// SPDX-FileCopyrightText: Copyright 2026 IFreemz
// SPDX-License-Identifier: MIT
//
// shadps4_dlss.dll / shadps4_dlss.so: the only part of the project that uses the NVIDIA DLSS (NGX)
// SDK. DLSS Frame Generation (Linux) follows bbport's (deadinside28/bloodborne_pc), with the
// settings it verified on an RTX 5090.

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "shadps4_dlss_bridge.h" // Vulkan first: the NGX headers expect its types

#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_vk.h>
#include <nvsdk_ngx_helpers_vk.h>
// Streamline runs Frame Generation on Windows, so the Windows bridge builds with SDKs older than
// its headers (310.7).
#ifndef _WIN32
#define DIRECT_FRAME_GENERATION
#include <nvsdk_ngx_helpers_dlssg_vk.h>
#endif

namespace {

constexpr char ProjectId[] = "ea385843-a44c-4ea5-8e6d-46a16125aef0";
constexpr char EngineVersion[] = "shadPS4-Bloodborne-DLSS";

struct State {
    ShadDlssLogFn log{};
    std::wstring dll_directory;
    std::wstring data_directory;
    const wchar_t* dll_path{};
    NVSDK_NGX_FeatureCommonInfo common{};
    NVSDK_NGX_FeatureDiscoveryInfo discovery{};
    NVSDK_NGX_FeatureDiscoveryInfo discovery_fg{};
    NVSDK_NGX_Parameter* capabilities{};
    NVSDK_NGX_Parameter* parameters{};
    NVSDK_NGX_Handle* feature{};
    ShadDlssFeature feature_desc{};
    // Frame Generation keeps a parameter map of its own.
    NVSDK_NGX_Parameter* fg_parameters{};
    NVSDK_NGX_Handle* fg_feature{};
    ShadDlssgFeature fg_desc{};
    bool fg_device_supported{};
    int32_t fg_max_count{};
    std::vector<VkExtensionProperties> instance_extensions, device_extensions;
    VkDevice device{};
    bool configured{};
    bool initialized{};
} state;

template <typename... Args>
void Log(int warning, const char* format, Args... args) {
    if (!state.log)
        return;
    std::array<char, 1024> text{};
    std::snprintf(text.data(), text.size(), format, args...);
    state.log(warning, text.data());
}

bool Check(const char* operation, NVSDK_NGX_Result result) {
    if (NVSDK_NGX_FAILED(result)) {
        Log(1, "%s failed: 0x%08x", operation, static_cast<unsigned>(result));
        return false;
    }
    return true;
}

void NVSDK_CONV NgxLog(const char* message, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature) {
    Log(0, "NGX: %s", message ? message : "");
}

int32_t Configure(const wchar_t* dll_directory, const wchar_t* data_directory,
                  ShadDlssLogFn log) {
    state.log = log;
    state.dll_directory = dll_directory ? dll_directory : L".";
    state.data_directory = data_directory ? data_directory : L".";
    state.dll_path = state.dll_directory.c_str();
    state.common.PathListInfo = {&state.dll_path, 1};
    state.common.LoggingInfo = {NgxLog, NVSDK_NGX_LOGGING_LEVEL_OFF, false};
    state.discovery.SDKVersion = NVSDK_NGX_Version_API;
    state.discovery.FeatureID = NVSDK_NGX_Feature_SuperSampling;
    state.discovery.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Project_Id;
    state.discovery.Identifier.v.ProjectDesc = {ProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM,
                                                EngineVersion};
    state.discovery.ApplicationDataPath = state.data_directory.c_str();
    state.discovery.FeatureInfo = &state.common;
#ifdef DIRECT_FRAME_GENERATION
    state.discovery_fg = state.discovery;
    state.discovery_fg.FeatureID = NVSDK_NGX_Feature_FrameGeneration;
#endif
    state.configured = true;
    return 1;
}

constexpr bool DirectFrameGeneration =
#ifdef DIRECT_FRAME_GENERATION
    true;
#else
    false;
#endif

void Merge(std::vector<VkExtensionProperties>& into, const VkExtensionProperties* add,
           uint32_t count) {
    for (uint32_t i = 0; add && i < count; ++i) {
        const auto same = std::find_if(into.begin(), into.end(), [&](const auto& have) {
            return std::strcmp(have.extensionName, add[i].extensionName) == 0;
        });
        if (same == into.end())
            into.push_back(add[i]);
        else
            same->specVersion = std::max(same->specVersion, add[i].specVersion);
    }
}

int32_t InstanceExtensions(uint32_t* count, const VkExtensionProperties** extensions) {
    VkExtensionProperties* required{};
    uint32_t required_count{};
    if (!state.configured ||
        !Check("Instance extension query",
               NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(
                   &state.discovery, &required_count, &required)))
        return 0;
    state.instance_extensions.clear();
    Merge(state.instance_extensions, required, required_count);
    required = nullptr;
    required_count = 0;
    if (DirectFrameGeneration &&
        NVSDK_NGX_SUCCEED(NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(
            &state.discovery_fg, &required_count, &required)))
        Merge(state.instance_extensions, required, required_count);
    *count = static_cast<uint32_t>(state.instance_extensions.size());
    *extensions = state.instance_extensions.data();
    return 1;
}

int32_t DeviceExtensions(VkInstance instance, VkPhysicalDevice physical, uint32_t* count,
                         const VkExtensionProperties** extensions) {
    if (!state.configured)
        return 0;
    NVSDK_NGX_FeatureRequirement support{};
    if (!Check("Feature requirement query",
               NVSDK_NGX_VULKAN_GetFeatureRequirements(instance, physical, &state.discovery,
                                                       &support)))
        return 0;
    if (support.FeatureSupported != NVSDK_NGX_FeatureSupportResult_Supported) {
        Log(1, "DLSS is not supported on this GPU/driver (code 0x%x)",
            static_cast<unsigned>(support.FeatureSupported));
        return 0;
    }
    VkExtensionProperties* required{};
    uint32_t required_count{};
    if (!Check("Device extension query",
               NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
                   instance, physical, &state.discovery, &required_count, &required)))
        return 0;
    state.device_extensions.clear();
    Merge(state.device_extensions, required, required_count);
    // Frame Generation's extensions only when this GPU and driver run it: device creation would
    // fail with them otherwise.
    if (DirectFrameGeneration) {
        NVSDK_NGX_FeatureRequirement fg_support{};
        required = nullptr;
        required_count = 0;
        state.fg_device_supported =
            NVSDK_NGX_SUCCEED(NVSDK_NGX_VULKAN_GetFeatureRequirements(
                instance, physical, &state.discovery_fg, &fg_support)) &&
            fg_support.FeatureSupported == NVSDK_NGX_FeatureSupportResult_Supported &&
            NVSDK_NGX_SUCCEED(NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
                instance, physical, &state.discovery_fg, &required_count, &required));
        if (state.fg_device_supported)
            Merge(state.device_extensions, required, required_count);
        else
            Log(0, "Frame Generation is not supported on this GPU/driver (code 0x%x)",
                static_cast<unsigned>(fg_support.FeatureSupported));
    }
    *count = static_cast<uint32_t>(state.device_extensions.size());
    *extensions = state.device_extensions.data();
    return 1;
}

int32_t Initialize(VkInstance instance, VkPhysicalDevice physical, VkDevice device,
                   PFN_vkGetInstanceProcAddr get_instance_proc,
                   PFN_vkGetDeviceProcAddr get_device_proc) {
    if (!state.configured || state.initialized)
        return 0;
    if (!Check("NGX initialization",
               NVSDK_NGX_VULKAN_Init_with_ProjectID(
                   ProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, EngineVersion,
                   state.data_directory.c_str(), instance, physical, device, get_instance_proc,
                   get_device_proc, &state.common)))
        return 0;
    state.initialized = true;
    state.device = device;
    if (!Check("Capability query", NVSDK_NGX_VULKAN_GetCapabilityParameters(&state.capabilities)))
        return 0;
    int available{}, needs_driver{};
    NVSDK_NGX_Parameter_GetI(state.capabilities, NVSDK_NGX_Parameter_SuperSampling_Available,
                             &available);
    NVSDK_NGX_Parameter_GetI(state.capabilities,
                             NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needs_driver);
    if (needs_driver) {
        Log(1, "DLSS needs a newer NVIDIA driver");
        return 0;
    }
    if (!available) {
        Log(1, "DLSS Super Resolution is not available on this system");
        return 0;
    }
#ifdef DIRECT_FRAME_GENERATION
    if (state.fg_device_supported) {
        int fg_available{}, fg_needs_driver{};
        unsigned fg_max{};
        NVSDK_NGX_Parameter_GetI(state.capabilities,
                                 NVSDK_NGX_Parameter_FrameGeneration_Available, &fg_available);
        NVSDK_NGX_Parameter_GetI(state.capabilities,
                                 NVSDK_NGX_Parameter_FrameGeneration_NeedsUpdatedDriver,
                                 &fg_needs_driver);
        if (NVSDK_NGX_FAILED(NVSDK_NGX_Parameter_GetUI(
                state.capabilities, NVSDK_NGX_DLSSG_Parameter_MultiFrameCountMax, &fg_max)))
            fg_max = 1;
        if (fg_available) {
            state.fg_max_count = static_cast<int32_t>(std::clamp(fg_max, 1u, 3u));
            Log(0, "Frame Generation available (up to %dx)", state.fg_max_count + 1);
        } else {
            Log(1, fg_needs_driver ? "Frame Generation needs a newer NVIDIA driver"
                                   : "Frame Generation unavailable (libnvidia-ngx-dlssg.so is "
                                     "missing, or the GPU is older than RTX 40)");
        }
    }
#endif
    return 1;
}

void ReleaseFeature() {
    if (state.feature) {
        NVSDK_NGX_VULKAN_ReleaseFeature(state.feature);
        state.feature = nullptr;
    }
    if (state.parameters) {
        NVSDK_NGX_VULKAN_DestroyParameters(state.parameters);
        state.parameters = nullptr;
    }
}

int32_t CreateFeature(VkCommandBuffer command, const ShadDlssFeature* desc) {
    if (!state.initialized || !state.capabilities || !desc)
        return 0;
    ReleaseFeature();
    if (!Check("Parameter allocation", NVSDK_NGX_VULKAN_AllocateParameters(&state.parameters)))
        return 0;
    for (const char* key : {NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA,
                            NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality,
                            NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced,
                            NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance,
                            NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance,
                            NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraQuality})
        NVSDK_NGX_Parameter_SetUI(state.parameters, key, desc->preset);
    static constexpr std::array qualities{
        NVSDK_NGX_PerfQuality_Value_DLAA, NVSDK_NGX_PerfQuality_Value_MaxQuality,
        NVSDK_NGX_PerfQuality_Value_Balanced, NVSDK_NGX_PerfQuality_Value_MaxPerf,
        NVSDK_NGX_PerfQuality_Value_UltraPerformance};
    NVSDK_NGX_DLSS_Create_Params settings{};
    settings.Feature.InWidth = desc->input_width;
    settings.Feature.InHeight = desc->input_height;
    settings.Feature.InTargetWidth = desc->output_width;
    settings.Feature.InTargetHeight = desc->output_height;
    settings.Feature.InPerfQualityValue = qualities[std::clamp(desc->quality, 0, 4)];
    settings.InFeatureCreateFlags =
        NVSDK_NGX_DLSS_Feature_Flags_MVLowRes |
        (desc->depth_inverted ? NVSDK_NGX_DLSS_Feature_Flags_DepthInverted : 0) |
        (desc->hdr ? NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_AutoExposure
                   : 0);
    const auto result = NGX_VULKAN_CREATE_DLSS_EXT1(state.device, command, 1, 1, &state.feature,
                                                   state.parameters, &settings);
    if (!Check("DLSS feature creation", result) || !state.feature) {
        state.feature = nullptr;
        return 0;
    }
    state.feature_desc = *desc;
    Log(0, "DLSS %ux%u -> %ux%u, quality %d, preset %u", desc->input_width, desc->input_height,
        desc->output_width, desc->output_height, desc->quality, desc->preset);
    return 1;
}

NVSDK_NGX_Resource_VK Wrap(const ShadDlssImage& image, bool writable) {
    return NVSDK_NGX_Create_ImageView_Resource_VK(image.view, image.image, image.range,
                                                  image.format, image.width, image.height,
                                                  writable);
}

int32_t Evaluate(VkCommandBuffer command, const ShadDlssEvaluate* evaluate) {
    if (!state.feature || !state.parameters || !evaluate)
        return 0;
    auto color = Wrap(evaluate->color, false);
    auto depth = Wrap(evaluate->depth, false);
    auto motion = Wrap(evaluate->motion, false);
    auto output = Wrap(evaluate->output, true);
    NVSDK_NGX_VK_DLSS_Eval_Params parameters{};
    parameters.Feature.pInColor = &color;
    parameters.Feature.pInOutput = &output;
    parameters.pInDepth = &depth;
    parameters.pInMotionVectors = &motion;
    parameters.InRenderSubrectDimensions = {state.feature_desc.input_width,
                                            state.feature_desc.input_height};
    parameters.InJitterOffsetX = evaluate->jitter_x;
    parameters.InJitterOffsetY = evaluate->jitter_y;
    parameters.InReset = evaluate->reset;
    parameters.InMVScaleX = parameters.InMVScaleY = 1.0f;
    parameters.InPreExposure = parameters.InExposureScale = 1.0f;
    parameters.InFrameTimeDeltaInMsec = evaluate->frame_ms;
    return Check("DLSS evaluation", NGX_VULKAN_EVALUATE_DLSS_EXT(command, state.feature,
                                                                state.parameters, &parameters))
               ? 1
               : 0;
}

int32_t FrameGenMaxCount() {
    return state.initialized ? state.fg_max_count : 0;
}

void FrameGenRelease() {
    if (state.fg_feature) {
        NVSDK_NGX_VULKAN_ReleaseFeature(state.fg_feature);
        state.fg_feature = nullptr;
    }
    if (state.fg_parameters) {
        NVSDK_NGX_VULKAN_DestroyParameters(state.fg_parameters);
        state.fg_parameters = nullptr;
    }
}

#ifdef DIRECT_FRAME_GENERATION
int32_t FrameGenCreate(VkCommandBuffer command, const ShadDlssgFeature* desc) {
    if (!state.initialized || state.fg_max_count < 1 || !desc)
        return 0;
    FrameGenRelease();
    if (!Check("Frame Generation parameter allocation",
               NVSDK_NGX_VULKAN_AllocateParameters(&state.fg_parameters)))
        return 0;
    NVSDK_NGX_Parameter_SetUI(state.fg_parameters, NVSDK_NGX_DLSSG_Parameter_Width, desc->width);
    NVSDK_NGX_Parameter_SetUI(state.fg_parameters, NVSDK_NGX_DLSSG_Parameter_Height, desc->height);
    NVSDK_NGX_Parameter_SetUI(state.fg_parameters, NVSDK_NGX_DLSSG_Parameter_InternalWidth,
                              desc->render_width);
    NVSDK_NGX_Parameter_SetUI(state.fg_parameters, NVSDK_NGX_DLSSG_Parameter_InternalHeight,
                              desc->render_height);
    NVSDK_NGX_Parameter_SetUI(state.fg_parameters, NVSDK_NGX_DLSSG_Parameter_DynamicResolution, 0);
    NVSDK_NGX_DLSSG_Create_Params settings{};
    settings.Width = desc->width;
    settings.Height = desc->height;
    settings.NativeBackbufferFormat = static_cast<unsigned>(desc->format);
    settings.RenderWidth = desc->render_width;
    settings.RenderHeight = desc->render_height;
    settings.DynamicResolutionScaling = false;
    const auto result =
        NGX_VK_CREATE_DLSSG(command, 1, 1, &state.fg_feature, state.fg_parameters, &settings);
    if (!Check("Frame Generation feature creation", result) || !state.fg_feature) {
        state.fg_feature = nullptr;
        return 0;
    }
    state.fg_desc = *desc;
    Log(0, "Frame Generation %ux%u (depth and motion %ux%u, format %u)", desc->width,
        desc->height, desc->render_width, desc->render_height,
        static_cast<unsigned>(desc->format));
    return 1;
}

int32_t FrameGenEvaluate(VkCommandBuffer command, const ShadDlssgEvaluate* evaluate) {
    if (!state.fg_feature || !state.fg_parameters || !evaluate)
        return 0;
    auto backbuffer = Wrap(evaluate->backbuffer, false);
    auto depth = Wrap(evaluate->depth, false);
    auto motion = Wrap(evaluate->motion, false);
    auto output = Wrap(evaluate->output, true);
    NVSDK_NGX_VK_DLSSG_Eval_Params resources{};
    resources.pBackbuffer = &backbuffer;
    resources.pDepth = &depth;
    resources.pMVecs = &motion;
    resources.pOutputInterpFrame = &output;
    NVSDK_NGX_Resource_VK hudless{};
    if (evaluate->hudless.image != VK_NULL_HANDLE) {
        hudless = Wrap(evaluate->hudless, false);
        resources.pHudless = &hudless;
    }
    NVSDK_NGX_DLSSG_Opt_Eval_Params options{};
    options.multiFrameCount = std::max(1u, evaluate->multi_frame_count);
    options.multiFrameIndex = std::clamp(evaluate->multi_frame_index, 1u, options.multiFrameCount);
    std::memcpy(options.cameraViewToClip, evaluate->view_to_clip, sizeof(options.cameraViewToClip));
    std::memcpy(options.clipToCameraView, evaluate->clip_to_view, sizeof(options.clipToCameraView));
    std::memcpy(options.clipToPrevClip, evaluate->clip_to_prev_clip,
                sizeof(options.clipToPrevClip));
    std::memcpy(options.prevClipToClip, evaluate->prev_clip_to_clip,
                sizeof(options.prevClipToClip));
    std::memset(options.clipToLensClip, 0, sizeof(options.clipToLensClip));
    for (int i = 0; i < 4; ++i)
        options.clipToLensClip[i][i] = 1.0f;
    options.jitterOffset[0] = evaluate->jitter_x;
    options.jitterOffset[1] = evaluate->jitter_y;
    options.mvecScale[0] = evaluate->mvec_scale_x;
    options.mvecScale[1] = evaluate->mvec_scale_y;
    for (int i = 0; i < 3; ++i) {
        options.cameraPos[i] = evaluate->camera_position[i];
        options.cameraUp[i] = evaluate->camera_up[i];
        options.cameraRight[i] = evaluate->camera_right[i];
        options.cameraFwd[i] = evaluate->camera_forward[i];
    }
    options.cameraNear = evaluate->camera_near;
    options.cameraFar = evaluate->camera_far;
    options.cameraFOV = evaluate->camera_fov;
    options.cameraAspectRatio = evaluate->camera_aspect;
    options.colorBuffersHDR = false;
    options.depthInverted = evaluate->depth_inverted != 0;
    options.cameraMotionIncluded = true;
    // No vector is marked invalid: 0, the zero-initialised value, would mark every still pixel.
    options.motionVectorsInvalidValue = std::numeric_limits<float>::max();
    options.motionVectorsDilated = false;
    options.menuDetectionEnabled = false;
    options.reset = evaluate->reset != 0 && options.multiFrameIndex == 1;
    // Depth and vectors are at the render size, smaller than the frame: a zero size would mean the
    // whole texture.
    options.mvecsSubrectSize = {evaluate->motion.width, evaluate->motion.height};
    options.depthSubrectSize = {evaluate->depth.width, evaluate->depth.height};
    options.backbufferSubrectSize = {evaluate->backbuffer.width, evaluate->backbuffer.height};
    options.outputInterpSubrectSize = {evaluate->output.width, evaluate->output.height};
    if (evaluate->hudless.image != VK_NULL_HANDLE)
        options.hudLessSubrectSize = {evaluate->hudless.width, evaluate->hudless.height};
    return Check("Frame Generation evaluation",
                 NGX_VK_EVALUATE_DLSSG(command, state.fg_feature, state.fg_parameters,
                                       &resources, &options))
               ? 1
               : 0;
}
#else
int32_t FrameGenCreate(VkCommandBuffer, const ShadDlssgFeature*) {
    return 0;
}

int32_t FrameGenEvaluate(VkCommandBuffer, const ShadDlssgEvaluate*) {
    return 0;
}
#endif

void Shutdown() {
    FrameGenRelease();
    ReleaseFeature();
    if (state.capabilities) {
        NVSDK_NGX_VULKAN_DestroyParameters(state.capabilities);
        state.capabilities = nullptr;
    }
    if (state.initialized) {
        NVSDK_NGX_VULKAN_Shutdown1(state.device);
        state.initialized = false;
        state.device = VK_NULL_HANDLE;
    }
}

const ShadDlssApi api{SHADPS4_DLSS_BRIDGE_ABI,
                      Configure,
                      InstanceExtensions,
                      DeviceExtensions,
                      Initialize,
                      CreateFeature,
                      Evaluate,
                      ReleaseFeature,
                      Shutdown,
                      FrameGenMaxCount,
                      FrameGenCreate,
                      FrameGenEvaluate,
                      FrameGenRelease};

} // namespace

#ifdef _WIN32
#define SHADPS4_DLSS_EXPORT __declspec(dllexport)
#else
#define SHADPS4_DLSS_EXPORT __attribute__((visibility("default")))
#endif

extern "C" SHADPS4_DLSS_EXPORT const ShadDlssApi* ShadDlssGetApi() {
    return &api;
}
