// SPDX-FileCopyrightText: Copyright 2026 IFreemz
// SPDX-License-Identifier: GPL-2.0-or-later
//
// FSR 4 add-on: loads the v07 model files of one preset and output tier and
// records the FSR 4 passes through FSR-Vulkan. The integration follows bbport's
// (deadinside28/bloodborne_pc).

#include <array>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "shadps4_fsr4.h"

extern "C" {
#include "ffx_vk_fsr4_v07.h"
#include "ffx_vk_fsr4_v07_assets.h"
#include "ffx_vk_fsr4_v07_schedule.h"
}

namespace {

struct State {
    ShadFsr4LogFn log{};
    std::filesystem::path directory;
    VkPhysicalDevice physical{};
    VkDevice device{};
    std::vector<uint8_t> scratch;
    FfxInterface backend{};
    bool backend_ok{};
    ffxContext context{};
    bool context_ok{};
    std::optional<ShadFsr4Context> current;
    FfxFsr4ModelPreset preset{};
    std::string problem;
} state;

void Log(bool warning, const std::string& message) {
    if (state.log)
        state.log(warning ? 1 : 0, message.c_str());
}

void Fail(const std::string& reason) {
    state.problem = reason;
    Log(true, reason);
}

FfxFsr4ModelPreset PresetFor(const ShadFsr4Context& c) {
    // The v07 performance and ultra performance models leave trails of moving objects in
    // Bloodborne (2x and 3x upscaling); the balanced model handles those ratios cleanly.
    const auto preset = ffxFsr4SelectModelPreset(c.render_width, c.output_width, false);
    return preset == FFX_FSR4_MODEL_PRESET_PERFORMANCE ||
                   preset == FFX_FSR4_MODEL_PRESET_ULTRA_PERFORMANCE
               ? FFX_FSR4_MODEL_PRESET_BALANCED
               : preset;
}

bool ReadFile(const std::filesystem::path& path, std::vector<uint8_t>& data) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        return false;
    data.resize(size_t(file.tellg()));
    file.seekg(0);
    return bool(file.read(reinterpret_cast<char*>(data.data()), std::streamsize(data.size())));
}

void Release() {
    if (state.context_ok)
        ffxFsr4V07DestroyContext(&state.context, nullptr);
    if (state.backend_ok)
        ffxFsr4VkDestroyContext(reinterpret_cast<FfxFsr4VkContext*>(state.scratch.data()));
    state.context = {};
    state.context_ok = false;
    state.backend = {};
    state.backend_ok = false;
    state.current.reset();
}

int32_t Configure(const wchar_t* directory, ShadFsr4LogFn log) {
    state.log = log;
    state.directory = directory ? std::filesystem::path{directory} : std::filesystem::path{};
    return 1;
}

int32_t Initialize(VkPhysicalDevice physical, VkDevice device) {
    state.physical = physical;
    state.device = device;
    return physical && device ? 1 : 0;
}

int32_t HasContext(const ShadFsr4Context* c) {
    return state.context_ok && state.current && c &&
           state.current->output_width == c->output_width &&
           state.current->output_height == c->output_height && state.preset == PresetFor(*c);
}

int32_t CreateContext(const ShadFsr4Context* c) {
    Release();
    if (!c || !state.device)
        return 0;
    const FfxFsr4ModelPreset preset = PresetFor(*c);
    FfxFsr4V07AssetSet assets{};
    if (!ffxFsr4V07BuildAssetSet(preset, c->output_width, c->output_height, &assets)) {
        Fail("FSR 4: this output size is not supported by the v07 model");
        return 0;
    }
    std::array<std::vector<uint8_t>, FFX_FSR4_VK_PASS_COUNT> code;
    std::vector<uint8_t> initializer, weights;
    const auto load = [&](const char* name, std::vector<uint8_t>& data) {
        // Optional faster or fixed passes in opt/ take precedence.
        if (ReadFile(state.directory / "opt" / name, data) ||
            ReadFile(state.directory / name, data))
            return true;
        Fail(std::string{"FSR 4: model file "} + name + " is missing from the fsr4 folder");
        return false;
    };
    if (!load(assets.pre, code[0]))
        return 0;
    for (uint32_t pass = 0; pass < FFX_FSR4_MODEL_PASS_COUNT; ++pass)
        if (!load(assets.model[pass], code[1 + pass]))
            return 0;
    if (!load(assets.post, code[13]) || !load(assets.rcas, code[14]) ||
        !load(assets.spdAutoExposure, code[15]) || !load(assets.initializer, initializer) ||
        !load(assets.prePassWeights, weights))
        return 0;
    if (initializer.size() != FFX_FSR4_V07_INITIALIZER_BYTES ||
        weights.size() != FFX_FSR4_V07_PRE_PASS_WEIGHTS_BYTES) {
        Fail("FSR 4: model weights have an unexpected size");
        return 0;
    }
    static const std::array<std::string, FFX_FSR4_VK_PASS_COUNT> entries = [] {
        std::array<std::string, FFX_FSR4_VK_PASS_COUNT> names;
        names.fill("main");
        for (uint32_t pass = 1; pass <= FFX_FSR4_MODEL_PASS_COUNT; ++pass)
            names[pass] = "fsr4_model_v07_i8_pass" + std::to_string(pass);
        return names;
    }();
    FfxFsr4VkCreateInfo info{};
    info.device = state.device;
    info.physicalDevice = state.physical;
    for (uint32_t i = 0; i < FFX_FSR4_VK_PASS_COUNT; ++i) {
        if (code[i].size() % 4) {
            Fail("FSR 4: invalid shader file");
            return 0;
        }
        info.shaders[i] = {reinterpret_cast<const uint32_t*>(code[i].data()), code[i].size(),
                           entries[i].c_str()};
    }
    info.modelInitializer = initializer.data();
    info.modelInitializerSize = initializer.size();
    info.prePassWeights = weights.data();
    info.prePassWeightsSize = weights.size();
    state.scratch.assign(ffxFsr4VkGetScratchMemorySize(), 0);
    info.scratchBuffer = state.scratch.data();
    info.scratchBufferSize = state.scratch.size();
    if (const VkResult result = ffxFsr4VkCreateContext(&info, &state.backend);
        result != VK_SUCCESS) {
        state.backend = {};
        Fail("FSR 4: Vulkan backend creation failed (" + std::to_string(int(result)) + ")");
        return 0;
    }
    state.backend_ok = true;

    ffxCreateContextDescUpscale desc{};
    desc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    // Any render size up to the output fits, so a render size change keeps the
    // context.
    desc.maxRenderSize = {(c->output_width + 7) & ~7u, (c->output_height + 7) & ~7u};
    desc.maxUpscaleSize = desc.maxRenderSize;
    ffxFsr4V07SetBackendInterface(&state.backend);
    const auto created = ffxFsr4V07CreateContext(&state.context, &desc.header, nullptr);
    ffxFsr4V07SetBackendInterface(nullptr);
    if (created != FFX_API_RETURN_OK) {
        Release();
        Fail("FSR 4: context creation failed (" + std::to_string(created) + ")");
        return 0;
    }
    state.context_ok = true;
    state.current = *c;
    state.preset = preset;
    state.problem.clear();
    Log(false, std::string{"FSR 4 v07 "} + ffxFsr4ModelPresetName(preset) + " model, " +
                   assets.tier + " tier, output " + std::to_string(c->output_width) + "x" +
                   std::to_string(c->output_height));
    return 1;
}

FfxApiResource Resource(const ShadFsr4Image& image, uint32_t resource_state) {
    FfxApiResource r{};
    r.resource = reinterpret_cast<void*>(image.view);
    r.description.type = FFX_RESOURCE_TYPE_TEXTURE2D;
    r.description.format = image.ffx_format;
    r.description.width = image.width;
    r.description.height = image.height;
    r.description.depth = 1;
    r.description.mipCount = 1;
    r.state = resource_state;
    return r;
}

VkResult Register(const ShadFsr4Image& image, bool writable) {
    const VkAccessFlags access =
        VK_ACCESS_SHADER_READ_BIT | (writable ? VK_ACCESS_SHADER_WRITE_BIT : 0u);
    const FfxFsr4VkExternalImageState external{
        .structSize = sizeof(FfxFsr4VkExternalImageState),
        .image = image.image,
        .view = image.view,
        .layout = image.layout,
        .stageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        .accessMask = access,
        .restoreLayout = image.layout,
        .restoreStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        .restoreAccessMask = access,
    };
    return ffxFsr4VkSetExternalImageState(&state.backend, &external);
}

int32_t Evaluate(VkCommandBuffer command, const ShadFsr4Evaluate* e, uint64_t frame_id) {
    if (!state.context_ok || !e)
        return SHADPS4_FSR4_FAILED;
    const VkResult begin = ffxFsr4VkBeginFrame(&state.backend, frame_id);
    if (begin == VK_NOT_READY)
        return SHADPS4_FSR4_BUSY;
    if (begin != VK_SUCCESS) {
        Fail("FSR 4: no free frame (" + std::to_string(int(begin)) + ")");
        return SHADPS4_FSR4_FAILED;
    }
    // From here the frame id must be retired even if nothing is recorded.
    if (Register(e->color, false) != VK_SUCCESS || Register(e->depth, false) != VK_SUCCESS ||
        Register(e->motion, false) != VK_SUCCESS || Register(e->output, true) != VK_SUCCESS) {
        Fail("FSR 4: image registration failed");
        return SHADPS4_FSR4_ABANDONED;
    }
    ffxDispatchDescUpscale d{};
    d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    d.commandList = command;
    d.color = Resource(e->color, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.depth = Resource(e->depth, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.motionVectors = Resource(e->motion, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.output = Resource(e->output, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
    d.jitterOffset = {e->jitter_x, e->jitter_y};
    d.motionVectorScale = {1.0f, 1.0f}; // render pixels; the provider divides by the render size
    d.renderSize = {e->color.width, e->color.height};
    d.upscaleSize = {e->output.width, e->output.height};
    d.enableSharpening = false; // shadPS4's composite pass sharpens for every upscaler
    d.enableAutoExposure = e->hdr == 1;
    d.frameTimeDelta = e->frame_ms;
    d.preExposure = 1.0f;
    d.reset = e->reset != 0;
    d.cameraNear = e->camera_near;
    d.cameraFar = e->camera_far;
    d.cameraFovAngleVertical = e->fov_y;
    d.viewSpaceToMetersFactor = 1.0f;
    if (const auto result = ffxFsr4V07Dispatch(&state.context, &d.header);
        result != FFX_API_RETURN_OK) {
        Fail("FSR 4: dispatch failed (" + std::to_string(result) + ")");
        return SHADPS4_FSR4_ABANDONED;
    }
    return SHADPS4_FSR4_OK;
}

void Retire(uint64_t completed_frame_id) {
    if (state.backend_ok)
        ffxFsr4VkRetireFrame(&state.backend, completed_frame_id);
}

void ReleaseContext() {
    Release();
}

const char* Problem() {
    return state.problem.c_str();
}

constexpr ShadFsr4Api Api{SHADPS4_FSR4_ABI, Configure, Initialize,     HasContext, CreateContext,
                          Evaluate,         Retire,    ReleaseContext, Problem};

} // namespace

#ifdef _WIN32
#define SHADPS4_FSR4_EXPORT __declspec(dllexport)
#else
#define SHADPS4_FSR4_EXPORT __attribute__((visibility("default")))
#endif

extern "C" SHADPS4_FSR4_EXPORT const ShadFsr4Api* ShadFsr4GetApi() {
    return &Api;
}
