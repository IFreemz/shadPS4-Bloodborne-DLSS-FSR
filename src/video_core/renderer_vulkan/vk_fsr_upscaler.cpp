// SPDX-FileCopyrightText: Copyright 2026 IFreemz
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_bb_temporal_dlss.h"
#include "video_core/renderer_vulkan/vk_fsr_upscaler.h"
#include "video_core/renderer_vulkan/vk_instance.h"

#ifndef _WIN32
// ffx_api.h marks its declarations __declspec(dllexport).
#define __declspec(x)
#endif
#include <ffx_api/ffx_upscale.h>
#include <ffx_api/vk/ffx_api_vk.h>
#ifndef _WIN32
#undef __declspec
#endif
#include "video_core/renderer_vulkan/vk_bb_library.h"

namespace Vulkan {

namespace {
constexpr char FsrName[] = BB_LIBRARY_NAME("amd_fidelityfx_vk.dll", "libamd_fidelityfx_vk.so");
} // namespace

bool FsrUpscaler::Present() {
    static const bool present = [] {
        const char* setting = std::getenv("SHADPS4_FSR");
        if (setting && std::string_view{setting} == "0")
            return false;
        return std::filesystem::is_regular_file(BbLibrary::ExecutableDirectory() / FsrName);
    }();
    return present;
}

#if defined(_WIN32) || defined(__linux__)

namespace {
void FfxMessage(uint32_t type, const wchar_t* message) {
    const auto text = std::filesystem::path{message}.string();
    if (type == FFX_API_MESSAGE_TYPE_ERROR)
        LOG_ERROR(Render_Vulkan, "[FSR] {}", text);
    else
        LOG_WARNING(Render_Vulkan, "[FSR] {}", text);
}

VKAPI_ATTR VkResult VKAPI_CALL NoObjectName(VkDevice, const VkDebugUtilsObjectNameInfoEXT*) {
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL NoLabel(VkCommandBuffer, const VkDebugUtilsLabelEXT*) {}
VKAPI_ATTR void VKAPI_CALL NoLabelEnd(VkCommandBuffer) {}
VKAPI_ATTR void VKAPI_CALL NoMarker(VkCommandBuffer, VkPipelineStageFlagBits, VkBuffer,
                                    VkDeviceSize, uint32_t) {}
VKAPI_ATTR void VKAPI_CALL NoMarker2(VkCommandBuffer, VkPipelineStageFlags2, VkBuffer, VkDeviceSize,
                                     uint32_t) {}

// The FidelityFX backend looks functions up by extension name and calls some without checking.
// Fall back to the core name (e.g. vkGetBufferMemoryRequirements2KHR -> ...2), and give debug
// label functions no-op stand-ins when debug utils are not enabled. On AMD cards it also checks
// for the buffer marker extension on the GPU rather than on the device shadPS4 created, so the
// marker functions get no-op stand-ins as well.
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL DeviceProcAddr(VkDevice device, const char* name) {
    const auto get = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;
    if (auto function = get(device, name))
        return function;
    const std::string_view view{name};
    for (const std::string_view suffix : {"KHR", "EXT"}) {
        if (view.ends_with(suffix)) {
            const std::string core{view.substr(0, view.size() - suffix.size())};
            if (auto function = get(device, core.c_str())) {
                LOG_INFO(Render_Vulkan, "[FSR] {} -> {}", name, core);
                return function;
            }
        }
    }
    if (view == "vkSetDebugUtilsObjectNameEXT")
        return reinterpret_cast<PFN_vkVoidFunction>(&NoObjectName);
    if (view == "vkCmdBeginDebugUtilsLabelEXT" || view == "vkCmdInsertDebugUtilsLabelEXT")
        return reinterpret_cast<PFN_vkVoidFunction>(&NoLabel);
    if (view == "vkCmdEndDebugUtilsLabelEXT")
        return reinterpret_cast<PFN_vkVoidFunction>(&NoLabelEnd);
    if (view == "vkCmdWriteBufferMarkerAMD")
        return reinterpret_cast<PFN_vkVoidFunction>(&NoMarker);
    if (view == "vkCmdWriteBufferMarker2AMD")
        return reinterpret_cast<PFN_vkVoidFunction>(&NoMarker2);
    LOG_WARNING(Render_Vulkan, "[FSR] Vulkan function {} is not available", name);
    return nullptr;
}

uint32_t SurfaceFormat(vk::Format format) {
    switch (format) {
    case vk::Format::eR8G8B8A8Unorm:
        return FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM;
    case vk::Format::eR32Sfloat:
        return FFX_API_SURFACE_FORMAT_R32_FLOAT;
    case vk::Format::eR16G16Sfloat:
        return FFX_API_SURFACE_FORMAT_R16G16_FLOAT;
    case vk::Format::eR16G16B16A16Sfloat:
        return FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT;
    default:
        return FFX_API_SURFACE_FORMAT_UNKNOWN;
    }
}

FfxApiResource ToFfx(const DlssNgx::Resource& resource, bool writable) {
    FfxApiResource result{};
    result.resource = reinterpret_cast<void*>(static_cast<VkImage>(resource.image));
    result.description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    result.description.format = SurfaceFormat(resource.format);
    result.description.width = resource.extent.width;
    result.description.height = resource.extent.height;
    result.description.depth = 1;
    result.description.mipCount = 1;
    result.description.flags = FFX_API_RESOURCE_FLAGS_NONE;
    result.description.usage =
        writable ? FFX_API_RESOURCE_USAGE_UAV : FFX_API_RESOURCE_USAGE_READ_ONLY;
    // Inputs arrive in SHADER_READ_ONLY_OPTIMAL, the output in GENERAL.
    result.state =
        writable ? FFX_API_RESOURCE_STATE_UNORDERED_ACCESS : FFX_API_RESOURCE_STATE_COMPUTE_READ;
    return result;
}
} // namespace

struct FsrUpscaler::Impl {
    void* module{};
    PfnFfxCreateContext create{};
    PfnFfxDestroyContext destroy{};
    PfnFfxQuery query{};
    PfnFfxDispatch dispatch{};
    vk::Device device;
    vk::PhysicalDevice physical;
    ffxContext context{};
    std::optional<ContextDesc> desc;
};

FsrUpscaler::FsrUpscaler() : impl{std::make_unique<Impl>()} {}

void* FsrUpscaler::FfxDeviceProcAddr() {
    return reinterpret_cast<void*>(&DeviceProcAddr);
}

std::unique_ptr<FsrUpscaler> FsrUpscaler::Create(const Instance& instance) {
    if (!Present())
        return {};
    auto fsr = std::unique_ptr<FsrUpscaler>{new FsrUpscaler};
    auto& s = *fsr->impl;
    s.module = BbLibrary::Load(BbLibrary::ExecutableDirectory() / FsrName);
    if (!s.module) {
        LOG_WARNING(Render_Vulkan, "[FSR] {} could not be loaded", FsrName);
        return {};
    }
    s.create = BbLibrary::Find<PfnFfxCreateContext>(s.module, "ffxCreateContext");
    s.destroy = BbLibrary::Find<PfnFfxDestroyContext>(s.module, "ffxDestroyContext");
    s.query = BbLibrary::Find<PfnFfxQuery>(s.module, "ffxQuery");
    s.dispatch = BbLibrary::Find<PfnFfxDispatch>(s.module, "ffxDispatch");
    if (!s.create || !s.destroy || !s.query || !s.dispatch) {
        LOG_WARNING(Render_Vulkan, "[FSR] {} is missing entry points", FsrName);
        BbLibrary::Free(s.module);
        s.module = {};
        return {};
    }
    s.device = instance.GetDevice();
    s.physical = instance.GetPhysicalDevice();
    BbTemporalDlssReportGpu(s.physical.getProperties().deviceName.data());
    LOG_INFO(Render_Vulkan, "[FSR] Ready");
    return fsr;
}

FsrUpscaler::~FsrUpscaler() {
    DestroyContextAfterGpuDrain();
    BbLibrary::Free(impl->module);
}

bool FsrUpscaler::HasContext(const ContextDesc& desc) const {
    return impl->desc && *impl->desc == desc;
}

bool FsrUpscaler::CreateContext(const ContextDesc& desc) {
    DestroyContextAfterGpuDrain();
    ffxCreateBackendVKDesc backend{};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
    backend.vkDevice = impl->device;
    backend.vkPhysicalDevice = impl->physical;
    backend.vkDeviceProcAddr = DeviceProcAddr;
    ffxCreateContextDescUpscale create{};
    create.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    create.header.pNext = &backend.header;
    create.flags = (desc.hdr ? FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE |
                                   (desc.auto_exposure ? FFX_UPSCALE_ENABLE_AUTO_EXPOSURE : 0u)
                             : FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE) |
                   (desc.depth_inverted ? FFX_UPSCALE_ENABLE_DEPTH_INVERTED : 0u);
    create.maxRenderSize = {desc.render.width, desc.render.height};
    create.maxUpscaleSize = {desc.output.width, desc.output.height};
    create.fpMessage = FfxMessage;
    const auto result = impl->create(&impl->context, &create.header, nullptr);
    if (result != FFX_API_RETURN_OK) {
        impl->context = {};
        LOG_ERROR(Render_Vulkan, "[FSR] Context creation failed ({})", result);
        return false;
    }
    impl->desc = desc;
    ffxQueryGetProviderVersion version{};
    version.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
    impl->query(&impl->context, &version.header);
    LOG_INFO(Render_Vulkan, "[FSR] Context {}x{} -> {}x{} ({})", desc.render.width,
             desc.render.height, desc.output.width, desc.output.height,
             version.versionName ? version.versionName : "unknown version");
    return true;
}

void FsrUpscaler::DestroyContextAfterGpuDrain() {
    if (impl->context)
        impl->destroy(&impl->context, nullptr);
    impl->context = {};
    impl->desc.reset();
}

bool FsrUpscaler::Evaluate(vk::CommandBuffer command, const DlssNgx::Resource& color,
                           const DlssNgx::Resource& depth, const DlssNgx::Resource& motion,
                           const DlssNgx::Resource& output, const DlssNgx::EvalDesc& eval,
                           const Camera& camera) {
    if (!impl->context || !impl->desc)
        return false;
    ffxDispatchDescUpscale dispatch{};
    dispatch.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    dispatch.commandList = static_cast<VkCommandBuffer>(command);
    dispatch.color = ToFfx(color, false);
    dispatch.depth = ToFfx(depth, false);
    dispatch.motionVectors = ToFfx(motion, false);
    dispatch.output = ToFfx(output, true);
    dispatch.jitterOffset = {eval.jitter_x, eval.jitter_y};
    dispatch.motionVectorScale = {1.0f, 1.0f}; // already in render pixels
    dispatch.renderSize = {color.extent.width, color.extent.height};
    dispatch.upscaleSize = {output.extent.width, output.extent.height};
    dispatch.enableSharpening = false; // the composite pass sharpens for every upscaler
    dispatch.frameTimeDelta = eval.frame_ms;
    dispatch.preExposure = 1.0f;
    dispatch.reset = eval.reset;
    dispatch.cameraNear = camera.near_plane;
    dispatch.cameraFar = camera.far_plane;
    dispatch.cameraFovAngleVertical = camera.fov_y;
    dispatch.viewSpaceToMetersFactor = 1.0f;
    dispatch.flags = impl->desc->hdr ? 0u : FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB;
    const auto result = impl->dispatch(&impl->context, &dispatch.header);
    if (result != FFX_API_RETURN_OK) {
        LOG_ERROR(Render_Vulkan, "[FSR] Dispatch failed ({})", result);
        return false;
    }
    return true;
}

#else

struct FsrUpscaler::Impl {};
FsrUpscaler::FsrUpscaler() = default;
FsrUpscaler::~FsrUpscaler() = default;
std::unique_ptr<FsrUpscaler> FsrUpscaler::Create(const Instance&) {
    return {};
}
bool FsrUpscaler::HasContext(const ContextDesc&) const {
    return false;
}
bool FsrUpscaler::CreateContext(const ContextDesc&) {
    return false;
}
void FsrUpscaler::DestroyContextAfterGpuDrain() {}
void* FsrUpscaler::FfxDeviceProcAddr() {
    return nullptr;
}
bool FsrUpscaler::Evaluate(vk::CommandBuffer, const DlssNgx::Resource&, const DlssNgx::Resource&,
                           const DlssNgx::Resource&, const DlssNgx::Resource&,
                           const DlssNgx::EvalDesc&, const Camera&) {
    return false;
}

#endif

} // namespace Vulkan
