// SPDX-FileCopyrightText: Copyright 2026 IFreemz
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string_view>

#include <fmt/format.h>

#include "common/logging/log.h"
#include "common/path_util.h"
#include "video_core/renderer_vulkan/vk_bb_temporal_dlss.h"
#include "video_core/renderer_vulkan/vk_dlss_ngx.h"

#include "../../../dlss_bridge/shadps4_dlss_bridge.h"
#include "video_core/renderer_vulkan/vk_bb_library.h"

namespace Vulkan {

namespace {
constexpr char BridgeName[] = BB_LIBRARY_NAME("shadps4_dlss.dll", "shadps4_dlss.so");

// The DLSS SDK names its Linux library after its version (libnvidia-ngx-dlss.so.310.9.16).
bool HasNgxLibrary(const std::filesystem::path& directory) {
#ifdef _WIN32
    return std::filesystem::is_regular_file(directory / "nvngx_dlss.dll");
#else
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator{directory, error})
        if (entry.path().filename().string().starts_with("libnvidia-ngx-dlss.so.") &&
            entry.is_regular_file(error))
            return true;
    return false;
#endif
}
} // namespace

bool DlssNgx::Present() {
    static const bool present = [] {
        const char* setting = std::getenv("SHADPS4_DLSS");
        if (setting && std::string_view{setting} == "0")
            return false;
        const auto directory = BbLibrary::ExecutableDirectory();
        return std::filesystem::is_regular_file(directory / BridgeName) && HasNgxLibrary(directory);
    }();
    return present;
}

int DlssNgx::QualityForScale(float scale) {
    // DLAA only for equal sizes: any real upscale needs at least the Quality mode.
    return scale >= 2.9f ? 4 : scale >= 1.95f ? 3 : scale >= 1.65f ? 2 : scale > 1.01f ? 1 : 0;
}

#if defined(_WIN32) || defined(__linux__)

namespace {
// quick_exit skips Instance destructors; release NGX before the process ends.
std::mutex live_mutex;
std::vector<DlssNgx*> live_contexts;
std::once_flag exit_handler_once;

void ShutdownLiveContexts() {
    std::scoped_lock lock{live_mutex};
    for (auto* context : live_contexts)
        context->Shutdown();
}

void BridgeLog(int warning, const char* message) {
    if (warning)
        LOG_WARNING(Render_Vulkan, "[DLSS] {}", message);
    else
        LOG_INFO(Render_Vulkan, "[DLSS] {}", message);
}

bool CollectExtensions(const VkExtensionProperties* required, u32 count,
                       const std::vector<vk::ExtensionProperties>& available,
                       std::vector<std::string>& storage, std::vector<const char*>& enabled) {
    storage.clear();
    for (u32 i = 0; i < count; ++i) {
        const auto& request = required[i];
        const auto match = std::ranges::find_if(available, [&](const auto& extension) {
            return std::strcmp(extension.extensionName.data(), request.extensionName) == 0 &&
                   extension.specVersion >= request.specVersion;
        });
        if (match == available.end()) {
            LOG_WARNING(Render_Vulkan, "[DLSS] Missing Vulkan extension {}", request.extensionName);
            return false;
        }
        storage.emplace_back(request.extensionName);
    }
    for (const auto& name : storage)
        if (std::ranges::none_of(enabled, [&](const char* current) { return name == current; }))
            enabled.push_back(name.c_str());
    return true;
}
} // namespace

struct DlssNgx::Impl {
    void* module{};
    const ShadDlssApi* api{};
    std::vector<std::string> instance_extensions, device_extensions;
    std::optional<FeatureDesc> feature;
    bool eligible{true}, instance_ready{}, device_ready{}, initialized{}, available{};

    void Disable(std::string_view reason) {
        eligible = available = false;
        LOG_WARNING(Render_Vulkan, "[DLSS] Disabled: {}; using normal rendering", reason);
        BbTemporalDlssReportProblem(fmt::format("DLSS unavailable: {}.", reason));
    }
};

DlssNgx::DlssNgx() : impl{std::make_unique<Impl>()} {
    const auto directory = BbLibrary::ExecutableDirectory();
    impl->module = BbLibrary::Load(directory / BridgeName);
    const auto get_api = BbLibrary::Find<ShadDlssGetApiFn>(impl->module, "ShadDlssGetApi");
    impl->api = get_api ? get_api() : nullptr;
    if (!impl->api || impl->api->abi != SHADPS4_DLSS_BRIDGE_ABI) {
        impl->Disable(fmt::format("{} is missing or from a different version", BridgeName));
        return;
    }
    const auto data = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "dlss";
    std::error_code error;
    std::filesystem::create_directories(data, error);
    if (!impl->api->Configure(directory.wstring().c_str(), data.wstring().c_str(), BridgeLog))
        impl->Disable("bridge configuration failed");
}

std::unique_ptr<DlssNgx> DlssNgx::Create() {
    if (!Present())
        return {};
    auto context = std::unique_ptr<DlssNgx>{new DlssNgx};
    std::call_once(exit_handler_once, [] {
        std::at_quick_exit(ShutdownLiveContexts);
        std::atexit(ShutdownLiveContexts);
    });
    std::scoped_lock lock{live_mutex};
    live_contexts.push_back(context.get());
    return context;
}

DlssNgx::~DlssNgx() {
    std::scoped_lock lock{live_mutex};
    Shutdown();
    std::erase(live_contexts, this);
}

void DlssNgx::AppendInstanceExtensions(std::vector<const char*>& enabled) {
    if (!impl->eligible)
        return;
    u32 count{};
    const VkExtensionProperties* required{};
    if (!impl->api->InstanceExtensions(&count, &required))
        return impl->Disable("instance extension query failed");
    const auto [result, extensions] = vk::enumerateInstanceExtensionProperties();
    if (result != vk::Result::eSuccess ||
        !CollectExtensions(required, count, extensions, impl->instance_extensions, enabled))
        return impl->Disable("required Vulkan instance extensions are missing");
    impl->instance_ready = true;
}

void DlssNgx::AppendDeviceExtensions(vk::Instance instance, vk::PhysicalDevice physical,
                                     std::vector<const char*>& enabled) {
    if (!impl->eligible || !impl->instance_ready)
        return;
    const auto properties = physical.getProperties();
    BbTemporalDlssReportGpu(properties.deviceName.data());
    if (properties.vendorID != 0x10de)
        return impl->Disable("this is not an NVIDIA GPU");
    u32 count{};
    const VkExtensionProperties* required{};
    if (!impl->api->DeviceExtensions(instance, physical, &count, &required))
        return impl->Disable("this GPU or driver does not support DLSS (RTX GPU required)");
    const auto [result, extensions] = physical.enumerateDeviceExtensionProperties();
    if (result != vk::Result::eSuccess ||
        !CollectExtensions(required, count, extensions, impl->device_extensions, enabled))
        return impl->Disable("required Vulkan device extensions are missing");
    impl->device_ready = true;
}

void DlssNgx::Initialize(vk::Instance instance, vk::PhysicalDevice physical, vk::Device device) {
    if (!impl->eligible || !impl->device_ready || impl->initialized)
        return;
    impl->initialized = true;
    if (!impl->api->Initialize(instance, physical, device,
                               VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr,
                               VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr))
        return impl->Disable("NVIDIA NGX initialization failed (update the GPU driver)");
    impl->available = true;
    LOG_INFO(Render_Vulkan, "[DLSS] Ready");
}

bool DlssNgx::IsAvailable() const {
    return impl->available;
}

bool DlssNgx::HasFeature(const FeatureDesc& desc) const {
    const auto& d = impl->feature;
    return d && d->input_width == desc.input_width && d->input_height == desc.input_height &&
           d->output_width == desc.output_width && d->output_height == desc.output_height &&
           d->quality == desc.quality && d->depth_inverted == desc.depth_inverted &&
           d->preset == desc.preset && d->hdr == desc.hdr;
}

bool DlssNgx::CreateFeature(vk::CommandBuffer command, const FeatureDesc& desc) {
    if (!impl->available)
        return false;
    impl->feature.reset();
    const ShadDlssFeature feature{desc.input_width,   desc.input_height, desc.output_width,
                                  desc.output_height, desc.quality,      desc.depth_inverted,
                                  desc.preset,        desc.hdr ? 1 : 0};
    if (!impl->api->CreateFeature(command, &feature))
        return false;
    impl->feature = desc;
    return true;
}

bool DlssNgx::Evaluate(vk::CommandBuffer command, const Resource& color, const Resource& depth,
                       const Resource& motion, const Resource& output, const EvalDesc& eval) {
    if (!impl->available || !impl->feature)
        return false;
    const auto image = [](const Resource& r) {
        return ShadDlssImage{r.image,
                             r.view,
                             static_cast<VkImageSubresourceRange>(r.range),
                             static_cast<VkFormat>(r.format),
                             r.extent.width,
                             r.extent.height};
    };
    const ShadDlssEvaluate parameters{image(color),  image(depth),  image(motion), image(output),
                                      eval.jitter_x, eval.jitter_y, eval.reset,    eval.frame_ms};
    return impl->api->Evaluate(command, &parameters) != 0;
}

void DlssNgx::ReleaseFeatureAfterGpuDrain() {
    impl->feature.reset();
    if (impl->api)
        impl->api->ReleaseFeature();
}

void DlssNgx::Shutdown() {
    impl->available = false;
    impl->feature.reset();
    if (impl->api && impl->initialized) {
        impl->api->Shutdown();
        impl->initialized = false;
    }
}

#else

struct DlssNgx::Impl {};
DlssNgx::DlssNgx() = default;
DlssNgx::~DlssNgx() = default;
std::unique_ptr<DlssNgx> DlssNgx::Create() {
    return {};
}
void DlssNgx::AppendInstanceExtensions(std::vector<const char*>&) {}
void DlssNgx::AppendDeviceExtensions(vk::Instance, vk::PhysicalDevice, std::vector<const char*>&) {}
void DlssNgx::Initialize(vk::Instance, vk::PhysicalDevice, vk::Device) {}
void DlssNgx::Shutdown() {}
bool DlssNgx::IsAvailable() const {
    return false;
}
bool DlssNgx::CreateFeature(vk::CommandBuffer, const FeatureDesc&) {
    return false;
}
bool DlssNgx::HasFeature(const FeatureDesc&) const {
    return false;
}
bool DlssNgx::Evaluate(vk::CommandBuffer, const Resource&, const Resource&, const Resource&,
                       const Resource&, const EvalDesc&) {
    return false;
}
void DlssNgx::ReleaseFeatureAfterGpuDrain() {}

#endif

} // namespace Vulkan
