// SPDX-FileCopyrightText: Copyright 2026 IFreemz
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string_view>

#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_fsr4_addon.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

#include "../../../fsr4_addon/shadps4_fsr4.h"
#include "video_core/renderer_vulkan/vk_bb_library.h"

namespace Vulkan {

namespace {
constexpr char AddonName[] = BB_LIBRARY_NAME("shadps4_fsr4.dll", "shadps4_fsr4.so");

std::filesystem::path AddonDirectory() {
    return BbLibrary::ExecutableDirectory() / "fsr4";
}
} // namespace

bool Fsr4Addon::Present() {
    static const bool present = [] {
        const char* setting = std::getenv("SHADPS4_FSR4");
        if (setting && std::string_view{setting} == "0")
            return false;
        return std::filesystem::is_regular_file(AddonDirectory() / AddonName);
    }();
    return present;
}

#if defined(_WIN32) || defined(__linux__)

namespace {
void AddonLog(int warning, const char* message) {
    if (warning)
        LOG_WARNING(Render_Vulkan, "[FSR4] {}", message);
    else
        LOG_INFO(Render_Vulkan, "[FSR4] {}", message);
}

uint32_t FfxFormat(vk::Format format) {
    // FFX surface format values shared by FidelityFX and FSR-Vulkan.
    switch (format) {
    case vk::Format::eR8G8B8A8Unorm:
        return 10;
    case vk::Format::eR32Sfloat:
        return 28;
    case vk::Format::eR16G16Sfloat:
        return 18;
    case vk::Format::eR16G16B16A16Sfloat:
        return 4;
    default:
        return 0;
    }
}

ShadFsr4Image Image(const DlssNgx::Resource& r, vk::ImageLayout layout) {
    return {static_cast<VkImage>(r.image),
            static_cast<VkImageView>(r.view),
            static_cast<VkImageLayout>(layout),
            r.extent.width,
            r.extent.height,
            FfxFormat(r.format)};
}
} // namespace

struct Fsr4Addon::Impl {
    Scheduler& scheduler;
    void* module{};
    const ShadFsr4Api* api{};
    std::optional<ContextDesc> desc;
    u64 next_frame{1};
    std::deque<std::pair<u64, u64>> in_flight; // add-on frame id, scheduler tick
};

Fsr4Addon::Fsr4Addon(Scheduler& scheduler) : impl{std::make_unique<Impl>(scheduler)} {}

std::unique_ptr<Fsr4Addon> Fsr4Addon::Create(const Instance& instance, Scheduler& scheduler) {
    if (!Present())
        return {};
    if (!instance.IsFsr4Supported()) {
        LOG_WARNING(Render_Vulkan,
                    "[FSR4] This GPU lacks INT8 dot products or compute shader derivatives");
        return {};
    }
    auto addon = std::unique_ptr<Fsr4Addon>{new Fsr4Addon{scheduler}};
    auto& s = *addon->impl;
    const auto directory = AddonDirectory();
    s.module = BbLibrary::Load(directory / AddonName);
    const auto get_api = BbLibrary::Find<ShadFsr4GetApiFn>(s.module, "ShadFsr4GetApi");
    s.api = get_api ? get_api() : nullptr;
    if (!s.api || s.api->abi != SHADPS4_FSR4_ABI) {
        LOG_WARNING(Render_Vulkan, "[FSR4] {} is missing or from another version", AddonName);
        return {};
    }
    s.api->Configure(directory.wstring().c_str(), AddonLog);
    if (!s.api->Initialize(static_cast<VkPhysicalDevice>(instance.GetPhysicalDevice()),
                           static_cast<VkDevice>(instance.GetDevice())))
        return {};
    LOG_INFO(Render_Vulkan, "[FSR4] Ready");
    return addon;
}

Fsr4Addon::~Fsr4Addon() {
    ReleaseContextAfterGpuDrain();
    BbLibrary::Free(impl->module);
}

bool Fsr4Addon::HasContext(const ContextDesc& desc) const {
    const ShadFsr4Context c{desc.render.width, desc.render.height, desc.output.width,
                            desc.output.height};
    return impl->api && impl->api->HasContext(&c);
}

bool Fsr4Addon::CreateContext(const ContextDesc& desc) {
    const ShadFsr4Context c{desc.render.width, desc.render.height, desc.output.width,
                            desc.output.height};
    impl->in_flight.clear(); // the GPU was drained; a new context starts over
    return impl->api && impl->api->CreateContext(&c);
}

void Fsr4Addon::ReleaseContextAfterGpuDrain() {
    impl->in_flight.clear();
    if (impl->api)
        impl->api->ReleaseContext();
}

void Fsr4Addon::RetireFinished() {
    auto& s = *impl;
    while (!s.in_flight.empty() && s.scheduler.IsFree(s.in_flight.front().second)) {
        s.api->Retire(s.in_flight.front().first);
        s.in_flight.pop_front();
    }
}

bool Fsr4Addon::Evaluate(vk::CommandBuffer command, const DlssNgx::Resource& color,
                         const DlssNgx::Resource& depth, const DlssNgx::Resource& motion,
                         const DlssNgx::Resource& output, const DlssNgx::EvalDesc& eval,
                         const FsrUpscaler::Camera& camera, int hdr_mode) {
    auto& s = *impl;
    if (!s.api)
        return false;
    const auto ro = vk::ImageLayout::eShaderReadOnlyOptimal;
    const ShadFsr4Evaluate e{Image(color, ro),   Image(depth, ro),
                             Image(motion, ro),  Image(output, vk::ImageLayout::eGeneral),
                             eval.jitter_x,      eval.jitter_y,
                             eval.frame_ms,      camera.near_plane,
                             camera.far_plane,   camera.fov_y,
                             eval.reset ? 1 : 0, hdr_mode};
    RetireFinished();
    const u64 frame = s.next_frame++;
    int32_t result = s.api->Evaluate(static_cast<VkCommandBuffer>(command), &e, frame);
    while (result == SHADPS4_FSR4_BUSY && !s.in_flight.empty()) {
        s.scheduler.Wait(s.in_flight.front().second);
        RetireFinished();
        result = s.api->Evaluate(static_cast<VkCommandBuffer>(command), &e, frame);
    }
    if (result == SHADPS4_FSR4_OK || result == SHADPS4_FSR4_ABANDONED)
        s.in_flight.emplace_back(frame, s.scheduler.CurrentTick());
    return result == SHADPS4_FSR4_OK;
}

#else

struct Fsr4Addon::Impl {};
Fsr4Addon::Fsr4Addon(Scheduler&) {}
Fsr4Addon::~Fsr4Addon() = default;
std::unique_ptr<Fsr4Addon> Fsr4Addon::Create(const Instance&, Scheduler&) {
    return {};
}
bool Fsr4Addon::HasContext(const ContextDesc&) const {
    return false;
}
bool Fsr4Addon::CreateContext(const ContextDesc&) {
    return false;
}
void Fsr4Addon::ReleaseContextAfterGpuDrain() {}
void Fsr4Addon::RetireFinished() {}
bool Fsr4Addon::Evaluate(vk::CommandBuffer, const DlssNgx::Resource&, const DlssNgx::Resource&,
                         const DlssNgx::Resource&, const DlssNgx::Resource&,
                         const DlssNgx::EvalDesc&, const FsrUpscaler::Camera&, int) {
    return false;
}

#endif

} // namespace Vulkan
