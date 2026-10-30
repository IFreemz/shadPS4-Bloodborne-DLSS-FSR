// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <span>
#include <stdexcept>
#include <vector>
#include "common/logging/log.h"
#include "core/emulator_settings.h"
#include "video_core/host_shaders/bb_ssr_denoise_prefilter_comp.h"
#include "video_core/host_shaders/bb_ssr_denoise_reproject_comp.h"
#include "video_core/host_shaders/bb_ssr_denoise_resolve_comp.h"
#include "video_core/host_shaders/bb_ssr_denoise_resolve_target_comp.h"
#include "video_core/host_shaders/bb_ssr_down_comp.h"
#include "video_core/host_shaders/bb_ssr_hiz_comp.h"
#include "video_core/host_shaders/bb_ssr_trace_comp.h"
#include "video_core/host_shaders/bb_ssr_trace_full_comp.h"
#include "video_core/renderer_vulkan/vk_bb_ssr.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/texture_cache/image.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/texture_cache.h"

namespace Vulkan {

namespace {

template <typename T>
T CheckOrThrow(vk::ResultValue<T> result) {
    if (result.result != vk::Result::eSuccess) {
        throw std::runtime_error(vk::to_string(result.result));
    }
    return std::move(result.value);
}

struct PyramidConstants {
    std::array<s32, 2> size;
    std::array<s32, 2> depth_size;
    s32 level;
};

struct TraceConstants {
    std::array<float, 2> size;
    std::array<float, 2> normal_scale;
    u32 offset;
    s32 max_level;
    s32 max_steps;
    float edge_fade;
    float cone;
    float max_lod;
};

struct DenoiseConstants {
    std::array<float, 2> size;
    std::array<float, 2> normal_scale;
    u32 offset;
    u32 has_gloss;
    u32 history;
    float roughness;
};

struct DownConstants {
    std::array<s32, 2> size;
    std::array<s32, 2> src_size;
    s32 src_level;
};

constexpr u32 PyramidLevels = 6; ///< cells of up to 32x32 pixels
constexpr s32 HalfSteps = 96;
constexpr s32 FullSteps = 128;
constexpr float EdgeFade = 0.08f;
/// The denoiser's roughness without the gloss (half resolution).
constexpr float HalfResRoughness = 0.3f;
constexpr u32 ConstantCount = 80; ///< of the game's constants read
constexpr u32 ColorLevels = 8;    ///< of the full-size copy of the last frame
/// Full resolution: the blur grows by this many pixels per pixel of ray length (up to the
/// game's blur for the surface's gloss), so reflections are sharp where they touch.
constexpr float Cone = 0.15f;
/// History, and the last frame's colour, older than this is not used (a pause, a loading
/// screen, the setting off for a while).
constexpr auto MaxAge = std::chrono::milliseconds(250);

constexpr auto Rgba16f = vk::Format::eR16G16B16A16Sfloat;
constexpr auto ComputeStage = vk::PipelineStageFlagBits2::eComputeShader;
constexpr auto ReaderStages =
    vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eFragmentShader;
constexpr auto Read = vk::AccessFlagBits2::eShaderRead;
constexpr auto Write = vk::AccessFlagBits2::eShaderWrite;

bool IsDepth32(vk::Format format) {
    return format == vk::Format::eD32Sfloat || format == vk::Format::eD32SfloatS8Uint;
}

vk::ImageMemoryBarrier2 ImageBarrier(vk::Image image, u32 base_level, u32 levels,
                                     vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src_access,
                                     vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access,
                                     vk::ImageLayout old_layout) {
    return vk::ImageMemoryBarrier2{
        .srcStageMask = src_stage,
        .srcAccessMask = src_access,
        .dstStageMask = dst_stage,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = vk::ImageLayout::eGeneral,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, base_level, levels, 0, 1},
    };
}

void Barriers(vk::CommandBuffer cmdbuf, std::span<const vk::ImageMemoryBarrier2> barriers) {
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = static_cast<u32>(barriers.size()),
        .pImageMemoryBarriers = barriers.data(),
    });
}

void Barrier(vk::CommandBuffer cmdbuf, const vk::ImageMemoryBarrier2& barrier) {
    Barriers(cmdbuf, std::span(&barrier, 1));
}

/// One descriptor of a pass (all in General layout).
struct Bind {
    u32 binding;
    vk::DescriptorType type;
    vk::ImageView view{};
    vk::Sampler sampler{};
    vk::DescriptorBufferInfo buffer{};
};

void PushBinds(vk::CommandBuffer cmdbuf, vk::PipelineLayout layout, std::span<const Bind> binds) {
    std::array<vk::DescriptorImageInfo, 16> infos{};
    std::array<vk::WriteDescriptorSet, 16> writes{};
    for (u32 i = 0; i < binds.size(); ++i) {
        const auto& bind = binds[i];
        infos[i] = {.sampler = bind.sampler,
                    .imageView = bind.view,
                    .imageLayout = vk::ImageLayout::eGeneral};
        const bool buffer = bind.type == vk::DescriptorType::eStorageBuffer;
        writes[i] = {.dstBinding = bind.binding,
                     .descriptorCount = 1,
                     .descriptorType = bind.type,
                     .pImageInfo = buffer ? nullptr : &infos[i],
                     .pBufferInfo = buffer ? &bind.buffer : nullptr};
    }
    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0,
                                std::span(writes.data(), binds.size()));
}

template <typename T>
void Dispatch(vk::CommandBuffer cmdbuf, vk::PipelineLayout layout, const T& push, u32 width,
              u32 height) {
    cmdbuf.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
    cmdbuf.dispatch((width + 7) / 8, (height + 7) / 8, 1);
}

using T = vk::DescriptorType;

struct Img {
    vk::Image image{};
    vk::ImageView view{};
};

/// What one trace records (handles only: it runs on the recording thread).
struct TraceJob {
    vk::Image pyramid{};
    vk::ImageView pyramid_view{};
    std::array<vk::ImageView, PyramidLevels> levels{};
    std::array<u32, 2> pyramid_size{};
    std::array<s32, 2> depth_size{};
    vk::ImageView depth_view{};
    vk::ImageView normal_view{};
    vk::ImageView color_view{};
    vk::ImageView gloss_view{}; ///< full resolution only
    vk::Sampler color_sampler{};
    vk::Sampler history_sampler{};
    vk::Image current{};
    vk::ImageView current_view{};
    vk::Image distance{};
    vk::ImageView distance_view{};
    /// The denoiser's images of this frame, and of this frame and the last (history: the
    /// reflections, their statistics and the surface).
    std::array<Img, 5> scratch{}; ///< reprojected, variance and frames, averages, filtered, var.
    std::array<Img, 3> write{}, read{};
    bool clear_history{};
    vk::DescriptorBufferInfo constants{};
    TraceConstants trace_push{};
    DenoiseConstants denoise_push{};
    std::array<u32, 2> average_size{};
    vk::ImageView target_view{}; ///< the resolve writes it too (half resolution)
    vk::Pipeline pyramid_pipeline{}, trace_pipeline{};
    std::array<vk::Pipeline, 3> denoise_pipelines{}; ///< reproject, prefilter, resolve
    vk::PipelineLayout layout{}, denoise_layout{};
};

void RecordTrace(vk::CommandBuffer cmdbuf, const TraceJob& job) {
    const auto layout = job.layout;
    // The game's constants may have just been copied in.
    const vk::MemoryBarrier2 constants_barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = ComputeStage,
        .dstAccessMask = Read,
    };
    // Rewritten whole each frame: their old contents are not kept.
    const std::array<vk::ImageMemoryBarrier2, 3> scratch = {
        ImageBarrier(job.pyramid, 0, PyramidLevels, ComputeStage, Read, ComputeStage, Write,
                     vk::ImageLayout::eUndefined),
        ImageBarrier(job.current, 0, 1, ComputeStage, Read, ComputeStage, Write,
                     vk::ImageLayout::eUndefined),
        ImageBarrier(job.distance, 0, 1, ComputeStage, Read, ComputeStage, Write,
                     vk::ImageLayout::eUndefined),
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1,
                                               .pMemoryBarriers = &constants_barrier,
                                               .imageMemoryBarrierCount = 3,
                                               .pImageMemoryBarriers = scratch.data()});

    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, job.pyramid_pipeline);
    for (u32 level = 0; level < PyramidLevels; ++level) {
        const std::array<Bind, 3> binds = {{
            {0, T::eSampledImage, job.depth_view},
            {1, T::eStorageImage, job.levels[level]},
            {2, T::eSampledImage, job.pyramid_view},
        }};
        PushBinds(cmdbuf, layout, binds);
        const PyramidConstants push{
            .size = {s32(job.pyramid_size[0] >> level), s32(job.pyramid_size[1] >> level)},
            .depth_size = job.depth_size,
            .level = s32(level),
        };
        Dispatch(cmdbuf, layout, push, u32(push.size[0]), u32(push.size[1]));
        Barrier(cmdbuf, ImageBarrier(job.pyramid, level, 1, ComputeStage, Write, ComputeStage, Read,
                                     vk::ImageLayout::eGeneral));
    }

    const u32 width = u32(job.trace_push.size[0]);
    const u32 height = u32(job.trace_push.size[1]);
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, job.trace_pipeline);
    {
        std::array<Bind, 7> binds = {{
            {0, T::eSampledImage, job.pyramid_view},
            {1, T::eStorageImage, job.current_view},
            {2, T::eSampledImage, job.normal_view},
            {3, T::eCombinedImageSampler, job.color_view, job.color_sampler},
            {4, T::eStorageBuffer, {}, {}, job.constants},
            {5, T::eStorageImage, job.distance_view},
            {6, T::eSampledImage, job.gloss_view},
        }};
        PushBinds(cmdbuf, layout, std::span(binds.data(), job.gloss_view ? 7 : 6));
        Dispatch(cmdbuf, layout, job.trace_push, width, height);
    }

    // The denoiser. Its history images live in General layout; new ones start from Undefined.
    // The lighting may have read the reflections.
    const auto history_layout =
        job.clear_history ? vk::ImageLayout::eUndefined : vk::ImageLayout::eGeneral;
    const auto undefined = vk::ImageLayout::eUndefined;
    const auto general = vk::ImageLayout::eGeneral;
    const auto [reprojected, stats, averages, filtered, filtered_variance] = job.scratch;
    {
        std::array<vk::ImageMemoryBarrier2, 11> barriers = {
            ImageBarrier(job.current, 0, 1, ComputeStage, Write, ComputeStage, Read, general),
            ImageBarrier(job.distance, 0, 1, ComputeStage, Write, ComputeStage, Read, general),
            ImageBarrier(reprojected.image, 0, 1, ComputeStage, Read, ComputeStage, Write,
                         undefined),
            ImageBarrier(stats.image, 0, 1, ComputeStage, Read, ComputeStage, Write, undefined),
            ImageBarrier(averages.image, 0, 1, ComputeStage, Read, ComputeStage, Write, undefined),
        };
        for (u32 i = 0; i < 3; ++i) {
            barriers[5 + i] = ImageBarrier(job.write[i].image, 0, 1, ReaderStages, Read | Write,
                                           ComputeStage, Write, history_layout);
            barriers[8 + i] = ImageBarrier(job.read[i].image, 0, 1, ReaderStages, Read | Write,
                                           ComputeStage, Read, history_layout);
        }
        Barriers(cmdbuf, barriers);
    }
    const auto dlayout = job.denoise_layout;
    const Bind constants{16, T::eStorageBuffer, {}, {}, job.constants};
    const Bind depth{2, T::eSampledImage, job.pyramid_view};
    const Bind normals{3, T::eSampledImage, job.normal_view};
    const Bind gloss{4, T::eSampledImage, job.gloss_view ? job.gloss_view : job.normal_view};
    const auto sampled = [&](u32 binding, vk::ImageView view) {
        return Bind{binding, T::eCombinedImageSampler, view, job.history_sampler};
    };
    const auto texture = [](u32 binding, vk::ImageView view) {
        return Bind{binding, T::eSampledImage, view};
    };
    const auto storage = [](u32 binding, vk::ImageView view) {
        return Bind{binding, T::eStorageImage, view};
    };

    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, job.denoise_pipelines[0]);
    {
        const std::array<Bind, 12> binds = {
            texture(0, job.current_view),
            texture(1, job.distance_view),
            depth,
            normals,
            gloss,
            sampled(8, job.read[0].view),
            sampled(9, job.read[1].view),
            sampled(10, job.read[2].view),
            storage(12, reprojected.view),
            storage(13, stats.view),
            storage(14, averages.view),
            constants,
        };
        PushBinds(cmdbuf, dlayout, binds);
        Dispatch(cmdbuf, dlayout, job.denoise_push, width, height);
    }
    {
        const std::array<vk::ImageMemoryBarrier2, 5> barriers = {
            ImageBarrier(reprojected.image, 0, 1, ComputeStage, Write, ComputeStage, Read, general),
            ImageBarrier(stats.image, 0, 1, ComputeStage, Write, ComputeStage, Read, general),
            ImageBarrier(averages.image, 0, 1, ComputeStage, Write, ComputeStage, Read, general),
            ImageBarrier(filtered.image, 0, 1, ComputeStage, Read, ComputeStage, Write, undefined),
            ImageBarrier(filtered_variance.image, 0, 1, ComputeStage, Read, ComputeStage, Write,
                         undefined),
        };
        Barriers(cmdbuf, barriers);
    }
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, job.denoise_pipelines[1]);
    {
        const std::array<Bind, 9> binds = {
            texture(0, job.current_view),
            texture(1, stats.view),
            depth,
            normals,
            gloss,
            sampled(8, averages.view),
            storage(12, filtered.view),
            storage(13, filtered_variance.view),
            constants,
        };
        PushBinds(cmdbuf, dlayout, binds);
        Dispatch(cmdbuf, dlayout, job.denoise_push, width, height);
    }
    {
        const std::array<vk::ImageMemoryBarrier2, 2> barriers = {
            ImageBarrier(filtered.image, 0, 1, ComputeStage, Write, ComputeStage, Read, general),
            ImageBarrier(filtered_variance.image, 0, 1, ComputeStage, Write, ComputeStage, Read,
                         general),
        };
        Barriers(cmdbuf, barriers);
    }
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, job.denoise_pipelines[2]);
    {
        const std::array<Bind, 13> binds = {
            texture(0, filtered.view),
            texture(1, filtered_variance.view),
            depth,
            normals,
            gloss,
            texture(5, stats.view),
            texture(6, reprojected.view),
            sampled(8, averages.view),
            storage(12, job.write[0].view),
            storage(13, job.write[1].view),
            storage(14, job.write[2].view),
            constants,
            storage(15, job.target_view),
        };
        PushBinds(cmdbuf, dlayout, std::span(binds.data(), job.target_view ? 13 : 12));
        Dispatch(cmdbuf, dlayout, job.denoise_push, width, height);
    }
    {
        const std::array<vk::ImageMemoryBarrier2, 3> barriers = {
            ImageBarrier(job.write[0].image, 0, 1, ComputeStage, Write, ReaderStages, Read,
                         general),
            ImageBarrier(job.write[1].image, 0, 1, ComputeStage, Write, ComputeStage, Read,
                         general),
            ImageBarrier(job.write[2].image, 0, 1, ComputeStage, Write, ComputeStage, Read,
                         general),
        };
        Barriers(cmdbuf, barriers);
    }
}

} // Anonymous namespace

struct BbSsr::Impl {
    const Instance& instance;
    Scheduler& scheduler;
    Runtime& runtime;
    VideoCore::TextureCache& texture_cache;
    vk::Device device;
    bool failed{};

    vk::UniqueDescriptorSetLayout desc_layout, denoise_desc_layout;
    vk::UniquePipelineLayout pipeline_layout, denoise_layout;
    vk::UniquePipeline pyramid_pipeline;
    vk::UniquePipeline trace_pipeline, trace_full_pipeline;
    vk::UniquePipeline reproject_pipeline, prefilter_pipeline, resolve_pipeline;
    vk::UniquePipeline resolve_target_pipeline;
    vk::UniquePipeline down_pipeline;
    vk::UniqueSampler sampler;     ///< linear, level 0
    vk::UniqueSampler mip_sampler; ///< linear, between levels

    /// An image with a view of all its levels and one of each.
    struct Leveled {
        std::unique_ptr<VideoCore::UniqueImage> image;
        vk::UniqueImageView view;
        std::vector<vk::UniqueImageView> levels;
        std::array<u32, 2> size{};
    };
    /// A trace's images at one size: the depth pyramid (level 0 padded to whole cells of the
    /// coarsest level, so each level is exactly half the one below), this frame's reflections,
    /// the denoiser's images of this frame, and its history (the reflections, their statistics
    /// and the surface; this frame's and the last one's, by turns).
    struct Traced {
        Leveled pyramid;
        Leveled current;
        Leveled distance; ///< from each pixel's surface to its rays' hits
        std::array<Leveled, 5> scratch;
        std::array<std::array<Leveled, 3>, 2> history;
        u32 history_index{};
        bool history_cleared{}; ///< the history images are in General layout
        std::chrono::steady_clock::time_point last_run{};
        std::array<u32, 2> size{};
    };
    Traced half, full;

    // Full resolution: the full-size copy of the last frame, and what the frame's passes showed.
    Leveled color;
    std::chrono::steady_clock::time_point color_time{};
    VideoCore::ImageId color_source{}; ///< the half-size colour the trace reads
    VideoCore::ImageId scene_depth{};
    u64 scene_depth_area{};
    VideoCore::ImageId gloss{};
    VAddr gloss_address{};
    vk::ImageView lighting_view{}; ///< this frame's full-size reflections, until the lighting

    struct Stats {
        u64 passes{}, traced{}, full{}, not_fitting{}, no_constants{}, color_copies{};
        u64 lighting_reads{};
    } stats;
    std::chrono::steady_clock::time_point next_log{};
    bool constants_logged{}, full_logged{};

    Impl(const Instance& instance_, Scheduler& scheduler_, Runtime& runtime_,
         VideoCore::TextureCache& texture_cache_)
        : instance{instance_}, scheduler{scheduler_}, runtime{runtime_},
          texture_cache{texture_cache_}, device{instance_.GetDevice()} {}

    void LogStats() {
        const auto now = std::chrono::steady_clock::now();
        if (now < next_log) {
            return;
        }
        if (next_log != std::chrono::steady_clock::time_point{} && stats.passes) {
            LOG_INFO(Render_Vulkan,
                     "Better reflections (5 s): game's reflection passes {}, traced {} (at full "
                     "resolution {}, read by the lighting {}; colour copies {}); game's kept: "
                     "targets not fitting {}, no constants {}",
                     stats.passes, stats.traced, stats.full, stats.lighting_reads,
                     stats.color_copies, stats.not_fitting, stats.no_constants);
        }
        stats = {};
        next_log = now + std::chrono::seconds(5);
    }

    void CreatePipelines() {
        const std::array<T, 7> types = {
            T::eSampledImage,  T::eStorageImage, T::eSampledImage, T::eCombinedImageSampler,
            T::eStorageBuffer, T::eStorageImage, T::eSampledImage};
        std::array<vk::DescriptorSetLayoutBinding, 7> bindings{};
        for (u32 i = 0; i < bindings.size(); ++i) {
            bindings[i] = {.binding = i,
                           .descriptorType = types[i],
                           .descriptorCount = 1,
                           .stageFlags = vk::ShaderStageFlagBits::eCompute};
        }
        desc_layout = CheckOrThrow(device.createDescriptorSetLayoutUnique({
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
            .bindingCount = static_cast<u32>(bindings.size()),
            .pBindings = bindings.data(),
        }));
        const vk::PushConstantRange range{
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .offset = 0,
            .size = u32(std::max({sizeof(PyramidConstants), sizeof(TraceConstants),
                                  sizeof(DenoiseConstants), sizeof(DownConstants)}))};
        const auto set = *desc_layout;
        pipeline_layout = CheckOrThrow(device.createPipelineLayoutUnique({
            .setLayoutCount = 1,
            .pSetLayouts = &set,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &range,
        }));
        // The denoiser's: textures 0 to 6, sampled 8 to 10, written 12 to 15, the constants 16.
        std::vector<vk::DescriptorSetLayoutBinding> denoise_bindings;
        for (u32 i = 0; i <= 16; ++i) {
            if (i == 7 || i == 11) {
                continue;
            }
            const auto type = i < 7    ? T::eSampledImage
                              : i < 11 ? T::eCombinedImageSampler
                              : i < 16 ? T::eStorageImage
                                       : T::eStorageBuffer;
            denoise_bindings.push_back({.binding = i,
                                        .descriptorType = type,
                                        .descriptorCount = 1,
                                        .stageFlags = vk::ShaderStageFlagBits::eCompute});
        }
        denoise_desc_layout = CheckOrThrow(device.createDescriptorSetLayoutUnique({
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
            .bindingCount = static_cast<u32>(denoise_bindings.size()),
            .pBindings = denoise_bindings.data(),
        }));
        const auto denoise_set = *denoise_desc_layout;
        denoise_layout = CheckOrThrow(device.createPipelineLayoutUnique({
            .setLayoutCount = 1,
            .pSetLayouts = &denoise_set,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &range,
        }));
        const auto create = [&](std::span<const u32> code,
                                vk::PipelineLayout layout = vk::PipelineLayout{}) {
            const auto module = CompileSPV(code, device);
            auto result = device.createComputePipelineUnique(
                {}, vk::ComputePipelineCreateInfo{
                        .stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                                  .module = module,
                                  .pName = "main"},
                        .layout = layout ? layout : *pipeline_layout,
                    });
            device.destroyShaderModule(module);
            return CheckOrThrow(std::move(result));
        };
        pyramid_pipeline = create(BB_SSR_HIZ_COMP);
        trace_pipeline = create(BB_SSR_TRACE_COMP);
        trace_full_pipeline = create(BB_SSR_TRACE_FULL_COMP);
        reproject_pipeline = create(BB_SSR_DENOISE_REPROJECT_COMP, *denoise_layout);
        prefilter_pipeline = create(BB_SSR_DENOISE_PREFILTER_COMP, *denoise_layout);
        resolve_pipeline = create(BB_SSR_DENOISE_RESOLVE_COMP, *denoise_layout);
        resolve_target_pipeline = create(BB_SSR_DENOISE_RESOLVE_TARGET_COMP, *denoise_layout);
        down_pipeline = create(BB_SSR_DOWN_COMP);
        const auto make_sampler = [&](bool mips) {
            return CheckOrThrow(device.createSamplerUnique(vk::SamplerCreateInfo{
                .magFilter = vk::Filter::eLinear,
                .minFilter = vk::Filter::eLinear,
                .mipmapMode = vk::SamplerMipmapMode::eLinear,
                .addressModeU = vk::SamplerAddressMode::eClampToEdge,
                .addressModeV = vk::SamplerAddressMode::eClampToEdge,
                .addressModeW = vk::SamplerAddressMode::eClampToEdge,
                .maxLod = mips ? float(ColorLevels) : 0.0f,
            }));
        };
        sampler = make_sampler(false);
        mip_sampler = make_sampler(true);
    }

    Leveled MakeLeveled(vk::Format format, u32 width, u32 height, u32 levels) {
        Leveled out;
        out.image = std::make_unique<VideoCore::UniqueImage>(device, instance.GetAllocator());
        out.image->Create(vk::ImageCreateInfo{
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent = {width, height, 1},
            .mipLevels = levels,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
                     vk::ImageUsageFlagBits::eTransferDst,
            .initialLayout = vk::ImageLayout::eUndefined,
        });
        const auto view = [&](u32 base, u32 count) {
            return CheckOrThrow(device.createImageViewUnique({
                .image = out.image->image,
                .viewType = vk::ImageViewType::e2D,
                .format = format,
                .subresourceRange = {vk::ImageAspectFlagBits::eColor, base, count, 0, 1},
            }));
        };
        out.view = view(0, levels);
        if (levels > 1) {
            for (u32 level = 0; level < levels; ++level) {
                out.levels.push_back(view(level, 1));
            }
        }
        out.size = {width, height};
        return out;
    }

    /// The images of a trace of `width` x `height`.
    void EnsureTraced(Traced& traced, u32 width, u32 height) {
        if (traced.current.image && traced.size == std::array{width, height}) {
            return;
        }
        if (traced.current.image) {
            // The old ones may still be in use by the GPU.
            scheduler.DeferOperation([old = std::move(traced)] {});
            traced = {};
        }
        constexpr u32 cell = 1u << (PyramidLevels - 1);
        traced.pyramid = MakeLeveled(vk::Format::eR32Sfloat, (width + cell - 1) / cell * cell,
                                     (height + cell - 1) / cell * cell, PyramidLevels);
        traced.current = MakeLeveled(Rgba16f, width, height, 1);
        traced.distance = MakeLeveled(vk::Format::eR32Sfloat, width, height, 1);
        for (u32 i = 0; i < traced.scratch.size(); ++i) {
            // The tile averages: one per 8x8 pixels.
            traced.scratch[i] = i == 2 ? MakeLeveled(Rgba16f, (width + 7) / 8, (height + 7) / 8, 1)
                                       : MakeLeveled(Rgba16f, width, height, 1);
        }
        for (auto& images : traced.history) {
            for (auto& image : images) {
                image = MakeLeveled(Rgba16f, width, height, 1);
            }
        }
        traced.size = {width, height};
    }

    /// A view the texture cache keeps with the image (made once, dropped with the image).
    vk::ImageView CacheView(VideoCore::Image& image, vk::Format format, bool storage) {
        VideoCore::ImageViewInfo info{};
        info.type = AmdGpu::ImageType::Color2D;
        info.format = format;
        info.is_storage = storage;
        return *image.FindView(info, false).image_view;
    }

    /// A colour image that is still `id` and can be read as one level of `width` x `height`.
    bool Readable(VideoCore::ImageId id, u32 width, u32 height) {
        if (!id || !texture_cache.IsImageAllocated(id)) {
            return false;
        }
        const auto& image = texture_cache.GetImage(id);
        return image.backing && image.info.num_samples == 1 && image.dropped_levels == 0 &&
               image.info.size.width == width && image.info.size.height == height;
    }

    bool Fits(VideoCore::ImageId target_id, vk::Format view_format, VideoCore::ImageId depth_id,
              VideoCore::ImageId normals_id, VideoCore::ImageId color_id) {
        if (!target_id || !depth_id || !normals_id || !color_id || target_id == depth_id ||
            target_id == normals_id || target_id == color_id) {
            return false;
        }
        const auto& target = texture_cache.GetImage(target_id);
        const auto& depth = texture_cache.GetImage(depth_id);
        const auto& normals = texture_cache.GetImage(normals_id);
        const auto& color = texture_cache.GetImage(color_id);
        const bool fits =
            target.info.pixel_format == Rgba16f &&
            (view_format == vk::Format::eUndefined || view_format == Rgba16f) &&
            bool(target.usage_flags & vk::ImageUsageFlagBits::eStorage) &&
            target.dropped_levels == 0 && target.info.num_samples == 1 &&
            IsDepth32(depth.info.pixel_format) && depth.info.num_samples == 1 &&
            depth.dropped_levels == 0 && target.info.size.width == depth.info.size.width &&
            target.info.size.height == depth.info.size.height && !normals.info.props.is_depth &&
            normals.backing && normals.info.num_samples == 1 && normals.dropped_levels == 0 &&
            !color.info.props.is_depth && color.backing && color.info.num_samples == 1 &&
            color.dropped_levels == 0;
        if (!fits) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                LOG_WARNING(Render_Vulkan,
                            "Better reflections: skipped (target {} {}x{} drawn as {}, depth {} "
                            "{}x{}, normals {} {}x{}, colour {} {}x{})",
                            vk::to_string(target.info.pixel_format), target.info.size.width,
                            target.info.size.height, vk::to_string(view_format),
                            vk::to_string(depth.info.pixel_format), depth.info.size.width,
                            depth.info.size.height, vk::to_string(normals.info.pixel_format),
                            normals.info.size.width, normals.info.size.height,
                            vk::to_string(color.info.pixel_format), color.info.size.width,
                            color.info.size.height);
            }
        }
        return fits;
    }

    /// Whether this frame can be traced at full resolution, at the size of `normals`; logs why
    /// not once.
    bool FullResReady(const VideoCore::Image& normals, std::chrono::steady_clock::time_point now) {
        if (!EmulatorSettings.IsBbSsrFullRes()) {
            return false;
        }
        const u32 width = normals.info.size.width;
        const u32 height = normals.info.size.height;
        const char* missing = nullptr;
        if (!Readable(scene_depth, width, height) ||
            !IsDepth32(texture_cache.GetImage(scene_depth).info.pixel_format)) {
            missing = "the scene's depth";
        } else if (!Readable(gloss, width, height) ||
                   texture_cache.GetImage(gloss).info.guest_address != gloss_address ||
                   texture_cache.GetImage(gloss).info.props.is_depth) {
            missing = "the gloss (from the lighting pass)";
        } else if (!color.image || color.size != std::array{width, height} ||
                   now - color_time > MaxAge) {
            missing = "the last frame's colour";
        }
        if (missing && !full_logged && stats.passes > 4) {
            full_logged = true;
            LOG_INFO(Render_Vulkan,
                     "Better reflections: not at full resolution yet ({} {}x{} not found)", missing,
                     width, height);
        }
        return !missing;
    }

    bool Run(VideoCore::ImageId target_id, vk::Format view_format, VideoCore::ImageId depth_id,
             VideoCore::ImageId normals_id, VideoCore::ImageId color_id, const Constants& constants,
             const void* guest_constants) {
        lighting_view = vk::ImageView{};
        if (!constants.buffer || (u64(constants.offset) + ConstantCount) * 4 > constants.size) {
            ++stats.no_constants;
            return false;
        }
        if (!Fits(target_id, view_format, depth_id, normals_id, color_id)) {
            ++stats.not_fitting;
            return false;
        }
        if (!pyramid_pipeline) {
            CreatePipelines();
        }
        color_source = color_id;
        auto& target = texture_cache.GetImage(target_id);
        auto& normals = texture_cache.GetImage(normals_id);
        const auto now = std::chrono::steady_clock::now();
        const bool full_res = FullResReady(normals, now);
        auto& depth = texture_cache.GetImage(full_res ? scene_depth : depth_id);
        const u32 width = depth.info.size.width;
        const u32 height = depth.info.size.height;
        auto& traced = full_res ? full : half;
        EnsureTraced(traced, width, height);

        if (!constants_logged && guest_constants) {
            constants_logged = true;
            const auto* c = static_cast<const float*>(guest_constants);
            LOG_INFO(Render_Vulkan,
                     "Better reflections: on, {}x{} (normals {}x{}); the game's lowest direction "
                     "{}, depth to view z {} {}, steps {}, thickness {} {}, farthest {}, step "
                     "scale {}",
                     width, height, normals.info.size.width, normals.info.size.height, c[64], c[68],
                     c[69], c[72], c[76], c[77], c[78], c[79]);
        }
        if (full_res && !full_logged) {
            full_logged = true;
            LOG_INFO(Render_Vulkan, "Better reflections: at full resolution, {}x{}", width, height);
        }

        TraceJob job{};
        job.pyramid = traced.pyramid.image->image;
        job.pyramid_view = *traced.pyramid.view;
        for (u32 level = 0; level < PyramidLevels; ++level) {
            job.levels[level] = *traced.pyramid.levels[level];
        }
        job.pyramid_size = traced.pyramid.size;
        job.depth_size = {s32(width), s32(height)};
        job.depth_view = CacheView(depth, vk::Format::eR32Sfloat, false);
        job.normal_view = CacheView(normals, normals.info.pixel_format, false);
        job.history_sampler = *sampler;
        job.current = traced.current.image->image;
        job.current_view = *traced.current.view;
        job.distance = traced.distance.image->image;
        job.distance_view = *traced.distance.view;
        const u32 write_index = traced.history_index ^ 1u;
        traced.history_index = write_index;
        const auto img = [](const Leveled& image) { return Img{image.image->image, *image.view}; };
        for (u32 i = 0; i < 5; ++i) {
            job.scratch[i] = img(traced.scratch[i]);
        }
        for (u32 i = 0; i < 3; ++i) {
            job.write[i] = img(traced.history[write_index][i]);
            job.read[i] = img(traced.history[write_index ^ 1u][i]);
        }
        // History from the last frame only.
        const bool history_valid = traced.history_cleared && now - traced.last_run < MaxAge;
        job.clear_history = !traced.history_cleared;
        traced.history_cleared = true;
        traced.last_run = now;
        job.constants = {constants.buffer, constants.binding_offset, constants.size};
        job.trace_push = TraceConstants{
            .size = {float(width), float(height)},
            .normal_scale = {float(normals.info.size.width) / float(width),
                             float(normals.info.size.height) / float(height)},
            .offset = constants.offset,
            .max_level = s32(PyramidLevels) - 1,
            .max_steps = full_res ? FullSteps : HalfSteps,
            .edge_fade = EdgeFade,
            .cone = Cone,
            .max_lod = float(ColorLevels - 1),
        };
        job.denoise_push = DenoiseConstants{
            .size = job.trace_push.size,
            .normal_scale = job.trace_push.normal_scale,
            .offset = constants.offset,
            .has_gloss = full_res ? 1u : 0u,
            .history = history_valid ? 1u : 0u,
            .roughness = HalfResRoughness,
        };
        job.pyramid_pipeline = *pyramid_pipeline;
        job.layout = *pipeline_layout;
        job.denoise_layout = *denoise_layout;
        const auto target_view = CacheView(target, Rgba16f, true);
        if (full_res) {
            auto& gloss_image = texture_cache.GetImage(gloss);
            job.gloss_view = CacheView(gloss_image, gloss_image.info.pixel_format, false);
            job.color_view = *color.view;
            job.color_sampler = *mip_sampler;
            job.trace_pipeline = *trace_full_pipeline;
            job.denoise_pipelines = {*reproject_pipeline, *prefilter_pipeline, *resolve_pipeline};
        } else {
            auto& color_image = texture_cache.GetImage(color_id);
            job.color_view = CacheView(color_image, color_image.info.pixel_format, false);
            job.color_sampler = *sampler;
            job.trace_pipeline = *trace_pipeline;
            job.denoise_pipelines = {*reproject_pipeline, *prefilter_pipeline,
                                     *resolve_target_pipeline};
            job.target_view = target_view;
        }

        scheduler.EndRendering();
        const auto read = [&](VideoCore::Image& image) {
            runtime.Transit(&image, vk::ImageLayout::eGeneral, ComputeStage, Read);
        };
        read(depth);
        read(normals);
        if (full_res) {
            read(texture_cache.GetImage(gloss));
        } else {
            read(texture_cache.GetImage(color_id));
        }
        // Only level 0: the game's own pass makes the others from it.
        runtime.Transit(&target, vk::ImageLayout::eGeneral, ComputeStage, Write,
                        VideoCore::SubresourceRange{.base = {0, 0}, .extent = {1, 1}});
        runtime.FlushBarriers();

        const DownConstants down_push{
            .size = {s32(target.info.size.width), s32(target.info.size.height)},
            .src_size = {s32(width), s32(height)},
            .src_level = 0,
        };
        scheduler.Record([job, full_res, target_view, down_push,
                          down_pipeline = *down_pipeline](vk::CommandBuffer cmdbuf) {
            RecordTrace(cmdbuf, job);
            if (!full_res) {
                return;
            }
            // The game's target gets the result halved, for its mips and the passes that read
            // them (the lighting reads the full-size result).
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, down_pipeline);
            const std::array<Bind, 2> binds = {{
                {0, T::eSampledImage, job.write[0].view},
                {1, T::eStorageImage, target_view},
            }};
            PushBinds(cmdbuf, job.layout, binds);
            Dispatch(cmdbuf, job.layout, down_push, u32(down_push.size[0]), u32(down_push.size[1]));
        });
        if (full_res) {
            lighting_view = job.write[0].view;
            ++stats.full;
        }
        return true;
    }

    /// Reflections off: the game's target cleared (no reflection anywhere) instead of traced.
    bool Clear(VideoCore::ImageId target_id, vk::Format view_format) {
        lighting_view = vk::ImageView{};
        if (!target_id) {
            return false;
        }
        auto& target = texture_cache.GetImage(target_id);
        if (target.info.pixel_format != Rgba16f ||
            (view_format != vk::Format::eUndefined && view_format != Rgba16f) ||
            !(target.usage_flags & vk::ImageUsageFlagBits::eTransferDst) ||
            target.dropped_levels != 0) {
            return false;
        }
        scheduler.EndRendering();
        // Only level 0: the game's own pass makes the others from it.
        runtime.Transit(&target, vk::ImageLayout::eTransferDstOptimal,
                        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
                        VideoCore::SubresourceRange{.base = {0, 0}, .extent = {1, 1}});
        runtime.FlushBarriers();
        scheduler.Record([image = target.GetImage()](vk::CommandBuffer cmdbuf) {
            const vk::ClearColorValue black{std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f}};
            const vk::ImageSubresourceRange range{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
            cmdbuf.clearColorImage(image, vk::ImageLayout::eTransferDstOptimal, black, range);
        });
        return true;
    }

    void ColorCopy(VideoCore::ImageId source_id) {
        auto& source = texture_cache.GetImage(source_id);
        const u32 width = source.info.size.width;
        const u32 height = source.info.size.height;
        if (source.info.pixel_format != Rgba16f || !Readable(source_id, width, height) ||
            width < 64 || height < 64) {
            return;
        }
        if (!pyramid_pipeline) {
            CreatePipelines();
        }
        if (!color.image || color.size != std::array{width, height}) {
            if (color.image) {
                scheduler.DeferOperation([old = std::move(color)] {});
            }
            color = MakeLeveled(Rgba16f, width, height, ColorLevels);
        }
        scheduler.EndRendering();
        runtime.Transit(&source, vk::ImageLayout::eTransferSrcOptimal,
                        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
        runtime.FlushBarriers();
        std::vector<vk::ImageView> levels;
        for (const auto& level : color.levels) {
            levels.push_back(*level);
        }
        scheduler.Record([src = source.GetImage(), dst = color.image->image, all = *color.view,
                          levels, width, height, pipeline = *down_pipeline,
                          layout = *pipeline_layout](vk::CommandBuffer cmdbuf) {
            // The last frame's copy is rewritten whole.
            Barrier(cmdbuf, ImageBarrier(dst, 0, ColorLevels, ComputeStage, Read,
                                         vk::PipelineStageFlagBits2::eTransfer | ComputeStage,
                                         vk::AccessFlagBits2::eTransferWrite | Write,
                                         vk::ImageLayout::eUndefined));
            const vk::ImageCopy region{
                .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                .extent = {width, height, 1},
            };
            cmdbuf.copyImage(src, vk::ImageLayout::eTransferSrcOptimal, dst,
                             vk::ImageLayout::eGeneral, region);
            Barrier(cmdbuf, ImageBarrier(dst, 0, 1, vk::PipelineStageFlagBits2::eTransfer,
                                         vk::AccessFlagBits2::eTransferWrite, ComputeStage, Read,
                                         vk::ImageLayout::eGeneral));
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
            for (u32 level = 1; level < ColorLevels; ++level) {
                const std::array<Bind, 2> binds = {{
                    {0, T::eSampledImage, all},
                    {1, T::eStorageImage, levels[level]},
                }};
                PushBinds(cmdbuf, layout, binds);
                const DownConstants push{
                    .size = {s32(std::max(width >> level, 1u)), s32(std::max(height >> level, 1u))},
                    .src_size = {s32(std::max(width >> (level - 1), 1u)),
                                 s32(std::max(height >> (level - 1), 1u))},
                    .src_level = s32(level - 1),
                };
                Dispatch(cmdbuf, layout, push, u32(push.size[0]), u32(push.size[1]));
                Barrier(cmdbuf, ImageBarrier(dst, level, 1, ComputeStage, Write, ComputeStage, Read,
                                             vk::ImageLayout::eGeneral));
            }
        });
        color_time = std::chrono::steady_clock::now();
        ++stats.color_copies;
    }
};

BbSsr::BbSsr(const Instance& instance, Scheduler& scheduler, Runtime& runtime,
             VideoCore::TextureCache& texture_cache)
    : impl{std::make_unique<Impl>(instance, scheduler, runtime, texture_cache)} {}

BbSsr::~BbSsr() = default;

bool BbSsr::Run(VideoCore::ImageId target, vk::Format view_format, VideoCore::ImageId depth,
                VideoCore::ImageId normals, VideoCore::ImageId color, const Constants& constants,
                const void* guest_constants) {
    auto& s = *impl;
    s.LogStats();
    ++s.stats.passes;
    if (s.failed) {
        return false;
    }
    try {
        const bool traced =
            s.Run(target, view_format, depth, normals, color, constants, guest_constants);
        s.stats.traced += traced ? 1 : 0;
        return traced;
    } catch (const std::exception& e) {
        s.failed = true;
        s.lighting_view = vk::ImageView{};
        LOG_ERROR(Render_Vulkan, "Better reflections: unavailable ({})", e.what());
        return false;
    }
}

void BbSsr::OnGBufferDraw(VideoCore::ImageId depth) {
    auto& s = *impl;
    if (!depth) {
        return;
    }
    const auto& image = s.texture_cache.GetImage(depth);
    const u64 area = u64(image.info.size.width) * image.info.size.height;
    if (area > s.scene_depth_area || depth == s.scene_depth) {
        s.scene_depth = depth;
        s.scene_depth_area = area;
    }
}

void BbSsr::OnColorCopy(VideoCore::ImageId source, VideoCore::ImageId target) {
    auto& s = *impl;
    if (s.failed || !source || !target || target != s.color_source) {
        return;
    }
    // The end of a frame: the next one finds its own scene depth, and makes its own reflections.
    s.lighting_view = vk::ImageView{};
    s.scene_depth_area = 0;
    if (!EmulatorSettings.IsBbSsrFullRes()) {
        return;
    }
    try {
        s.ColorCopy(source);
    } catch (const std::exception& e) {
        s.failed = true;
        LOG_ERROR(Render_Vulkan, "Better reflections: unavailable ({})", e.what());
    }
}

void BbSsr::OnLighting(VideoCore::ImageId gloss) {
    auto& s = *impl;
    s.stats.lighting_reads += s.lighting_view ? 1 : 0;
    if (!gloss) {
        return;
    }
    s.gloss = gloss;
    s.gloss_address = s.texture_cache.GetImage(gloss).info.guest_address;
}

bool BbSsr::Clear(VideoCore::ImageId target, vk::Format view_format) {
    return impl->Clear(target, view_format);
}

vk::ImageView BbSsr::LightingReflections() const {
    return impl->lighting_view;
}

} // namespace Vulkan
