// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-FileCopyrightText: Copyright 2026 bbport contributors (deadinside28/bloodborne_pc)
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <span>
#include <stdexcept>
#include "common/logging/log.h"
#include "core/emulator_settings.h"
#include "video_core/host_shaders/bb_vbao_blur_comp.h"
#include "video_core/host_shaders/bb_vbao_blur_rgba_comp.h"
#include "video_core/host_shaders/bb_vbao_comp.h"
#include "video_core/host_shaders/bb_vbao_temporal_comp.h"
#include "video_core/renderer_vulkan/vk_bb_vbao.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/texture_cache/host_compatibility.h"
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

struct PushConstants {
    std::array<std::array<float, 4>, 3> view_rows; // world to view rotation
    std::array<float, 4> proj;         // x scale, y scale, z scale, z offset (ndc +y down)
    std::array<float, 2> size;         // output pixels
    std::array<float, 2> normal_scale; // normal texels per output pixel
    float depth_scale;                 // depth pixels per output pixel
    float radius;                      // view units (metres)
    float thickness;                   // assumed thickness of the depth layer, view units
    float intensity;                   // exponent on the visibility
    float max_radius_px;
    u32 frame;       // sample pattern rotation; 0 when nothing accumulates frames
    u32 normal_mode; // 1: G-buffer normals through view_rows
};

struct BlurConstants {
    std::array<float, 2> size; // output pixels
    s32 radius;
    float tolerance;
    float normal_power;
    float upscale;                 // output pixels per raw texel (1 or 2)
    std::array<float, 2> raw_size; // raw texels
    float depth_scale;             // depth texels per output pixel
    float z_scale;                 // the projection's z scale and offset
    float z_offset;
};
static_assert(sizeof(BlurConstants) <= sizeof(PushConstants));

struct TemporalConstants {
    std::array<std::array<float, 4>, 3> reproject; // previous view * inverse current view (3x4)
    std::array<float, 4> proj;                     // current projection
    std::array<float, 4> prev_proj;                // previous projection
    std::array<float, 2> size;                     // raw texels
    float blend;                                   // history weight (0: none)
    float tolerance;                               // relative depth difference that drops history
    float texel_pos; // where in a texel its depth was read (0.5, or 0.75 at half size)
};
static_assert(sizeof(TemporalConstants) <= sizeof(PushConstants));

constexpr float TemporalBlend = 0.85f; // about 7 frames of history
constexpr float TemporalTolerance = 0.05f;

// 3x4 affine matrices (rows; the last column is the translation), as in bbport's camera motion.
std::array<float, 12> Multiply(const std::array<float, 12>& a, const std::array<float, 12>& b) {
    std::array<float, 12> out{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            float v = c == 3 ? a[r * 4 + 3] : 0.0f;
            for (int k = 0; k < 3; ++k) {
                v += a[r * 4 + k] * b[k * 4 + c];
            }
            out[r * 4 + c] = v;
        }
    }
    return out;
}

/// The inverse of [M | t]: [M^-1 | -M^-1 t].
std::array<float, 12> InverseAffine(const std::array<float, 12>& m) {
    const float a = m[0], b = m[1], c = m[2], d = m[4], e = m[5], f = m[6], g = m[8], h = m[9],
                i = m[10];
    const float det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (std::abs(det) < 1e-12f) {
        return m;
    }
    const float s = 1.0f / det;
    const std::array<float, 9> r = {(e * i - f * h) * s, (c * h - b * i) * s, (b * f - c * e) * s,
                                    (f * g - d * i) * s, (a * i - c * g) * s, (c * d - a * f) * s,
                                    (d * h - e * g) * s, (b * g - a * h) * s, (a * e - b * d) * s};
    std::array<float, 12> out{};
    for (int row = 0; row < 3; ++row) {
        out[row * 4 + 0] = r[row * 3 + 0];
        out[row * 4 + 1] = r[row * 3 + 1];
        out[row * 4 + 2] = r[row * 3 + 2];
        out[row * 4 + 3] =
            -(r[row * 3 + 0] * m[3] + r[row * 3 + 1] * m[7] + r[row * 3 + 2] * m[11]);
    }
    return out;
}

constexpr float Thickness = 0.4f;
constexpr s32 BlurRadius = 2;      // taps per side (5x5)
constexpr float BlurDepth = 0.05f; // relative depth difference that stops the blur
constexpr float BlurNormal = 8.0f; // how sharply the blur stops at normal changes

} // Anonymous namespace

struct BbVbao::Impl {
    const Instance& instance;
    Scheduler& scheduler;
    Runtime& runtime;
    VideoCore::TextureCache& texture_cache;
    vk::Device device;
    bool failed{};

    vk::UniqueDescriptorSetLayout desc_layout;
    vk::UniquePipelineLayout pipeline_layout;
    vk::UniquePipeline pipeline;           ///< the raw AO, view depth and normal
    vk::UniquePipeline blur_pipeline;      ///< denoise into R8 at the scene size
    vk::UniquePipeline blur_rgba_pipeline; ///< into the red channel of the half-size RGBA8
    vk::UniquePipeline temporal_pipeline;  ///< the raw AO blended with the last frames'
    /// RGBA16F at the raw size: AO, view z, normal xy.
    struct Scratch {
        std::unique_ptr<VideoCore::UniqueImage> image;
        vk::UniqueImageView view;
    };
    Scratch raw;        ///< this frame's AO
    Scratch history[2]; ///< accumulated AO: this frame's and the last one's, by turns
    u32 history_index{};
    bool history_cleared{}; ///< the history images are in General layout
    u32 raw_width{}, raw_height{};
    u32 frame{};
    // Temporal accumulation: the camera of the last frame whose AO went into the history, and
    // which AO pass that was (history only from the one just before).
    u64 ao_pass_index{};
    u64 history_pass{~0ull};
    std::array<float, 12> history_view{};
    std::array<float, 4> history_proj{};

    // The scene: the largest G-buffer pass of the frame, and its camera.
    VideoCore::ImageId depth_id{};
    u64 depth_area{};
    bool side_pass{};       ///< in a G-buffer pass smaller than the scene's (a reflection)
    bool pass_has_camera{}; ///< the scene pass's camera is known
    bool ao_done{};         ///< the frame's AO pass ran: the next G-buffer pass is a new frame
    float x_sign{1.0f}, y_sign{1.0f};
    std::array<float, 12> view{};
    std::array<float, 4> proj{};
    std::array<u32, 2> render_size{};
    VideoCore::ImageId ssao_normals{}; ///< this frame's, from the SSAO pass

    // Diagnostics, logged every 5 s: what happened at the game's AO passes.
    struct Stats {
        u64 ao_passes{}, ran{}, no_camera{}, no_depth{}, not_fitting{}, with_normals{};
        u64 ssao_passes{}, ssao_no_texture{}, game_draws_skipped{}, game_format_views{};
        u64 depth_changes{}, side_passes{}, cameras{}, proj_changes{};
        u64 accumulated{}, with_history{};
    } stats;
    VideoCore::ImageId last_depth{};
    std::array<float, 4> last_proj{};
    std::chrono::steady_clock::time_point next_log{};

    void LogStats() {
        const auto now = std::chrono::steady_clock::now();
        if (now < next_log) {
            return;
        }
        if (next_log != std::chrono::steady_clock::time_point{} && stats.ao_passes) {
            LOG_INFO(Render_Vulkan,
                     "Better AO (5 s): AO passes {}, VBAO {} (with G-buffer normals {}, through "
                     "a view in the game's format {}; SSAO passes {}, without a texture 1 {}; "
                     "game's AO draws left out {}; temporal {} of them with history {}), "
                     "game's AO "
                     "kept: no camera {}, no depth {}, targets not fitting {}; scene depth "
                     "changed {}, smaller G-buffer passes {}, cameras {}, projection changes {} "
                     "(now {:.4f} {:.4f} {:.6f} {:.6f})",
                     stats.ao_passes, stats.ran, stats.with_normals, stats.game_format_views,
                     stats.ssao_passes, stats.ssao_no_texture, stats.game_draws_skipped,
                     stats.accumulated, stats.with_history, stats.no_camera, stats.no_depth,
                     stats.not_fitting, stats.depth_changes, stats.side_passes, stats.cameras,
                     stats.proj_changes, proj[0], proj[1], proj[2], proj[3]);
        }
        stats = {};
        next_log = now + std::chrono::seconds(5);
    }

    Impl(const Instance& instance_, Scheduler& scheduler_, Runtime& runtime_,
         VideoCore::TextureCache& texture_cache_)
        : instance{instance_}, scheduler{scheduler_}, runtime{runtime_},
          texture_cache{texture_cache_}, device{instance_.GetDevice()} {}

    void CreatePipelines() {
        const std::array<vk::DescriptorSetLayoutBinding, 3> bindings = {{
            {.binding = 0,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 1,
             .descriptorType = vk::DescriptorType::eStorageImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 2,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
        }};
        desc_layout = CheckOrThrow(device.createDescriptorSetLayoutUnique({
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
            .bindingCount = static_cast<u32>(bindings.size()),
            .pBindings = bindings.data(),
        }));
        const vk::PushConstantRange range{.stageFlags = vk::ShaderStageFlagBits::eCompute,
                                          .offset = 0,
                                          .size = sizeof(PushConstants)};
        const auto set = *desc_layout;
        pipeline_layout = CheckOrThrow(device.createPipelineLayoutUnique({
            .setLayoutCount = 1,
            .pSetLayouts = &set,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &range,
        }));
        const auto create = [&](std::span<const u32> code) {
            const auto module = CompileSPV(code, device);
            auto result = device.createComputePipelineUnique(
                {}, vk::ComputePipelineCreateInfo{
                        .stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                                  .module = module,
                                  .pName = "main"},
                        .layout = *pipeline_layout,
                    });
            device.destroyShaderModule(module);
            return CheckOrThrow(std::move(result));
        };
        pipeline = create(BB_VBAO_COMP);
        blur_pipeline = create(BB_VBAO_BLUR_COMP);
        blur_rgba_pipeline = create(BB_VBAO_BLUR_RGBA_COMP);
        temporal_pipeline = create(BB_VBAO_TEMPORAL_COMP);
    }

    Scratch MakeScratch(u32 width, u32 height) {
        Scratch out;
        out.image = std::make_unique<VideoCore::UniqueImage>(device, instance.GetAllocator());
        out.image->Create(vk::ImageCreateInfo{
            .imageType = vk::ImageType::e2D,
            .format = vk::Format::eR16G16B16A16Sfloat,
            .extent = {width, height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
            .initialLayout = vk::ImageLayout::eUndefined,
        });
        out.view = CheckOrThrow(device.createImageViewUnique({
            .image = out.image->image,
            .viewType = vk::ImageViewType::e2D,
            .format = vk::Format::eR16G16B16A16Sfloat,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        }));
        return out;
    }

    void EnsureScratch(u32 width, u32 height) {
        if (raw.image && raw_width == width && raw_height == height) {
            return;
        }
        if (raw.image) {
            // The old ones may still be in use by the GPU.
            scheduler.DeferOperation([old = std::move(raw), old0 = std::move(history[0]),
                                      old1 = std::move(history[1])] {});
        }
        raw = MakeScratch(width, height);
        history[0] = MakeScratch(width, height);
        history[1] = MakeScratch(width, height);
        history_cleared = false;
        history_pass = ~0ull;
        raw_width = width;
        raw_height = height;
    }

    /// A view the texture cache keeps with the image (made once, dropped with the image).
    vk::ImageView CacheView(VideoCore::Image& image, vk::Format format, bool storage) {
        VideoCore::ImageViewInfo info{};
        info.type = AmdGpu::ImageType::Color2D;
        info.format = format;
        info.is_storage = storage;
        return *image.FindView(info, false).image_view;
    }

    /// The format VBAO writes `target` in: the one the game draws it in (`view_format`), which
    /// may differ from the image's own (memory first used in another format of the same size
    /// per pixel, e.g. kept as R32 float). Undefined when VBAO can't write it.
    vk::Format AoFormat(const VideoCore::Image& ao, vk::Format view_format) const {
        const auto image_format = ao.info.pixel_format;
        if (view_format == vk::Format::eUndefined) {
            view_format = image_format;
        }
        if (view_format != vk::Format::eR8Unorm && view_format != vk::Format::eR8G8B8A8Unorm) {
            return vk::Format::eUndefined;
        }
        if (view_format != image_format &&
            !VideoCore::IsVulkanFormatCompatible(image_format, view_format)) {
            return vk::Format::eUndefined;
        }
        return view_format;
    }

    /// Whether VBAO can replace the game's AO in `target` this frame (checked before the
    /// game's draw, which is then left out).
    bool CanReplace(VideoCore::ImageId target, vk::Format view_format) {
        if (failed || !target || !depth_id || !pass_has_camera) {
            return false;
        }
        auto& depth = texture_cache.GetImage(depth_id);
        auto& ao = texture_cache.GetImage(target);
        const auto depth_format = depth.info.pixel_format;
        const auto ao_format = AoFormat(ao, view_format);
        // The half-size path: an RGBA8 target of half the depth's size (rounded up).
        const u32 scale = ao_format == vk::Format::eR8G8B8A8Unorm ? 2 : 1;
        return (depth_format == vk::Format::eD32Sfloat ||
                depth_format == vk::Format::eD32SfloatS8Uint) &&
               ao_format != vk::Format::eUndefined && depth.info.num_samples == 1 &&
               ao.info.num_samples == 1 &&
               bool(ao.usage_flags & vk::ImageUsageFlagBits::eStorage) &&
               ao.info.size.width == (depth.info.size.width + scale - 1) / scale &&
               ao.info.size.height == (depth.info.size.height + scale - 1) / scale;
    }

    /// Whether VBAO went into `target`.
    bool Run(VideoCore::ImageId target, vk::Format view_format, VideoCore::ImageId normals_id) {
        auto& depth = texture_cache.GetImage(depth_id);
        auto& ao = texture_cache.GetImage(target);
        const auto depth_format = depth.info.pixel_format;
        if (!CanReplace(target, view_format)) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                LOG_WARNING(Render_Vulkan,
                            "Better AO: skipped (depth {} {}x{}, AO target {} {}x{} drawn as {})",
                            vk::to_string(depth_format), depth.info.size.width,
                            depth.info.size.height, vk::to_string(ao.info.pixel_format),
                            ao.info.size.width, ao.info.size.height, vk::to_string(view_format));
            }
            ++stats.not_fitting;
            return false;
        }
        const auto ao_format = AoFormat(ao, view_format);
        if (ao_format != ao.info.pixel_format) {
            ++stats.game_format_views;
        }
        const bool half = ao_format == vk::Format::eR8G8B8A8Unorm;
        const u32 scale = half ? 2 : 1;
        // Half resolution (the full-size path): the AO computed at half the scene's size and
        // brought up to it by the blur, guided by the full-size depth (about a quarter of the
        // cost). The half-size path's target is already half size.
        const u32 down = !half && EmulatorSettings.IsBbVbaoHalfRes() ? 2 : 1;
        const u32 raw_width_now = (ao.info.size.width + down - 1) / down;
        const u32 raw_height_now = (ao.info.size.height + down - 1) / down;
        if (!pipeline) {
            CreatePipelines();
        }
        // The scene's viewport; the targets may be allocated larger (aligned height).
        auto [width, height] = render_size;
        width =
            width ? std::min((width + scale - 1) / scale, ao.info.size.width) : ao.info.size.width;
        height = height ? std::min((height + scale - 1) / scale, ao.info.size.height)
                        : ao.info.size.height;
        {
            static bool logged[2] = {};
            if (!logged[half]) {
                logged[half] = true;
                LOG_INFO(Render_Vulkan,
                         "Better AO: on, {}x{} into the game's {} AO ({} {}x{}, kept as {})", width,
                         height, half ? "half-size" : "upsampled", vk::to_string(ao_format),
                         ao.info.size.width, ao.info.size.height,
                         vk::to_string(ao.info.pixel_format));
            }
        }

        // The G-buffer normals: a colour image covering the depth (normal-map detail).
        VideoCore::Image* normals = nullptr;
        if (normals_id) {
            auto& image = texture_cache.GetImage(normals_id);
            const bool fits = !image.info.props.is_depth && image.backing &&
                              image.info.num_samples == 1 && normals_id != target &&
                              image.info.size.width >= depth.info.size.width / 2 &&
                              image.info.size.height >= depth.info.size.height / 2;
            static int logged = 0;
            if (logged < 2 && (logged == 0 || fits)) {
                ++logged;
                LOG_INFO(Render_Vulkan, "Better AO: G-buffer normals {} {}x{}{}",
                         vk::to_string(image.info.pixel_format), image.info.size.width,
                         image.info.size.height, fits ? "" : ": not used");
            }
            if (fits) {
                normals = &image;
                ++stats.with_normals;
            }
        }

        const float strength =
            std::clamp(float(EmulatorSettings.GetBbVbaoStrength()) * 0.01f, 0.1f, 5.0f);
        const float radius =
            std::clamp(float(EmulatorSettings.GetBbVbaoRadius()) * 0.01f, 0.1f, 10.0f);
        const u32 compute_width = (width + down - 1) / down;
        const u32 compute_height = (height + down - 1) / down;
        const PushConstants push{
            .view_rows = {{{view[0], view[1], view[2], 0.0f},
                           {view[4], view[5], view[6], 0.0f},
                           {view[8], view[9], view[10], 0.0f}}},
            .proj = proj,
            .size = {float(compute_width), float(compute_height)},
            .normal_scale = {normals ? float(normals->info.size.width) / float(raw_width_now)
                                     : 1.0f,
                             normals ? float(normals->info.size.height) / float(raw_height_now)
                                     : 1.0f},
            .depth_scale = float(scale * down),
            .radius = radius,
            .thickness = Thickness,
            .intensity = strength,
            .max_radius_px = 0.1f * float(compute_height),
            // The pattern turns each frame; the temporal accumulation averages the frames.
            .frame = ++frame,
            .normal_mode = normals ? 1u : 0u,
        };
        const BlurConstants blur_push{
            .size = {float(width), float(height)},
            // Upscaling, 3x3 half-size texels cover about what the full-size 5x5 blur does; 5x5
            // spread the darkness of narrow grooves (between floor tiles) over the flat ground
            // around them, and half resolution looked darker than full.
            .radius = down > 1 ? 1 : BlurRadius,
            .tolerance = BlurDepth,
            .normal_power = BlurNormal,
            .upscale = float(down),
            .raw_size = push.size,
            .depth_scale = float(scale),
            .z_scale = proj[2],
            .z_offset = proj[3],
        };

        EnsureScratch(raw_width_now, raw_height_now);
        // R32 float: a depth-only view of the depth image (ImageView picks the depth aspect).
        const auto depth_view = CacheView(depth, vk::Format::eR32Sfloat, false);
        const auto ao_view = CacheView(ao, ao_format, true);
        // Without the G-buffer normals the depth stands in for binding 2 (not read).
        const auto normal_view =
            normals ? CacheView(*normals, normals->info.pixel_format, false) : depth_view;

        // Temporal accumulation: this frame's AO blended with the last frames', moved to where
        // the camera now sees them; dropped where the depth doesn't match (newly seen places).
        const u64 pass = ao_pass_index;
        const bool history_valid = history_pass + 1 == pass;
        const u32 write_index = history_index ^ 1u;
        TemporalConstants temporal_push{
            .proj = proj,
            .prev_proj = history_proj,
            .size = push.size,
            .blend = history_valid ? TemporalBlend : 0.0f,
            .tolerance = TemporalTolerance,
            .texel_pos = scale * down == 2 ? 0.75f : 0.5f,
        };
        if (history_valid) {
            const auto reproject = Multiply(history_view, InverseAffine(view));
            for (int row = 0; row < 3; ++row) {
                for (int c = 0; c < 4; ++c) {
                    temporal_push.reproject[row][c] = reproject[row * 4 + c];
                }
            }
        }
        history_index = write_index;
        history_pass = pass;
        history_view = view;
        history_proj = proj;
        ++stats.accumulated;
        stats.with_history += history_valid ? 1 : 0;
        const bool clear_history = !history_cleared;
        history_cleared = true;

        scheduler.EndRendering();
        if (normals) {
            runtime.Transit(normals, vk::ImageLayout::eGeneral,
                            vk::PipelineStageFlagBits2::eComputeShader,
                            vk::AccessFlagBits2::eShaderRead);
        }
        runtime.Transit(&depth, vk::ImageLayout::eGeneral,
                        vk::PipelineStageFlagBits2::eComputeShader,
                        vk::AccessFlagBits2::eShaderRead);
        runtime.Transit(&ao, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eComputeShader,
                        half ? vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite
                             : vk::AccessFlagBits2::eShaderWrite);
        runtime.FlushBarriers();

        scheduler.Record([push, blur_push, temporal_push, depth_view, ao_view, normal_view,
                          raw = raw.image->image, raw_view = *raw.view,
                          hist_write = history[write_index].image->image,
                          hist_write_view = *history[write_index].view,
                          hist_read = history[write_index ^ 1u].image->image,
                          hist_read_view = *history[write_index ^ 1u].view, clear_history,
                          pipeline = *pipeline, temporal_pipeline = *temporal_pipeline,
                          blur_pipeline = half ? *blur_rgba_pipeline : *blur_pipeline,
                          layout = *pipeline_layout](vk::CommandBuffer cmdbuf) {
            const auto push_images = [&](vk::ImageView sampled, vk::ImageView storage,
                                         vk::ImageView extra) {
                const vk::DescriptorImageInfo sampled_info{
                    .imageView = sampled, .imageLayout = vk::ImageLayout::eGeneral};
                const vk::DescriptorImageInfo storage_info{
                    .imageView = storage, .imageLayout = vk::ImageLayout::eGeneral};
                const vk::DescriptorImageInfo extra_info{.imageView = extra,
                                                         .imageLayout = vk::ImageLayout::eGeneral};
                const std::array<vk::WriteDescriptorSet, 3> writes = {{
                    {.dstBinding = 0,
                     .descriptorCount = 1,
                     .descriptorType = vk::DescriptorType::eSampledImage,
                     .pImageInfo = &sampled_info},
                    {.dstBinding = 1,
                     .descriptorCount = 1,
                     .descriptorType = vk::DescriptorType::eStorageImage,
                     .pImageInfo = &storage_info},
                    {.dstBinding = 2,
                     .descriptorCount = 1,
                     .descriptorType = vk::DescriptorType::eSampledImage,
                     .pImageInfo = &extra_info},
                }};
                cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0,
                                            std::span(writes.data(), extra ? 3 : 2));
            };
            const u32 groups_x = (u32(push.size[0]) + 7) / 8;
            const u32 groups_y = (u32(push.size[1]) + 7) / 8;
            const u32 out_groups_x = (u32(blur_push.size[0]) + 7) / 8;
            const u32 out_groups_y = (u32(blur_push.size[1]) + 7) / 8;
            // The raw image is rewritten whole each frame: its old contents are not kept.
            vk::ImageMemoryBarrier2 barrier{
                .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
                .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
                .oldLayout = vk::ImageLayout::eUndefined,
                .newLayout = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = raw,
                .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
            };
            cmdbuf.pipelineBarrier2(
                vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
            push_images(depth_view, raw_view, normal_view);
            cmdbuf.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
            cmdbuf.dispatch(groups_x, groups_y, 1);

            barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
            barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
            barrier.oldLayout = vk::ImageLayout::eGeneral;
            cmdbuf.pipelineBarrier2(
                vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
            {
                // The history images live in General layout; new ones start from Undefined.
                std::array<vk::ImageMemoryBarrier2, 2> history_barriers{};
                for (int i = 0; i < 2; ++i) {
                    history_barriers[i] = vk::ImageMemoryBarrier2{
                        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                        .srcAccessMask =
                            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                        .dstAccessMask = i == 0 ? vk::AccessFlagBits2::eShaderWrite
                                                : vk::AccessFlagBits2::eShaderRead,
                        .oldLayout =
                            clear_history ? vk::ImageLayout::eUndefined : vk::ImageLayout::eGeneral,
                        .newLayout = vk::ImageLayout::eGeneral,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = i == 0 ? hist_write : hist_read,
                        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
                    };
                }
                cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                    .imageMemoryBarrierCount = 2, .pImageMemoryBarriers = history_barriers.data()});
                cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, temporal_pipeline);
                push_images(raw_view, hist_write_view, hist_read_view);
                cmdbuf.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0,
                                     sizeof(temporal_push), &temporal_push);
                cmdbuf.dispatch(groups_x, groups_y, 1);
                auto written = history_barriers[0];
                written.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
                written.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
                written.oldLayout = vk::ImageLayout::eGeneral;
                cmdbuf.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                                           .pImageMemoryBarriers = &written});
            }
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, blur_pipeline);
            // The accumulated AO; the depth guides the upscale.
            push_images(hist_write_view, ao_view, depth_view);
            cmdbuf.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(blur_push),
                                 &blur_push);
            cmdbuf.dispatch(out_groups_x, out_groups_y, 1);
        });
        return true;
    }
};

BbVbao::BbVbao(const Instance& instance, Scheduler& scheduler, Runtime& runtime,
               VideoCore::TextureCache& texture_cache)
    : impl{std::make_unique<Impl>(instance, scheduler, runtime, texture_cache)} {}

BbVbao::~BbVbao() = default;

void BbVbao::OnGBufferDraw(VideoCore::ImageId depth, float x_sign, float y_sign) {
    auto& s = *impl;
    gbuffer_draw = false;
    if (!depth) {
        return;
    }
    if (s.ao_done) {
        // A new frame.
        s.ao_done = false;
        s.depth_id = {};
        s.depth_area = 0;
        s.pass_has_camera = false;
        s.side_pass = false;
        s.ssao_normals = {};
    }
    if (depth == s.depth_id) {
        s.side_pass = false; // the scene's pass again (after a smaller one)
    } else {
        // Another depth target: another G-buffer pass, with its own camera. One smaller than the
        // frame's pass so far is a side view (a water reflection): the scene's depth and camera
        // stay.
        const auto& image = s.texture_cache.GetImage(depth);
        const u64 area = u64(image.info.size.width) * image.info.size.height;
        if (s.depth_id && area < s.depth_area) {
            if (!s.side_pass) {
                ++s.stats.side_passes;
            }
            s.side_pass = true;
            return;
        }
        if (depth != s.last_depth) {
            ++s.stats.depth_changes;
            s.last_depth = depth;
        }
        s.depth_id = depth;
        s.depth_area = area;
        s.side_pass = false;
        s.pass_has_camera = false;
    }
    s.x_sign = x_sign;
    s.y_sign = y_sign;
    gbuffer_draw = true;
}

void BbVbao::ObserveSceneConstants(const void* data, u64 size) {
    auto& s = *impl;
    if (!gbuffer_draw || s.pass_has_camera || s.side_pass || !data || size < 64 * sizeof(float)) {
        return;
    }
    const auto* f = static_cast<const float*>(data);
    // Scene constants: far plane 3000, 1/far, and the render size.
    if (f[0] != 3000.0f || std::abs(f[1] * f[0] - 1.0f) > 1e-3f || f[4] < 64.0f || f[5] < 64.0f) {
        return;
    }
    if (f[52] == 0.0f || f[57] == 0.0f || !std::isfinite(f[62]) || !std::isfinite(f[63])) {
        return;
    }
    std::memcpy(s.view.data(), f + 8, sizeof(s.view));
    // The projection as the shader uses it: ndc +y down the screen; the G-buffer pass's viewport
    // decides which way the game's goes.
    s.proj = {f[52] * s.x_sign, f[57] * s.y_sign, f[62], f[63]};
    s.render_size = {u32(f[4]), u32(f[5])};
    s.pass_has_camera = true;
    ++s.stats.cameras;
    if (s.proj != s.last_proj) {
        ++s.stats.proj_changes;
        s.last_proj = s.proj;
    }
}

void BbVbao::OnSsaoPass(VideoCore::ImageId normals) {
    auto& s = *impl;
    s.ssao_normals = normals;
    ++s.stats.ssao_passes;
    if (!normals) {
        ++s.stats.ssao_no_texture;
    }
}

bool BbVbao::ReplacesSsaoPass() {
    auto& s = *impl;
    // The SSAO pass only feeds the upsample, which VBAO overwrites: left out once the scene's
    // depth and camera are known (the upsample's own target is checked at its draw).
    const bool skip = !s.failed && s.depth_id && s.pass_has_camera;
    s.stats.game_draws_skipped += skip ? 1 : 0;
    return skip;
}

bool BbVbao::ReplacesAoDraw(VideoCore::ImageId target, vk::Format view_format) {
    auto& s = *impl;
    const bool skip = s.CanReplace(target, view_format);
    s.stats.game_draws_skipped += skip ? 1 : 0;
    return skip;
}

void BbVbao::OnAoPass(VideoCore::ImageId target, vk::Format view_format,
                      VideoCore::ImageId normals) {
    auto& s = *impl;
    s.LogStats();
    ++s.stats.ao_passes;
    ++s.ao_pass_index;
    const bool ready = target && s.depth_id && s.pass_has_camera;
    s.ao_done = true;
    if (!ready || s.failed) {
        ++(s.depth_id ? s.stats.no_camera : s.stats.no_depth);
        return;
    }
    try {
        s.stats.ran += s.Run(target, view_format, normals ? normals : s.ssao_normals) ? 1 : 0;
    } catch (const std::exception& e) {
        s.failed = true;
        LOG_ERROR(Render_Vulkan, "Better AO: unavailable ({})", e.what());
    }
}

} // namespace Vulkan
