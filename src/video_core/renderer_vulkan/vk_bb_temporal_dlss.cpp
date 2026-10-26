// SPDX-FileCopyrightText: Copyright 2026 IFreemz
// SPDX-License-Identifier: GPL-2.0-or-later
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <fmt/format.h>
#include "common/bb_tuning.h"
#include "common/crash_crumbs.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "core/bb_game_fixes.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/host_shaders/bb_dlss_composite_comp.h"
#include "video_core/host_shaders/bb_dlss_composite_hdr_comp.h"
#include "video_core/host_shaders/bb_dlss_linearize_comp.h"
#include "video_core/host_shaders/bb_dlss_motion_comp.h"
#include "video_core/host_shaders/bb_dlss_motion_debug_comp.h"
#include "video_core/host_shaders/bb_dlss_sharpen_hdr_comp.h"
#include "video_core/host_shaders/bb_hud_layer_comp.h"
#include "video_core/host_shaders/bb_hudless_dim_comp.h"
#include "video_core/renderer_vulkan/vk_bb_object_motion.h"
#include "video_core/renderer_vulkan/vk_bb_resolution.h"
#include "video_core/renderer_vulkan/vk_bb_temporal_dlss.h"
#include "video_core/renderer_vulkan/vk_bb_velocity_mirror.h"
#include "video_core/renderer_vulkan/vk_dlss_ngx.h"
#include "video_core/renderer_vulkan/vk_fsr4_addon.h"
#include "video_core/renderer_vulkan/vk_fsr_upscaler.h"
#include "video_core/renderer_vulkan/vk_gpu_profiler.h"
#include "video_core/renderer_vulkan/vk_hdr_mod.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/texture_cache/texture_cache.h"

namespace Vulkan {

bool BbTemporalDlssRequested() {
    return DlssNgx::Present() || FsrUpscaler::Present() || Fsr4Addon::Present();
}

std::filesystem::path BbDlssSettingsPath() {
    return Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "dlss.ini";
}

namespace {
std::mutex status_mutex;
BbDlssStatus status;
std::string ngx_problem;
std::chrono::steady_clock::time_point last_evaluation;

void ReportFrame(std::string_view reason) {
    std::scoped_lock lock{status_mutex};
    status.reason = reason;
}

std::string DlssProblem() {
    std::scoped_lock lock{status_mutex};
    return ngx_problem;
}
} // namespace

namespace {
std::atomic<float> scene_mip_bias{0.0f};
std::atomic<s64> scene_mip_bias_ms{0}; // steady clock of the evaluation that set it
s64 SteadyMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
} // namespace

float BbSceneMipBias() {
    // Menus and loading screens (no evaluation for a second): the game's own bias.
    if (SteadyMs() - scene_mip_bias_ms.load(std::memory_order_relaxed) > 1000)
        return 0.0f;
    return scene_mip_bias.load(std::memory_order_relaxed);
}

BbDlssStatus BbTemporalDlssStatus() {
    std::scoped_lock lock{status_mutex};
    auto copy = status;
    copy.active = std::chrono::steady_clock::now() - last_evaluation < std::chrono::seconds{1};
    copy.dlss_problem = ngx_problem;
    copy.fsr4_installed = Fsr4Addon::Present();
    if (!BbTemporalDlssRequested())
        copy.reason = "No upscaler files were found next to shadPS4.exe.";
    else if (!copy.active && copy.reason.empty())
        copy.reason = "Waiting for gameplay (menus and loading screens use the normal image).";
    return copy;
}

void BbTemporalDlssReportGpu(std::string_view gpu) {
    std::scoped_lock lock{status_mutex};
    status.gpu = gpu;
}

void BbTemporalDlssReportProblem(std::string_view problem) {
    std::scoped_lock lock{status_mutex};
    ngx_problem = problem;
}

namespace {
// Inverse of an affine 3x4 row-major transform, in double precision.
std::array<double, 12> InverseAffine(const std::array<double, 12>& m) {
    const double a = m[0], b = m[1], c = m[2], d = m[4], e = m[5], f = m[6], g = m[8], h = m[9],
                 i = m[10];
    const double c0 = e * i - f * h, c1 = f * g - d * i, c2 = d * h - e * g;
    const double det = a * c0 + b * c1 + c * c2;
    if (det == 0.0 || !std::isfinite(det))
        return m;
    const double r = 1.0 / det;
    std::array<double, 12> inv{};
    inv[0] = c0 * r;
    inv[1] = (c * h - b * i) * r;
    inv[2] = (b * f - c * e) * r;
    inv[4] = c1 * r;
    inv[5] = (a * i - c * g) * r;
    inv[6] = (c * d - a * f) * r;
    inv[8] = c2 * r;
    inv[9] = (b * g - a * h) * r;
    inv[10] = (a * e - b * d) * r;
    for (int row = 0; row < 3; ++row)
        inv[row * 4 + 3] =
            -(inv[row * 4 + 0] * m[3] + inv[row * 4 + 1] * m[7] + inv[row * 4 + 2] * m[11]);
    return inv;
}

constexpr u64 DepthProducer = 0xd3c8bb21;  // samples R32 scene depth at slot 0
constexpr u64 DisplayCopy = 0x38d65b32;    // UI target -> VideoOut, LUT at slot 1
constexpr u64 OpaqueVelocity = 0x34bc187c; // packing scale at binding 0
constexpr u64 AlphaVelocityA = 0x749e4f9e; // packing scale at binding 1
constexpr u64 AlphaVelocityB = 0xb25e4fae;
// Motion blur: two full-screen passes, each blurring the HDR scene (slot 0) along the half-size
// motion buffer (slot 1) that the DepthProducer pass builds. Skipping them keeps that data.
constexpr u64 MotionBlur = 0xe0305cef;
// The same blur in some areas (Forbidden Woods): another pixel shader, same passes and slots.
constexpr u64 MotionBlurAlt = 0x29e06868;
// Copies the HDR scene (slot 0) into the target post-processing runs on. With the Decoupled UI
// patches the scene is smaller than that target and this pass is the game's upscale.
constexpr u64 HdrSceneCopy = 0xccbf44a6;
constexpr u64 CameraBytes = 4096, VelocityOffset = 4096, CoefficientBytes = 4096 + 256;
// HUD coverage counters, one per frame in flight, each in its own aligned slot.
constexpr u64 CoverageSlots = 3, CoverageStride = 256;
constexpr u64 DimmingBytes = 2 * 64 * sizeof(u32); // HUD dimming histogram, GPU only
constexpr vk::ImageSubresourceRange Range{
    .aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1};

bool IsUiShader(u64 hash) {
    return hash == 0x34e8a281 || hash == 0x81d336ce || hash == 0x09957251 || hash == 0x24042a9b ||
           hash == 0xa400228b;
}

template <typename T>
T Make(vk::ResultValue<T> result, const char* what) {
    if (result.result != vk::Result::eSuccess)
        throw std::runtime_error(std::string{what} + ": " + vk::to_string(result.result));
    return std::move(result.value);
}

float Halton(u32 index, u32 base) {
    float f = 1.f, r = 0.f;
    while (index > 0) {
        f /= static_cast<float>(base);
        r += f * static_cast<float>(index % base);
        index /= base;
    }
    return r;
}

struct OwnedImage {
    VideoCore::UniqueImage image;
    vk::UniqueImageView view;
    vk::UniqueImageView rgb_view, bgr_view; // opaque presentation views
    vk::Format format;
    vk::Extent2D extent;
    vk::ImageLayout layout{vk::ImageLayout::eUndefined};
    bool fresh{};
    bool swapped{}; // red and blue swapped, as the game's display copy writes them
    std::optional<FrameGen::FrameInputs> frame_gen; // with a presentable output
    // FSR frame generation: the same frame without the game's HUD, fresh along with it.
    std::unique_ptr<OwnedImage> hudless;
    bool hudless_fresh{};

    // Decoupled UI HUD-less frames: a view in the presenter's format for the game's frame.
    vk::Device device;
    vk::UniqueImageView frame_view;
    vk::Format frame_view_format{};

    OwnedImage(const Instance& instance, vk::Format format_, vk::Extent2D extent_,
               vk::ImageUsageFlags usage, bool presentable = false, vk::ImageCreateFlags flags = {})
        : image{instance.GetDevice(), instance.GetAllocator()}, format{format_}, extent{extent_},
          device{instance.GetDevice()} {
        image.Create(vk::ImageCreateInfo{.flags = flags,
                                         .imageType = vk::ImageType::e2D,
                                         .format = format,
                                         .extent = {extent.width, extent.height, 1},
                                         .mipLevels = 1,
                                         .arrayLayers = 1,
                                         .samples = vk::SampleCountFlagBits::e1,
                                         .tiling = vk::ImageTiling::eOptimal,
                                         .usage = usage});
        const auto device = instance.GetDevice();
        const auto make_view = [&](vk::ComponentMapping mapping) {
            return Make(device.createImageViewUnique(
                            vk::ImageViewCreateInfo{.image = image.image,
                                                    .viewType = vk::ImageViewType::e2D,
                                                    .format = format,
                                                    .components = mapping,
                                                    .subresourceRange = Range}),
                        "temporal DLSS image view");
        };
        view = make_view({});
        if (presentable) {
            using S = vk::ComponentSwizzle;
            rgb_view = make_view({S::eR, S::eG, S::eB, S::eOne});
            bgr_view = make_view({S::eB, S::eG, S::eR, S::eOne});
        }
    }
    // Decoupled UI HUD-less frames: written as the game writes its frame. The VideoOut buffers
    // are not all the same format (UNORM and sRGB), and the game writes them all as UNORM: an
    // sRGB one written as sRGB is encoded twice (washed out, every other frame).
    vk::UniqueImageView target_view;
    vk::Format target_view_format{};
    vk::ImageView TargetView(vk::Format view_format) {
        if (view_format == format || view_format == vk::Format::eUndefined)
            return *view;
        if (!target_view || target_view_format != view_format) {
            target_view = Make(device.createImageViewUnique(
                                   vk::ImageViewCreateInfo{.image = image.image,
                                                           .viewType = vk::ImageViewType::e2D,
                                                           .format = view_format,
                                                           .subresourceRange = Range}),
                               "temporal DLSS target view");
            target_view_format = view_format;
        }
        return *target_view;
    }
    // Opaque view as `view_format`, like the presenter's view of the game's own frame (the
    // image needs eMutableFormat).
    vk::ImageView FrameView(vk::Format view_format) {
        if (!frame_view || frame_view_format != view_format) {
            using S = vk::ComponentSwizzle;
            frame_view = Make(device.createImageViewUnique(vk::ImageViewCreateInfo{
                                  .image = image.image,
                                  .viewType = vk::ImageViewType::e2D,
                                  .format = view_format,
                                  .components = {S::eIdentity, S::eIdentity, S::eIdentity, S::eOne},
                                  .subresourceRange = Range}),
                              "temporal DLSS frame view");
            frame_view_format = view_format;
        }
        return *frame_view;
    }
    DlssNgx::Resource Resource() const {
        return {image.image, *view, Range, format, extent};
    }
    void Transit(vk::CommandBuffer command, vk::ImageLayout next, vk::PipelineStageFlags2 stage,
                 vk::AccessFlags2 access) {
        const vk::ImageMemoryBarrier2 barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask =
                layout == vk::ImageLayout::eUndefined
                    ? vk::AccessFlagBits2::eNone
                    : vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = stage,
            .dstAccessMask = access,
            .oldLayout = layout,
            .newLayout = next,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image.image,
            .subresourceRange = Range};
        command.pipelineBarrier2(
            vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
        layout = next;
    }
};

struct ComputePass {
    vk::UniqueDescriptorSetLayout descriptors;
    vk::UniquePipelineLayout layout;
    vk::UniquePipeline pipeline;

    ComputePass(vk::Device device, std::span<const vk::DescriptorType> types,
                std::span<const u32> code, u32 push_bytes) {
        std::vector<vk::DescriptorSetLayoutBinding> bindings;
        for (u32 i = 0; i < types.size(); ++i)
            bindings.push_back({.binding = i,
                                .descriptorType = types[i],
                                .descriptorCount = 1,
                                .stageFlags = vk::ShaderStageFlagBits::eCompute});
        descriptors = Make(device.createDescriptorSetLayoutUnique(
                               {.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptor,
                                .bindingCount = static_cast<u32>(bindings.size()),
                                .pBindings = bindings.data()}),
                           "temporal DLSS descriptor layout");
        const vk::PushConstantRange push{
            .stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = push_bytes};
        layout = Make(device.createPipelineLayoutUnique({.setLayoutCount = 1,
                                                         .pSetLayouts = &descriptors.get(),
                                                         .pushConstantRangeCount = 1,
                                                         .pPushConstantRanges = &push}),
                      "temporal DLSS pipeline layout");
        const auto module = CompileSPV(code, device);
        if (!module)
            throw std::runtime_error("temporal DLSS shader unavailable");
        auto result = device.createComputePipelineUnique(
            {}, vk::ComputePipelineCreateInfo{.stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                                                        .module = module,
                                                        .pName = "main"},
                                              .layout = *layout});
        device.destroyShaderModule(module);
        pipeline = Make(std::move(result), "temporal DLSS pipeline");
    }
};

constexpr char DefaultSettings[] =
    R"(# Upscaler settings for shadPS4. Press F1 in game to change them, or edit this file;
# saved changes apply within a second.

# 1 = upscaling on, 0 = the game's normal image
enabled=1

# auto = DLSS on NVIDIA RTX cards, else FSR 4 when its add-on is installed, else FSR 3.1;
# or dlss, fsr4, fsr
upscaler=auto

# DLSS model: 13 = M, 11 = K, 10 = J, 12 = L, 0 = NVIDIA default
preset=13

# Sharpening after upscaling, 0 (off) to 1
sharpness=0.6

# The game's motion blur: 0 = off (sharper in motion), 1 = on
motion_blur=0

# Exact motion for characters, cloth and enemies (less smearing): 1 = on, 0 = off.
# Turning it on or off completely needs a restart.
character_motion=1

# The HUD, menus and dialogue drawn as their own layer over the upscaled scene, so the scene
# does not shimmer through their see-through parts: 1 = on, 0 = off
menu_fix=1

# -1 = automatic, 0 = DLAA, 1 = Quality, 2 = Balanced, 3 = Performance, 4 = Ultra Performance
quality=-1

# Output resolution, e.g. output=3840x2160. Empty = the shadPS4 window size.
output=
)";

struct Tune {
    bool enabled = true;
    bool jitter = true;
    // Test: also jitter single quads drawn with the scene depth that test and write it
    // (billboards, cards, sprites); full-screen passes do neither.
    bool jitter_small = false;
    float sign_x = 1.f, sign_y = 1.f;
    bool mirror = true;
    bool hud = true;
    bool debug_motion = false;
    bool frame_gen = false; // DLSS frame generation (Streamline loads at launch)
    bool fps_counter = false;
    u32 fg_multiplier = 2; // frames shown per rendered frame with frame generation
    bool fg_reflex = true, fg_reflex_sleep = false; // hidden: Reflex mode and sleep
    bool fsr4_linear = true; // FSR 4 gets linear-light colour instead of sRGB-encoded
    bool camera_snap = true;
    bool scene_camera = true;
    bool camera_majority = true; // the camera most scene draws use (see ObserveSceneConstants)
    float sharpness = 0.6f;
    bool swap = false;
    bool depth_inverted = false;
    u32 preset = 13;       // M: user-tuned default
    vk::Extent2D output{}; // zero: the window size
    int quality = -1;      // -1: from scale
    int upscaler = 0;      // 0 automatic, 1 DLSS, 2 FSR 3.1, 3 FSR 4
    bool motion_blur = false;
    bool character_motion = true; // skinned meshes' own motion vectors (vk_bb_object_motion)
    // The HUD as its own layer, as see-through as measured between frames (see
    // bb_hud_layer.comp), over the upscaled scene.
    bool menu_fix = true;
    bool auto_exposure = true;    // HDR input (decoupled UI): FSR computes the exposure
    bool texture_lod_bias = true; // sharper scene textures while upscaling (BbSceneMipBias)
};

enum class Backend { None, Dlss, Fsr, Fsr4 };

const char* BackendName(Backend backend) {
    return backend == Backend::Dlss ? "DLSS" : backend == Backend::Fsr4 ? "FSR 4" : "FSR 3.1";
}
} // namespace

struct BbTemporalDlss::Impl {
    bool requested{}, failed{}, stopped{};
    Tune tune;
    std::filesystem::path tune_path;
    std::filesystem::file_time_type tune_time{};

    // Per-frame (display copy to display copy) state.
    // Jitter frames run from one DLSS evaluation to the next: the game records part of the next
    // frame's geometry before the display copy of the previous one.
    bool ui_phase{}, dlss_ready{}, dlss_pre_hud{}, history_valid{}, depth_learned{};
    bool evaluated_since_copy{};
    std::optional<VideoCore::ImageId> ui_target;
    u32 jittered_before_copy{};
    bool camera_constants{}, velocity_constants{};
    u32 jittered_draws{};
    // Scene-depth draws of one quad (4-6 indices) this frame: left unjittered, and those of them
    // that test and write depth (jittered with jitter_small).
    u32 small_draws{}, small_depth_draws{};
    u64 small_report_frame{};
    u64 frame{};
    std::array<float, 2> jitter{};
    std::chrono::steady_clock::time_point last_frame{};
    float frame_ms{1000.f / 60.f};
    struct Tracked {
        VideoCore::ImageId id;
        u64 uid;
    };
    std::optional<Tracked> scene_depth, r32_depth;
    // Some areas (the Altar of Despair) switch the game's motion blur off, and with it the depth
    // producer pass. The scene depth is then taken from another pass that samples the memory
    // the producer last sampled; motion then comes from the scene camera alone.
    VAddr producer_depth_address{};
    bool depth_fallback{};        // this frame's scene depth came from another pass
    bool depth_fallback_logged{}; // the switch to and from the fallback is logged once each
    struct Bound {
        VideoCore::ImageId id;
        VideoCore::ImageViewInfo view;
    };
    std::optional<Bound> copy_source, copy_lut, blur_source, hdr_copy_source;
    std::optional<DrawReplacement> replacement;
    std::optional<DisplayCopyReplay> display_copy_replay;
    OwnedImage* replay_output{};
    std::unique_ptr<OwnedImage> pre_lut;         // RenoDX: the composite before the display LUT
    std::unique_ptr<OwnedImage> pre_lut_hudless; // and the same without the HUD
    OwnedImage* replay_hudless{};
    // Decoupled UI + FSR frame generation: the game's UI target just before its first HUD draw,
    // and the game's display copy of it per VideoOut buffer (the frame without the HUD).
    std::optional<VideoCore::ImageId> last_copy_source;
    vk::Format last_copy_format{};
    // The display copy's whole view of its source (format and swizzle: Bloodborne's swaps red
    // and blue), for the Decoupled UI HUD-less snapshot.
    std::optional<VideoCore::ImageViewInfo> last_copy_view;
    std::unique_ptr<OwnedImage> hud_snapshot, hud_dimmed;
    bool hud_snapshot_taken{};
    std::unordered_map<VAddr, std::unique_ptr<OwnedImage>> decoupled_hudless;
    // Decoupled UI: the render-size HDR scene image, whose depth attachment is the scene depth.
    std::optional<VideoCore::ImageId> decoupled_scene;
    bool decoupled{}, decoupled_frame{};
    // A Decoupled UI patch the upscaler cannot handle (or that failed): the game scales itself.
    bool decoupled_unsupported{};
    u64 blur_skips{};
    vk::Extent2D render{}, output{};
    u64 evaluations{}, composites{}, fallbacks{};

    std::unique_ptr<VideoCore::Buffer> coefficients;
    // Scene constants (see bbport docs/upscaler.md): far 3000 at [0], render size at [4..5],
    // view 3x4 at [8..19], projection x/y/z/offset at [52, 57, 62, 63], inverse view at [180..191].
    struct Camera {
        std::array<double, 12> view{}, inv_view{};
        std::array<float, 4> proj{};
        bool valid{};
    } camera, previous_camera;
    bool frame_has_camera{}, camera_this_frame{};
    // Where two map sections are on screen (outside Oedon Chapel), the game sets up the camera
    // once per section, each in that section's coordinates, and the first one of a frame isn't
    // the one most draws use: its motion was off by several pixels (FSR smeared, DLSS lost
    // detail). Each distinct camera of the frame is counted and the most used one is taken.
    struct FrameCamera {
        Camera camera;
        u32 uses{};
    };
    std::vector<FrameCamera> frame_cameras;
    u64 scene_camera_frames{};
    // Motion blur watch (the whole picture but the character blurred after a respawn): the
    // game's own motion blur matrix from the depth producer's constants, against the camera.
    struct BlurWatch {
        bool sampled{};
        std::array<float, 16> matrix{};
        float shift_px{};
        std::array<float, 2> velocity_scale{};
        u32 blur_draws{};
        u32 frames{}, sampled_frames{}, still_frames{}, still_blurred{}, blur_frames{};
        float max_shift{}, sum_shift{};
        u32 streak{};
        bool loading{};
        u64 after_load_frame{};
        std::chrono::steady_clock::time_point window_start{std::chrono::steady_clock::now()};
    } blur;
    std::unique_ptr<OwnedImage> snapshot, motion, upscaled;
    // The HUD as its own layer (render size): this frame's and the last frame's game frame and
    // scene (with k and its confidence), alternating, and the layer itself.
    std::array<std::unique_ptr<OwnedImage>, 2> hud_finals, hud_scenes;
    std::unique_ptr<OwnedImage> hud_layer_image;
    std::unique_ptr<VideoCore::Buffer> hud_dimming; // its histogram of the even darkening
    u32 hud_last{};
    bool hud_history{};
    bool hud_layer_on{};                      // the last composite drew the HUD layer
    std::unique_ptr<OwnedImage> linear_color; // FSR 4 input
    std::unique_ptr<OwnedImage> sharpened;    // decoupled UI: the upscaled scene, sharpened
    // Frame generation reads depth and motion when the frame is presented, after later frames
    // may have overwritten them: each upscaled frame copies them into the next slot.
    struct FrameGenSlot {
        std::unique_ptr<OwnedImage> depth, motion;
    };
    std::array<std::shared_ptr<FrameGenSlot>, 4> frame_gen_slots;
    u32 frame_gen_next{};
    std::optional<FrameGen::FrameInputs> frame_gen_pending;
    // Decoupled UI: frame generation inputs for the game's own frame in each VideoOut buffer.
    std::unordered_map<VAddr, FrameGen::FrameInputs> decoupled_frame_gen;
    bool upscaled_linear{};
    bool upscaled_tone_mapped{}; // RenoDX: reversible tone map, values above 1 kept
    u32 producer_constants{}, producer_depths{}, velocity_draw_constants{};
    std::unordered_map<VAddr, std::unique_ptr<OwnedImage>> outputs;
    std::unique_ptr<ComputePass> motion_pass, composite_pass, composite_hdr_pass, linearize_pass,
        sharpen_pass, hudless_dim_pass, motion_debug_pass, hud_layer_pass;
    vk::UniqueSampler nearest, linear;
    Backend backend{};
    // Menus cover much of the screen. Where the HUD layer can't be drawn (RenoDX), the scene is
    // not jittered while one is open, so the render-size scene seen through the see-through menu
    // panels cannot shimmer. The scene stays upscaled.
    std::unique_ptr<VideoCore::Buffer> coverage, dimming;
    bool menu{};
    std::unique_ptr<FsrUpscaler> fsr;
    std::unique_ptr<Fsr4Addon> fsr4;
    std::optional<Fsr4Addon::ContextDesc> fsr4_rejected_size;
    bool fsr_tried{}, fsr4_tried{}, camera_logged{};
    BbObjectMotion* object_motion{};
    u64 character_motion_frames{};

    Impl() {
        requested = BbTemporalDlssRequested();
        if (!requested)
            return;
        tune_path = BbDlssSettingsPath();
        std::error_code error;
        if (!std::filesystem::exists(tune_path, error))
            std::ofstream{tune_path} << DefaultSettings;
        PollTune();
        LOG_INFO(Render_Vulkan, "[DLSS] Settings file {}", tune_path.string());
    }

    void PollTune() {
        std::error_code error;
        const auto time = std::filesystem::last_write_time(tune_path, error);
        if (error || time == tune_time)
            return;
        tune_time = time;
        std::ifstream file{tune_path};
        std::string line;
        auto next = tune;
        while (std::getline(file, line)) {
            const auto eq = line.find('=');
            if (line.empty() || line[0] == '#' || line[0] == ';' || eq == std::string::npos)
                continue;
            const auto key = line.substr(0, eq);
            const auto value = line.substr(eq + 1);
            const int number = std::atoi(value.c_str());
            if (key == "enabled")
                next.enabled = number != 0;
            else if (key == "jitter")
                next.jitter = number != 0;
            else if (key == "jitter_small")
                next.jitter_small = number != 0;
            else if (key == "jitter_sign_x") // multiplier sent to DLSS (sign and scale)
                next.sign_x = std::clamp(float(std::atof(value.c_str())), -4.f, 4.f);
            else if (key == "jitter_sign_y")
                next.sign_y = std::clamp(float(std::atof(value.c_str())), -4.f, 4.f);
            else if (key == "depth_inverted")
                next.depth_inverted = number != 0;
            else if (key == "preset")
                next.preset = u32(std::clamp(number, 0, 15));
            else if (key == "jitter_swap")
                next.swap = number != 0;
            else if (key == "object_motion")
                next.mirror = number != 0;
            else if (key == "hud")
                next.hud = number != 0;
            else if (key == "sharpness")
                next.sharpness = std::clamp(float(std::atof(value.c_str())), 0.f, 1.f);
            else if (key == "scene_camera")
                next.scene_camera = number != 0;
            else if (key == "camera_majority")
                next.camera_majority = number != 0;
            else if (key == "camera_snap")
                next.camera_snap = number != 0;
            else if (key == "debug_motion")
                next.debug_motion = number != 0;
            else if (key == "frame_gen")
                next.frame_gen = number != 0;
            else if (key == "fps_counter")
                next.fps_counter = number != 0;
            else if (key == "fg_multiplier")
                next.fg_multiplier = u32(std::clamp(number, 2, 6));
            else if (key == "fg_reflex")
                next.fg_reflex = number != 0;
            else if (key == "fg_reflex_sleep")
                next.fg_reflex_sleep = number != 0;
            else if (key == "fsr4_linear")
                next.fsr4_linear = number != 0;
            else if (key == "auto_exposure")
                next.auto_exposure = number != 0;
            else if (key == "menu_fix")
                next.menu_fix = number != 0;
            else if (key == "texture_lod_bias")
                next.texture_lod_bias = number != 0;
            else if (key == "motion_blur")
                next.motion_blur = number != 0;
            else if (key == "character_motion")
                next.character_motion = number != 0;
            else if (key == "upscaler")
                next.upscaler = value.starts_with("dlss")   ? 1
                                : value.starts_with("fsr4") ? 3
                                : value.starts_with("fsr")  ? 2
                                                            : 0;
            else if (key == "quality")
                next.quality = std::clamp(number, -1, 4);
            else if (key == "output") {
                u32 w{}, h{};
                next.output = std::sscanf(value.c_str(), "%ux%u", &w, &h) == 2 && w && h
                                  ? vk::Extent2D{w, h}
                                  : vk::Extent2D{};
            }
        }
        tune = next;
        if (object_motion)
            object_motion->SetActive(tune.character_motion);
        FrameGen::SetEnabled(tune.frame_gen);
        FrameGen::SetCounterVisible(tune.fps_counter);
        FrameGen::SetMultiplier(tune.fg_multiplier);
        FrameGen::SetReflex(tune.fg_reflex, tune.fg_reflex_sleep);
        LOG_INFO(Render_Vulkan,
                 "[DLSS-TEMPORAL] Settings: enabled={} upscaler={} jitter={} sign=({},{}) "
                 "object_motion={} character_motion={} texture_lod_bias={} hud={} "
                 "debug_motion={} quality={} output={}x{}",
                 tune.enabled, tune.upscaler, tune.jitter, tune.sign_x, tune.sign_y, tune.mirror,
                 tune.character_motion, tune.texture_lod_bias, tune.hud, tune.debug_motion,
                 tune.quality, tune.output.width, tune.output.height);
    }

    void EnsurePipelines(const Instance& instance) {
        if (motion_pass)
            return;
        const auto device = instance.GetDevice();
        using T = vk::DescriptorType;
        static constexpr std::array motion_types{T::eCombinedImageSampler, T::eCombinedImageSampler,
                                                 T::eStorageImage, T::eStorageBuffer,
                                                 T::eCombinedImageSampler};
        static constexpr std::array composite_types{
            T::eCombinedImageSampler, T::eCombinedImageSampler, T::eCombinedImageSampler,
            T::eCombinedImageSampler, T::eStorageImage,         T::eCombinedImageSampler,
            T::eStorageBuffer,        T::eStorageBuffer,        T::eCombinedImageSampler};
        motion_pass = std::make_unique<ComputePass>(device, motion_types, BB_DLSS_MOTION_COMP, 96);
        composite_pass =
            std::make_unique<ComputePass>(device, composite_types, BB_DLSS_COMPOSITE_COMP, 8);
        composite_hdr_pass =
            std::make_unique<ComputePass>(device, composite_types, BB_DLSS_COMPOSITE_HDR_COMP, 8);
        static constexpr std::array linearize_types{T::eCombinedImageSampler, T::eStorageImage};
        linearize_pass =
            std::make_unique<ComputePass>(device, linearize_types, BB_DLSS_LINEARIZE_COMP, 4);
        sharpen_pass =
            std::make_unique<ComputePass>(device, linearize_types, BB_DLSS_SHARPEN_HDR_COMP, 4);
        static constexpr std::array motion_debug_types{T::eCombinedImageSampler,
                                                       T::eCombinedImageSampler, T::eStorageImage};
        motion_debug_pass =
            std::make_unique<ComputePass>(device, motion_debug_types, BB_DLSS_MOTION_DEBUG_COMP, 4);
        static constexpr std::array hudless_dim_types{T::eCombinedImageSampler,
                                                      T::eCombinedImageSampler, T::eStorageImage,
                                                      T::eStorageBuffer};
        hudless_dim_pass =
            std::make_unique<ComputePass>(device, hudless_dim_types, BB_HUDLESS_DIM_COMP, 4);
        static constexpr std::array hud_layer_types{
            T::eCombinedImageSampler, T::eCombinedImageSampler, T::eCombinedImageSampler,
            T::eCombinedImageSampler, T::eStorageImage,         T::eStorageImage,
            T::eStorageImage,         T::eStorageBuffer};
        hud_layer_pass =
            std::make_unique<ComputePass>(device, hud_layer_types, BB_HUD_LAYER_COMP, 4);
        const auto sampler = [&](vk::Filter filter) {
            return Make(device.createSamplerUnique(vk::SamplerCreateInfo{
                            .magFilter = filter,
                            .minFilter = filter,
                            .mipmapMode = vk::SamplerMipmapMode::eNearest,
                            .addressModeU = vk::SamplerAddressMode::eClampToEdge,
                            .addressModeV = vk::SamplerAddressMode::eClampToEdge,
                            .addressModeW = vk::SamplerAddressMode::eClampToEdge,
                            .maxLod = 0}),
                        "temporal DLSS sampler");
        };
        nearest = sampler(vk::Filter::eNearest);
        linear = sampler(vk::Filter::eLinear);
    }

    vk::Extent2D OutputFor(vk::Extent2D in) const {
        if (tune.output.width && tune.output.height)
            return BbUpscaleOutput(in, tune.output, true);
        // F1's Regular resolution mode: its output size.
        if (const u64 f1 = Common::BbTuning::upscale_output.load(std::memory_order_relaxed))
            return BbUpscaleOutput(in, {u32(f1 >> 32), u32(f1)}, true);
        const auto window = display.load();
        return BbUpscaleOutput(in, {u32(window >> 32), u32(window)});
    }
    std::atomic<u64> display{};

    void AdvanceJitter() {
        const float scale =
            output.width && render.width ? float(output.width) / float(render.width) : 1.5f;
        const u32 phases = std::max<u32>(8, u32(std::ceil(8.f * scale * scale)));
        const u32 index = u32(frame % phases) + 1;
        jitter = {Halton(index, 2) - 0.5f, Halton(index, 3) - 0.5f};
    }

    // After each evaluation attempt: per-frame guide inputs restart and the jitter advances.
    // Largest shift, in pixels at 1280x720, that the game's motion blur matrix (previous to
    // current clip space, rows) gives a grid of points on screen at a near and a far depth.
    static float BlurShiftPx(const std::array<float, 16>& rows) {
        float worst = 0.0f;
        for (const float x : {-0.8f, 0.0f, 0.8f}) {
            for (const float y : {-0.8f, 0.0f, 0.8f}) {
                for (const float z : {0.1f, 0.9f}) {
                    const std::array<float, 4> p{x, y, z, 1.0f};
                    std::array<float, 4> q{};
                    for (int i = 0; i < 4; ++i) {
                        for (int k = 0; k < 4; ++k) {
                            q[i] += rows[i * 4 + k] * p[k];
                        }
                    }
                    if (!std::isfinite(q[0]) || !std::isfinite(q[1]) || !(std::abs(q[3]) > 1e-6f))
                        return std::numeric_limits<float>::infinity();
                    worst = std::max(
                        worst, std::hypot((q[0] / q[3] - x) * 640.0f, (q[1] / q[3] - y) * 360.0f));
                }
            }
        }
        return worst;
    }

    // How far the game's blur matrix (last frame's position of each pixel) is from "nothing
    // moved": its largest difference from the identity. With the camera still it should be ~0;
    // BlurShiftPx's test depths (0.1, 0.9) read huge shifts from matrices that are fine near the
    // depths the game actually uses, so it cannot tell a stuck matrix from a good one.
    static float MatrixOffIdentity(const std::array<float, 16>& rows) {
        float worst = 0.0f;
        for (int i = 0; i < 16; ++i) {
            const float expected = (i % 5 == 0) ? 1.0f : 0.0f;
            const float diff = std::abs(rows[i] - expected);
            if (!std::isfinite(diff))
                return std::numeric_limits<float>::infinity();
            worst = std::max(worst, diff);
        }
        return worst;
    }

    float CameraChange() const {
        if (!camera.valid || !previous_camera.valid)
            return -1.0f;
        double change = 0.0;
        for (size_t i = 0; i < camera.view.size(); ++i)
            change = std::max(change, std::abs(camera.view[i] - previous_camera.view[i]));
        return float(change);
    }

    std::string DescribeBlurFrame(float camera_change) const {
        return fmt::format(
            "game blur matrix shifts the image up to {:.1f} px, {:.4f} from not moving | camera "
            "change {} | blur passes "
            "{} (game motion blur {}) | velocity scale {:.4g} {:.4g} | depth {} at {:#x} | "
            "matrix rows [{:.5f} {:.5f} {:.5f} {:.5f}] [{:.5f} {:.5f} {:.5f} {:.5f}] [{:.5f} "
            "{:.5f} {:.5f} {:.5f}] [{:.5f} {:.5f} {:.5f} {:.5f}]",
            blur.shift_px, MatrixOffIdentity(blur.matrix),
            camera_change < 0.0f ? std::string{"unknown"} : fmt::format("{:.5f}", camera_change),
            blur.blur_draws, tune.motion_blur ? "on" : "off", blur.velocity_scale[0],
            blur.velocity_scale[1], depth_fallback ? "from another pass" : "from the producer",
            producer_depth_address, blur.matrix[0], blur.matrix[1], blur.matrix[2], blur.matrix[3],
            blur.matrix[4], blur.matrix[5], blur.matrix[6], blur.matrix[7], blur.matrix[8],
            blur.matrix[9], blur.matrix[10], blur.matrix[11], blur.matrix[12], blur.matrix[13],
            blur.matrix[14], blur.matrix[15]);
    }

    void FinishBlurFrame() {
        // A still camera whose blur matrix is this far from "nothing moved" blurs the picture.
        constexpr float MovingMatrix = 0.01f;
        constexpr float StillCamera = 1e-3f;
        const float camera_change = CameraChange();
        ++blur.frames;
        blur.blur_frames += blur.blur_draws != 0;
        if (blur.sampled) {
            ++blur.sampled_frames;
            blur.max_shift = std::max(blur.max_shift, blur.shift_px);
            blur.sum_shift += std::isfinite(blur.shift_px) ? blur.shift_px : 0.0f;
            const bool still = camera_change >= 0.0f && camera_change < StillCamera;
            blur.still_frames += still;
            if (still && MatrixOffIdentity(blur.matrix) > MovingMatrix) {
                ++blur.still_blurred;
                if (++blur.streak == 30)
                    LOG_WARNING(Render_Vulkan,
                                "[BB-BLUR] Camera still for 30 frames but the game's motion "
                                "blur matrix still moves the image (the picture is blurred): {}",
                                DescribeBlurFrame(camera_change));
            } else {
                if (blur.streak >= 30)
                    LOG_INFO(Render_Vulkan,
                             "[BB-BLUR] Game blur matrix back to the camera after {} frames",
                             blur.streak);
                blur.streak = 0;
            }
        }
        const bool loading = Core::BbGameFixes::IsLoadingScreen();
        if (blur.loading && !loading)
            blur.after_load_frame = frame;
        blur.loading = loading;
        if (blur.after_load_frame && frame - blur.after_load_frame <= 300 &&
            (frame - blur.after_load_frame) % 60 == 0)
            LOG_INFO(Render_Vulkan, "[BB-BLUR] {} frames after the loading screen: {}{}",
                     frame - blur.after_load_frame, blur.sampled ? "" : "(no depth producer) ",
                     DescribeBlurFrame(camera_change));
        const auto now = std::chrono::steady_clock::now();
        if (now - blur.window_start >= std::chrono::seconds{10}) {
            if (blur.frames != 0)
                LOG_INFO(Render_Vulkan,
                         "[BB-BLUR] last 10 s: {} frames, {} with the depth producer, {} with "
                         "motion blur passes | game blur matrix shift avg {:.1f} px, max {:.1f} "
                         "px | camera still in {} frames, {} of them with a moving blur matrix "
                         "(over {:.2f} from not moving)",
                         blur.frames, blur.sampled_frames, blur.blur_frames,
                         blur.sampled_frames ? blur.sum_shift / blur.sampled_frames : 0.0f,
                         blur.max_shift, blur.still_frames, blur.still_blurred, MovingMatrix);
            const auto streak = blur.streak;
            const auto loading_state = blur.loading;
            const auto after_load = blur.after_load_frame;
            blur = {};
            blur.streak = streak;
            blur.loading = loading_state;
            blur.after_load_frame = after_load;
            blur.window_start = now;
        }
        blur.sampled = false;
        blur.blur_draws = 0;
    }

    void FrameBoundary(Scheduler& scheduler, BbVelocityMirror& mirror) {
        FinishBlurFrame();
        if (frame - small_report_frame >= 600) {
            small_report_frame = frame;
            LOG_INFO(Render_Vulkan,
                     "[DLSS-TEMPORAL] Jitter: {} draws jittered this frame; {} single-quad scene "
                     "draws not jittered by the usual rule, {} of them testing and writing "
                     "depth (jitter_small={})",
                     jittered_draws, small_draws, small_depth_draws, tune.jitter_small);
        }
        small_draws = small_depth_draws = 0;
        camera_constants = velocity_constants = depth_learned = false;
        jittered_draws = jittered_before_copy = 0;
        producer_constants = producer_depths = velocity_draw_constants = 0;
        r32_depth.reset();
        depth_fallback = false;
        mirror.ConsumeFrame(scheduler.CommandBuffer()); // start a fresh mirror every frame
        if (object_motion)
            object_motion->OnFrameStart();
        ++frame;
        AdvanceJitter();
    }

    // At the game's display copy: presentation bookkeeping only.
    void DisplayCopyDone() {
        decoupled_frame = false;
        if (!evaluated_since_copy)
            history_valid = false;
        evaluated_since_copy = false;
        // Camera frames run display copy to display copy: HUD/post passes after the evaluation
        // may bind the same scene constants again.
        if (!frame_has_camera)
            camera.valid = previous_camera.valid = false;
        frame_has_camera = false;
        ui_phase = dlss_ready = dlss_pre_hud = false;
        jittered_before_copy = jittered_draws;
        copy_source.reset();
        copy_lut.reset();
        const auto now = std::chrono::steady_clock::now();
        if (last_frame.time_since_epoch().count())
            frame_ms = std::clamp(
                std::chrono::duration<float, std::milli>(now - last_frame).count(), 1.f, 100.f);
        last_frame = now;
        if (++copies % 30 == 0)
            PollTune();
    }
    u64 copies{};

    Backend ChooseBackend(const Instance& instance, Scheduler& scheduler, vk::Extent2D in,
                          vk::Extent2D out) {
        const auto* ngx = instance.GetDlssNgx();
        const bool dlss = ngx && ngx->IsAvailable();
        const bool rejected_size = fsr4_rejected_size && fsr4_rejected_size->render == in &&
                                   fsr4_rejected_size->output == out;
        const auto fsr_ready = [&] {
            if (!fsr && !fsr_tried) { // load the DLL only when FSR is actually used
                fsr_tried = true;
                fsr = FsrUpscaler::Create(instance);
            }
            return fsr != nullptr;
        };
        const auto fsr4_ready = [&] {
            if (rejected_size)
                return false;
            if (!fsr4 && !fsr4_tried) {
                fsr4_tried = true;
                fsr4 = Fsr4Addon::Create(instance, scheduler);
            }
            return fsr4 != nullptr;
        };
        switch (tune.upscaler) {
        case 1:
            return dlss ? Backend::Dlss : Backend::None;
        case 2:
            return fsr_ready() ? Backend::Fsr : Backend::None;
        case 3:
            if (fsr4_ready())
                return Backend::Fsr4;
            return rejected_size && fsr_ready() ? Backend::Fsr : Backend::None;
        default:
            return dlss           ? Backend::Dlss
                   : fsr4_ready() ? Backend::Fsr4
                   : fsr_ready()  ? Backend::Fsr
                                  : Backend::None;
        }
    }

    std::string NoUpscalerReason() const {
        const auto dlss = DlssProblem();
        const std::string no_fsr =
            FsrUpscaler::Present()
                ? "FSR could not be loaded (amd_fidelityfx_vk.dll is damaged or incompatible)."
                : "FSR needs amd_fidelityfx_vk.dll next to shadPS4.exe.";
        if (tune.upscaler == 1)
            return dlss.empty() ? "DLSS is not available on this system." : dlss;
        if (tune.upscaler == 2)
            return no_fsr;
        if (tune.upscaler == 3) {
            if (fsr4_rejected_size)
                return "FSR 4 is unavailable at this resolution. " + no_fsr;
            return Fsr4Addon::Present()
                       ? "FSR 4 could not start (see the log; it needs a Radeon RX 6000 / RTX "
                         "or newer GPU and the complete fsr4 folder)."
                       : "FSR 4 needs the FSR 4 add-on (the fsr4 folder next to shadPS4.exe).";
        }
        return (dlss.empty() ? std::string{} : dlss + " ") + no_fsr;
    }

    // FSR uses the camera planes and field of view to judge depth differences. Depth is
    // p2 + p3 / z for view distance z, so depth 0 and 1 give the near and far planes.
    FsrUpscaler::Camera FsrCamera() {
        FsrUpscaler::Camera result{0.1f, 3000.0f, 1.0f};
        const auto& p = camera.proj;
        if (camera.valid && p[2] != 0.0f && p[2] != 1.0f && p[3] != 0.0f) {
            const float a = std::abs(-p[3] / p[2]), b = std::abs(p[3] / (1.0f - p[2]));
            if (std::isfinite(a) && std::isfinite(b) && std::min(a, b) > 0.0f) {
                result.near_plane = std::min(a, b);
                result.far_plane = std::max(a, b);
            }
        }
        if (camera.valid && p[1] != 0.0f)
            result.fov_y = 2.0f * std::atan(1.0f / std::abs(p[1]));
        if (!camera_logged && camera.valid) {
            camera_logged = true;
            LOG_INFO(Render_Vulkan, "[FSR] Camera near={} far={} fovY={} (proj {} {} {} {})",
                     result.near_plane, result.far_plane, result.fov_y, p[0], p[1], p[2], p[3]);
        }
        return result;
    }

    // Copies this frame's depth and motion for frame generation and describes its camera.
    void CaptureFrameGen(const Instance& instance, Runtime& runtime, Scheduler& scheduler,
                         vk::CommandBuffer command, VideoCore::Image& depth, vk::Extent2D in,
                         const DlssNgx::EvalDesc& eval) {
        frame_gen_pending.reset();
        if (!FrameGen::Active() || !frame_has_camera || !camera.valid || !previous_camera.valid ||
            !(depth.backing->image.image_ci.usage & vk::ImageUsageFlagBits::eTransferSrc))
            return;
        auto& owner = frame_gen_slots[frame_gen_next];
        frame_gen_next = (frame_gen_next + 1) % frame_gen_slots.size();
        // A queued present (or a VideoOut buffer) can still own this slot on a slow backend.
        if (!owner || owner.use_count() != 1)
            owner = std::make_shared<FrameGenSlot>();
        auto& slot = *owner;
        // Also retain captures which are dropped before reaching the presenter.
        scheduler.DeferOperation([resources = owner] {});
        using U = vk::ImageUsageFlagBits;
        if (!slot.depth || slot.depth->extent != in) {
            slot.depth =
                std::make_unique<OwnedImage>(instance, vk::Format::eR32Sfloat, in,
                                             U::eSampled | U::eTransferDst | U::eTransferSrc);
            slot.motion =
                std::make_unique<OwnedImage>(instance, vk::Format::eR16G16Sfloat, in,
                                             U::eSampled | U::eTransferDst | U::eTransferSrc);
        }
        const vk::ImageCopy region{
            .srcSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 1},
            .dstSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 1},
            .extent = {in.width, in.height, 1}};
        runtime.Transit(&depth, vk::ImageLayout::eTransferSrcOptimal,
                        vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead);
        runtime.FlushBarriers();
        motion->Transit(command, vk::ImageLayout::eTransferSrcOptimal,
                        vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead);
        for (auto* image : {slot.depth.get(), slot.motion.get()})
            image->Transit(command, vk::ImageLayout::eTransferDstOptimal,
                           vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite);
        command.copyImage(depth.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                          slot.depth->image.image, vk::ImageLayout::eTransferDstOptimal, region);
        command.copyImage(motion->image.image, vk::ImageLayout::eTransferSrcOptimal,
                          slot.motion->image.image, vk::ImageLayout::eTransferDstOptimal, region);
        // The motion image is sampled again later this frame (HUD composite).
        for (auto* image : {slot.depth.get(), slot.motion.get(), motion.get()})
            image->Transit(command, vk::ImageLayout::eShaderReadOnlyOptimal,
                           vk::PipelineStageFlagBits2::eAllCommands,
                           vk::AccessFlagBits2::eShaderRead);

        // Column-vector matrices, row by row, in double.
        using Mat = std::array<double, 16>;
        static constexpr Mat Identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        const auto mul = [](const Mat& a, const Mat& b) {
            Mat r{};
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    for (int k = 0; k < 4; ++k)
                        r[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
            return r;
        };
        const auto invert = [](Mat m) {
            Mat r = Identity;
            for (int c = 0; c < 4; ++c) {
                int pivot = c;
                for (int i = c + 1; i < 4; ++i)
                    if (std::abs(m[i * 4 + c]) > std::abs(m[pivot * 4 + c]))
                        pivot = i;
                for (int j = 0; j < 4; ++j) {
                    std::swap(m[c * 4 + j], m[pivot * 4 + j]);
                    std::swap(r[c * 4 + j], r[pivot * 4 + j]);
                }
                const double d = m[c * 4 + c];
                if (d == 0.0)
                    return Identity;
                for (int j = 0; j < 4; ++j) {
                    m[c * 4 + j] /= d;
                    r[c * 4 + j] /= d;
                }
                for (int i = 0; i < 4; ++i) {
                    if (i == c)
                        continue;
                    const double f = m[i * 4 + c];
                    for (int j = 0; j < 4; ++j) {
                        m[i * 4 + j] -= f * m[c * 4 + j];
                        r[i * 4 + j] -= f * r[c * 4 + j];
                    }
                }
            }
            return r;
        };
        // Streamline wants row-vector matrices: the transposes.
        const auto out = [](const Mat& m) {
            std::array<float, 16> r{};
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    r[i * 4 + j] = float(m[j * 4 + i]);
            return r;
        };
        // D3D-style clip space: z_clip = zs * z + zo * w, w_clip = z.
        const auto projection = [](const std::array<float, 4>& p) {
            return Mat{p[0], 0, 0, 0, 0, p[1], 0, 0, 0, 0, p[2], p[3], 0, 0, 1, 0};
        };
        Mat view_to_prev_view{};
        const auto& a = previous_camera.view;
        const auto b = InverseAffine(camera.view); // see RunDlss: the game's is float32
        for (int r = 0; r < 3; ++r)
            for (int col = 0; col < 4; ++col) {
                double value = a[r * 4 + 0] * b[0 * 4 + col] + a[r * 4 + 1] * b[1 * 4 + col] +
                               a[r * 4 + 2] * b[2 * 4 + col];
                if (col == 3)
                    value += a[r * 4 + 3];
                view_to_prev_view[r * 4 + col] = value;
            }
        view_to_prev_view[15] = 1.0;
        const Mat view_to_clip = projection(camera.proj);
        const Mat clip_to_view = invert(view_to_clip);
        const Mat clip_to_prev_clip =
            mul(mul(projection(previous_camera.proj), view_to_prev_view), clip_to_view);

        FrameGen::FrameInputs inputs{};
        inputs.resources = owner;
        inputs.depth = slot.depth->image.image;
        inputs.depth_view = *slot.depth->view;
        inputs.motion = slot.motion->image.image;
        inputs.motion_view = *slot.motion->view;
        inputs.render = in;
        inputs.view_to_clip = out(view_to_clip);
        inputs.clip_to_view = out(clip_to_view);
        inputs.clip_to_prev_clip = out(clip_to_prev_clip);
        inputs.prev_clip_to_clip = out(invert(clip_to_prev_clip));
        // The inverse view's columns: camera right, up, forward and position in the world.
        for (int i = 0; i < 3; ++i) {
            inputs.right[i] = float(b[i * 4 + 0]);
            inputs.up[i] = float(b[i * 4 + 1]);
            inputs.forward[i] = float(b[i * 4 + 2]);
            inputs.position[i] = float(b[i * 4 + 3]);
        }
        const auto lens = FsrCamera();
        inputs.near_plane = lens.near_plane;
        inputs.far_plane = lens.far_plane;
        inputs.fov_y = lens.fov_y;
        inputs.aspect = camera.proj[1] / camera.proj[0];
        inputs.jitter = {eval.jitter_x, eval.jitter_y};
        inputs.depth_inverted = tune.depth_inverted;
        inputs.reset = eval.reset;
        frame_gen_pending = inputs;
    }

    void InvalidateOutputs() {
        for (auto& [_, image] : outputs)
            image->fresh = false;
    }

    // With `upscale_to`, the decoupled-UI mode: `source` is the HDR scene, upscaled to that size
    // and copied over the game's own upscale instead of being composited at display time.
    bool RunDlss(const Instance& instance, Runtime& runtime, Scheduler& scheduler,
                 VideoCore::TextureCache& cache, BbVelocityMirror& mirror, VideoCore::Image& source,
                 bool pre_hud, std::optional<vk::Extent2D> upscale_to = std::nullopt);
    void SnapshotHud(const Instance& instance, Runtime& runtime, Scheduler& scheduler,
                     VideoCore::Image& target) {
        scheduler.EndRendering();
        EnsurePipelines(instance);
        const vk::Extent2D size{target.info.size.width, target.info.size.height};
        // Float, copied by a shader through the game's own view: RenoDX widens the target and
        // keeps highlights above 1 in it.
        if (!hud_snapshot || hud_snapshot->extent != size) {
            if (hud_snapshot)
                scheduler.Finish();
            using U = vk::ImageUsageFlagBits;
            hud_snapshot = std::make_unique<OwnedImage>(instance, vk::Format::eR16G16B16A16Sfloat,
                                                        size, U::eSampled | U::eStorage);
            hud_dimmed = std::make_unique<OwnedImage>(instance, vk::Format::eR16G16B16A16Sfloat,
                                                      size, U::eSampled | U::eStorage);
        }
        const auto command = scheduler.CommandBuffer(); // after Finish(), see RunDlss
        // Read as the display copy reads it, swizzle included: the replay feeds this to the
        // display copy in place of its source, and the dimming pass compares it with that view.
        VideoCore::ImageViewInfo view_info{};
        if (last_copy_view)
            view_info = *last_copy_view;
        else
            view_info.format = last_copy_format != vk::Format::eUndefined
                                   ? last_copy_format
                                   : target.info.pixel_format;
        const auto& view = target.FindView(view_info);
        runtime.Transit(&target, vk::ImageLayout::eShaderReadOnlyOptimal,
                        vk::PipelineStageFlagBits2::eComputeShader,
                        vk::AccessFlagBits2::eShaderRead);
        runtime.FlushBarriers();
        hud_snapshot->Transit(command, vk::ImageLayout::eGeneral,
                              vk::PipelineStageFlagBits2::eComputeShader,
                              vk::AccessFlagBits2::eShaderWrite);
        const std::array images{
            vk::DescriptorImageInfo{*nearest, *view.image_view,
                                    vk::ImageLayout::eShaderReadOnlyOptimal},
            vk::DescriptorImageInfo{{}, *hud_snapshot->view, vk::ImageLayout::eGeneral}};
        const std::array writes{
            vk::WriteDescriptorSet{.dstBinding = 0,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                   .pImageInfo = &images[0]},
            vk::WriteDescriptorSet{.dstBinding = 1,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eStorageImage,
                                   .pImageInfo = &images[1]}};
        const u32 copy_flags = 2; // plain copy
        command.bindPipeline(vk::PipelineBindPoint::eCompute, *linearize_pass->pipeline);
        command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *linearize_pass->layout, 0,
                                     writes);
        command.pushConstants(*linearize_pass->layout, vk::ShaderStageFlagBits::eCompute, 0,
                              sizeof(copy_flags), &copy_flags);
        command.dispatch((size.width + 7) / 8, (size.height + 7) / 8, 1);
        hud_snapshot->Transit(command, vk::ImageLayout::eShaderReadOnlyOptimal,
                              vk::PipelineStageFlagBits2::eFragmentShader,
                              vk::AccessFlagBits2::eShaderRead);
        scheduler.GetDynamicState().Invalidate();
        hud_snapshot_taken = true;
    }

    // At a Decoupled UI display copy: draw it again from the HUD-less snapshot.
    void ReplayHudless(const Instance& instance, Runtime& runtime, Scheduler& scheduler,
                       VideoCore::TextureCache& cache, VideoCore::Image& target,
                       vk::Format target_view_format) {
        auto& slot = decoupled_hudless[target.info.guest_address];
        const vk::Extent2D size{target.info.size.width, target.info.size.height};
        scheduler.EndRendering();
        // RenoDX's display copy writes values above 1: a float image, presented like the
        // regular path's outputs.
        const bool renodx = RenoDxLoaded();
        const auto format = renodx ? vk::Format::eR16G16B16A16Sfloat : target.info.pixel_format;
        if (!dimming)
            dimming = std::make_unique<VideoCore::Buffer>(instance, 0, DimmingBytes,
                                                          VideoCore::MemoryType::DeviceLocal);
        if (!slot || slot->extent != size || slot->format != format) {
            if (slot)
                scheduler.Finish();
            using U = vk::ImageUsageFlagBits;
            slot = renodx ? std::make_unique<OwnedImage>(instance, format, size,
                                                         U::eSampled | U::eColorAttachment, true)
                          : std::make_unique<OwnedImage>(instance, format, size,
                                                         U::eSampled | U::eColorAttachment, false,
                                                         vk::ImageCreateFlagBits::eMutableFormat);
            slot->swapped = renodx;
        }
        const auto command = scheduler.CommandBuffer();

        // The snapshot with the dimming most of the finished frame shows (fades, dialogs).
        auto& final_image = cache.GetImage(copy_source->id);
        const auto& final_view = final_image.FindView(copy_source->view);
        runtime.Transit(&final_image, vk::ImageLayout::eShaderReadOnlyOptimal,
                        vk::PipelineStageFlagBits2::eComputeShader |
                            vk::PipelineStageFlagBits2::eFragmentShader,
                        vk::AccessFlagBits2::eShaderRead);
        runtime.FlushBarriers();
        hud_dimmed->Transit(command, vk::ImageLayout::eGeneral,
                            vk::PipelineStageFlagBits2::eComputeShader,
                            vk::AccessFlagBits2::eShaderWrite);
        const vk::BufferMemoryBarrier2 clear_barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = dimming->Handle(),
            .offset = 0,
            .size = DimmingBytes};
        command.pipelineBarrier2(vk::DependencyInfo{.bufferMemoryBarrierCount = 1,
                                                    .pBufferMemoryBarriers = &clear_barrier});
        command.fillBuffer(dimming->Handle(), 0, DimmingBytes, 0);
        auto pass_barrier = clear_barrier;
        pass_barrier.srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
        pass_barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
        pass_barrier.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader;
        pass_barrier.dstAccessMask =
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
        command.pipelineBarrier2(vk::DependencyInfo{.bufferMemoryBarrierCount = 1,
                                                    .pBufferMemoryBarriers = &pass_barrier});
        const auto ro = vk::ImageLayout::eShaderReadOnlyOptimal;
        const std::array images{
            vk::DescriptorImageInfo{*nearest, *final_view.image_view, ro},
            vk::DescriptorImageInfo{*nearest, *hud_snapshot->view, ro},
            vk::DescriptorImageInfo{{}, *hud_dimmed->view, vk::ImageLayout::eGeneral}};
        const vk::DescriptorBufferInfo dimming_info{dimming->Handle(), 0, DimmingBytes};
        std::array<vk::WriteDescriptorSet, 4> writes;
        for (u32 i = 0; i < images.size(); ++i)
            writes[i] = {.dstBinding = i,
                         .descriptorCount = 1,
                         .descriptorType = i == 2 ? vk::DescriptorType::eStorageImage
                                                  : vk::DescriptorType::eCombinedImageSampler,
                         .pImageInfo = &images[i]};
        writes[3] = {.dstBinding = 3,
                     .descriptorCount = 1,
                     .descriptorType = vk::DescriptorType::eStorageBuffer,
                     .pBufferInfo = &dimming_info};
        command.bindPipeline(vk::PipelineBindPoint::eCompute, *hudless_dim_pass->pipeline);
        command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *hudless_dim_pass->layout, 0,
                                     writes);
        for (const u32 mode : {0u, 1u}) {
            if (mode == 1) {
                pass_barrier.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
                pass_barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
                command.pipelineBarrier2(vk::DependencyInfo{
                    .bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &pass_barrier});
            }
            command.pushConstants(*hudless_dim_pass->layout, vk::ShaderStageFlagBits::eCompute, 0,
                                  sizeof(mode), &mode);
            command.dispatch((size.width + 7) / 8, (size.height + 7) / 8, 1);
        }
        hud_dimmed->Transit(command, ro, vk::PipelineStageFlagBits2::eFragmentShader,
                            vk::AccessFlagBits2::eShaderRead);

        slot->Transit(command, vk::ImageLayout::eColorAttachmentOptimal,
                      vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                      vk::AccessFlagBits2::eColorAttachmentWrite);
        display_copy_replay = DisplayCopyReplay{
            .shader_hash = DisplayCopy,
            .extent = size,
            .hudless_input = *hud_dimmed->view,
            .hudless_output = renodx ? *slot->view : slot->TargetView(target_view_format)};
        replay_hudless = slot.get();
        slot->fresh = true;
        scheduler.GetDynamicState().Invalidate();
    }

    void Composite(const Instance& instance, Runtime& runtime, Scheduler& scheduler,
                   VideoCore::TextureCache& cache, VideoCore::Image& target);
};

bool BbTemporalDlss::Impl::RunDlss(const Instance& instance, Runtime& runtime, Scheduler& scheduler,
                                   VideoCore::TextureCache& cache, BbVelocityMirror& mirror,
                                   VideoCore::Image& source, bool pre_hud,
                                   std::optional<vk::Extent2D> upscale_to) {
    if (auto* profiler = GpuProfiler::Get(); profiler && profiler->Records(&scheduler)) {
        profiler->Mark(0xD155ull, [] { return std::string{"DLSS (temporal upscale and passes)"}; });
    }
    const bool hdr = upscale_to.has_value();
    Common::Crumbs::Add("upscaler: evaluate (width, height)", source.info.size.width,
                        source.info.size.height);
    auto* ngx = instance.GetDlssNgx();
    const vk::Extent2D in{source.info.size.width, source.info.size.height};
    VideoCore::Image* depth =
        r32_depth ? cache.TryGetImage(r32_depth->id, r32_depth->uid) : nullptr;
    const auto reject = [&](const char* reason) {
        ReportFrame(std::string_view{reason} == "no upscaler"
                        ? NoUpscalerReason()
                        : "Waiting for gameplay (menus and loading screens use the normal image).");
        if (++fallbacks <= 8 || fallbacks % 600 == 0)
            LOG_INFO(Render_Vulkan, "[DLSS-TEMPORAL] Stock frame ({}), fallbacks={}", reason,
                     fallbacks);
        return false;
    };
    const auto out = upscale_to ? *upscale_to : OutputFor(in);
    const auto chosen = ChooseBackend(instance, scheduler, in, out);
    if (chosen == Backend::None)
        return reject("no upscaler");
    if (!source.backing || source.backing->image.image_ci.samples != vk::SampleCountFlagBits::e1 ||
        source.backing->image.image_ci.extent != vk::Extent3D{in.width, in.height, 1})
        return reject("unsupported color target");
    if (!depth || !depth->backing || depth->info.size.width != in.width ||
        depth->info.size.height != in.height ||
        depth->backing->image.image_ci.format != vk::Format::eR32Sfloat ||
        depth->backing->image.image_ci.samples != vk::SampleCountFlagBits::e1)
        return reject("scene depth not observed");
    // Without the depth producer, motion comes from the scene camera alone (its constants
    // feed only the motion-blur matrix the scene camera replaces).
    const bool camera_only =
        tune.scene_camera && frame_has_camera && camera.valid && previous_camera.valid;
    if (!camera_constants && !camera_only)
        return reject("camera constants not observed");
    if (!coefficients)
        coefficients = std::make_unique<VideoCore::Buffer>(instance, 0, CoefficientBytes,
                                                           VideoCore::MemoryType::DeviceLocal);
    if (depth_fallback != depth_fallback_logged) {
        depth_fallback_logged = depth_fallback;
        LOG_INFO(Render_Vulkan, "[DLSS-TEMPORAL] {}",
                 depth_fallback ? "Game motion blur off here: scene depth from another pass"
                                : "Game motion blur back: depth from the motion blur pass");
    }

    scheduler.EndRendering();
    EnsurePipelines(instance);
    // RenoDX widens the game's 8-bit targets to floating point and keeps values above 1 there;
    // the scene is copied with a shader into a float image so they survive.
    const bool renodx = RenoDxLoaded() && !hdr;
    const vk::Format color_format =
        renodx ? vk::Format::eR16G16B16A16Sfloat : source.info.pixel_format;
    if (!snapshot || snapshot->extent != in || snapshot->format != color_format || !upscaled ||
        upscaled->extent != out) {
        scheduler.Finish(); // owned images may still be in flight
        using U = vk::ImageUsageFlagBits;
        snapshot = std::make_unique<OwnedImage>(instance, color_format, in,
                                                renodx ? U::eSampled | U::eStorage
                                                       : U::eSampled | U::eTransferDst);
        motion = std::make_unique<OwnedImage>(instance, vk::Format::eR16G16Sfloat, in,
                                              U::eSampled | U::eStorage | U::eTransferSrc);
        upscaled = std::make_unique<OwnedImage>(instance, vk::Format::eR16G16B16A16Sfloat, out,
                                                U::eSampled | U::eStorage | U::eTransferSrc);
        linear_color = std::make_unique<OwnedImage>(instance, vk::Format::eR16G16B16A16Sfloat, in,
                                                    U::eSampled | U::eStorage);
        sharpened = upscale_to
                        ? std::make_unique<OwnedImage>(instance, vk::Format::eR16G16B16A16Sfloat,
                                                       out, U::eStorage | U::eTransferSrc)
                        : nullptr;
        outputs.clear();
        history_valid = false;
        LOG_INFO(Render_Vulkan, "[DLSS-TEMPORAL] Resources {}x{} -> {}x{}", in.width, in.height,
                 out.width, out.height);
    }
    render = in;
    output = out;
    if (chosen != backend) {
        scheduler.Finish();
        if (backend == Backend::Dlss && ngx)
            ngx->ReleaseFeatureAfterGpuDrain();
        if (backend == Backend::Fsr && fsr)
            fsr->DestroyContextAfterGpuDrain();
        if (backend == Backend::Fsr4 && fsr4)
            fsr4->ReleaseContextAfterGpuDrain();
        backend = chosen;
        history_valid = false;
        LOG_INFO(Render_Vulkan, "[DLSS-TEMPORAL] Upscaler: {}", BackendName(backend));
    }
    if (backend == Backend::Dlss) {
        const DlssNgx::FeatureDesc desc{
            in.width, in.height, out.width, out.height,
            tune.quality >= 0 ? tune.quality
                              : DlssNgx::QualityForScale(float(out.width) / float(in.width)),
            tune.depth_inverted, // UID47/R32 depth measured near < far
            tune.preset,
            // With RenoDX DLSS gets the scene as linear 16-bit float (see `linear` below).
            hdr || RenoDxLoaded()};
        if (!ngx->HasFeature(desc)) {
            scheduler.Finish();
            if (!ngx->CreateFeature(scheduler.CommandBuffer(), desc)) {
                (upscale_to ? decoupled_unsupported : failed) = true;
                LOG_ERROR(Render_Vulkan,
                          "[DLSS-TEMPORAL] Feature creation failed; stock rendering");
                return false;
            }
            history_valid = false;
        }
    } else if (backend == Backend::Fsr4) {
        const Fsr4Addon::ContextDesc desc{in, out};
        if (!fsr4->HasContext(desc)) {
            ReportFrame("Initializing FSR 4...");
            const auto started = std::chrono::steady_clock::now();
            scheduler.Finish();
            if (!fsr4->CreateContext(desc)) {
                fsr4_rejected_size = desc;
                backend = Backend::None;
                ReportFrame("FSR 4 could not initialize; trying FSR 3.1.");
                LOG_WARNING(Render_Vulkan,
                            "[FSR4] Context {}x{} -> {}x{} unavailable; trying FSR 3.1", in.width,
                            in.height, out.width, out.height);
                return false;
            }
            LOG_INFO(Render_Vulkan, "[FSR4] Context ready in {:.0f} ms",
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                               started)
                         .count());
            fsr4_rejected_size.reset();
            history_valid = false;
        }
    } else {
        const FsrUpscaler::ContextDesc desc{in, out, tune.depth_inverted, hdr,
                                            hdr && tune.auto_exposure};
        if (!fsr->HasContext(desc)) {
            scheduler.Finish();
            if (!fsr->CreateContext(desc)) {
                (upscale_to ? decoupled_unsupported : failed) = true;
                LOG_ERROR(Render_Vulkan, "[FSR] Context creation failed; stock rendering");
                return false;
            }
            history_valid = false;
        }
    }

    // Only now: the Finish() calls above submit the command buffer, and recording into a
    // submitted one crashes AMD's driver.
    const auto command = scheduler.CommandBuffer();
    // Snapshot of the scene as DLSS color input (pre-HUD at the first Scaleform draw).
    VideoCore::ImageViewInfo source_info{};
    source_info.format = vk::Format::eR8G8B8A8Unorm;
    const auto& source_view = source.FindView(source_info);
    if (renodx)
        runtime.Transit(&source, vk::ImageLayout::eShaderReadOnlyOptimal,
                        vk::PipelineStageFlagBits2::eComputeShader,
                        vk::AccessFlagBits2::eShaderRead);
    else
        runtime.Transit(&source, vk::ImageLayout::eTransferSrcOptimal,
                        vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead);
    VideoCore::ImageViewInfo depth_info{};
    depth_info.format = vk::Format::eR32Sfloat;
    const auto& depth_view = depth->FindView(depth_info);
    runtime.Transit(depth, vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead);
    runtime.FlushBarriers();
    if (renodx) {
        snapshot->Transit(command, vk::ImageLayout::eGeneral,
                          vk::PipelineStageFlagBits2::eComputeShader,
                          vk::AccessFlagBits2::eShaderWrite);
        const std::array images{
            vk::DescriptorImageInfo{*nearest, *source_view.image_view,
                                    vk::ImageLayout::eShaderReadOnlyOptimal},
            vk::DescriptorImageInfo{{}, *snapshot->view, vk::ImageLayout::eGeneral}};
        const std::array writes{
            vk::WriteDescriptorSet{.dstBinding = 0,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                   .pImageInfo = &images[0]},
            vk::WriteDescriptorSet{.dstBinding = 1,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eStorageImage,
                                   .pImageInfo = &images[1]}};
        const u32 copy_flags = 2; // plain copy
        command.bindPipeline(vk::PipelineBindPoint::eCompute, *linearize_pass->pipeline);
        command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *linearize_pass->layout, 0,
                                     writes);
        command.pushConstants(*linearize_pass->layout, vk::ShaderStageFlagBits::eCompute, 0,
                              sizeof(copy_flags), &copy_flags);
        command.dispatch((in.width + 7) / 8, (in.height + 7) / 8, 1);
    } else {
        snapshot->Transit(command, vk::ImageLayout::eTransferDstOptimal,
                          vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite);
        const vk::ImageCopy region{
            .srcSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 1},
            .dstSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 1},
            .extent = {in.width, in.height, 1}};
        command.copyImage(source.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                          snapshot->image.image, vk::ImageLayout::eTransferDstOptimal, region);
    }
    snapshot->Transit(command, vk::ImageLayout::eShaderReadOnlyOptimal,
                      vk::PipelineStageFlagBits2::eComputeShader |
                          vk::PipelineStageFlagBits2::eFragmentShader,
                      vk::AccessFlagBits2::eShaderRead);

    // Motion: camera reprojection, replaced by object velocity where the mirror covers.
    const auto mirror_frame = mirror.ConsumeFrame(command);
    mirror.SetTargetSize(in);
    const bool use_mirror =
        tune.mirror && velocity_constants && mirror_frame && mirror_frame->extent == in;
    const vk::BufferMemoryBarrier2 constants_barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = coefficients->Handle(),
        .offset = 0,
        .size = CoefficientBytes};
    command.pipelineBarrier2(vk::DependencyInfo{.bufferMemoryBarrierCount = 1,
                                                .pBufferMemoryBarriers = &constants_barrier});
    motion->Transit(command, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eComputeShader,
                    vk::AccessFlagBits2::eShaderWrite);
    const bool jittered = tune.jitter && jittered_draws > 0;
    const std::array<float, 2> applied = jittered ? jitter : std::array<float, 2>{};
    // Character motion: the skinned meshes' own vectors, written with the G-buffer.
    bool character_valid = false;
    vk::ImageView character_view{};
    if (object_motion && object_motion->Enabled())
        character_view = object_motion->PrepareRead(command, in.width, in.height, character_valid);
    if (character_valid)
        ++character_motion_frames;
    {
        const std::array images{
            vk::DescriptorImageInfo{*nearest, *depth_view.image_view,
                                    vk::ImageLayout::eShaderReadOnlyOptimal},
            vk::DescriptorImageInfo{*nearest, use_mirror ? mirror_frame->view : *snapshot->view,
                                    vk::ImageLayout::eShaderReadOnlyOptimal},
            vk::DescriptorImageInfo{{}, *motion->view, vk::ImageLayout::eGeneral},
            vk::DescriptorImageInfo{*nearest, character_view ? character_view : *snapshot->view,
                                    character_view ? vk::ImageLayout::eGeneral
                                                   : vk::ImageLayout::eShaderReadOnlyOptimal}};
        const vk::DescriptorBufferInfo buffer{coefficients->Handle(), 0, CoefficientBytes};
        const std::array writes{
            vk::WriteDescriptorSet{.dstBinding = 0,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                   .pImageInfo = &images[0]},
            vk::WriteDescriptorSet{.dstBinding = 1,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                   .pImageInfo = &images[1]},
            vk::WriteDescriptorSet{.dstBinding = 2,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eStorageImage,
                                   .pImageInfo = &images[2]},
            vk::WriteDescriptorSet{.dstBinding = 3,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eStorageBuffer,
                                   .pBufferInfo = &buffer},
            vk::WriteDescriptorSet{.dstBinding = 4,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                   .pImageInfo = &images[3]}};
        struct {
            float reproject[12];
            float proj[4], prev_proj[4];
            float jitter[2];
            u32 use_mirror, flags;
        } push{};
        if (tune.camera_majority && frame_has_camera && frame_cameras.size() > 1) {
            const auto most = std::ranges::max_element(frame_cameras, {}, &FrameCamera::uses);
            camera = most->camera;
        }
        // Crossing into another map section moves the coordinates by ~200 units in one frame:
        // that frame takes the game's matrix instead.
        const bool section_switch =
            std::abs(camera.inv_view[3] - previous_camera.inv_view[3]) +
                std::abs(camera.inv_view[7] - previous_camera.inv_view[7]) +
                std::abs(camera.inv_view[11] - previous_camera.inv_view[11]) >
            20.0;
        const bool scene_camera = tune.scene_camera && frame_has_camera && camera.valid &&
                                  previous_camera.valid && !section_switch;
        if (scene_camera) {
            // previous view * current inverse view, both affine 3x4, in double precision. The
            // inverse is built here from the view: the game's own is float32 and off by a
            // fraction of a pixel far from the map origin (the world shook after stopping).
            const auto& a = previous_camera.view;
            const auto b = InverseAffine(camera.view);
            for (int r = 0; r < 3; ++r)
                for (int col = 0; col < 4; ++col) {
                    double value = a[r * 4 + 0] * b[0 * 4 + col] + a[r * 4 + 1] * b[1 * 4 + col] +
                                   a[r * 4 + 2] * b[2 * 4 + col];
                    if (col == 3)
                        value += a[r * 4 + 3];
                    push.reproject[r * 4 + col] = float(value);
                }
            std::copy(camera.proj.begin(), camera.proj.end(), push.proj);
            std::copy(previous_camera.proj.begin(), previous_camera.proj.end(), push.prev_proj);
            ++scene_camera_frames;
        }
        push.jitter[0] = applied[0];
        push.jitter[1] = applied[1];
        push.use_mirror = use_mirror ? 1u : 0u;
        push.flags =
            (tune.camera_snap ? 1u : 0u) | (scene_camera ? 2u : 0u) | (character_valid ? 4u : 0u);
        command.bindPipeline(vk::PipelineBindPoint::eCompute, *motion_pass->pipeline);
        command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *motion_pass->layout, 0,
                                     writes);
        command.pushConstants(*motion_pass->layout, vk::ShaderStageFlagBits::eCompute, 0,
                              sizeof(push), &push);
        command.dispatch((in.width + 7) / 8, (in.height + 7) / 8, 1);
    }
    runtime.AccessBuffer(coefficients.get(), 0, CoefficientBytes,
                         vk::PipelineStageFlagBits2::eComputeShader,
                         vk::AccessFlagBits2::eShaderRead);
    motion->Transit(command, vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::PipelineStageFlagBits2::eComputeShader |
                        vk::PipelineStageFlagBits2::eFragmentShader,
                    vk::AccessFlagBits2::eShaderRead);
    upscaled->Transit(command, vk::ImageLayout::eGeneral,
                      vk::PipelineStageFlagBits2::eComputeShader,
                      vk::AccessFlagBits2::eShaderWrite);

    // FSR 4 treats its colour as linear light and has no flag for display-encoded input, so
    // it gets the scene decoded to linear; the composite encodes its output again.
    // With RenoDX loaded DLSS outputs black for the 8-bit display-encoded scene (it works with
    // the 16-bit HDR scene of the Decoupled UI patches), so it gets the scene the same way.
    // With RenoDX every upscaler gets it, reversibly tone mapped so highlights above 1 survive.
    const bool linear = (backend == Backend::Fsr4 && !hdr && tune.fsr4_linear) || renodx;
    if (linear) {
        linear_color->Transit(command, vk::ImageLayout::eGeneral,
                              vk::PipelineStageFlagBits2::eComputeShader,
                              vk::AccessFlagBits2::eShaderWrite);
        const std::array images{
            vk::DescriptorImageInfo{*nearest, *snapshot->view,
                                    vk::ImageLayout::eShaderReadOnlyOptimal},
            vk::DescriptorImageInfo{{}, *linear_color->view, vk::ImageLayout::eGeneral}};
        const std::array writes{
            vk::WriteDescriptorSet{.dstBinding = 0,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                   .pImageInfo = &images[0]},
            vk::WriteDescriptorSet{.dstBinding = 1,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eStorageImage,
                                   .pImageInfo = &images[1]}};
        const u32 linear_flags = renodx ? 1u : 0u;
        command.bindPipeline(vk::PipelineBindPoint::eCompute, *linearize_pass->pipeline);
        command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *linearize_pass->layout, 0,
                                     writes);
        command.pushConstants(*linearize_pass->layout, vk::ShaderStageFlagBits::eCompute, 0,
                              sizeof(linear_flags), &linear_flags);
        command.dispatch((in.width + 7) / 8, (in.height + 7) / 8, 1);
        linear_color->Transit(command, vk::ImageLayout::eShaderReadOnlyOptimal,
                              vk::PipelineStageFlagBits2::eComputeShader,
                              vk::AccessFlagBits2::eShaderRead);
    }
    if (linear != upscaled_linear || renodx != upscaled_tone_mapped)
        history_valid = false; // the history is in the other encoding
    upscaled_linear = linear;
    upscaled_tone_mapped = renodx;
    const DlssNgx::Resource depth_resource{depth->backing->image.image, *depth_view.image_view,
                                           Range, vk::Format::eR32Sfloat, in};
    const bool reset = !history_valid;
    const DlssNgx::EvalDesc eval{(tune.swap ? applied[1] : applied[0]) * tune.sign_x,
                                 (tune.swap ? applied[0] : applied[1]) * tune.sign_y, reset,
                                 frame_ms};
    const bool success =
        backend == Backend::Dlss
            ? ngx->Evaluate(command, (linear ? linear_color : snapshot)->Resource(), depth_resource,
                            motion->Resource(), upscaled->Resource(), eval)
        : backend == Backend::Fsr4
            ? fsr4->Evaluate(command, (linear ? linear_color : snapshot)->Resource(),
                             depth_resource, motion->Resource(), upscaled->Resource(), eval,
                             FsrCamera(), hdr ? (tune.auto_exposure ? 1 : 2) : 0)
            : fsr->Evaluate(command, (linear ? linear_color : snapshot)->Resource(), depth_resource,
                            motion->Resource(), upscaled->Resource(), eval, FsrCamera());
    scheduler.GetDynamicState().Invalidate(); // the upscaler records its own Vulkan state
    if (!success) {
        (upscale_to ? decoupled_unsupported : failed) = true;
        LOG_ERROR(Render_Vulkan, "[DLSS-TEMPORAL] Evaluation failed; stock rendering");
        return false;
    }
    if (upscale_to && tune.debug_motion) {
        // The composite (and its motion view) does not run when the upscaled image replaces
        // the game's own upscale: draw the motion view over the upscaled image instead.
        motion->Transit(command, vk::ImageLayout::eShaderReadOnlyOptimal,
                        vk::PipelineStageFlagBits2::eComputeShader,
                        vk::AccessFlagBits2::eShaderRead);
        upscaled->Transit(command, vk::ImageLayout::eGeneral,
                          vk::PipelineStageFlagBits2::eComputeShader,
                          vk::AccessFlagBits2::eShaderWrite);
        const std::array images{
            vk::DescriptorImageInfo{*nearest, *motion->view,
                                    vk::ImageLayout::eShaderReadOnlyOptimal},
            vk::DescriptorImageInfo{*nearest, character_view ? character_view : *motion->view,
                                    character_view ? vk::ImageLayout::eGeneral
                                                   : vk::ImageLayout::eShaderReadOnlyOptimal},
            vk::DescriptorImageInfo{{}, *upscaled->view, vk::ImageLayout::eGeneral}};
        const std::array writes{
            vk::WriteDescriptorSet{.dstBinding = 0,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                   .pImageInfo = &images[0]},
            vk::WriteDescriptorSet{.dstBinding = 1,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                   .pImageInfo = &images[1]},
            vk::WriteDescriptorSet{.dstBinding = 2,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eStorageImage,
                                   .pImageInfo = &images[2]}};
        const u32 debug_character = character_valid ? 1u : 0u;
        command.bindPipeline(vk::PipelineBindPoint::eCompute, *motion_debug_pass->pipeline);
        command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *motion_debug_pass->layout, 0,
                                     writes);
        command.pushConstants(*motion_debug_pass->layout, vk::ShaderStageFlagBits::eCompute, 0,
                              sizeof(debug_character), &debug_character);
        command.dispatch((out.width + 7) / 8, (out.height + 7) / 8, 1);
    }
    if (upscale_to && tune.sharpness > 0 && sharpened) {
        // Sharpened into its own image: the upscaler keeps the unsharpened one as history.
        upscaled->Transit(command, vk::ImageLayout::eShaderReadOnlyOptimal,
                          vk::PipelineStageFlagBits2::eComputeShader,
                          vk::AccessFlagBits2::eShaderRead);
        sharpened->Transit(command, vk::ImageLayout::eGeneral,
                           vk::PipelineStageFlagBits2::eComputeShader,
                           vk::AccessFlagBits2::eShaderWrite);
        const std::array images{
            vk::DescriptorImageInfo{*nearest, *upscaled->view,
                                    vk::ImageLayout::eShaderReadOnlyOptimal},
            vk::DescriptorImageInfo{{}, *sharpened->view, vk::ImageLayout::eGeneral}};
        const std::array writes{
            vk::WriteDescriptorSet{.dstBinding = 0,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                   .pImageInfo = &images[0]},
            vk::WriteDescriptorSet{.dstBinding = 1,
                                   .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eStorageImage,
                                   .pImageInfo = &images[1]}};
        command.bindPipeline(vk::PipelineBindPoint::eCompute, *sharpen_pass->pipeline);
        command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *sharpen_pass->layout, 0,
                                     writes);
        command.pushConstants(*sharpen_pass->layout, vk::ShaderStageFlagBits::eCompute, 0,
                              sizeof(tune.sharpness), &tune.sharpness);
        command.dispatch((out.width + 7) / 8, (out.height + 7) / 8, 1);
        sharpened->Transit(command, vk::ImageLayout::eTransferSrcOptimal,
                           vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead);
        replacement = DrawReplacement{.owned = sharpened->image.image};
        decoupled_frame = true;
    } else if (upscale_to) {
        // Replaces the game's own upscale; the game then post-processes and draws its UI.
        upscaled->Transit(command, vk::ImageLayout::eTransferSrcOptimal,
                          vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead);
        replacement = DrawReplacement{.owned = upscaled->image.image};
        decoupled_frame = true;
    } else {
        upscaled->Transit(command, vk::ImageLayout::eShaderReadOnlyOptimal,
                          vk::PipelineStageFlagBits2::eComputeShader,
                          vk::AccessFlagBits2::eShaderRead);
        dlss_ready = true;
    }
    CaptureFrameGen(instance, runtime, scheduler, command, *depth, in, eval);
    dlss_pre_hud = pre_hud;
    evaluated_since_copy = true;
    history_valid = true;
    {
        std::scoped_lock lock{status_mutex};
        last_evaluation = std::chrono::steady_clock::now();
        status.render_width = in.width;
        status.render_height = in.height;
        status.output_width = out.width;
        status.output_height = out.height;
        status.backend = BackendName(backend);
        status.decoupled = upscale_to.has_value();
        status.reason.clear();
    }
    {
        // Quarter steps, so a changing render size doesn't make a new sampler set every frame.
        float bias = 0.0f;
        if (tune.texture_lod_bias && in.width < out.width)
            bias = std::round((std::log2(float(in.width) / float(out.width)) - 1.0f) * 4.0f) / 4.0f;
        scene_mip_bias.store(bias, std::memory_order_relaxed);
        scene_mip_bias_ms.store(SteadyMs(), std::memory_order_relaxed);
    }
    if (++evaluations <= 3 || evaluations % 1800 == 0)
        LOG_INFO(Render_Vulkan,
                 "[DLSS-TEMPORAL] {} evaluations={} {}x{}->{}x{} preHUD={} "
                 "jitter=({:.3f},{:.3f}) "
                 "jitteredDraws={} (before display copy {}) objectMotion={} reset={} fallbacks={} "
                 "sceneCameraFrames={} characterMotion={} (frames with it {})",
                 BackendName(backend), evaluations, in.width, in.height, out.width, out.height,
                 pre_hud, applied[0], applied[1], jittered_draws, jittered_before_copy, use_mirror,
                 reset, fallbacks, scene_camera_frames, character_valid, character_motion_frames);
    return true;
}

void BbTemporalDlss::Impl::Composite(const Instance& instance, Runtime& runtime,
                                     Scheduler& scheduler, VideoCore::TextureCache& cache,
                                     VideoCore::Image& target) {
    auto& outputs_slot = outputs[target.info.guest_address];
    auto& source = cache.GetImage(copy_source->id);
    auto& lut = cache.GetImage(copy_lut->id);
    if (source.info.size.width != render.width || source.info.size.height != render.height) {
        if (outputs_slot)
            outputs_slot->fresh = false;
        return;
    }
    scheduler.EndRendering();
    // With RenoDX the game's display copy (RenoDX's HDR version of it) is drawn again at the
    // output size into an image like the game's target; the composite stops before the LUT.
    const bool replay = RenoDxLoaded();
    // RenoDX's display copy writes values above 1: the image it is redrawn into is float.
    const auto output_format =
        replay ? vk::Format::eR16G16B16A16Sfloat : vk::Format::eR8G8B8A8Unorm;
    const auto pre_lut_format =
        replay ? vk::Format::eR16G16B16A16Sfloat : vk::Format::eR8G8B8A8Unorm;
    if (!outputs_slot || outputs_slot->extent != output || outputs_slot->format != output_format ||
        (replay && (!pre_lut || pre_lut->extent != output))) {
        if (outputs_slot || pre_lut)
            scheduler.Finish();
        using U = vk::ImageUsageFlagBits;
        outputs_slot = std::make_unique<OwnedImage>(
            instance, output_format, output,
            replay ? U::eSampled | U::eColorAttachment : U::eSampled | U::eStorage, true);
        if (replay)
            pre_lut = std::make_unique<OwnedImage>(instance, pre_lut_format, output,
                                                   U::eSampled | U::eStorage);
    }
    // FSR frame generation moves whatever differs from its HUD-less image as UI instead of
    // interpolating it, so it gets the frame composited a second time without the HUD.
    const bool hudless = frame_gen_pending && dlss_pre_hud && tune.hud &&
                         std::string_view{FrameGen::BackendName()} == "FSR";
    if (hudless && (!outputs_slot->hudless || outputs_slot->hudless->extent != output ||
                    outputs_slot->hudless->format != output_format ||
                    (replay && (!pre_lut_hudless || pre_lut_hudless->extent != output)))) {
        scheduler.Finish();
        using U = vk::ImageUsageFlagBits;
        outputs_slot->hudless = std::make_unique<OwnedImage>(
            instance, output_format, output,
            replay ? U::eSampled | U::eColorAttachment : U::eSampled | U::eStorage, true);
        if (replay)
            pre_lut_hudless = std::make_unique<OwnedImage>(instance, pre_lut_format, output,
                                                           U::eSampled | U::eStorage);
    }
    // The HUD as its own layer (see bb_hud_layer.comp): its images at the render size.
    const bool hud_layer =
        tune.menu_fix && dlss_pre_hud && tune.hud && !replay && !upscaled_tone_mapped;
    hud_layer_on = hud_layer;
    if (hud_layer && (!hud_layer_image || hud_layer_image->extent != render)) {
        if (hud_layer_image)
            scheduler.Finish();
        using U = vk::ImageUsageFlagBits;
        const auto make = [&] {
            return std::make_unique<OwnedImage>(instance, vk::Format::eR16G16B16A16Sfloat, render,
                                                U::eSampled | U::eStorage);
        };
        for (u32 i = 0; i < 2; ++i) {
            hud_finals[i] = make();
            hud_scenes[i] = make();
        }
        hud_layer_image = make();
        hud_history = false;
    }
    if (hud_layer && !hud_dimming)
        hud_dimming = std::make_unique<VideoCore::Buffer>(instance, 0, DimmingBytes,
                                                          VideoCore::MemoryType::DeviceLocal);
    auto& written = replay ? *pre_lut : *outputs_slot;
    auto* written_hudless = !hudless ? nullptr
                            : replay ? pre_lut_hudless.get()
                                     : outputs_slot->hudless.get();
    const auto command = scheduler.CommandBuffer(); // after Finish(), see RunDlss
    VideoCore::ImageViewInfo source_info{};
    source_info.format = vk::Format::eR8G8B8A8Unorm;
    const auto& source_view = source.FindView(source_info);
    const auto& lut_view = lut.FindView(copy_lut->view);
    runtime.Transit(&source, vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::PipelineStageFlagBits2::eComputeShader |
                        vk::PipelineStageFlagBits2::eFragmentShader,
                    vk::AccessFlagBits2::eShaderRead);
    runtime.Transit(&lut, vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::PipelineStageFlagBits2::eComputeShader |
                        vk::PipelineStageFlagBits2::eFragmentShader,
                    vk::AccessFlagBits2::eShaderRead);
    runtime.FlushBarriers();
    written.Transit(command, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eComputeShader,
                    vk::AccessFlagBits2::eShaderWrite);
    if (written_hudless)
        written_hudless->Transit(command, vk::ImageLayout::eGeneral,
                                 vk::PipelineStageFlagBits2::eComputeShader,
                                 vk::AccessFlagBits2::eShaderWrite);
    const auto ro = vk::ImageLayout::eShaderReadOnlyOptimal;
    if (!coverage)
        coverage = std::make_unique<VideoCore::Buffer>(instance, 0, CoverageSlots * CoverageStride,
                                                       VideoCore::MemoryType::HostCached);
    // The slot written three composites ago holds a finished count by now.
    const u64 slot = (composites % CoverageSlots) * CoverageStride;
    if (composites >= CoverageSlots) {
        u32 changed{};
        std::memcpy(&changed, coverage->mapped_data.data() + slot, sizeof(changed));
        const double share = double(changed) / (double(output.width) * output.height);
        const bool was_menu = menu;
        menu = tune.menu_fix && (menu ? share > 0.25 : share > 0.40);
        if (menu != was_menu)
            LOG_INFO(Render_Vulkan, "[DLSS-TEMPORAL] Menu {} (HUD covers {:.0f}%)",
                     menu ? "open" : "closed", share * 100.0);
    }
    if (!dimming)
        dimming = std::make_unique<VideoCore::Buffer>(instance, 0, DimmingBytes,
                                                      VideoCore::MemoryType::DeviceLocal);
    command.fillBuffer(coverage->Handle(), slot, sizeof(u32), 0);
    command.fillBuffer(dimming->Handle(), 0, DimmingBytes, 0);
    const vk::BufferMemoryBarrier2 coverage_barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = coverage->Handle(),
        .offset = slot,
        .size = sizeof(u32)};
    auto dimming_barrier = coverage_barrier;
    dimming_barrier.buffer = dimming->Handle();
    dimming_barrier.offset = 0;
    dimming_barrier.size = DimmingBytes;
    const std::array clear_barriers{coverage_barrier, dimming_barrier};
    command.pipelineBarrier2(vk::DependencyInfo{.bufferMemoryBarrierCount = 2,
                                                .pBufferMemoryBarriers = clear_barriers.data()});
    // The HUD as its own layer, measured at the render size before the composite.
    if (hud_layer) {
        const vk::Extent2D size{render.width, render.height};
        const u32 last = hud_last, next = hud_last ^ 1;
        for (auto* image : {hud_finals[last].get(), hud_scenes[last].get()})
            image->Transit(command, ro, vk::PipelineStageFlagBits2::eComputeShader,
                           vk::AccessFlagBits2::eShaderRead);
        for (auto* image : {hud_finals[next].get(), hud_scenes[next].get(), hud_layer_image.get()})
            image->Transit(command, vk::ImageLayout::eGeneral,
                           vk::PipelineStageFlagBits2::eComputeShader,
                           vk::AccessFlagBits2::eShaderWrite);
        const std::array layer_images{
            vk::DescriptorImageInfo{*nearest, *source_view.image_view, ro},
            vk::DescriptorImageInfo{*nearest, *snapshot->view, ro},
            vk::DescriptorImageInfo{*nearest, *hud_finals[last]->view, ro},
            vk::DescriptorImageInfo{*nearest, *hud_scenes[last]->view, ro},
            vk::DescriptorImageInfo{{}, *hud_finals[next]->view, vk::ImageLayout::eGeneral},
            vk::DescriptorImageInfo{{}, *hud_scenes[next]->view, vk::ImageLayout::eGeneral},
            vk::DescriptorImageInfo{{}, *hud_layer_image->view, vk::ImageLayout::eGeneral}};
        std::array<vk::WriteDescriptorSet, 8> layer_writes;
        for (u32 i = 0; i < layer_images.size(); ++i)
            layer_writes[i] = {.dstBinding = i,
                               .descriptorCount = 1,
                               .descriptorType = i >= 4 ? vk::DescriptorType::eStorageImage
                                                        : vk::DescriptorType::eCombinedImageSampler,
                               .pImageInfo = &layer_images[i]};
        const vk::DescriptorBufferInfo layer_dimming{hud_dimming->Handle(), 0, DimmingBytes};
        layer_writes[7] = {.dstBinding = 7,
                           .descriptorCount = 1,
                           .descriptorType = vk::DescriptorType::eStorageBuffer,
                           .pBufferInfo = &layer_dimming};
        // First the histogram of the even darkening, then the layer.
        auto histogram_barrier = dimming_barrier;
        histogram_barrier.buffer = hud_dimming->Handle();
        histogram_barrier.dstStageMask = vk::PipelineStageFlagBits2::eTransfer;
        histogram_barrier.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
        command.pipelineBarrier2(vk::DependencyInfo{.bufferMemoryBarrierCount = 1,
                                                    .pBufferMemoryBarriers = &histogram_barrier});
        command.fillBuffer(hud_dimming->Handle(), 0, DimmingBytes, 0);
        histogram_barrier.srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
        histogram_barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
        histogram_barrier.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader;
        histogram_barrier.dstAccessMask =
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
        command.pipelineBarrier2(vk::DependencyInfo{.bufferMemoryBarrierCount = 1,
                                                    .pBufferMemoryBarriers = &histogram_barrier});
        command.bindPipeline(vk::PipelineBindPoint::eCompute, *hud_layer_pass->pipeline);
        command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *hud_layer_pass->layout, 0,
                                     layer_writes);
        const u32 layer_push = hud_history ? 1u : 0u;
        const u32 count_push = layer_push | 4u;
        command.pushConstants(*hud_layer_pass->layout, vk::ShaderStageFlagBits::eCompute, 0,
                              sizeof(count_push), &count_push);
        command.dispatch((size.width + 7) / 8, (size.height + 7) / 8, 1);
        histogram_barrier.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
        histogram_barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
        command.pipelineBarrier2(vk::DependencyInfo{.bufferMemoryBarrierCount = 1,
                                                    .pBufferMemoryBarriers = &histogram_barrier});
        command.pushConstants(*hud_layer_pass->layout, vk::ShaderStageFlagBits::eCompute, 0,
                              sizeof(layer_push), &layer_push);
        command.dispatch((size.width + 7) / 8, (size.height + 7) / 8, 1);
        hud_layer_image->Transit(command, ro, vk::PipelineStageFlagBits2::eComputeShader,
                                 vk::AccessFlagBits2::eShaderRead);
        hud_last = next;
        hud_history = true;
    } else {
        hud_history = false;
    }
    const std::array images{vk::DescriptorImageInfo{*nearest, *upscaled->view, ro},
                            vk::DescriptorImageInfo{*linear, *source_view.image_view, ro},
                            vk::DescriptorImageInfo{*linear, *snapshot->view, ro},
                            vk::DescriptorImageInfo{*linear, *lut_view.image_view, ro},
                            vk::DescriptorImageInfo{{}, *written.view, vk::ImageLayout::eGeneral},
                            vk::DescriptorImageInfo{*linear, *motion->view, ro}};
    const vk::DescriptorImageInfo layer_info{
        *linear, hud_layer ? *hud_layer_image->view : *snapshot->view, ro};
    std::array<vk::WriteDescriptorSet, 9> writes;
    for (u32 i = 0; i < images.size(); ++i)
        writes[i] = {.dstBinding = i,
                     .descriptorCount = 1,
                     .descriptorType = i == 4 ? vk::DescriptorType::eStorageImage
                                              : vk::DescriptorType::eCombinedImageSampler,
                     .pImageInfo = &images[i]};
    const vk::DescriptorBufferInfo coverage_info{coverage->Handle(), slot, sizeof(u32)};
    writes[6] = {.dstBinding = 6,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageBuffer,
                 .pBufferInfo = &coverage_info};
    const vk::DescriptorBufferInfo dimming_info{dimming->Handle(), 0, DimmingBytes};
    writes[7] = {.dstBinding = 7,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageBuffer,
                 .pBufferInfo = &dimming_info};
    writes[8] = {.dstBinding = 8,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                 .pImageInfo = &layer_info};
    const u32 flags = (dlss_pre_hud && tune.hud ? 1u : 0u) | (replay ? 32u : 2u) |
                      (tune.debug_motion ? 4u : 0u) | (upscaled_linear ? 16u : 0u) |
                      (upscaled_tone_mapped ? 128u : 0u) | (hud_layer ? 512u : 0u);
    const auto& pass = replay ? *composite_hdr_pass : *composite_pass;
    command.bindPipeline(vk::PipelineBindPoint::eCompute, *pass.pipeline);
    command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *pass.layout, 0, writes);
    struct {
        u32 flags;
        float sharpness;
    } push{flags, tune.sharpness};
    command.pushConstants(*pass.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
    command.dispatch((output.width + 7) / 8, (output.height + 7) / 8, 1);
    if (written_hudless) {
        // The HUD-less pass reads the dimming histogram the main pass just wrote.
        auto histogram_barrier = dimming_barrier;
        histogram_barrier.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
        histogram_barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
        command.pipelineBarrier2(vk::DependencyInfo{.bufferMemoryBarrierCount = 1,
                                                    .pBufferMemoryBarriers = &histogram_barrier});
        const vk::DescriptorImageInfo hudless_image{
            {}, *written_hudless->view, vk::ImageLayout::eGeneral};
        writes[4].pImageInfo = &hudless_image;
        command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *pass.layout, 0, writes);
        push.flags = (flags & ~1u) | 64u;
        command.pushConstants(*pass.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push),
                              &push);
        command.dispatch((output.width + 7) / 8, (output.height + 7) / 8, 1);
    }
    outputs_slot->hudless_fresh = written_hudless != nullptr;
    if (replay) {
        pre_lut->Transit(command, ro, vk::PipelineStageFlagBits2::eFragmentShader,
                         vk::AccessFlagBits2::eShaderRead);
        outputs_slot->Transit(command, vk::ImageLayout::eColorAttachmentOptimal,
                              vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                              vk::AccessFlagBits2::eColorAttachmentWrite);
        display_copy_replay =
            DisplayCopyReplay{DisplayCopy, *pre_lut->view, *outputs_slot->view, output};
        if (written_hudless) {
            pre_lut_hudless->Transit(command, ro, vk::PipelineStageFlagBits2::eFragmentShader,
                                     vk::AccessFlagBits2::eShaderRead);
            outputs_slot->hudless->Transit(command, vk::ImageLayout::eColorAttachmentOptimal,
                                           vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                           vk::AccessFlagBits2::eColorAttachmentWrite);
            display_copy_replay->hudless_input = *pre_lut_hudless->view;
            display_copy_replay->hudless_output = *outputs_slot->hudless->view;
            outputs_slot->hudless->swapped = true;
            replay_hudless = outputs_slot->hudless.get();
        }
        replay_output = outputs_slot.get();
        outputs_slot->swapped = true;
    } else {
        outputs_slot->Transit(command, ro, vk::PipelineStageFlagBits2::eFragmentShader,
                              vk::AccessFlagBits2::eShaderRead);
        if (written_hudless)
            written_hudless->Transit(command, ro, vk::PipelineStageFlagBits2::eFragmentShader,
                                     vk::AccessFlagBits2::eShaderRead);
    }
    scheduler.GetDynamicState().Invalidate();
    outputs_slot->fresh = true;
    outputs_slot->frame_gen = std::exchange(frame_gen_pending, std::nullopt);
    if (++composites <= 3 || composites % 1800 == 0)
        LOG_INFO(Render_Vulkan, "[DLSS-TEMPORAL] Display composites={} VideoOut={:#x} {}x{}",
                 composites, target.info.guest_address, output.width, output.height);
}

BbTemporalDlss::BbTemporalDlss() : impl{std::make_unique<Impl>()} {}
BbTemporalDlss::~BbTemporalDlss() = default;

bool BbTemporalDlss::Requested() const {
    return impl->requested;
}

bool BbTemporalDlss::CharacterMotionRequested() const {
    return impl->requested && impl->tune.character_motion;
}

bool BbTemporalDlss::ObjectMotionRequested() const {
    return impl->requested && impl->tune.mirror;
}

void BbTemporalDlss::SetObjectMotion(BbObjectMotion* object_motion) {
    impl->object_motion = object_motion;
    if (object_motion)
        object_motion->SetActive(impl->tune.character_motion);
}

void BbTemporalDlss::ObserveTexture(const VideoCore::Image& image, VideoCore::ImageId id,
                                    const VideoCore::ImageViewInfo& view, u64 hash, u32 slot) {
    if (!impl->requested || impl->failed || impl->stopped)
        return;
    if (hash == DepthProducer && slot == 0 && image.info.pixel_format == vk::Format::eR32Sfloat &&
        ++impl->producer_depths) {
        impl->r32_depth = Impl::Tracked{id, image.image_uid};
        impl->producer_depth_address = image.info.guest_address;
        impl->depth_fallback = false;
    } else if (!impl->r32_depth && image.info.pixel_format == vk::Format::eR32Sfloat &&
               image.info.guest_address != 0 &&
               image.info.guest_address == impl->producer_depth_address) {
        impl->r32_depth = Impl::Tracked{id, image.image_uid};
        impl->depth_fallback = true;
    } else if (hash == DisplayCopy && slot == 0)
        impl->copy_source = Impl::Bound{id, view};
    else if (hash == DisplayCopy && slot == 1)
        impl->copy_lut = Impl::Bound{id, view};
    else if ((hash == MotionBlur || hash == MotionBlurAlt) && slot == 0)
        impl->blur_source = Impl::Bound{id, view};
    else if (hash == HdrSceneCopy && slot == 0)
        impl->hdr_copy_source = Impl::Bound{id, view};
}

void BbTemporalDlss::ObserveBlurInputs(const void* data, u64 size, u64 hash, u32 binding) {
    auto& s = *impl;
    if (!s.requested || s.failed || s.stopped || !data)
        return;
    const auto* f = static_cast<const float*>(data);
    if (hash == DepthProducer && binding == 0 && !s.blur.sampled &&
        size >= (0x380 + 16) * sizeof(float)) {
        std::copy_n(f + 0x380, 16, s.blur.matrix.begin());
        s.blur.shift_px = Impl::BlurShiftPx(s.blur.matrix);
        s.blur.sampled = true;
    } else if (hash == OpaqueVelocity && binding == 0 && size >= 2 * sizeof(float)) {
        s.blur.velocity_scale = {f[0], f[1]};
    }
}

void BbTemporalDlss::ObserveSceneConstants(const void* data, u64 size) {
    auto& s = *impl;
    if (!s.requested || s.failed || s.stopped || !data || size < 192 * sizeof(float))
        return;
    const auto* f = static_cast<const float*>(data);
    if (f[0] != 3000.0f || std::abs(f[1] * f[0] - 1.0f) > 1e-3f || f[4] < 64.0f || f[5] < 64.0f)
        return;
    Impl::Camera seen;
    for (int i = 0; i < 12; ++i) {
        seen.view[i] = f[8 + i];
        seen.inv_view[i] = f[180 + i];
    }
    seen.proj = {f[52], f[57], f[62], f[63]};
    seen.valid = f[52] != 0.0f && f[57] != 0.0f && std::isfinite(f[62]) && std::isfinite(f[63]);
    if (!s.frame_has_camera) {
        // The first scene constants after an evaluation belong to the main camera of the next
        // frame.
        s.previous_camera = s.camera;
        s.camera = seen;
        s.frame_has_camera = true;
        s.frame_cameras.clear();
    }
    auto it = std::ranges::find_if(s.frame_cameras, [&](const Impl::FrameCamera& known) {
        return known.camera.view == seen.view && known.camera.proj == seen.proj;
    });
    if (it != s.frame_cameras.end())
        ++it->uses;
    else if (s.frame_cameras.size() < 8)
        s.frame_cameras.push_back({seen, 1});
}

void BbTemporalDlss::ObserveConstants(Runtime& runtime, const Instance& instance,
                                      const VideoCore::Buffer* source, u64 offset, u64 size,
                                      u64 hash, u32 binding) {
    if (!impl->requested || impl->failed || impl->stopped || !source)
        return;
    impl->producer_constants += hash == DepthProducer && binding == 0;
    impl->velocity_draw_constants +=
        (hash == OpaqueVelocity && binding == 0) ||
        ((hash == AlphaVelocityA || hash == AlphaVelocityB) && binding == 1);
    const bool camera = hash == DepthProducer && binding == 0 && !impl->camera_constants;
    const bool velocity = !impl->velocity_constants &&
                          ((hash == OpaqueVelocity && binding == 0) ||
                           ((hash == AlphaVelocityA || hash == AlphaVelocityB) && binding == 1));
    if (!camera && !velocity)
        return;
    const u64 bytes = camera ? CameraBytes : 8;
    if (size < bytes || offset > source->SizeBytes() || bytes > source->SizeBytes() - offset)
        return;
    try {
        if (!impl->coefficients)
            impl->coefficients = std::make_unique<VideoCore::Buffer>(
                instance, 0, CoefficientBytes, VideoCore::MemoryType::DeviceLocal);
        const vk::BufferCopy copy{
            .srcOffset = offset, .dstOffset = camera ? 0 : VelocityOffset, .size = bytes};
        runtime.CopyBuffer(source, impl->coefficients.get(), std::span{&copy, 1});
        (camera ? impl->camera_constants : impl->velocity_constants) = true;
    } catch (const std::exception& e) {
        impl->failed = true;
        LOG_ERROR(Render_Vulkan, "[DLSS-TEMPORAL] Disabled: {}", e.what());
    }
}

std::array<float, 2> BbTemporalDlss::OnDraw(const Instance& instance, Runtime& runtime,
                                            Scheduler& scheduler, VideoCore::TextureCache& cache,
                                            BbVelocityMirror& mirror, const DrawInfo& draw) {
    auto& s = *impl;
    if (!s.requested || s.failed || s.stopped)
        return {};
    auto* color = draw.color ? &cache.GetImage(draw.color) : nullptr;
    try {
        if (draw.ps_hash == MotionBlur || draw.ps_hash == MotionBlurAlt) {
            ++s.blur.blur_draws;
            const auto source = std::exchange(s.blur_source, std::nullopt);
            if (s.tune.motion_blur || !color || !source || source->id == draw.color)
                return {};
            const auto& image = cache.GetImage(source->id);
            if (image.info.size.width != color->info.size.width ||
                image.info.size.height != color->info.size.height ||
                image.info.pixel_format != color->info.pixel_format)
                return {};
            s.replacement = DrawReplacement{.image = source->id};
            if (++s.blur_skips <= 2)
                LOG_INFO(Render_Vulkan, "[DLSS-TEMPORAL] Motion blur pass skipped");
            return {};
        }
        if (draw.ps_hash == HdrSceneCopy && color) {
            const auto source = std::exchange(s.hdr_copy_source, std::nullopt);
            if (!source || source->id == draw.color)
                return {};
            auto& scene = cache.GetImage(source->id);
            const vk::Extent2D target{color->info.size.width, color->info.size.height};
            // Same size: the normal copy before post-processing. Smaller: a Decoupled UI patch,
            // and this draw is the game's upscale of the scene to its UI resolution.
            if (scene.info.size.width == target.width && scene.info.size.height == target.height)
                return {};
            // Upscales of up to 3x only: the variants that render above the UI resolution, or
            // far below it, keep the game's own scaling.
            const bool supported = scene.info.size.width < target.width &&
                                   scene.info.size.height < target.height &&
                                   target.width <= 3 * scene.info.size.width;
            if (!supported || s.decoupled_unsupported) {
                if (!s.decoupled_unsupported)
                    LOG_INFO(Render_Vulkan,
                             "[DLSS-TEMPORAL] Decoupled UI {}x{} -> {}x{} is not upscaled",
                             scene.info.size.width, scene.info.size.height, target.width,
                             target.height);
                s.decoupled_unsupported = true;
                ReportFrame("This Decoupled UI patch is not supported: use one that renders "
                            "between 640x360 and 1600x900, or a regular Resolution Patch.");
                return {};
            }
            if (!s.decoupled)
                LOG_INFO(Render_Vulkan, "[DLSS-TEMPORAL] Decoupled UI: scene {}x{}, UI {}x{}",
                         scene.info.size.width, scene.info.size.height, target.width,
                         target.height);
            s.decoupled = true;
            s.decoupled_scene = source->id;
            if (s.tune.enabled && color->info.pixel_format == vk::Format::eR16G16B16A16Sfloat &&
                scene.info.pixel_format == vk::Format::eR16G16B16A16Sfloat)
                s.RunDlss(instance, runtime, scheduler, cache, mirror, scene, false, target);
            s.FrameBoundary(scheduler, mirror);
            return {};
        }
        if (draw.ps_hash == DisplayCopy && color && s.copy_source && s.copy_lut) {
            // In decoupled-UI mode the scene was upscaled before post-processing already.
            if (s.tune.enabled && !s.dlss_ready && !s.decoupled && !s.decoupled_unsupported) {
                // No Scaleform draw this frame: upscale the finished frame as a whole.
                s.RunDlss(instance, runtime, scheduler, cache, mirror,
                          cache.GetImage(s.copy_source->id), false);
                s.FrameBoundary(scheduler, mirror);
            }
            if (s.tune.enabled && s.dlss_ready && !s.failed)
                s.Composite(instance, runtime, scheduler, cache, *color);
            else if (auto it = s.outputs.find(color->info.guest_address); it != s.outputs.end())
                it->second->fresh = false;
            // Decoupled UI: the game presents its own frame, generated from this frame's inputs.
            auto& hudless = s.decoupled_hudless[color->info.guest_address];
            if (hudless)
                hudless->fresh = false;
            if (auto inputs = std::exchange(s.frame_gen_pending, std::nullopt);
                inputs && s.decoupled && !s.dlss_ready) {
                s.decoupled_frame_gen[color->info.guest_address] = *inputs;
                if (s.hud_snapshot_taken && s.hud_snapshot &&
                    s.hud_snapshot->extent ==
                        vk::Extent2D{color->info.size.width, color->info.size.height})
                    s.ReplayHudless(instance, runtime, scheduler, cache, *color,
                                    draw.color_view_format);
            } else {
                s.decoupled_frame_gen.erase(color->info.guest_address);
            }
            s.last_copy_source = s.copy_source->id;
            s.last_copy_format = s.copy_source->view.format;
            s.last_copy_view = s.copy_source->view;
            s.hud_snapshot_taken = false;
            s.DisplayCopyDone();
            return {};
        }
        if (color && !s.outputs.empty()) {
            // Anything else drawing into a tracked VideoOut buffer makes our output stale.
            if (auto it = s.outputs.find(color->info.guest_address); it != s.outputs.end())
                it->second->fresh = false;
        }
        const auto learn_depth = [&] {
            if (!draw.depth || !color || s.depth_learned)
                return;
            const auto& depth = cache.GetImage(draw.depth);
            if (depth.info.size.width == color->info.size.width &&
                depth.info.size.height == color->info.size.height) {
                s.scene_depth = Impl::Tracked{draw.depth, depth.image_uid};
                s.depth_learned = true;
            }
        };
        // Scaleform also renders movies into offscreen targets; only the scene-sized target
        // (same size as this frame's sampled scene depth) starts the HUD phase.
        const auto scene_sized = [&](const VideoCore::Image& image) {
            const auto* depth =
                s.r32_depth ? cache.TryGetImage(s.r32_depth->id, s.r32_depth->uid) : nullptr;
            return depth && depth->info.size.width == image.info.size.width &&
                   depth->info.size.height == image.info.size.height;
        };
        // Decoupled UI + FSR frame generation: the UI target as it is before the HUD.
        if (s.decoupled && !s.hud_snapshot_taken && s.frame_gen_pending && color &&
            IsUiShader(draw.vs_hash) && s.last_copy_source && draw.color == *s.last_copy_source &&
            std::string_view{FrameGen::BackendName()} == "FSR")
            s.SnapshotHud(instance, runtime, scheduler, *color);
        if (!s.ui_phase && IsUiShader(draw.vs_hash) && color &&
            (color->info.pixel_format == vk::Format::eR8G8B8A8Unorm ||
             color->info.pixel_format == vk::Format::eR8G8B8A8Srgb) &&
            scene_sized(*color)) {
            s.ui_phase = true;
            s.ui_target = draw.color;
            learn_depth();
            if (s.tune.enabled && !s.dlss_ready)
                s.RunDlss(instance, runtime, scheduler, cache, mirror, *color, true);
            s.FrameBoundary(scheduler, mirror);
            return {};
        }
        if (IsUiShader(draw.vs_hash)) {
            if (s.ui_phase)
                learn_depth();
            return {};
        }
        // Decoupled UI: the depth used while drawing the HDR scene is the scene depth.
        if (s.decoupled && s.decoupled_scene && color && draw.color == *s.decoupled_scene &&
            draw.depth) {
            const auto& depth = cache.GetImage(draw.depth);
            if (depth.info.size.width == color->info.size.width &&
                depth.info.size.height == color->info.size.height &&
                (!s.scene_depth || s.scene_depth->id != draw.depth))
                s.scene_depth = Impl::Tracked{draw.depth, depth.image_uid};
        }
        // HUD draws share the scene depth for stencil; anything into the HUD target stays put.
        if (s.ui_target && draw.color == *s.ui_target)
            return {};
        // The HUD's stencil masks (no colour, no depth test) share the scene depth too: a
        // jittered mask moves the edge of what it masks (the stamina bar's end line).
        if (s.ui_phase && !color && draw.stencil && !draw.depth_test)
            return {};
        // In menus the jitter stops where the HUD layer can't be drawn; the HUD layer measures
        // the HUD against the moving scene, so there it keeps running.
        if (s.tune.enabled && s.tune.jitter && !(s.menu && !s.hud_layer_on) && s.scene_depth &&
            draw.depth == s.scene_depth->id &&
            cache.GetImage(draw.depth).image_uid == s.scene_depth->uid) {
            const bool small = !draw.indirect && draw.num_indices <= 6 && draw.num_instances <= 1;
            if (!small) {
                ++s.jittered_draws;
                return s.jitter;
            }
            ++s.small_draws;
            if (draw.depth_test && draw.depth_write) {
                ++s.small_depth_draws;
                if (s.tune.jitter_small) {
                    ++s.jittered_draws;
                    return s.jitter;
                }
            }
        }
        return {};
    } catch (const std::exception& e) {
        s.failed = true;
        s.InvalidateOutputs();
        scheduler.GetDynamicState().Invalidate();
        LOG_ERROR(Render_Vulkan, "[DLSS-TEMPORAL] Disabled: {}", e.what());
        return {};
    }
}

std::optional<BbTemporalDlss::DrawReplacement> BbTemporalDlss::TakeDrawReplacement() {
    return std::exchange(impl->replacement, std::nullopt);
}

std::optional<BbTemporalDlss::DisplayCopyReplay> BbTemporalDlss::TakeDisplayCopyReplay() {
    return std::exchange(impl->display_copy_replay, std::nullopt);
}

void BbTemporalDlss::FinishDisplayCopyReplay(vk::CommandBuffer command, bool drawn) {
    if (auto* hudless = std::exchange(impl->replay_hudless, nullptr)) {
        hudless->Transit(command, vk::ImageLayout::eShaderReadOnlyOptimal,
                         vk::PipelineStageFlagBits2::eFragmentShader,
                         vk::AccessFlagBits2::eShaderRead);
        if (!drawn)
            hudless->fresh = false;
    }
    auto* output = std::exchange(impl->replay_output, nullptr);
    if (!output)
        return;
    output->Transit(command, vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderRead);
    if (!drawn) {
        output->fresh = false;
        static bool logged{};
        if (!std::exchange(logged, true))
            LOG_WARNING(Render_Vulkan, "[DLSS-TEMPORAL] RenoDX display copy could not be redrawn");
    }
}

void BbTemporalDlss::SetDisplaySize(u32 width, u32 height) {
    impl->display = (u64{width} << 32) | height;
}

std::optional<BbTemporalDlss::Presentation> BbTemporalDlss::TakePresentation(
    VAddr address, vk::Format frame_view_format) {
    auto& s = *impl;
    if (!s.requested || s.failed || s.stopped || !s.tune.enabled)
        return {};
    const auto it = s.outputs.find(address);
    if (it == s.outputs.end() || !it->second->fresh)
        return {};
    const bool bgr = (frame_view_format == vk::Format::eB8G8R8A8Srgb) != it->second->swapped;
    if (frame_view_format != vk::Format::eB8G8R8A8Srgb &&
        frame_view_format != vk::Format::eR8G8B8A8Srgb)
        return {};
    Presentation presentation{bgr ? *it->second->bgr_view : *it->second->rgb_view,
                              it->second->extent, it->second->frame_gen};
    if (const auto& hudless = it->second->hudless; hudless && it->second->hudless_fresh)
        presentation.hudless = bgr ? *hudless->bgr_view : *hudless->rgb_view;
    return presentation;
}

std::optional<FrameGen::FrameInputs> BbTemporalDlss::TakeFrameGen(VAddr address,
                                                                  vk::Format frame_view_format,
                                                                  vk::ImageView* hudless) {
    auto& s = *impl;
    *hudless = vk::ImageView{};
    const auto it = s.decoupled_frame_gen.find(address);
    if (it == s.decoupled_frame_gen.end())
        return {};
    const auto inputs = it->second;
    s.decoupled_frame_gen.erase(it);
    if (!s.requested || s.failed || s.stopped || !s.tune.enabled)
        return {};
    if (const auto slot = s.decoupled_hudless.find(address);
        slot != s.decoupled_hudless.end() && slot->second && slot->second->fresh) {
        slot->second->fresh = false;
        const auto& image = *slot->second;
        if (image.rgb_view) {
            const bool bgr = (frame_view_format == vk::Format::eB8G8R8A8Srgb) != image.swapped;
            *hudless = bgr ? *image.bgr_view : *image.rgb_view;
        } else {
            *hudless = slot->second->FrameView(frame_view_format);
        }
    }
    return inputs;
}

void BbTemporalDlss::Shutdown(const Instance& instance, Scheduler& scheduler) {
    auto& s = *impl;
    if (!s.requested || s.stopped)
        return;
    s.stopped = true;
    scheduler.EndRendering();
    scheduler.Finish();
    s.fsr.reset();
    s.fsr4.reset();
    if (auto* ngx = instance.GetDlssNgx()) {
        ngx->ReleaseFeatureAfterGpuDrain();
        ngx->Shutdown();
    }
    s.outputs.clear();
    s.snapshot.reset();
    s.motion.reset();
    s.upscaled.reset();
    s.pre_lut.reset();
    s.pre_lut_hudless.reset();
    s.hud_snapshot.reset();
    s.decoupled_hudless.clear();
    s.sharpened.reset();
    s.linear_color.reset();
    s.coefficients.reset();
    s.coverage.reset();
    s.dimming.reset();
    s.motion_pass.reset();
    s.composite_pass.reset();
    s.composite_hdr_pass.reset();
    s.linearize_pass.reset();
    s.sharpen_pass.reset();
    s.motion_debug_pass.reset();
    s.hudless_dim_pass.reset();
    LOG_INFO(Render_Vulkan, "[DLSS-TEMPORAL] Teardown: evaluations={} composites={} fallbacks={}",
             s.evaluations, s.composites, s.fallbacks);
}

} // namespace Vulkan
