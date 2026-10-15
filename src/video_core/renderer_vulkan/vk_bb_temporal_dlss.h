// SPDX-FileCopyrightText: Copyright 2026 IFreemz
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_bb_frame_gen.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/types.h"

namespace VideoCore {
struct Buffer;
struct Image;
class TextureCache;
} // namespace VideoCore
namespace Vulkan {
class Instance;
class Runtime;
class Scheduler;
class BbVelocityMirror;
class BbObjectMotion;

// Temporal upscaling for Bloodborne with DLSS (shadps4_dlss.dll + nvngx_dlss.dll) or FSR 3.1
// (amd_fidelityfx_vk.dll), active when either is present next to the executable. Scene
// draws are jittered, the pre-HUD scene is upscaled with camera/object motion at the first
// Scaleform draw, and the game's display copy is re-done at output resolution with the HUD delta
// and the game's gamma LUT. Any missing input falls back to the stock presentation path for that
// frame.
bool BbTemporalDlssRequested();
// user/dlss.ini, shared by the renderer and the F1 panel.
std::filesystem::path BbDlssSettingsPath();

// Thread-safe status for the in-game panel.
struct BbDlssStatus {
    bool active{};
    std::string reason;       // why upscaling is not active, in plain words
    std::string backend;      // upscaler of the latest evaluation: DLSS, FSR 4 or FSR 3.1
    std::string dlss_problem; // why DLSS cannot run on this system, if it cannot
    std::string gpu;
    bool fsr4_installed{};
    bool decoupled{}; // the Decoupled UI patch: the scene is upscaled to the game's UI size
    u32 render_width{}, render_height{}, output_width{}, output_height{};
};
BbDlssStatus BbTemporalDlssStatus();
// Extra texture LOD bias for the scene's material textures while the upscaler runs (NVIDIA's
// DLSS guide, AMD's FSR guide: log2(render / output) - 1), 0 otherwise or when turned off in F1.
float BbSceneMipBias();
void BbTemporalDlssReportGpu(std::string_view gpu);
void BbTemporalDlssReportProblem(std::string_view problem);

class BbTemporalDlss {
public:
    BbTemporalDlss();
    ~BbTemporalDlss();
    bool Requested() const;
    // dlss.ini character_motion at startup: G-buffer pipelines get the motion attachment.
    bool CharacterMotionRequested() const;
    void SetObjectMotion(BbObjectMotion* object_motion);

    struct DrawInfo {
        u64 vs_hash;
        u64 ps_hash;
        VideoCore::ImageId color;
        VideoCore::ImageId depth;
        u32 num_indices;
        u32 num_instances;
        bool indirect;
        bool depth_test, depth_write;
    };
    // Called after resources are bound and before the guest render pass begins. Returns the
    // viewport offset (render pixels) for this draw.
    std::array<float, 2> OnDraw(const Instance& instance, Runtime& runtime, Scheduler& scheduler,
                                VideoCore::TextureCache& cache, BbVelocityMirror& mirror,
                                const DrawInfo& draw);
    // An image whose copy replaces the draw that was just passed to OnDraw (motion blur turned
    // off, or the upscaled scene in decoupled-UI mode). Size and format match the draw's color
    // target.
    struct DrawReplacement {
        std::optional<VideoCore::ImageId> image; // a texture cache image, or
        vk::Image owned;                         // an image left in TRANSFER_SRC_OPTIMAL
    };
    std::optional<DrawReplacement> TakeDrawReplacement();
    // With RenoDX, the display copy that was just passed to OnDraw is drawn a second time at the
    // output size: `input` (the upscaled frame before the display LUT) in place of the texture
    // the shader `shader_hash` reads at slot 0, into `output`. RenoDX's replacement of that
    // shader then gives the upscaled frame its HDR encoding.
    struct DisplayCopyReplay {
        u64 shader_hash;
        vk::ImageView input;
        vk::ImageView output;
        vk::Extent2D extent;
        // FSR frame generation: drawn a second time from the frame without the HUD (or null).
        vk::ImageView hudless_input, hudless_output;
    };
    std::optional<DisplayCopyReplay> TakeDisplayCopyReplay();
    // After the replay was recorded (or could not be), on the scheduler's command buffer.
    void FinishDisplayCopyReplay(vk::CommandBuffer command, bool drawn);
    void ObserveTexture(const VideoCore::Image& image, VideoCore::ImageId id,
                        const VideoCore::ImageViewInfo& view, u64 shader_hash, u32 slot);
    // CPU-visible bytes of any read-only constant binding; finds the game's scene constants.
    void ObserveSceneConstants(const void* data, u64 size);
    void ObserveConstants(Runtime& runtime, const Instance& instance,
                          const VideoCore::Buffer* source, u64 offset, u64 size, u64 shader_hash,
                          u32 binding);

    struct Presentation {
        vk::ImageView view;
        vk::Extent2D extent;
        // Frame generation inputs for this frame, when frame generation is on.
        std::optional<FrameGen::FrameInputs> frame_gen;
        // FSR frame generation: the same frame without the game's HUD (or null).
        vk::ImageView hudless;
    };
    // Output for the VideoOut buffer at this address, if the latest display copy into it was
    // produced by this pass. The view yields sRGB-encoded values in the presenter's order.
    std::optional<Presentation> TakePresentation(VAddr address, vk::Format frame_view_format);
    // Decoupled UI patches: frame generation inputs for the game's own frame in the VideoOut
    // buffer at this address, which the presenter shows instead of an output of this pass.
    // With FSR frame generation `hudless` gets the same frame without the game's HUD, viewed
    // as `frame_view_format` like the game's frame (or null).
    std::optional<FrameGen::FrameInputs> TakeFrameGen(VAddr address, vk::Format frame_view_format,
                                                      vk::ImageView* hudless);
    // The presenter's frame size, which DLSS upscales to unless an output size is configured.
    void SetDisplaySize(u32 width, u32 height);
    // GPU command thread only.
    void Shutdown(const Instance& instance, Scheduler& scheduler);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace Vulkan
