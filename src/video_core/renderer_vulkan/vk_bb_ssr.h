// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <memory>
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/types.h"

namespace VideoCore {
class TextureCache;
}

namespace Vulkan {
class Instance;
class Runtime;
class Scheduler;

/// Better screen-space reflections (the game's, turned on by its "Enable Screen Space
/// Reflections" patch). The game traces its reflections at half the scene's size (pixel shader
/// 6c2323c2: depth, G-buffer normals and a half-size copy of the last frame's colour in, an
/// RGBA16F target out), makes that target's mips, and its lighting (pixel shader 3fe9e964) reads
/// them at a mip chosen by each surface's gloss. That draw is left out and the target written
/// with rays traced through a depth pyramid with the game's own constants, then denoised (AMD
/// FidelityFX's reflection denoiser, bb_ssr_denoise.comp).
/// At full resolution (the setting), the rays are traced at the scene's size from a full-size
/// copy of the last frame (taken at the game's own half-size copy, pixel shader 86e96401), blurred
/// by gloss there, and the lighting reads that full-size result instead of the game's target
/// (which still gets it, halved, for the other passes that read it).
class BbSsr {
public:
    static constexpr u64 TraceShader = 0x6c2323c2;
    /// The game's lighting pass, and which of its textures are the gloss and the reflections.
    static constexpr u64 LightingShader = 0x3fe9e964;
    static constexpr u32 LightingGlossSlot = 0;
    static constexpr u32 LightingReflectionSlot = 6;
    /// The game's half-size copy of the scene's colour (a plain copy shader, used elsewhere too).
    static constexpr u64 ColorCopyShader = 0x86e96401;

    /// The game's constants of the trace, as bound to its pixel shader (buffer 0): `offset`
    /// bytes into the binding, which is `size` bytes from `binding_offset` in `buffer`.
    struct Constants {
        vk::Buffer buffer{};
        u64 binding_offset{};
        u64 size{};
        u32 offset{};
    };

    BbSsr(const Instance& instance, Scheduler& scheduler, Runtime& runtime,
          VideoCore::TextureCache& texture_cache);
    ~BbSsr();

    /// At the game's trace draw (before it is recorded): writes `target` (drawn as
    /// `view_format`) from the game's `depth`, `normals` and previous frame's `color` (its pixel
    /// shader's textures 0 to 2) and `constants`. Whether it did (the game's draw is then left
    /// out).
    [[nodiscard]] bool Run(VideoCore::ImageId target, vk::Format view_format,
                           VideoCore::ImageId depth, VideoCore::ImageId normals,
                           VideoCore::ImageId color, const Constants& constants,
                           const void* guest_constants);

    /// A G-buffer draw into `depth`: the largest of a frame is the scene's.
    void OnGBufferDraw(VideoCore::ImageId depth);

    /// After a draw of the copy shader from `source` into `target`: when `target` is the colour
    /// the trace reads, `source` is the full-size scene, copied for the next frame's rays.
    void OnColorCopy(VideoCore::ImageId source, VideoCore::ImageId target);

    /// After the lighting draw, which read `gloss`.
    void OnLighting(VideoCore::ImageId gloss);

    /// Reflections off, at the game's trace draw: `target` (drawn as `view_format`) cleared
    /// instead (no reflection anywhere). Whether it was (the game's draw is then left out).
    [[nodiscard]] bool Clear(VideoCore::ImageId target, vk::Format view_format);

    /// The view the lighting reads for its reflections this frame instead of the game's
    /// target, or null.
    [[nodiscard]] vk::ImageView LightingReflections() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Vulkan
