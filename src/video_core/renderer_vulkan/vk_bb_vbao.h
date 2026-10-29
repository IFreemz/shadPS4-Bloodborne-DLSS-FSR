// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <memory>
#include "common/types.h"
#include "video_core/texture_cache/types.h"

namespace VideoCore {
class TextureCache;
}

namespace Vulkan {
class Instance;
class Runtime;
class Scheduler;

/// Better ambient occlusion: GT-VBAO (ground-truth AO with a visibility bitmask, Therrien et al.
/// 2023), after bbport 0.7 (deadinside28/bloodborne_pc). The game keeps its own AO passes; after
/// the last one its AO target is overwritten with VBAO computed from the scene depth, so the
/// game's lighting uses it as its own. The game has two AO paths: a temporal one ending in an
/// upsample into an R8 target at the scene size (pixel shader c45c51cb), and a cheaper one whose
/// half-size RGBA8 target is read directly (pixel shader cbbca5dc; the AO is its red channel).
class BbVbao {
public:
    static constexpr u64 AoUpsampleShader = 0xc45c51cb;
    static constexpr u64 AoHalfShader = 0xcbbca5dc;
    /// Pixel shader of the temporal path's SSAO pass: its texture 1 is the G-buffer's normals.
    static constexpr u64 SsaoShader = 0x83da8467;

    BbVbao(const Instance& instance, Scheduler& scheduler, Runtime& runtime,
           VideoCore::TextureCache& texture_cache);
    ~BbVbao();

    /// The pixel shader of a draw that ends one of the game's AO paths.
    [[nodiscard]] static bool AoShader(u64 ps_hash) noexcept {
        return ps_hash == AoUpsampleShader || ps_hash == AoHalfShader;
    }

    /// A G-buffer draw (5+ color targets) with depth target `depth` and the signs of its
    /// viewport's scales: the scene's depth and camera come from the largest such pass.
    void OnGBufferDraw(VideoCore::ImageId depth, float x_sign, float y_sign);
    /// Not a G-buffer draw (its constants are not the scene camera's).
    void OnOtherDraw() noexcept {
        gbuffer_draw = false;
    }

    /// Constant data bound to a draw: the first scene constants of the G-buffer pass are its
    /// camera.
    void ObserveSceneConstants(const void* data, u64 size);

    /// The temporal path's SSAO pass read `normals` (its texture 1): the G-buffer's normals.
    void OnSsaoPass(VideoCore::ImageId normals);

    /// The draw that ends the game's AO wrote `target`; `normals` is the G-buffer normal texture
    /// it read (the half-size path), or null (then those the SSAO pass read). Overwrites the AO;
    /// `temporal`: a temporal upscaler accumulates frames, so the sample pattern rotates.
    void OnAoPass(VideoCore::ImageId target, VideoCore::ImageId normals, bool temporal);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    bool gbuffer_draw{};
};

} // namespace Vulkan
