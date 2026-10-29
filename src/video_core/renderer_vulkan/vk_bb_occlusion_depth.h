// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <memory>
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore {
struct Image;
}

namespace Vulkan {
class Instance;
class Runtime;
class Scheduler;

/// Occlusion tests tolerant of a pixel: a copy of the scene's depth where each pixel holds the
/// farthest depth around it, for the game's occlusion test boxes. A box that shows through a gap
/// of about a pixel then counts a few pixels every frame instead of 0 in some (the camera's
/// sub-pixel jitter for the upscaler, and sub-pixel movement, moved such gaps on and off pixel
/// centres; a 0 hid the area behind for a frame).
class BbOcclusionDepth {
public:
    BbOcclusionDepth(const Instance& instance, Runtime& runtime, Scheduler& scheduler);
    ~BbOcclusionDepth();

    /// Draw recorder, outside a render pass: makes the copy of `depth` (single sampled, in its
    /// tracked layout; left in DEPTH_STENCIL_READ_ONLY) and returns its view for a test pass, in
    /// DEPTH_STENCIL_ATTACHMENT_OPTIMAL. `farthest_is_max`: the tests pass for smaller depths.
    /// Null when it can't (the pass then uses the game's depth).
    vk::ImageView Build(VideoCore::Image& depth, u32 width, u32 height, bool farthest_is_max);

    /// Whether the format has stencil (the test pass then binds it too).
    [[nodiscard]] static bool HasStencil(vk::Format format);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Vulkan
