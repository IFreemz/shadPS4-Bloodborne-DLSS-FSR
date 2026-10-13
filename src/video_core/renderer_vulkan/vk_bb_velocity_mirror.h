// SPDX-FileCopyrightText: Copyright 2026 IFreemz
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <functional>
#include <memory>
#include <optional>
#include <vector>
#include "common/unique_function.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
namespace VideoCore {
struct Image;
}
namespace Vulkan {
class Instance;
class Scheduler;
class Runtime;
class GraphicsPipeline;
struct RenderState;

// Replays the game's object-velocity draws at render resolution for DLSS motion vectors.
class BbVelocityMirror {
public:
    BbVelocityMirror();
    ~BbVelocityMirror();
    bool Requested() const;
    /// Graphics setting: off skips the replays (the upscaler gets camera motion only).
    void SetEnabled(bool enabled);
    /// The commands of one deferred mirror draw (bindings and draw, recorded later in order).
    using Commands = std::vector<Common::UniqueFunction<void, vk::CommandBuffer>>;
    /// Records a mirror draw's bindings and draw into `out` (scheduler capture), with the
    /// mirror's viewport and scissor.
    using Capture =
        std::function<void(Commands& out, const Viewports& viewports, const Scissors& scissors)>;
    /// Defers this guest draw's replay into the mirror pass of the current batch (one pass per
    /// run of mirrored draws instead of a pass break out of and back into the guest pass for
    /// each). The frame's first mirrored draw seeds the mirror depth, which records directly
    /// and ends the guest pass: `guest_pass_ended` reports it. Returns whether it deferred.
    bool Defer(const Instance& instance, Runtime& runtime, Scheduler& scheduler,
               const GraphicsPipeline& pipeline, const RenderState& guest_state,
               VideoCore::Image* guest_depth, u32 depth_layer, const Capture& capture,
               bool& guest_pass_ended);
    /// Whether deferred draws wait for FlushBatch.
    [[nodiscard]] bool HasBatch() const;
    /// Batches flushed so far (a flush rebinds the pipeline, descriptors and dynamic state).
    [[nodiscard]] u64 Flushes() const;
    /// Records the deferred draws in one mirror pass. Call outside a render pass (the
    /// scheduler's pass-end hook does).
    void FlushBatch(Scheduler& scheduler);
    struct Frame {
        vk::ImageView view;
        vk::Extent2D extent;
    };
    // Temporal DLSS: hands over this frame's mirror (shader-read layout) and starts a new one.
    std::optional<Frame> ConsumeFrame(vk::CommandBuffer command);
    // Render resolution the mirror replays the game's 160x90 velocity draws at.
    void SetTargetSize(vk::Extent2D size);
    void Shutdown(Scheduler& scheduler);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace Vulkan
