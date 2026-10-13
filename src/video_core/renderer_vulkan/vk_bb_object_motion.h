// SPDX-License-Identifier: GPL-2.0-or-later
// Character motion vectors for the temporal upscaler, ported from bbport's ObjectMotion.
#pragma once

#include <array>
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_bb_motion_history.h"
#include "video_core/renderer_vulkan/vk_common.h"

#include <vk_mem_alloc.h>

namespace Vulkan {

class Instance;
class Scheduler;
struct RenderState;

// Skinned G-buffer draws (characters, cloth, enemies) store each vertex's clip position and
// read last frame's (runtime_info.h, MotionVectors). Their fragment shaders write the true
// screen-space motion to an extra RGBA32F attachment: rg = previous - current in render pixels,
// b = valid, a = depth. The upscaler's motion pass prefers it where the depth still matches.
class BbObjectMotion {
public:
    static constexpr u32 FrameSlots = 4;
    static constexpr u32 ParamsPerFrame = 8192;
    static constexpr u32 PositionsPerFrame = 2u << 20; // vertices; 2 x 32 MiB of history
    static constexpr u32 ParamBytes = 32;

    BbObjectMotion(const Instance& instance, Scheduler& scheduler, bool requested);
    ~BbObjectMotion();

    BbObjectMotion(const BbObjectMotion&) = delete;
    BbObjectMotion& operator=(const BbObjectMotion&) = delete;

    bool Enabled() const {
        return enabled;
    }
    // Runtime switch (dlss.ini character_motion). Off: draws get no history and the upscaler
    // ignores the image; the pipelines keep the attachment until restart.
    void SetActive(bool value) {
        active = value;
    }
    bool Active() const {
        return enabled && active;
    }

    // Device address of the all-zero parameters: no stores, no loads.
    u64 NoParam() const {
        return params_address;
    }
    // Parameters for one skinned draw; returns their device address (NoParam() when skipped).
    u64 PrepareDraw(const Motion::Draw& draw);
    bool Moving(const Motion::History::GateKey& key, u64 palette) {
        return history.Moving(key, palette);
    }
    template <class Scan>
    Motion::IndexRangeCache::Result IndexRange(const Motion::IndexRangeCache::Key& key,
                                               Scan&& scan) {
        return index_ranges.Get(key, frame, std::forward<Scan>(scan));
    }

    // Adds the motion attachment to a G-buffer render state (cleared on first use each frame).
    void Attach(RenderState& state);
    // Between frames, after the upscaler has read the image.
    void OnFrameStart();
    // The motion image for the upscaler, in GENERAL layout; `valid` when it was written this
    // frame at this size. Records its barrier on `cmd` (outside a render pass).
    vk::ImageView PrepareRead(vk::CommandBuffer cmd, u32 width, u32 height, bool& valid);

private:
    struct Target {
        vk::Image image{};
        VmaAllocation allocation{};
        vk::ImageView view{};
        u32 width{}, height{};
        u64 used_frame{};
        vk::ImageLayout layout{vk::ImageLayout::eUndefined};
        bool written{};
    };
    Target& GetTarget(u32 width, u32 height);
    void DestroyTarget(Target& target, bool deferred);

    const Instance& instance;
    Scheduler& scheduler;
    bool enabled{}, active{true};
    vk::Buffer params_buffer{}, positions_buffer{};
    VmaAllocation params_allocation{}, positions_allocation{};
    u32* params_mapped{};
    u64 params_address{}, positions_address{};
    u32 params_used{};
    std::array<u64, FrameSlots> params_ticks{};
    u64 frame{};
    Motion::History history{PositionsPerFrame};
    Motion::IndexRangeCache index_ranges;
    std::array<Target, 4> targets{};
    Target* read_target{};
    u64 report_frame{};
};

} // namespace Vulkan
