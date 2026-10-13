// SPDX-License-Identifier: GPL-2.0-or-later
// Character motion vectors for the temporal upscaler, ported from bbport's ObjectMotion.

#include "video_core/renderer_vulkan/vk_bb_object_motion.h"

#include <cstring>
#include "common/assert.h"
#include "common/logging/log.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

namespace {

bool CreateBuffer(const Instance& instance, vk::DeviceSize size, bool host, vk::Buffer& buffer,
                  VmaAllocation& allocation, void** mapped, u64& address) {
    const VkBufferCreateInfo buffer_ci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    };
    VmaAllocationCreateInfo alloc_ci{
        .usage = VMA_MEMORY_USAGE_AUTO,
    };
    if (host) {
        alloc_ci.requiredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        alloc_ci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                         VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    } else {
        alloc_ci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    }
    VkBuffer raw{};
    VmaAllocationInfo info{};
    if (vmaCreateBuffer(instance.GetAllocator(), &buffer_ci, &alloc_ci, &raw, &allocation,
                        &info) != VK_SUCCESS) {
        return false;
    }
    buffer = raw;
    if (mapped) {
        *mapped = info.pMappedData;
    }
    address = instance.GetDevice().getBufferAddress({.buffer = buffer});
    return address != 0;
}

} // namespace

BbObjectMotion::BbObjectMotion(const Instance& instance_, Scheduler& scheduler_, bool requested)
    : instance{instance_}, scheduler{scheduler_} {
    if (!requested) {
        return;
    }
    const auto features = instance.GetPhysicalDevice().getFeatures();
    if (!features.vertexPipelineStoresAndAtomics) {
        LOG_WARNING(Render_Vulkan,
                    "[DLSS-TEMPORAL] Character motion: vertex stores unsupported, off");
        return;
    }
    void* mapped = nullptr;
    if (!CreateBuffer(instance, vk::DeviceSize(1 + FrameSlots * ParamsPerFrame) * ParamBytes, true,
                      params_buffer, params_allocation, &mapped, params_address) ||
        !CreateBuffer(instance, vk::DeviceSize(1 + 2 * u64(PositionsPerFrame)) * 16, false,
                      positions_buffer, positions_allocation, nullptr, positions_address) ||
        !mapped) {
        LOG_WARNING(Render_Vulkan, "[DLSS-TEMPORAL] Character motion: buffers failed, off");
        return;
    }
    params_mapped = static_cast<u32*>(mapped);
    std::memset(params_mapped, 0, ParamBytes); // entry 0: no stores, no loads
    Shader::MotionVectors::enabled = true;
    enabled = true;
    LOG_INFO(Render_Vulkan, "[DLSS-TEMPORAL] Character motion: on ({} vertices per frame)",
             PositionsPerFrame);
}

BbObjectMotion::~BbObjectMotion() {
    if (!enabled) {
        return;
    }
    Shader::MotionVectors::enabled = false;
    scheduler.Finish();
    for (auto& target : targets) {
        DestroyTarget(target, false);
    }
    const auto allocator = instance.GetAllocator();
    if (params_buffer) {
        vmaDestroyBuffer(allocator, params_buffer, params_allocation);
    }
    if (positions_buffer) {
        vmaDestroyBuffer(allocator, positions_buffer, positions_allocation);
    }
}

void BbObjectMotion::OnFrameStart() {
    if (!enabled) {
        return;
    }
    if (frame - report_frame >= 600 && history.stats.draws) {
        report_frame = frame;
        const auto& s = history.stats;
        LOG_INFO(Render_Vulkan,
                 "[DLSS-TEMPORAL] Character motion: {} draws, {} stored, {} with history, {} "
                 "unmatched, {} over capacity, {} invalid, {} still; last frame {}/{} vertices; "
                 "index ranges {} reused, {} scanned",
                 s.draws, s.stored, s.loaded, s.unmatched, s.exhausted, s.invalid, s.still,
                 history.Used(), PositionsPerFrame, index_ranges.stats.hits,
                 index_ranges.stats.scans);
        history.stats = {};
        index_ranges.stats = {};
    }
    history.NextFrame();
    ++frame;
    if (frame % Motion::IndexRangeCache::Unused == 0) {
        index_ranges.Trim(frame);
    }
    const u32 slot = u32(frame % FrameSlots);
    // Do not overwrite parameters still referenced by queued GPU work.
    if (params_ticks[slot]) {
        scheduler.Wait(params_ticks[slot]);
        params_ticks[slot] = 0;
    }
    params_used = 0;
    for (auto& target : targets) {
        target.written = false;
    }
    scheduler.EndRendering();
    // Last frame's stores must be visible to this frame's loads, and old loads must finish
    // before that half of the history is written again.
    scheduler.Record([](vk::CommandBuffer cmd) {
        const vk::MemoryBarrier2 barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eVertexShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eVertexShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
        };
        cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
    });
}

u64 BbObjectMotion::PrepareDraw(const Motion::Draw& draw) {
    if (!Active() || params_used >= ParamsPerFrame) {
        return params_address;
    }
    const auto allocation = history.Prepare(draw);
    if (!allocation.store && !allocation.load) {
        return params_address;
    }
    const u32 slot = u32(frame % FrameSlots);
    const u32 index = 1 + slot * ParamsPerFrame + params_used++;
    u32* param = params_mapped + index * (ParamBytes / 4);
    const u64 store = allocation.store ? positions_address + u64(allocation.store) * 16 : 0;
    const u64 load = allocation.load ? positions_address + u64(allocation.load) * 16 : 0;
    param[0] = u32(store);
    param[1] = u32(store >> 32);
    param[2] = u32(load);
    param[3] = u32(load >> 32);
    param[4] = allocation.vertices;
    param[5] = allocation.first_vertex;
    param[6] = allocation.first_instance;
    param[7] = allocation.instances;
    params_ticks[slot] = scheduler.CurrentTick();
    return params_address + u64(index) * ParamBytes;
}

void BbObjectMotion::DestroyTarget(Target& target, bool deferred) {
    if (!target.image) {
        return;
    }
    const auto device = instance.GetDevice();
    const auto allocator = instance.GetAllocator();
    auto destroy = [device, allocator, image = target.image, allocation = target.allocation,
                    view = target.view] {
        device.destroyImageView(view);
        vmaDestroyImage(allocator, image, allocation);
    };
    if (deferred) {
        scheduler.DeferOperation(std::move(destroy));
    } else {
        destroy();
    }
    target = {};
}

BbObjectMotion::Target& BbObjectMotion::GetTarget(u32 width, u32 height) {
    for (auto& target : targets) {
        if (target.image && target.width == width && target.height == height) {
            target.used_frame = frame;
            return target;
        }
    }
    // A free slot, else the one used longest ago (not the one being read).
    Target* slot = nullptr;
    for (auto& target : targets) {
        if (!target.image) {
            slot = &target;
            break;
        }
        if (&target != read_target && (!slot || target.used_frame < slot->used_frame)) {
            slot = &target;
        }
    }
    DestroyTarget(*slot, true);
    const VkImageCreateInfo image_ci{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_R32G32B32A32_SFLOAT,
        .extent = {width, height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    const VmaAllocationCreateInfo alloc_ci{.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    VkImage raw{};
    if (vmaCreateImage(instance.GetAllocator(), &image_ci, &alloc_ci, &raw, &slot->allocation,
                       nullptr) != VK_SUCCESS) {
        UNREACHABLE_MSG("Character motion image {}x{} could not be created", width, height);
    }
    slot->image = raw;
    slot->view = instance.GetDevice()
                     .createImageView({
                         .image = slot->image,
                         .viewType = vk::ImageViewType::e2D,
                         .format = vk::Format::eR32G32B32A32Sfloat,
                         .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
                     })
                     .value;
    slot->width = width;
    slot->height = height;
    slot->written = false;
    slot->layout = vk::ImageLayout::eUndefined;
    slot->used_frame = frame;
    LOG_INFO(Render_Vulkan, "[DLSS-TEMPORAL] Character motion image {}x{}", width, height);
    return *slot;
}

void BbObjectMotion::Attach(RenderState& state) {
    constexpr u32 slot = Shader::MotionVectors::Output;
    Target& target = GetTarget(state.width, state.height);
    auto& attachment = state.color_attachments[slot];
    attachment = {};
    attachment.image_view = target.view;
    attachment.image_layout = vk::ImageLayout::eColorAttachmentOptimal;
    if (!target.written || target.layout != vk::ImageLayout::eColorAttachmentOptimal) {
        scheduler.EndRendering();
        const vk::ImageMemoryBarrier2 barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite |
                             vk::AccessFlagBits2::eColorAttachmentRead,
            .oldLayout = target.layout,
            .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .image = target.image,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
        scheduler.Record([barrier](vk::CommandBuffer cmd) {
            cmd.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
        });
        target.layout = vk::ImageLayout::eColorAttachmentOptimal;
        // Cleared once per frame (zero = no vector); later passes load what is there.
        attachment.is_clear = !target.written;
        target.written = true;
    }
    state.num_color_attachments = std::max<u16>(state.num_color_attachments, slot + 1);
}

vk::ImageView BbObjectMotion::PrepareRead(vk::CommandBuffer cmd, u32 width, u32 height,
                                          bool& valid) {
    valid = false;
    if (!enabled) {
        return {};
    }
    Target* target = nullptr;
    for (auto& candidate : targets) {
        if (candidate.image && candidate.width == width && candidate.height == height) {
            target = &candidate;
        }
    }
    if (!target) {
        // Any image: the pass binds one, its vectors unused.
        for (auto& candidate : targets) {
            if (candidate.image && (!target || candidate.used_frame > target->used_frame)) {
                target = &candidate;
            }
        }
    }
    if (!target) {
        target = &GetTarget(1, 1);
    }
    valid = Active() && target->written && target->width == width && target->height == height;
    const vk::ImageMemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .oldLayout = target->layout,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = target->image,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    };
    cmd.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
    target->layout = vk::ImageLayout::eGeneral;
    read_target = target;
    return target->view;
}

} // namespace Vulkan
