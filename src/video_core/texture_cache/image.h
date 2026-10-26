// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <mutex>
#include <optional>
#include <source_location>

#include "common/enum.h"
#include "common/incremental_id.h"
#include "common/lru_cache.h"
#include "common/small_vector.h"
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/image_view.h"

namespace Vulkan {
class Instance;
class Runtime;
} // namespace Vulkan

VK_DEFINE_HANDLE(VmaAllocation)
VK_DEFINE_HANDLE(VmaAllocator)

namespace VideoCore {

enum ImageFlagBits : u32 {
    Empty = 0,
    MaybeCpuDirty = 1 << 0, ///< The page this image is in was touched before the image address
    CpuDirty = 1 << 1,      ///< Contents have been modified from the CPU
    GpuDirty = 1 << 2, ///< Contents have been modified from the GPU (valid data in buffer cache)
    Dirty = MaybeCpuDirty | CpuDirty | GpuDirty,
    GpuModified = 1 << 3, ///< Contents have been modified from the GPU
    Registered = 1 << 6,  ///< True when the image is registered
};
DECLARE_ENUM_FLAG_OPERATORS(ImageFlagBits)

/// Texture quality: an image a shader reads at its exact size keeps all its levels from now on.
void KeepFullQuality(VAddr guest_address);

struct UniqueImage {
    explicit UniqueImage() = default;
    explicit UniqueImage(vk::Device device, VmaAllocator allocator)
        : device{device}, allocator{allocator} {}
    ~UniqueImage();

    UniqueImage(const UniqueImage&) = delete;
    UniqueImage& operator=(const UniqueImage&) = delete;

    UniqueImage(UniqueImage&& other)
        : allocator{std::exchange(other.allocator, VK_NULL_HANDLE)},
          allocation{std::exchange(other.allocation, VK_NULL_HANDLE)},
          image{std::exchange(other.image, VK_NULL_HANDLE)}, image_ci{std::move(other.image_ci)},
          vram_owner{std::exchange(other.vram_owner, ~0u)} {}
    UniqueImage& operator=(UniqueImage&& other) {
        image = std::exchange(other.image, VK_NULL_HANDLE);
        allocator = std::exchange(other.allocator, VK_NULL_HANDLE);
        allocation = std::exchange(other.allocation, VK_NULL_HANDLE);
        image_ci = std::move(other.image_ci);
        vram_owner = std::exchange(other.vram_owner, ~0u);
        return *this;
    }

    /// `owner`: who made it, for the VRAM report (by default the calling file and line).
    void Create(const vk::ImageCreateInfo& image_ci,
                std::source_location owner = std::source_location::current());

    void Destroy();

    operator vk::Image() const {
        return image;
    }

    operator bool() const {
        return image;
    }

public:
    vk::Device device{};
    VmaAllocator allocator{};
    VmaAllocation allocation{};
    vk::Image image{};
    vk::ImageCreateInfo image_ci{};
    vk::DeviceSize size_bytes{};
    u32 vram_owner{~0u}; ///< VramOwners slot of a device-local allocation
};

struct Image : public Common::LRUNode<> {
    explicit Image(const Vulkan::Instance& instance, Vulkan::Runtime& runtime,
                   Common::SlotVector<ImageView>& slot_image_views, const ImageInfo& info);
    ~Image();

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    Image(Image&&) = delete;
    Image& operator=(Image&&) = delete;

    bool Overlaps(VAddr addr, size_t size) const noexcept {
        return info.guest_address < (addr + size) && addr < (info.guest_address + info.guest_size);
    }

    vk::Image GetImage() const {
        return backing->image.image;
    }

    bool IsUntracked() {
        return track_addr == 0 || track_addr_end == 0;
    }

    bool SafeToDownload() const {
        return True(flags & ImageFlagBits::GpuModified) && False(flags & ImageFlagBits::Dirty);
    }

    void AssociateDepth(ImageId depth_image_id, u64 depth_image_uid) {
        depth_id = depth_image_id;
        depth_uid = depth_image_uid;
    }

    void DisassociateDepth() {
        depth_id = {};
        depth_uid = {};
    }

    /// VRAM the image takes: all its backings (sample counts), as allocated.
    vk::DeviceSize SizeBytes() const {
        vk::DeviceSize size = 0;
        for (const auto& backing_image : backing_images) {
            size += backing_image.image.size_bytes;
        }
        return size;
    }

    ImageView& FindView(const ImageViewInfo& view_info, bool ensure_guest_samples = true);

    /// The levels the Vulkan image has: the guest's less those left out by texture quality.
    SubresourceExtent HostResources() const {
        return {static_cast<u16>(info.resources.levels - dropped_levels), info.resources.layers};
    }

    /// A range of guest levels as levels of the Vulkan image (the guest levels left out read
    /// the largest one kept).
    SubresourceRange HostRange(const SubresourceRange& range) const {
        if (dropped_levels == 0) {
            return range;
        }
        const u32 host_levels = info.resources.levels - dropped_levels;
        const u32 base = std::min<u32>(
            range.base.level >= dropped_levels ? range.base.level - dropped_levels : 0,
            host_levels - 1);
        const u32 end = std::min<u32>(
            std::max<u32>(range.base.level + range.extent.levels, dropped_levels + 1) -
                dropped_levels,
            host_levels);
        return {.base = {static_cast<u16>(base), range.base.layer},
                .extent = {static_cast<u16>(std::max<u32>(end - base, 1)), range.extent.layers}};
    }

    using Barriers = SmallVector<vk::ImageMemoryBarrier2, 32>;
    void GetBarriers(Barriers& out_barriers, vk::ImageLayout dst_layout, vk::AccessFlags2 dst_mask,
                     vk::PipelineStageFlags2 dst_stage,
                     std::optional<SubresourceRange> subres_range = {});

public:
    Vulkan::Runtime* runtime;
    Common::SlotVector<ImageView>* slot_image_views;
    std::mutex mutex;
    ImageInfo info;
    vk::ImageAspectFlags aspect_mask = vk::ImageAspectFlagBits::eColor;
    vk::SampleCountFlags supported_samples = vk::SampleCountFlagBits::e1;
    ImageFlagBits flags = ImageFlagBits::Dirty;
    VAddr track_addr = 0;
    VAddr track_addr_end = 0;
    /// Threaded renderer: GC tick + 1 at which UpdateImage last found this image clean, fully
    /// tracked and touched; 0 once anything changes that. Only accessed through atomic_ref.
    u64 clean_tick = 0;
    ImageId depth_id{};
    u64 depth_uid{};

    vk::ImageUsageFlags usage_flags;
    vk::FormatFeatureFlags2 format_features;
    struct State {
        vk::PipelineStageFlags2 pl_stage = vk::PipelineStageFlagBits2::eAllCommands;
        vk::AccessFlags2 access_mask = vk::AccessFlagBits2::eNone;
        vk::ImageLayout layout = vk::ImageLayout::eUndefined;
    };
    struct BackingImage {
        UniqueImage image;
        State state;
        std::vector<State> subresource_states;
        SmallVector<ImageViewInfo, 2> image_view_infos;
        SmallVector<ImageViewId, 2> image_view_ids;
        u32 num_samples;
    };
    SmallVector<BackingImage, 2> backing_images;
    BackingImage* backing{};
    /// While registered: the texture cache's count of image memory (a backing added later, for
    /// another sample count, is counted too).
    std::atomic<u64>* memory_usage_counter{};
    u64 image_uid{};
    u64 lru_id{};
    u64 tick_accessed_last{};
    u64 last_use_gc_tick{}; ///< VRAM cleanup: GC tick of the last use (atomic_ref access)
    u64 hash{};
    /// Shader::ExactTexelRead bits of the shaders that read it (atomic_ref access).
    u8 exact_texel_reads{};
    /// Texture quality: the largest guest levels the Vulkan image leaves out (it is that much
    /// smaller, and its level 0 is guest level dropped_levels).
    u32 dropped_levels{};

    struct {
        u32 texture : 1;
        u32 storage : 1;
        u32 render_target : 1;
        u32 depth_target : 1;
        u32 vo_surface : 1;
    } usage{};

    struct {
        u32 is_bound : 1;
        u32 is_target : 1;
        u32 needs_rebind : 1;
        u32 force_general : 1;
    } binding{};

private:
    static Common::IncrementalIdProvider<u64> global_image_uid;
};

} // namespace VideoCore
