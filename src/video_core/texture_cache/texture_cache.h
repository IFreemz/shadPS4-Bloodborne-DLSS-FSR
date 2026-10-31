// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <boost/container/small_vector.hpp>
#include <queue>
#include <tsl/robin_map.h>

#include "common/bb_tuning.h"
#include "common/guest_stats.h"
#include "common/lru_cache.h"
#include "common/multi_level_page_table.h"
#include "common/slot_vector.h"
#include "shader_recompiler/resource.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/texture_cache/blit_helper.h"
#include "video_core/texture_cache/image.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/sampler.h"
#include "video_core/texture_cache/tile_manager.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Vulkan {
class Runtime;
}

namespace VideoCore {

class BufferCache;
class PageManager;

class TextureCache {
    static constexpr s64 DEFAULT_PRESSURE_GC_MEMORY = 1_GB + 512_MB;
    static constexpr s64 DEFAULT_CRITICAL_GC_MEMORY = 3_GB;
    static constexpr s64 TARGET_GC_THRESHOLD = 8_GB;

    struct BucketEntry {
        u32 key;
        u32 size;
        ImageId id;

        bool Overlaps(VAddr addr, size_t size) const noexcept {
            const VAddr base = Address();
            return base < (addr + size) && addr < (base + this->size);
        }

        VAddr Address() const noexcept {
            return VAddr(key) << 8;
        }
    };

    struct alignas(64) Bucket {
        SmallVector<BucketEntry, 4, u32> entries;
    };
    static_assert(sizeof(Bucket) == 64);

    struct Traits {
        using Entry = Bucket;
        static constexpr size_t ADDRESS_SPACE_BITS = 40;
        static constexpr size_t L1_BITS = 10;
        static constexpr size_t PAGE_BITS = 18;
        static constexpr bool NULL_CHECK = true;
    };
    using PageTable = Common::MultiLevelPageTable<Traits>;

public:
    enum class BindingType : u32 {
        Texture,
        Storage,
        RenderTarget,
        DepthTarget,
        VideoOut,
    };

    struct ImageDesc {
        ImageInfo info;
        ImageViewInfo view_info;
        BindingType type{BindingType::Texture};

        ImageDesc() = default;
        ImageDesc(const AmdGpu::Image& image, const Shader::ImageResource& desc)
            : info{image, desc}, view_info{image, desc},
              type{desc.is_written ? BindingType::Storage : BindingType::Texture} {}
        ImageDesc(const AmdGpu::ColorBuffer& buffer, AmdGpu::CbDbExtent hint)
            : info{buffer, hint}, view_info{buffer}, type{BindingType::RenderTarget} {}
        ImageDesc(const AmdGpu::DepthBuffer& buffer, AmdGpu::DepthView view,
                  AmdGpu::DepthControl ctl, VAddr htile_address, AmdGpu::CbDbExtent hint,
                  bool write_buffer = false)
            : info{buffer, view.NumSlices(), htile_address, hint, write_buffer},
              view_info{buffer, view, ctl}, type{BindingType::DepthTarget} {}
        ImageDesc(const Libraries::VideoOut::BufferAttributeGroup& group, VAddr cpu_address)
            : info{group, cpu_address}, type{BindingType::VideoOut} {}
    };
    static_assert(std::is_trivially_destructible_v<ImageDesc>);

public:
    TextureCache(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                 Vulkan::Runtime& runtime, AmdGpu::Liverpool* liverpool, BufferCache& buffer_cache,
                 PageManager& tracker);
    ~TextureCache();

    TileManager& GetTileManager() noexcept {
        return tile_manager;
    }

    /// Invalidates any image in the logical page range.
    void InvalidateMemory(VAddr addr, size_t size);

    /// Marks an image as dirty if it exists at the provided address.
    void InvalidateMemoryFromGPU(VAddr address, size_t max_size);

    /// Evicts any images that overlap the unmapped range.
    void UnmapMemory(VAddr cpu_addr, size_t size);

    /// Schedules a copy of pending images for download back to CPU memory.
    void ProcessDownloadImages();

    /// Retrieves the image handle of the image with the provided attributes.
    [[nodiscard]] ImageId FindImage(ImageDesc& desc, bool exact_fmt = false);

    /// Retrieves image whose address matches provided
    [[nodiscard]] ImageId FindImageFromRange(VAddr address, size_t size, bool ensure_valid = true);

    /// The image of which [address, +size) is exactly one layer (an array or cube face, one
    /// level, all levels kept): `layer` is set to which. Null when none.
    [[nodiscard]] ImageId FindImageLayer(VAddr address, size_t size, u32& layer);

    /// Retrieves an image view with the properties of the specified image id.
    [[nodiscard]] ImageView& FindTexture(ImageId image_id, const ImageDesc& desc);

    /// Retrieves the render target with specified properties
    [[nodiscard]] ImageView& FindRenderTarget(ImageId image_id, const ImageDesc& desc);

    /// Retrieves the depth target with specified properties
    [[nodiscard]] ImageView& FindDepthTarget(ImageId image_id, const ImageDesc& desc);

    /// Links the stencil plane of a depth surface to its depth image.
    void AssociateStencil(ImageId depth_id, const ImageInfo& depth_info);

    /// Updates image contents if it was modified by CPU.
    void UpdateImage(ImageId image_id) {
        Image& image = slot_images[image_id];
        // Threaded renderer fast path (after bbport): skip the work below when this image was
        // already found clean, fully tracked and touched in this GC tick. Only clean_tick is read
        // here: everything that would make the image need work again clears it under the
        // image's lock, so a fault racing with this check is the same as one arriving just after.
        if (Common::BbTuning::Threaded() &&
            std::atomic_ref<u64>(image.clean_tick).load(std::memory_order_acquire) ==
                std::atomic_ref<u64>(gc_tick).load(std::memory_order_acquire) + 1) {
            return;
        }
        TouchImage(image);
        // Protection on a helper thread: the protection calls of newly tracked pages go with the
        // upload's background copies (applied here when there is none).
        ProtectRanges deferred;
        const bool defer = ProtectOnHelperThread();
        {
            Common::GuestStats::DetailTimer timer{Common::GuestStats::Detail::TexTrack};
            TrackImage(image_id, defer ? &deferred : nullptr);
        }
        if (defer) {
            pending_protect = &deferred;
            RefreshImage(image);
            pending_protect = nullptr;
            ApplyProtection(deferred);
        } else {
            RefreshImage(image);
        }
        // Under the image's lock: a fault handler untracking it (and clearing clean_tick) must
        // not be overwritten by a stale "clean".
        std::scoped_lock lk{image.mutex};
        if (False(image.flags & ImageFlagBits::Dirty) &&
            True(image.flags & ImageFlagBits::Registered) &&
            image.track_addr == image.info.guest_address &&
            image.track_addr_end == image.info.guest_address + image.info.guest_size) {
            std::atomic_ref<u64>(image.clean_tick).store(gc_tick + 1, std::memory_order_release);
        }
    }

    /// UpdateImage for a sampled texture: while an area load is spread over frames, its upload
    /// may wait for a later frame (it stays dirty: the next draw binding it tries again).
    void UpdateTexture(ImageId image_id) {
        if (slot_images[image_id].detail_pending) {
            TryRestoreDetail(slot_images[image_id]);
        }
        // Lower detail first applies to the uploads of sampled textures only.
        if (spread_loads_enabled && !IsClean(image_id) && DeferUpload(slot_images[image_id])) {
            TouchImage(slot_images[image_id]);
            return;
        }
        const bool was_sampled = std::exchange(sampled_update, true);
        Common::GuestStats::UploadPathScope path{Common::GuestStats::UploadPath::Texture};
        UpdateImage(image_id);
        sampled_update = was_sampled;
    }

    /// Spread area loading: the per-frame upload budget starts again (each game frame).
    void OnGameFrame() {
        spread_frame_bytes.store(0, std::memory_order_relaxed);
        detail_frame_bytes.store(0, std::memory_order_relaxed);
        detail_frame_restores.store(0, std::memory_order_relaxed);
    }

    /// UpdateImage for the images a draw binds, with the uploads of those that need one recorded
    /// together (Batched texture uploads setting): detiles after one barrier, copies after
    /// another, one render pass break.
    void UpdateImages(std::span<const ImageId> image_ids);

    /// Protection on a helper thread (setting, threaded renderer with background copies).
    [[nodiscard]] static bool ProtectOnHelperThread();

    /// Applies deferred protection calls now (none left over afterwards).
    void ApplyProtection(ProtectRanges& ranges);

    /// Lower detail first: uploads the largest level an image still waits for, now (before a
    /// use other than sampling: render target, storage, copy source).
    void EnsureFullDetail(Image& image);

    /// Clear new shader-written images: the image's first contents are zeros cleared on the GPU
    /// instead of its guest memory (a shader about to overwrite all of it).
    void ClearInsteadOfUpload(Image& image);

    /// The slot holds an image (it may have been freed since an id was taken).
    [[nodiscard]] bool IsImageAllocated(ImageId image_id) const {
        return slot_images.IsAllocated(image_id);
    }

    /// UpdateImage would return at once: the image was found clean, fully tracked and touched
    /// in this GC tick (threaded renderer memos reuse what it resolved).
    [[nodiscard]] bool IsClean(ImageId image_id) {
        Image& image = slot_images[image_id];
        return std::atomic_ref<u64>(image.clean_tick).load(std::memory_order_acquire) ==
               std::atomic_ref<u64>(gc_tick).load(std::memory_order_acquire) + 1;
    }

    /// The image needs UpdateImage's full path again (called with the lock held).
    static void ClearCleanTick(Image& image) {
        std::atomic_ref<u64>(image.clean_tick).store(0, std::memory_order_release);
    }

    /// Resolves overlap between existing cache image and pending merged image
    [[nodiscard]] std::tuple<ImageId, int, int> ResolveOverlap(const ImageInfo& info,
                                                               BindingType binding,
                                                               ImageId cache_img_id,
                                                               ImageId merged_image_id);

    /// Resolves depth overlap and either re-creates the image or returns existing one
    [[nodiscard]] ImageId ResolveDepthOverlap(const ImageInfo& requested_info, BindingType binding,
                                              ImageId cache_img_id);

    /// Creates a new image with provided image info and copies subresources from image_id
    [[nodiscard]] ImageId ExpandImage(const ImageInfo& info, ImageId image_id);

    /// Reuploads image contents.
    void RefreshImage(Image& image);

    /// RefreshImage's checks and copy regions: false when there is nothing to upload.
    bool PrepareRefresh(Image& image, SmallVector<vk::BufferImageCopy, 14>& copies);

    /// Spread area loading: whether this texture's upload waits for a later frame. A new
    /// texture is filled with zeros (black) meanwhile.
    bool DeferUpload(Image& image);

    /// Lower detail first: for a new sampled texture in a burst, drops the copies of its
    /// largest level (it waits, the image is marked detail_pending). Returns the guest bytes in
    /// front an upload does not need to copy (LeftOutBytes, or up to the next level).
    u32 UploadSkip(Image& image, SmallVector<vk::BufferImageCopy, 14>& copies);

    /// UpdateImages' work (with lower detail first's sampled-upload mark set).
    void UpdateImagesSampled(std::span<const ImageId> image_ids);

    /// Lower detail first: uploads the waiting largest level when this frame's budget allows.
    void TryRestoreDetail(Image& image);

    /// Retrieves the sampler that matches the provided S# descriptor.
    [[nodiscard]] vk::Sampler GetSampler(const AmdGpu::Sampler& sampler,
                                         AmdGpu::BorderColorBuffer border_color_base, bool is_depth,
                                         float extra_lod_bias = 0.0f);

    /// Retrieves the image with the specified id.
    [[nodiscard]] Image& GetImage(ImageId id) {
        auto& image = slot_images[id];
        TouchImage(image);
        return image;
    }

    /// Changes whenever an image is registered or unregistered: FindImage results remembered
    /// at one generation hold as long as it stays the same (threaded renderer memos).
    [[nodiscard]] u64 RegistryGeneration() const noexcept {
        return registry_generation.load(std::memory_order_acquire);
    }

    /// Changes whenever samplers are destroyed: remembered sampler handles hold while it stays.
    [[nodiscard]] u64 SamplerGeneration() const noexcept {
        return sampler_generation.load(std::memory_order_acquire);
    }
    /// The garbage collection tick (samplers and images not touched for a while are destroyed).
    [[nodiscard]] u64 GcTick() const noexcept {
        return std::atomic_ref<u64>(const_cast<u64&>(gc_tick)).load(std::memory_order_acquire);
    }
    /// GetSampler's LRU touch for a remembered sampler; once per GC tick is enough.
    void TouchSampler(const AmdGpu::Sampler& sampler, bool is_depth, float extra_lod_bias = 0.0f);

    /// FindImage's bookkeeping for an image found through such a memo.
    void TouchFoundImage(ImageId id) {
        auto& image = slot_images[id];
        image.tick_accessed_last = scheduler.CurrentTick();
        TouchImage(image);
    }

    /// Texture quality estimate: a shader read this texture in a way that needs its exact size
    /// (Shader::ExactTexelRead bits).
    /// Texture quality: an image with levels left out is freed at the next cleanup and loaded
    /// again with all of them.
    void NoteExactTexelReads(ImageId image_id, u8 reads, u64 pgm_hash);

    /// Returns the image if the slot still holds the same image (uid), without refreshing it.
    [[nodiscard]] Image* TryGetImage(ImageId id, u64 uid) {
        if (!slot_images.IsAllocated(id))
            return nullptr;
        auto& image = slot_images[id];
        if (image.image_uid != uid || False(image.flags & ImageFlagBits::Registered))
            return nullptr;
        TouchImage(image);
        return &image;
    }

    /// Retrieves the image view with the specified id.
    [[nodiscard]] ImageView& GetImageView(ImageId id) {
        return slot_image_views[id];
    }

    /// Get the associated depth stencil image if it is still valid.
    ImageId GetAssociatedDepth(Image& image) {
        if (!image.depth_id) {
            return {};
        }
        if (slot_images.IsAllocated(image.depth_id)) {
            auto& depth_image = slot_images[image.depth_id];
            if (depth_image.image_uid == image.depth_uid &&
                depth_image.flags & ImageFlagBits::Registered) {
                return image.depth_id;
            }
        }
        // The linked depth image is no longer valid, disassociate it.
        image.DisassociateDepth();
        return {};
    }

    enum class MetaType {
        CMask,
        FMask,
        HTile,
    };

    struct MetaDataInfo {
        MetaType type;
        ImageId owner_id{};
        u64 owner_uid{};
        u32 num_samples = 1;
        s32 clear_mask = -1;
    };

    /// Returns metadata information if the specified address belongs to a live image.
    std::optional<MetaDataInfo> GetMetaInfo(VAddr address) const {
        const auto it = surface_metas.find(address);
        if (it == surface_metas.end()) {
            return std::nullopt;
        }
        const auto& meta = it->second;
        if (!meta.owner_id || !slot_images.IsAllocated(meta.owner_id)) {
            return std::nullopt;
        }
        const auto& owner = slot_images[meta.owner_id];
        if (owner.image_uid != meta.owner_uid || False(owner.flags & ImageFlagBits::Registered)) {
            return std::nullopt;
        }
        return meta;
    }

    /// Returns meta type if the specified address is a metadata surface.
    std::optional<MetaType> IsMeta(VAddr address) const {
        auto it = surface_metas.find(address);
        if (it != surface_metas.end()) {
            return it->second.type;
        }
        return std::nullopt;
    }

    /// Returns true if a slice of the specified metadata surface has been cleared.
    bool IsMetaCleared(VAddr address, u32 slice) const {
        const auto& it = surface_metas.find(address);
        if (it != surface_metas.end()) {
            return it->second.clear_mask & (1u << slice);
        }
        return false;
    }

    /// Clears all slices of the specified metadata surface.
    bool ClearMeta(VAddr address) {
        auto it = surface_metas.find(address);
        if (it != surface_metas.end()) {
            it->second.clear_mask = u32(-1);
            return true;
        }
        return false;
    }

    /// Updates the state of a slice of the specified metadata surface.
    bool TouchMeta(VAddr address, u32 slice, bool is_clear) {
        auto it = surface_metas.find(address);
        if (it != surface_metas.end()) {
            if (is_clear) {
                it->second.clear_mask |= 1u << slice;
            } else {
                it->second.clear_mask &= ~(1u << slice);
            }
            return true;
        }
        return false;
    }

    /// Runs the garbage collector.
    void RunGarbageCollector();

    /// What the cached images hold, by what the garbage collector can do with them.
    struct MemoryReport {
        struct Group {
            u64 count{};
            u64 bytes{};
            u64 stale_count{}; ///< not used for 16+ collector ticks (old enough to collect)
            u64 stale_bytes{};
        };
        Group clean;   ///< copies of guest memory: freed whenever they are old enough
        Group written; ///< GPU-written linear images: freed (written back) only under pressure
        Group stuck;   ///< GPU-written tiled images: never freed (no tiled write-back)
        u64 extra_backing_bytes{}; ///< additional backing images (other sample counts)
        u64 allocated_images{};    ///< all image slots, including ones awaiting deletion
        u64 trigger{}, pressure{}, critical{}, gc_used{};
        u64 evicted{}, written_back{}, skipped_stuck{}, skipped_written{}; ///< since last report
        u64 gc_runs{};
        u64 loads{};           ///< area loads seen (upload bursts) since last report
        u64 spread_loads{};    ///< loads spread over several frames
        u64 spread_deferred{}; ///< uploads put off to a later frame (each try counted)
        u64 spread_zero_filled{}, spread_zero_bytes{}; ///< new images shown black meanwhile
    };
    /// Fills a report and resets the collector counters. Call on the thread running the collector.
    MemoryReport TakeMemoryReport();

    /// Totals of the VRAM cleanup counters (Common::GuestStats), for reports over a period.
    struct CleanupCounters {
        u64 passes{}, pressure_passes{}, aggressive_passes{}, ns{}, usage_query_ns{};
        u64 freed_images{}, freed_bytes{}, writebacks{}, writeback_bytes{}, writeback_done_ns{};
        u64 written_bytes{}, reuploads{}, reupload_bytes{}, reupload_age_ms{};
        u64 buffer_evictions{}, buffer_eviction_bytes{}, buffer_eviction_ns{};
        u64 buffer_eviction_unbind_ns{}, uploaded_bytes{}, budget_limit_passes{};
        u64 scratch_releases{}, scratch_released_bytes{}, scratch_allocations{};
        u64 scratch_allocation_bytes{};
    };
    [[nodiscard]] static CleanupCounters ReadCleanupCounters();

    template <typename Func>
    void ForEachImageInRegion(VAddr cpu_addr, size_t size, Func&& func) {
        using FuncReturn = typename std::invoke_result<Func, ImageId, Image&>::type;
        static constexpr bool BOOL_BREAK = std::is_same_v<FuncReturn, bool>;
        const u64 first_page = cpu_addr >> Traits::PAGE_BITS;
        ForEachPage(cpu_addr, size, [this, first_page, cpu_addr, size, func](u64 page) {
            const auto it = page_table.find(page);
            if (it == nullptr) {
                if constexpr (BOOL_BREAK) {
                    return false;
                } else {
                    return;
                }
            }
            for (const auto& entry : it->entries) {
                const u64 base_page = entry.Address() >> Traits::PAGE_BITS;
                if (page != std::max(first_page, base_page)) {
                    continue;
                }
                if (!entry.Overlaps(cpu_addr, size)) {
                    continue;
                }
                Image& image = slot_images[entry.id];
                if constexpr (BOOL_BREAK) {
                    if (func(entry.id, image)) {
                        return true;
                    }
                } else {
                    func(entry.id, image);
                }
            }
            if constexpr (BOOL_BREAK) {
                return false;
            }
        });
    }

    template <typename Func>
    void ForEachImageWithAddress(VAddr cpu_addr, Func&& func) {
        const auto it = page_table.find(cpu_addr >> Traits::PAGE_BITS);
        if (it == nullptr) {
            return;
        }
        const u32 key = u32(cpu_addr >> 8);
        for (const auto& entry : it->entries) {
            if (entry.key == key) {
                func(entry.id, slot_images[entry.id]);
            }
        }
    }

private:
    /// Iterate over all page indices in a range
    template <typename Func>
    static void ForEachPage(PAddr addr, size_t size, Func&& func) {
        static constexpr bool RETURNS_BOOL = std::is_same_v<std::invoke_result<Func, u64>, bool>;
        const u64 page_end = (addr + size - 1) >> Traits::PAGE_BITS;
        for (u64 page = addr >> Traits::PAGE_BITS; page <= page_end; ++page) {
            if constexpr (RETURNS_BOOL) {
                if (func(page)) {
                    break;
                }
            } else {
                func(page);
            }
        }
    }

    /// Copies image memory back to CPU.
    bool DownloadImageMemory(ImageId image_id, bool sync = false);

    /// VRAM cleanup: synchronously writes an image back to guest memory before it is freed.
    bool BeginWriteBack(ImageId image_id);
    void QueueWriteBack(ImageId image_id, const Image& image,
                        const Vulkan::StagingBufferRef& download, u64 size);
    void CompleteWriteBacks();
    bool SafeToWriteBack(const Image& image);

    /// Register image in the page table
    void RegisterImage(ImageId image);

    /// Unregister image from the page table
    void UnregisterImage(ImageId image);

    /// Track CPU reads and writes for image
    /// With `deferred`, the protection calls are added to it instead (ApplyProtection).
    void TrackImage(ImageId image_id, ProtectRanges* deferred = nullptr);

    /// Stop tracking CPU reads and writes for image
    void UntrackImage(ImageId image_id);
    void UntrackImageHead(ImageId image_id);
    void UntrackImageTail(ImageId image_id);

    void MarkAsMaybeDirty(ImageId image_id, Image& image);

    void RegisterMeta(VAddr address, MetaDataInfo info);
    void UnregisterMeta(VAddr address, u64 owner_uid);

    /// Removes the image and any views/surface metas that reference it.
    void DeleteImage(ImageId image_id);

    /// Touch the image in the LRU cache.
    void TouchImage(Image& image) {
        image_lru_cache.Touch(image, gc_tick);
        // VRAM cleanup: a write-back recorded before this use is out of date.
        std::atomic_ref<u64>(image.last_use_gc_tick).store(gc_tick, std::memory_order_relaxed);
    }

    /// Removes image from the cache and schedules it for deletion.
    void FreeImage(ImageId image_id) {
        {
            std::scoped_lock lk{slot_images[image_id].mutex};
            UntrackImage(image_id);
        }
        UnregisterImage(image_id);
        DeleteImage(image_id);
    }

    void GarbageCollectImages();
    void GarbageCollectSamplers();
    void LogCleanupReport();
    void LogVramBreakdown();
    void LogTextureQualityEstimate(u64 tick_30);

private:
    void RestoreFullQuality();
    void NoteEvicted(const Image& image);
    void NoteUpload(Image& image);
    void LoadScreenCleanup();
    void ReloadTexturesFromGuest();
    void UpdateVramLimits();
    void ReportPastBudget();

    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    Vulkan::Runtime& runtime;
    AmdGpu::Liverpool* liverpool;
    BufferCache& buffer_cache;
    PageManager& tracker;
    PageTable page_table;
    Common::SlotVector<Image> slot_images;
    Common::SlotVector<ImageView> slot_image_views;
    Common::SlotVector<Sampler> slot_samplers;
    BlitHelper blit_helper;
    TileManager tile_manager;
    absl::flat_hash_map<u64, SamplerId> samplers;
    absl::flat_hash_set<ImageId> download_images;
    std::atomic<u64> total_used_memory{0}; ///< image memory (backings as allocated)
    u64 trigger_gc_memory = 0;
    u64 pressure_gc_memory = 0;
    u64 critical_gc_memory = 0;
    u64 total_used_samplers = 0;
    u64 trigger_gc_samplers = 0;
    u64 pressure_gc_samplers = 0;
    u64 critical_gc_samplers = 0;
    u64 gc_tick = 0;
    /// VRAM cleanup: images whose write-back failed, and the tick they may be tried again.
    std::unordered_map<u32, u64> writeback_retry;
    struct PendingWriteBack {
        ImageId image_id;
        u64 image_uid;
        u64 last_use; ///< the image's last_use_gc_tick when the copy was recorded
        u64 tick;     ///< scheduler tick that completes the copy
        VAddr address;
        u64 size;
        Vulkan::StagingBufferRef download;
    };
    std::vector<PendingWriteBack> pending_writebacks;
    double writeback_budget = 0.0;
    std::chrono::steady_clock::time_point writeback_budget_time{};
    std::array<u64, 64> gc_tick_at_second{}; ///< VRAM cleanup: gc_tick at each of the last 64 s
    u64 gc_second = 0;
    std::mutex restore_mutex;
    std::vector<std::pair<ImageId, u64>> restore_full_quality; ///< image and its uid

    u64 load_window_bytes = 0;          ///< image upload bytes at the start of this second
    u64 load_window_reupload_bytes = 0; ///< of them re-uploads of freed images, at that start
    u64 load_burst_ignored_second = 0;
    u64 load_until_second = 0; ///< VRAM cleanup frees ahead of a load until this second
    u64 gc_loads = 0;
    bool spread_loads_enabled = false;
    std::atomic<u64> spread_frame_bytes{0};    ///< uploaded this game frame while spreading
    std::atomic<u64> detail_frame_bytes{0};    ///< texture bytes uploaded this game frame
    std::atomic<u32> detail_frame_restores{0}; ///< largest levels uploaded later this frame
    static inline thread_local bool sampled_update = false; ///< in UpdateTexture/UpdateImages
    /// Protection on a helper thread: the image being refreshed's deferred protection calls.
    static inline thread_local ProtectRanges* pending_protect = nullptr;
    u64 spread_window_second = 0, spread_window_uploads = 0, spread_window_bytes = 0;
    u64 spread_until_second = 0;
    std::unique_ptr<Buffer> zero_buffer; ///< zeros a new texture is filled from
    u64 spread_loads = 0, spread_deferred = 0, spread_zero_filled = 0, spread_zero_bytes = 0;
    u64 gc_evicted = 0, gc_written_back = 0, gc_skipped_stuck = 0, gc_skipped_written = 0;
    u64 gc_runs = 0;
    struct EvictedImage {
        std::chrono::steady_clock::time_point time;
        u64 size;
    };
    std::mutex evicted_mutex;
    std::unordered_map<VAddr, EvictedImage> recently_evicted; ///< freed by the cleanup, by address
    std::chrono::steady_clock::time_point cleanup_report_time{};
    std::chrono::steady_clock::time_point slow_pass_report_time{};
    CleanupCounters cleanup_last{};
    u64 cleanup_host_usage{};
    std::atomic<u64> pending_free_bytes{0}; ///< images deleted, released once the GPU is done
    std::chrono::steady_clock::time_point scratch_report_time{};
    /// "Use more VRAM": marks from the driver's live budget while on, the usual ones otherwise.
    bool more_vram_enabled = false;
    bool live_limits_active = false;
    bool standard_limits_saved = false;
    u64 standard_trigger = 0, standard_pressure = 0, standard_critical = 0;
    u64 live_limit = 0, logged_limit = 0;
    std::chrono::steady_clock::time_point limits_time{};
    std::chrono::steady_clock::time_point past_budget_report_time{};
    u64 past_budget_first_budget = 0, past_budget_first_host = 0;
    /// After a loading screen's cleanup, the next area's first uploads aren't an area load to
    /// free ahead of: there is room already.
    std::chrono::steady_clock::time_point free_ahead_paused_until{};
    /// VRAM breakdown report (every 10 s): gc_tick at each of the last 64 seconds, and uploads
    /// by size since the last report.
    std::chrono::steady_clock::time_point breakdown_time{};
    std::array<u64, 64> breakdown_tick_at_second{};
    u64 breakdown_second{};
    static constexpr size_t SizeClasses = 5;
    std::array<std::atomic<u64>, SizeClasses> upload_count_by_size{};
    std::array<std::atomic<u64>, SizeClasses> upload_bytes_by_size{};
    u32 quality_report_count{};
    /// Cleanup at loading screens: the one in progress.
    struct LoadScreen {
        bool active{};
        bool marked{}, images_done{}, buffers_done{}, reported{};
        std::chrono::steady_clock::time_point start{};
        u64 mark_tick{}; ///< images last used before this tick are the previous area's
        u64 start_vram{};
        u64 freed{}, freed_bytes{}, kept_written{}, kept_written_bytes{};
        u64 free_ns{}, passes{}, scratch_bytes{};
    } load_screen;
    bool load_screen_cleanup_enabled = false;
    Common::LRUCache<Image> image_lru_cache;
    Common::LRUCache<Sampler> sampler_lru_cache;
    const bool readback_linear_images;
    const bool vram_reclaim; ///< VRAM cleanup setting, read at start
    std::atomic<u64> registry_generation{0};
    std::atomic<u64> sampler_generation{0};
    std::mutex download_images_mutex;
    absl::flat_hash_map<VAddr, MetaDataInfo> surface_metas;
};

} // namespace VideoCore
