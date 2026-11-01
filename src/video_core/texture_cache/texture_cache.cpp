// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <bit>
#include <chrono>
#include <iterator>
#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <xxhash.h>

#include "common/assert.h"
#include "common/crash_crumbs.h"
#include "common/debug.h"
#include "common/div_ceil.h"
#include "common/game_waits.h"
#include "common/guest_stats.h"
#include "common/hash.h"
#include "common/scope_exit.h"
#include "core/bb_game_fixes.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/cache_memory_policy.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/host_compatibility.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/texture_cache/tile_manager.h"
#include "video_core/vram_owners.h"

namespace VideoCore {

static constexpr u32 MAX_IMAGES = std::numeric_limits<u16>::max();
static constexpr u32 MAX_IMAGE_VIEWS = std::numeric_limits<u16>::max();
static constexpr size_t MAX_GC_IMAGE_SCANS = 256;
static constexpr size_t MAX_GC_WRITEBACKS = 4;
static constexpr u64 WRITEBACK_RETRY_TICKS = 600;
// Write-back copies in flight at once (each holds a staging buffer the size of its image).
static constexpr size_t MAX_PENDING_WRITEBACKS = 8;
// Write-backs are GPU copies into system memory: started together (a cleanup pass over the
// pressure mark) they showed as 30-75 ms GPU waits while turning the camera. So they are paced:
// about 128 MiB a second, at most 16 MiB at once (an image larger than that goes on its own).
static constexpr u64 WRITEBACK_BYTES_PER_SECOND = 128_MB;
static constexpr u64 WRITEBACK_BURST_BYTES = 16_MB;
static constexpr u32 MAX_SAMPLERS = std::numeric_limits<u16>::max();

static u64 HashBackingMemory(VAddr virtual_addr, u64 size) {
    static constexpr std::array<u8, Core::GUEST_PAGE_SIZE> zeros{};
    auto* state = XXH3_createState();
    const auto& backing_pages = Core::Memory::Instance()->GetAddressSpace().BackingPages();
    XXH3_64bits_reset(state);
    while (size) {
        const u64 page = virtual_addr >> Core::GUEST_PAGE_BITS;
        const u64 offset_in_page = virtual_addr & (Core::GUEST_PAGE_SIZE - 1);
        const u64 hash_size = std::min<u64>(Core::GUEST_PAGE_SIZE - offset_in_page, size);
        if (auto* entry = backing_pages.find(page); entry && *entry) {
            XXH3_64bits_update(state, *entry + offset_in_page, hash_size);
        } else {
            XXH3_64bits_update(state, zeros.data(), hash_size);
        }
        size -= hash_size;
        virtual_addr += hash_size;
    }
    const u64 hash = XXH3_64bits_digest(state);
    XXH3_freeState(state);
    return hash;
}

TextureCache::TextureCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                           Vulkan::Runtime& runtime_, AmdGpu::Liverpool* liverpool_,
                           BufferCache& buffer_cache_, PageManager& page_manager_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_}, liverpool{liverpool_},
      buffer_cache{buffer_cache_}, page_manager{page_manager_}, slot_images{MAX_IMAGES},
      slot_image_views{MAX_IMAGE_VIEWS}, slot_samplers{MAX_SAMPLERS},
      blit_helper{instance, scheduler},
      tile_manager{instance, scheduler, runtime, buffer_cache.GetStreamBuffer()},
      readback_linear_images{EmulatorSettings.IsReadbackLinearImagesEnabled()},
      vram_reclaim{EmulatorSettings.IsBbVramReclaim()} {

    u32 max_samplers = instance.GetMaxSamplerAllocationCount();
    trigger_gc_samplers = max_samplers * 3 / 4;
    pressure_gc_samplers = max_samplers * 7 / 8;
    critical_gc_samplers = max_samplers * 15 / 16;

    // Set up garbage collection parameters.
    if (!instance.CanReportMemoryUsage()) {
        trigger_gc_memory = 0;
        pressure_gc_memory = DEFAULT_PRESSURE_GC_MEMORY;
        critical_gc_memory = DEFAULT_CRITICAL_GC_MEMORY;
        return;
    }

    const s64 device_local_memory = static_cast<s64>(instance.GetTotalMemoryBudget());
    const s64 min_spacing_expected = device_local_memory - 1_GB;
    const s64 min_spacing_critical = device_local_memory - 512_MB;
    const s64 mem_threshold = std::min<s64>(device_local_memory, TARGET_GC_THRESHOLD);
    const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
    const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
    const s64 min_pressure_floor =
        std::clamp<s64>(device_local_memory / 4, 256_MB, DEFAULT_PRESSURE_GC_MEMORY);
    const s64 min_critical_floor =
        std::clamp<s64>(device_local_memory / 2, 512_MB, DEFAULT_CRITICAL_GC_MEMORY);
    pressure_gc_memory = static_cast<u64>(
        std::max<s64>(std::min(device_local_memory - min_vacancy_expected, min_spacing_expected),
                      min_pressure_floor));
    critical_gc_memory = static_cast<u64>(
        std::max<s64>(std::min(device_local_memory - min_vacancy_critical, min_spacing_critical),
                      min_critical_floor));
    trigger_gc_memory = static_cast<u64>((device_local_memory - mem_threshold) / 2);

    // VRAM cleanup on large budgets: start writeback and eviction before the cache approaches
    // the full Vulkan budget. Smaller budgets already start early, and keep their thresholds.
    // This is a soft target: active or unsafe-to-download images remain.
    if (vram_reclaim && device_local_memory >= static_cast<s64>(LargeBudget)) {
        pressure_gc_memory = std::min(pressure_gc_memory, CachePressureTarget(device_local_memory));
        critical_gc_memory = std::min(critical_gc_memory, CacheCriticalTarget(device_local_memory));
        trigger_gc_memory = std::min(trigger_gc_memory, pressure_gc_memory);
    } else if (vram_reclaim) {
        // Smaller budgets (after bbport): the original marks (1.8 / 2.7 GB on a 6 GB card) sit
        // below what the game uses in normal play, so the collector stayed in pressure mode,
        // evicting textures used a second ago and loading them again (stutter, lower FPS).
        // Pressure at 85 % and critical at 95 % of the budget; below that, images go once
        // unused for GcIdleSeconds (GarbageCollectImages), which keeps VRAM from growing.
        pressure_gc_memory = static_cast<u64>(device_local_memory) / 100 * 85;
        critical_gc_memory = static_cast<u64>(device_local_memory) / 100 * 95;
        trigger_gc_memory = 0;
    }
    if (vram_reclaim) {
        LOG_INFO(Render,
                 "VRAM cleanup on. Image cache targets: scan={} MiB pressure={} MiB "
                 "critical={} MiB",
                 trigger_gc_memory >> 20, pressure_gc_memory >> 20, critical_gc_memory >> 20);
    }
}

TextureCache::~TextureCache() = default;

void TextureCache::ProcessDownloadImages() {
    std::unique_lock lk{download_images_mutex};
    for (const ImageId image_id : download_images) {
        DownloadImageMemory(image_id, DownloadMemoryFlags::Sync);
    }
    download_images.clear();
}

bool TextureCache::DownloadImageMemory(ImageId image_id, DownloadMemoryFlags flags) {
    Image& image = slot_images[image_id];
    // (Texture quality: an image without its largest level can't give the guest's contents.)
    if (False(image.flags & ImageFlagBits::GpuModified) || image.dropped_levels != 0) {
        return true;
    }
    const auto image_addr = image.info.guest_address;
    const auto image_mips = image.info.resources.levels;
    const auto is_sync = True(flags & DownloadMemoryFlags::Sync);
    const auto is_priority = True(flags & DownloadMemoryFlags::Priority);
    const auto discard_buffer_cache = True(flags & DownloadMemoryFlags::DiscardBufferCache);
    ASSERT(!(is_sync && is_priority));

    // Every level, re-tiled to the guest's layout where the image is tiled (0.21.0).
    u32 copy_size = 0;
    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    for (u32 mip = 0; mip < image_mips; ++mip) {
        const u32 width = std::max(image.info.size.width >> mip, 1u);
        const u32 height = std::max(image.info.size.height >> mip, 1u);
        const u32 depth =
            image.info.props.is_volume ? std::max(image.info.size.depth >> mip, 1u) : 1u;
        const auto [mip_size, mip_pitch, mip_height, mip_offset] = image.info.mips_layout[mip];
        const u32 extent_width = mip_pitch ? std::min<u32>(mip_pitch, width) : width;
        const u32 extent_height = mip_height ? std::min<u32>(mip_height, height) : height;
        buffer_copies.push_back(vk::BufferImageCopy{
            .bufferOffset = mip_offset,
            .bufferRowLength = mip_pitch,
            .bufferImageHeight = mip_height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = mip,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {extent_width, extent_height, depth},
        });
        copy_size += mip_size;
    }
    if (buffer_copies.empty()) {
        return true;
    }
    ASSERT(copy_size <= image.info.guest_size);
    const auto download =
        runtime.GetStagingPool().Request(copy_size, MemoryType::HostCached, 16, !is_sync);

    tile_manager.TileImage(image, buffer_copies, download.buffer, download.offset);

    const auto write_data = [this, image_addr, copy_size, download, discard_buffer_cache,
                             is_sync] {
        download.Invalidate();
        const bool written = Core::Memory::Instance()->TryWriteBacking(
            std::bit_cast<u8*>(image_addr), download.mapped, copy_size);
        if (written && discard_buffer_cache) {
            buffer_cache.UnmarkRegionAsGpuModified(image_addr, copy_size, true);
        }
        if (!is_sync) {
            runtime.GetStagingPool().FreeDeferred(download);
        }
        return written;
    };

    if (is_sync) {
        scheduler.Finish();
        return write_data();
    }
    if (is_priority) {
        scheduler.DeferPriorityOperation([write_data] { write_data(); });
    } else {
        scheduler.DeferOperation([write_data] { write_data(); });
    }
    return true;
}

// VRAM cleanup: an image may be written back over its guest memory only while that memory still
// belongs to it. A CPU write next to the image untracks its first or last page without marking it
// dirty, after which writes inside the image there go unseen; a GPU buffer write that doesn't
// start at the image's address doesn't mark it either.
bool TextureCache::SafeToWriteBack(const Image& image) {
    const VAddr begin = image.info.guest_address;
    const VAddr end = begin + image.info.guest_size;
    if (image.track_addr != begin || image.track_addr_end != end) {
        return false;
    }
    return !buffer_cache.IsRegionGpuModified(begin, image.info.guest_size);
}

// VRAM cleanup: records the copy of a GPU-modified image (tiled ones re-tiled to their guest
// layout) into a staging buffer, to be written to guest memory and the image freed once the GPU
// has done it. False when the image can't be written back completely.
bool TextureCache::BeginWriteBack(ImageId image_id) {
    Image& image = slot_images[image_id];
    if (False(image.flags & ImageFlagBits::GpuModified) || image.dropped_levels != 0) {
        return false;
    }
    const auto& info = image.info;
    if (info.guest_address == 0 || info.guest_size == 0 || info.resources.levels == 0 ||
        info.resources.levels > info.mips_layout.size() || info.resources.layers == 0) {
        return false;
    }
    if (image.info.props.is_tiled) {
        const auto& meta = info.meta_info;
        if (meta.cmask_addr || meta.fmask_addr || meta.htile_addr || info.stencil_addr ||
            info.props.is_depth || info.props.has_stencil || info.num_samples != 1) {
            return false;
        }
        switch (info.num_bits) {
        case 8:
        case 16:
        case 32:
        case 64:
        case 96:
        case 128:
            break;
        default:
            return false;
        }
        // The tiling shader dispatches whole groups of 64 texels.
        if (info.guest_size % ((info.num_bits / 8) * 64) != 0) {
            return false;
        }
        // Validate the complete mip layout before acquiring a writable buffer.
        for (u32 mip = 0; mip < info.resources.levels; ++mip) {
            const auto& mip_info = info.mips_layout[mip];
            if (mip_info.size == 0 || mip_info.pitch == 0 || mip_info.height == 0 ||
                mip_info.offset > info.guest_size ||
                mip_info.size > info.guest_size - mip_info.offset) {
                return false;
            }
        }
        boost::container::small_vector<vk::BufferImageCopy, 14> image_copies;
        for (u32 mip = 0; mip < image.info.resources.levels; ++mip) {
            const auto& mip_info = image.info.mips_layout[mip];
            image_copies.push_back({
                .bufferOffset = mip_info.offset,
                .bufferRowLength = mip_info.pitch,
                .bufferImageHeight = mip_info.height,
                .imageSubresource =
                    {
                        .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                        .mipLevel = mip,
                        .baseArrayLayer = 0,
                        .layerCount = image.info.resources.layers,
                    },
                .imageOffset = {0, 0, 0},
                .imageExtent = {std::max(image.info.size.width >> mip, 1u),
                                std::max(image.info.size.height >> mip, 1u),
                                image.info.props.is_volume
                                    ? std::max(image.info.size.depth >> mip, 1u)
                                    : 1u},
            });
        }
        if (image_copies.empty()) {
            return false;
        }
        // Tiled straight into the staging buffer (the tiling shader writes it as a storage
        // buffer), so guest buffers in the GPU aren't touched.
        const auto download =
            runtime.GetStagingPool().Request(info.guest_size, MemoryType::HostCached, 256, true);
        tile_manager.TileImage(image, image_copies, download.buffer, download.offset);
        QueueWriteBack(image_id, image, download, info.guest_size);
        return true;
    }
    // The existing linear path copies only mip zero and one aspect. Do not
    // evict multi-mip or multi-aspect images through an incomplete writeback.
    if ((info.resources.levels != 1 || info.props.has_stencil || info.stencil_addr ||
         info.props.is_block || info.num_samples != 1)) {
        return false;
    }
    const u64 download_size = u64{info.pitch} * info.size.height * info.size.depth *
                              info.resources.layers * (info.num_bits / 8);
    if (download_size == 0 || download_size > info.guest_size) {
        return false;
    }
    const auto download =
        runtime.GetStagingPool().Request(download_size, MemoryType::HostCached, 16, true);
    const vk::BufferImageCopy image_download = {
        .bufferOffset = download.offset,
        .bufferRowLength = image.info.pitch,
        .bufferImageHeight = image.info.size.height,
        .imageSubresource =
            {
                .aspectMask = image.info.props.is_depth ? vk::ImageAspectFlagBits::eDepth
                                                        : vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
        .imageOffset = {0, 0, 0},
        .imageExtent = {image.info.size.width, image.info.size.height, image.info.size.depth},
    };
    runtime.DownloadImage(&image, download.buffer, std::span{&image_download, 1});
    QueueWriteBack(image_id, image, download, download_size);
    return true;
}

void TextureCache::QueueWriteBack(ImageId image_id, const Image& image,
                                  const Vulkan::StagingBufferRef& download, u64 size) {
    pending_writebacks.push_back({
        .image_id = image_id,
        .image_uid = image.image_uid,
        .last_use = image.last_use_gc_tick,
        .tick = scheduler.CurrentTick(),
        .address = image.info.guest_address,
        .size = size,
        .download = download,
    });
}

// VRAM cleanup: finishes the write-backs whose copies the GPU has done. The image is freed only
// if nothing used or changed it since its copy was recorded; otherwise the copy is dropped.
void TextureCache::CompleteWriteBacks() {
    if (pending_writebacks.empty()) {
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    SCOPE_EXIT {
        Common::GuestStats::gc_writeback_done_ns.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                                 start)
                .count(),
            std::memory_order_relaxed);
    };
    auto pending = std::move(pending_writebacks);
    pending_writebacks.clear();
    for (PendingWriteBack& writeback : pending) {
        if (!scheduler.IsFree(writeback.tick)) {
            pending_writebacks.push_back(writeback);
            continue;
        }
        bool written = false;
        if (slot_images.IsAllocated(writeback.image_id)) {
            Image& image = slot_images[writeback.image_id];
            if (image.image_uid == writeback.image_uid &&
                image.last_use_gc_tick == writeback.last_use && image.HasGpuValidData() &&
                SafeToWriteBack(image)) {
                writeback.download.Invalidate();
                // Like a CPU write: guest buffers in the GPU holding this range upload it again.
                buffer_cache.InvalidateMemory(writeback.address, writeback.size);
                written = Core::Memory::Instance()->TryWriteBacking(
                    std::bit_cast<u8*>(writeback.address), writeback.download.mapped,
                    writeback.size);
                if (written) {
                    ++gc_written_back;
                    ++gc_evicted;
                    Common::GuestStats::gc_writeback_written_bytes.fetch_add(
                        writeback.size, std::memory_order_relaxed);
                    NoteEvicted(image);
                    FreeImage(writeback.image_id);
                } else {
                    writeback_retry[writeback.image_id.index] = gc_tick + WRITEBACK_RETRY_TICKS;
                }
            }
        }
        runtime.GetStagingPool().FreeDeferred(writeback.download);
    }
}

// Upload statistics: the first moment since its last upload that the image became stale.
static void NoteDirty(Image& image, Common::GuestStats::DirtySource source) {
    if (image.dirty_since_ns == 0) {
        image.dirty_since_ns = static_cast<u64>(
            std::chrono::steady_clock::now().time_since_epoch() / std::chrono::nanoseconds{1});
        image.dirty_source = static_cast<u8>(source);
    }
}

void TextureCache::MarkAsMaybeDirty(ImageId image_id, Image& image) {
    NoteDirty(image, Common::GuestStats::DirtySource::MaybeCpu);
    if (image.cpu_hash == 0) {
        // Initialize hash
        const u8* addr = std::bit_cast<u8*>(image.info.guest_address);
        image.cpu_hash = XXH3_64bits(addr, image.info.guest_size);
    }
    image.flags |= ImageFlagBits::MaybeCpuDirty;
    ClearCleanTick(image);
    UntrackImage(image_id);
}

void TextureCache::InvalidateMemory(VAddr addr, size_t size) {
    const auto pages_start = PageManager::GetPageAddr(addr);
    const auto pages_end = PageManager::GetNextPageAddr(addr + size - 1);

    SmallVector<ImageId, 8> image_ids;
    ForEachImageInRegion(pages_start, pages_end - pages_start,
                         [&](ImageId image_id, Image&) { image_ids.push_back(image_id); });

    for (const auto image_id : image_ids) {
        Image& image = slot_images[image_id];
        std::unique_lock lk{image.mutex, std::defer_lock};
        {
            const Common::GameWaits::Scope wait{"write fault: image lock"};
            lk.lock();
        }

        const auto image_begin = image.info.guest_address;
        const auto image_end = image.info.guest_address + image.info.guest_size;
        if (image.Overlaps(addr, size)) {
            // Modified region overlaps image, so the image was definitely accessed by this fault.
            // Untrack the image, so that the range is unprotected and the guest can write freely.
            image.flags |= ImageFlagBits::CpuDirty;
            NoteDirty(image, Common::GuestStats::DirtySource::Cpu);
            ClearCleanTick(image);
            UntrackImage(image_id);
        } else if (pages_end < image_end) {
            // This page access may or may not modify the image.
            // We should not mark it as dirty now. If it really was modified
            // it will receive more invalidations on its other pages.
            // Remove tracking from this page only.
            UntrackImageHead(image_id);
        } else if (image_begin < pages_start) {
            // This page access does not modify the image but the page should be untracked.
            // We should not mark this image as dirty now. If it really was modified
            // it will receive more invalidations on its other pages.
            UntrackImageTail(image_id);
        } else {
            // Image begins and ends on this page so it can not receive any more invalidations.
            // We will check it's hash later to see if it really was modified.
            MarkAsMaybeDirty(image_id, image);
        }
    }
}

void TextureCache::InvalidateMemoryFromGPU(VAddr address, size_t max_size) {
    ForEachImageInRegion(address, max_size, [&](ImageId image_id, Image& image) {
        // Only consider images that match base address.
        // TODO: Maybe also consider subresources
        if (image.info.guest_address != address) {
            return;
        }
        // Ensure image is reuploaded when accessed again.
        image.flags |= ImageFlagBits::GpuDirty;
        NoteDirty(image, Common::GuestStats::DirtySource::Gpu);
        ClearCleanTick(image);
    });
}

void TextureCache::MarkAsMaybeGpuDirty(VAddr addr, size_t size) {
    ForEachImageInRegion(addr, size, [&](ImageId image_id, Image& image) {
        if (image.gpu_hash == 0) {
            // Read backing pages directly, without faulting through the guest readback path.
            image.gpu_hash = HashBackingMemory(image.info.guest_address, image.info.guest_size);
        }
    });
}

void TextureCache::UnmapMemory(VAddr cpu_addr, size_t size) {
    SmallVector<ImageId, 16> deleted_images;
    ForEachImageInRegion(cpu_addr, size, [&](ImageId id, Image&) { deleted_images.push_back(id); });
    for (const ImageId id : deleted_images) {
        // TODO: Download image data back to host.
        FreeImage(id);
    }
}

ImageId TextureCache::ResolveDepthOverlap(const ImageInfo& requested_info, BindingType binding,
                                          ImageId cache_image_id) {
    auto& cache_image = slot_images[cache_image_id];

    if (!cache_image.info.props.is_depth && !requested_info.props.is_depth) {
        return {};
    }

    const bool stencil_match =
        requested_info.props.has_stencil == cache_image.info.props.has_stencil;
    const bool bpp_match = requested_info.num_bits == cache_image.info.num_bits;

    // If an image in the cache has less slices we need to expand it
    bool recreate = !cache_image.info.resources.Contains(requested_info.resources);

    switch (binding) {
    case BindingType::Texture:
        // The guest requires a depth sampled texture, but cache can offer only Rxf. Need to
        // recreate the image.
        recreate |= requested_info.props.is_depth && !cache_image.info.props.is_depth;
        break;
    case BindingType::Storage:
        // If the guest is going to use previously created depth as storage, the image needs to be
        // recreated. (TODO: Probably a case with linear rgba8 aliasing is legit)
        recreate |= cache_image.info.props.is_depth;
        break;
    case BindingType::RenderTarget:
        // Render target can have only Rxf format. If the cache contains only Dx[S8] we need to
        // re-create the image.
        ASSERT(!requested_info.props.is_depth);
        recreate |= cache_image.info.props.is_depth;
        break;
    case BindingType::DepthTarget:
        // The guest has requested previously allocated texture to be bound as a depth target.
        // In this case we need to convert Rx float to a Dx[S8] as requested
        recreate |= !cache_image.info.props.is_depth;

        // The guest is trying to bind a depth target and cache has it. Need to be sure that aspects
        // and bpp match
        recreate |= cache_image.info.props.is_depth && !(stencil_match && bpp_match);
        break;
    default:
        break;
    }

    if (recreate) {
        auto new_info = requested_info;
        new_info.resources = std::max(requested_info.resources, cache_image.info.resources);
        new_info.UpdateSize();
        EnsureFullDetail(cache_image);
        const auto new_image_id = slot_images.Insert(instance, runtime, slot_image_views, new_info);
        RegisterImage(new_image_id);

        // Inherit image usage
        auto& new_image = slot_images[new_image_id];
        new_image.usage = cache_image.usage;
        if (new_info.num_samples == 1 &&
            (new_info.resources.layers > cache_image.info.resources.layers ||
             new_info.resources.levels > cache_image.info.resources.levels)) {
            Common::GuestStats::UploadPathScope path{Common::GuestStats::UploadPath::Overlap};
            RefreshImage(new_image);
        }
        new_image.flags &= ~ImageFlagBits::Dirty;
        // When creating a depth buffer through overlap resolution don't clear it on first use.
        new_image.info.meta_info.htile_clear_mask = 0;
        runtime.CopyColorAndDepth(&cache_image, &new_image);

        // Free the cache image.
        FreeImage(cache_image_id);
        return new_image_id;
    }

    // Will be handled by view
    return cache_image_id;
}

std::tuple<ImageId, int, int> TextureCache::ResolveOverlap(const ImageInfo& image_info,
                                                           BindingType binding,
                                                           ImageId cache_image_id,
                                                           ImageId merged_image_id) {
    static constexpr u64 NUM_FRAMES_BEFORE_REMOVAL = 32;

    auto& cache_image = slot_images[cache_image_id];
    const bool safe_to_delete =
        scheduler.CurrentTick() - cache_image.tick_accessed_last > NUM_FRAMES_BEFORE_REMOVAL;

    // Equal address
    if (image_info.guest_address == cache_image.info.guest_address) {
        const u32 lhs_block_size = image_info.num_bits * image_info.num_samples;
        const u32 rhs_block_size = cache_image.info.num_bits * cache_image.info.num_samples;
        if (image_info.BlockDim() != cache_image.info.BlockDim() ||
            lhs_block_size != rhs_block_size) {
            // Very likely this kind of overlap is caused by allocation from a pool.
            if (safe_to_delete) {
                FreeImage(cache_image_id);
            }
            return {merged_image_id, -1, -1};
        }

        if (const auto depth_image_id = ResolveDepthOverlap(image_info, binding, cache_image_id)) {
            return {depth_image_id, -1, -1};
        }

        // Compressed view of uncompressed image with same block size.
        if (image_info.props.is_block && !cache_image.info.props.is_block) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        if (image_info.guest_size == cache_image.info.guest_size &&
            (image_info.type == AmdGpu::ImageType::Color3D ||
             cache_image.info.type == AmdGpu::ImageType::Color3D)) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        const bool pow2_padding_only =
            image_info.props.is_pow2 != cache_image.info.props.is_pow2 &&
            image_info.tile_mode == cache_image.info.tile_mode &&
            image_info.size == cache_image.info.size &&
            image_info.pitch == cache_image.info.pitch && image_info.resources.levels == 1 &&
            cache_image.info.resources.levels == 1 && image_info.resources.layers == 1 &&
            cache_image.info.resources.layers == 1;

        // Size and resources are less than or equal, use image view.
        if (image_info.pixel_format != cache_image.info.pixel_format ||
            image_info.guest_size <= cache_image.info.guest_size || pow2_padding_only) {
            auto result_id = merged_image_id ? merged_image_id : cache_image_id;
            const auto& result_image = slot_images[result_id];
            const bool is_compatible =
                IsVulkanFormatCompatible(result_image.info.pixel_format, image_info.pixel_format);
            return {is_compatible ? result_id : ImageId{}, -1, -1};
        }

        // Size and resources are greater, expand the image.
        if (image_info.type == cache_image.info.type &&
            !cache_image.info.resources.Contains(image_info.resources)) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        // Size is greater but resources are not, because the tiling mode is different.
        // Likely the address is reused for a image with a different tiling mode.
        if (image_info.tile_mode != cache_image.info.tile_mode) {
            if (safe_to_delete) {
                FreeImage(cache_image_id);
            }
            return {merged_image_id, -1, -1};
        }

        // Enhanced debug logging for unreachable case
        // Calculate expected size based on format and dimensions
        u64 expected_size =
            (static_cast<u64>(image_info.size.width) * static_cast<u64>(image_info.size.height) *
             static_cast<u64>(image_info.size.depth) * static_cast<u64>(image_info.num_bits) / 8);
        LOG_ERROR(Render_Vulkan,
                  "Unresolvable image overlap with equal memory address:\n"
                  "=== OLD IMAGE (cached) ===\n"
                  "  Address:        {:#x}\n"
                  "  Size:           {:#x} bytes\n"
                  "  Format:         {}\n"
                  "  Type:           {}\n"
                  "  Width:          {}\n"
                  "  Height:         {}\n"
                  "  Depth:          {}\n"
                  "  Pitch:          {}\n"
                  "  Mip levels:     {}\n"
                  "  Array layers:   {}\n"
                  "  Samples:        {}\n"
                  "  Tile mode:      {:#x}\n"
                  "  Block size:     {} bits\n"
                  "  Is block-comp:  {}\n"
                  "  Guest size:     {:#x}\n"
                  "  Last accessed:  tick {}\n"
                  "  Safe to delete: {}\n"
                  "  isPow2:         {}\n"
                  "  Alt tile:       {}\n"
                  "\n"
                  "=== NEW IMAGE (requested) ===\n"
                  "  Address:        {:#x}\n"
                  "  Size:           {:#x} bytes\n"
                  "  Format:         {}\n"
                  "  Type:           {}\n"
                  "  Width:          {}\n"
                  "  Height:         {}\n"
                  "  Depth:          {}\n"
                  "  Pitch:          {}\n"
                  "  Mip levels:     {}\n"
                  "  Array layers:   {}\n"
                  "  Samples:        {}\n"
                  "  Tile mode:      {:#x}\n"
                  "  Block size:     {} bits\n"
                  "  Is block-comp:  {}\n"
                  "  Guest size:     {:#x}\n"
                  "  isPow2:         {}\n"
                  "  Alt tile:       {}\n"
                  "\n"
                  "=== COMPARISON ===\n"
                  "  Same format:           {}\n"
                  "  Same type:             {}\n"
                  "  Same tile mode:        {}\n"
                  "  Same block size:       {}\n"
                  "  Same BlockDim:         {}\n"
                  "  Same pitch:            {}\n"
                  "  Same pow2:             {}\n"
                  "  Same alt tile:         {}\n"
                  "  Old resources <= new:  {} (old: {}, new: {})\n"
                  "  Old size <= new size:  {}\n"
                  "  Expected size (calc):  {} bytes\n"
                  "  Size ratio (new/expected): {:.2f}x\n"
                  "  Size ratio (new/old):  {:.2f}x\n"
                  "  Old vs expected diff:  {} bytes ({:+.2f}%)\n"
                  "  New vs expected diff:  {} bytes ({:+.2f}%)\n"
                  "  Merged image ID:       {}\n"
                  "  Binding type:          {}\n"
                  "  Current tick:          {}\n"
                  "  Age (ticks since last access): {}",

                  // Old image details
                  cache_image.info.guest_address, cache_image.info.guest_size,
                  vk::to_string(cache_image.info.pixel_format),
                  static_cast<int>(cache_image.info.type), cache_image.info.size.width,
                  cache_image.info.size.height, cache_image.info.size.depth, cache_image.info.pitch,
                  cache_image.info.resources.levels, cache_image.info.resources.layers,
                  cache_image.info.num_samples, static_cast<u32>(cache_image.info.tile_mode),
                  cache_image.info.num_bits, +cache_image.info.props.is_block,
                  cache_image.info.guest_size, cache_image.tick_accessed_last, safe_to_delete,
                  bool(cache_image.info.props.is_pow2), cache_image.info.alt_tile,

                  // New image details
                  image_info.guest_address, image_info.guest_size,
                  vk::to_string(image_info.pixel_format), static_cast<int>(image_info.type),
                  image_info.size.width, image_info.size.height, image_info.size.depth,
                  image_info.pitch, image_info.resources.levels, image_info.resources.layers,
                  image_info.num_samples, static_cast<u32>(image_info.tile_mode),
                  image_info.num_bits, image_info.props.is_block, image_info.guest_size,
                  bool(image_info.props.is_pow2), image_info.alt_tile,

                  // Comparison
                  (image_info.pixel_format == cache_image.info.pixel_format),
                  (image_info.type == cache_image.info.type),
                  (image_info.tile_mode == cache_image.info.tile_mode),
                  (image_info.num_bits == cache_image.info.num_bits),
                  (image_info.BlockDim() == cache_image.info.BlockDim()),
                  (image_info.pitch == cache_image.info.pitch),
                  (image_info.props.is_pow2 == cache_image.info.props.is_pow2),
                  (image_info.alt_tile == cache_image.info.alt_tile),
                  (cache_image.info.resources <= image_info.resources),
                  cache_image.info.resources.levels, image_info.resources.levels,
                  (cache_image.info.guest_size <= image_info.guest_size), expected_size,

                  // Size ratios
                  static_cast<double>(image_info.guest_size) / expected_size,
                  static_cast<double>(image_info.guest_size) / cache_image.info.guest_size,

                  // Difference between actual and expected sizes with percentages
                  static_cast<s64>(cache_image.info.guest_size) - static_cast<s64>(expected_size),
                  (static_cast<double>(cache_image.info.guest_size) / expected_size - 1.0) * 100.0,

                  static_cast<s64>(image_info.guest_size) - static_cast<s64>(expected_size),
                  (static_cast<double>(image_info.guest_size) / expected_size - 1.0) * 100.0,

                  merged_image_id.index, static_cast<int>(binding), scheduler.CurrentTick(),
                  scheduler.CurrentTick() - cache_image.tick_accessed_last);

        UNREACHABLE_MSG("Encountered unresolvable image overlap with equal memory address.");
    }

    // Right overlap, the image requested is a possible subresource of the image from cache.
    if (image_info.guest_address > cache_image.info.guest_address) {
        if (auto mip = image_info.MipOf(cache_image.info); mip >= 0) {
            if (auto slice = image_info.SliceOf(cache_image.info, mip); slice >= 0) {
                return {cache_image_id, mip, slice};
            }
        }

        // Image isn't a subresource but a chance overlap.
        if (safe_to_delete) {
            FreeImage(cache_image_id);
        }

        return {{}, -1, -1};
    } else {
        // Left overlap, the image from cache is a possible subresource of the image requested
        if (auto mip = cache_image.info.MipOf(image_info); mip >= 0) {
            if (auto slice = cache_image.info.SliceOf(image_info, mip); slice >= 0) {
                // We have a larger image created and a separate one, representing a subres of it
                // bound as render target. In this case we need to rebind render target.
                if (cache_image.binding.is_target) {
                    cache_image.binding.needs_rebind = 1u;
                    if (merged_image_id) {
                        GetImage(merged_image_id).binding.is_target = 1u;
                    }

                    FreeImage(cache_image_id);
                    return {merged_image_id, -1, -1};
                }

                // If we don't have an allocated image, we create one here
                // Otherwise it's later created from guest memory and the GPU data is lost
                // And reads/writes go to different images.
                if (!merged_image_id) {
                    merged_image_id =
                        slot_images.Insert(instance, runtime, slot_image_views, image_info);
                    RegisterImage(merged_image_id);
                    RefreshImage(slot_images[merged_image_id]);
                    TrackImage(merged_image_id);
                }
                auto& merged_image = slot_images[merged_image_id];
                EnsureFullDetail(cache_image);
                runtime.CopyMip(&cache_image, &merged_image, mip, slice);
                FreeImage(cache_image_id);
            }
        }
    }

    return {merged_image_id, -1, -1};
}

ImageId TextureCache::ExpandImage(const ImageInfo& info, ImageId image_id) {
    const auto new_image_id = slot_images.Insert(instance, runtime, slot_image_views, info);
    RegisterImage(new_image_id);

    auto& src_image = slot_images[image_id];
    auto& new_image = slot_images[new_image_id];

    EnsureFullDetail(src_image);
    Common::GuestStats::UploadPathScope path{Common::GuestStats::UploadPath::Overlap};
    // Only carry over the old contents when they hold GPU-written data that guest memory lacks.
    // A CPU-dirty source is stale (e.g. a streamed texture rewritten in a reused pool slot);
    // copying it would overwrite the fresh upload with old data.
    const bool src_has_gpu_data = src_image.HasGpuValidData();
    RefreshImage(new_image);

    if (src_image.binding.is_bound || src_image.binding.is_target) {
        src_image.binding.needs_rebind = 1u;
    }

    TrackImage(new_image_id);
    new_image.flags &= ~ImageFlagBits::Dirty;
    if (src_has_gpu_data) {
        runtime.CopyImage(&src_image, &new_image);
        // The new image holds the old one's GPU-written contents now: the VRAM cleanup must write
        // them back before freeing it, not reload stale guest memory.
        new_image.flags |= src_image.flags & ImageFlagBits::GpuModified;
    }
    FreeImage(image_id);
    return new_image_id;
}

ImageId TextureCache::FindImage(ImageDesc& desc, bool exact_fmt) {
    const auto& info = desc.info;

    ASSERT(info.guest_address != 0);

    SmallVector<ImageId, 8> image_ids;
    ForEachImageInRegion(info.guest_address, info.guest_size,
                         [&](ImageId image_id, Image& image) { image_ids.push_back(image_id); });

    ImageId image_id{};

    // Check for a perfect match first
    for (const auto& cache_id : image_ids) {
        auto& cache_image = slot_images[cache_id];
        if (cache_image.info.guest_address != info.guest_address) {
            continue;
        }
        if (cache_image.info.guest_size != info.guest_size) {
            continue;
        }
        if (cache_image.info.size != info.size) {
            continue;
        }
        if (!IsVulkanFormatCompatible(cache_image.info.pixel_format, info.pixel_format) ||
            (cache_image.info.type != info.type && info.size != Extent3D{1, 1, 1})) {
            continue;
        }
        if (exact_fmt && info.pixel_format != cache_image.info.pixel_format) {
            continue;
        }
        image_id = cache_id;
    }

    // Try to resolve overlaps (if any)
    int view_mip{-1};
    int view_slice{-1};
    if (!image_id) {
        for (const auto& cache_id : image_ids) {
            view_mip = -1;
            view_slice = -1;

            const auto& merged_info = image_id ? slot_images[image_id].info : info;
            auto [overlap_image_id, overlap_view_mip, overlap_view_slice] =
                ResolveOverlap(merged_info, desc.type, cache_id, image_id);
            if (overlap_image_id) {
                image_id = overlap_image_id;
                view_mip = overlap_view_mip;
                view_slice = overlap_view_slice;
            }
        }
    }

    if (image_id) {
        Image& image_resolved = slot_images[image_id];
        if (exact_fmt && info.pixel_format != image_resolved.info.pixel_format) {
            // Cannot reuse this image as we need the exact requested format.
            image_id = {};
        } else if (!image_resolved.info.resources.Contains(info.resources)) {
            // The image was clearly picked up wrong.
            FreeImage(image_id);
            image_id = {};
            LOG_WARNING(Render_Vulkan, "Image overlap resolve failed");
        }
    }
    // Create and register a new image
    if (!image_id) {
        Common::GuestStats::DetailTimer timer{Common::GuestStats::Detail::TexCreate};
        image_id = slot_images.Insert(instance, runtime, slot_image_views, info);
        RegisterImage(image_id);
    }

    Image& image = slot_images[image_id];
    image.tick_accessed_last = scheduler.CurrentTick();
    TouchImage(image);

    // If the image requested is a subresource of the image from cache record its location.
    if (view_mip > 0) {
        desc.view_info.range.base.level = view_mip;
    }
    if (view_slice > 0) {
        desc.view_info.range.base.layer = view_slice;
    }

    return image_id;
}

ImageId TextureCache::FindImageFromRange(VAddr address, size_t size, bool ensure_valid) {
    SmallVector<ImageId, 4> image_ids;
    ForEachImageWithAddress(address, [&](ImageId image_id, Image& image) {
        if (ensure_valid && !image.HasGpuValidData()) {
            return;
        }
        image_ids.push_back(image_id);
    });
    if (image_ids.size() == 1) {
        // Sometimes image size might not exactly match with requested buffer size
        // If we only found 1 candidate image use it without too many questions.
        Image& image = slot_images[image_ids[0]];
        TouchImage(image);
        return image_ids.back();
    }
    if (!image_ids.empty()) {
        for (s32 i = 0; i < image_ids.size(); ++i) {
            Image& image = slot_images[image_ids[i]];
            if (image.info.guest_size == size) {
                TouchImage(image);
                return image_ids[i];
            }
        }
        LOG_WARNING(Render_Vulkan,
                    "Failed to find exact image match for copy addr={:#x}, size={:#x}", address,
                    size);
    }
    return {};
}

ImageId TextureCache::FindImageLayer(VAddr address, size_t size, u32& layer) {
    // (After bbport 0.7: the point lights' shadow cube is cleared a face at a time.)
    ImageId found{};
    ForEachImageInRegion(address, size, [&](ImageId image_id, Image& image) {
        const auto& info = image.info;
        const u32 layers = info.resources.layers;
        if (found || image.dropped_levels != 0 || info.resources.levels != 1 || layers < 2 ||
            info.guest_size % layers != 0 || address < info.guest_address) {
            return;
        }
        const u64 layer_size = info.guest_size / layers;
        const u64 offset = address - info.guest_address;
        if (size == layer_size && offset % layer_size == 0 && offset / layer_size < layers) {
            layer = u32(offset / layer_size);
            found = image_id;
        }
    });
    return found;
}

ImageView& TextureCache::FindTexture(ImageId image_id, const ImageDesc& desc) {
    Image& image = slot_images[image_id];
    if (desc.type == BindingType::Storage) {
        Common::GuestStats::UploadPathScope path{Common::GuestStats::UploadPath::Storage};
        image.flags |= ImageFlagBits::GpuModified;
        if (readback_linear_images && (!image.info.props.is_tiled || image.info.size.width <= 8) &&
            image.info.guest_address != 0) {
            std::unique_lock lk{download_images_mutex};
            download_images.emplace(image_id);
        }
        EnsureFullDetail(image);
        UpdateImage(image_id);
    } else {
        UpdateTexture(image_id);
        if (image.detail_pending) {
            // Lower detail first: sampled from the next level down until the largest arrives
            // (a view of the largest level alone, or exact texel reads, need it now).
            const SubresourceRange host = image.HostRange(desc.view_info.range);
            if (host.base.level == 0 && host.extent.levels >= 2 &&
                std::atomic_ref<u8>(image.exact_texel_reads).load(std::memory_order_relaxed) == 0) {
                ImageViewInfo soft = desc.view_info;
                soft.min_lod = std::max(soft.min_lod, (image.dropped_levels + 1) * 256);
                Common::GuestStats::DetailTimer timer{Common::GuestStats::Detail::TexView};
                return image.FindView(soft);
            }
            if (host.base.level == 0) {
                EnsureFullDetail(image);
            }
        }
    }
    Common::GuestStats::DetailTimer timer{Common::GuestStats::Detail::TexView};
    return image.FindView(desc.view_info);
}

void TextureCache::RegisterMeta(VAddr address, MetaDataInfo info) {
    const auto it = surface_metas.find(address);
    if (it == surface_metas.end()) {
        surface_metas.emplace(address, info);
        return;
    }

    auto& registered = it->second;
    if (registered.type != info.type || registered.owner_uid != info.owner_uid) {
        registered = info;
        return;
    }

    // Preserve the current clear state when the same render target is rebound.
    registered.owner_id = info.owner_id;
    registered.num_samples = info.num_samples;
}

void TextureCache::UnregisterMeta(VAddr address, u64 owner_uid) {
    const auto it = surface_metas.find(address);
    if (it != surface_metas.end() && it->second.owner_uid == owner_uid) {
        surface_metas.erase(address);
    }
}

ImageView& TextureCache::FindRenderTarget(ImageId image_id, const ImageDesc& desc) {
    Image& image = slot_images[image_id];
    Common::GuestStats::UploadPathScope path{Common::GuestStats::UploadPath::Target};
    EnsureFullDetail(image);
    image.flags |= ImageFlagBits::GpuModified;
    if (readback_linear_images && (!image.info.props.is_tiled || image.info.size.width <= 8)) {
        std::unique_lock lk{download_images_mutex};
        download_images.emplace(image_id);
    }
    image.usage.render_target = 1u;
    UpdateImage(image_id);

    // Register meta data for this color buffer
    const VAddr cmask_addr = desc.info.meta_info.cmask_addr;
    if (image.info.meta_info.cmask_addr != cmask_addr) {
        UnregisterMeta(image.info.meta_info.cmask_addr, image.image_uid);
        image.info.meta_info.cmask_addr = cmask_addr;
    }
    if (cmask_addr) {
        RegisterMeta(cmask_addr, MetaDataInfo{
                                     .type = MetaType::CMask,
                                     .owner_id = image_id,
                                     .owner_uid = image.image_uid,
                                     .num_samples = image.info.num_samples,
                                 });
    }

    const VAddr fmask_addr = desc.info.meta_info.fmask_addr;
    if (image.info.meta_info.fmask_addr != fmask_addr) {
        UnregisterMeta(image.info.meta_info.fmask_addr, image.image_uid);
        image.info.meta_info.fmask_addr = fmask_addr;
    }
    if (fmask_addr) {
        RegisterMeta(fmask_addr, MetaDataInfo{
                                     .type = MetaType::FMask,
                                     .owner_id = image_id,
                                     .owner_uid = image.image_uid,
                                     .num_samples = image.info.num_samples,
                                 });
    }

    return image.FindView(desc.view_info, false);
}

ImageView& TextureCache::FindDepthTarget(ImageId image_id, const ImageDesc& desc) {
    Image& image = slot_images[image_id];
    Common::GuestStats::UploadPathScope path{Common::GuestStats::UploadPath::Target};
    EnsureFullDetail(image);
    image.flags |= ImageFlagBits::GpuModified;
    image.usage.depth_target = 1u;
    UpdateImage(image_id);

    // Register meta data for this depth buffer
    const VAddr htile_addr = desc.info.meta_info.htile_addr;
    if (image.info.meta_info.htile_addr != htile_addr) {
        UnregisterMeta(image.info.meta_info.htile_addr, image.image_uid);
        image.info.meta_info.htile_addr = htile_addr;
    }
    if (htile_addr) {
        RegisterMeta(htile_addr, MetaDataInfo{
                                     .type = MetaType::HTile,
                                     .owner_id = image_id,
                                     .owner_uid = image.image_uid,
                                     .num_samples = image.info.num_samples,
                                     .clear_mask = image.info.meta_info.htile_clear_mask,
                                 });
    }

    AssociateStencil(image_id, desc.info);

    return image.FindView(desc.view_info, false);
}

void TextureCache::AssociateStencil(ImageId depth_id, const ImageInfo& depth_info) {
    if (depth_info.stencil_addr == 0 || depth_info.stencil_addr == depth_info.guest_address) {
        return;
    }
    ImageId stencil_id{};
    ForEachImageInRegion(
        depth_info.stencil_addr, depth_info.stencil_size, [&](ImageId image_id, Image& image) {
            if (image.info.guest_address != depth_info.stencil_addr) {
                return;
            }
            if (image.info.pixel_format == vk::Format::eUndefined ||
                Vulkan::LiverpoolToVK::IsFormatStencilCompatible(image.info.pixel_format)) {
                stencil_id = image_id;
            }
        });
    if (!stencil_id) {
        ImageInfo info{};
        info.guest_address = depth_info.stencil_addr;
        info.guest_size = depth_info.stencil_size;
        info.size = depth_info.size;
        stencil_id = slot_images.Insert(instance, runtime, slot_image_views, info);
        RegisterImage(stencil_id);
    }
    Image& stencil_image = slot_images[stencil_id];
    TouchImage(stencil_image);
    stencil_image.AssociateDepth(depth_id, slot_images[depth_id].image_uid);
}

// Texture quality: the copies of the guest levels the image leaves out are dropped, the rest go
// to the image's levels.
static void DropLeftOutLevels(const Image& image, SmallVector<vk::BufferImageCopy, 14>& copies) {
    if (image.dropped_levels == 0) {
        return;
    }
    copies.erase(std::remove_if(copies.begin(), copies.end(),
                                [&](const vk::BufferImageCopy& copy) {
                                    return copy.imageSubresource.mipLevel < image.dropped_levels;
                                }),
                 copies.end());
    for (auto& copy : copies) {
        copy.imageSubresource.mipLevel -= image.dropped_levels;
    }
}

// Guest bytes in front of guest level `first` when every level before it lies there (an upload
// starting at `first` does not need to copy them); 0 otherwise.
static u32 BytesBeforeLevel(const Image& image, u32 first) {
    const u32 levels = image.info.resources.levels;
    if (first == 0 || first >= levels) {
        return 0;
    }
    u32 kept_start = image.info.guest_size;
    for (u32 m = first; m < levels; ++m) {
        kept_start = std::min(kept_start, image.info.mips_layout[m].offset);
    }
    for (u32 m = 0; m < first; ++m) {
        if (image.info.mips_layout[m].offset >= kept_start) {
            return 0;
        }
    }
    return kept_start;
}

// Copy only kept texture levels: the guest bytes in front of the first level the image keeps
// (its left-out levels), which an upload does not need to copy. 0 when a left-out level lies
// after a kept one, or with the setting off.
static u32 LeftOutBytes(const Image& image) {
    if (!EmulatorSettings.IsBbSkipLeftOutLevels()) {
        return 0;
    }
    return BytesBeforeLevel(image, image.dropped_levels);
}

static u64 SteadyNowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

// Upload staging and detile output only from the first level copied: the bytes in front (whole
// levels left out or waiting) get no memory when aligned for it (256 bytes; tiled, 64 texels for
// the detile dispatch).
static u32 TrimBytes(const Image& image, u32 skip) {
    if (skip == 0 || skip >= image.info.guest_size || skip % 256 != 0) {
        return 0;
    }
    if (image.info.props.is_tiled) {
        const u32 bytes_per_texel = image.info.num_bits / 8;
        if (bytes_per_texel == 0 || (skip / bytes_per_texel) % 64 != 0) {
            return 0;
        }
    }
    return skip;
}

u32 TextureCache::UploadSkip(Image& image, SmallVector<vk::BufferImageCopy, 14>& copies) {
    // A burst: this game frame already uploaded this much (a traversal load uploads 200-440 MiB
    // of new textures in one frame; usual frames a few MiB). Only textures this large wait.
    constexpr u64 BurstFrameBytes = 32_MB;
    constexpr u32 MinImageBytes = 1_MB;
    namespace GS = Common::GuestStats;
    if (image.detail_pending) {
        // This upload includes the largest level again.
        image.detail_pending = false;
        const u64 soft_ns = SteadyNowNs() - image.detail_pending_since_ns;
        GS::detail_restored.fetch_add(1, std::memory_order_relaxed);
        GS::detail_restored_bytes.fetch_add(image.info.guest_size, std::memory_order_relaxed);
        GS::detail_soft_ns.fetch_add(soft_ns, std::memory_order_relaxed);
        if (soft_ns > GS::detail_soft_max_ns.load(std::memory_order_relaxed)) {
            GS::detail_soft_max_ns.store(soft_ns, std::memory_order_relaxed);
        }
    }
    const u32 left_out = LeftOutBytes(image);
    u32 skip = left_out;
    const u32 host_levels = image.info.resources.levels - image.dropped_levels;
    if (sampled_update && host_levels >= 2 &&
        detail_frame_bytes.load(std::memory_order_relaxed) >= BurstFrameBytes &&
        EmulatorSettings.IsBbLowerDetailFirst() && instance.IsImageViewMinLodSupported() &&
        image.info.guest_size >= MinImageBytes && True(image.flags & ImageFlagBits::CpuDirty) &&
        False(image.flags & ImageFlagBits::GpuModified) && image.info.num_samples == 1 &&
        !image.info.props.is_depth && !image.info.props.has_stencil && !image.info.stencil_addr &&
        !image.info.props.is_volume && !image.usage.render_target && !image.usage.depth_target &&
        !image.usage.storage &&
        std::atomic_ref<u8>(image.exact_texel_reads).load(std::memory_order_relaxed) == 0 &&
        image.backing && image.backing->state.layout == vk::ImageLayout::eUndefined &&
        image.backing->subresource_states.empty()) {
        // A new texture (never written): its largest level waits.
        const u32 next = BytesBeforeLevel(image, image.dropped_levels + 1);
        if (next > left_out) {
            copies.erase(std::remove_if(copies.begin(), copies.end(),
                                        [](const vk::BufferImageCopy& copy) {
                                            return copy.imageSubresource.mipLevel == 0;
                                        }),
                         copies.end());
            skip = next;
            image.detail_pending = true;
            image.detail_pending_since_ns = SteadyNowNs();
            GS::detail_deferred.fetch_add(1, std::memory_order_relaxed);
            GS::detail_deferred_bytes.fetch_add(next - left_out, std::memory_order_relaxed);
        }
    }
    detail_frame_bytes.fetch_add(image.info.guest_size - skip, std::memory_order_relaxed);
    return skip;
}

void TextureCache::TryRestoreDetail(Image& image) {
    // The largest levels arrive in frames that upload little else: up to this much a frame
    // (~2-3 ms of copying), or one a frame once a texture waited this long.
    constexpr u64 RestoreFrameBytes = 24_MB;
    constexpr u64 OverdueNs = 500'000'000;
    const bool fits = detail_frame_bytes.load(std::memory_order_relaxed) + image.info.guest_size <=
                      RestoreFrameBytes;
    const bool overdue = detail_frame_restores.load(std::memory_order_relaxed) == 0 &&
                         SteadyNowNs() - image.detail_pending_since_ns >= OverdueNs;
    if (!fits && !overdue && EmulatorSettings.IsBbLowerDetailFirst()) {
        return;
    }
    detail_frame_restores.fetch_add(1, std::memory_order_relaxed);
    {
        std::scoped_lock lk{image.mutex};
        image.flags |= ImageFlagBits::CpuDirty;
        ClearCleanTick(image);
    }
    const bool was_sampled = std::exchange(sampled_update, false);
    RefreshImage(image);
    sampled_update = was_sampled;
    image.detail_pending = false;
}

void TextureCache::ClearInsteadOfUpload(Image& image) {
    runtime.SetBackingSamples(&image, image.info.num_samples, false);
    const SubresourceRange range{
        .base = {0, 0},
        .extent = {static_cast<u16>(image.info.resources.levels),
                   static_cast<u16>(image.info.resources.layers)},
    };
    runtime.ClearImage(&image, range, vk::ClearValue{});
    Common::GuestStats::new_storage_cleared_bytes.fetch_add(image.info.guest_size,
                                                            std::memory_order_relaxed);
}

void TextureCache::EnsureFullDetail(Image& image) {
    if (!image.detail_pending) {
        return;
    }
    Common::GuestStats::detail_forced.fetch_add(1, std::memory_order_relaxed);
    {
        std::scoped_lock lk{image.mutex};
        image.flags |= ImageFlagBits::CpuDirty;
        ClearCleanTick(image);
    }
    const bool was_sampled = std::exchange(sampled_update, false);
    RefreshImage(image);
    sampled_update = was_sampled;
    image.detail_pending = false;
}

bool TextureCache::PrepareRefresh(Image& image,
                                  SmallVector<vk::BufferImageCopy, 14>& image_copies) {
    if (False(image.flags & ImageFlagBits::Dirty) || image.info.num_samples > 1) {
        return false;
    }

    RENDERER_TRACE;
    TRACE_HINT(fmt::format("{:x}:{:x}", image.info.guest_address, image.info.guest_size));

    if (True(image.flags & ImageFlagBits::MaybeCpuDirty) &&
        False(image.flags & ImageFlagBits::CpuDirty)) {
        // The image size should be less than page size to be considered MaybeCpuDirty
        // So this calculation should be very uncommon and reasonably fast
        // For now we'll just check up to 64 first pixels
        const auto addr = std::bit_cast<u8*>(image.info.guest_address);
        const u32 w = std::min(image.info.size.width, u32(8));
        const u32 h = std::min(image.info.size.height, u32(8));

        const u32 s_w = image.info.props.is_block ? Common::DivCeil(w, 4u) : w;
        const u32 s_h = image.info.props.is_block ? Common::DivCeil(h, 4u) : h;
        const u32 size = s_w * s_h * (image.info.num_bits / 8);
        const u64 hash = XXH3_64bits(addr, size);
        if (image.cpu_hash == hash) {
            image.flags &= ~ImageFlagBits::MaybeCpuDirty;
            return false;
        }
        image.cpu_hash = hash;
    }

    image.gpu_hash = 0;

    const u32 num_layers = image.info.resources.layers;
    const u32 num_mips = image.info.resources.levels;

    image_copies.clear();
    for (u32 m = 0; m < num_mips; m++) {
        const u32 width = std::max(image.info.size.width >> m, 1u);
        const u32 height = std::max(image.info.size.height >> m, 1u);
        const u32 depth =
            image.info.props.is_volume ? std::max(image.info.size.depth >> m, 1u) : 1u;
        const auto [mip_size, mip_pitch, mip_height, mip_offset] = image.info.mips_layout[m];
        const u32 extent_width = mip_pitch ? std::min<u32>(mip_pitch, width) : width;
        const u32 extent_height = mip_height ? std::min<u32>(mip_height, height) : height;
        image_copies.push_back({
            .bufferOffset = mip_offset,
            .bufferRowLength = mip_pitch,
            .bufferImageHeight = mip_height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = m,
                .baseArrayLayer = 0,
                .layerCount = num_layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {extent_width, extent_height, depth},
        });
    }
    DropLeftOutLevels(image, image_copies);

    if (image_copies.empty()) {
        image.flags &= ~ImageFlagBits::Dirty;
        return false;
    }
    return true;
}

bool TextureCache::DeferUpload(Image& image) {
    // A load: this many uploads (and bytes) in one second; normal play stays well below (a
    // burst of a few big textures, 280 MiB in 22, too). Then for SpreadSeconds each game
    // frame uploads up to SpreadFrameBytes, the rest waits. A DLC respawn: ~1000 textures,
    // ~450 MiB in one frame (130 ms) before; on the first load two more bursts follow 2-3 s
    // later (22 textures, 280 MiB, then 180, 310 MiB): those are spread too.
    constexpr u64 SpreadBurstUploads = 300;
    constexpr u64 SpreadBurstBytes = 128_MB;
    constexpr u64 SpreadSeconds = 6;
    constexpr u64 SpreadFrameBytes = 64_MB;
    constexpr u64 ZeroBufferBytes = 32_MB;
    if (!spread_loads_enabled || !Common::BbTuning::Threaded() ||
        False(image.flags & ImageFlagBits::CpuDirty) ||
        True(image.flags & ImageFlagBits::GpuModified) || image.info.num_samples > 1 ||
        image.info.props.is_depth || image.info.props.has_stencil || image.info.stencil_addr ||
        image.info.guest_size > ZeroBufferBytes || !image.backing) {
        return false;
    }
    namespace GS = Common::GuestStats;
    const u64 second = static_cast<u64>(std::chrono::duration_cast<std::chrono::seconds>(
                                            std::chrono::steady_clock::now().time_since_epoch())
                                            .count());
    const u64 uploads = GS::image_uploads.load(std::memory_order_relaxed);
    const u64 bytes = GS::image_upload_bytes.load(std::memory_order_relaxed);
    if (second != spread_window_second) {
        spread_window_second = second;
        spread_window_uploads = uploads;
        spread_window_bytes = bytes;
    }
    if (uploads - spread_window_uploads >= SpreadBurstUploads &&
        bytes - spread_window_bytes >= SpreadBurstBytes) {
        if (second >= spread_until_second) {
            ++spread_loads;
        }
        spread_until_second = second + SpreadSeconds;
    }
    if (second >= spread_until_second) {
        return false;
    }
    const u64 used = spread_frame_bytes.load(std::memory_order_relaxed);
    if (used == 0 || used + image.info.guest_size <= SpreadFrameBytes) {
        spread_frame_bytes.store(used + image.info.guest_size, std::memory_order_relaxed);
        return false;
    }
    ++spread_deferred;
    // Never written yet (an image's first contents): zeros rather than whatever the memory held.
    if (image.backing->state.layout == vk::ImageLayout::eUndefined &&
        image.backing->subresource_states.empty()) {
        if (!zero_buffer) {
            zero_buffer = std::make_unique<Buffer>(instance, 0, ZeroBufferBytes,
                                                   MemoryType::DeviceLocal, "SpreadLoads:Zeros");
            runtime.FillBuffer(zero_buffer.get(), 0, ZeroBufferBytes, 0);
        }
        SmallVector<vk::BufferImageCopy, 14> copies;
        for (u32 m = 0; m < image.info.resources.levels; m++) {
            const u32 width = std::max(image.info.size.width >> m, 1u);
            const u32 height = std::max(image.info.size.height >> m, 1u);
            const u32 depth =
                image.info.props.is_volume ? std::max(image.info.size.depth >> m, 1u) : 1u;
            const auto [mip_size, mip_pitch, mip_height, mip_offset] = image.info.mips_layout[m];
            copies.push_back({
                .bufferOffset = 0,
                .bufferRowLength = mip_pitch,
                .bufferImageHeight = mip_height,
                .imageSubresource{
                    .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                    .mipLevel = m,
                    .baseArrayLayer = 0,
                    .layerCount = image.info.resources.layers,
                },
                .imageOffset = {0, 0, 0},
                .imageExtent = {mip_pitch ? std::min<u32>(mip_pitch, width) : width,
                                mip_height ? std::min<u32>(mip_height, height) : height, depth},
            });
        }
        DropLeftOutLevels(image, copies);
        // UploadImage marks the image clean: it still needs its real upload.
        const auto dirty = image.flags & ImageFlagBits::Dirty;
        runtime.UploadImage(&image, zero_buffer.get(), copies);
        image.flags |= dirty;
        ++spread_zero_filled;
        spread_zero_bytes += image.info.guest_size;
    }
    return true;
}

void TextureCache::RefreshImage(Image& image) {
    Common::Crumbs::Add("image upload (address, size)", image.info.guest_address,
                        image.info.guest_size);
    SmallVector<vk::BufferImageCopy, 14> image_copies;
    if (!PrepareRefresh(image, image_copies)) {
        return;
    }

    scheduler.EndRendering();

    const auto upload_start = std::chrono::steady_clock::now();
    namespace GSD = Common::GuestStats;
    std::optional<GSD::DetailTimer> timer{std::in_place, GSD::Detail::TexStaging};
    const u32 skip = UploadSkip(image, image_copies);
    const u32 trim = TrimBytes(image, skip);
    // Protection on a helper thread: the copies protect the pages they read first.
    ProtectRanges protect;
    if (pending_protect) {
        protect = std::move(*pending_protect);
        pending_protect->clear();
    }
    const auto [in_buffer, in_offset] = buffer_cache.ObtainBufferForImage(
        image.info.guest_address, image.info.guest_size, skip, trim, std::move(protect));
    timer.emplace(GSD::Detail::TexDetile);
    const auto [buffer, offset] =
        tile_manager.DetileImage(in_buffer, in_offset, image.info, true, trim);
    for (auto& copy : image_copies) {
        copy.bufferOffset += offset;
    }

    timer.emplace(GSD::Detail::TexUpload);
    runtime.UploadImage(&image, buffer, image_copies);
    timer.reset();
    namespace GS = Common::GuestStats;
    GS::image_uploads.fetch_add(1, std::memory_order_relaxed);
    GS::image_upload_bytes.fetch_add(image.info.guest_size, std::memory_order_relaxed);
    GS::NoteUploadPath(image.info.guest_size);
    NoteUpload(image);
    GS::image_upload_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::steady_clock::now() - upload_start)
                                      .count(),
                                  std::memory_order_relaxed);
}

void TextureCache::UpdateImages(std::span<const ImageId> image_ids) {
    for (const ImageId image_id : image_ids) {
        if (image_id && slot_images[image_id].detail_pending) {
            TryRestoreDetail(slot_images[image_id]);
        }
    }
    const bool was_sampled = std::exchange(sampled_update, true);
    Common::GuestStats::UploadPathScope path{Common::GuestStats::UploadPath::Texture};
    UpdateImagesSampled(image_ids);
    sampled_update = was_sampled;
}

void TextureCache::UpdateImagesSampled(std::span<const ImageId> image_ids) {
    // Spread area loading: uploads past this frame's budget wait (the image stays dirty, so a
    // later draw binding it tries again).
    SmallVector<ImageId, 16> deferred;
    if (spread_loads_enabled) {
        SmallVector<ImageId, 16> checked;
        for (const ImageId image_id : image_ids) {
            if (!image_id || IsClean(image_id) ||
                std::ranges::find(checked, image_id) != checked.end()) {
                continue;
            }
            checked.push_back(image_id);
            Image& image = slot_images[image_id];
            if (DeferUpload(image)) {
                TouchImage(image);
                deferred.push_back(image_id);
            }
        }
    }
    const auto is_deferred = [&](ImageId image_id) {
        return !deferred.empty() && std::ranges::find(deferred, image_id) != deferred.end();
    };
    // Images that need an upload; the rest go through UpdateImage as before.
    SmallVector<ImageId, 16> pending;
    if (EmulatorSettings.IsBbBatchedUploads()) {
        for (const ImageId image_id : image_ids) {
            if (!image_id || IsClean(image_id) || is_deferred(image_id)) {
                continue;
            }
            Image& image = slot_images[image_id];
            // (Images checked by hash first keep the one-at-a-time path.)
            if (False(image.flags & ImageFlagBits::Dirty) || image.info.num_samples > 1 ||
                (True(image.flags & ImageFlagBits::MaybeCpuDirty) &&
                 False(image.flags & ImageFlagBits::CpuDirty)) ||
                std::ranges::find(pending, image_id) != pending.end()) {
                continue;
            }
            pending.push_back(image_id);
        }
    }
    if (pending.size() < 2) {
        for (const ImageId image_id : image_ids) {
            if (image_id && !is_deferred(image_id)) {
                UpdateImage(image_id);
            }
        }
        return;
    }

    // Protected before their contents are read (as UpdateImage does); with protection on a
    // helper thread, by that thread (the copies wait for it).
    const bool defer_protect = ProtectOnHelperThread();
    SmallVector<ProtectRanges, 16> deferred_protect(pending.size());
    for (size_t i = 0; i < pending.size(); ++i) {
        TouchImage(slot_images[pending[i]]);
        Common::GuestStats::DetailTimer timer{Common::GuestStats::Detail::TexTrack};
        TrackImage(pending[i], defer_protect ? &deferred_protect[i] : nullptr);
    }

    struct Job {
        Image* image;
        SmallVector<vk::BufferImageCopy, 14> copies;
        const Buffer* buffer;
        u64 offset;
        u32 trim;
    };
    SmallVector<Job, 16> jobs;
    scheduler.EndRendering();
    const auto upload_start = std::chrono::steady_clock::now();
    namespace GSD = Common::GuestStats;
    {
        GSD::DetailTimer timer{GSD::Detail::TexStaging};
        for (size_t i = 0; i < pending.size(); ++i) {
            Image& image = slot_images[pending[i]];
            Job job{.image = &image};
            if (!PrepareRefresh(image, job.copies)) {
                ApplyProtection(deferred_protect[i]);
                continue;
            }
            const u32 skip = UploadSkip(image, job.copies);
            job.trim = TrimBytes(image, skip);
            const auto [buffer, offset] =
                buffer_cache.ObtainBufferForImage(image.info.guest_address, image.info.guest_size,
                                                  skip, job.trim, std::move(deferred_protect[i]));
            deferred_protect[i].clear();
            job.buffer = buffer;
            job.offset = offset;
            jobs.push_back(std::move(job));
        }
    }
    if (!jobs.empty()) {
        {
            // One barrier for every source, then the detiles back to back.
            GSD::DetailTimer timer{GSD::Detail::TexDetile};
            runtime.FlushBarriers();
            for (Job& job : jobs) {
                const auto [buffer, offset] = tile_manager.DetileImage(
                    job.buffer, job.offset, job.image->info, false, job.trim);
                job.buffer = buffer;
                for (auto& copy : job.copies) {
                    copy.bufferOffset += offset;
                }
            }
        }
        GSD::DetailTimer timer{GSD::Detail::TexUpload};
        // Every image to transfer layout and the detiles finished: one barrier, then the copies.
        for (Job& job : jobs) {
            runtime.SetBackingSamples(job.image, job.image->info.num_samples, false);
            runtime.Transit(job.image, vk::ImageLayout::eTransferDstOptimal,
                            vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite);
        }
        runtime.FlushBarriers();
        for (Job& job : jobs) {
            runtime.RecordUploadCopies(job.image, job.buffer, job.copies);
            GSD::image_uploads.fetch_add(1, std::memory_order_relaxed);
            GSD::image_upload_bytes.fetch_add(job.image->info.guest_size,
                                              std::memory_order_relaxed);
            GSD::NoteUploadPath(job.image->info.guest_size);
            NoteUpload(*job.image);
        }
    }
    GSD::image_upload_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - upload_start)
                                       .count(),
                                   std::memory_order_relaxed);
    // The usual bookkeeping (clean marking); the uploads are done.
    for (const ImageId image_id : image_ids) {
        if (image_id && !is_deferred(image_id)) {
            UpdateImage(image_id);
        }
    }
}

vk::Sampler TextureCache::GetSampler(const AmdGpu::Sampler& sharp,
                                     AmdGpu::BorderColorBuffer border_color_base,
                                     const bool is_depth, const float extra_lod_bias) {
    // Compare and plain uses of one S# need separate samplers, and so do upscaler LOD biases.
    const u64 hash = HashCombine(HashCombine(XXH3_64bits(&sharp, sizeof(sharp)), is_depth),
                                 u64(std::bit_cast<u32>(extra_lod_bias)));
    const auto [it, new_sampler] = samplers.try_emplace(hash);
    if (new_sampler) {
        it->second =
            slot_samplers.Insert(instance, sharp, border_color_base, is_depth, extra_lod_bias);
        Sampler& sampler = slot_samplers[it->second];
        sampler.hash = hash;
        sampler_lru_cache.Insert(sampler, gc_tick);
    }

    Sampler& sampler = slot_samplers[it->second];
    sampler_lru_cache.Touch(sampler, gc_tick);
    return sampler.Handle();
}

void TextureCache::TouchSampler(const AmdGpu::Sampler& sampler, const bool is_depth,
                                const float extra_lod_bias) {
    // The same hash as GetSampler.
    const u64 hash = HashCombine(HashCombine(XXH3_64bits(&sampler, sizeof(sampler)), is_depth),
                                 u64(std::bit_cast<u32>(extra_lod_bias)));
    if (const auto it = samplers.find(hash); it != samplers.end()) {
        sampler_lru_cache.Touch(slot_samplers[it->second], gc_tick);
    }
}

void TextureCache::RegisterImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered),
               "Trying to register an already registered image");
    image.flags |= ImageFlagBits::Registered;
    registry_generation.fetch_add(1, std::memory_order_acq_rel);
    image.memory_usage_counter = &total_used_memory;
    total_used_memory += Common::AlignUp(image.SizeBytes(), 1024);
    image_lru_cache.Insert(image, gc_tick);
    const auto& info = image.info;
    ASSERT_MSG((info.guest_address & 0xff) == 0, "Trying to register an unaligned image");
    ForEachPage(info.guest_address, info.guest_size, [this, image_id, info](u64 page) {
        page_table[page].entries.emplace_back(BucketEntry{
            .key = u32(info.guest_address >> 8),
            .size = info.guest_size,
            .id = image_id,
        });
    });
}

void TextureCache::UnregisterImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(True(image.flags & ImageFlagBits::Registered),
               "Trying to unregister an already unregistered image");
    image.flags &= ~ImageFlagBits::Registered;
    ClearCleanTick(image);
    registry_generation.fetch_add(1, std::memory_order_acq_rel);
    image_lru_cache.Free(image);
    image.memory_usage_counter = nullptr;
    total_used_memory -= Common::AlignUp(image.SizeBytes(), 1024);
    ForEachPage(image.info.guest_address, image.info.guest_size, [this, image_id](u64 page) {
        const auto page_it = page_table.find(page);
        ASSERT_MSG(page_it, "Unregistering unregistered page={:#x}", page << Traits::PAGE_BITS);
        auto& entries = page_it->entries;
        const auto vector_it = std::ranges::find(entries, image_id, &BucketEntry::id);
        ASSERT_MSG(vector_it != entries.end(), "Unregistering unregistered image in page={:#x}",
                   page << Traits::PAGE_BITS);
        entries.erase(vector_it);
    });
}

bool TextureCache::ProtectOnHelperThread() {
    return Common::BbTuning::Threaded() && EmulatorSettings.IsBbProtectHelperThread() &&
           EmulatorSettings.IsBbBackgroundTextureCopies();
}

void TextureCache::ApplyProtection(ProtectRanges& ranges) {
    for (const auto& [address, size] : ranges) {
        page_manager.ApplyProtection(address, size);
    }
    ranges.clear();
}

void TextureCache::TrackImage(ImageId image_id, ProtectRanges* deferred) {
    auto& image = slot_images[image_id];
    if (False(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_begin = image.info.guest_address;
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (image_begin == image.track_addr && image_end == image.track_addr_end) {
        return;
    }

    std::scoped_lock lk{image.mutex};
    if (image.IsUntracked()) {
        page_manager.UpdatePageWatchers(image_begin, image.info.guest_size, PageOp::Track, deferred);
    } else {
        if (image_begin < image.track_addr) {
            page_manager.UpdatePageWatchers(image_begin, image.track_addr - image_begin, PageOp::Track,
                                       deferred);
        }
        if (image.track_addr_end < image_end) {
            page_manager.UpdatePageWatchers(image.track_addr_end, image_end - image.track_addr_end,
                                       PageOp::Track, deferred);
        }
    }
    image.track_addr = image_begin;
    image.track_addr_end = image_end;
    MarkDirtyIfEdgesModified(image);
    image.head_hash = 0;
    image.tail_hash = 0;
}

void TextureCache::UntrackImage(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (image.IsUntracked()) {
        return;
    }
    ClearCleanTick(image);
    const auto addr = image.track_addr;
    const auto size = image.track_addr_end - image.track_addr;
    if (size != 0) {
        page_manager.UpdatePageWatchers(addr, size, PageOp::Untrack);
    }
    image.track_addr = 0;
    image.track_addr_end = 0;
}

void TextureCache::UntrackImageHead(ImageId image_id) {
    auto& image = slot_images[image_id];
    const auto image_begin = image.info.guest_address;
    if (image.IsUntracked() || image_begin < image.track_addr) {
        return;
    }
    ClearCleanTick(image);
    const auto addr = page_manager.GetNextPageAddr(image_begin);
    const auto size = addr - image_begin;
    image.head_hash = HashBackingMemory(image_begin, size);
    page_manager.UpdatePageWatchers(image_begin, size, PageOp::Untrack);

    image.track_addr = addr;
    if (image.track_addr == image.track_addr_end) {
        // This image spans only 2 pages and both are modified,
        // but the image itself was not directly affected.
        // Cehck its hash later.
        MarkAsMaybeDirty(image_id, image);
    }
}

void TextureCache::UntrackImageTail(ImageId image_id) {
    auto& image = slot_images[image_id];
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (image.IsUntracked() || image.track_addr_end < image_end) {
        return;
    }
    ASSERT(image.track_addr_end != 0);
    ClearCleanTick(image);
    const auto addr = page_manager.GetPageAddr(image_end);
    const auto size = image_end - addr;
    image.tail_hash = HashBackingMemory(addr, size);
    image.track_addr_end = addr;
    if (image.track_addr == image.track_addr_end) {
        // This image spans only 2 pages and both are modified,
        // but the image itself was not directly affected.
        // Cehck its hash later.
        MarkAsMaybeDirty(image_id, image);
    }
    page_manager.UpdatePageWatchers(addr, size, PageOp::Untrack);
}

void TextureCache::MarkDirtyIfEdgesModified(Image& image) {
    const auto image_begin = image.info.guest_address;
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (image.head_hash != 0) {
        const auto head_end = page_manager.GetNextPageAddr(image_begin);
        if (HashBackingMemory(image_begin, head_end - image_begin) != image.head_hash) {
            image.flags |= ImageFlagBits::CpuDirty;
        }
    }
    if (image.tail_hash != 0) {
        const auto tail_begin = page_manager.GetPageAddr(image_end);
        if (HashBackingMemory(tail_begin, image_end - tail_begin) != image.tail_hash) {
            image.flags |= ImageFlagBits::CpuDirty;
        }
    }
}

void TextureCache::GarbageCollectImages() {
    namespace GS = Common::GuestStats;
    // VRAM cleanup measures the memory in use: free space inside the allocator's blocks and the
    // buffer slabs, and images freed but not yet released (the GPU may still use them), take new
    // data without more from the driver, which counts them as used. Measured with them, every
    // pass saw the space freed by the last ones as still full and freed more (in pressure mode
    // most of the time on a 6 GB card with a texture pack). At the budget itself, the driver's
    // figure decides: past it, the driver moves memory to system RAM.
    UpdateVramLimits();
    bool over_budget = false;
    const auto measure = [&] {
        const auto query_start = std::chrono::steady_clock::now();
        const u64 driver_usage = instance.GetDeviceMemoryUsage();
        u64 usage = driver_usage;
        if (vram_reclaim) {
            const auto allocator = instance.GetAllocatorUsage();
            const u64 unused = (allocator.held - std::min(allocator.held, allocator.in_use)) +
                               buffer_cache.GetPoolUsage().free +
                               pending_free_bytes.load(std::memory_order_relaxed);
            usage -= std::min(usage, unused);
            over_budget =
                driver_usage >= (live_limits_active ? live_limit : instance.GetTotalMemoryBudget());
        }
        total_used_memory = usage;
        GS::gc_usage_query_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                            std::chrono::steady_clock::now() - query_start)
                                            .count(),
                                        std::memory_order_relaxed);
    };
    if (instance.CanReportMemoryUsage()) {
        measure();
        ReportPastBudget();
        // Under pressure, idle upload scratch in graphics card memory goes before any texture:
        // a streaming burst grew it (hundreds of MiB, kept for seconds after each use), and the
        // collector was evicting textures the game used instead.
        if (vram_reclaim && (total_used_memory >= pressure_gc_memory || over_budget)) {
            const u64 before = total_used_memory;
            const auto released = runtime.GetStagingPool().ReleaseIdleDeviceLocal();
            if (released.bytes != 0) {
                GS::scratch_releases.fetch_add(1, std::memory_order_relaxed);
                GS::scratch_released_bytes.fetch_add(released.bytes, std::memory_order_relaxed);
                measure();
                const auto now = std::chrono::steady_clock::now();
                if (now - scratch_report_time >= std::chrono::seconds(2)) {
                    scratch_report_time = now;
                    LOG_INFO(Render,
                             "Under VRAM pressure: released {} MiB of idle upload scratch ({} "
                             "blocks, {} large buffers), {} MiB still held | in use {} MiB "
                             "before, {} MiB after (pressure {}, critical {})",
                             released.bytes >> 20, released.ring_blocks, released.large_buffers,
                             runtime.GetStagingPool().DeviceLocalBytes() >> 20, before >> 20,
                             total_used_memory.load() >> 20, pressure_gc_memory >> 20,
                             critical_gc_memory >> 20);
                }
            }
        }
    }
    if (total_used_memory < trigger_gc_memory) {
        return;
    }
    // VRAM cleanup: ages also in seconds (after bbport). gc_tick (one per submission, ~100 a
    // second) at each of the last 64 seconds; an image goes once unused for GcIdleSeconds
    // normally (the textures of an area left behind), GcPressureIdleSeconds over the pressure
    // mark and 1 s over the critical one. The submission counts alone (16/80/160, a fraction
    // of a second) evicted textures briefly off screen and uploaded them again.
    constexpr u64 GcIdleSeconds = 20;
    constexpr u64 GcPressureIdleSeconds = 5;
    // Loading an area (a respawn in the DLC: 430 MiB of new textures in one frame) arrives while
    // the old area's textures are still there, and VRAM went over budget (the driver then moved
    // memory out: 200 ms GPU waits). So after a burst of uploads, textures unused for 2 s go
    // from LoadHeadroom below the pressure mark, for LoadSeconds. Only textures new to the cache
    // count: re-uploads of ones the collector freed (it brought back half the uploads in a 6 GB
    // card's session) aren't an area load, and counting them kept it freeing ahead.
    constexpr u64 LoadBurstBytes = 192_MB;
    constexpr u64 LoadSeconds = 10;
    constexpr u64 LoadIdleSeconds = 2;
    constexpr u64 LoadHeadroom = 512_MB;
    const u64 second = static_cast<u64>(std::chrono::duration_cast<std::chrono::seconds>(
                                            std::chrono::steady_clock::now().time_since_epoch())
                                            .count());
    if (vram_reclaim && second != gc_second) {
        if (gc_second == 0) {
            gc_tick_at_second.fill(gc_tick);
        }
        gc_second = second;
        gc_tick_at_second[second % gc_tick_at_second.size()] = gc_tick;
        load_window_bytes = GS::image_upload_bytes.load(std::memory_order_relaxed);
        load_window_reupload_bytes = GS::gc_reupload_bytes.load(std::memory_order_relaxed);
    }
    if (vram_reclaim) {
        const u64 reuploaded =
            GS::gc_reupload_bytes.load(std::memory_order_relaxed) - load_window_reupload_bytes;
        const u64 uploaded =
            GS::image_upload_bytes.load(std::memory_order_relaxed) - load_window_bytes;
        const u64 new_bytes = uploaded - std::min(uploaded, reuploaded);
        const bool after_cleanup = std::chrono::steady_clock::now() < free_ahead_paused_until;
        if (new_bytes >= LoadBurstBytes && after_cleanup) {
            if (load_burst_ignored_second != second) {
                load_burst_ignored_second = second;
                LOG_INFO(Render,
                         "Upload burst not taken as an area load: {} MiB uploaded this second, "
                         "just after a loading screen's cleanup made room",
                         uploaded >> 20);
            }
        } else if (new_bytes >= LoadBurstBytes) {
            if (second >= load_until_second) {
                ++gc_loads;
            }
            load_until_second = second + LoadSeconds;
        } else if (uploaded >= LoadBurstBytes && load_burst_ignored_second != second) {
            load_burst_ignored_second = second;
            LOG_INFO(Render,
                     "Upload burst not taken as an area load: {} MiB uploaded this second, {} MiB "
                     "of it textures the cleanup had freed",
                     uploaded >> 20, reuploaded >> 20);
        }
    }
    const bool loading = vram_reclaim && second < load_until_second;
    const auto tick_seconds_ago = [&](u64 seconds) {
        return gc_tick_at_second[(second - seconds) % gc_tick_at_second.size()];
    };
    const auto pass_start = std::chrono::steady_clock::now();
    const u64 writeback_done_before = GS::gc_writeback_done_ns.load(std::memory_order_relaxed);
    ++gc_runs;
    CompleteWriteBacks();
    bool pressured = false;
    bool aggresive = false;
    int worst_mode = 0;
    u64 ticks_to_destroy = 0;
    u64 below_tick = 0;
    size_t num_deletions = 0;
    size_t scanned = 0;
    size_t deleted = 0;
    size_t skipped_surface = 0;
    size_t skipped_pressure = 0;
    size_t failed_writebacks = 0;
    const u64 observed_usage = total_used_memory;

    const auto configure = [&](bool allow_aggressive) {
        pressured = total_used_memory >= pressure_gc_memory ||
                    (loading && total_used_memory + LoadHeadroom >= pressure_gc_memory);
        aggresive = allow_aggressive && (total_used_memory >= critical_gc_memory || over_budget);
        ticks_to_destroy = aggresive ? 160 : pressured ? 80 : 16;
        ticks_to_destroy = std::min(ticks_to_destroy, gc_tick);
        below_tick = gc_tick - ticks_to_destroy;
        if (vram_reclaim) {
            const u64 idle = aggresive              ? 1
                             : loading && pressured ? LoadIdleSeconds
                             : pressured            ? GcPressureIdleSeconds
                                                    : GcIdleSeconds;
            below_tick = std::min(below_tick, tick_seconds_ago(idle));
        }
        num_deletions = aggresive ? 40 : pressured ? 20 : 10;
        worst_mode = std::max(worst_mode, aggresive ? 2 : pressured ? 1 : 0);
    };
    // An image that cannot go in this mode moves to the back of the queue (after bbport):
    // otherwise every pass scans the same oldest ones again (thousands a second).
    const auto keep = [&](const Image& image) {
        image_lru_cache.Touch(const_cast<Image&>(image), gc_tick);
        return false;
    };
    // VRAM cleanup off: the original collector. On: tiled images are written back too (so they
    // can be freed at all), and only images whose write-back completed are freed.
    size_t writebacks = 0;
    u64 free_ns = 0;
    u64 begin_writeback_ns = 0;
    const auto clean_up = [&](Image& image) {
        if (num_deletions == 0) {
            return true;
        }
        if (!vram_reclaim) {
            --num_deletions;
        }
        ++scanned;
        MarkDirtyIfEdgesModified(image);
        const auto image_id = slot_images.GetSlotId(image);
        const bool download = image.HasGpuValidData();
        if (!vram_reclaim) {
            const bool tiled = image.info.IsTiled();
            if (tiled && download) {
                // This is a workaround for now. We can't handle non-linear image downloads.
                ++gc_skipped_stuck;
                return false;
            }
            if (download && !pressured) {
                ++gc_skipped_written;
                return false;
            }
            if (download) {
                DownloadImageMemory(image_id, DownloadMemoryFlags::Priority);
                ++gc_written_back;
            }
        } else {
            const auto& meta = image.info.meta_info;
            const bool has_auxiliary_surface = meta.cmask_addr || meta.fmask_addr ||
                                               meta.htile_addr || image.info.stencil_addr ||
                                               image.info.num_samples > 1;
            if (download && image.info.IsTiled() && has_auxiliary_surface) {
                // This writeback path only preserves the tiled pixel surface. Keep images with
                // separate compression, depth/stencil, or multisample state resident.
                ++skipped_surface;
                return keep(image);
            }
            if (download && !pressured) {
                ++skipped_pressure;
                return keep(image);
            }
            if (download && !SafeToWriteBack(image)) {
                // Part of its memory may hold newer CPU or GPU buffer data: writing the image
                // back would overwrite it (seen as game crashes while loading an area).
                ++gc_skipped_written;
                return keep(image);
            }
            if (download) {
                // Written back without waiting for the GPU: the copy is recorded now and the
                // image freed by a later pass, once the copy is done (CompleteWriteBacks).
                if (std::ranges::any_of(pending_writebacks, [&](const PendingWriteBack& pending) {
                        return pending.image_id == image_id;
                    })) {
                    return keep(image);
                }
                if (writebacks >= MAX_GC_WRITEBACKS ||
                    pending_writebacks.size() >= MAX_PENDING_WRITEBACKS) {
                    return false;
                }
                const auto now = std::chrono::steady_clock::now();
                const double elapsed =
                    std::chrono::duration<double>(now - writeback_budget_time).count();
                writeback_budget_time = now;
                writeback_budget = std::min<double>(
                    writeback_budget + elapsed * WRITEBACK_BYTES_PER_SECOND, WRITEBACK_BURST_BYTES);
                if (writeback_budget <= 0.0) {
                    return false;
                }
                // A failed write-back is likely to fail again.
                const auto retry = writeback_retry.find(image_id.index);
                if (retry != writeback_retry.end() && gc_tick < retry->second) {
                    return false;
                }
                ++writebacks;
                writeback_budget -= static_cast<double>(image.info.guest_size);
                const auto begin_start = std::chrono::steady_clock::now();
                const bool began = BeginWriteBack(image_id);
                begin_writeback_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now() - begin_start)
                                          .count();
                if (!began) {
                    writeback_budget += static_cast<double>(image.info.guest_size);
                    writeback_retry[image_id.index] = gc_tick + WRITEBACK_RETRY_TICKS;
                    ++failed_writebacks;
                    return false;
                }
                GS::gc_writebacks_started.fetch_add(1, std::memory_order_relaxed);
                GS::gc_writeback_started_bytes.fetch_add(image.info.guest_size,
                                                         std::memory_order_relaxed);
                if (retry != writeback_retry.end()) {
                    writeback_retry.erase(retry);
                }
                return keep(image);
            }
        }
        ++gc_evicted;
        NoteEvicted(image);
        const auto free_start = std::chrono::steady_clock::now();
        FreeImage(image_id);
        free_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now() - free_start)
                       .count();
        if (vram_reclaim) {
            --num_deletions;
            ++deleted;
        }
        if (vram_reclaim) {
            // Freed only until back under the mark of its mode (the least recently used first):
            // the next pass measures again. Going on with the mode's short idle time freed
            // textures in use, which came back a few seconds later.
            if (num_deletions == 0) {
                return true;
            }
            if (aggresive) {
                return total_used_memory < critical_gc_memory && !over_budget;
            }
            if (pressured) {
                return total_used_memory < pressure_gc_memory &&
                       (!loading || total_used_memory + LoadHeadroom < pressure_gc_memory);
            }
            return false;
        }
        if (total_used_memory < critical_gc_memory) {
            if (aggresive) {
                num_deletions >>= 2;
                aggresive = false;
                return false;
            }
            if (pressured && total_used_memory < pressure_gc_memory &&
                (!loading || total_used_memory + LoadHeadroom < pressure_gc_memory)) {
                num_deletions >>= 1;
                pressured = false;
            }
        }
        return false;
    };
    const auto collect = [&] {
        if (vram_reclaim) {
            image_lru_cache.ForEachItemBelow(below_tick, MAX_GC_IMAGE_SCANS, clean_up);
        } else {
            image_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);
        }
    };
    SCOPE_EXIT {
        constexpr u64 SlowPassNs = 2'000'000;
        constexpr std::array<const char*, 3> mode_names{"normal", "pressure", "aggressive"};
        const auto now = std::chrono::steady_clock::now();
        const u64 pass_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(now - pass_start).count();
        GS::gc_passes.fetch_add(1, std::memory_order_relaxed);
        GS::gc_ns.fetch_add(pass_ns, std::memory_order_relaxed);
        if (worst_mode == 2) {
            GS::gc_aggressive_passes.fetch_add(1, std::memory_order_relaxed);
        } else if (worst_mode == 1) {
            GS::gc_pressure_passes.fetch_add(1, std::memory_order_relaxed);
        }
        u64 slowest = GS::gc_slowest_ns.load(std::memory_order_relaxed);
        while (pass_ns > slowest && !GS::gc_slowest_ns.compare_exchange_weak(
                                        slowest, pass_ns, std::memory_order_relaxed)) {
        }
        if (pass_ns < SlowPassNs || now - slow_pass_report_time < std::chrono::milliseconds(500)) {
            return;
        }
        slow_pass_report_time = now;
        const u64 finishing_ns =
            GS::gc_writeback_done_ns.load(std::memory_order_relaxed) - writeback_done_before;
        LOG_WARNING(Render,
                    "Slow image cleanup pass: {:.1f} ms in {} mode{} | VRAM {} MiB (pressure {}, "
                    "critical {}) | scanned {}, freed {} images in {:.1f} ms, {} write-backs "
                    "started in {:.1f} ms ({} failed), finishing earlier write-backs {:.1f} ms, {} "
                    "pending",
                    pass_ns / 1e6, mode_names[worst_mode], loading ? " (loading an area)" : "",
                    observed_usage >> 20, pressure_gc_memory >> 20, critical_gc_memory >> 20,
                    scanned, deleted, free_ns / 1e6, writebacks, begin_writeback_ns / 1e6,
                    failed_writebacks, finishing_ns / 1e6, pending_writebacks.size());
    };

    // Try to remove anything old enough and not high priority.
    configure(false);
    collect();

    if (total_used_memory >= critical_gc_memory || over_budget) {
        // If we are still over the critical limit, run an aggressive GC
        if (over_budget && total_used_memory < critical_gc_memory) {
            GS::gc_budget_limit_passes.fetch_add(1, std::memory_order_relaxed);
        }
        configure(true);
        collect();
    }
    if (vram_reclaim && gc_tick % 300 == 0) {
        LOG_DEBUG(Render,
                  "Image GC: usage={} MiB pressure={} MiB critical={} MiB scanned={} deleted={} "
                  "retained_surface={} retained_until_pressure={} writeback_rejected={}",
                  observed_usage >> 20, pressure_gc_memory >> 20, critical_gc_memory >> 20, scanned,
                  deleted, skipped_surface, skipped_pressure, failed_writebacks);
    }
}

static constexpr size_t MaxRecentlyEvicted = 16384;
static constexpr auto ReuploadWindow = std::chrono::seconds(60);

void TextureCache::NoteEvicted(const Image& image) {
    namespace GS = Common::GuestStats;
    GS::gc_freed_images.fetch_add(1, std::memory_order_relaxed);
    GS::gc_freed_bytes.fetch_add(image.info.guest_size, std::memory_order_relaxed);
    const auto now = std::chrono::steady_clock::now();
    std::scoped_lock lock{evicted_mutex};
    if (recently_evicted.size() >= MaxRecentlyEvicted) {
        std::erase_if(recently_evicted,
                      [&](const auto& entry) { return now - entry.second.time > ReuploadWindow; });
        if (recently_evicted.size() >= MaxRecentlyEvicted) {
            recently_evicted.clear();
        }
    }
    recently_evicted[image.info.guest_address] = {now, image.info.guest_size};
}

static size_t SizeClassOf(u64 bytes) {
    return bytes < 256_KB ? 0 : bytes < 1_MB ? 1 : bytes < 4_MB ? 2 : bytes < 16_MB ? 3 : 4;
}

static constexpr std::array<const char*, 5> SizeClassNames{"under 256 KiB", "256 KiB-1 MiB",
                                                           "1-4 MiB", "4-16 MiB", "16 MiB+"};

void TextureCache::NoteUpload(Image& image) {
    {
        namespace GS = Common::GuestStats;
        const u64 now = static_cast<u64>(std::chrono::steady_clock::now().time_since_epoch() /
                                         std::chrono::nanoseconds{1});
        size_t source = static_cast<size_t>(GS::DirtySource::New);
        size_t bucket = 0;
        if (image.dirty_since_ns != 0) {
            source = image.dirty_source;
            constexpr u64 FrameNs = 16'666'667;
            const u64 frames = (now - std::min(now, image.dirty_since_ns)) / FrameNs;
            bucket = frames < 1 ? 0 : frames < 2 ? 1 : frames < 6 ? 2 : frames < 30 ? 3 : 4;
        }
        GS::upload_age_bytes[source * GS::UploadAgeBuckets + bucket].fetch_add(
            image.info.guest_size, std::memory_order_relaxed);
        image.dirty_since_ns = 0;
    }
    const size_t size_class = SizeClassOf(image.info.guest_size);
    upload_count_by_size[size_class].fetch_add(1, std::memory_order_relaxed);
    upload_bytes_by_size[size_class].fetch_add(image.info.guest_size, std::memory_order_relaxed);
    std::scoped_lock lock{evicted_mutex};
    const auto it = recently_evicted.find(image.info.guest_address);
    if (it == recently_evicted.end()) {
        return;
    }
    const auto age = std::chrono::steady_clock::now() - it->second.time;
    recently_evicted.erase(it);
    if (age > ReuploadWindow) {
        return;
    }
    namespace GS = Common::GuestStats;
    GS::gc_reuploads.fetch_add(1, std::memory_order_relaxed);
    GS::gc_reupload_bytes.fetch_add(image.info.guest_size, std::memory_order_relaxed);
    GS::gc_reupload_age_ms.fetch_add(
        std::chrono::duration_cast<std::chrono::milliseconds>(age).count(),
        std::memory_order_relaxed);
}

TextureCache::CleanupCounters TextureCache::ReadCleanupCounters() {
    namespace GS = Common::GuestStats;
    const auto read = [](const std::atomic<u64>& counter) {
        return counter.load(std::memory_order_relaxed);
    };
    return {
        .passes = read(GS::gc_passes),
        .pressure_passes = read(GS::gc_pressure_passes),
        .aggressive_passes = read(GS::gc_aggressive_passes),
        .ns = read(GS::gc_ns),
        .usage_query_ns = read(GS::gc_usage_query_ns),
        .freed_images = read(GS::gc_freed_images),
        .freed_bytes = read(GS::gc_freed_bytes),
        .writebacks = read(GS::gc_writebacks_started),
        .writeback_bytes = read(GS::gc_writeback_started_bytes),
        .writeback_done_ns = read(GS::gc_writeback_done_ns),
        .written_bytes = read(GS::gc_writeback_written_bytes),
        .reuploads = read(GS::gc_reuploads),
        .reupload_bytes = read(GS::gc_reupload_bytes),
        .reupload_age_ms = read(GS::gc_reupload_age_ms),
        .buffer_evictions = read(GS::buffer_evictions),
        .buffer_eviction_bytes = read(GS::buffer_eviction_bytes),
        .buffer_eviction_ns = read(GS::buffer_eviction_ns),
        .buffer_eviction_unbind_ns = read(GS::buffer_eviction_unbind_ns),
        .uploaded_bytes = read(GS::image_upload_bytes),
        .budget_limit_passes = read(GS::gc_budget_limit_passes),
        .scratch_releases = read(GS::scratch_releases),
        .scratch_released_bytes = read(GS::scratch_released_bytes),
        .scratch_allocations = read(GS::scratch_allocations),
        .scratch_allocation_bytes = read(GS::scratch_allocation_bytes),
    };
}

void TextureCache::LogCleanupReport() {
    const auto now = std::chrono::steady_clock::now();
    if (cleanup_report_time == std::chrono::steady_clock::time_point{}) {
        cleanup_report_time = now;
        cleanup_last = ReadCleanupCounters();
        cleanup_host_usage = instance.GetMemoryHeapReport().host_usage;
        return;
    }
    if (now - cleanup_report_time < std::chrono::seconds(1)) {
        return;
    }
    const double seconds = std::chrono::duration<double>(now - cleanup_report_time).count();
    cleanup_report_time = now;
    const CleanupCounters current = ReadCleanupCounters();
    const CleanupCounters last = std::exchange(cleanup_last, current);
    const u64 slowest_ns = Common::GuestStats::gc_slowest_ns.exchange(0, std::memory_order_relaxed);
    const u64 pressure = current.pressure_passes - last.pressure_passes;
    const u64 aggressive = current.aggressive_passes - last.aggressive_passes;
    const u64 freed = current.freed_images - last.freed_images;
    const u64 started = current.writebacks - last.writebacks;
    const u64 reuploads = current.reuploads - last.reuploads;
    const u64 evictions = current.buffer_evictions - last.buffer_evictions;
    const u64 scratch = current.scratch_releases - last.scratch_releases;
    constexpr u64 QuietFreedImages = 16;
    if (pressure + aggressive + started + reuploads + evictions + scratch == 0 &&
        freed < QuietFreedImages) {
        return;
    }
    const auto heaps = instance.GetMemoryHeapReport();
    const auto mib = [](u64 bytes) { return bytes >> 20; };
    const auto ms = [](u64 ns) { return ns / 1e6; };
    const double reupload_seconds =
        reuploads != 0 ? (current.reupload_age_ms - last.reupload_age_ms) / 1000.0 / reuploads
                       : 0.0;
    const std::string text = fmt::format(
        "VRAM cleanup, last {:.1f} s: image collector {} passes ({} under pressure, {} "
        "aggressive, {} of those only for being at the budget) taking {:.1f} ms, slowest {:.1f} "
        "ms, reading memory use {:.1f} ms | freed {} "
        "images ({} MiB), {} write-backs started ({} MiB), {} MiB copied to guest memory in {:.1f} "
        "ms, {} pending | uploaded again at an address freed in the last 60 s: {} images ({} "
        "MiB, {:.1f} s after being freed on average) of {} MiB uploaded | buffer evictions {} ({} "
        "MiB, {:.1f} ms, unbind {:.1f} ms) | upload scratch in VRAM: released {} MiB ({} times), "
        "allocated {} MiB ({}), {} MiB held | VRAM {} of {} MiB, {} MiB of it in use (pressure "
        "{}, critical {}), system memory {} MiB ({:+} MiB)",
        seconds, current.passes - last.passes, pressure, aggressive,
        current.budget_limit_passes - last.budget_limit_passes, ms(current.ns - last.ns),
        ms(slowest_ns), ms(current.usage_query_ns - last.usage_query_ns), freed,
        mib(current.freed_bytes - last.freed_bytes), started,
        mib(current.writeback_bytes - last.writeback_bytes),
        mib(current.written_bytes - last.written_bytes),
        ms(current.writeback_done_ns - last.writeback_done_ns), pending_writebacks.size(),
        reuploads, mib(current.reupload_bytes - last.reupload_bytes), reupload_seconds,
        mib(current.uploaded_bytes - last.uploaded_bytes), evictions,
        mib(current.buffer_eviction_bytes - last.buffer_eviction_bytes),
        ms(current.buffer_eviction_ns - last.buffer_eviction_ns),
        ms(current.buffer_eviction_unbind_ns - last.buffer_eviction_unbind_ns),
        mib(current.scratch_released_bytes - last.scratch_released_bytes),
        current.scratch_releases - last.scratch_releases,
        mib(current.scratch_allocation_bytes - last.scratch_allocation_bytes),
        current.scratch_allocations - last.scratch_allocations,
        mib(runtime.GetStagingPool().DeviceLocalBytes()), mib(heaps.local_usage),
        mib(heaps.local_budget), mib(total_used_memory.load()), mib(pressure_gc_memory),
        mib(critical_gc_memory), mib(heaps.host_usage),
        (static_cast<s64>(heaps.host_usage) - static_cast<s64>(cleanup_host_usage)) / (1 << 20));
    const bool urgent = aggressive != 0 || slowest_ns >= 4'000'000 || reuploads >= 8;
    if (urgent) {
        LOG_WARNING(Render, "{}", text);
    } else {
        LOG_INFO(Render, "{}", text);
    }
}

// "Go past the budget": whether Windows makes room (the budget rises) or moves the emulator's
// own memory to system RAM (system memory rises), every 5 s while past the budget.
void TextureCache::ReportPastBudget() {
    if (!live_limits_active || !EmulatorSettings.IsBbVramPastBudget()) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - past_budget_report_time < std::chrono::seconds(5)) {
        return;
    }
    const auto heaps = instance.GetMemoryHeapReport();
    if (past_budget_first_budget == 0) {
        past_budget_first_budget = heaps.local_budget;
        past_budget_first_host = heaps.host_usage;
    }
    if (heaps.local_usage <= heaps.local_budget) {
        return;
    }
    past_budget_report_time = now;
    const auto mib = [](u64 bytes) { return static_cast<s64>(bytes >> 20); };
    LOG_INFO(Render,
             "Past the driver's VRAM budget by {} MiB: {} of {} MiB (budget {:+} MiB since the "
             "start, Windows making room), system memory {} MiB ({:+} MiB since the start, the "
             "emulator's memory moved out) | limit {} MiB, in use {} MiB",
             mib(heaps.local_usage) - mib(heaps.local_budget), mib(heaps.local_usage),
             mib(heaps.local_budget), mib(heaps.local_budget) - mib(past_budget_first_budget),
             mib(heaps.host_usage), mib(heaps.host_usage) - mib(past_budget_first_host),
             mib(live_limit), mib(total_used_memory.load()));
}

void TextureCache::UpdateVramLimits() {
    if (!vram_reclaim || !instance.CanReportMemoryUsage()) {
        return;
    }
    if (!standard_limits_saved) {
        standard_limits_saved = true;
        standard_trigger = trigger_gc_memory;
        standard_pressure = pressure_gc_memory;
        standard_critical = critical_gc_memory;
    }
    const auto mib = [](u64 bytes) { return bytes >> 20; };
    if (!more_vram_enabled) {
        if (live_limits_active) {
            live_limits_active = false;
            trigger_gc_memory = standard_trigger;
            pressure_gc_memory = standard_pressure;
            critical_gc_memory = standard_critical;
            LOG_INFO(Render,
                     "VRAM limits: the usual marks again (pressure {} MiB, critical {} MiB)",
                     mib(pressure_gc_memory), mib(critical_gc_memory));
        }
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (live_limits_active && now - limits_time < std::chrono::seconds(1)) {
        return;
    }
    limits_time = now;
    const auto limits = instance.GetLiveVramLimits();
    trigger_gc_memory = 0;
    pressure_gc_memory = limits.pressure;
    critical_gc_memory = limits.critical;
    live_limit = limits.limit;
    const u64 change =
        live_limit > logged_limit ? live_limit - logged_limit : logged_limit - live_limit;
    if (!live_limits_active || change >= 64_MB) {
        logged_limit = live_limit;
        LOG_INFO(Render,
                 "VRAM limits (Use more VRAM{}): driver budget {} MiB, so the limit is {} MiB, "
                 "pressure {} MiB, critical {} MiB (usual marks: pressure {}, critical {})",
                 EmulatorSettings.IsBbVramPastBudget() ? ", past the budget" : "",
                 mib(limits.budget), mib(limits.limit), mib(limits.pressure), mib(limits.critical),
                 mib(standard_pressure), mib(standard_critical));
    }
    live_limits_active = true;
}

// Cleanup at loading screens. The game's flag goes up as the transition starts, while the fade
// still draws the area; a second later only the loading screen draws. Images not used since then
// are the area's: the clean ones (copies of guest memory) are freed, a few hundred a frame, and
// so are buffer ranges idle since. GPU-written images stay (their data isn't in guest memory).
static constexpr auto LoadScreenMarkDelay = std::chrono::seconds(1);
static constexpr auto LoadScreenCleanupDelay = std::chrono::seconds(2);
static constexpr u64 LoadScreenImagesPerPass = 256;
static constexpr auto LoadScreenFreeAheadPause = std::chrono::seconds(15);

void TextureCache::LoadScreenCleanup() {
    const bool loading = Core::BbGameFixes::IsLoadingScreen();
    const auto now = std::chrono::steady_clock::now();
    const auto mib = [](u64 bytes) { return bytes >> 20; };
    LoadScreen& screen = load_screen;
    const bool enabled = vram_reclaim && load_screen_cleanup_enabled;
    if (loading && !screen.active) {
        screen = LoadScreen{
            .active = true,
            .start = now,
            .start_vram = instance.GetMemoryHeapReport().local_usage,
        };
        Common::Crumbs::Add("loading screen: started (VRAM MiB)", mib(screen.start_vram));
        LOG_INFO(Render, "Loading screen started | VRAM {} MiB{}", mib(screen.start_vram),
                 !vram_reclaim                  ? " (VRAM cleanup is off: nothing freed)"
                 : !load_screen_cleanup_enabled ? " (cleanup at loading screens is off)"
                                                : "");
        return;
    }
    if (!loading && screen.active) {
        screen.active = false;
        Common::Crumbs::Add("loading screen: ended");
        if (screen.reported) {
            free_ahead_paused_until = now + LoadScreenFreeAheadPause;
            load_until_second = 0;
        }
        LOG_INFO(Render, "Loading screen ended after {:.1f} s | VRAM {} MiB, {} MiB at its start{}",
                 std::chrono::duration<double>(now - screen.start).count(),
                 mib(instance.GetMemoryHeapReport().local_usage), mib(screen.start_vram),
                 enabled && !screen.reported ? " (too short for the cleanup to finish)" : "");
        return;
    }
    if (!screen.active || !enabled || screen.reported) {
        return;
    }
    if (!screen.marked) {
        if (now - screen.start >= LoadScreenMarkDelay) {
            screen.marked = true;
            screen.mark_tick = gc_tick;
        }
        return;
    }
    if (now - screen.start < LoadScreenCleanupDelay) {
        return;
    }
    if (!screen.images_done) {
        Common::Crumbs::Add("loading screen: freeing textures (freed so far)", screen.freed);
        const auto pass_start = std::chrono::steady_clock::now();
        u64 freed_now = 0;
        bool more = false;
        screen.kept_written = 0;
        screen.kept_written_bytes = 0;
        image_lru_cache.ForEachItemBelow(screen.mark_tick, [&](Image& image) {
            if (freed_now >= LoadScreenImagesPerPass) {
                more = true;
                return true;
            }
            if (image.HasGpuValidData()) {
                ++screen.kept_written;
                screen.kept_written_bytes += image.info.guest_size;
                return false;
            }
            ++freed_now;
            ++screen.freed;
            screen.freed_bytes += image.info.guest_size;
            FreeImage(slot_images.GetSlotId(image));
            return false;
        });
        screen.free_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now() - pass_start)
                              .count();
        ++screen.passes;
        screen.images_done = !more;
        if (!screen.images_done) {
            return;
        }
    }
    if (!screen.buffers_done) {
        Common::Crumbs::Add("loading screen: freeing idle buffers");
        screen.buffers_done = buffer_cache.EvictIdleBuffersAtLoadScreen();
        if (!screen.buffers_done) {
            return;
        }
        const auto released = runtime.GetStagingPool().ReleaseIdleDeviceLocal();
        screen.scratch_bytes = released.bytes;
        if (released.bytes != 0) {
            namespace GS = Common::GuestStats;
            GS::scratch_releases.fetch_add(1, std::memory_order_relaxed);
            GS::scratch_released_bytes.fetch_add(released.bytes, std::memory_order_relaxed);
        }
    }
    screen.reported = true;
    LOG_INFO(Render,
             "Loading screen cleanup: freed {} textures ({} MiB) in {:.1f} ms over {} frames, kept "
             "{} GPU-written ones ({} MiB), released {} MiB of idle upload scratch | VRAM {} MiB "
             "now, {} MiB when the loading screen started",
             screen.freed, mib(screen.freed_bytes), screen.free_ns / 1e6, screen.passes,
             screen.kept_written, mib(screen.kept_written_bytes), mib(screen.scratch_bytes),
             mib(instance.GetMemoryHeapReport().local_usage), mib(screen.start_vram));
}

// Every 10 s with VRAM cleanup on: where VRAM goes, how much of the cached textures the game used
// lately (if that is more than fits, no collector avoids uploading some again), and textures by
// size.
void TextureCache::LogVramBreakdown() {
    if (!vram_reclaim || !instance.CanReportMemoryUsage()) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    const u64 second = static_cast<u64>(
        std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count());
    if (second != breakdown_second) {
        if (breakdown_second == 0) {
            breakdown_tick_at_second.fill(gc_tick);
        }
        breakdown_second = second;
        breakdown_tick_at_second[second % breakdown_tick_at_second.size()] = gc_tick;
    }
    if (breakdown_time == std::chrono::steady_clock::time_point{}) {
        breakdown_time = now;
        return;
    }
    if (now - breakdown_time < std::chrono::seconds(10)) {
        return;
    }
    breakdown_time = now;
    const auto tick_seconds_ago = [&](u64 seconds) {
        return breakdown_tick_at_second[(second - seconds) % breakdown_tick_at_second.size()];
    };
    const u64 tick_5 = tick_seconds_ago(5);
    const u64 tick_30 = tick_seconds_ago(30);
    const u64 tick_60 = tick_seconds_ago(60);

    u64 clean_bytes = 0, written_bytes = 0, image_count = 0;
    u64 used_5 = 0, used_30 = 0, used_60 = 0;
    std::array<u64, SizeClasses> cached_count{}, cached_bytes{};
    image_lru_cache.ForEachItemBelow(gc_tick, [&](Image& image) {
        u64 bytes = 0;
        for (const auto& backing : image.backing_images) {
            bytes += backing.image.size_bytes;
        }
        ++image_count;
        (image.HasGpuValidData() ? written_bytes : clean_bytes) += bytes;
        const u64 last_use =
            std::atomic_ref<u64>(image.last_use_gc_tick).load(std::memory_order_relaxed);
        used_5 += last_use >= tick_5 ? bytes : 0;
        used_30 += last_use >= tick_30 ? bytes : 0;
        used_60 += last_use >= tick_60 ? bytes : 0;
        const size_t size_class = SizeClassOf(bytes);
        ++cached_count[size_class];
        cached_bytes[size_class] += bytes;
    });

    const auto mib = [](u64 bytes) { return bytes >> 20; };
    const auto heaps = instance.GetMemoryHeapReport();
    const auto allocator = instance.GetAllocatorUsage();
    const auto pool = buffer_cache.GetPoolUsage();
    const u64 images = clean_bytes + written_bytes;
    const u64 other_allocator = allocator.in_use - std::min(allocator.in_use, images);
    const u64 known = allocator.held + pool.allocated;
    const s64 rest = static_cast<s64>(heaps.local_usage) - static_cast<s64>(known);
    LOG_INFO(Render,
             "VRAM breakdown: {} of {} MiB (driver), {} MiB in system memory | textures {} MiB "
             "in {} (copies of game memory {} MiB, GPU-written {} MiB), other emulator memory {} "
             "MiB (upload scratch {} MiB), free inside the allocator {} MiB | guest buffers {} MiB "
             "in {} MiB of slabs "
             "({} MiB free inside) | driver and the rest {} MiB | freed, released once the GPU "
             "is done {} MiB | in use {} MiB (pressure {}, critical {})",
             mib(heaps.local_usage), mib(heaps.local_budget), mib(heaps.host_usage), mib(images),
             image_count, mib(clean_bytes), mib(written_bytes), mib(other_allocator),
             mib(runtime.GetStagingPool().DeviceLocalBytes()),
             mib(allocator.held - std::min(allocator.held, allocator.in_use)),
             mib(buffer_cache.ResidentBytes()), mib(pool.allocated), mib(pool.free),
             rest / (1 << 20), mib(pending_free_bytes.load(std::memory_order_relaxed)),
             mib(total_used_memory.load()), mib(pressure_gc_memory), mib(critical_gc_memory));

    std::string cached;
    std::string uploaded;
    for (size_t i = 0; i < SizeClasses; ++i) {
        const u64 count = upload_count_by_size[i].exchange(0, std::memory_order_relaxed);
        const u64 bytes = upload_bytes_by_size[i].exchange(0, std::memory_order_relaxed);
        fmt::format_to(std::back_inserter(cached), "{}{} {} ({} MiB)", i ? ", " : "",
                       SizeClassNames[i], cached_count[i], mib(cached_bytes[i]));
        fmt::format_to(std::back_inserter(uploaded), "{}{} {} ({} MiB)", i ? ", " : "",
                       SizeClassNames[i], count, mib(bytes));
    }
    LOG_INFO(Render,
             "VRAM textures: used in the last 5 s {} MiB, 30 s {} MiB, 60 s {} MiB, of {} MiB "
             "cached | cached by size: {} | uploaded in the last 10 s by size: {}",
             mib(used_5), mib(used_30), mib(used_60), mib(images), cached, uploaded);

    u64 tracked = 0;
    std::string by_owner;
    for (const auto& owner : VramOwners::Snapshot()) {
        tracked += owner.bytes;
        if (owner.bytes >= 1_MB) {
            fmt::format_to(std::back_inserter(by_owner), "{}{} {} MiB ({})",
                           by_owner.empty() ? "" : ", ", owner.name, mib(owner.bytes), owner.count);
        }
    }
    LOG_INFO(Render,
             "VRAM by owner: {} MiB of the allocator's {} MiB in use, the rest not tracked {} "
             "MiB | {}",
             mib(tracked), mib(allocator.in_use),
             mib(allocator.in_use - std::min(allocator.in_use, tracked)), by_owner);
    if (quality_report_count++ % 3 == 0) {
        LogTextureQualityEstimate(tick_30);
    }
}

// Estimate only (nothing changes): what a texture quality setting would save. Each step leaves
// out a texture's sharpest level, so it is stored at half the width and height: "High" for
// textures larger than 2048, "Medium" also for those larger than 1024. Only textures the game
// samples qualify: not drawn into, written by the GPU, used for storage or shown, with a smaller
// level to take the top one's place. Whether a shader reads one by texel position or asks its
// size (those would need the full one) is not known here.
static bool IsQualityCandidate(const Image& image) {
    const auto& info = image.info;
    if (image.usage.render_target || image.usage.depth_target || image.usage.storage ||
        image.usage.vo_surface || info.props.is_depth || image.HasGpuValidData()) {
        return false;
    }
    return info.resources.levels >= 2 && info.guest_size != 0 &&
           std::max(info.size.width, info.size.height) > 1024;
}

static std::string DescribeExactTexelReads(u8 reads) {
    std::string text;
    const auto add = [&](u8 bit, std::string_view name) {
        if (reads & bit) {
            fmt::format_to(std::back_inserter(text), "{}{}", text.empty() ? "" : ", ", name);
        }
    };
    add(Shader::ExactTexelFetch, "texel fetch");
    add(Shader::ExactTexelSizeQuery, "size query");
    add(Shader::ExactTexelLodQuery, "LOD query");
    add(Shader::ExactTexelUnnormalized, "pixel coordinates");
    return text;
}

void TextureCache::NoteExactTexelReads(ImageId image_id, u8 reads, u64 pgm_hash) {
    Image& image = slot_images[image_id];
    std::atomic_ref<u8> image_reads{image.exact_texel_reads};
    const u8 before = image_reads.fetch_or(reads, std::memory_order_relaxed);
    if ((before | reads) == before || !IsQualityCandidate(image)) {
        return;
    }
    if (image.dropped_levels != 0) {
        KeepFullQuality(image.info.guest_address);
        std::scoped_lock lock{restore_mutex};
        restore_full_quality.emplace_back(image_id, image.image_uid);
    }
    LOG_INFO(Render,
             "Texture quality: {}x{} {} ({} levels) at {:#x} needs its exact size, read by "
             "shader {:#x} with {}",
             image.info.size.width, image.info.size.height, vk::to_string(image.info.pixel_format),
             image.info.resources.levels, image.info.guest_address, pgm_hash,
             DescribeExactTexelReads(reads & ~before));
}

void TextureCache::RestoreFullQuality() {
    std::vector<std::pair<ImageId, u64>> pending;
    {
        std::scoped_lock lock{restore_mutex};
        pending.swap(restore_full_quality);
    }
    for (const auto& [image_id, uid] : pending) {
        if (!slot_images.IsAllocated(image_id)) {
            continue;
        }
        const Image& image = slot_images[image_id];
        if (image.image_uid != uid || False(image.flags & ImageFlagBits::Registered)) {
            continue;
        }
        LOG_INFO(Render, "Texture quality: {}x{} at {:#x} loaded again at full size",
                 image.info.size.width, image.info.size.height, image.info.guest_address);
        FreeImage(image_id);
    }
}

void TextureCache::LogTextureQualityEstimate(u64 tick_30) {
    struct Group {
        u64 count{};
        u64 bytes{};
        u64 saved{};
        u64 recent_saved{};
    };
    Group high, medium, medium_safe, medium_exact, reduced;
    std::array<u64, 4> exact_saved_by_read{};
    u64 total = 0, recent_total = 0;
    u64 targets = 0, gpu_written = 0, single_level = 0, small = 0;
    std::map<std::string, Group> formats;
    image_lru_cache.ForEachItemBelow(gc_tick, [&](Image& image) {
        u64 bytes = 0;
        for (const auto& backing : image.backing_images) {
            bytes += backing.image.size_bytes;
        }
        total += bytes;
        const u64 last_use =
            std::atomic_ref<u64>(image.last_use_gc_tick).load(std::memory_order_relaxed);
        const bool recent = last_use >= tick_30;
        recent_total += recent ? bytes : 0;
        const auto& info = image.info;
        if (image.dropped_levels != 0) {
            const u64 kept_guest = info.guest_size - info.mips_layout[0].size;
            const u64 saved = kept_guest ? bytes * info.mips_layout[0].size / kept_guest : 0;
            ++reduced.count;
            reduced.bytes += bytes;
            reduced.saved += saved;
            reduced.recent_saved += recent ? saved : 0;
            return;
        }
        if (image.usage.render_target || image.usage.depth_target || image.usage.storage ||
            image.usage.vo_surface || info.props.is_depth) {
            targets += bytes;
            return;
        }
        if (image.HasGpuValidData()) {
            gpu_written += bytes;
            return;
        }
        if (info.resources.levels < 2 || info.guest_size == 0) {
            single_level += bytes;
            return;
        }
        const u32 largest = std::max(info.size.width, info.size.height);
        if (largest <= 1024) {
            small += bytes;
            return;
        }
        const u64 saved = bytes * info.mips_layout[0].size / info.guest_size;
        const auto add = [&](Group& group) {
            ++group.count;
            group.bytes += bytes;
            group.saved += saved;
            group.recent_saved += recent ? saved : 0;
        };
        add(medium);
        const u8 reads =
            std::atomic_ref<u8>(image.exact_texel_reads).load(std::memory_order_relaxed);
        add(reads ? medium_exact : medium_safe);
        for (u32 bit = 0; bit < exact_saved_by_read.size(); ++bit) {
            exact_saved_by_read[bit] += (reads >> bit) & 1 ? saved : 0;
        }
        if (largest > 2048) {
            add(high);
        }
        add(formats[fmt::format("{}x{} {}", info.size.width, info.size.height,
                                vk::to_string(info.pixel_format))]);
    });
    std::vector<std::pair<std::string, Group>> sorted(formats.begin(), formats.end());
    std::ranges::sort(sorted, std::greater{}, [](const auto& entry) { return entry.second.bytes; });
    const auto mib = [](u64 bytes) { return bytes >> 20; };
    std::string top;
    for (size_t i = 0; i < std::min<size_t>(sorted.size(), 8); ++i) {
        const auto& [name, group] = sorted[i];
        fmt::format_to(std::back_inserter(top), "{}{} x{} ({} MiB)", i ? ", " : "", name,
                       group.count, mib(group.bytes));
    }
    LOG_INFO(Render,
             "Texture quality estimate (nothing changed): High would halve {} textures ({} MiB) "
             "and save {} MiB, Medium {} textures ({} MiB) saving {} MiB, of {} MiB cached | of "
             "the {} MiB used in the last 30 s, High saves {} MiB, Medium {} MiB | left out: "
             "render targets, depth and storage {} MiB, GPU-written {} MiB, one level only {} "
             "MiB, 1024 or smaller {} MiB | largest candidates: {}",
             high.count, mib(high.bytes), mib(high.saved), medium.count, mib(medium.bytes),
             mib(medium.saved), mib(total), mib(recent_total), mib(high.recent_saved),
             mib(medium.recent_saved), mib(targets), mib(gpu_written), mib(single_level),
             mib(small), top);
    LOG_INFO(Render,
             "Texture quality estimate, shader reads: of Medium's {} MiB saving, {} MiB from {} "
             "textures only sampled (safe to halve), {} MiB from {} textures some shader needs at "
             "their exact size: texel fetch {} MiB, size query {} MiB, LOD query {} MiB, pixel "
             "coordinates {} MiB (a texture can count under several)",
             mib(medium.saved), mib(medium_safe.saved), medium_safe.count, mib(medium_exact.saved),
             medium_exact.count, mib(exact_saved_by_read[0]), mib(exact_saved_by_read[1]),
             mib(exact_saved_by_read[2]), mib(exact_saved_by_read[3]));
    if (const int quality = EmulatorSettings.GetBbTextureQuality(); quality > 0 || reduced.count) {
        const char* quality_name = quality >= 2 ? "Medium" : quality == 1 ? "High" : "Full";
        LOG_INFO(Render,
                 "Texture quality {}: {} textures stored without their largest level ({} MiB), "
                 "saving about {} MiB ({} MiB of it in textures used in the last 30 s); the "
                 "estimate above counts only the textures still at full size",
                 quality_name, reduced.count, mib(reduced.bytes), mib(reduced.saved),
                 mib(reduced.recent_saved));
    }
}

TextureCache::MemoryReport TextureCache::TakeMemoryReport() {
    MemoryReport report{};
    const auto group_of = [&report](const Image& image) -> MemoryReport::Group& {
        if (!image.HasGpuValidData()) {
            return report.clean;
        }
        return image.info.IsTiled() ? report.stuck : report.written;
    };
    const auto image_bytes = [](const Image& image) {
        u64 bytes = 0;
        for (const auto& backing : image.backing_images) {
            bytes += backing.image.size_bytes;
        }
        return bytes;
    };
    image_lru_cache.ForEachItemBelow(gc_tick, [&](Image& image) {
        const u64 bytes = image_bytes(image);
        auto& group = group_of(image);
        ++group.count;
        group.bytes += bytes;
        if (image.backing) {
            report.extra_backing_bytes += bytes - image.backing->image.size_bytes;
        }
    });
    // The collector only looks at images untouched for at least 16 ticks.
    if (gc_tick >= 16) {
        image_lru_cache.ForEachItemBelow(gc_tick - 16, [&](Image& image) {
            auto& group = group_of(image);
            ++group.stale_count;
            group.stale_bytes += image_bytes(image);
        });
    }
    report.allocated_images = slot_images.Size();
    report.trigger = trigger_gc_memory;
    report.pressure = pressure_gc_memory;
    report.critical = critical_gc_memory;
    report.gc_used = total_used_memory;
    report.evicted = std::exchange(gc_evicted, 0);
    report.written_back = std::exchange(gc_written_back, 0);
    report.skipped_stuck = std::exchange(gc_skipped_stuck, 0);
    report.skipped_written = std::exchange(gc_skipped_written, 0);
    report.gc_runs = std::exchange(gc_runs, 0);
    report.loads = std::exchange(gc_loads, 0);
    report.spread_loads = std::exchange(spread_loads, 0);
    report.spread_deferred = std::exchange(spread_deferred, 0);
    report.spread_zero_filled = std::exchange(spread_zero_filled, 0);
    report.spread_zero_bytes = std::exchange(spread_zero_bytes, 0);
    return report;
}

void TextureCache::GarbageCollectSamplers() {
    total_used_samplers = samplers.size();
    if (total_used_samplers < trigger_gc_samplers) {
        return;
    }
    bool pressured = false;
    bool aggresive = false;
    u64 ticks_to_destroy = 0;
    size_t num_deletions = 0;

    const auto configure = [&](bool allow_aggressive) {
        pressured = total_used_samplers >= pressure_gc_samplers;
        aggresive = allow_aggressive && total_used_samplers >= critical_gc_samplers;
        ticks_to_destroy = aggresive ? 160 : pressured ? 80 : 16;
        ticks_to_destroy = std::min(ticks_to_destroy, gc_tick);
        num_deletions = aggresive ? 40 : pressured ? 20 : 10;
    };
    const auto clean_up = [&](Sampler& sampler) {
        if (num_deletions == 0) {
            return true;
        }
        --num_deletions;
        sampler_generation.fetch_add(1, std::memory_order_acq_rel);
        sampler_lru_cache.Free(sampler);
        samplers.erase(sampler.hash);
        const auto sampler_id = slot_samplers.GetSlotId(sampler);
        slot_samplers.Erase(sampler_id);
        return false;
    };

    // Try to remove anything old enough and not high priority.
    configure(false);
    sampler_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);

    if (total_used_samplers >= critical_gc_samplers) {
        // If we are still over the critical limit, run an aggressive GC
        configure(true);
        sampler_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);
    }
}

void TextureCache::RunGarbageCollector() {
    spread_loads_enabled = EmulatorSettings.IsBbSpreadLoads();
    load_screen_cleanup_enabled = EmulatorSettings.IsBbLoadScreenCleanup();
    more_vram_enabled = EmulatorSettings.IsBbMoreVram();
    SCOPE_EXIT {
        // Atomic: UpdateImage's fast path reads it without the lock.
        std::atomic_ref<u64>(gc_tick).store(gc_tick + 1, std::memory_order_release);
    };

    RestoreFullQuality();
    if (Common::GuestStats::texture_reload_requested.exchange(false, std::memory_order_relaxed))
        [[unlikely]] {
        ReloadTexturesFromGuest();
    }
    LoadScreenCleanup();
    GarbageCollectImages();
    LogCleanupReport();
    LogVramBreakdown();
    GarbageCollectSamplers();
}

void TextureCache::ReloadTexturesFromGuest() {
    u32 reloaded = 0, gpu_written = 0, pending = 0, pending_used = 0;
    u64 reloaded_bytes = 0;
    const u64 now = SteadyNowNs();
    image_lru_cache.ForEachItemBelow(gc_tick + 1, [&](Image& image) {
        if (False(image.flags & ImageFlagBits::Registered)) {
            return false;
        }
        const ImageId image_id = slot_images.GetSlotId(image);
        const bool used_recently =
            gc_tick - std::atomic_ref<u64>(image.last_use_gc_tick).load(std::memory_order_relaxed) <
            120;
        if (image.detail_pending) {
            ++pending;
            pending_used += used_recently;
            if (pending <= 32) {
                LOG_INFO(Render_Vulkan,
                         "Texture reload: {:#x} {}x{} {} ({} levels, {} KiB) still waits for its "
                         "largest level after {} ms{}",
                         image.info.guest_address, image.info.size.width, image.info.size.height,
                         vk::to_string(image.info.pixel_format), image.info.resources.levels,
                         image.info.guest_size / 1024,
                         (now - image.detail_pending_since_ns) / 1'000'000,
                         used_recently ? ", used in the last frames" : "");
            }
        }
        if (True(image.flags & ImageFlagBits::GpuModified)) {
            ++gpu_written;
            return false;
        }
        std::scoped_lock lk{image.mutex};
        image.flags |= ImageFlagBits::CpuDirty;
        ClearCleanTick(image);
        UntrackImage(image_id);
        ++reloaded;
        reloaded_bytes += image.info.guest_size;
        return false;
    });
    LOG_INFO(Render_Vulkan,
             "Texture reload: {} textures ({} MiB) will be uploaded again from game memory as "
             "they are used, {} written by the GPU kept | {} waiting for their largest level ({} "
             "of them used in the last frames)",
             reloaded, reloaded_bytes >> 20, gpu_written, pending, pending_used);
}

void TextureCache::DeleteImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    Common::Crumbs::Add("image deleted (address, size)", image.info.guest_address,
                        image.info.guest_size);
    ASSERT_MSG(image.IsUntracked(), "Image was not untracked");
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered), "Image was not unregistered");

    // Remove any registered meta areas.
    const auto& meta_info = image.info.meta_info;
    if (meta_info.cmask_addr) {
        UnregisterMeta(meta_info.cmask_addr, image.image_uid);
    }
    if (meta_info.fmask_addr) {
        UnregisterMeta(meta_info.fmask_addr, image.image_uid);
    }
    if (meta_info.htile_addr) {
        UnregisterMeta(meta_info.htile_addr, image.image_uid);
    }

    {
        std::unique_lock lk{download_images_mutex};
        if (download_images.contains(image_id)) {
            download_images.erase(image_id);
        }
    }

    // Reclaim image and any image views it references.
    u64 image_bytes = 0;
    for (const auto& backing : image.backing_images) {
        image_bytes += backing.image.size_bytes;
    }
    pending_free_bytes.fetch_add(image_bytes, std::memory_order_relaxed);
    scheduler.DeferOperation([this, image_id, image_bytes] {
        Image& image = slot_images[image_id];
        for (auto& backing : image.backing_images) {
            for (const ImageViewId image_view_id : backing.image_view_ids) {
                slot_image_views.Erase(image_view_id);
            }
        }
        slot_images.Erase(image_id);
        pending_free_bytes.fetch_sub(image_bytes, std::memory_order_relaxed);
    });
}

} // namespace VideoCore
