// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <deque>
#include <map>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <boost/container/small_vector.hpp>

#include "common/interval_set.h"
#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/fault_manager.h"
#include "video_core/buffer_cache/range_set.h"
#include "video_core/renderer_vulkan/vk_semaphore.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {
class GraphicsPipeline;
struct SubmitInfo;
class Runtime;
class StagingBufferPool;
} // namespace Vulkan

namespace VideoCore {

class TextureCache;
class MemoryTracker;
class PageManager;

class BufferCache {
    static constexpr u64 ADDRESS_SPACE_BITS = 40;
    static constexpr u64 ARENA_PAGE_BITS = 32;
    static constexpr u64 ARENA_PAGE_SIZE = u64{1} << ARENA_PAGE_BITS;
    static constexpr u64 NUM_ARENA_PAGES = u64{1} << (ADDRESS_SPACE_BITS - ARENA_PAGE_BITS);
    static constexpr u64 MIN_BLOCK_SIZE = 16_KB;
    static constexpr u64 STREAM_THRESHOLD = 16_KB;

public:
    /// Read-only buffers up to this size are copied into the stream buffer on every binding.
    static constexpr u64 StreamThreshold() {
        return STREAM_THRESHOLD;
    }

    explicit BufferCache(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         Vulkan::Runtime& runtime, AmdGpu::Liverpool* liverpool,
                         TextureCache& texture_cache, PageManager& tracker);
    ~BufferCache();

    /// Returns a pointer to GDS device local buffer.
    [[nodiscard]] const Buffer* GetGdsBuffer() const noexcept {
        return &gds_buffer;
    }

    [[nodiscard]] const Buffer* GetNullIndexBuffer() const noexcept {
        return null_index_buffer.get();
    }

    /// Retrieves the device local DBA page table buffer.
    [[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept {
        return bda_pagetable_buffer.get();
    }

    /// Retrieves the fault buffer.
    [[nodiscard]] Buffer* GetFaultBuffer() noexcept {
        return fault_manager->GetFaultBuffer();
    }

    /// Retrieves the stream buffer.
    [[nodiscard]] StreamBuffer& GetStreamBuffer() noexcept {
        return stream_buffer;
    }

    /// Returns minimum granularity of a sparse memory bind.
    u32 GetSparsePageShift() const noexcept {
        return block_shift;
    }

    void TickFrame();

    /// Loading screen: frees buffer memory idle since before it. False: try again next frame.
    bool EvictIdleBuffersAtLoadScreen() {
        return GarbageCollectResidentMemory(true);
    }

    /// Invalidates any buffer in the logical page range.
    void InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks = false);

    /// After a guest write fault at device_addr: also unprotects the rest of the aligned
    /// window around it (see RegionManager::ExtendWriteFault).
    void ExtendWriteFault(VAddr device_addr);

    /// Flushes any GPU modified buffer in the logical page range back to CPU memory.
    void ReadMemory(VAddr device_addr, u64 size, bool is_write = false, bool assume_locks = false);

    /// Diagnostics: GPU readbacks since the last call (where, who, how long, and whether the
    /// GPU work that wrote the data had been submitted or had finished), logged every 10 s.
    void LogReadbackReport();

    /// Early write-back (threaded renderer): memory a game thread had to wait for a readback of
    /// is "hot" for a while. GPU work writing hot memory asks for its batch to be sent at once,
    /// and once the GPU has finished it the data is copied back to guest memory, so the game's
    /// next write finds it there instead of waiting for the GPU.
    void SetEarlyWriteback(bool enabled) noexcept {
        early_writeback.store(enabled, std::memory_order_relaxed);
    }
    [[nodiscard]] bool EarlyWriteback() const noexcept {
        return early_writeback.load(std::memory_order_relaxed);
    }
    /// The batch being recorded writes hot memory (cleared by the call).
    [[nodiscard]] bool TakeWritebackFlush() noexcept {
        const bool wanted = writeback_flush.exchange(false, std::memory_order_acq_rel);
        writeback_flushes.fetch_add(wanted ? 1 : 0, std::memory_order_relaxed);
        return wanted;
    }
    struct PendingWriteback {
        VAddr address;
        u64 size;
        u64 tick;
    };
    /// Hot writes whose GPU work has finished (removed from the pending list).
    [[nodiscard]] std::vector<PendingWriteback> TakeFinishedWritebacks();
    /// Copies a finished hot write back to guest memory (GPU command thread, draw pipe idle).
    void WriteBack(const PendingWriteback& writeback);

    /// Finds a buffer for the specified region.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBuffer(VAddr device_addr, u32 size,
                                                             bool is_written,
                                                             bool is_texel_buffer = false);

    /// Vertex and index data: like ObtainBuffer(read-only), but with the threaded renderer a
    /// clean range of 1 KiB or more is used where it is in the arena instead of being copied
    /// into the stream buffer for every draw (static meshes keep their geometry).
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainGeometryBuffer(VAddr device_addr, u32 size);

    /// Attempts to obtain a buffer without modifying the cache contents.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBufferForImage(VAddr device_addr, u32 size);

    /// Writes a GPU-produced buffer range back to emulated memory.
    bool DownloadMemoryFromBuffer(const Buffer* arena, VAddr device_addr, u64 size);

    /// Return true when a region is modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(VAddr addr, size_t size);

    /// Return true when a region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(VAddr addr, size_t size);

    /// Synchronizes all buffers needed for DMA.
    void SynchronizeDmaBuffers();

    /// VRAM made resident for guest buffers so far (it's never released).
    [[nodiscard]] u64 ResidentBytes() const noexcept {
        return resident_bytes.load(std::memory_order_relaxed);
    }

    /// Buffer eviction's memory slabs: allocated, and of it free for reuse (the driver counts all
    /// of it as used). Call on the thread running the collectors.
    struct PoolUsage {
        u64 allocated{};
        u64 free{};
    };
    [[nodiscard]] PoolUsage GetPoolUsage() const {
        PoolUsage usage{};
        for (const auto& [raw, slab] : resident_slabs) {
            usage.allocated += slab.size_blocks << block_shift;
            usage.free += (slab.size_blocks - slab.live_blocks) << block_shift;
        }
        return usage;
    }

    /// Commits pending sparse buffer memory binds. Must be called before every scheduler submit.
    void SubmitPendingArenaBinds(Vulkan::SubmitInfo& info);

private:
    struct ArenaBinds {
        const Buffer* arena;
        boost::container::small_vector<vk::SparseMemoryBind, 32> binds;
    };

    ArenaBinds* BindsForArena(const Buffer* arena) {
        auto it = std::ranges::find(pending_binds, arena, &ArenaBinds::arena);
        if (it != pending_binds.end()) {
            return std::addressof(*it);
        }
        return &pending_binds.emplace_back(arena);
    }

    const Buffer* GetArena(u64 first_block, u64 last_block);

    void EnsureResident(const Buffer* arena, u64 first_block, u64 last_block);

    /// load_screen: at a loading screen, every buffer range idle for a couple of seconds goes,
    /// whatever the VRAM use. False when it couldn't run yet (binds still pending).
    bool GarbageCollectResidentMemory(bool load_screen = false);

    bool DownloadMemory(const Buffer* arena, VAddr device_addr, u64 size);

    bool SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size, bool is_written,
                           bool is_texel_buffer);

    bool SynchronizeMemoryFromImage(const Buffer* arena, VAddr device_addr, u32 size);

    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    Vulkan::Runtime& runtime;
    Vulkan::StagingBufferPool& staging_pool;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    TextureCache& texture_cache;
    std::unique_ptr<MemoryTracker> memory_tracker;

    StreamBuffer stream_buffer;
    Buffer gds_buffer;
    std::unique_ptr<Buffer> null_index_buffer;
    RangeSet gpu_modified_ranges;

    std::unique_ptr<FaultManager> fault_manager;
    std::unique_ptr<Buffer> bda_pagetable_buffer;
    bool fault_process_pending{};

    std::array<const Buffer*, NUM_ARENA_PAGES> address_space{};
    std::deque<Buffer> arenas;
    std::vector<ArenaBinds> pending_binds;
    Vulkan::Semaphore memory_semaphore;

    struct Backing : public Interval {
        vk::DeviceMemory memory;
        u64 offset;
        constexpr bool CanMergeWith(const Backing& other) const noexcept {
            return memory == other.memory && offset + (end - start) == other.offset;
        }
        constexpr Backing SubRange(u64 a, u64 b) const noexcept {
            return {{a, b}, memory, offset + (a - start)};
        }
    };
    class ResidentRangeList : public IntervalList<Backing> {
    public:
        void Remove(u64 start, u64 end) {
            auto [first, last] = OverlapRun(start, end);
            if (first == last) {
                return;
            }
            Backing remaining[2];
            size_t count = 0;
            if (first->start < start) {
                remaining[count++] = first->SubRange(first->start, start);
            }
            const auto previous = std::prev(last);
            if (previous->end > end) {
                remaining[count++] = previous->SubRange(end, previous->end);
            }
            const auto index = std::distance(intervals.begin(), first);
            Splice(first, last, remaining, count);
            Coalesce(index, count);
        }
    };
    ResidentRangeList resident_ranges;
    std::unordered_map<u64, u64> resident_chunk_use; ///< 2 MiB chunk -> second of last use
    /// Chunks already marked used this second (MarkResidentUse skips the map write).
    struct ChunkUseMemo {
        u64 chunk{~0ull};
        u64 second{};
    };
    std::array<ChunkUseMemo, 1024> chunk_use_memo{};
    /// Block ranges known to be fully resident (EnsureResident skips the gap search). Valid
    /// while resident_generation is unchanged: it changes whenever residency is taken away.
    struct ResidentMemo {
        u64 first_block{~0ull};
        u64 last_block{};
        u64 generation{};
    };
    std::array<ResidentMemo, 1024> resident_memo{};
    u64 resident_generation{1};
    /// Buffer eviction: resident memory slabs, handed out and taken back block by block.
    struct ResidentSlab {
        u64 size_blocks{};
        u64 live_blocks{};
        std::map<u64, u64> free; ///< free runs: first block -> count
    };
    std::unordered_map<VkDeviceMemory, ResidentSlab> resident_slabs;
    struct ResidentBlocks {
        vk::DeviceMemory memory;
        u64 start;
        u64 count;
    };
    void MakeResidentPooled(const Buffer* arena, const IntervalList<>& bind_ranges,
                            u64 resident_blocks);
    ResidentBlocks TakeResidentBlocks(u64 wanted);
    /// Allocates resident memory; when the driver refuses, the VRAM state goes to the log.
    vk::DeviceMemory AllocateResidentMemory(const vk::MemoryAllocateInfo& alloc_info,
                                            const char* pool);
    /// Returns the slab's size in blocks when nothing of it is bound any more (to be freed).
    u64 ReturnResidentBlocks(vk::DeviceMemory memory, u64 start, u64 count);
    void MarkResidentUse(VAddr device_addr, u64 size);
    u64 cache_second{};
    struct RetiredMemory {
        vk::DeviceMemory memory;
        u64 second;
        u64 unbind_tick; ///< memory_semaphore tick of its last unbind
    };
    u64 last_unbind_tick{}; ///< memory_semaphore tick of the last unbind (buffer eviction)
    std::chrono::steady_clock::time_point last_slow_bind_report{};
    std::vector<RetiredMemory> retired_memory; ///< evicted, freed a few seconds later
    bool vram_reclaim{};                       ///< VRAM cleanup setting, read at start
    u64 cache_tick{};
    u64 last_resident_gc_tick{};
    u64 last_resident_diag_tick{};

    /// Readback diagnostics (diagnostics setting only).
    struct ReadbackRegion {
        u64 count{}, bytes{}, gpu_wait_ns{}, guest_blocked_ns{};
        std::array<u64, 4> writer{}; ///< current batch, in flight, finished, unknown
        std::map<std::string, u64> threads;
    };
    std::mutex readback_mutex;
    std::unordered_map<u64, ReadbackRegion> readback_regions; ///< by 1 MiB region
    std::array<u64, 4> readback_writer_ns{};                  ///< GPU wait time by writer state
    u64 readback_max_wait_ns{};
    std::mutex gpu_write_mutex;
    std::unordered_map<u64, u64> gpu_write_ticks;    ///< 64 KiB block -> tick of the last GPU write
    u64 readback_queue_count{}, readback_queue_ns{}; ///< readbacks on the second queue (report)

    /// Early write-back state (see SetEarlyWriteback), under writeback_mutex.
    std::atomic<bool> early_writeback{false};
    std::atomic<bool> writeback_flush{false};
    std::mutex writeback_mutex;
    std::unordered_map<u64, std::chrono::steady_clock::time_point> hot_blocks; ///< 64 KiB blocks
    std::vector<PendingWriteback> pending_writebacks;
    u64 writebacks_done{}, writeback_bytes{}; ///< since the last report (writeback_mutex)
    std::atomic<u64> writeback_flushes{};
    void MarkHot(VAddr device_addr, u64 size);
    void NoteHotWrite(VAddr device_addr, u64 size);

    /// Notes that GPU work recorded now writes [device_addr, device_addr + size).
    void NoteGpuWrite(VAddr device_addr, u64 size);
    /// Tick of the last recorded GPU write into the range (0: none known).
    u64 LastGpuWriteTick(VAddr device_addr, u64 size);
    /// Copies `copies` from the arena into the download buffer on the readback queue once the
    /// work signalled with `tick` is done, and waits for the copy alone.
    void ReadbackOnSecondQueue(const Buffer* arena, const Buffer* destination,
                               std::span<const vk::BufferCopy> copies, u64 tick);
    vk::UniqueCommandPool readback_pool;
    vk::UniqueCommandBuffer readback_cmdbuf;
    vk::UniqueFence readback_fence;
    std::mutex readback_queue_mutex;
    std::chrono::steady_clock::time_point readback_report_time{};
    std::atomic<u64> resident_bytes{};
    static constexpr u64 ResidentSlabSize = 64ULL << 20;
    static constexpr u64 EvictionSlabSize = 16ULL << 20;
    vk::DeviceMemory resident_slab{};
    u64 resident_slab_size{};
    u64 resident_slab_used{};

    u32 arena_memory_type_index{};
    u32 block_size{};
    u32 block_shift{};
    u32 blocks_per_arena_page{};
    u32 blocks_per_arena_page_shift{};
};

} // namespace VideoCore
