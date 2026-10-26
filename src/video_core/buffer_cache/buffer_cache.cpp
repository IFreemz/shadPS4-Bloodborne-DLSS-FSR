// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <optional>
#include <unordered_set>
#include <magic_enum/magic_enum.hpp>

#include "common/alignment.h"
#include "common/bb_tuning.h"
#include "common/copy_pool.h"
#include "common/crash_crumbs.h"
#include "common/game_waits.h"
#include "common/guest_stats.h"
#include "common/scope_exit.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/buffer_cache/region_definitions.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

#include <vk_mem_alloc.h>

namespace VideoCore {

static constexpr size_t GDS_BUFFER_SIZE = 64_KB;
static constexpr size_t NULL_INDEX_BUFFER_SIZE = 16;
static constexpr size_t STREAM_BUFFER_SIZE = 128_MB;
// Background texture copies (test setting): smaller copies are done in place.
static constexpr u64 BackgroundCopyMin = 16_KB;
static constexpr u64 BackgroundCopyPiece = 1_MB;
// Buffer uploads at least this large go to the copy threads (streamed geometry; the small
// per-frame updates are cheaper to copy in place).
static constexpr u64 BackgroundBufferCopyMin = 64_KB;
// Buffer eviction: the "Also free idle buffers" setting (on by default, with VRAM cleanup) or
// SHADPS4_BUFFER_EVICTION=1.
// It crashed with the threaded renderer when it evicted memory unused for 300 frames; this
// version only takes guest memory unused for RESIDENT_BUFFER_IDLE_SECONDS that holds no
// GPU-written data, and frees an allocation RESIDENT_BUFFER_FREE_DELAY_SECONDS after the last
// range in it was unbound.
static bool EvictIdleBuffers() {
    static const bool on = [] {
        const char* value = std::getenv("SHADPS4_BUFFER_EVICTION");
        return value ? value[0] == '1' : EmulatorSettings.IsBbBufferEviction();
    }();
    return on;
}
static constexpr u64 RESIDENT_BUFFER_IDLE_SECONDS = 30;
// Use is tracked, and memory evicted, per 2 MiB of guest memory.
static constexpr u64 RESIDENT_CHUNK_SHIFT = 21;
static constexpr u64 RESIDENT_BUFFER_REPORT_TICKS = 3000;
static constexpr u64 RESIDENT_BUFFER_FREE_DELAY_SECONDS = 3;
// A sparse bind or unbind slower than this is reported (it holds up the graphics queue).
static constexpr double SLOW_SPARSE_CALL_MS = 5.0;
static constexpr u64 RESIDENT_BUFFER_GC_COOLDOWN = 120;
static constexpr u64 RESIDENT_BUFFER_GC_TARGET = 256_MB;
// Less than this idle isn't worth a pass: each one waits for the GPU.
static constexpr u64 RESIDENT_BUFFER_GC_MIN = 32_MB;
// At a loading screen: buffer ranges unused this long are freed (the loading screen's own are in
// use all along).
static constexpr u64 LOAD_SCREEN_BUFFER_IDLE_SECONDS = 2;
constexpr std::optional<u32> CmaskColorExpandedValue(u32 num_samples) {
    switch (num_samples) {
    case 1:
        return 0xFFFFFFFF;
    case 2:
        return 0xDDDDDDDD;
    case 4:
        return 0xEEEEEEEE;
    case 8:
        return 0xFFFFFFFF;
    default:
        return 0;
    }
}
static_assert(CmaskColorExpandedValue(1) == 0xFFFFFFFF);
static_assert(CmaskColorExpandedValue(2) == 0xDDDDDDDD);
static_assert(CmaskColorExpandedValue(4) == 0xEEEEEEEE);
static_assert(CmaskColorExpandedValue(8) == 0xFFFFFFFF);

static constexpr auto ARENA_USAGE =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
    vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;

std::optional<u32> FindMemoryType(const vk::PhysicalDeviceMemoryProperties& properties,
                                  vk::MemoryPropertyFlags wanted, u32 memory_type_bits) {
    for (u32 i = 0; i < properties.memoryTypeCount; ++i) {
        if (((memory_type_bits >> i) & 1) == 0) {
            continue;
        }
        const auto flags = properties.memoryTypes[i].propertyFlags;
        if ((flags & wanted) == wanted) {
            return i;
        }
    }
    return std::nullopt;
}

BufferCache::BufferCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                         Vulkan::Runtime& runtime_, AmdGpu::Liverpool* liverpool_,
                         TextureCache& texture_cache_, PageManager& tracker)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_},
      staging_pool{runtime_.GetStagingPool()}, liverpool{liverpool_},
      memory{Core::Memory::Instance()}, texture_cache{texture_cache_},
      memory_tracker{std::make_unique<MemoryTracker>(tracker)},
      stream_buffer{instance, scheduler, DrawDataMemoryType(instance), STREAM_BUFFER_SIZE},
      gds_buffer{instance, 0, GDS_BUFFER_SIZE, MemoryType::Stream, "GDS Buffer"},
      memory_semaphore{instance} {
    vram_reclaim = EvictIdleBuffers() && EmulatorSettings.IsBbVramReclaim();
    if (vram_reclaim) {
        LOG_INFO(Render, "Buffer eviction on");
    }
    const auto families = instance.GetBufferQueueFamilies();
    const vk::BufferCreateInfo probe_ci = {
        .flags =
            vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency,
        .size = ARENA_PAGE_SIZE,
        .usage = ARENA_USAGE,
        .sharingMode =
            families.size() > 1 ? vk::SharingMode::eConcurrent : vk::SharingMode::eExclusive,
        .queueFamilyIndexCount = families.size() > 1 ? static_cast<u32>(families.size()) : 0u,
        .pQueueFamilyIndices = families.size() > 1 ? families.data() : nullptr,
    };
    const vk::DeviceBufferMemoryRequirements req_info = {
        .pCreateInfo = &probe_ci,
    };
    const auto device = instance.GetDevice();
    const auto reqs = device.getBufferMemoryRequirements(req_info).memoryRequirements;
    block_size = Common::AlignUp(std::max<u64>(reqs.alignment, MIN_BLOCK_SIZE), reqs.alignment);
    ASSERT_MSG(std::popcount(block_size) == 1, "Sparse block size {} is not a power of 2",
               block_size);
    block_shift = std::bit_width(block_size) - 1;
    blocks_per_arena_page = ARENA_PAGE_SIZE / block_size;
    blocks_per_arena_page_shift = ARENA_PAGE_BITS - block_shift;
    arena_memory_type_index =
        FindMemoryType(instance.GetMemoryProperties(), vk::MemoryPropertyFlagBits::eDeviceLocal,
                       reqs.memoryTypeBits)
            .value();

    const u64 bda_pagetable_size =
        (blocks_per_arena_page * NUM_ARENA_PAGES) * sizeof(vk::DeviceAddress);
    fault_manager = std::make_unique<FaultManager>(instance, scheduler, *this, block_shift,
                                                   blocks_per_arena_page * NUM_ARENA_PAGES);
    bda_pagetable_buffer = std::make_unique<Buffer>(
        instance, 0, bda_pagetable_size, MemoryType::DeviceLocal, "BDA Page Table Buffer");
    runtime.FillBuffer(bda_pagetable_buffer.get(), 0u, bda_pagetable_size, 0u);

    if (instance.HasReadbackQueue()) {
        readback_pool = Vulkan::Check(device.createCommandPoolUnique({
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer |
                     vk::CommandPoolCreateFlagBits::eTransient,
            .queueFamilyIndex = instance.GetReadbackQueueFamily(),
        }));
        auto buffers = Vulkan::Check(device.allocateCommandBuffersUnique({
            .commandPool = *readback_pool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1,
        }));
        readback_cmdbuf = std::move(buffers[0]);
        readback_fence = Vulkan::Check(device.createFenceUnique({}));
    }
    if (!instance.IsMaintenance6Supported()) {
        null_index_buffer = std::make_unique<Buffer>(instance, 0, NULL_INDEX_BUFFER_SIZE,
                                                     MemoryType::DeviceLocal, "Null Index Buffer");
        runtime.FillBuffer(null_index_buffer.get(), 0u, NULL_INDEX_BUFFER_SIZE, 0u);
    }
}

BufferCache::~BufferCache() {
    for (const RetiredMemory& retired : retired_memory) {
        instance.GetDevice().freeMemory(retired.memory);
    }
}

void BufferCache::TickFrame() {
    ++cache_tick;
    if (vram_reclaim) {
        cache_second = static_cast<u64>(std::chrono::duration_cast<std::chrono::seconds>(
                                            std::chrono::steady_clock::now().time_since_epoch())
                                            .count());
        // Evicted memory is freed some seconds after its unbind: nothing still in flight on
        // another queue or thread (presenter, frame generation, readbacks) can reach it.
        if (!retired_memory.empty()) {
            memory_semaphore.Refresh();
        }
        std::erase_if(retired_memory, [&](const RetiredMemory& retired) {
            if (cache_second - retired.second < RESIDENT_BUFFER_FREE_DELAY_SECONDS ||
                !memory_semaphore.IsFree(retired.unbind_tick)) {
                return false;
            }
            instance.GetDevice().freeMemory(retired.memory);
            return true;
        });
    }
    if (std::exchange(fault_process_pending, false)) {
        fault_manager->ProcessFaultBuffer();
    }
    GarbageCollectResidentMemory();
}

void BufferCache::InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks) {
    memory_tracker->InvalidateRegion(device_addr, size, [this, device_addr, size, assume_locks] {
        ReadMemory(device_addr, size, true, assume_locks);
    });
}

void BufferCache::ExtendWriteFault(VAddr device_addr) {
    // The game fills its per-frame buffers page after page, and every 4 KiB page cost one
    // protection fault. bbport measured ~7x fewer faults and +22% FPS with a 64 KiB window,
    // and halved the faults again with 256 KiB.
    constexpr u64 FaultWindow = 256_KB;
    static_assert(HIGHER_PAGE_SIZE % FaultWindow == 0);
    memory_tracker->ExtendWriteFault(Common::AlignDown(device_addr, FaultWindow), FaultWindow);
}

namespace {
/// Readback report: the game thread whose fault is being handled, on the thread downloading.
struct ReadbackContext {
    std::string thread;
    bool is_write{};
    bool active{};
};
thread_local ReadbackContext readback_context;
} // namespace

void BufferCache::ReadMemory(VAddr device_addr, u64 size, bool is_write, bool assume_locks) {
    const bool diagnostics = Common::GuestStats::diagnostics.load(std::memory_order_relaxed);
    std::string thread = diagnostics ? Common::GetCurrentThreadName() : std::string{};
    const auto blocked_start = std::chrono::steady_clock::now();
    // The game frame report: how long a game thread's readback waited for the GPU thread to take
    // it, then for the download.
    std::chrono::steady_clock::time_point taken{};
    const bool for_game_thread = !assume_locks;
    const auto flush_request = [this, device_addr, size, is_write, thread, &taken,
                                for_game_thread] {
        taken = std::chrono::steady_clock::now();
        readback_context = {thread, is_write, !thread.empty()};
        const u32 first_block = device_addr >> block_shift;
        const u32 last_block = (device_addr + size - 1) >> block_shift;
        const auto* arena = GetArena(first_block, last_block);

        // GPU-modified ranges come as many small scattered islands,
        // so the download is widened to a window around the request
        constexpr u64 WindowSize = 512_KB;
        const VAddr arena_end = arena->cpu_addr + arena->size_bytes;
        const VAddr window_start =
            std::max<VAddr>(Common::AlignDown(device_addr, WindowSize), arena->cpu_addr);
        const VAddr window_end = std::min<VAddr>(
            std::max<VAddr>(window_start + WindowSize, device_addr + size), arena_end);
        if (for_game_thread && EarlyWriteback()) {
            MarkHot(window_start, window_end - window_start);
        }
        const bool downloaded = DownloadMemory(arena, window_start, window_end - window_start);
        readback_context.active = false;
        if (!downloaded) {
            LOG_WARNING(Render, "Guest buffer readback failed; retaining GPU-modified state");
            return;
        }
        if (is_write) {
            memory_tracker->MarkRegionAsCpuModified(device_addr, size);
        }
    };
    if (assume_locks) {
        flush_request();
    } else {
        liverpool->SendCommand<true>(std::move(flush_request));
        if (Common::GameWaits::IsGameThread() && taken != std::chrono::steady_clock::time_point{}) {
            const auto done = std::chrono::steady_clock::now();
            Common::GameWaits::Record("readback: waiting for the GPU thread", {},
                                      taken - blocked_start);
            Common::GameWaits::Record("readback: download", {}, done - taken);
        }
    }
    if (diagnostics) {
        // How long the game's thread was stopped (queued behind the GPU thread, then the wait).
        const u64 blocked = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - blocked_start)
                                .count();
        std::scoped_lock lock{readback_mutex};
        readback_regions[device_addr >> 20].guest_blocked_ns += blocked;
    }
}

void BufferCache::LogReadbackReport() {
    const auto now = std::chrono::steady_clock::now();
    if (readback_report_time == std::chrono::steady_clock::time_point{}) {
        readback_report_time = now;
        return;
    }
    if (now - readback_report_time < std::chrono::seconds(10)) {
        return;
    }
    readback_report_time = now;
    std::unordered_map<u64, ReadbackRegion> regions;
    std::array<u64, 4> writer_ns{};
    u64 max_wait{}, queue_count{}, queue_ns{};
    {
        std::scoped_lock lock{readback_mutex};
        regions.swap(readback_regions);
        writer_ns = std::exchange(readback_writer_ns, {});
        max_wait = std::exchange(readback_max_wait_ns, 0);
        queue_count = std::exchange(readback_queue_count, 0);
        queue_ns = std::exchange(readback_queue_ns, 0);
    }
    if (EarlyWriteback()) {
        u64 done, bytes;
        size_t pending, hot;
        {
            std::scoped_lock lock{writeback_mutex};
            done = std::exchange(writebacks_done, 0);
            bytes = std::exchange(writeback_bytes, 0);
            pending = pending_writebacks.size();
            hot = hot_blocks.size();
        }
        LOG_INFO(Render,
                 "Early write-back (10 s): {} copies back to guest memory ({} KiB), {} batches "
                 "sent early | {} waiting for the GPU, {} hot 64 KiB blocks",
                 done, bytes >> 10, writeback_flushes.exchange(0, std::memory_order_relaxed),
                 pending, hot);
    }
    u64 count = 0, wait = 0, blocked = 0;
    std::array<u64, 4> writer{};
    for (const auto& [key, region] : regions) {
        count += region.count;
        wait += region.gpu_wait_ns;
        blocked += region.guest_blocked_ns;
        for (size_t i = 0; i < writer.size(); ++i) {
            writer[i] += region.writer[i];
        }
    }
    if (count == 0) {
        return;
    }
    LOG_INFO(Render,
             "Readbacks (10 s): {} downloads, GPU thread waited {:.1f} ms (longest {:.1f}), game "
             "threads stopped {:.1f} ms | data written by: the unsubmitted batch {} ({:.1f} ms), "
             "submitted work still running {} ({:.1f} ms), finished work {} ({:.1f} ms), unknown "
             "{} ({:.1f} ms) | on the second queue {} ({:.1f} ms), on the graphics queue {} "
             "({:.1f} ms)",
             count, wait / 1e6, max_wait / 1e6, blocked / 1e6, writer[0], writer_ns[0] / 1e6,
             writer[1], writer_ns[1] / 1e6, writer[2], writer_ns[2] / 1e6, writer[3],
             writer_ns[3] / 1e6, queue_count, queue_ns / 1e6, count - queue_count,
             (wait - queue_ns) / 1e6);
    std::vector<std::pair<u64, const ReadbackRegion*>> sorted;
    for (const auto& [key, region] : regions) {
        if (region.count != 0) {
            sorted.emplace_back(key, &region);
        }
    }
    std::ranges::sort(sorted, [](const auto& a, const auto& b) {
        return a.second->gpu_wait_ns > b.second->gpu_wait_ns;
    });
    for (size_t i = 0; i < std::min<size_t>(sorted.size(), 6); ++i) {
        const auto& [key, region] = sorted[i];
        std::string threads;
        for (const auto& [name, n] : region->threads) {
            threads += fmt::format("{}{} x{}", threads.empty() ? "" : ", ", name, n);
        }
        LOG_INFO(Render,
                 "  region {:#x}: {} downloads, {:.0f} KiB each, waited {:.1f} ms | written by: "
                 "unsubmitted {}, running {}, finished {}, unknown {} | threads: {}",
                 key << 20, region->count, region->bytes / 1024.0 / region->count,
                 region->gpu_wait_ns / 1e6, region->writer[0], region->writer[1], region->writer[2],
                 region->writer[3], threads.empty() ? "-" : threads);
    }
}

void BufferCache::MarkHot(VAddr device_addr, u64 size) {
    const auto now = std::chrono::steady_clock::now();
    std::scoped_lock lock{writeback_mutex};
    for (u64 block = device_addr >> 16; block <= (device_addr + size - 1) >> 16; ++block) {
        hot_blocks[block] = now;
    }
}

void BufferCache::NoteHotWrite(VAddr device_addr, u64 size) {
    // Hot for 10 s after the last readback a game thread waited for.
    constexpr auto HotFor = std::chrono::seconds(10);
    constexpr size_t MaxPending = 512;
    const auto now = std::chrono::steady_clock::now();
    std::scoped_lock lock{writeback_mutex};
    if (hot_blocks.empty()) {
        return;
    }
    bool hot = false;
    for (u64 block = device_addr >> 16; block <= (device_addr + size - 1) >> 16 && !hot; ++block) {
        const auto it = hot_blocks.find(block);
        hot = it != hot_blocks.end() && now - it->second < HotFor;
    }
    if (!hot) {
        return;
    }
    const u64 tick = scheduler.CurrentTick();
    const auto same = std::ranges::find_if(pending_writebacks, [&](const PendingWriteback& w) {
        return w.address == device_addr && w.size == size;
    });
    if (same != pending_writebacks.end()) {
        same->tick = tick;
    } else if (pending_writebacks.size() < MaxPending) {
        pending_writebacks.push_back({device_addr, size, tick});
    }
    writeback_flush.store(true, std::memory_order_release);
}

std::vector<BufferCache::PendingWriteback> BufferCache::TakeFinishedWritebacks() {
    std::vector<PendingWriteback> finished;
    std::scoped_lock lock{writeback_mutex};
    if (pending_writebacks.empty()) {
        return finished;
    }
    const u64 current = scheduler.CurrentTick();
    std::erase_if(pending_writebacks, [&](const PendingWriteback& w) {
        if (w.tick >= current || !scheduler.IsFree(w.tick)) {
            return false;
        }
        finished.push_back(w);
        return true;
    });
    // Expired hot blocks are dropped now and then.
    const auto now = std::chrono::steady_clock::now();
    std::erase_if(hot_blocks,
                  [&](const auto& block) { return now - block.second > std::chrono::seconds(30); });
    return finished;
}

void BufferCache::WriteBack(const PendingWriteback& writeback) {
    // Only finished work on the second queue (no wait for everything queued), and not memory
    // that newer GPU work writes again (its own write-back follows).
    if (!readback_fence || !Common::BbTuning::ReadbackQueue() ||
        !IsRegionGpuModified(writeback.address, writeback.size) ||
        LastGpuWriteTick(writeback.address, writeback.size) > writeback.tick) {
        return;
    }
    ReadMemory(writeback.address, writeback.size, false, true);
    std::scoped_lock lock{writeback_mutex};
    ++writebacks_done;
    writeback_bytes += writeback.size;
}

void BufferCache::NoteGpuWrite(VAddr device_addr, u64 size) {
    if (size != 0 && EarlyWriteback()) {
        NoteHotWrite(device_addr, size);
    }
    // Only needed for the second readback queue and the readback report.
    if (size == 0 ||
        (!readback_fence && !Common::GuestStats::diagnostics.load(std::memory_order_relaxed))) {
        return;
    }
    const u64 tick = scheduler.CurrentTick();
    std::scoped_lock lock{gpu_write_mutex};
    for (u64 block = device_addr >> 16; block <= (device_addr + size - 1) >> 16; ++block) {
        gpu_write_ticks[block] = tick;
    }
}

u64 BufferCache::LastGpuWriteTick(VAddr device_addr, u64 size) {
    u64 tick = 0;
    std::scoped_lock lock{gpu_write_mutex};
    for (u64 block = device_addr >> 16; block <= (device_addr + size - 1) >> 16; ++block) {
        if (const auto it = gpu_write_ticks.find(block); it != gpu_write_ticks.end()) {
            tick = std::max(tick, it->second);
        }
    }
    return tick;
}

void BufferCache::ReadbackOnSecondQueue(const Buffer* arena, const Buffer* destination,
                                        std::span<const vk::BufferCopy> copies, u64 tick) {
    std::scoped_lock lock{readback_queue_mutex};
    const auto device = instance.GetDevice();
    const auto cmdbuf = *readback_cmdbuf;
    Vulkan::Check(cmdbuf.reset());
    Vulkan::Check(cmdbuf.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit}));
    cmdbuf.copyBuffer(arena->Handle(), destination->Handle(), static_cast<u32>(copies.size()),
                      copies.data());
    // The CPU reads the result after the fence.
    const vk::MemoryBarrier2 to_host = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead,
    };
    cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &to_host});
    Vulkan::Check(cmdbuf.end());
    // Waits on the graphics timeline for the submission that wrote the data (its signal makes
    // the writes available), not for the work queued after it.
    const vk::SemaphoreSubmitInfo wait = {
        .semaphore = scheduler.GetWorkSemaphore()->Handle(),
        .value = tick,
        .stageMask = vk::PipelineStageFlagBits2::eAllTransfer,
    };
    const vk::CommandBufferSubmitInfo command = {.commandBuffer = cmdbuf};
    const vk::SubmitInfo2 submit = {
        .waitSemaphoreInfoCount = 1,
        .pWaitSemaphoreInfos = &wait,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &command,
    };
    Vulkan::Check(device.resetFences(*readback_fence));
    Vulkan::Check(instance.GetReadbackQueue().submit2(submit, *readback_fence));
    Vulkan::Check(device.waitForFences(*readback_fence, true, std::numeric_limits<u64>::max()));
}

bool BufferCache::DownloadMemory(const Buffer* arena, VAddr device_addr, u64 size) {
    boost::container::small_vector<vk::BufferCopy, 1> copies;
    u64 total_size_bytes = 0;
    const VAddr arena_base = arena->cpu_addr;
    memory_tracker->ForEachDownloadRange<false>(device_addr, size, [&](u64 address, u64 size) {
        const auto add_download = [&](VAddr start, VAddr end) {
            const u64 new_offset = start - arena_base;
            const u64 new_size = end - start;
            copies.push_back(vk::BufferCopy{
                .srcOffset = new_offset,
                .dstOffset = total_size_bytes,
                .size = new_size,
            });
            // Align up to avoid cache conflicts
            constexpr u64 align = 64ULL;
            constexpr u64 mask = ~(align - 1ULL);
            total_size_bytes += (new_size + align - 1) & mask;
        };
        gpu_modified_ranges.ForEachInRange(address, size, add_download);
    });
    if (total_size_bytes == 0) {
        return true;
    }
    // Readback report: was the GPU work that wrote this data still unsubmitted, running, or
    // finished?
    const bool diagnostics = Common::GuestStats::diagnostics.load(std::memory_order_relaxed);
    const u64 tick = LastGpuWriteTick(device_addr, size);
    const bool submitted = tick != 0 && tick < scheduler.CurrentTick();
    size_t writer = 3;
    if (diagnostics && tick != 0) {
        writer = !submitted ? 0 : scheduler.IsFree(tick) ? 2 : 1;
    }
    const auto download = staging_pool.Request(total_size_bytes, VideoCore::MemoryType::HostCached);
    for (auto& copy : copies) {
        copy.dstOffset += download.offset;
    }
    const auto wait_start = std::chrono::steady_clock::now();
    // The data was written by work already submitted: copy it on the second queue, which
    // waits only for that work. Otherwise (unsubmitted or unknown writer) copy in order on the
    // graphics queue and wait for everything queued.
    const bool second_queue = submitted && readback_fence && Common::BbTuning::ReadbackQueue();
    if (second_queue) {
        ReadbackOnSecondQueue(arena, download.buffer, copies, tick);
    } else {
        runtime.CopyBuffer(arena, download.buffer, copies);
        scheduler.Finish();
    }
    if (diagnostics) {
        const u64 waited = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - wait_start)
                               .count();
        std::scoped_lock lock{readback_mutex};
        auto& region = readback_regions[device_addr >> 20];
        ++region.count;
        region.bytes += total_size_bytes;
        region.gpu_wait_ns += waited;
        ++region.writer[writer];
        readback_writer_ns[writer] += waited;
        readback_max_wait_ns = std::max(readback_max_wait_ns, waited);
        if (second_queue) {
            ++readback_queue_count;
            readback_queue_ns += waited;
        }
        const std::string& thread =
            readback_context.active ? readback_context.thread : std::string{"GPU thread"};
        ++region.threads[fmt::format(
            "{} ({})", thread,
            readback_context.active && !readback_context.is_write ? "read" : "write")];
    }

    download.buffer->Invalidate(download.offset, download.size);
    for (const auto& copy : copies) {
        auto* dst_addr = std::bit_cast<u8*>(arena_base + copy.srcOffset);
        if (!memory->TryWriteBacking(dst_addr, download.mapped + (copy.dstOffset - download.offset),
                                     copy.size)) {
            return false;
        }
    }
    // Keep all dirty tracking until every copy has reached guest memory.
    for (const auto& copy : copies) {
        gpu_modified_ranges.Subtract(arena_base + copy.srcOffset, copy.size);
    }
    memory_tracker->UnmarkRegionAsGpuModified(device_addr, size, false);
    return true;
}

bool BufferCache::DownloadMemoryFromBuffer(const Buffer* arena, VAddr device_addr, u64 size) {
    gpu_modified_ranges.Add(device_addr, size);
    memory_tracker->MarkRegionAsGpuModified(device_addr, size);
    return DownloadMemory(arena, device_addr, size);
}

std::pair<const Buffer*, u64> BufferCache::ObtainBuffer(VAddr device_addr, u32 size,
                                                        bool is_written, bool is_texel_buffer) {
    // For read-only buffers use device local stream buffer to reduce renderpass breaks.
    if (!is_written && size <= STREAM_THRESHOLD && !IsRegionGpuModified(device_addr, size)) {
        const auto [data, offset] = stream_buffer.Map(size, instance.UniformMinAlignment());
        memory->CopySparseMemory(device_addr, data, size);
        stream_buffer.Commit();
        return {&stream_buffer, offset};
    }
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    EnsureResident(arena, first_block, last_block);
    if (vram_reclaim) {
        MarkResidentUse(device_addr, size);
    }
    SynchronizeMemory(arena, device_addr, size, is_written, is_texel_buffer);
    if (is_written) {
        gpu_modified_ranges.Add(device_addr, size);
        NoteGpuWrite(device_addr, size);
    }
    return {arena, arena->Offset(device_addr)};
}

std::pair<const Buffer*, u64> BufferCache::ObtainGeometryBuffer(VAddr device_addr, u32 size) {
    static constexpr u32 CopyBelow = 1_KB;
    if (size < CopyBelow || size > STREAM_THRESHOLD || !Common::BbTuning::Threaded() ||
        !EmulatorSettings.IsBbGeometryInPlace() || IsRegionCpuModified(device_addr, size)) {
        // Small or changed data is copied as before (no upload inside a render pass).
        return ObtainBuffer(device_addr, size, false);
    }
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    EnsureResident(arena, first_block, last_block);
    if (vram_reclaim) {
        MarkResidentUse(device_addr, size);
    }
    SynchronizeMemory(arena, device_addr, size, false, false);
    return {arena, arena->Offset(device_addr)};
}

std::pair<const Buffer*, u64> BufferCache::ObtainBufferForImage(VAddr device_addr, u32 size) {
    if (IsRegionGpuModified(device_addr, size)) {
        return ObtainBuffer(device_addr, size, false);
    }
    const auto staging = staging_pool.Request(size, VideoCore::MemoryType::HostUncached,
                                              instance.StorageMinAlignment());
    const auto start = std::chrono::steady_clock::now();
    if (staging.size >= BackgroundCopyMin && Common::BbTuning::Threaded() &&
        EmulatorSettings.IsBbBackgroundTextureCopies()) {
        // Streaming a new area uploads ~1000 textures, ~200 KiB each, in a frame: copied one by
        // one they held up the draw recorder for tens of ms. On the copy threads, in pieces of up
        // to 1 MiB; waited for before the submission that reads the staging memory
        // (Scheduler::SubmitExecution) and before the guest sees a fence (Rasterizer::OnFence).
        for (u64 offset = 0; offset < staging.size; offset += BackgroundCopyPiece) {
            const u64 length = std::min(BackgroundCopyPiece, staging.size - offset);
            Common::GuestStats::async_copies.fetch_add(1, std::memory_order_relaxed);
            Common::CopyPool::Async([memory = memory, source = device_addr + offset,
                                     destination = staging.mapped + offset, length,
                                     buffer = staging.buffer, flush = staging.offset + offset] {
                memory->CopySparseMemory(source, destination, length);
                buffer->Flush(flush, length);
            });
        }
        return {staging.buffer, staging.offset};
    }
    if (Common::BbTuning::Streaming()) {
        // Streaming a new area copies 100+ MiB of textures in a frame: large ones in parallel.
        Common::CopyPool::SplitCopy(staging.size, [&](u64 offset, u64 length) {
            memory->CopySparseMemory(device_addr + offset, staging.mapped + offset, length);
        });
    } else {
        memory->CopySparseMemory(device_addr, staging.mapped, staging.size);
    }
    Common::GuestStats::image_copy_ns.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                             start)
            .count(),
        std::memory_order_relaxed);
    staging.Flush();
    return {staging.buffer, staging.offset};
}

bool BufferCache::IsRegionCpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionCpuModified(addr, size);
}

bool BufferCache::IsRegionGpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionGpuModified(addr, size);
}

void BufferCache::SynchronizeDmaBuffers() {
    fault_process_pending = true;
    for (const auto& range : resident_ranges) {
        const u64 page = range.start >> (ARENA_PAGE_BITS - block_shift);
        const VAddr device_addr = range.start << block_shift;
        const u64 size = (range.end - range.start) << block_shift;
        SynchronizeMemory(address_space[page], device_addr, size, false, false);
    }
}

const Buffer* BufferCache::GetArena(u64 first_block, u64 last_block) {
    const u64 first_page = first_block >> blocks_per_arena_page_shift;
    const u64 last_page = last_block >> blocks_per_arena_page_shift;
    ASSERT_MSG(last_page - first_page <= 1,
               "Buffer request cannot span more than two VA arena pages");

    const auto* first_arena = address_space[first_page];
    const auto* last_arena = address_space[last_page];
    if (first_arena == last_arena) {
        if (!first_arena) {
            const u64 base_block = Common::AlignDownPow2<u64>(first_block, blocks_per_arena_page);
            const u64 num_pages = last_page - first_page + 1;
            const auto* new_arena =
                &arenas.emplace_back(instance, base_block << block_shift,
                                     num_pages << ARENA_PAGE_BITS, MemoryType::Sparse);
            address_space[first_page] = new_arena;
            address_space[last_page] = new_arena;
        }
        return address_space[first_page];
    }

    const u64 first_addr = first_arena ? first_arena->cpu_addr : (first_page << ARENA_PAGE_BITS);
    const u64 first_size = first_arena ? first_arena->size_bytes : ARENA_PAGE_SIZE;
    const u64 last_size = last_arena ? last_arena->size_bytes : ARENA_PAGE_SIZE;

    const u64 base_block = first_addr >> block_shift;
    const u64 total_size = first_size + last_size;
    LOG_WARNING(Render,
                "Migrating arena: pages {} and {} ({} and {} MiB, {}) into one of {} MiB, {} "
                "arenas so far",
                first_page, last_page, first_size >> 20, last_size >> 20,
                first_arena ? "first exists" : "first new", total_size >> 20, arenas.size());
    Common::Crumbs::Add("buffer arena: migrate (first page, last page)", first_page, last_page);
    const u64 end_block = (first_addr + total_size) >> block_shift;
    auto* new_arena = &arenas.emplace_back(instance, first_addr, total_size, MemoryType::Sparse);
    auto* bind = BindsForArena(new_arena);
    resident_ranges.ForEachInRange(base_block, end_block, [&](const Backing& backing) {
        const u64 start = std::max(base_block, backing.start);
        const u64 end = std::min(end_block, backing.end);
        bind->binds.push_back(vk::SparseMemoryBind{
            .resourceOffset = (start - base_block) << block_shift,
            .size = (end - start) << block_shift,
            .memory = backing.memory,
            .memoryOffset = (backing.offset + start - backing.start) << block_shift,
        });
    });

    u64 base_page = first_addr >> ARENA_PAGE_BITS;
    for (u32 page = 0; page < (first_size >> ARENA_PAGE_BITS); ++page) {
        address_space[base_page + page] = new_arena;
    }
    base_page = last_page;
    for (u32 page = 0; page < (last_size >> ARENA_PAGE_BITS); ++page) {
        address_space[base_page + page] = new_arena;
    }
    return new_arena;
}

void BufferCache::EnsureResident(const Buffer* arena, u64 first_block, u64 last_block) {
    // Threaded renderer (streaming optimisations): most lookups are for ranges made resident
    // long ago; remembered ones skip the gap search until residency is taken away.
    const bool memo = Common::BbTuning::Streaming();
    auto& entry = resident_memo[(first_block * 0x9E3779B97F4A7C15ull ^ last_block) >> 54];
    if (memo && entry.first_block == first_block && entry.last_block == last_block &&
        entry.generation == resident_generation) {
        return;
    }
    if (memo) {
        // Resident once this returns (bound now, or queued before any use of it).
        entry = {first_block, last_block, resident_generation};
    }
    u32 resident_blocks{};
    IntervalList bind_ranges;
    resident_ranges.ForEachGap(first_block, last_block + 1, [&](u64 start, u64 end) {
        resident_blocks += end - start;
        bind_ranges.Add({start, end});
    });

    if (bind_ranges.Empty()) {
        return;
    }
    Common::Crumbs::Add("buffer residency: make resident (MiB, first block)",
                        (u64(resident_blocks) << block_shift) >> 20, first_block);
    const auto resident_start = std::chrono::steady_clock::now();
    SCOPE_EXIT {
        Common::GuestStats::resident_ns.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                                 resident_start)
                .count(),
            std::memory_order_relaxed);
    };

    if (vram_reclaim) {
        MakeResidentPooled(arena, bind_ranges, resident_blocks);
        MarkResidentUse(first_block << block_shift, (last_block - first_block + 1) << block_shift);
        // Pooled memory comes back holding another range's bytes (space an eviction unbound),
        // and only the part a binding names was uploaded at its use. The GPU also reads the rest
        // of the block: vertex fetch past a binding's end, shaders through the page table. Garbage
        // there (indices, draw counts) hung the GPU after loading screens with buffer eviction on.
        // So the whole range bound now is filled from guest memory (evicted ranges were marked
        // CPU-modified, so all of it uploads).
        for (const auto& range : bind_ranges) {
            const VAddr address = range.start << block_shift;
            const u64 size = (range.end - range.start) << block_shift;
            for (u64 offset = 0; offset < size; offset += 1_GB) {
                SynchronizeMemory(arena, address + offset, u32(std::min<u64>(size - offset, 1_GB)),
                                  false, false);
            }
        }
        return;
    }

    // Resident memory is never released, so it's carved from 64 MiB slabs: one allocation per
    // slab instead of one per range (streaming a new area made hundreds resident in a frame).
    // Streaming optimisations off: one allocation per range, as originally.
    const u64 bind_bytes = u64(resident_blocks) << block_shift;
    if (!Common::BbTuning::Streaming()) {
        resident_slab = vk::DeviceMemory{};
    }
    if (!resident_slab || resident_slab_used + bind_bytes > resident_slab_size) {
        // Arenas have buffer device addresses: their sparse backing needs the flag too.
        const vk::MemoryAllocateFlagsInfo alloc_flags = {
            .flags = vk::MemoryAllocateFlagBits::eDeviceAddress,
        };
        const vk::MemoryAllocateInfo alloc_info = {
            .pNext = &alloc_flags,
            .allocationSize = Common::BbTuning::Streaming()
                                  ? std::max<u64>(ResidentSlabSize, bind_bytes)
                                  : bind_bytes,
            .memoryTypeIndex = arena_memory_type_index,
        };
        resident_slab = AllocateResidentMemory(alloc_info, "slab");
        resident_slab_size = alloc_info.allocationSize;
        resident_slab_used = 0;
    }
    const auto device_memory = resident_slab;
    resident_bytes.fetch_add(bind_bytes, std::memory_order_relaxed);

    boost::container::small_vector<vk::BufferCopy, 8> copies;
    const auto staging =
        staging_pool.Request(resident_blocks * sizeof(vk::DeviceAddress), MemoryType::HostUncached);

    u64 memory_offset = resident_slab_used;
    resident_slab_used += bind_bytes;
    ArenaBinds* binds = BindsForArena(arena);
    auto* bda_addrs = reinterpret_cast<vk::DeviceAddress*>(staging.mapped);
    u64 offset = staging.offset;
    for (const auto& range : bind_ranges) {
        Backing backing;
        backing.start = range.start;
        backing.end = range.end;
        backing.memory = device_memory;
        backing.offset = memory_offset >> block_shift;
        resident_ranges.Add(backing);

        // Trace only: with synchronous logging, hundreds of these per frame cost tens of ms.
        LOG_TRACE(Render, "Making range start={}, end={} resident", backing.start, backing.end);
        Common::GuestStats::arena_binds.fetch_add(1, std::memory_order_relaxed);

        const auto& bind = binds->binds.emplace_back(vk::SparseMemoryBind{
            .resourceOffset = (range.start << block_shift) - arena->cpu_addr,
            .size = (range.end - range.start) << block_shift,
            .memory = device_memory,
            .memoryOffset = memory_offset,
        });
        memory_offset += bind.size;

        for (u32 block = 0; block < bind.size; block += block_size) {
            *(bda_addrs++) = arena->BufferDeviceAddress() + bind.resourceOffset + block;
        }
        const u64 copy_size = (backing.end - backing.start) * sizeof(vk::DeviceAddress);
        copies.emplace_back(offset, backing.start * sizeof(vk::DeviceAddress), copy_size);
        offset += copy_size;
    }

    staging.Flush();
    runtime.CopyBuffer(staging.buffer, bda_pagetable_buffer.get(), copies);
}

// Buffer eviction: resident memory comes from slabs whose blocks are handed out and taken back
// one by one. Space an eviction unbound is reused by the next ranges made resident (a sparse
// range doesn't need contiguous memory), and a slab is freed once none of it is bound. Without
// this, evicted space stayed allocated as long as anything else in its slab was in use.
void BufferCache::MakeResidentPooled(const Buffer* arena, const IntervalList<>& bind_ranges,
                                     u64 resident_blocks) {
    const auto staging =
        staging_pool.Request(resident_blocks * sizeof(vk::DeviceAddress), MemoryType::HostUncached);
    boost::container::small_vector<vk::BufferCopy, 8> copies;
    ArenaBinds* binds = BindsForArena(arena);
    auto* bda_addrs = reinterpret_cast<vk::DeviceAddress*>(staging.mapped);
    u64 staging_offset = staging.offset;
    for (const auto& range : bind_ranges) {
        for (u64 block = range.start; block < range.end;) {
            const auto [memory, memory_block, count] = TakeResidentBlocks(range.end - block);
            Backing backing;
            backing.start = block;
            backing.end = block + count;
            backing.memory = memory;
            backing.offset = memory_block;
            resident_ranges.Add(backing);
            Common::GuestStats::arena_binds.fetch_add(1, std::memory_order_relaxed);

            const auto& bind = binds->binds.emplace_back(vk::SparseMemoryBind{
                .resourceOffset = (block << block_shift) - arena->cpu_addr,
                .size = count << block_shift,
                .memory = memory,
                .memoryOffset = memory_block << block_shift,
            });
            for (u64 offset = 0; offset < bind.size; offset += block_size) {
                *(bda_addrs++) = arena->BufferDeviceAddress() + bind.resourceOffset + offset;
            }
            const u64 copy_size = count * sizeof(vk::DeviceAddress);
            copies.emplace_back(staging_offset, block * sizeof(vk::DeviceAddress), copy_size);
            staging_offset += copy_size;
            block += count;
        }
    }
    resident_bytes.fetch_add(resident_blocks << block_shift, std::memory_order_relaxed);
    staging.Flush();
    runtime.CopyBuffer(staging.buffer, bda_pagetable_buffer.get(), copies);
}

BufferCache::ResidentBlocks BufferCache::TakeResidentBlocks(u64 wanted) {
    // A free run that fits whole, else the first one (the range is split over several).
    ResidentSlab* slab = nullptr;
    vk::DeviceMemory memory{};
    std::map<u64, u64>::iterator run;
    for (auto& [raw, candidate] : resident_slabs) {
        for (auto it = candidate.free.begin(); it != candidate.free.end(); ++it) {
            if (it->second >= wanted || !slab) {
                slab = &candidate;
                memory = vk::DeviceMemory{raw};
                run = it;
                if (it->second >= wanted) {
                    break;
                }
            }
        }
        if (slab && run->second >= wanted) {
            break;
        }
    }
    if (!slab) {
        const u64 slab_blocks = std::max<u64>(EvictionSlabSize >> block_shift, wanted);
        const vk::MemoryAllocateFlagsInfo alloc_flags = {
            .flags = vk::MemoryAllocateFlagBits::eDeviceAddress,
        };
        const vk::MemoryAllocateInfo alloc_info = {
            .pNext = &alloc_flags,
            .allocationSize = slab_blocks << block_shift,
            .memoryTypeIndex = arena_memory_type_index,
        };
        memory = AllocateResidentMemory(alloc_info, "pooled slab");
        auto& created = resident_slabs[static_cast<VkDeviceMemory>(memory)];
        created.size_blocks = slab_blocks;
        created.free.emplace(0, slab_blocks);
        slab = &created;
        run = created.free.begin();
    }
    const u64 start = run->first;
    const u64 count = std::min(wanted, run->second);
    const u64 rest = run->second - count;
    slab->free.erase(run);
    if (rest != 0) {
        slab->free.emplace(start + count, rest);
    }
    slab->live_blocks += count;
    return {memory, start, count};
}

vk::DeviceMemory BufferCache::AllocateResidentMemory(const vk::MemoryAllocateInfo& requested,
                                                     const char* pool) {
    // Memory priorities on: guest buffers ask to be the last of the emulator's memory Windows moves
    // to system RAM (the GPU reads them in every draw).
    const vk::MemoryPriorityAllocateInfoEXT priority_info = {
        .pNext = requested.pNext,
        .priority = Vulkan::Instance::HighBufferPriority,
    };
    vk::MemoryAllocateInfo alloc_info = requested;
    if (instance.HasMemoryPriority()) {
        alloc_info.pNext = &priority_info;
    }
    const u64 size_mib = alloc_info.allocationSize >> 20;
    Common::Crumbs::Add("buffer memory: allocate (MiB, resident MiB)", size_mib,
                        resident_bytes.load(std::memory_order_relaxed) >> 20);
    if (alloc_info.allocationSize > ResidentSlabSize) {
        const auto heaps = instance.GetMemoryHeapReport();
        LOG_INFO(Render, "Large buffer memory allocation ({}): {} MiB | VRAM {} of {} MiB", pool,
                 size_mib, heaps.local_usage >> 20, heaps.local_budget >> 20);
    }
    auto allocated = instance.GetDevice().allocateMemory(alloc_info);
    if (allocated.result != vk::Result::eSuccess) {
        const auto heaps = instance.GetMemoryHeapReport();
        u64 pool_blocks = 0;
        u64 pool_free_blocks = 0;
        for (const auto& [raw, slab] : resident_slabs) {
            pool_blocks += slab.size_blocks;
            pool_free_blocks += slab.size_blocks - slab.live_blocks;
        }
        LOG_CRITICAL(Render,
                     "Buffer memory allocation failed ({}): {} MiB requested, {}, memory type {} "
                     "| VRAM {} of {} MiB (emulator budget {} MiB), system memory {} MiB | buffer "
                     "memory {} MiB resident, {} MiB in {} pooled slabs ({} MiB free for reuse), "
                     "{} evicted allocations waiting to be freed, buffer eviction {}",
                     pool, size_mib, vk::to_string(allocated.result), alloc_info.memoryTypeIndex,
                     heaps.local_usage >> 20, heaps.local_budget >> 20,
                     instance.GetTotalMemoryBudget() >> 20, heaps.host_usage >> 20,
                     resident_bytes.load(std::memory_order_relaxed) >> 20,
                     (pool_blocks << block_shift) >> 20, resident_slabs.size(),
                     (pool_free_blocks << block_shift) >> 20, retired_memory.size(),
                     vram_reclaim ? "on" : "off");
        Common::Crumbs::Add("buffer memory: allocation failed (MiB)", size_mib);
        // Out of graphics card memory: what can go at once goes, then the allocation is tried
        // again. Evicted buffer memory whose unbind the GPU has done is bound to nothing (it waits
        // a few seconds only as a precaution), and idle upload scratch is taken back on demand.
        u64 retired_freed = 0;
        if (!retired_memory.empty()) {
            memory_semaphore.Refresh();
            std::erase_if(retired_memory, [&](const RetiredMemory& retired) {
                if (!memory_semaphore.IsFree(retired.unbind_tick)) {
                    return false;
                }
                instance.GetDevice().freeMemory(retired.memory);
                ++retired_freed;
                return true;
            });
        }
        const auto scratch = staging_pool.ReleaseIdleDeviceLocal(true);
        allocated = instance.GetDevice().allocateMemory(alloc_info);
        const auto after = instance.GetMemoryHeapReport();
        if (allocated.result == vk::Result::eSuccess) {
            LOG_WARNING(Render,
                        "Buffer memory allocation ({}, {} MiB) succeeded on a second try after "
                        "freeing {} evicted buffer allocations early and {} MiB of upload "
                        "scratch | VRAM {} of {} MiB",
                        pool, size_mib, retired_freed, scratch.bytes >> 20, after.local_usage >> 20,
                        after.local_budget >> 20);
            Common::Crumbs::Add("buffer memory: allocation recovered (MiB)", size_mib);
        } else {
            LOG_CRITICAL(Render,
                         "Buffer memory allocation ({}, {} MiB) failed again after freeing {} "
                         "evicted buffer allocations early and {} MiB of upload scratch: {} | "
                         "VRAM {} of {} MiB",
                         pool, size_mib, retired_freed, scratch.bytes >> 20,
                         vk::to_string(allocated.result), after.local_usage >> 20,
                         after.local_budget >> 20);
        }
    }
    return Vulkan::Check(std::move(allocated));
}

u64 BufferCache::ReturnResidentBlocks(vk::DeviceMemory memory, u64 start, u64 count) {
    const auto it = resident_slabs.find(static_cast<VkDeviceMemory>(memory));
    if (it == resident_slabs.end()) {
        return 0;
    }
    ResidentSlab& slab = it->second;
    slab.live_blocks -= std::min(slab.live_blocks, count);
    if (slab.live_blocks == 0) {
        // Nothing of it is bound any more: the caller frees it.
        const u64 size_blocks = slab.size_blocks;
        resident_slabs.erase(it);
        return size_blocks;
    }
    // Merged with its neighbours.
    u64 run_start = start;
    u64 run_count = count;
    auto next = slab.free.lower_bound(start);
    if (next != slab.free.begin()) {
        const auto previous = std::prev(next);
        if (previous->first + previous->second == start) {
            run_start = previous->first;
            run_count += previous->second;
            slab.free.erase(previous);
        }
    }
    if (next != slab.free.end() && next->first == start + count) {
        run_count += next->second;
        slab.free.erase(next);
    }
    slab.free.emplace(run_start, run_count);
    return 0;
}

void BufferCache::MarkResidentUse(VAddr device_addr, u64 size) {
    const u64 first = device_addr >> RESIDENT_CHUNK_SHIFT;
    const u64 last = (device_addr + size - 1) >> RESIDENT_CHUNK_SHIFT;
    for (u64 chunk = first; chunk <= last; ++chunk) {
        auto& seen = chunk_use_memo[chunk % chunk_use_memo.size()];
        if (seen.chunk == chunk && seen.second == cache_second) {
            continue; // already marked this second
        }
        seen = {chunk, cache_second};
        resident_chunk_use[chunk] = cache_second;
    }
}

bool BufferCache::GarbageCollectResidentMemory(bool load_screen) {
    if (!vram_reclaim || !instance.CanReportMemoryUsage() || resident_ranges.Empty()) {
        return true;
    }
    if (!pending_binds.empty()) {
        return false;
    }
    const u64 budget = instance.GetTotalMemoryBudget();
    if (budget == 0) {
        return true;
    }
    // Memory in use: free space in the allocator's blocks and in the slabs here takes new data
    // without more from the driver, which counts it as used all the same.
    const u64 driver_usage = instance.GetDeviceMemoryUsage();
    const auto allocator_usage = instance.GetAllocatorUsage();
    const u64 unused =
        (allocator_usage.held - std::min(allocator_usage.held, allocator_usage.in_use)) +
        GetPoolUsage().free;
    const u64 usage = driver_usage - std::min(driver_usage, unused);
    const u64 reserve = std::min<u64>(1_GB, budget / 8);
    // Only near the real limit: each pass waits for the GPU, and evicted buffers the game uses
    // again are made resident again (on 8 GB cards a lower target evicted every few seconds).
    // A loading screen hides the GPU wait, and what the next area needs is uploaded anyway.
    const u64 pressure_limit =
        EmulatorSettings.IsBbMoreVram() ? instance.GetLiveVramLimits().pressure : budget - reserve;
    if (!load_screen && (cache_tick - last_resident_gc_tick < RESIDENT_BUFFER_GC_COOLDOWN ||
                         usage < pressure_limit)) {
        return true;
    }
    const u64 idle_seconds =
        load_screen ? LOAD_SCREEN_BUFFER_IDLE_SECONDS : RESIDENT_BUFFER_IDLE_SECONDS;
    const u64 evict_target =
        load_screen ? std::numeric_limits<u64>::max() : RESIDENT_BUFFER_GC_TARGET;
    const u64 evict_min = load_screen ? 1 : RESIDENT_BUFFER_GC_MIN;
    last_resident_gc_tick = cache_tick;
    const auto pass_start = std::chrono::steady_clock::now();

    // Pieces of resident memory, at most one 2 MiB chunk of guest memory each. The memory itself
    // is shared by many ranges (slabs), so it is unbound piece by piece and freed once nothing
    // is bound to it any more.
    struct Piece {
        Backing backing;
        u64 last_use;
    };
    const u64 chunk_blocks = u64{1} << (RESIDENT_CHUNK_SHIFT - block_shift);
    std::vector<Piece> pieces;
    u64 recent_bytes = 0;
    u64 gpu_written_bytes = 0;
    for (const Backing& backing : resident_ranges) {
        for (u64 start = backing.start; start < backing.end;) {
            const u64 end = std::min(backing.end, (start + chunk_blocks) & ~(chunk_blocks - 1));
            const VAddr address = start << block_shift;
            const u64 size = (end - start) << block_shift;
            const auto use = resident_chunk_use.find(address >> RESIDENT_CHUNK_SHIFT);
            const u64 last_use = use != resident_chunk_use.end() ? use->second : 0;
            if (cache_second - last_use < idle_seconds) {
                recent_bytes += size;
            } else if (gpu_modified_ranges.Intersects(address, size) ||
                       memory_tracker->IsRegionGpuModified(address, size)) {
                // Data the GPU wrote isn't in guest memory: it stays.
                gpu_written_bytes += size;
            } else {
                pieces.push_back({backing.SubRange(start, end), last_use});
            }
            start = end;
        }
    }
    std::ranges::sort(pieces, {}, &Piece::last_use);
    std::vector<Backing> evict;
    u64 evict_size = 0;
    for (const Piece& piece : pieces) {
        evict.push_back(piece.backing);
        evict_size += (piece.backing.end - piece.backing.start) << block_shift;
        if (evict_size >= evict_target) {
            break;
        }
    }
    if (evict_size < evict_min) {
        if (!load_screen && cache_tick - last_resident_diag_tick >= RESIDENT_BUFFER_REPORT_TICKS) {
            last_resident_diag_tick = cache_tick;
            LOG_INFO(Render,
                     "Guest buffers under VRAM pressure ({} of {} MiB): nothing to evict | {} MiB "
                     "used in the last {} s, {} MiB holds GPU-written data, {} MiB idle",
                     usage >> 20, budget >> 20, recent_bytes >> 20, idle_seconds,
                     gpu_written_bytes >> 20, evict_size >> 20);
        }
        return true;
    }

    // No GPU wait (each one was a stutter): the unbind below waits on the GPU for the work
    // submitted so far, and later binds wait for the unbind (SubmitPendingArenaBinds).
    scheduler.EndRendering();

    std::vector<ArenaBinds> unbinds;
    for (const Backing& backing : evict) {
        const u64 guest_addr = backing.start << block_shift;
        const u64 guest_size = (backing.end - backing.start) << block_shift;
        // The next use uploads it again from guest memory.
        memory_tracker->MarkRegionAsCpuModified(guest_addr, guest_size);

        runtime.FillBuffer(bda_pagetable_buffer.get(), backing.start * sizeof(vk::DeviceAddress),
                           (backing.end - backing.start) * sizeof(vk::DeviceAddress), 0);

        for (const Buffer& old_arena : arenas) {
            const u64 arena_start = old_arena.CpuAddr() >> block_shift;
            const u64 arena_end = (old_arena.CpuAddr() + old_arena.SizeBytes()) >> block_shift;
            const u64 start = std::max(backing.start, arena_start);
            const u64 end = std::min(backing.end, arena_end);
            if (start >= end) {
                continue;
            }
            auto bind_it = std::ranges::find(unbinds, &old_arena, &ArenaBinds::arena);
            if (bind_it == unbinds.end()) {
                unbinds.push_back(ArenaBinds{.arena = &old_arena});
                bind_it = std::prev(unbinds.end());
            }
            bind_it->binds.push_back(vk::SparseMemoryBind{
                .resourceOffset = (start << block_shift) - old_arena.CpuAddr(),
                .size = (end - start) << block_shift,
                .memory = {},
                .memoryOffset = 0,
            });
        }
    }

    // Everything recorded so far, the page table clears included, runs before the unbind.
    const u64 work_tick = scheduler.CurrentTick();
    scheduler.Flush();
    std::vector<vk::SparseBufferMemoryBindInfo> buffer_binds;
    buffer_binds.reserve(unbinds.size());
    for (const ArenaBinds& arena_binds : unbinds) {
        buffer_binds.push_back(vk::SparseBufferMemoryBindInfo{
            .buffer = arena_binds.arena->Handle(),
            .bindCount = static_cast<u32>(arena_binds.binds.size()),
            .pBinds = arena_binds.binds.data(),
        });
    }

    const u64 signal_tick = memory_semaphore.NextTick();
    const vk::Semaphore signal_sema = memory_semaphore.Handle();
    const vk::Semaphore wait_sema = scheduler.GetWorkSemaphore()->Handle();
    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = 1u,
        .pWaitSemaphoreValues = &work_tick,
        .signalSemaphoreValueCount = 1u,
        .pSignalSemaphoreValues = &signal_tick,
    };
    const vk::BindSparseInfo sparse_info = {
        .pNext = &timeline_si,
        .waitSemaphoreCount = 1u,
        .pWaitSemaphores = &wait_sema,
        .bufferBindCount = static_cast<u32>(buffer_binds.size()),
        .pBufferBinds = buffer_binds.data(),
        .signalSemaphoreCount = 1u,
        .pSignalSemaphores = &signal_sema,
    };
    vk::Result result;
    // Timed: on some drivers (bbport saw NVIDIA on Linux take 10 s and more) this call is slow.
    const auto unbind_start = std::chrono::steady_clock::now();
    {
        // The graphics queue is shared with the presenter (and frame generation): queue
        // operations must be serialized like every other submission.
        std::scoped_lock submit_lock{Vulkan::Scheduler::submit_mutex};
        Common::Crumbs::Add("buffer eviction: sparse unbind (ranges)", buffer_binds.size());
        result = instance.GetGraphicsQueue().bindSparse(sparse_info);
        Common::Crumbs::Add("buffer eviction: sparse unbind done");
    }
    const double unbind_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - unbind_start)
            .count();
    ASSERT_MSG(result == vk::Result::eSuccess, "Failed to unbind sparse guest memory: {}",
               vk::to_string(result));
    last_unbind_tick = signal_tick;

    u64 freed_bytes = 0;
    u64 freed_allocations = 0;
    for (const Backing& backing : evict) {
        resident_ranges.Remove(backing.start, backing.end);
        ++resident_generation;
        const u64 freed_blocks =
            ReturnResidentBlocks(backing.memory, backing.offset, backing.end - backing.start);
        if (freed_blocks == 0) {
            continue;
        }
        freed_bytes += freed_blocks << block_shift;
        ++freed_allocations;
        retired_memory.push_back({backing.memory, cache_second, signal_tick});
    }
    resident_bytes.fetch_sub(evict_size, std::memory_order_relaxed);
    u64 pool_blocks = 0;
    u64 pool_free_blocks = 0;
    for (const auto& [raw, slab] : resident_slabs) {
        pool_blocks += slab.size_blocks;
        pool_free_blocks += slab.size_blocks - slab.live_blocks;
    }
    const double pass_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pass_start)
            .count();
    namespace GS = Common::GuestStats;
    GS::buffer_evictions.fetch_add(1, std::memory_order_relaxed);
    GS::buffer_eviction_bytes.fetch_add(evict_size, std::memory_order_relaxed);
    GS::buffer_eviction_ns.fetch_add(static_cast<u64>(pass_ms * 1e6), std::memory_order_relaxed);
    GS::buffer_eviction_unbind_ns.fetch_add(static_cast<u64>(unbind_ms * 1e6),
                                            std::memory_order_relaxed);
    LOG_INFO(Render,
             "{}Evicted {} MiB of guest buffers unused for {}+ s ({} pieces), freed {} MiB ({} "
             "allocations) | VRAM {} of {} MiB | kept: {} MiB used recently, {} MiB GPU-written "
             "| buffer memory {} MiB in {} allocations, {} MiB of it free for reuse | unbind call "
             "{:.2f} ms ({} ranges), whole pass {:.2f} ms",
             load_screen ? "Loading screen: " : "", evict_size >> 20, idle_seconds, evict.size(),
             freed_bytes >> 20, freed_allocations, usage >> 20, budget >> 20, recent_bytes >> 20,
             gpu_written_bytes >> 20, (pool_blocks << block_shift) >> 20, resident_slabs.size(),
             (pool_free_blocks << block_shift) >> 20, unbind_ms, evict.size(), pass_ms);
    if (unbind_ms >= SLOW_SPARSE_CALL_MS) {
        LOG_WARNING(Render,
                    "Buffer eviction: the driver took {:.1f} ms to unbind {} ranges. If this "
                    "repeats, turn off \"Also free idle buffers\" and report it with this log",
                    unbind_ms, evict.size());
    }
    return true;
}

bool BufferCache::SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size,
                                    bool is_written, bool is_texel_buffer) {
    boost::container::small_vector<vk::BufferCopy, 4> copies;
    size_t total_size_bytes{};
    std::optional<Common::GuestStats::DetailTimer> timer{std::in_place,
                                                         Common::GuestStats::Detail::BufTrack};
    memory_tracker->ForEachUploadRange(device_addr, size, is_written, [&](u64 addr, u64 size) {
        copies.emplace_back(total_size_bytes, addr, size);
        total_size_bytes += size;
    });
    if (copies.empty()) {
        timer.reset();
    } else {
        timer.emplace(Common::GuestStats::Detail::BufCopy);
    }
    if (!copies.empty()) {
        const auto start = std::chrono::steady_clock::now();
        const auto staging = staging_pool.Request(total_size_bytes, MemoryType::HostUncached);
        bool background = false;
        if (total_size_bytes >= BackgroundBufferCopyMin && Common::BbTuning::Threaded() &&
            EmulatorSettings.IsBbBackgroundTextureCopies()) {
            // Streaming a new area uploads 100+ MiB of vertex and index data in one frame, in a
            // few hundred buffers of ~0.5 MiB: copied here they held up the draw recorder for
            // 30+ ms. On the copy threads instead, like background texture copies: waited for
            // before the submission that reads the staging memory and before the guest sees a
            // fence. The GPU copy below is recorded now, so the order of GPU work is unchanged.
            for (const auto& copy : copies) {
                for (u64 offset = 0; offset < copy.size; offset += BackgroundCopyPiece) {
                    const u64 length = std::min(BackgroundCopyPiece, copy.size - offset);
                    const u64 staged = copy.srcOffset + offset;
                    Common::GuestStats::async_copies.fetch_add(1, std::memory_order_relaxed);
                    Common::CopyPool::Async([memory = memory, source = copy.dstOffset + offset,
                                             destination = staging.mapped + staged, length,
                                             buffer = staging.buffer,
                                             flush = staging.offset + staged] {
                        memory->CopySparseMemory(source, destination, length);
                        buffer->Flush(flush, length);
                    });
                }
            }
            background = true;
        } else if (total_size_bytes >= Common::CopyPool::ParallelThreshold &&
                   Common::BbTuning::Streaming()) {
            // Large uploads (newly streamed data): copied in parallel, in 512 KiB pieces.
            struct Piece {
                VAddr source;
                u64 destination;
                u64 size;
            };
            boost::container::small_vector<Piece, 64> pieces;
            for (const auto& copy : copies) {
                for (u64 offset = 0; offset < copy.size; offset += Common::CopyPool::ChunkSize) {
                    pieces.push_back({copy.dstOffset + offset, copy.srcOffset + offset,
                                      std::min(Common::CopyPool::ChunkSize, copy.size - offset)});
                }
            }
            Common::CopyPool::ParallelFor(pieces.size(), [&](std::size_t i) {
                memory->CopySparseMemory(pieces[i].source, staging.mapped + pieces[i].destination,
                                         pieces[i].size);
            });
        } else {
            for (const auto& copy : copies) {
                memory->CopySparseMemory(copy.dstOffset, staging.mapped + copy.srcOffset,
                                         copy.size);
            }
        }
        for (auto& copy : copies) {
            copy.srcOffset += staging.offset;
            copy.dstOffset -= arena->cpu_addr;
        }
        if (!background) {
            staging.Flush(); // background pieces flush their own range
        }
        runtime.CopyBuffer(staging.buffer, arena, copies);
        namespace GS = Common::GuestStats;
        GS::buffer_upload_bytes.fetch_add(total_size_bytes, std::memory_order_relaxed);
        GS::buffer_upload_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::steady_clock::now() - start)
                                           .count(),
                                       std::memory_order_relaxed);
    }
    if (is_texel_buffer && !is_written) {
        return SynchronizeMemoryFromImage(arena, device_addr, size);
    }
    return false;
}

bool BufferCache::SynchronizeMemoryFromImage(const Buffer* arena, VAddr device_addr, u32 size) {
    if (const auto meta = texture_cache.GetMetaInfo(device_addr)) {
        if (meta->type == TextureCache::MetaType::HTile) {
            static constexpr u32 ZmaskUncompressed = 0xf;
            runtime.FillBuffer(arena, arena->Offset(device_addr), size, ZmaskUncompressed);
            return true;
        } else if (meta->type == TextureCache::MetaType::CMask) {
            const auto expanded_value = CmaskColorExpandedValue(meta->num_samples);
            if (!expanded_value) {
                LOG_WARNING(Render_Vulkan,
                            "Unsupported CMask sample count: address={:#x}, size={:#x}, "
                            "samples={}, owner={}:{}",
                            device_addr, size, meta->num_samples, meta->owner_id.index,
                            meta->owner_uid);
                return false;
            }
            runtime.FillBuffer(arena, arena->Offset(device_addr), size, *expanded_value);
            return true;
        } else {
            LOG_WARNING(Render_Vulkan, "Unhandled metadata type {}",
                        magic_enum::enum_name(meta->type));
        }
    }
    const ImageId image_id = texture_cache.FindImageFromRange(device_addr, size);
    if (!image_id) {
        return false;
    }
    Image& image = texture_cache.GetImage(image_id);
    if (image.dropped_levels != 0) {
        return false;
    }
    ASSERT_MSG(device_addr == image.info.guest_address,
               "Texel buffer aliases image subresources {:x} : {:x}", device_addr,
               image.info.guest_address);
    const u64 arena_offset = arena->Offset(device_addr);
    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    for (u32 mip = 0; mip < image.info.resources.levels; mip++) {
        const auto& mip_info = image.info.mips_layout[mip];
        const u32 width = std::max(image.info.size.width >> mip, 1u);
        const u32 height = std::max(image.info.size.height >> mip, 1u);
        const u32 depth = std::max(image.info.size.depth >> mip, 1u);
        if (arena_offset + mip_info.offset + mip_info.size > arena->size_bytes) {
            break;
        }
        buffer_copies.push_back(vk::BufferImageCopy{
            .bufferOffset = mip_info.offset,
            .bufferRowLength = mip_info.pitch,
            .bufferImageHeight = mip_info.height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = mip,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, depth},
        });
    }
    if (buffer_copies.empty()) {
        return false;
    }
    auto& tile_manager = texture_cache.GetTileManager();
    tile_manager.TileImage(image, buffer_copies, arena, arena_offset);
    NoteGpuWrite(device_addr, size);
    return true;
}

void BufferCache::SubmitPendingArenaBinds(Vulkan::SubmitInfo& info) {
    if (pending_binds.empty()) {
        return;
    }

    std::vector<vk::SparseBufferMemoryBindInfo> buffer_binds;
    buffer_binds.reserve(pending_binds.size());

    for (const auto& binds : pending_binds) {
        buffer_binds.emplace_back(vk::SparseBufferMemoryBindInfo{
            .buffer = binds.arena->Handle(),
            .bindCount = static_cast<u32>(binds.binds.size()),
            .pBinds = binds.binds.data(),
        });
    }

    const u64 signal_tick = memory_semaphore.NextTick();
    const auto signal_sema = memory_semaphore.Handle();

    // Buffer eviction: binds wait for the last unbind, which may still be waiting on the GPU
    // (the same range can be unbound there and bound again here).
    const bool after_unbind = last_unbind_tick != 0;
    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = after_unbind ? 1u : 0u,
        .pWaitSemaphoreValues = after_unbind ? &last_unbind_tick : nullptr,
        .signalSemaphoreValueCount = 1u,
        .pSignalSemaphoreValues = &signal_tick,
    };

    const vk::BindSparseInfo sparse_info = {
        .pNext = &timeline_si,
        .waitSemaphoreCount = after_unbind ? 1u : 0u,
        .pWaitSemaphores = after_unbind ? &signal_sema : nullptr,
        .bufferBindCount = static_cast<u32>(buffer_binds.size()),
        .pBufferBinds = buffer_binds.data(),
        .signalSemaphoreCount = 1u,
        .pSignalSemaphores = &signal_sema,
    };

    info.AddWait(signal_sema, signal_tick);
    const auto bind_start = std::chrono::steady_clock::now();
    Common::Crumbs::Add("buffer memory: sparse bind (ranges)", buffer_binds.size());
    auto submit_result = instance.GetGraphicsQueue().bindSparse(sparse_info);
    Common::Crumbs::Add("buffer memory: sparse bind done");
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");
    if (vram_reclaim) {
        // Buffer eviction makes binding more frequent (evicted ranges come back): report slow
        // driver calls, at most once a second.
        const auto now = std::chrono::steady_clock::now();
        const double bind_ms = std::chrono::duration<double, std::milli>(now - bind_start).count();
        if (bind_ms >= SLOW_SPARSE_CALL_MS &&
            now - last_slow_bind_report >= std::chrono::seconds{1}) {
            last_slow_bind_report = now;
            size_t ranges = 0;
            for (const auto& binds : pending_binds) {
                ranges += binds.binds.size();
            }
            LOG_WARNING(Render, "Sparse bind of {} ranges took {:.1f} ms in the driver", ranges,
                        bind_ms);
        }
    }

    pending_binds.clear();
}

} // namespace VideoCore
