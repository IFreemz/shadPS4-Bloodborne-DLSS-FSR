// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Counters of guest-side waits (mutex contention, sleeps), read by the threaded renderer's
// periodic statistics. Only slow paths count, so the counters cost nothing when uncontended.

#pragma once

#include <array>
#include <atomic>
#include <chrono>

#include "common/types.h"

namespace Common::GuestStats {

inline std::atomic<u64> mutex_waits{0};        ///< mutex locks that had to block
inline std::atomic<u64> sleeps{0};             ///< guest sleeps (usleep, nanosleep)
inline std::atomic<u64> sleep_requested_ns{0}; ///< total time asked for
inline std::atomic<u64> sleep_actual_ns{0};    ///< total time actually slept

/// Page protection changes (GPU write tracking) and the time spent in them.
inline std::atomic<u64> protect_calls{0};
inline std::atomic<u64> protect_pages{0};
inline std::atomic<u64> protect_ns{0};
/// Background pipeline compiles and shader translations started, and draws skipped while one
/// was not ready.
inline std::atomic<u64> background_pipelines_started{0};
inline std::atomic<u64> background_shaders_started{0};
inline std::atomic<u64> background_wait_ns{0}; ///< waits for one needed at once
inline std::atomic<u64> background_skipped_draws{0};
/// Read-only buffer uploads that found nothing to upload without taking the region lock.
inline std::atomic<u64> upload_checks_skipped{0};

/// Renderer events for the threaded renderer's hitch report (frames much slower than usual).
inline std::atomic<u64> gpu_wait_ns{0}; ///< CPU time spent waiting for the GPU
inline std::atomic<u64> gpu_waits{0};
inline std::atomic<u64> slowest_wait_ns{0}; ///< slowest single GPU wait since the last report
inline std::atomic<const char*> slowest_wait_file{nullptr};
inline std::atomic<u32> slowest_wait_line{0};
inline std::atomic<u64> pipelines_compiled{0};
inline std::atomic<u64> arena_binds{0}; ///< buffer arena ranges made resident
/// Texture uploads from guest memory (copy, detile, upload) and the GPU thread time they took.
inline std::atomic<u64> image_uploads{0};
inline std::atomic<u64> image_upload_bytes{0};
inline std::atomic<u64> image_upload_ns{0};
/// Keep rendering while presenting: GPU queue operations that waited for a present to return
/// instead of their thread waiting.
inline std::atomic<u64> queue_ops_behind_present{0};
/// Guest memory copied into staging for texture uploads (part of image_upload_ns).
inline std::atomic<u64> image_copy_ns{0};
/// Texture uploads by what made the texture stale and how long before its upload: bytes per
/// [source][age]. Sources: a new texture (when its data was written is unknown), a game (CPU)
/// write, a GPU write to its memory, a possible game write (small textures, checked by hash).
/// Ages in 60 Hz frames: under 1, 1-2, 2-6, 6-30, 30 or more.
enum class DirtySource : u8 { New, Cpu, Gpu, MaybeCpu, Count };
inline constexpr size_t UploadAgeBuckets = 5;
inline std::array<std::atomic<u64>, static_cast<size_t>(DirtySource::Count) * UploadAgeBuckets>
    upload_age_bytes{};
/// Copy only kept texture levels: guest bytes of left-out levels not copied for uploads.
inline std::atomic<u64> image_skipped_bytes{0};
/// Texture memory allocations (vmaCreateImage): all, and those over 1 ms (a new memory block).
inline std::atomic<u64> image_allocs{0};
inline std::atomic<u64> image_alloc_ns{0};
inline std::atomic<u64> image_slow_allocs{0};
inline std::atomic<u64> image_slow_alloc_ns{0};
/// Lower detail first: new textures whose largest level waited (and its bytes), those whose
/// largest level arrived later (within the per-frame budget, or at once for another use), and
/// how long they were shown softer (total and longest).
inline std::atomic<u64> detail_deferred{0};
inline std::atomic<u64> detail_deferred_bytes{0};
inline std::atomic<u64> detail_restored{0};
inline std::atomic<u64> detail_restored_bytes{0};
inline std::atomic<u64> detail_forced{0};
inline std::atomic<u64> detail_soft_ns{0};
inline std::atomic<u64> detail_soft_max_ns{0};
/// Texture uploads by what asked for them (a texture a draw or dispatch samples, a storage
/// image, a render or depth target, merging overlapping images, the displayed frame, anything
/// else) and by the draw recorder packet running then (draws, indirect draws, dispatches,
/// indirect dispatches, queued jobs such as DMA copies, or not on the draw recorder).
enum class UploadPath : u8 { Texture, Storage, Target, Overlap, Display, Other, Count };
inline constexpr size_t UploadPackets = 6;
inline std::array<std::atomic<u64>, static_cast<size_t>(UploadPath::Count)> upload_path_bytes{};
inline std::array<std::atomic<u64>, UploadPackets> upload_packet_bytes{};
inline thread_local UploadPath upload_path = UploadPath::Other;
inline thread_local u8 upload_packet = UploadPackets - 1;
inline void NoteUploadPath(u64 bytes) {
    upload_path_bytes[static_cast<size_t>(upload_path)].fetch_add(bytes, std::memory_order_relaxed);
    upload_packet_bytes[upload_packet].fetch_add(bytes, std::memory_order_relaxed);
}
struct UploadPathScope {
    explicit UploadPathScope(UploadPath path) : previous{upload_path} {
        upload_path = path;
    }
    ~UploadPathScope() {
        upload_path = previous;
    }
    UploadPathScope(const UploadPathScope&) = delete;
    UploadPathScope& operator=(const UploadPathScope&) = delete;
    UploadPath previous;
};
struct UploadPacketScope {
    explicit UploadPacketScope(u8 packet) : previous{upload_packet} {
        upload_packet = packet < UploadPackets - 1 ? packet : UploadPackets - 1;
    }
    ~UploadPacketScope() {
        upload_packet = previous;
    }
    UploadPacketScope(const UploadPacketScope&) = delete;
    UploadPacketScope& operator=(const UploadPacketScope&) = delete;
    u8 previous;
};
/// New images a shader writes (storage) uploaded from guest memory before the write, by how
/// the shader uses them: only written, the whole image bound and enough compute threads for
/// every texel; also read (image loads or atomics); only part of the image bound; coverage
/// unclear (fewer threads than texels, an indirect dispatch or not compute). And of all of them,
/// those whose guest memory looked all zero (64 places sampled).
enum class NewStorageUse : u8 { Whole, AlsoRead, Part, Unclear, Count };
inline std::array<std::atomic<u64>, static_cast<size_t>(NewStorageUse::Count)> new_storage_bytes{};
inline std::atomic<u64> new_storage_zero_bytes{0};
/// GPU command thread (stutter detail): time and count by PM4 packet type (graphics queue,
/// after any wait for the draw recorder; packets that wait for the game include that wait), and
/// work done on that thread: page protection, buffers made resident, buffer uploads.
inline std::array<std::atomic<u64>, 256> cmd_opcode_ns{};
inline std::array<std::atomic<u64>, 256> cmd_opcode_count{};
inline std::atomic<u64> cmd_protect_calls{0};
inline std::atomic<u64> cmd_protect_ns{0};
inline std::atomic<u64> cmd_resident_ns{0};
inline std::atomic<u64> cmd_buffer_upload_bytes{0};
inline std::atomic<u64> cmd_buffer_upload_ns{0};
/// Clear new shader-written images: of them, cleared on the GPU instead of uploaded.
inline std::atomic<u64> new_storage_cleared_bytes{0};
/// Direct texture copies: guest bytes copied from the game's own addresses.
inline std::atomic<u64> image_direct_bytes{0};
/// Buffer data uploaded from guest memory, and the time the copies took.
inline std::atomic<u64> buffer_upload_bytes{0};
inline std::atomic<u64> buffer_upload_ns{0};
/// Staging memory the upload pool had to allocate (it keeps and reuses what it has).
inline std::atomic<u64> staging_allocations{0};
inline std::atomic<u64> staging_allocation_bytes{0};
inline std::atomic<u64> staging_allocation_ns{0};
/// Time spent making buffer memory resident (allocation and page table updates).
inline std::atomic<u64> resident_ns{0};
/// VRAM cleanup: the image collector's passes (the modes they reached and their time), what it
/// freed and wrote back, uploads at addresses it had freed, and buffer evictions.
inline std::atomic<u64> gc_passes{0};
inline std::atomic<u64> gc_pressure_passes{0};
inline std::atomic<u64> gc_aggressive_passes{0};
inline std::atomic<u64> gc_budget_limit_passes{0}; ///< aggressive only for the driver's figure
inline std::atomic<u64> gc_ns{0};
inline std::atomic<u64> gc_slowest_ns{0}; ///< slowest single pass since the last cleanup report
inline std::atomic<u64> gc_usage_query_ns{0};
inline std::atomic<u64> gc_freed_images{0};
inline std::atomic<u64> gc_freed_bytes{0};
inline std::atomic<u64> gc_writebacks_started{0};
inline std::atomic<u64> gc_writeback_started_bytes{0};
inline std::atomic<u64> gc_writeback_done_ns{0}; ///< finishing write-backs (copy to guest memory)
inline std::atomic<u64> gc_writeback_written_bytes{0};
inline std::atomic<u64> gc_reuploads{0};
inline std::atomic<u64> gc_reupload_bytes{0};
inline std::atomic<u64> gc_reupload_age_ms{0}; ///< summed time between the free and the upload
inline std::atomic<u64> buffer_evictions{0};
inline std::atomic<u64> buffer_eviction_bytes{0};
inline std::atomic<u64> buffer_eviction_ns{0};
inline std::atomic<u64> buffer_eviction_unbind_ns{0};
/// Upload scratch in graphics card memory (detiling, copies): released under VRAM pressure, and
/// allocated (again) when needed.
inline std::atomic<u64> scratch_releases{0};
inline std::atomic<u64> scratch_released_bytes{0};
inline std::atomic<u64> scratch_allocations{0};
inline std::atomic<u64> scratch_allocation_bytes{0};
/// Background texture copies: started, and time spent waiting for them (submission, fences).
inline std::atomic<u64> async_copies{0};
/// Presentation (the swap thread): time in acquiring a window image, in presenting it, and
/// waiting for a free presentation frame (its previous present done), and frames presented.
inline std::atomic<u64> present_acquire_ns{0};
inline std::atomic<u64> present_queue_ns{0};
inline std::atomic<u64> present_frame_wait_ns{0};
inline std::atomic<u64> presents{0};
/// GPU timeline of the draw scheduler's submissions (diagnostics; results come in a frame or
/// two late): time executing them and the gaps between them, and submissions measured.
inline std::atomic<u64> gpu_timeline_busy_ns{0};
inline std::atomic<u64> gpu_timeline_idle_ns{0};
inline std::atomic<u64> gpu_timeline_submits{0};
/// Time the CPU spent in the queue submit call itself (the driver can block there).
inline std::atomic<u64> queue_submit_ns{0};
inline std::atomic<u64> async_copy_wait_ns{0};
/// Diagnostics setting is on (the rasterizer sets it): extra bookkeeping for the reports.
inline std::atomic<bool> diagnostics{false};
/// Files the game opened (streaming).
inline std::atomic<u64> file_opens{0};

/// What the GPU command thread is doing, sampled with the draw recorder's state to find where
/// frame time goes (Rasterizer's state sampler).
enum class CmdState : u8 {
    Idle,        ///< no guest work queued: waiting for the game to submit
    Work,        ///< decoding packets, selecting pipelines, preparing draws
    DrainWait,   ///< waiting for the draw recorder to run dry
    GuestWait,   ///< an unmet wait (WaitRegMem, semaphore) on memory the game/GPU writes
    DisplayWait, ///< waiting for a display buffer to be released by a flip (VO label)
    Command,     ///< running queued commands (flips, readbacks, frame preparation)
    GpuWait,     ///< waiting for the GPU (frame limit, readbacks, finishes)
    Count,
};
inline std::atomic<u8> cmd_state{0};
inline thread_local bool is_cmd_thread = false;
inline std::atomic<bool> recorder_busy{false};

/// Sets the command thread's state and restores the previous one at scope exit.
class CmdStateScope {
public:
    explicit CmdStateScope(CmdState state) {
        if (is_cmd_thread) {
            active = true;
            previous = cmd_state.exchange(static_cast<u8>(state), std::memory_order_relaxed);
        }
    }
    ~CmdStateScope() {
        if (active) {
            cmd_state.store(previous, std::memory_order_relaxed);
        }
    }
    CmdStateScope(const CmdStateScope&) = delete;
    CmdStateScope& operator=(const CmdStateScope&) = delete;

private:
    bool active = false;
    u8 previous = 0;
};

/// "Detailed stutter reports": time of the steps of new textures and buffers (Detail), and the
/// page protection done on the draw recorder. Only timed while `detail_timing` is set.
enum class Detail : u32 {
    TexTrack,        ///< texture memory protected (TrackImage)
    TexStaging,      ///< texture data to staging (copies queued or done)
    TexDetile,       ///< texture detiling recorded
    TexUpload,       ///< texture copy recorded
    TexView,         ///< texture view found or created
    TexCreate,       ///< new texture created and registered
    BufTrack,        ///< buffer ranges to upload found, marked and protected
    BufCopy,         ///< buffer staging and copy queued or recorded
    RecorderProtect, ///< page protection calls made on the draw recorder
    BufStream,       ///< small read-only buffers copied into the stream buffer
    BufResident,     ///< buffer memory made resident (arena blocks bound)
    BufArena,        ///< buffer arena looked up
    Count,
};
inline std::atomic<bool> detail_timing{false};
inline std::array<std::atomic<u64>, static_cast<size_t>(Detail::Count)> detail_ns{};
inline std::atomic<u64> recorder_protect_calls{0};
inline thread_local bool on_recorder = false;
/// The detail step running on this thread (Count: none), and the draw recorder's page
/// protection by the step it happened in (Count: outside any).
inline thread_local Detail current_detail = Detail::Count;
inline std::array<std::atomic<u64>, static_cast<size_t>(Detail::Count) + 1>
    recorder_protect_ns_by{};
inline std::array<std::atomic<u64>, static_cast<size_t>(Detail::Count) + 1>
    recorder_protect_calls_by{};
/// Buffers: small read-only copies into the stream buffer, and uploads from guest memory (buffers
/// with data to upload, their ranges, and the bytes copied right away rather than on the copy
/// threads).
inline std::atomic<u64> stream_copies{0};
inline std::atomic<u64> stream_copy_bytes{0};
inline std::atomic<u64> buffer_syncs{0};
inline std::atomic<u64> buffer_sync_ranges{0};
inline std::atomic<u64> buffer_sync_inline_bytes{0};
/// F1's "Log motion blur inputs": the next motion blur passes (and the depth producer) log
/// every input they read.
inline std::atomic<bool> blur_dump_requested{false};
inline std::atomic<bool> texture_reload_requested{false};
/// Cheaper buffer refill: evicted buffer data uploaded straight from guest memory when made
/// resident again (no page protection change).
inline std::atomic<u64> evicted_refill_bytes{0};
inline std::atomic<u64> evicted_refills{0};
/// Buffer eviction: bytes evicted, and evicted data made resident again, by how long after its
/// eviction: under 10 s, 10-60 s, 1-5 min, later.
inline std::atomic<u64> evicted_bytes_total{0};
/// Protection on a helper thread: its protection calls and their time.
inline thread_local bool on_protect_helper = false;
inline std::atomic<u64> helper_protect_calls{0};
inline std::atomic<u64> helper_protect_ns{0};
inline std::array<std::atomic<u64>, 4> evicted_returned_bytes{};
/// Buffer memory blocks asked from the driver while making buffers resident (and the time).
inline std::atomic<u64> buffer_mem_allocs{0};
inline std::atomic<u64> buffer_mem_alloc_ns{0};
/// Constant engine (stutter detail): constant dumps to game memory, those that first waited
/// for the draw recorder and that wait, the dumps' own time (writes that fault included), and
/// the constant engine's time when run at the start or end of a submission (outside
/// WaitOnCeCounter).
inline std::atomic<u64> ce_dumps{0};
inline std::atomic<u64> ce_dump_drains{0};
inline std::atomic<u64> ce_dump_drain_ns{0};
inline std::atomic<u64> ce_dump_write_ns{0};
inline std::atomic<u64> ce_outside_ns{0};
/// Why constant dumps waited: queued draws read the memory later, queued work (tasks, shader
/// storage writes) writes it, too many queued reads to tell.
inline std::atomic<u64> ce_wait_reads{0};
inline std::atomic<u64> ce_wait_writes{0};
inline std::atomic<u64> ce_wait_overflow{0};
/// Constant dumps past huge writes: dumps written at once that only huge shader writes held up.
inline std::atomic<u64> ce_huge_write_skips{0};
/// Flips (the patched flip NOP on the command thread): count, waiting for the draw recorder
/// first, and the flip signal itself (the game's flip handling, the presenter's frame wait).
inline std::atomic<u64> flips{0};
inline std::atomic<u64> flip_drain_ns{0};
inline std::atomic<u64> flip_signal_ns{0};
/// GPU command thread (stutter detail): waits for the draw recorder before packets, and the
/// compute queues' packets.
inline std::atomic<u64> cmd_drain_ns{0};
inline std::atomic<u64> cmd_compute_ns{0}; ///< compute queue packets (waits included)
inline std::atomic<u64> cmd_compute_packets{0};

/// Adds the scope's time to `detail` while detail timing is on.
class DetailTimer {
public:
    explicit DetailTimer(Detail detail_)
        : detail{detail_}, on{detail_timing.load(std::memory_order_relaxed)},
          previous{current_detail} {
        current_detail = detail;
        if (on) {
            start = std::chrono::steady_clock::now();
        }
    }
    ~DetailTimer() {
        current_detail = previous;
        if (on) {
            detail_ns[static_cast<size_t>(detail)].fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - start)
                    .count(),
                std::memory_order_relaxed);
        }
    }
    DetailTimer(const DetailTimer&) = delete;
    DetailTimer& operator=(const DetailTimer&) = delete;

private:
    Detail detail;
    bool on;
    Detail previous;
    std::chrono::steady_clock::time_point start{};
};

} // namespace Common::GuestStats
