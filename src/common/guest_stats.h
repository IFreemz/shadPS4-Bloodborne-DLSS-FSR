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
    Count,
};
inline std::atomic<bool> detail_timing{false};
inline std::array<std::atomic<u64>, static_cast<size_t>(Detail::Count)> detail_ns{};
inline std::atomic<u64> recorder_protect_calls{0};
inline thread_local bool on_recorder = false;

/// Adds the scope's time to `detail` while detail timing is on.
class DetailTimer {
public:
    explicit DetailTimer(Detail detail_)
        : detail{detail_}, on{detail_timing.load(std::memory_order_relaxed)} {
        if (on) {
            start = std::chrono::steady_clock::now();
        }
    }
    ~DetailTimer() {
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
    std::chrono::steady_clock::time_point start{};
};

} // namespace Common::GuestStats
