// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Counters of guest-side waits (mutex contention, sleeps), read by the threaded renderer's
// periodic statistics. Only slow paths count, so the counters cost nothing when uncontended.

#pragma once

#include <atomic>

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
/// Background texture copies: started, and time spent waiting for them (submission, fences).
inline std::atomic<u64> async_copies{0};
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

} // namespace Common::GuestStats
