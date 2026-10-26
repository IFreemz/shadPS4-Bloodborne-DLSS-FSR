// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "mutex.h"

#include "common/assert.h"
#include "common/bb_tuning.h"
#include "common/game_waits.h"
#include "common/guest_stats.h"

#ifdef _WIN64
#include <immintrin.h>
#pragma comment(lib, "synchronization.lib")
#endif

namespace Libraries::Kernel {

#ifdef _WIN64

TimedMutex::TimedMutex() : fast{Common::BbTuning::FastLocks()} {
    if (!fast) {
        mtx = CreateMutex(nullptr, false, nullptr);
        ASSERT(mtx);
    }
}

TimedMutex::~TimedMutex() {
    if (mtx) {
        CloseHandle(mtx);
    }
}

bool TimedMutex::try_lock() {
    if (!fast) {
        return WaitForSingleObjectEx(mtx, 0, true) == WAIT_OBJECT_0;
    }
    u32 expected = Free;
    return state.compare_exchange_strong(expected, Locked, std::memory_order_acquire,
                                         std::memory_order_relaxed);
}

void TimedMutex::Wait(u32 timeout_ms) {
    Common::GuestStats::mutex_waits.fetch_add(1, std::memory_order_relaxed);
    const Common::GameWaits::Scope wait{"mutex"};
    u32 contended = Contended;
    WaitOnAddress(&state, &contended, sizeof(contended), timeout_ms);
}

void TimedMutex::lock() {
    if (!fast) {
        for (;;) {
            u64 res = WaitForSingleObjectEx(mtx, INFINITE, true);
            if (res == WAIT_OBJECT_0) {
                return;
            }
        }
    }
    if (try_lock()) {
        return;
    }
    // Holders are usually done within a few hundred cycles: spin briefly before sleeping.
    for (u32 spins = 0; spins < 128; ++spins) {
        _mm_pause();
        if (state.load(std::memory_order_relaxed) == Free && try_lock()) {
            return;
        }
    }
    while (state.exchange(Contended, std::memory_order_acquire) != Free) {
        Wait(INFINITE);
    }
}

void TimedMutex::unlock() {
    if (!fast) {
        ReleaseMutex(mtx);
        return;
    }
    if (state.exchange(Free, std::memory_order_release) == Contended) {
        WakeByAddressSingle(&state);
    }
}

#else

TimedMutex::TimedMutex() = default;
TimedMutex::~TimedMutex() = default;

void TimedMutex::lock() {
    mtx.lock();
}

bool TimedMutex::try_lock() {
    return mtx.try_lock();
}

void TimedMutex::unlock() {
    mtx.unlock();
}

#endif

} // namespace Libraries::Kernel
