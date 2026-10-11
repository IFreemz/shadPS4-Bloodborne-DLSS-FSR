// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <chrono>

#include "common/types.h"

#ifdef _WIN64
#include <atomic>
#include <windows.h>
#else
#include <mutex>
#endif

namespace Libraries::Kernel {

class TimedMutex {
public:
    TimedMutex();
    ~TimedMutex();

    void lock();
    bool try_lock();

    void unlock();

    template <class Rep, class Period>
    bool try_lock_for(const std::chrono::duration<Rep, Period>& rel_time) {
#ifdef _WIN64
        constexpr auto zero = std::chrono::duration<Rep, Period>::zero();
        const auto now = std::chrono::steady_clock::now();

        std::chrono::steady_clock::time_point abs_time = now;
        if (rel_time > zero) {
            constexpr auto max = (std::chrono::steady_clock::time_point::max)();
            if (abs_time < max - rel_time) {
                abs_time += rel_time;
            } else {
                abs_time = max;
            }
        }

        return try_lock_until(abs_time);
#else
        return mtx.try_lock_for(rel_time);
#endif
    }

    template <class Clock, class Duration>
    bool try_lock_until(const std::chrono::time_point<Clock, Duration>& abs_time) {
#ifdef _WIN64
        if (!fast) {
            // The original Win32 mutex.
            for (;;) {
                const auto now = Clock::now();
                if (abs_time <= now) {
                    return false;
                }
                const auto rel_ms = std::chrono::ceil<std::chrono::milliseconds>(abs_time - now);
                u64 res = WaitForSingleObjectEx(mtx, static_cast<u64>(rel_ms.count()), true);
                if (res == WAIT_OBJECT_0) {
                    return true;
                } else if (res == WAIT_TIMEOUT) {
                    return false;
                }
            }
        }
        if (try_lock()) {
            return true;
        }
        for (;;) {
            if (state.exchange(Contended, std::memory_order_acquire) == Free) {
                return true;
            }
            const auto now = Clock::now();
            if (abs_time <= now) {
                return false;
            }
            const auto rel_ms = std::chrono::ceil<std::chrono::milliseconds>(abs_time - now);
            Wait(static_cast<u32>(std::min<s64>(static_cast<s64>(rel_ms.count()), 0x7FFFFFFF)));
        }
#else
        return mtx.try_lock_until(abs_time);
#endif
    }

private:
#ifdef _WIN64
    // Locked in user space (an atomic word); the kernel is entered only to wait for a holder
    // (WaitOnAddress). A Win32 mutex object cost two system calls per lock/unlock pair.
    // Only with the faster locks (Bloodborne threaded renderer toggle, chosen at creation);
    // otherwise the original Win32 mutex object.
    static constexpr u32 Free = 0, Locked = 1, Contended = 2;
    void Wait(u32 timeout_ms);
    bool fast;
    std::atomic<u32> state{Free};
    HANDLE mtx{};
#else
    std::timed_mutex mtx;
#endif
};

} // namespace Libraries::Kernel