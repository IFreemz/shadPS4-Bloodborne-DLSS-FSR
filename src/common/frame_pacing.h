// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Frame pacing trace (threaded renderer diagnostics): when each game frame is submitted, when it's
// ready, when the vblank thread flips it and when it's presented. Writes every event to
// frame_pacing.csv in the log folder and a summary to the log every 5 seconds.

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "common/logging/log.h"
#include "common/path_util.h"
#include "common/types.h"

namespace Common::FramePacing {

enum class Event : u8 {
    Submit,  ///< the game asked to flip a buffer (its frame is finished)
    Ready,   ///< the emulator prepared the frame (after waiting for frames in flight)
    Flip,    ///< the vblank thread took the frame (value: vblank count)
    Present, ///< the frame was handed to the swapchain
    Count,
};

constexpr std::array<const char*, static_cast<size_t>(Event::Count)> EventNames = {
    "submit", "ready", "flip", "present"};

inline std::atomic<bool> enabled{false};

namespace Detail {
struct Series {
    s64 last_us = -1;
    u64 last_value = 0;
    std::vector<s64> intervals_us;
    std::array<u32, 5> vblank_gaps{}; ///< flips only: gaps of 1, 2, 3, 4 and 5+ vblanks
};
inline std::mutex mutex;
inline std::array<Series, static_cast<size_t>(Event::Count)> series;
inline std::FILE* file = nullptr;
inline bool file_tried = false;
inline s64 report_us = 0;
inline u32 lines = 0;

inline s64 NowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

inline void Summarize(s64 now_us) {
    const double window_s = (now_us - report_us) / 1e6;
    report_us = now_us;
    std::string text;
    for (size_t i = 0; i < series.size(); ++i) {
        auto& s = series[i];
        auto& v = s.intervals_us;
        if (v.empty()) {
            continue;
        }
        // Jitter: average change between consecutive frame times (0 = perfectly even).
        double jitter = 0;
        for (size_t k = 1; k < v.size(); ++k) {
            jitter += std::abs(static_cast<double>(v[k] - v[k - 1]));
        }
        jitter = v.size() > 1 ? jitter / (v.size() - 1) : 0;
        std::sort(v.begin(), v.end());
        const auto pct = [&v](double p) {
            return v[std::min(v.size() - 1, static_cast<size_t>(p * v.size()))] / 1000.0;
        };
        text += fmt::format(" | {} {:.1f}/s: median {:.2f} ms, 90% {:.2f}, 99% {:.2f}, max {:.2f}, "
                            "jitter {:.2f}",
                            EventNames[i], v.size() / window_s, pct(0.5), pct(0.9), pct(0.99),
                            v.back() / 1000.0, jitter / 1000.0);
        if (static_cast<Event>(i) == Event::Flip) {
            const auto& g = s.vblank_gaps;
            text += fmt::format(" (vblanks apart: 1x{} 2x{} 3x{} 4x{} 5+x{})", g[0], g[1], g[2],
                                g[3], g[4]);
            s.vblank_gaps = {};
        }
        v.clear();
    }
    if (!text.empty()) {
        LOG_INFO(Render, "Frame pacing{}", text);
    }
    if (file) {
        std::fflush(file);
    }
}
} // namespace Detail

/// Records an event. `value` is the vblank count for flips.
inline void Record(Event event, u64 value = 0) {
    if (!enabled.load(std::memory_order_relaxed)) {
        return;
    }
    using namespace Detail;
    const s64 now = NowUs();
    std::scoped_lock lock{mutex};
    if (!file_tried) {
        file_tried = true;
        const auto path = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "frame_pacing.csv";
#ifdef _WIN32
        file = _wfopen(path.c_str(), L"w");
#else
        file = std::fopen(path.c_str(), "w");
#endif
        if (file) {
            std::fputs("time_us,event,interval_us,value\n", file);
            LOG_INFO(Render, "Frame pacing trace: {}", path.string());
        }
        report_us = now;
    }
    auto& s = series[static_cast<size_t>(event)];
    const s64 interval = s.last_us < 0 ? 0 : now - s.last_us;
    if (s.last_us >= 0) {
        s.intervals_us.push_back(interval);
        if (event == Event::Flip) {
            const u64 gap = value - s.last_value;
            ++s.vblank_gaps[std::clamp<u64>(gap, 1, 5) - 1];
        }
    }
    s.last_us = now;
    s.last_value = value;
    if (file) {
        std::fprintf(file, "%lld,%s,%lld,%llu\n", static_cast<long long>(now),
                     EventNames[static_cast<size_t>(event)], static_cast<long long>(interval),
                     static_cast<unsigned long long>(value));
    }
    if (now - report_us >= 5'000'000) {
        Summarize(now);
    }
}

/// Turns the trace on or off (diagnostics setting). Restarts the intervals when it turns on.
inline void SetEnabled(bool on) {
    if (enabled.exchange(on, std::memory_order_relaxed) == on || !on) {
        return;
    }
    using namespace Detail;
    std::scoped_lock lock{mutex};
    for (auto& s : series) {
        s = {};
    }
    report_us = NowUs();
}

} // namespace Common::FramePacing
