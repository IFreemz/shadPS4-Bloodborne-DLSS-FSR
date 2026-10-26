// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <mutex>
#include <string>
#include <vector>
#include <fmt/format.h>

#include "common/game_waits.h"
#include "common/logging/log.h"

namespace Common::GameWaits {

namespace {

using Clock = std::chrono::steady_clock;

constexpr u32 MaxThreads = 128;
constexpr u32 MaxKinds = 16;
constexpr u32 LabelSize = 48;
constexpr u32 OtherKind = MaxKinds - 1;
constexpr u32 HistorySize = 120;

enum : u32 { Free, Alive, Ended };

struct Kind {
    char label[LabelSize]{};
    std::atomic<u64> ns{0};
    std::atomic<u64> count{0};
    std::atomic<u64> longest{0};
};

struct Thread {
    std::atomic<u32> state{Free};
    std::atomic<u32> generation{0};
    char name[32]{};
    std::atomic<u32> kinds_used{0};
    std::array<Kind, MaxKinds> kinds;
    /// Time in blocking calls, finished ones, and the start of the one in progress (0: none).
    std::atomic<u64> wait_ns{0};
    std::atomic<s64> wait_since{0};
    u32 depth{}; ///< nested blocking calls (only the outermost counts as waiting)
};

std::array<Thread, MaxThreads> threads;
std::mutex register_mutex;
thread_local Thread* current = nullptr;

s64 Ticks(Clock::time_point time) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(time.time_since_epoch()).count();
}

struct Snapshot {
    u32 generation{~0u};
    u64 waited{}; ///< wait_ns plus the part of a wait in progress
    std::array<u64, MaxKinds> ns{};
    std::array<u64, MaxKinds> count{};
};

struct Delta {
    u32 slot{};
    double busy_ms{};
    double wait_ms{};
    std::array<double, MaxKinds> ns_ms{};
    std::array<u64, MaxKinds> count{};
    std::array<double, MaxKinds> longest_ms{};
};

struct Reporter {
    std::mutex mutex;
    Clock::time_point last_frame{};
    Clock::time_point period_start{};
    Clock::time_point last_slow_log{};
    std::array<double, HistorySize> history{};
    u64 frames{};
    u64 period_frames{};
    u64 period_slow{};
    u64 suppressed{};
    double period_worst_ms{};
    std::array<Snapshot, MaxThreads> frame_snaps{};
    std::array<Snapshot, MaxThreads> period_snaps{};
};
Reporter reporter;

void Take(const Thread& thread, Snapshot& snap, s64 now) {
    snap.generation = thread.generation.load(std::memory_order_acquire);
    const s64 since = thread.wait_since.load(std::memory_order_acquire);
    snap.waited = thread.wait_ns.load(std::memory_order_acquire) +
                  (since != 0 && now > since ? static_cast<u64>(now - since) : 0);
    const u32 used = thread.kinds_used.load(std::memory_order_acquire);
    for (u32 k = 0; k < MaxKinds; ++k) {
        const bool valid = k < used || k == OtherKind;
        snap.ns[k] = valid ? thread.kinds[k].ns.load(std::memory_order_relaxed) : 0;
        snap.count[k] = valid ? thread.kinds[k].count.load(std::memory_order_relaxed) : 0;
    }
}

Delta Diff(u32 slot, const Snapshot& before, const Snapshot& after, double interval_ms) {
    Delta delta{.slot = slot};
    delta.wait_ms = after.waited > before.waited ? (after.waited - before.waited) / 1e6 : 0.0;
    delta.wait_ms = std::min(delta.wait_ms, interval_ms);
    delta.busy_ms = interval_ms - delta.wait_ms;
    for (u32 k = 0; k < MaxKinds; ++k) {
        delta.ns_ms[k] = after.ns[k] > before.ns[k] ? (after.ns[k] - before.ns[k]) / 1e6 : 0.0;
        delta.count[k] = after.count[k] > before.count[k] ? after.count[k] - before.count[k] : 0;
    }
    return delta;
}

std::string Describe(const Delta& delta, double per, bool with_longest) {
    const Thread& thread = threads[delta.slot];
    std::array<u32, MaxKinds> order{};
    for (u32 k = 0; k < MaxKinds; ++k) {
        order[k] = k;
    }
    std::ranges::sort(order, std::greater{}, [&](u32 k) { return delta.ns_ms[k]; });
    std::string text = fmt::format("{}: busy {:.1f}, waited {:.1f}", thread.name,
                                   delta.busy_ms / per, delta.wait_ms / per);
    std::string kinds;
    for (u32 i = 0; i < 5; ++i) {
        const u32 k = order[i];
        if (delta.ns_ms[k] / per < 0.05) {
            break;
        }
        fmt::format_to(std::back_inserter(kinds), "{}{} {:.1f} x{:.1f}", kinds.empty() ? "" : "; ",
                       thread.kinds[k].label, delta.ns_ms[k] / per,
                       static_cast<double>(delta.count[k]) / per);
        if (with_longest) {
            fmt::format_to(std::back_inserter(kinds), " longest {:.1f}", delta.longest_ms[k]);
        }
    }
    if (!kinds.empty()) {
        fmt::format_to(std::back_inserter(text), " ({})", kinds);
    }
    return text;
}

// The flipping thread and the game's main thread first, then the busiest others. Threads that
// made no call into the emulator in the interval are left out: they wait in calls this does not
// see (or spin), so their busy time says nothing. Threads that only waited are left out too.
std::string Busiest(std::vector<Delta>& deltas, const Thread* flipper, double per,
                    bool with_longest) {
    const auto rank = [&](const Delta& delta) {
        const Thread& thread = threads[delta.slot];
        if (&thread == flipper) {
            return 0;
        }
        return std::strcmp(thread.name, "Game:Main") == 0 ? 1 : 2;
    };
    std::ranges::sort(deltas, [&](const Delta& a, const Delta& b) {
        if (rank(a) != rank(b)) {
            return rank(a) < rank(b);
        }
        return a.busy_ms > b.busy_ms;
    });
    std::string text;
    u32 shown = 0;
    for (const Delta& delta : deltas) {
        u64 calls = 0;
        for (const u64 count : delta.count) {
            calls += count;
        }
        const bool pinned = rank(delta) < 2;
        if (shown == 6 || (!pinned && (calls == 0 || delta.busy_ms / per < 0.5))) {
            continue;
        }
        const char* mark = &threads[delta.slot] == flipper ? "* " : "";
        fmt::format_to(std::back_inserter(text), "{}{}{}{}", shown ? " | " : "", mark,
                       Describe(delta, per, with_longest),
                       calls == 0 ? " (no emulator calls: may be waiting where this cannot see)"
                                  : "");
        ++shown;
    }
    return text;
}

} // namespace

void BeginGameThread(std::string_view name) {
    std::scoped_lock lock{register_mutex};
    for (auto& thread : threads) {
        if (thread.state.load(std::memory_order_acquire) != Free) {
            continue;
        }
        const size_t length = std::min(name.size(), sizeof(thread.name) - 1);
        std::memcpy(thread.name, name.data(), length);
        thread.name[length] = '\0';
        for (auto& kind : thread.kinds) {
            kind.ns.store(0, std::memory_order_relaxed);
            kind.count.store(0, std::memory_order_relaxed);
            kind.longest.store(0, std::memory_order_relaxed);
            kind.label[0] = '\0';
        }
        std::snprintf(thread.kinds[OtherKind].label, LabelSize, "other");
        thread.kinds_used.store(0, std::memory_order_relaxed);
        thread.wait_ns.store(0, std::memory_order_relaxed);
        thread.wait_since.store(0, std::memory_order_relaxed);
        thread.depth = 0;
        thread.generation.fetch_add(1, std::memory_order_release);
        thread.state.store(Alive, std::memory_order_release);
        current = &thread;
        return;
    }
}

void EndGameThread() {
    if (current) {
        current->state.store(Ended, std::memory_order_release);
        current = nullptr;
    }
}

bool IsGameThread() {
    return current != nullptr;
}

void BeginWait(Clock::time_point start) {
    Thread* thread = current;
    if (thread && thread->depth++ == 0) {
        thread->wait_since.store(Ticks(start), std::memory_order_release);
    }
}

void EndWait(Clock::time_point end) {
    Thread* thread = current;
    if (!thread || thread->depth == 0 || --thread->depth != 0) {
        return;
    }
    const s64 since = thread->wait_since.load(std::memory_order_relaxed);
    const s64 now = Ticks(end);
    thread->wait_ns.fetch_add(now > since ? static_cast<u64>(now - since) : 0,
                              std::memory_order_release);
    thread->wait_since.store(0, std::memory_order_release);
}

void Record(const char* kind, std::string_view name, std::chrono::nanoseconds duration) {
    Thread* thread = current;
    if (!thread) {
        return;
    }
    char label[LabelSize];
    if (name.empty()) {
        std::snprintf(label, sizeof(label), "%s", kind);
    } else {
        std::snprintf(label, sizeof(label), "%s %.*s", kind, static_cast<int>(name.size()),
                      name.data());
    }
    const u32 used = thread->kinds_used.load(std::memory_order_relaxed);
    u32 index = OtherKind;
    for (u32 k = 0; k < used; ++k) {
        if (std::strcmp(thread->kinds[k].label, label) == 0) {
            index = k;
            break;
        }
    }
    if (index == OtherKind && used < OtherKind) {
        index = used;
        std::memcpy(thread->kinds[index].label, label, sizeof(label));
        thread->kinds_used.store(used + 1, std::memory_order_release);
    }
    auto& slot = thread->kinds[index];
    const u64 ns = static_cast<u64>(std::max<s64>(duration.count(), 0));
    slot.ns.fetch_add(ns, std::memory_order_relaxed);
    slot.count.fetch_add(1, std::memory_order_relaxed);
    u64 longest = slot.longest.load(std::memory_order_relaxed);
    while (ns > longest &&
           !slot.longest.compare_exchange_weak(longest, ns, std::memory_order_relaxed)) {
    }
}

void FrameBoundary(bool diagnostics, bool loading_screen) {
    enabled.store(diagnostics, std::memory_order_relaxed);
    auto& r = reporter;
    std::scoped_lock lock{r.mutex};
    const auto now = Clock::now();
    if (!diagnostics) {
        r.last_frame = {};
        return;
    }
    const bool first = r.last_frame == Clock::time_point{};
    const double frame_ms =
        first ? 0.0 : std::chrono::duration<double, std::milli>(now - r.last_frame).count();
    r.last_frame = now;
    if (first) {
        r.period_start = now;
        r.frames = r.period_frames = r.period_slow = r.suppressed = 0;
        r.period_worst_ms = 0.0;
    }

    const s64 ticks = Ticks(now);
    std::vector<Delta> frame_deltas;
    for (u32 slot = 0; slot < MaxThreads; ++slot) {
        Thread& thread = threads[slot];
        if (thread.state.load(std::memory_order_acquire) == Free) {
            continue;
        }
        Snapshot snap;
        Take(thread, snap, ticks);
        auto& before = r.frame_snaps[slot];
        if (before.generation != snap.generation) {
            before = snap;
            r.period_snaps[slot] = snap;
        }
        Delta delta = Diff(slot, before, snap, frame_ms);
        for (u32 k = 0; k < MaxKinds; ++k) {
            delta.longest_ms[k] =
                thread.kinds[k].longest.exchange(0, std::memory_order_relaxed) / 1e6;
        }
        before = snap;
        frame_deltas.push_back(delta);
    }
    if (loading_screen) {
        r.period_start = now;
        r.period_frames = r.period_slow = 0;
        r.period_worst_ms = 0.0;
        r.period_snaps = r.frame_snaps;
    }
    if (first || loading_screen) {
        return;
    }

    ++r.period_frames;
    r.period_worst_ms = std::max(r.period_worst_ms, frame_ms);
    const u64 filled = std::min<u64>(r.frames, HistorySize);
    double usual_ms = frame_ms;
    if (filled >= 30) {
        std::array<double, HistorySize> sorted = r.history;
        const auto end = sorted.begin() + filled;
        std::nth_element(sorted.begin(), sorted.begin() + filled / 2, end);
        usual_ms = sorted[filled / 2];
    }
    r.history[r.frames % HistorySize] = frame_ms;
    ++r.frames;
    if (filled >= 30 && frame_ms > usual_ms * 1.3 && frame_ms > usual_ms + 4.0) {
        ++r.period_slow;
        if (now - r.last_slow_log >= std::chrono::milliseconds(500)) {
            r.last_slow_log = now;
            LOG_WARNING(Render,
                        "Game frame {:.1f} ms (usual {:.1f}){} | game threads in this frame, "
                        "ms (* flips the frames; busy: running the game's code): {}",
                        frame_ms, usual_ms,
                        r.suppressed ? fmt::format(", {} more slow frames since the last shown",
                                                   r.suppressed)
                                     : std::string{},
                        Busiest(frame_deltas, current, 1.0, true));
            r.suppressed = 0;
        } else {
            ++r.suppressed;
        }
    }

    if (now - r.period_start < std::chrono::seconds(10)) {
        return;
    }
    const double period_ms =
        std::chrono::duration<double, std::milli>(now - r.period_start).count();
    std::vector<Delta> period_deltas;
    {
        std::scoped_lock register_lock{register_mutex};
        for (u32 slot = 0; slot < MaxThreads; ++slot) {
            Thread& thread = threads[slot];
            const u32 state = thread.state.load(std::memory_order_acquire);
            if (state == Free) {
                continue;
            }
            period_deltas.push_back(
                Diff(slot, r.period_snaps[slot], r.frame_snaps[slot], period_ms));
            r.period_snaps[slot] = r.frame_snaps[slot];
            if (state == Ended) {
                r.frame_snaps[slot].generation = ~0u;
                thread.state.store(Free, std::memory_order_release);
            }
        }
    }
    const double per_frame = static_cast<double>(std::max<u64>(r.period_frames, 1));
    LOG_INFO(Render,
             "Game threads, last {:.0f} s: {} frames ({:.1f} ms on average, {} slow, slowest "
             "{:.1f} ms) | ms per frame (* flips the frames; busy: running the game's code): {}",
             period_ms / 1000.0, r.period_frames, period_ms / per_frame, r.period_slow,
             r.period_worst_ms, Busiest(period_deltas, current, per_frame, false));
    r.period_start = now;
    r.period_frames = r.period_slow = 0;
    r.period_worst_ms = 0.0;
}

} // namespace Common::GameWaits
