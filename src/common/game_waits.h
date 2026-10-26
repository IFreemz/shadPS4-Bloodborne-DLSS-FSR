// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Where the game's own threads spend a frame: each blocking call into the emulator (sleeps, event
// queues, semaphores, condition variables, contended mutexes, file reads, GPU submissions, write
// faults) and the time between them (busy: running the game's code). A frame much slower than
// usual is logged with the busiest threads in it; every 10 s a summary per thread. Only with
// diagnostics, outside loading screens.

#pragma once

#include <atomic>
#include <chrono>
#include <string_view>

#include "common/types.h"

namespace Common::GameWaits {

inline std::atomic<bool> enabled{false};

/// A game thread starts (or ends): only these threads are measured.
void BeginGameThread(std::string_view name);
void EndGameThread();
bool IsGameThread();

void Record(const char* kind, std::string_view name, std::chrono::nanoseconds duration);
void BeginWait(std::chrono::steady_clock::time_point start);
void EndWait(std::chrono::steady_clock::time_point end);

/// The game submitted a frame (its flip): the last frame's breakdown is measured here.
void FrameBoundary(bool diagnostics, bool loading_screen);

class Scope {
public:
    explicit Scope(const char* kind_, std::string_view name_ = {}) : kind{kind_}, name{name_} {
        if (enabled.load(std::memory_order_relaxed) && IsGameThread()) {
            start = std::chrono::steady_clock::now();
            BeginWait(start);
        }
    }
    ~Scope() {
        if (start != std::chrono::steady_clock::time_point{}) {
            const auto end = std::chrono::steady_clock::now();
            EndWait(end);
            Record(kind, name, end - start);
        }
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    const char* kind;
    std::string_view name;
    std::chrono::steady_clock::time_point start{};
};

} // namespace Common::GameWaits
