// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Bloodborne tuning switches. Everything the threaded renderer work changed outside the
// renderer itself is behind one of these, so turning the threaded renderer off (or one of the
// toggles under it) gives the emulator's original behaviour back.

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <string_view>

namespace Common::BbTuning {

inline bool IsBloodborne(std::string_view serial) {
    static constexpr std::array<std::string_view, 8> serials{"CUSA00207", "CUSA00208", "CUSA00299",
                                                             "CUSA00900", "CUSA01363", "CUSA03014",
                                                             "CUSA03023", "CUSA03173"};
    return std::ranges::find(serials, serial) != serials.end();
}

/// Faster Windows locks and sleeps. Decided when the game starts (before its threads run), as
/// locks can't change implementation while in use: a restart applies a new setting.
inline std::atomic<bool> fast_locks{false};

/// Set by the rasterizer at every submission: the threaded renderer is running and the toggle
/// is on. Off: the original code paths.
inline std::atomic<bool> threaded{false};      ///< the threaded renderer itself
inline std::atomic<bool> streaming{false};     ///< parallel copies, resident memory slabs, staging
inline std::atomic<bool> merged_copies{false}; ///< copy shader batches in one dispatch
inline std::atomic<bool> merged_copy_commands{false}; ///< test: those batches as copy commands
inline std::atomic<bool> readback_queue{false};       ///< GPU readbacks on the second queue
/// The graphics card's own memory in bytes (0: not known yet), set when the GPU is opened.
inline std::atomic<unsigned long long> vram_bytes{0};
/// The size of the game's mods folder in bytes (texture mods such as BB PC Remaster), measured
/// at game start: big ones need a bigger graphics heap.
inline std::atomic<unsigned long long> mods_bytes{0};
/// How many of the game's biggest map textures are much bigger than the game's own (a texture
/// pack such as BB PC Remaster, merged into the game folder or in the mods folder).
inline std::atomic<unsigned> bigger_map_textures{0};

inline bool FastLocks() {
    return fast_locks.load(std::memory_order_relaxed);
}
inline bool Threaded() {
    return threaded.load(std::memory_order_relaxed);
}
inline bool Streaming() {
    return streaming.load(std::memory_order_relaxed);
}
inline bool MergedCopies() {
    return merged_copies.load(std::memory_order_relaxed);
}
inline bool MergedCopyCommands() {
    return merged_copy_commands.load(std::memory_order_relaxed);
}
inline bool ReadbackQueue() {
    return readback_queue.load(std::memory_order_relaxed);
}

} // namespace Common::BbTuning
