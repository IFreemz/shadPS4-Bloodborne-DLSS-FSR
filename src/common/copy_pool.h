// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Parallel copies of guest memory (after bbport's copy threads). Streaming a new area uploads
// 50-250 MiB of textures and buffers in one frame; copied on one thread that is tens of
// milliseconds, a visible stutter. Large copies are split across a few threads instead; the
// caller still waits for them, so nothing about ordering changes.

#pragma once

#include <cstddef>
#include <functional>

#include "common/types.h"

namespace Common::CopyPool {

/// Copies at least this large are split across the pool.
constexpr u64 ParallelThreshold = 1ULL << 20;
/// Size of the pieces a large copy is split into.
constexpr u64 ChunkSize = 256ULL << 10;

/// Runs `task(i)` for every i in [0, count) on the pool threads and the calling thread, and
/// returns when all are done. Runs inline when count < 2 or another caller is using the pool.
void ParallelFor(std::size_t count, const std::function<void(std::size_t)>& task);

/// Copies [0, size) in ChunkSize pieces with `copy(offset, length)`, in parallel when size is at
/// least ParallelThreshold.
void SplitCopy(u64 size, const std::function<void(u64, u64)>& copy);

} // namespace Common::CopyPool
