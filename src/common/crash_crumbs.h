// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

// Crash breadcrumbs: each thread keeps its last steps (what it was doing, with two numbers) in a
// small ring of its own, and a crash prints them for every thread. A crash inside the GPU driver
// then shows which draw, bind or memory change the renderer was in the middle of.
namespace Common::Crumbs {

// A step of the current thread. `what` must be a string literal (only the pointer is kept).
// About 20 ns: fine for every draw.
void Add(const char* what, u64 a = 0, u64 b = 0) noexcept;

// All threads' last steps into the log, oldest first (once; called by the crash handlers).
void Dump() noexcept;

} // namespace Common::Crumbs
