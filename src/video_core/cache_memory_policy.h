// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>

#include "common/types.h"

namespace VideoCore {

// Shared soft targets for reclaimable image and buffer caches. Live resources
// and driver allocations may still exceed these targets.
constexpr u64 CachePressureTarget(u64 budget) {
    // Higher-headroom experiment: keep the conservative policy on smaller
    // budgets, and leave at least a third of larger budgets available.
    if (budget >= 12_GB) {
        return std::min<u64>(9_GB, budget * 2 / 3);
    }
    return std::min<u64>(6_GB, budget / 2);
}

constexpr u64 CacheCriticalTarget(u64 budget) {
    if (budget >= 12_GB) {
        return std::min<u64>(10_GB, budget * 3 / 4);
    }
    return std::min<u64>(7_GB, budget * 3 / 5);
}

} // namespace VideoCore
