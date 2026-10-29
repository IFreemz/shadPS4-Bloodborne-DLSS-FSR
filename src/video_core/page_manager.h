// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <boost/container/small_vector.hpp>
#include "common/alignment.h"
#include "common/types.h"
#include "video_core/buffer_cache/region_definitions.h"

namespace Vulkan {
class Rasterizer;
}

namespace VideoCore {

struct UffdImpl;
struct SignalImpl;

/// Protection on the copy threads: page ranges whose protection a texture's background copies
/// apply just before reading them (address, size).
using ProtectRanges = boost::container::small_vector<std::pair<VAddr, u64>, 4>;

enum class PageOp : s8 {
    None = 0,
    Track = 1,
    Untrack = -1,
};

class PageManager {
    static constexpr size_t PM_PAGE_BITS = 12;
    static constexpr size_t PM_PAGE_SIZE = 1ULL << PM_PAGE_BITS;

public:
    explicit PageManager(Vulkan::Rasterizer* rasterizer);
    ~PageManager();

    /// Register a range of mapped gpu memory.
    void OnGpuMap(VAddr address, size_t size);

    /// Unregister a range of gpu memory that was unmapped.
    void OnGpuUnmap(VAddr address, size_t size);

    /// Updates watches in the pages touching the specified region. With `deferred` (tracking
    /// only), the page states change now but the protection calls are left to ApplyProtection:
    /// their ranges are added to `deferred`.
    void UpdatePageWatchers(VAddr addr, u64 size, PageOp write_op,
                            ProtectRanges* deferred = nullptr) const;

    /// Watches (Track) or stops watching (Untrack) reads of the page with `addr`: no access
    /// while watched. False when the page is not GPU mapped (nothing changed).
    bool UpdatePageReadWatcher(VAddr addr, PageOp read_op) const;

    /// Gives the pages touching the region the protection their states ask for now (deferred
    /// tracking: before the memory is read).
    void ApplyProtection(VAddr addr, u64 size) const;

    /// Protection on a helper thread: ApplyProtection for the ranges, done by one helper thread
    /// in order. Returns the ticket WaitProtection waits for.
    u64 QueueProtection(ProtectRanges ranges);
    /// Waits until the helper thread has protected everything up to `ticket`.
    void WaitProtection(u64 ticket);

    /// Updates watches in the pages touching the inclusive bounds using a mask.
    void UpdatePageWatchersForRegion(VAddr base_addr, const Bounds& bounds,
                                     const RegionBits& write_mask, const RegionBits& read_mask,
                                     PageOp write_op, PageOp read_op) const;

    /// Returns page aligned address.
    static constexpr VAddr GetPageAddr(VAddr addr) {
        return Common::AlignDown(addr, PM_PAGE_SIZE);
    }

    /// Returns address of the next page.
    static constexpr VAddr GetNextPageAddr(VAddr addr) {
        return Common::AlignUp(addr + 1, PM_PAGE_SIZE);
    }

private:
    friend struct UffdImpl;
    friend struct SignalImpl;
    struct Impl;
    struct ProtectWorker;
    std::unique_ptr<Impl> impl;
    /// Protection on a helper thread (after impl: stopped before impl goes).
    std::unique_ptr<ProtectWorker> protect_worker;
};

} // namespace VideoCore
