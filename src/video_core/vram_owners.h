// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>
#include <vk_mem_alloc.h>

#include "common/types.h"

// Graphics card memory by owner (VRAM report): every allocator allocation in device-local memory,
// counted under the name of what made it (a buffer's name, the file and line that created an
// image, or a fixed name).
namespace VideoCore::VramOwners {

inline constexpr u32 NoOwner = ~0u;

/// "Go past the budget": the allocator's pool for block-compressed textures, at a low priority so
/// Windows moves them to system RAM before anything else of the emulator's (null when off).
inline VmaPool low_priority_textures = VK_NULL_HANDLE;

namespace Detail {
struct Slot {
    std::atomic<s64> bytes{0};
    std::atomic<s64> count{0};
    std::string name;
};
inline constexpr u32 MaxSlots = 256;
inline std::array<Slot, MaxSlots> slots;
inline u32 used = 0;
inline std::mutex mutex;
} // namespace Detail

/// The owner's slot (made on first use). Owners past the table's size share its last slot.
inline u32 Register(std::string_view name) {
    using namespace Detail;
    std::scoped_lock lock{mutex};
    for (u32 i = 0; i < used; ++i) {
        if (slots[i].name == name) {
            return i;
        }
    }
    if (used == MaxSlots - 1) {
        slots[used].name = "(other owners)";
        return used;
    }
    if (used == MaxSlots) {
        return MaxSlots - 1;
    }
    slots[used].name = name;
    return used++;
}

/// Counts an allocation for its owner once made (add) or just before it's freed (!add).
inline void Track(u32 owner, VmaAllocator allocator, VmaAllocation allocation, bool add) {
    if (owner == NoOwner || !allocator || !allocation) {
        return;
    }
    VkMemoryPropertyFlags flags{};
    vmaGetAllocationMemoryProperties(allocator, allocation, &flags);
    if (!(flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
        return;
    }
    VmaAllocationInfo info{};
    vmaGetAllocationInfo(allocator, allocation, &info);
    auto& slot = Detail::slots[owner];
    slot.bytes.fetch_add(add ? s64(info.size) : -s64(info.size), std::memory_order_relaxed);
    slot.count.fetch_add(add ? 1 : -1, std::memory_order_relaxed);
}

struct Entry {
    std::string name;
    u64 bytes;
    u64 count;
};

/// Owners holding memory now, the largest first.
inline std::vector<Entry> Snapshot() {
    using namespace Detail;
    std::vector<Entry> entries;
    {
        std::scoped_lock lock{mutex};
        for (u32 i = 0; i < used; ++i) {
            const s64 bytes = slots[i].bytes.load(std::memory_order_relaxed);
            if (bytes > 0) {
                entries.push_back(
                    {slots[i].name, u64(bytes),
                     u64(std::max<s64>(0, slots[i].count.load(std::memory_order_relaxed)))});
            }
        }
    }
    std::ranges::sort(entries, std::greater{}, &Entry::bytes);
    return entries;
}

} // namespace VideoCore::VramOwners
