// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <numeric>
#include <string>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "core/emulator_settings.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/vram_owners.h"

#include <vk_mem_alloc.h>

namespace VideoCore {

MemoryType DrawDataMemoryType(const Vulkan::Instance& instance) {
    static const MemoryType type = [&] {
        // Resizable BAR (AMD: Smart Access Memory): the CPU-visible part of graphics card memory
        // covers (nearly) all of it, not the classic 256 MiB window.
        const auto props = instance.GetPhysicalDevice().getMemoryProperties();
        constexpr auto bar_flags =
            vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible;
        u64 bar_heap = 0;
        for (u32 i = 0; i < props.memoryTypeCount; ++i)
            if ((props.memoryTypes[i].propertyFlags & bar_flags) == bar_flags)
                bar_heap =
                    std::max<u64>(bar_heap, props.memoryHeaps[props.memoryTypes[i].heapIndex].size);
        const bool rebar = bar_heap > 1024_MB;
        const int setting = EmulatorSettings.GetBbDrawDataMemory(); // 0 auto, 1 card, 2 system
        const bool card = setting == 1 || (setting == 0 && rebar);
        LOG_INFO(Render_Vulkan, "Draw data memory: {} ({}, Resizable BAR {}: {} MiB visible)",
                 card ? "graphics card" : "system RAM", setting == 0 ? "auto" : "setting",
                 rebar ? "on" : "off", bar_heap >> 20);
        return card ? MemoryType::Stream : MemoryType::HostCached;
    }();
    return type;
}

std::string_view BufferTypeName(MemoryType type) {
    switch (type) {
    case MemoryType::HostUncached:
        return "HostUncached";
    case MemoryType::HostCached:
        return "HostCached";
    case MemoryType::Stream:
        return "Stream";
    case MemoryType::DeviceLocal:
        return "DeviceLocal";
    case MemoryType::Sparse:
        return "Sparse";
    default:
        return "Invalid";
    }
}

[[nodiscard]] VkMemoryPropertyFlags MemoryUsagePreferredVmaFlags(MemoryType type) {
    return type != MemoryType::DeviceLocal ? VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                                           : VkMemoryPropertyFlagBits{};
}

[[nodiscard]] VmaAllocationCreateFlags MemoryUsageVmaFlags(MemoryType type) {
    switch (type) {
    case MemoryType::HostUncached:
    case MemoryType::Stream:
        return VMA_ALLOCATION_CREATE_MAPPED_BIT |
               VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    case MemoryType::HostCached:
        return VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    case MemoryType::DeviceLocal:
    default:
        return {};
    }
}

[[nodiscard]] VmaMemoryUsage MemoryUsageVma(MemoryType type) {
    switch (type) {
    case MemoryType::DeviceLocal:
    case MemoryType::Stream:
        return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    case MemoryType::HostUncached:
    case MemoryType::HostCached:
        return VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    default:
        return VMA_MEMORY_USAGE_UNKNOWN;
    }
}

UniqueBuffer::UniqueBuffer(vk::Device device_, VmaAllocator allocator_)
    : device{device_}, allocator{allocator_} {}

UniqueBuffer::~UniqueBuffer() {
    Destroy();
}

void UniqueBuffer::Destroy() {
    if (allocation) {
        VramOwners::Track(vram_owner, allocator, allocation, false);
        vram_owner = VramOwners::NoOwner;
        vmaDestroyBuffer(allocator, buffer, allocation);
    } else if (buffer) {
        device.destroyBuffer(buffer);
    }
    buffer = VK_NULL_HANDLE;
    allocation = VK_NULL_HANDLE;
    bda_addr = 0;
}

void UniqueBuffer::Create(vk::BufferCreateInfo& buffer_ci, MemoryType mem_type,
                          VmaAllocationInfo* out_alloc_info, std::string_view owner_name) {
    const bool with_bda = bool(buffer_ci.usage & vk::BufferUsageFlagBits::eShaderDeviceAddress);
    if (mem_type != MemoryType::Sparse) {
        const VmaAllocationCreateFlags bda_flag =
            with_bda ? VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT : 0;
        const VmaAllocationCreateInfo alloc_ci = {
            .flags =
                VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | bda_flag | MemoryUsageVmaFlags(mem_type),
            .usage = MemoryUsageVma(mem_type),
            .requiredFlags = 0,
            .preferredFlags = MemoryUsagePreferredVmaFlags(mem_type),
            .pool = VK_NULL_HANDLE,
            .pUserData = nullptr,
        };

        const VkBufferCreateInfo buffer_ci_unsafe = static_cast<VkBufferCreateInfo>(buffer_ci);
        VkBuffer unsafe_buffer{};
        VkResult result = vmaCreateBuffer(allocator, &buffer_ci_unsafe, &alloc_ci, &unsafe_buffer,
                                          &allocation, out_alloc_info);
        ASSERT_MSG(result == VK_SUCCESS, "Failed allocating buffer with error {}",
                   vk::to_string(vk::Result{result}));
        buffer = vk::Buffer{unsafe_buffer};
        const std::string name = owner_name.empty()
                                     ? fmt::format("{} buffers", BufferTypeName(mem_type))
                                     : fmt::format("{} ({})", owner_name, BufferTypeName(mem_type));
        vram_owner = VramOwners::Register(name);
        VramOwners::Track(vram_owner, allocator, allocation, true);
    } else {
        buffer_ci.flags |=
            vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency;
        buffer = Vulkan::Check(device.createBuffer(buffer_ci));
    }

    if (with_bda) {
        const vk::BufferDeviceAddressInfo bda_info = {
            .buffer = buffer,
        };
        auto bda_result = device.getBufferAddress(bda_info);
        ASSERT_MSG(bda_result != 0, "Failed to get buffer device address");
        bda_addr = bda_result;
    }
}

Buffer::Buffer(const Vulkan::Instance& instance, VAddr cpu_addr_, u64 size_bytes_,
               MemoryType mem_type_, std::string_view debug_name)
    : cpu_addr{cpu_addr_}, size_bytes{size_bytes_}, mem_type{mem_type_},
      buffer{instance.GetDevice(), instance.GetAllocator()} {

    // Shared with the readback queue's family when it has one of its own (GPU readbacks).
    const auto families = instance.GetBufferQueueFamilies();
    vk::BufferCreateInfo buffer_ci = {
        .size = size_bytes,
        .usage = AllFlags,
        .sharingMode =
            families.size() > 1 ? vk::SharingMode::eConcurrent : vk::SharingMode::eExclusive,
        .queueFamilyIndexCount = families.size() > 1 ? static_cast<u32>(families.size()) : 0u,
        .pQueueFamilyIndices = families.size() > 1 ? families.data() : nullptr,
    };
    VmaAllocationInfo alloc_info{};
    buffer.Create(buffer_ci, mem_type, &alloc_info, debug_name);

    const auto device = instance.GetDevice();
    if (!debug_name.empty()) {
        Vulkan::SetObjectName(device, Handle(), debug_name);
    } else {
        Vulkan::SetObjectName(device, Handle(), "Buffer {:#x}:{:#x}", cpu_addr, size_bytes);
    }

    if (mem_type != MemoryType::Sparse) {
        VkMemoryPropertyFlags property_flags{};
        vmaGetAllocationMemoryProperties(instance.GetAllocator(), buffer.allocation,
                                         &property_flags);
        if (alloc_info.pMappedData) {
            mapped_data = std::span<u8>{std::bit_cast<u8*>(alloc_info.pMappedData), size_bytes};
        }
        is_coherent = property_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    }
}

void Buffer::Flush(u64 offset, u64 size) {
    if (mapped_data.empty() || is_coherent) {
        return;
    }
    vmaFlushAllocation(buffer.allocator, buffer.allocation, offset, size);
}

void Buffer::Invalidate(u64 offset, u64 size) {
    if (mapped_data.empty() || is_coherent) {
        return;
    }
    vmaInvalidateAllocation(buffer.allocator, buffer.allocation, offset, size);
}

constexpr u64 WATCHES_INITIAL_RESERVE = 0x100;
constexpr u64 WATCHES_RESERVE_CHUNK = 0x100;

StreamBuffer::StreamBuffer(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler_,
                           MemoryType mem_type, u64 size_bytes)
    : Buffer{instance, 0, size_bytes, mem_type}, scheduler{scheduler_},
      non_coherent_atom_size{instance.NonCoherentAtomSize()} {
    ReserveWatches(current_watches, WATCHES_INITIAL_RESERVE);
    ReserveWatches(previous_watches, WATCHES_INITIAL_RESERVE);
    Vulkan::SetObjectName(instance.GetDevice(), Handle(), "StreamBuffer({}):{:#x}",
                          BufferTypeName(mem_type), size_bytes);
}

bool StreamBuffer::PrepareMap(u64 size, u64 alignment, bool allow_wait) {
    if (!mapped_data.empty() && !is_coherent) {
        size = Common::AlignUp(size, non_coherent_atom_size);
        alignment =
            alignment > 0 ? std::lcm(alignment, non_coherent_atom_size) : non_coherent_atom_size;
    }

    if (size > this->size_bytes) {
        return false;
    }

    if (alignment > 0) {
        offset = Common::AlignUp(offset, alignment);
    }

    if (offset + size > this->size_bytes) {
        // The buffer would overflow, save the amount of used watches and reset the state.
        invalidation_mark = current_watch_cursor;
        current_watch_cursor = 0;
        offset = 0;

        // Swap watches and reset waiting cursors.
        std::swap(previous_watches, current_watches);
        wait_cursor = 0;
        wait_bound = 0;
    }

    if (!WaitPendingOperations(offset + size, allow_wait)) {
        return false;
    }

    mapped_size = size;
    return true;
}

std::pair<u8*, u64> StreamBuffer::Map(u64 size, u64 alignment, bool allow_wait) {
    if (!PrepareMap(size, alignment, allow_wait)) {
        return {nullptr, 0};
    }
    u8* const data = mapped_data.empty() ? nullptr : mapped_data.data() + offset;
    return {data, offset};
}

void StreamBuffer::Commit() {
    if (mem_type == MemoryType::HostCached) {
        Invalidate(offset, mapped_size);
    } else {
        Flush(offset, mapped_size);
    }
    AdvanceAndWatch();
}

std::optional<u64> StreamBuffer::Reserve(u64 size, u64 alignment, bool allow_wait) {
    if (!PrepareMap(size, alignment, allow_wait)) {
        return std::nullopt;
    }
    const u64 reserved_offset = offset;
    AdvanceAndWatch();
    return reserved_offset;
}

void StreamBuffer::AdvanceAndWatch() {
    offset += mapped_size;
    const u64 tick = scheduler.CurrentTick();
    last_tick = tick;

    // Extend the last watch if it belongs to the same tick.
    if (current_watch_cursor != 0 && current_watches[current_watch_cursor - 1].tick == tick) {
        current_watches[current_watch_cursor - 1].upper_bound = offset;
        return;
    }

    if (current_watch_cursor + 1 >= current_watches.size()) {
        // Ensure that there are enough watches.
        ReserveWatches(current_watches, WATCHES_RESERVE_CHUNK);
    }

    auto& watch = current_watches[current_watch_cursor++];
    watch.upper_bound = offset;
    watch.tick = tick;
}

void StreamBuffer::ReserveWatches(std::vector<Watch>& watches, std::size_t grow_size) {
    watches.resize(watches.size() + grow_size);
}

bool StreamBuffer::WaitPendingOperations(u64 requested_upper_bound, bool allow_wait) {
    if (!invalidation_mark) {
        return true;
    }
    while (requested_upper_bound > wait_bound && wait_cursor < *invalidation_mark) {
        auto& watch = previous_watches[wait_cursor];
        if (!scheduler.IsFree(watch.tick) && !allow_wait) {
            return false;
        }
        scheduler.Wait(watch.tick);
        wait_bound = watch.upper_bound;
        ++wait_cursor;
    }
    return true;
}

} // namespace VideoCore
