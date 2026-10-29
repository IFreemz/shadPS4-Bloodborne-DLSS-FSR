// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <chrono>

#include <array>
#include <memory>
#include <vector>

#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"

namespace Vulkan {

class Instance;
class Scheduler;

struct StagingBufferRef {
    VideoCore::Buffer* buffer{};
    u64 offset{};
    u64 size{};
    u8* mapped{};
    VideoCore::MemoryType type{};

    void Flush() const noexcept {
        buffer->Flush(offset, size);
    }

    void Invalidate() const noexcept {
        buffer->Invalidate(offset, size);
    }
};

class StagingBufferPool {
public:
    explicit StagingBufferPool(const Instance& instance, Scheduler& scheduler);
    ~StagingBufferPool();

    StagingBufferPool(const StagingBufferPool&) = delete;
    StagingBufferPool& operator=(const StagingBufferPool&) = delete;

    /// Requests size bytes of memory of the given type.
    [[nodiscard]] StagingBufferRef Request(u64 size, VideoCore::MemoryType type, u64 alignment = 0,
                                           bool deferred = false, bool unsynchronized = false);

    /// Releases a deferred allocation. It becomes reusable once the current tick completes.
    void FreeDeferred(const StagingBufferRef& ref);

    /// Releases idle memory.
    void TickFrame();

    /// VRAM cleanup under pressure: frees the graphics card memory blocks and buffers (scratch
    /// for detiling and copies) unused for about a second whose GPU work is done.
    struct ReleasedMemory {
        u64 bytes{};
        u32 ring_blocks{};
        u32 large_buffers{};
    };
    /// emergency (an allocation failed): everything the GPU is done with, no floor or idle time.
    ReleasedMemory ReleaseIdleDeviceLocal(bool emergency = false);

    /// Graphics card memory the pool holds now.
    [[nodiscard]] u64 DeviceLocalBytes() const;

private:
    struct Block {
        std::unique_ptr<VideoCore::StreamBuffer> buffer;
        u64 last_used_frame;
    };

    struct Ring {
        std::vector<Block> blocks;
        size_t current = 0;
    };

    struct LargeBuffer {
        std::unique_ptr<VideoCore::Buffer> buffer;
        u64 tick;
        u64 last_used_frame;
        bool held;
    };

    static constexpr size_t NUM_TYPES = 3;

    StagingBufferRef RequestFromRing(Ring& ring, u64 size, u64 alignment,
                                     VideoCore::MemoryType type, bool unsynchronized);
    StagingBufferRef RequestLarge(u64 size, VideoCore::MemoryType type, bool deferred,
                                  bool unsynchronized);

    void TrimRing(Ring& ring, size_t min_blocks);

    void CountAllocation(u64 bytes, std::chrono::steady_clock::time_point start);
    void TrimLarge(std::vector<LargeBuffer>& cache);

    void PublishStats() const;

    const Instance& instance;
    Scheduler& scheduler;

    std::array<Ring, NUM_TYPES> rings;
    std::array<std::vector<LargeBuffer>, NUM_TYPES> large_caches;
    u64 frame = 0;
    size_t min_upload_blocks = 0;
    /// The memory texture uploads are staged in (BufferCache::ObtainBufferForImage): its ring
    /// keeps min_upload_blocks.
    VideoCore::MemoryType TextureStagingType() const;
    /// Blocks each ring keeps (TrimRing): min_upload_blocks for texture staging, a few for
    /// buffer uploads.
    size_t KeptBlocks(VideoCore::MemoryType type) const;
    std::chrono::steady_clock::time_point frame_time{};
};

} // namespace Vulkan
