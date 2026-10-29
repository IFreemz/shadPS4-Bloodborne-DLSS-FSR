// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <unordered_map>
#include <utility>
#include <boost/container_hash/hash.hpp>

#include "common/types.h"
#include "video_core/amdgpu/tiling.h"
#include "video_core/buffer_cache/buffer.h"

namespace Vulkan {
struct Runtime;
}

namespace VideoCore {

struct ImageInfo;
struct Image;

class TileManager {
    struct TilingKey {
        AmdGpu::TileMode tile_mode;
        u32 num_bits;
        u32 num_samples;
        bool is_tiler;

        bool operator==(const TilingKey&) const = default;

        struct Hash {
            size_t operator()(const TilingKey& key) const {
                size_t hash = 0;
                boost::hash_combine(hash, key.tile_mode);
                boost::hash_combine(hash, key.num_bits);
                boost::hash_combine(hash, key.num_samples);
                boost::hash_combine(hash, key.is_tiler);
                return hash;
            }
        };
    };

public:
    explicit TileManager(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         Vulkan::Runtime& runtime, StreamBuffer& stream_buffer);
    ~TileManager();

    void TileImage(Image& in_image, std::span<vk::BufferImageCopy> buffer_copies,
                   const VideoCore::Buffer* out_buffer, u64 out_offset);

    /// `flush_barriers` false: the caller has flushed the barriers its inputs need (batched
    /// uploads record several detiles after one barrier), and the output gets memory no other
    /// detile of the batch uses. `trim`: the bytes in front not needed (whole levels, a multiple
    /// of 64 texels and of 256 bytes) are not detiled and get no output memory; offsets (in and
    /// out) are where guest byte 0 would be.
    std::pair<const Buffer*, u64> DetileImage(const VideoCore::Buffer* in_buffer, u64 in_offset,
                                              const ImageInfo& info, bool flush_barriers = true,
                                              u32 trim = 0);

private:
    vk::Pipeline GetTilingPipeline(const ImageInfo& info, bool is_tiler);

private:
    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    Vulkan::Runtime& runtime;
    StreamBuffer& stream_buffer;
    vk::UniqueDescriptorSetLayout desc_layout;
    vk::UniquePipelineLayout pl_layout;
    std::unordered_map<TilingKey, vk::UniquePipeline, TilingKey::Hash> tiling_pipelines;
};

} // namespace VideoCore
