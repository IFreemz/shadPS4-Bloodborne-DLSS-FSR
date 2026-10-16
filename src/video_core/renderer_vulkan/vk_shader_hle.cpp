// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include "common/alignment.h"
#include "common/bb_tuning.h"
#include "core/emulator_settings.h"
#include "shader_recompiler/info.h"
#include "video_core/host_shaders/buffer_multi_copy_comp.h"
#include "video_core/renderer_vulkan/vk_gpu_profiler.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_hle.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

extern std::unique_ptr<AmdGpu::Liverpool> liverpool;

namespace Vulkan {

static constexpr u64 COPY_SHADER_HASH = 0xfefebf9f;

// Diagnostics (F3 "Diagnostics in the log"): what the copy shader HLE does, every 10 s.
namespace {
struct CopyStats {
    std::atomic<u64> dispatches, batches, commands, multi, copies, bytes, flushes, cpu_ns;
    std::atomic<u64> small, unaligned, overlapping, too_large;
    std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
};
CopyStats copy_stats;
void Count(std::atomic<u64>& counter, u64 n = 1) {
    counter.fetch_add(n, std::memory_order_relaxed);
}
void LogCopyStats() {
    auto& c = copy_stats;
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - c.last).count();
    if (seconds < 10.0) {
        return;
    }
    c.last = now;
    const auto take = [](std::atomic<u64>& v) { return v.exchange(0, std::memory_order_relaxed); };
    const u64 dispatches = take(c.dispatches), batches = take(c.batches);
    const u64 commands = take(c.commands), multi = take(c.multi), copies = take(c.copies);
    const u64 bytes = take(c.bytes), flushes = take(c.flushes), cpu_ns = take(c.cpu_ns);
    const u64 small = take(c.small), unaligned = take(c.unaligned);
    const u64 overlapping = take(c.overlapping), too_large = take(c.too_large);
    if (dispatches == 0 || commands == 0) {
        return;
    }
    LOG_INFO(Render_Vulkan,
             "Copy shader: {:.0f} game dispatches/s, {:.1f} batches and {:.1f} GPU copy commands "
             "per dispatch, {:.0f} copies/s, {:.1f} MiB/s; commands: multi-copy dispatch "
             "{:.0f}%, vkCmdCopyBuffer {:.0f}% (fewer than 8 copies {:.0f}%, unaligned {:.0f}%, "
             "too large {:.0f}%, copies overlap {:.0f}%); barriers before a command {:.0f}%; "
             "CPU {:.0f} us per dispatch",
             dispatches / seconds, double(batches) / dispatches, double(commands) / dispatches,
             copies / seconds, bytes / seconds / (1 << 20), 100.0 * multi / commands,
             100.0 * (commands - multi) / commands, 100.0 * small / commands,
             100.0 * unaligned / commands, 100.0 * too_large / commands,
             100.0 * overlapping / commands, 100.0 * flushes / commands,
             cpu_ns / 1000.0 / dispatches);
}

// A batch's ranges in its host buffers (barrier checks and access tracking stay this precise).
struct BatchRange {
    u64 src_offset, src_size, dst_offset, dst_size;
};

// True when no destination overlaps another and, within one buffer, no source overlaps a
// destination: the copies may then run in any order, as the game's single dispatch runs them.
bool CopiesIndependent(const VideoCore::Buffer* src, const VideoCore::Buffer* dst,
                       std::span<const vk::BufferCopy> copies) {
    thread_local std::vector<std::pair<u64, u64>> dsts;
    dsts.clear();
    for (const auto& copy : copies) {
        dsts.emplace_back(copy.dstOffset, copy.dstOffset + copy.size);
    }
    std::ranges::sort(dsts);
    for (size_t i = 1; i < dsts.size(); ++i) {
        if (dsts[i].first < dsts[i - 1].second) {
            return false;
        }
    }
    if (src == dst) {
        // Sorted and disjoint, so the ends are sorted too: the first destination ending after a
        // source's start is the only one that can overlap it.
        for (const auto& copy : copies) {
            const auto it =
                std::ranges::upper_bound(dsts, copy.srcOffset, {}, &std::pair<u64, u64>::second);
            if (it != dsts.end() && it->first < copy.srcOffset + copy.size) {
                return false;
            }
        }
    }
    return true;
}

bool RangesAccessed(Runtime& runtime, const VideoCore::Buffer* src, const VideoCore::Buffer* dst,
                    std::span<const BatchRange> ranges) {
    for (const auto& range : ranges) {
        if (runtime.IsBufferAccessed(src, range.src_offset, range.src_size) ||
            runtime.IsBufferAccessed(dst, range.dst_offset, range.dst_size, true)) {
            return true;
        }
    }
    return false;
}
} // namespace

// The copy shader runs ~60-80 times per frame with up to ~1024 small ranges each. As one
// vkCmdCopyBuffer with that many regions the GPU spends ~37 ns per range (bbport measured ~2 ms
// per frame); buffer_multi_copy.comp copies all ranges in one dispatch (after bbport).
// `copies` must be independent (CopiesIndependent).
static bool MultiCopy(Rasterizer& rasterizer, const VideoCore::Buffer* src,
                      const VideoCore::Buffer* dst, std::span<const vk::BufferCopy> copies,
                      std::span<const BatchRange> ranges, bool diagnostics) {
    if (copies.size() < 8) {
        if (diagnostics) {
            Count(copy_stats.small);
        }
        return false;
    }
    u64 src_min = ~0ull, src_max = 0, dst_min = ~0ull, dst_max = 0;
    for (const auto& copy : copies) {
        if ((copy.srcOffset | copy.dstOffset | copy.size) & 3) {
            if (diagnostics) {
                Count(copy_stats.unaligned);
            }
            return false; // dword copies only
        }
        src_min = std::min(src_min, copy.srcOffset);
        src_max = std::max(src_max, copy.srcOffset + copy.size);
        dst_min = std::min(dst_min, copy.dstOffset);
        dst_max = std::max(dst_max, copy.dstOffset + copy.size);
    }
    auto& runtime = rasterizer.GetRuntime();
    auto& scheduler = runtime.GetScheduler();
    const auto& instance = runtime.GetInstance();
    const u64 align = instance.StorageMinAlignment();
    src_min = Common::AlignDown(src_min, align);
    dst_min = Common::AlignDown(dst_min, align);
    // Storage buffer ranges every device supports (maxStorageBufferRange is at least 2^27).
    constexpr u64 MaxRange = 128ull << 20;
    if (src_max - src_min > MaxRange || dst_max - dst_min > MaxRange || copies.size() > 65535) {
        if (diagnostics) {
            Count(copy_stats.too_large);
        }
        return false;
    }

    struct Pipeline {
        vk::UniqueDescriptorSetLayout set_layout;
        vk::UniquePipelineLayout layout;
        vk::UniquePipeline pipeline;
    };
    static Pipeline pipe = [&] {
        const auto device = instance.GetDevice();
        Pipeline p;
        std::array<vk::DescriptorSetLayoutBinding, 3> bindings{};
        for (u32 i = 0; i < 3; ++i) {
            bindings[i] = {.binding = i,
                           .descriptorType = vk::DescriptorType::eStorageBuffer,
                           .descriptorCount = 1,
                           .stageFlags = vk::ShaderStageFlagBits::eCompute};
        }
        p.set_layout = Check(device.createDescriptorSetLayoutUnique({
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
            .bindingCount = static_cast<u32>(bindings.size()),
            .pBindings = bindings.data(),
        }));
        const vk::PushConstantRange range{
            .stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = sizeof(u32)};
        p.layout = Check(device.createPipelineLayoutUnique({
            .setLayoutCount = 1,
            .pSetLayouts = &*p.set_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &range,
        }));
        const auto module = CompileSPV(BUFFER_MULTI_COPY_COMP, device);
        p.pipeline = Check(device.createComputePipelineUnique(
            {}, vk::ComputePipelineCreateInfo{
                    .stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                              .module = module,
                              .pName = "main"},
                    .layout = *p.layout,
                }));
        device.destroyShaderModule(module);
        return p;
    }();

    // The region table goes into the stream buffer (host writes, visible at submission).
    thread_local std::vector<u32> regions;
    regions.resize(copies.size() * 4);
    for (size_t i = 0; i < copies.size(); ++i) {
        regions[i * 4 + 0] = static_cast<u32>((copies[i].srcOffset - src_min) / 4);
        regions[i * 4 + 1] = static_cast<u32>((copies[i].dstOffset - dst_min) / 4);
        regions[i * 4 + 2] = static_cast<u32>(copies[i].size / 4);
        regions[i * 4 + 3] = 0;
    }
    auto& stream = rasterizer.GetBufferCache().GetStreamBuffer();
    const u64 table_size = regions.size() * sizeof(u32);
    const u64 table_offset = stream.Copy(regions.data(), table_size, align);

    scheduler.EndRendering();
    if (RangesAccessed(runtime, src, dst, ranges)) {
        if (diagnostics) {
            Count(copy_stats.flushes);
        }
        runtime.FlushBarriers();
    }
    const u64 src_size = src_max - src_min, dst_size = dst_max - dst_min;
    const u32 count = static_cast<u32>(copies.size());
    scheduler.Record([src = src->Handle(), dst = dst->Handle(), table = stream.Handle(), src_min,
                      src_size, dst_min, dst_size, table_offset, table_size,
                      count](vk::CommandBuffer cmdbuf) {
        const std::array<vk::DescriptorBufferInfo, 3> infos{{
            {src, src_min, src_size},
            {dst, dst_min, dst_size},
            {table, table_offset, table_size},
        }};
        std::array<vk::WriteDescriptorSet, 3> writes{};
        for (u32 i = 0; i < 3; ++i) {
            writes[i] = {.dstBinding = i,
                         .descriptorCount = 1,
                         .descriptorType = vk::DescriptorType::eStorageBuffer,
                         .pBufferInfo = &infos[i]};
        }
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, *pipe.pipeline);
        cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *pipe.layout, 0, writes);
        cmdbuf.pushConstants(*pipe.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(count),
                             &count);
        cmdbuf.dispatch(count, 1, 1);
    });
    // Tracked per batch, not over the whole span: later work only waits for what was written.
    for (const auto& range : ranges) {
        runtime.AccessBuffer(src, range.src_offset, range.src_size,
                             vk::PipelineStageFlagBits2::eComputeShader,
                             vk::AccessFlagBits2::eShaderRead);
        runtime.AccessBuffer(dst, range.dst_offset, range.dst_size,
                             vk::PipelineStageFlagBits2::eComputeShader,
                             vk::AccessFlagBits2::eShaderWrite);
    }
    return true;
}

static bool ExecuteCopyShaderHLE(const Shader::Info& info, const AmdGpu::ComputeProgram& cs_program,
                                 Rasterizer& rasterizer) {
    auto& runtime = rasterizer.GetRuntime();
    auto& buffer_cache = rasterizer.GetBufferCache();
    const bool diagnostics = EmulatorSettings.IsThreadedRendererDiagnostics();
    const auto start =
        diagnostics ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    // Copy shader defines three formatted buffers as inputs: control, source, and destination.
    const auto ctl_buf_sharp = info.buffers[0].GetSharp(info);
    const auto src_buf_sharp = info.buffers[1].GetSharp(info);
    const auto dst_buf_sharp = info.buffers[2].GetSharp(info);
    const auto buf_stride = src_buf_sharp.GetStride();
    ASSERT(buf_stride == dst_buf_sharp.GetStride());

    struct CopyShaderControl {
        u32 dst_idx;
        u32 src_idx;
        u32 end;
    };
    static_assert(sizeof(CopyShaderControl) == 12);
    ASSERT(ctl_buf_sharp.GetStride() == sizeof(CopyShaderControl));
    const auto ctl_buf = reinterpret_cast<const CopyShaderControl*>(ctl_buf_sharp.base_address);

    static std::vector<vk::BufferCopy> copies;
    copies.clear();
    copies.reserve(cs_program.dim_x);

    for (u32 i = 0; i < cs_program.dim_x; i++) {
        const auto& [dst_idx, src_idx, end] = ctl_buf[i];
        const u32 local_dst_offset = dst_idx * buf_stride;
        const u32 local_src_offset = src_idx * buf_stride;
        const u32 local_size = (end + 1) * buf_stride;
        copies.emplace_back(local_src_offset, local_dst_offset, local_size);
    }

    // 64 KiB instead of 64 MiB (bbport): the copies are a few KiB spread over up to 57 MiB, and
    // each batch synchronizes (uploads, marks GPU-modified) its whole range. bbport measured the
    // copy shader's GPU time 1.5 -> 0.6 ms/frame.
    // Off (threaded renderer toggle): the original 64 MiB and one copy command per batch.
    const bool merged = Common::BbTuning::MergedCopies();
    const vk::DeviceSize MaxDistanceForMerge = merged ? 64_KB : 64_MB;
    struct Batch {
        u32 start, end;
        const VideoCore::Buffer* src;
        const VideoCore::Buffer* dst;
        BatchRange range;
    };
    static std::vector<Batch> batches;
    batches.clear();
    u32 batch_start = 0;
    u32 batch_end = 0;

    while (batch_end < copies.size()) {
        // Place first copy into the current batch
        const auto& copy = copies[batch_start];
        auto src_offset_min = copy.srcOffset;
        auto src_offset_max = copy.srcOffset + copy.size;
        auto dst_offset_min = copy.dstOffset;
        auto dst_offset_max = copy.dstOffset + copy.size;

        for (++batch_end; batch_end < copies.size(); batch_end++) {
            // Compute new src and dst bounds if we were to batch this copy
            const auto& [src_offset, dst_offset, size] = copies[batch_end];
            auto new_src_offset_min = std::min(src_offset_min, src_offset);
            auto new_src_offset_max = std::max(src_offset_max, src_offset + size);
            if (new_src_offset_max - new_src_offset_min > MaxDistanceForMerge) {
                break;
            }

            auto new_dst_offset_min = std::min(dst_offset_min, dst_offset);
            auto new_dst_offset_max = std::max(dst_offset_max, dst_offset + size);
            if (new_dst_offset_max - new_dst_offset_min > MaxDistanceForMerge) {
                break;
            }

            // We can batch this copy
            src_offset_min = new_src_offset_min;
            src_offset_max = new_src_offset_max;
            dst_offset_min = new_dst_offset_min;
            dst_offset_max = new_dst_offset_max;
        }

        // Obtain buffers for the total source and destination ranges.
        const auto [src_buf, src_buf_offset] = buffer_cache.ObtainBuffer(
            src_buf_sharp.base_address + src_offset_min, src_offset_max - src_offset_min, false);
        const auto [dst_buf, dst_buf_offset] = buffer_cache.ObtainBuffer(
            dst_buf_sharp.base_address + dst_offset_min, dst_offset_max - dst_offset_min, true);

        // Apply found buffer base.
        for (auto& c : std::span{copies}.subspan(batch_start, batch_end - batch_start)) {
            c.srcOffset = c.srcOffset - src_offset_min + src_buf_offset;
            c.dstOffset = c.dstOffset - dst_offset_min + dst_buf_offset;
        }
        LOG_TRACE(Render_Vulkan, "HLE buffer copy: src_size = {}, dst_size = {}",
                  src_offset_max - src_offset_min, dst_offset_max - dst_offset_min);
        batches.push_back({batch_start, batch_end, src_buf, dst_buf,
                           BatchRange{src_buf_offset, src_offset_max - src_offset_min,
                                      dst_buf_offset, dst_offset_max - dst_offset_min}});
        batch_start = batch_end;
    }

    const auto batch_copies = [&](u32 first, u32 last) {
        return std::span{copies}.subspan(batches[first].start,
                                         batches[last - 1].end - batches[first].start);
    };
    const auto record = [&](const VideoCore::Buffer* src, const VideoCore::Buffer* dst,
                            std::span<vk::BufferCopy> group, std::span<const BatchRange> ranges,
                            bool independent) {
        if (auto* profiler = GpuProfiler::Get()) {
            profiler->Mark(0xC0B1ull, [] { return std::string{"copy shader HLE"}; });
        }
        if (diagnostics) {
            Count(copy_stats.commands);
            u64 bytes = 0;
            for (const auto& c : group) {
                bytes += c.size;
            }
            Count(copy_stats.copies, group.size());
            Count(copy_stats.bytes, bytes);
        }
        if (independent && MultiCopy(rasterizer, src, dst, group, ranges, diagnostics)) {
            if (diagnostics) {
                Count(copy_stats.multi);
            }
            return;
        }
        if (diagnostics && RangesAccessed(runtime, src, dst, ranges)) {
            Count(copy_stats.flushes);
        }
        runtime.CopyBuffer(src, dst, group);
    };

    if (diagnostics) {
        Count(copy_stats.dispatches);
        Count(copy_stats.batches, batches.size());
    }
    if (!merged) {
        for (u32 b = 0; b < batches.size(); ++b) {
            record(batches[b].src, batches[b].dst, batch_copies(b, b + 1),
                   std::span{&batches[b].range, 1}, false);
        }
    } else {
        // The game runs all its copies in one dispatch, in no particular order: consecutive
        // batches in the same buffers become one command (no barriers between them), as long
        // as the copies are independent. Otherwise each batch on its own, in order.
        thread_local std::vector<BatchRange> ranges;
        for (u32 first = 0; first < batches.size();) {
            // Within the storage buffer range every device supports (MultiCopy's limit).
            constexpr u64 MaxSpan = 128ull << 20;
            u64 src_lo = batches[first].range.src_offset;
            u64 src_hi = src_lo + batches[first].range.src_size;
            u64 dst_lo = batches[first].range.dst_offset;
            u64 dst_hi = dst_lo + batches[first].range.dst_size;
            u32 last = first + 1;
            for (; last < batches.size(); ++last) {
                const auto& next = batches[last];
                if (next.src != batches[first].src || next.dst != batches[first].dst) {
                    break;
                }
                const u64 s_lo = std::min(src_lo, next.range.src_offset);
                const u64 s_hi = std::max(src_hi, next.range.src_offset + next.range.src_size);
                const u64 d_lo = std::min(dst_lo, next.range.dst_offset);
                const u64 d_hi = std::max(dst_hi, next.range.dst_offset + next.range.dst_size);
                if (s_hi - s_lo > MaxSpan / 2 || d_hi - d_lo > MaxSpan / 2) {
                    break;
                }
                src_lo = s_lo, src_hi = s_hi, dst_lo = d_lo, dst_hi = d_hi;
            }
            const auto* src = batches[first].src;
            const auto* dst = batches[first].dst;
            ranges.clear();
            for (u32 b = first; b < last; ++b) {
                ranges.push_back(batches[b].range);
            }
            const auto group = batch_copies(first, last);
            if (CopiesIndependent(src, dst, group)) {
                record(src, dst, group, ranges, true);
            } else {
                if (diagnostics) {
                    Count(copy_stats.overlapping);
                }
                for (u32 b = first; b < last; ++b) {
                    record(src, dst, batch_copies(b, b + 1), std::span{&batches[b].range, 1},
                           false);
                }
            }
            first = last;
        }
    }

    if (diagnostics) {
        Count(copy_stats.cpu_ns, std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now() - start)
                                     .count());
        LogCopyStats();
    }
    return true;
}

bool ExecuteShaderHLE(const Shader::Info& info, const AmdGpu::Regs& regs,
                      const AmdGpu::ComputeProgram& cs_program, Rasterizer& rasterizer) {
    switch (info.pgm_hash) {
    case COPY_SHADER_HASH:
        return ExecuteCopyShaderHLE(info, cs_program, rasterizer);
    default:
        return false;
    }
}

} // namespace Vulkan
