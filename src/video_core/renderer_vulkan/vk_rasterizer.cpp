// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstring>
#include <string_view>
#include "common/bb_tuning.h"
#include "common/copy_pool.h"
#include "common/crash_crumbs.h"
#include "common/debug.h"
#include "common/elf_info.h"
#include "common/game_waits.h"
#include "common/guest_stats.h"
#include "common/hash.h"
#include "common/scope_exit.h"
#include "common/thread.h"
#include "core/bb_resolution.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "shader_recompiler/resource.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_gpu_profiler.h"
#include "video_core/renderer_vulkan/vk_hdr_mod.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_hle.h"
#include "video_core/texture_cache/host_compatibility.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/texture_cache.h"

#include <mutex>
#include <optional>
#include <unordered_set>
#include <vk_mem_alloc.h>
#include <xxhash.h>

namespace Vulkan {

static Shader::PushData MakeUserData(const AmdGpu::Regs& regs) {
    // TODO(roamic): Add support for multiple viewports and geometry shaders when ViewportIndex
    // is encountered and implemented in the recompiler.
    Shader::PushData push_data{};
    push_data.xoffset = regs.viewport_control.xoffset_enable ? regs.viewports[0].xoffset : 0.f;
    push_data.xscale = regs.viewport_control.xscale_enable ? regs.viewports[0].xscale : 1.f;
    push_data.yoffset = regs.viewport_control.yoffset_enable ? regs.viewports[0].yoffset : 0.f;
    push_data.yscale = regs.viewport_control.yscale_enable ? regs.viewports[0].yscale : 1.f;
    return push_data;
}

Rasterizer::Rasterizer(const Instance& instance_, Scheduler& scheduler_, Runtime& runtime_,
                       AmdGpu::Liverpool* liverpool_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_}, page_manager{this},
      buffer_cache{instance, scheduler, runtime, liverpool_, texture_cache, page_manager},
      texture_cache{instance, scheduler, runtime, liverpool_, buffer_cache, page_manager},
      liverpool{liverpool_}, memory{Core::Memory::Instance()},
      pipeline_cache{instance, scheduler, liverpool, buffer_cache.GetSparsePageShift()},
      host_markers_enabled{EmulatorSettings.IsVkHostMarkersEnabled()},
      guest_markers_enabled{EmulatorSettings.IsVkGuestMarkersEnabled()} {
    {
        const int copy_threads = EmulatorSettings.GetBbCopyThreads();
        const bool bloodborne =
            Common::BbTuning::IsBloodborne(Common::ElfInfo::Instance().GameSerial());
        const u32 workers =
            Common::CopyPool::Configure(!bloodborne        ? 0u
                                        : copy_threads > 0 ? static_cast<u32>(copy_threads)
                                                           : Common::CopyPool::AutoWorkers());
        LOG_INFO(Render_Vulkan, "Copy threads: {} (setting {})", workers,
                 copy_threads > 0 ? std::to_string(copy_threads) : std::string{"auto"});
    }
    if (!EmulatorSettings.IsNullGPU()) {
        liverpool->BindRasterizer(this);
    }
    memory->SetRasterizer(this);

    // Before any pipeline is selected: G-buffer pipelines only get the motion attachment
    // while this is enabled. Not under RenoDX: it replaces shaders by their exact code, and
    // the motion code would hide the character shaders from it.
    object_motion = std::make_unique<BbObjectMotion>(
        instance, scheduler,
        temporal_dlss.Requested() && temporal_dlss.CharacterMotionRequested() && !RenoDxLoaded());
    temporal_dlss.SetObjectMotion(object_motion.get());
    // Upscaler jitter in the vertex shaders, not the viewport (runtime_info.h, ClipJitter). Not
    // under RenoDX, for the same reason as the motion code.
    Shader::ClipJitter::enabled = temporal_dlss.Requested() && !RenoDxLoaded();
    if (temporal_dlss.Requested()) {
        LOG_INFO(Render_Vulkan, "[DLSS-TEMPORAL] Jitter in {}",
                 Shader::ClipJitter::enabled ? "the vertex shaders" : "the viewport");
    }

    scheduler.SetSubmitCallback([this](Vulkan::SubmitInfo& info) {
        {
            // The tick being submitted (already advanced past it).
            const u64 tick = scheduler.CurrentTick() - 1;
            auto& entry = submit_times[tick % submit_times.size()];
            entry.ns.store(std::chrono::steady_clock::now().time_since_epoch().count(),
                           std::memory_order_relaxed);
            entry.tick.store(tick, std::memory_order_release);
        }
        runtime.FlushBarriers();
        buffer_cache.SubmitPendingArenaBinds(info);
    });

    // Threaded renderer: Bloodborne only (the drains assume its command streams), and not
    // with userfaultfd tracking (its fault thread routes the recording thread's faults to the
    // GPU command thread, which would wait for that thread).
    const auto serial = Common::ElfInfo::Instance().GameSerial();
    if (!EmulatorSettings.IsNullGPU() && !EmulatorSettings.IsUserfaultfdTracking() &&
        Common::BbTuning::IsBloodborne(serial)) {
        pipe_regs = std::make_unique<AmdGpu::Regs>();
        pipe_enabled = EmulatorSettings.IsThreadedRenderer();
        target_memo_enabled = pipe_enabled.load();
        texture_memo = std::make_unique<std::array<TextureMemoEntry, TextureMemoSize>>();
        texture_sets = std::make_unique<std::array<TextureSet, TextureSetCount>>();
        scheduler.EnableThreadedRecording();
        sampler_memo = std::make_unique<std::array<SamplerMemoEntry, SamplerMemoSize>>();
        const_ring = std::make_unique<VideoCore::Buffer>(
            instance, 0, ConstRingSize, VideoCore::DrawDataMemoryType(instance), "Constant ring");
        if (const_ring->mapped_data.empty()) {
            const_ring.reset();
        }
        if (const_ring && EmulatorSettings.IsBbCopyThread()) {
            copy_jobs = std::make_unique<CopyJob[]>(CopyJobCount);
            copy_thread_on = true;
            copy_thread = std::jthread([this](std::stop_token stop) { RunCopyThread(stop); });
        }
        draw_pipe = std::make_unique<DrawPipe>(&Rasterizer::RunDrawPacket, this);
        state_sampler = std::jthread([this](std::stop_token stop) {
            Common::SetCurrentThreadName("shadPS4:StateSampler");
            while (!stop.stop_requested()) {
                if (!diagnostics.load(std::memory_order_relaxed)) {
                    Common::AccurateSleep(std::chrono::milliseconds(100), nullptr, false);
                    continue;
                }
                Common::AccurateSleep(std::chrono::microseconds(250), nullptr, false);
                const u32 a = Common::GuestStats::cmd_state.load(std::memory_order_relaxed);
                const u32 b = Common::GuestStats::recorder_busy.load(std::memory_order_relaxed);
                if (a < static_cast<u32>(Common::GuestStats::CmdState::Count)) {
                    state_samples[a * 2 + b].fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
        LOG_INFO(Render_Vulkan, "Threaded renderer available, {}",
                 pipe_enabled ? "enabled" : "disabled in the settings");
    }
    // Velocity mirror batches are recorded when the guest pass they belong to ends.
    scheduler.SetPassEndHook([this] {
        if (velocity_mirror.HasBatch()) {
            velocity_mirror.FlushBatch(scheduler);
        }
    });
}

Rasterizer::~Rasterizer() {
    scheduler.SetPassEndHook({});
}

bool Rasterizer::FilterDrawPasses() const {
    // FilterDraw's checks without its side effects: false when it would skip the draw or run a
    // pass of its own (fast clear elimination, FMask decompression, resolve, depth copy).
    const auto& regs = Regs();
    using Mode = AmdGpu::ColorControl::OperationMode;
    const auto mode = regs.color_control.mode;
    if (mode == Mode::EliminateFastClear || mode == Mode::FmaskDecompress ||
        mode == Mode::Resolve || regs.primitive_type == AmdGpu::PrimitiveType::None) {
        return false;
    }
    const bool depth_copy =
        regs.depth_render_override.force_z_dirty && regs.depth_render_override.force_z_valid &&
        regs.depth_buffer.DepthValid() && regs.depth_buffer.DepthWriteValid() &&
        regs.depth_buffer.DepthAddress() != regs.depth_buffer.DepthWriteAddress();
    const bool stencil_copy =
        regs.depth_render_override.force_stencil_dirty &&
        regs.depth_render_override.force_stencil_valid && regs.depth_buffer.StencilValid() &&
        regs.depth_buffer.StencilWriteValid() &&
        regs.depth_buffer.StencilAddress() != regs.depth_buffer.StencilWriteAddress();
    return !(mode == Mode::Disable && (depth_copy || stencil_copy));
}

bool Rasterizer::DrawWritesNothing(const GraphicsPipeline* pipeline) {
    if (pipeline->GetStages()[static_cast<u32>(Shader::SwStage::Fragment)]) {
        return false; // colour, depth export, discard or memory writes: drawn
    }
    if (pipeline != no_write_pipeline) {
        no_write_pipeline = pipeline;
        no_write_shaders = true;
        for (const auto* stage : pipeline->GetStages()) {
            if (!stage) {
                continue;
            }
            const bool writes =
                stage->uses_dma || stage->has_storage_images ||
                std::ranges::any_of(stage->buffers, [](const auto& b) { return b.is_written; }) ||
                std::ranges::any_of(stage->images, [](const auto& i) { return i.is_written; });
            no_write_shaders &= !writes;
        }
    }
    if (!no_write_shaders) {
        return false;
    }
    const auto& regs = Regs();
    if (regs.vgt_strmout_config.raw) {
        return false;
    }
    const bool depth = regs.depth_control.depth_enable && regs.depth_buffer.DepthValid();
    const bool stencil = regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid();
    if (depth || stencil) {
        // Not skipped: they can write depth or stencil, and a bound target can start a clear.
        if (!regs.depth_control.depth_write_enable && !stencil) {
            ++depth_test_only_draws;
        }
        return false;
    }
    ++empty_draws_skipped;
    return EmulatorSettings.IsBbSkipEmptyDraws();
}

bool Rasterizer::FilterDraw() {
    const auto& regs = Regs();
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::EliminateFastClear) {
        // Clears the render target if FCE is launched before any draws
        EliminateFastClear();
        return false;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::FmaskDecompress) {
        // TODO: check for a valid MRT1 to promote the draw to the resolve pass.
        LOG_TRACE(Render_Vulkan, "FMask decompression pass skipped");
        ScopedMarkerInsert("FmaskDecompress");
        EliminateFastClear();
        return false;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Resolve) {
        LOG_TRACE(Render_Vulkan, "Resolve pass");
        Resolve();
        return false;
    }
    if (regs.primitive_type == AmdGpu::PrimitiveType::None) {
        LOG_TRACE(Render_Vulkan, "Primitive type 'None' skipped");
        ScopedMarkerInsert("PrimitiveTypeNone");
        return false;
    }

    const bool cb_disabled =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    const auto depth_copy =
        regs.depth_render_override.force_z_dirty && regs.depth_render_override.force_z_valid &&
        regs.depth_buffer.DepthValid() && regs.depth_buffer.DepthWriteValid() &&
        regs.depth_buffer.DepthAddress() != regs.depth_buffer.DepthWriteAddress();
    const auto stencil_copy =
        regs.depth_render_override.force_stencil_dirty &&
        regs.depth_render_override.force_stencil_valid && regs.depth_buffer.StencilValid() &&
        regs.depth_buffer.StencilWriteValid() &&
        regs.depth_buffer.StencilAddress() != regs.depth_buffer.StencilWriteAddress();
    if (cb_disabled && (depth_copy || stencil_copy)) {
        // Games may disable color buffer and enable force depth/stencil dirty and valid to
        // do a copy from one depth-stencil surface to another, without a pixel shader.
        // We need to detect this case and perform the copy, otherwise it will have no effect.
        LOG_TRACE(Render_Vulkan, "Performing depth-stencil override copy");
        DepthStencilCopy(depth_copy, stencil_copy);
        return false;
    }

    return true;
}

VideoCore::ImageId Rasterizer::FindTargetImage(u32 slot, ImageDesc& desc, const void* key,
                                               u32 key_bytes) {
    // `desc` is constructed here from the key's registers on a miss; the key holds exactly the
    // inputs of that construction.
    auto& memo = target_lookup_memo[slot];
    const u32 key_words = key_bytes / sizeof(u32);
    ASSERT(key_bytes % sizeof(u32) == 0 && key_words <= memo.key.size());
    const bool memo_on = target_memo_enabled.load(std::memory_order_relaxed);
    const u64 generation = texture_cache.RegistryGeneration();
    if (memo_on && memo.valid && memo.generation == generation && memo.key_words == key_words &&
        std::memcmp(memo.key.data(), key, key_bytes) == 0) {
        image_memo_hits.fetch_add(1, std::memory_order_relaxed);
        desc = memo.desc;
        texture_cache.TouchFoundImage(memo.image_id);
        return memo.image_id;
    }
    image_memo_misses.fetch_add(1, std::memory_order_relaxed);
    if (slot < AmdGpu::NUM_COLOR_BUFFERS) {
        const auto* k = static_cast<const AmdGpu::ColorBuffer*>(key);
        const auto hint = *reinterpret_cast<const AmdGpu::CbDbExtent*>(static_cast<const u8*>(key) +
                                                                       sizeof(AmdGpu::ColorBuffer));
        std::construct_at(&desc, *k, hint);
    } else {
        const auto& regs = Regs();
        std::construct_at(&desc, regs.depth_buffer, regs.depth_view, regs.depth_control,
                          regs.depth_htile_data_base.GetAddress(), DbExtent());
    }
    const auto image_id = texture_cache.FindImage(desc);
    if (memo_on) {
        memo.valid = true;
        memo.generation = generation;
        memo.key_words = key_words;
        std::memcpy(memo.key.data(), key, key_bytes);
        memo.image_id = image_id;
        memo.desc = desc;
    } else {
        memo.valid = false;
    }
    return image_id;
}

void Rasterizer::PrepareRenderState(const GraphicsPipeline* pipeline) {
    // Prefetch render targets to handle overlaps with bound textures (e.g. mipgen)
    const auto& key = pipeline->GetGraphicsKey();
    const auto& regs = Regs();
    if (regs.color_control.degamma_enable) {
        LOG_WARNING(Render_Vulkan, "Color buffers require gamma correction");
    }

    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    for (s32 cb = 0; cb < std::bit_width(key.mrt_mask); ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (skip_cb_binding || !col_buf || !target_mask || (key.mrt_mask & (1 << cb)) == 0) {
            image_id = {};
            continue;
        }
        const auto hint = CbExtent(cb);
        struct {
            AmdGpu::ColorBuffer buffer;
            AmdGpu::CbDbExtent hint;
        } key{col_buf, hint};
        image_id = bound_images.emplace_back(FindTargetImage(cb, desc, &key, sizeof(key)));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    }

    if ((regs.depth_control.depth_enable && regs.depth_buffer.DepthValid()) ||
        (regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid())) {
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto hint = DbExtent();
        auto& [image_id, desc] = db_desc;
        struct {
            AmdGpu::DepthBuffer buffer;
            AmdGpu::DepthView view;
            AmdGpu::DepthControl control;
            AmdGpu::CbDbExtent hint;
            u32 pad;
            VAddr htile_address;
        } key{regs.depth_buffer, regs.depth_view, regs.depth_control, hint, 0, htile_address};
        image_id = bound_images.emplace_back(
            FindTargetImage(AmdGpu::NUM_COLOR_BUFFERS, desc, &key, sizeof(key)));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    } else {
        db_desc.image_id = {};
    }
}

static std::pair<u32, u32> GetDrawOffsets(const AmdGpu::Regs& regs, const Shader::Info& info,
                                          const Shader::Gcn::FetchShaderData& fetch_shader) {
    u32 vertex_offset = regs.index_offset;
    u32 instance_offset = 0;
    if (!fetch_shader.Empty()) {
        if (vertex_offset == 0 && fetch_shader.vertex_offset_sgpr != -1) {
            vertex_offset = info.UserData()[fetch_shader.vertex_offset_sgpr];
        }
        if (fetch_shader.instance_offset_sgpr != -1) {
            instance_offset = info.UserData()[fetch_shader.instance_offset_sgpr];
        }
    }
    return {vertex_offset, instance_offset};
}

Rasterizer::MotionPrep Rasterizer::ComputeMotionPrep(const GraphicsPipeline* pipeline,
                                                     bool is_indexed, u32 index_offset) {
    MotionPrep prep{};
    const auto& regs = Regs();
    const auto& vs = pipeline->GetStage(Shader::SwStage::Vertex);
    const auto& fetch = pipeline->GetFetchShader();
    if (fetch.Empty()) {
        return prep;
    }
    // Every vertex stream: identical position buffers can go with different skinning data.
    std::array<AmdGpu::Buffer, 32> streams;
    u32 num_streams = 0;
    for (const auto& attribute : fetch.attributes) {
        if (num_streams < streams.size()) {
            streams[num_streams++] = attribute.GetSharp(vs);
        }
    }
    prep.geometry = XXH3_64bits(streams.data(), num_streams * sizeof(AmdGpu::Buffer));
    std::tie(prep.vertex_offset, prep.instance_offset) = GetDrawOffsets(regs, vs, fetch);
    prep.index_size = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16 ? 2u : 4u;
    prep.index_address =
        is_indexed ? regs.index_base_address.Address<VAddr>() + u64(index_offset) * prep.index_size
                   : 0;
    prep.num_indices = regs.num_indices;
    prep.num_instances = regs.num_instances.NumInstances();
    prep.gated = true;
    for (const auto& resource : vs.buffers) {
        if (resource.IsSpecial()) {
            continue;
        }
        const auto buffer = resource.GetSharp(vs);
        const u32 size = u32(buffer.GetSize());
        if (!buffer.Valid() || buffer.GetStride() != 16) {
            continue;
        }
        const auto role = Motion::ClassifyBuffer(size);
        if (role == Motion::BufferRole::Skeleton) {
            prep.gated = false;
            break;
        }
        const VAddr address = buffer.base_address;
        if (role == Motion::BufferRole::SmallSkeleton && memory->IsValidGpuMapping(address, 0) &&
            memory->ClampRangeSize(address, size) == size) {
            prep.palette =
                XXH3_64bits_withSeed(reinterpret_cast<const void*>(address), size, prep.palette);
        }
    }
    // Small skeletons (weapons, props, mostly static world pieces) only get history while
    // their bone palette differs from the last time this draw was seen.
    if (prep.gated) {
        const std::array<u64, 6> identity{vs.pgm_hash,        prep.geometry,
                                          prep.index_address, prep.num_indices,
                                          prep.num_instances, prep.instance_offset};
        const u64 key = XXH3_64bits(identity.data(), sizeof(identity));
        if (motion_gate.size() > 262144) {
            motion_gate.clear();
        }
        // Identical pieces (fence sections, gravestones) share the identity. As in bbport,
        // the k-th copy this frame is compared with the k-th copy last frame; otherwise each
        // copy is compared with another copy, looks moving and loads that copy's positions.
        const u64 frame = stat_frames;
        if (frame != motion_gate_frame) {
            motion_gate_counts.clear();
            motion_gate_frame = frame;
        }
        const u32 occurrence = motion_gate_counts[key]++;
        const u64 gate_key = XXH3_64bits_withSeed(&occurrence, sizeof(occurrence), key);
        auto [it, inserted] = motion_gate.try_emplace(gate_key, MotionGate{prep.palette, frame});
        auto& gate = it.value();
        const bool moving = !inserted && gate.palette != prep.palette && gate.frame + 1 == frame;
        gate = {prep.palette, frame};
        if (!moving) {
            return prep; // valid = false: camera motion is right for it
        }
    }
    // A pseudo-frame for the index range cache (about a frame of skinned draws).
    const u64 frame = ++motion_preps >> 10;
    if ((motion_preps & ((u64(1) << 10) - 1)) == 0 &&
        frame % Motion::IndexRangeCache::Unused == 0) {
        motion_index_ranges.Trim(frame);
    }
    prep.range = {prep.vertex_offset, prep.num_indices};
    if (is_indexed) {
        prep.range = {};
        const u32 count = prep.num_indices;
        const u32 index_size = prep.index_size;
        const VAddr index_address = prep.index_address;
        const u64 bytes = u64(count) * index_size;
        if (!index_address || !memory->IsValidGpuMapping(index_address, 0) ||
            memory->ClampRangeSize(index_address, bytes) != bytes) {
            return prep;
        }
        const bool restart = (regs.enable_primitive_restart & 1) != 0;
        const s32 base = s32(prep.vertex_offset);
        const auto scanned =
            motion_index_ranges.Get({index_address, count, index_size, base, restart}, frame, [&] {
                const auto* data = reinterpret_cast<const void*>(index_address);
                const auto scan_range =
                    index_size == 2
                        ? Motion::IndexedRange(std::span(static_cast<const u16*>(data), count),
                                               base, restart)
                        : Motion::IndexedRange(std::span(static_cast<const u32*>(data), count),
                                               base, restart);
                return Motion::IndexRangeCache::Result{scan_range, XXH3_64bits(data, bytes)};
            });
        prep.range = scanned.range;
        prep.topology = scanned.topology;
    }
    prep.valid = true;
    return prep;
}

// Character motion vectors (ported from bbport): history parameters for a skinned G-buffer draw.
// The inputs come from ComputeMotionPrep (on stage A with the threaded renderer); only the
// per-frame history bookkeeping is done here.
void Rasterizer::PrepareObjectMotion(const GraphicsPipeline* pipeline, bool is_indexed,
                                     u32 index_offset, u32, u32) {
    if (!object_motion->Active()) {
        return;
    }
    const MotionPrep prep =
        packet_motion ? *packet_motion : ComputeMotionPrep(pipeline, is_indexed, index_offset);
    if (!prep.valid) {
        return;
    }
    const u64 param = object_motion->PrepareDraw({
        .shader = pipeline->GetStage(Shader::SwStage::Vertex).pgm_hash,
        .geometry = prep.geometry,
        .indices = prep.index_address,
        .topology = prep.topology,
        .index_count = prep.num_indices,
        .instances = prep.num_instances,
        .first_instance = prep.instance_offset,
        .vertices = prep.range,
    });
    push_data.motion_param = {u32(param), u32(param >> 32)};
}

void Rasterizer::EliminateFastClear() {
    auto& col_buf = Regs().color_buffers[0];
    if (!col_buf || !col_buf.info.fast_clear) {
        return;
    }
    VideoCore::TextureCache::ImageDesc desc(col_buf, CbExtent(0));
    const auto image_id = texture_cache.FindImage(desc);
    const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
    if (!texture_cache.IsMetaCleared(col_buf.CmaskAddress(), col_buf.view.slice_start)) {
        return;
    }
    for (u32 slice = col_buf.view.slice_start; slice <= col_buf.view.slice_max; ++slice) {
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);
    }
    auto& image = texture_cache.GetImage(image_id);
    const auto clear_value = LiverpoolToVK::ColorBufferClearValue(col_buf);
    if (BlurWatched(col_buf.Address())) {
        LOG_INFO(Render_Vulkan, "[BB-BLUR] writer: fast clear of {:#x} carried out (cmask {:#x})",
                 col_buf.Address(), col_buf.CmaskAddress());
    }

    ScopeMarkerBegin(fmt::format("EliminateFastClear:MRT={:#x}:M={:#x}", col_buf.Address(),
                                 col_buf.CmaskAddress()));
    ClearImageAs(image, desc.view_info.range, desc.view_info.format, clear_value);
    ScopeMarkerEnd();
}

void Rasterizer::TemporalDlssDraw(const GraphicsPipeline* pipeline, bool indirect) {
    draw_jitter = {};
    if (!temporal_dlss.Requested()) {
        return;
    }
    const auto& regs = Regs();
    const auto* vs = pipeline->GetStages()[static_cast<u32>(Shader::SwStage::Vertex)];
    const auto* ps = pipeline->GetStages()[static_cast<u32>(Shader::SwStage::Fragment)];
    const BbTemporalDlss::DrawInfo info{
        .vs_hash = vs ? vs->pgm_hash : 0,
        .ps_hash = ps ? ps->pgm_hash : 0,
        .color = std::popcount(pipeline->GetGraphicsKey().mrt_mask) >= 1 ? cb_descs[0].image_id
                                                                         : VideoCore::ImageId{},
        .color_view_format = cb_descs[0].desc.view_info.format,
        .depth = db_desc.image_id,
        .num_indices = regs.num_indices,
        .num_instances = regs.num_instances.NumInstances(),
        .indirect = indirect,
        .depth_test = regs.depth_control.depth_enable != 0,
        .depth_write = regs.depth_control.depth_write_enable != 0,
        .stencil = regs.depth_control.stencil_enable != 0,
    };
    draw_jitter =
        temporal_dlss.OnDraw(instance, runtime, scheduler, texture_cache, velocity_mirror, info);
    // In the vertex shader when it has the code: a viewport with a fractional edge loses its
    // last column/row on some AMD drivers. The viewport of the draw is 2 * scale pixels wide.
    if ((draw_jitter[0] != 0.f || draw_jitter[1] != 0.f) && pipeline->UsesClipJitter() &&
        push_data.xscale != 0.f && push_data.yscale != 0.f) {
        push_data.clip_jitter = {draw_jitter[0] / push_data.xscale,
                                 draw_jitter[1] / push_data.yscale};
        draw_jitter = {};
    }
}

void Rasterizer::BetterAoAfterDraw(const GraphicsPipeline* pipeline) {
    const auto* ps = pipeline->GetStages()[static_cast<u32>(Shader::SwStage::Fragment)];
    if (!ps || !EmulatorSettings.IsBbVbao()) {
        return;
    }
    // The G-buffer normals the game's SSAO reads: its pixel shader's texture 1. The temporal
    // path's last pass (the upsample) reads others.
    const VideoCore::ImageId texture1 =
        ps->images.size() > 1 ? stage_textures[1] : VideoCore::ImageId{};
    if (ps->pgm_hash == BbVbao::SsaoShader) {
        vbao->OnSsaoPass(texture1);
        return;
    }
    if (!BbVbao::AoShader(ps->pgm_hash) || std::popcount(pipeline->GetGraphicsKey().mrt_mask) < 1) {
        return;
    }
    vbao->OnAoPass(cb_descs[0].image_id, cb_descs[0].desc.view_info.format,
                   ps->pgm_hash == BbVbao::AoHalfShader ? texture1 : VideoCore::ImageId{});
}

bool Rasterizer::BetterAoLeavesOutDraw(const GraphicsPipeline* pipeline) {
    const auto* ps = pipeline->GetStages()[static_cast<u32>(Shader::SwStage::Fragment)];
    if (!ps || !EmulatorSettings.IsBbVbao()) {
        return false;
    }
    if (ps->pgm_hash == BbVbao::SsaoShader) {
        return vbao->ReplacesSsaoPass();
    }
    // The upsample writes only the AO; the half-size path's target also holds other channels.
    if (ps->pgm_hash == BbVbao::AoUpsampleShader &&
        std::popcount(pipeline->GetGraphicsKey().mrt_mask) == 1) {
        return vbao->ReplacesAoDraw(cb_descs[0].image_id, cb_descs[0].desc.view_info.format);
    }
    return false;
}

bool Rasterizer::BetterSsrReplacesDraw(const GraphicsPipeline* pipeline) {
    const auto* ps = pipeline->GetStages()[static_cast<u32>(Shader::SwStage::Fragment)];
    const int mode = EmulatorSettings.GetBbSsrMode();
    if (!ps || ps->pgm_hash != BbSsr::TraceShader || (mode != 1 && mode != 2)) {
        return false;
    }
    const auto constants = std::exchange(ssr_constants, {});
    const auto* guest_constants = std::exchange(ssr_guest_constants, nullptr);
    if (ps->images.size() < 3 || std::popcount(pipeline->GetGraphicsKey().mrt_mask) != 1) {
        return false;
    }
    if (!ssr) {
        ssr = std::make_unique<BbSsr>(instance, scheduler, runtime, texture_cache);
    }
    if (mode == 2) {
        return ssr->Clear(cb_descs[0].image_id, cb_descs[0].desc.view_info.format);
    }
    return ssr->Run(cb_descs[0].image_id, cb_descs[0].desc.view_info.format, stage_textures[0],
                    stage_textures[1], stage_textures[2], constants, guest_constants);
}

void Rasterizer::BetterSsrAfterDraw(const GraphicsPipeline* pipeline) {
    const auto* ps = pipeline->GetStages()[static_cast<u32>(Shader::SwStage::Fragment)];
    if (!ps || EmulatorSettings.GetBbSsrMode() != 1 ||
        std::popcount(pipeline->GetGraphicsKey().mrt_mask) < 1) {
        return;
    }
    if (ps->pgm_hash == BbSsr::ColorCopyShader && !ps->images.empty()) {
        ssr->OnColorCopy(stage_textures[0], cb_descs[0].image_id);
    } else if (ps->pgm_hash == BbSsr::LightingShader &&
               ps->images.size() > BbSsr::LightingReflectionSlot) {
        ssr->OnLighting(stage_textures[BbSsr::LightingGlossSlot]);
    }
}

// Fills color target 0 with a copy of another image of the same size and format, with the same
// bookkeeping a draw into it gets.
void Rasterizer::CopyInsteadOfDraw(const BbTemporalDlss::DrawReplacement& replacement) {
    auto& [target_id, desc] = cb_descs[0];
    Common::GuestStats::UploadPathScope path{Common::GuestStats::UploadPath::Target};
    texture_cache.UpdateImage(target_id);
    texture_cache.FindRenderTarget(target_id, desc);
    texture_cache.TouchMeta(Regs().color_buffers[0].CmaskAddress(), desc.view_info.range.base.layer,
                            false);
    auto& target = texture_cache.GetImage(target_id);
    scheduler.EndRendering();
    vk::Image source_image = replacement.owned;
    if (replacement.image) {
        auto& source = texture_cache.GetImage(*replacement.image);
        runtime.Transit(&source, vk::ImageLayout::eTransferSrcOptimal,
                        vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead);
        source_image = source.GetImage();
    }
    runtime.Transit(&target, vk::ImageLayout::eTransferDstOptimal,
                    vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite);
    runtime.FlushBarriers();
    const vk::ImageCopy region{
        .srcSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 1},
        .dstSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                           .baseArrayLayer = desc.view_info.range.base.layer,
                           .layerCount = 1},
        .extent = {target.info.size.width, target.info.size.height, 1}};
    scheduler.CommandBuffer().copyImage(source_image, vk::ImageLayout::eTransferSrcOptimal,
                                        target.GetImage(), vk::ImageLayout::eTransferDstOptimal,
                                        region);
}

// Draws the display copy again at the output size, reading the upscaled frame (see
// BbTemporalDlss::DisplayCopyReplay).
void Rasterizer::ReplayDisplayCopy(const GraphicsPipeline* pipeline, const RenderState& state,
                                   const std::function<void()>& draw) {
    const auto replay = temporal_dlss.TakeDisplayCopyReplay();
    if (!replay) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    const auto input =
        std::ranges::find(first_texture_infos, replay->shader_hash, &std::pair<u64, u32>::first);
    auto& dynamic = scheduler.GetDynamicState();
    scheduler.EndRendering();
    if (input == first_texture_infos.end() || input->second >= image_infos.size() ||
        state.num_color_attachments != 1 || state.num_layers != 1 ||
        state.depth_stencil_attachment.has_depth || state.width == 0 || state.height == 0 ||
        dynamic.viewports.size() != 1 || dynamic.scissors.size() != 1) {
        temporal_dlss.FinishDisplayCopyReplay(cmdbuf, false);
        return;
    }
    auto& info = image_infos[input->second];
    const auto guest_info = info;
    const auto viewports = dynamic.viewports;
    const auto scissors = dynamic.scissors;
    const float fx = float(replay->extent.width) / state.width;
    const float fy = float(replay->extent.height) / state.height;
    auto scaled_viewports = viewports;
    auto& viewport = scaled_viewports[0];
    viewport.x *= fx;
    viewport.y *= fy;
    viewport.width *= fx;
    viewport.height *= fy;
    auto scaled_scissors = scissors;
    auto& scissor = scaled_scissors[0];
    scissor.offset.x = s32(std::lround(scissor.offset.x * fx));
    scissor.offset.y = s32(std::lround(scissor.offset.y * fy));
    scissor.extent.width = u32(std::lround(scissor.extent.width * fx));
    scissor.extent.height = u32(std::lround(scissor.extent.height * fy));
    dynamic.SetViewports(scaled_viewports);
    dynamic.SetScissors(scaled_scissors);
    dynamic.Commit(instance, cmdbuf);
    const auto replay_into = [&](vk::ImageView input_view, vk::ImageView output_view) {
        info = vk::DescriptorImageInfo{guest_info.sampler, input_view,
                                       vk::ImageLayout::eShaderReadOnlyOptimal};
        pipeline->BindResources(set_writes, push_data);
        RenderState target{};
        target.color_attachments[0].image_view = output_view;
        target.color_attachments[0].image_layout = vk::ImageLayout::eColorAttachmentOptimal;
        target.width = u16(replay->extent.width);
        target.height = u16(replay->extent.height);
        target.num_layers = 1;
        target.num_color_attachments = 1;
        scheduler.BeginRendering(target);
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline->Handle());
        draw();
        scheduler.EndRendering();
    };
    if (replay->output)
        replay_into(replay->input, replay->output);
    if (replay->hudless_output)
        replay_into(replay->hudless_input, replay->hudless_output);
    temporal_dlss.FinishDisplayCopyReplay(cmdbuf, true);
    // Back to the guest's bindings and state for whatever is recorded next.
    info = guest_info;
    dynamic.SetViewports(viewports);
    dynamic.SetScissors(scissors);
    dynamic.Commit(instance, cmdbuf);
}

void Rasterizer::ReplayVelocityMirror(const GraphicsPipeline* pipeline, const RenderState& state,
                                      bool is_indexed, const std::function<void()>& draw) {
    if (!velocity_mirror.Requested()) {
        return;
    }
    auto* depth = db_desc.image_id ? &texture_cache.GetImage(db_desc.image_id) : nullptr;
    const auto layer = db_desc.image_id ? db_desc.desc.view_info.range.base.layer : 0;
    bool guest_pass_ended = false;
    // Deferred into one mirror pass per run of mirrored draws (recorded when the guest pass
    // ends): the draw's bindings and draw are captured with the mirror's viewport.
    velocity_mirror.Defer(
        instance, runtime, scheduler, *pipeline, state, depth, layer,
        [&](BbVelocityMirror::Commands& out, const Viewports& viewports, const Scissors& scissors) {
            scheduler.BeginCapture(&out);
            RecordDrawBindings(pipeline, is_indexed, &viewports, &scissors);
            draw();
            scheduler.EndCapture();
        },
        guest_pass_ended);
    if (guest_pass_ended) {
        mirror_rebind = true;
    }
}

void Rasterizer::RecordDrawBindings(const GraphicsPipeline* pipeline, bool is_indexed,
                                    const Viewports* viewports, const Scissors* scissors) {
    RecordVertexBindings();
    if (is_indexed) {
        RecordIndexBinding();
    }
    pipeline->BindResources(set_writes, push_data);
    auto snapshot = scheduler.GetDynamicState();
    snapshot.Invalidate();
    if (viewports && scissors) {
        snapshot.viewports = *viewports;
        snapshot.scissors = *scissors;
    }
    scheduler.Record([snapshot, &instance = instance](vk::CommandBuffer cmdbuf) mutable {
        snapshot.Commit(instance, cmdbuf);
    });
}

void Rasterizer::RebindAfterMirror(const GraphicsPipeline* pipeline, const RenderState& state,
                                   bool is_indexed, u64 flushes_before) {
    if (!mirror_rebind && velocity_mirror.Flushes() == flushes_before) {
        return;
    }
    // A mirror batch (or the frame's depth seed) was recorded after this draw's bindings:
    // the guest pass, pipeline, descriptors, vertex input and dynamic state are bound again.
    mirror_rebind = false;
    scheduler.BeginRendering(state);
    scheduler.Record([handle = pipeline->Handle()](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, handle);
    });
    RecordDrawBindings(pipeline, is_indexed, nullptr, nullptr);
    scheduler.GetDynamicState().MarkCommitted(instance);
}

void Rasterizer::RecordVertexBindings() {
    const auto& v = last_vertex;
    if (v.dynamic) {
        scheduler.Record(
            [bindings = v.bindings, attributes = v.attributes](vk::CommandBuffer cmdbuf) {
                cmdbuf.setVertexInputEXT(bindings, attributes);
            });
    }
    if (v.num_buffers == 0) {
        return;
    }
    if (v.dynamic) {
        scheduler.Record([num_buffers = v.num_buffers, buffers = v.buffers,
                          offsets = v.offsets](vk::CommandBuffer cmdbuf) {
            cmdbuf.bindVertexBuffers(0, num_buffers, buffers.data(), offsets.data());
        });
    } else {
        scheduler.Record([num_buffers = v.num_buffers, buffers = v.buffers, offsets = v.offsets,
                          sizes = v.sizes, strides = v.strides](vk::CommandBuffer cmdbuf) {
            cmdbuf.bindVertexBuffers2(0, num_buffers, buffers.data(), offsets.data(), sizes.data(),
                                      strides.data());
        });
    }
}

void Rasterizer::RecordIndexBinding() {
    // A null handle (no index data mapped) is legal with maintenance6; with maintenance5 the
    // size keeps reads inside the guest's index buffer.
    scheduler.Record([handle = last_index.handle, offset = last_index.offset,
                      size = last_index.size, type = last_index.type,
                      maintenance5 = instance.IsMaintenance5Supported()](vk::CommandBuffer cmdbuf) {
        if (maintenance5) {
            cmdbuf.bindIndexBuffer2KHR(handle, offset, size, type);
        } else {
            cmdbuf.bindIndexBuffer(handle, offset, type);
        }
    });
}

void Rasterizer::Draw(bool is_indexed, u32 index_offset) {
    RENDERER_TRACE;
    ++stat_draws;
    occlusion_a.draws += occlusion_a.open ? 1 : 0;
    const bool redirected = OcclusionRedirectBegin();
    SCOPE_EXIT {
        if (redirected) {
            OcclusionRedirectEnd();
        }
    };
    if (occlusion_a.open) {
        // Diagnostics (one-frame hidden blips): the depth this box is drawn against.
        const auto& regs = liverpool->regs;
        occlusion_a.depth_address = regs.depth_buffer.DepthAddress();
        occlusion_a.depth_extent = liverpool->last_db_extent.raw;
        occlusion_a.depth_control = std::bit_cast<u32>(regs.depth_control);
    }

    // Threaded renderer: stage A selects the pipeline and queues the rest of the draw. Draws
    // FilterDraw handles itself (fast clear elimination, resolve, depth copy, skips) run here.
    if (UseDrawPipe() && FilterDrawPasses()) {
        ABeginDraw();
        // A new scene pipeline may compile in the background: the draw is skipped until then.
        const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline({}, true);
        ALap(ASection::Pipeline);
        if (pipeline && !DrawWritesNothing(pipeline)) {
            QueueDraw(PacketKind::Draw, pipeline, is_indexed, index_offset, {});
        }
        AEndDraw();
        return;
    }
    if (UseDrawPipe() && !DrainForSpecialPackets()) {
        // Draws FilterDraw handles itself (fast clear elimination, resolve, depth copy, skips)
        // run on the draw recorder, in order: a packet without a pipeline.
        QueueDraw(PacketKind::Draw, nullptr, is_indexed, index_offset, {});
        return;
    }
    if (UseDrawPipe() && Regs().primitive_type == AmdGpu::PrimitiveType::None) {
        // FilterDraw only skips these (fast clear, resolve and copy modes are checked first by
        // FilterDrawPasses): no need to wait for the draw recorder.
        using Mode = AmdGpu::ColorControl::OperationMode;
        const auto mode = Regs().color_control.mode;
        if (mode != Mode::EliminateFastClear && mode != Mode::FmaskDecompress &&
            mode != Mode::Resolve) {
            return;
        }
    }
    DrainDrawPipe();
    scheduler.PopPendingOperations();

    if (!FilterDraw()) {
        return;
    }

    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline();
    if (!pipeline) {
        return;
    }
    DrawRecord(pipeline, is_indexed, index_offset);
}

void Rasterizer::DrawRecord(const GraphicsPipeline* pipeline, bool is_indexed, u32 index_offset) {
    const u64 mirror_flushes = velocity_mirror.Flushes();
    mirror_rebind = false;
    occlusion_b.draws_seen += occlusion_b.open ? 1 : 0;
    if (DrawPipe::OnStageB()) {
        scheduler.PopPendingOperations();
        if (target_memo_enabled.load(std::memory_order_relaxed)) {
            scheduler.KickRecording(); // nobody holds the command buffer here
        }
    }
    ProfileLap(Section::Pending);
    const auto& regs = Regs();

    const auto stage_hash = [pipeline](Shader::SwStage stage) -> u64 {
        const auto* info = pipeline->GetStages()[static_cast<u32>(stage)];
        return info ? info->pgm_hash : 0;
    };
    Common::Crumbs::Add("draw: render targets (vs, ps)", stage_hash(Shader::SwStage::Vertex),
                        stage_hash(Shader::SwStage::Fragment));
    PrepareRenderState(pipeline);
    ProfileLap(Section::Targets);
    Common::Crumbs::Add("draw: textures and buffers");
    if (!BindResources(pipeline)) {
        return;
    }
    TemporalDlssDraw(pipeline, false);
    if (const auto source = temporal_dlss.TakeDrawReplacement()) {
        CopyInsteadOfDraw(*source);
        ResetBindings(false);
        return;
    }
    ProfileLap(Section::Dlss);
    Common::Crumbs::Add("draw: begin rendering");
    auto state = BeginRendering(pipeline);
    ProfileLap(Section::BeginRendering);

    Common::Crumbs::Add("draw: vertex and index buffers");
    BindVertexBuffers(pipeline);
    ProfileLap(Section::VertexBuffers);
    if (is_indexed) {
        BindIndexBuffer(index_offset);
    }
    ProfileLap(Section::Vertex);

    if (needs_barrier) {
        runtime.FlushBarriers();
    }
    ProfileLap(Section::Barriers);

    const auto& vs_info = pipeline->GetStage(Shader::SwStage::Vertex);
    const auto& fetch_shader = pipeline->GetFetchShader();
    const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs_info, fetch_shader);
    if (pipeline->GetGraphicsKey().motion_vectors) {
        PrepareObjectMotion(pipeline, is_indexed, index_offset, vertex_offset, instance_offset);
    }
    ProfileLap(Section::Motion);

    Common::Crumbs::Add("draw: descriptors");
    pipeline->BindResources(set_writes, push_data);
    ProfileLap(Section::Descriptors);
    UpdateDynamicState(pipeline, is_indexed);
    MarkPass(pipeline, state);
    scheduler.BeginRendering(state);
    ProfileLap(Section::Dynamic);

    const auto pipeline_handle = pipeline->Handle();
    scheduler.Record([pipeline_handle](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline_handle);
    });

    Common::Crumbs::Add("draw: draw (indices, instances)", regs.num_indices,
                        regs.num_instances.NumInstances());
    const u32 num_indices = regs.num_indices;
    const u32 num_instances = regs.num_instances.NumInstances();
    const auto record_draw = [=](vk::CommandBuffer cmdbuf) {
        if (is_indexed) {
            cmdbuf.drawIndexed(num_indices, num_instances, 0, s32(vertex_offset), instance_offset);
        } else {
            cmdbuf.draw(num_indices, num_instances, vertex_offset, instance_offset);
        }
    };
    // The display copy replay records directly (CommandBuffer() waits for the recording
    // thread first); the velocity mirror goes through Record().
    const auto draw = [&] { record_draw(scheduler.CommandBuffer()); };
    ReplayVelocityMirror(pipeline, state, is_indexed, [&] { scheduler.Record(record_draw); });
    RebindAfterMirror(pipeline, state, is_indexed, mirror_flushes);
    if (occlusion_b.open) {
        occlusion_b.depth_view = reinterpret_cast<u64>(
            static_cast<VkImageView>(state.depth_stencil_attachment.image_view));
        occlusion_b.render_size = u32(state.width) | u32(state.height) << 16;
        occlusion_b.depth_clear = state.depth_stencil_attachment.depth_clear;
    }
    const u32 query = occlusion_b.open ? OcclusionQueryBegin() : NoOcclusionQuery;
    // Better AO: the game's own AO draws whose result VBAO replaces are left out (their GPU time
    // back); the bookkeeping above still ran, as for any draw.
    // Better reflections: the game's trace is written by its own pass instead.
    if (!BetterSsrReplacesDraw(pipeline) && !(vbao && BetterAoLeavesOutDraw(pipeline))) {
        scheduler.Record(record_draw);
    }
    OcclusionQueryEnd(query);
    ReplayDisplayCopy(pipeline, state, draw);
    if (vbao) {
        BetterAoAfterDraw(pipeline);
    }
    if (ssr) {
        BetterSsrAfterDraw(pipeline);
    }
    DebugState.IncDrawCall();
    draw_jitter = {};
    ProfileLap(Section::Record);

    ResetBindings(false);
    ProfileLap(Section::Reset);
}

void Rasterizer::DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 stride,
                              u32 max_count, VAddr count_address, u16 vertex_sgpr_offset,
                              u16 instance_sgpr_offset) {
    RENDERER_TRACE;
    ++stat_draws;
    occlusion_a.draws += occlusion_a.open ? 1 : 0;

    const DrawIndirectParams params = {
        .vertex_sgpr_offset = vertex_sgpr_offset,
        .instance_sgpr_offset = instance_sgpr_offset,
        .is_indirect = true,
    };
    const IndirectArgs args{
        .address = arg_address,
        .count_address = count_address,
        .offset = offset,
        .stride = stride,
        .max_count = max_count,
    };
    if (UseDrawPipe() && FilterDrawPasses()) {
        const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline(params, true);
        if (pipeline) {
            QueueDraw(PacketKind::DrawIndirect, pipeline, is_indexed, 0, args);
        }
        return;
    }
    DrainDrawPipe();
    scheduler.PopPendingOperations();

    if (!FilterDraw()) {
        return;
    }

    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline(params);
    if (!pipeline) {
        return;
    }
    DrawIndirectRecord(pipeline, is_indexed, args);
}

void Rasterizer::DrawIndirectRecord(const GraphicsPipeline* pipeline, bool is_indexed,
                                    const IndirectArgs& args) {
    const u64 mirror_flushes = velocity_mirror.Flushes();
    mirror_rebind = false;
    occlusion_b.dropped |= occlusion_b.open; // not counted: the test answers "visible"
    if (DrawPipe::OnStageB()) {
        scheduler.PopPendingOperations();
        if (target_memo_enabled.load(std::memory_order_relaxed)) {
            scheduler.KickRecording(); // nobody holds the command buffer here
        }
    }
    Common::Crumbs::Add("indirect draw (vs, args)",
                        pipeline->GetStage(Shader::SwStage::Vertex).pgm_hash, args.address);
    const VAddr arg_address = args.address;
    const VAddr count_address = args.count_address;
    const u32 offset = args.offset;
    const u32 stride = args.stride;
    const u32 max_count = args.max_count;

    PrepareRenderState(pipeline);
    if (!BindResources(pipeline)) {
        return;
    }
    TemporalDlssDraw(pipeline, true);
    const auto state = BeginRendering(pipeline);

    BindVertexBuffers(pipeline);
    if (is_indexed) {
        BindIndexBuffer(0, true);
    }

    const auto size = stride * max_count;
    const auto [buffer, base] = buffer_cache.ObtainBuffer(arg_address + offset, size);
    needs_barrier |= runtime.IsBufferAccessed(buffer, base, size);
    bound_buffers.emplace_back(buffer, base, size, vk::AccessFlagBits2::eIndirectCommandRead);

    const VideoCore::Buffer* count_buffer;
    u64 count_offset;
    if (count_address != 0) {
        std::tie(count_buffer, count_offset) = buffer_cache.ObtainBuffer(count_address, 4);
        needs_barrier |= runtime.IsBufferAccessed(count_buffer, count_offset, 4);
        bound_buffers.emplace_back(count_buffer, count_offset, 4,
                                   vk::AccessFlagBits2::eIndirectCommandRead);
    }

    if (needs_barrier) {
        runtime.FlushBarriers();
    }

    pipeline->BindResources(set_writes, push_data);
    UpdateDynamicState(pipeline, is_indexed);
    MarkPass(pipeline, state);
    scheduler.BeginRendering(state);

    const auto pipeline_handle = pipeline->Handle();
    scheduler.Record([pipeline_handle](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline_handle);
    });

    const auto buffer_handle = buffer->Handle();
    const auto count_handle = count_address != 0 ? count_buffer->Handle() : vk::Buffer{};
    const auto record_draw = [=](vk::CommandBuffer cmdbuf) {
        if (is_indexed) {
            if (count_address != 0) {
                cmdbuf.drawIndexedIndirectCount(buffer_handle, base, count_handle, count_offset,
                                                max_count, stride);
            } else {
                cmdbuf.drawIndexedIndirect(buffer_handle, base, max_count, stride);
            }
        } else {
            if (count_address != 0) {
                cmdbuf.drawIndirectCount(buffer_handle, base, count_handle, count_offset, max_count,
                                         stride);
            } else {
                cmdbuf.drawIndirect(buffer_handle, base, max_count, stride);
            }
        }
    };
    ASSERT(!is_indexed || sizeof(VkDrawIndexedIndirectCommand) == stride);
    ASSERT(is_indexed || sizeof(VkDrawIndirectCommand) == stride);
    const auto draw = [&] { record_draw(scheduler.CommandBuffer()); };
    ReplayVelocityMirror(pipeline, state, is_indexed, [&] { scheduler.Record(record_draw); });
    RebindAfterMirror(pipeline, state, is_indexed, mirror_flushes);
    scheduler.Record(record_draw);
    ReplayDisplayCopy(pipeline, state, draw);
    DebugState.IncDrawCall();
    draw_jitter = {};

    ResetBindings(false);
}

void Rasterizer::DispatchDirect() {
    RENDERER_TRACE;

    // Only dispatches of the graphics queue are queued: compute queues run with stage B idle.
    if (UseDrawPipe() && liverpool->CurrentQueue() == AmdGpu::Liverpool::GfxQueueId) {
        const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
        if (pipeline) {
            QueueDraw(PacketKind::Dispatch, pipeline, false, 0, {});
        }
        return;
    }
    DrainDrawPipe();
    scheduler.PopPendingOperations();

    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }
    DispatchRecord(pipeline);
}

void Rasterizer::DispatchRecord(const ComputePipeline* pipeline) {
    if (DrawPipe::OnStageB()) {
        scheduler.PopPendingOperations();
        if (target_memo_enabled.load(std::memory_order_relaxed)) {
            scheduler.KickRecording(); // nobody holds the command buffer here
        }
    }
    const auto& cs_program = CsRegs();

    const auto& cs = pipeline->GetStage(Shader::SwStage::Compute);
    Common::Crumbs::Add("dispatch (cs)", cs.pgm_hash);
    // Diagnostics: a dispatch that held up the draw recorder over 2 ms (first 64): its shader,
    // whether the HLE took it, and the buffers it binds.
    const bool timed = diagnostics.load(std::memory_order_relaxed);
    const auto dispatch_start =
        timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto log_slow = [&](const char* how) {
        static u32 logged = 0;
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - dispatch_start)
                              .count();
        if (ms < 2.0 || logged >= 64) {
            return;
        }
        ++logged;
        std::string buffers;
        for (const auto& desc : cs.buffers) {
            if (desc.IsSpecial()) {
                continue;
            }
            const auto vsharp = desc.GetSharp(cs);
            buffers +=
                fmt::format("{}{:#x} {} KiB{}", buffers.empty() ? "" : ", ", vsharp.base_address,
                            vsharp.GetSize() >> 10, desc.is_written ? " written" : "");
        }
        LOG_INFO(Render_Vulkan,
                 "Slow dispatch: shader {:#x} took {:.1f} ms on the draw recorder ({}) | {}x{}x{} "
                 "groups, {} images | buffers: {}",
                 cs.pgm_hash, ms, how, cs_program.dim_x, cs_program.dim_y, cs_program.dim_z,
                 cs.images.size(), buffers.empty() ? std::string{"-"} : buffers);
    };
    if (ExecuteShaderHLE(cs, Regs(), cs_program, *this)) {
        if (timed) {
            log_slow("done by the HLE");
        }
        return;
    }

    if (!BindResources(pipeline)) {
        return;
    }
    if (timed) {
        log_slow("binding its resources");
    }

    if (needs_barrier) {
        runtime.FlushBarriers();
    }

    scheduler.EndRendering();
    MarkDispatch(pipeline);
    pipeline->BindResources(set_writes, push_data);

    scheduler.Record([handle = pipeline->Handle(), x = cs_program.dim_x, y = cs_program.dim_y,
                      z = cs_program.dim_z](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, handle);
        cmdbuf.dispatch(x, y, z);
    });
    DebugState.IncDispatch();

    ResetBindings(true);
}

void Rasterizer::DispatchIndirect(VAddr address, u32 offset, u32 size) {
    RENDERER_TRACE;

    const IndirectArgs args{.address = address, .offset = offset, .stride = size};
    if (UseDrawPipe() && liverpool->CurrentQueue() == AmdGpu::Liverpool::GfxQueueId) {
        const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
        if (pipeline) {
            QueueDraw(PacketKind::DispatchIndirect, pipeline, false, 0, args);
        }
        return;
    }
    DrainDrawPipe();
    scheduler.PopPendingOperations();

    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }
    DispatchIndirectRecord(pipeline, args);
}

void Rasterizer::DispatchIndirectRecord(const ComputePipeline* pipeline, const IndirectArgs& args) {
    if (DrawPipe::OnStageB()) {
        scheduler.PopPendingOperations();
        if (target_memo_enabled.load(std::memory_order_relaxed)) {
            scheduler.KickRecording(); // nobody holds the command buffer here
        }
    }
    const VAddr address = args.address;
    const u32 offset = args.offset;
    const u32 size = args.stride;

    if (!BindResources(pipeline)) {
        return;
    }

    const auto [buffer, base] = buffer_cache.ObtainBuffer(address + offset, size);
    needs_barrier |= runtime.IsBufferAccessed(buffer, base, size);
    bound_buffers.emplace_back(buffer, base, size, vk::AccessFlagBits2::eIndirectCommandRead);

    if (needs_barrier) {
        runtime.FlushBarriers();
    }

    scheduler.EndRendering();
    MarkDispatch(pipeline);
    pipeline->BindResources(set_writes, push_data);

    scheduler.Record([handle = pipeline->Handle(), buffer_handle = buffer->Handle(),
                      base](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, handle);
        cmdbuf.dispatchIndirect(buffer_handle, base);
    });
    DebugState.IncDispatch();

    ResetBindings(true);
}

u64 Rasterizer::Flush() {
    DrainDrawPipe();
    const u64 current_tick = scheduler.CurrentTick();
    SubmitInfo info{};
    scheduler.Flush(info);
    return current_tick;
}

void Rasterizer::OnSubmissionDecoded() {
    if (!buffer_cache.EarlyWriteback()) {
        return;
    }
    // Only while the draw recorder has nothing queued: draining it here made the GPU command
    // thread wait for every queued draw mid-frame, so the two threads stopped overlapping (frames
    // ~10 ms longer for as long as some memory stayed hot, seconds of judder). With draws queued,
    // the early flush waits for a later submission and the write-backs for the frame's end.
    if (draw_pipe && !draw_pipe->Idle()) {
        return;
    }
    if (buffer_cache.TakeWritebackFlush()) {
        Flush();
    }
    const auto finished = buffer_cache.TakeFinishedWritebacks();
    if (finished.empty()) {
        return;
    }
    DrainDrawPipe();
    for (const auto& writeback : finished) {
        buffer_cache.WriteBack(writeback);
    }
}

void Rasterizer::Finish() {
    DrainDrawPipe();
    scheduler.Finish();
}

void Rasterizer::OnSubmit() {
    ++stat_frames;
    occlusion_frame.fetch_add(1, std::memory_order_relaxed);
    DrainDrawPipe();
    // Early write-back: the frame's finished copies, now that the draw recorder is idle anyway.
    if (buffer_cache.EarlyWriteback()) {
        for (const auto& writeback : buffer_cache.TakeFinishedWritebacks()) {
            buffer_cache.WriteBack(writeback);
        }
    }
    // F1's Object motion vectors: the velocity draws are replayed only while an upscaler uses them.
    velocity_mirror.SetEnabled(temporal_dlss.ObjectMotionRequested());
    if (draw_pipe) {
        // The setting applies from the next submission (the pipe is idle here).
        pipe_enabled = EmulatorSettings.IsThreadedRenderer();
        target_memo_enabled.store(pipe_enabled, std::memory_order_relaxed);
        const bool diag = pipe_enabled && EmulatorSettings.IsThreadedRendererDiagnostics();
        if (diag && !diagnostics.load(std::memory_order_relaxed)) {
            // Start a fresh window (counters ran while off).
            pipe_stats_time = {};
            hitch_last = {};
            hitch_frames = 0;
        }
        diagnostics.store(diag, std::memory_order_relaxed);
        Common::GuestStats::diagnostics.store(diag, std::memory_order_relaxed);
        // The toggles under the threaded renderer (off with it: original behaviour).
        namespace BB = Common::BbTuning;
        BB::threaded.store(pipe_enabled, std::memory_order_relaxed);
        BB::streaming.store(pipe_enabled && EmulatorSettings.IsBbStreaming(),
                            std::memory_order_relaxed);
        BB::merged_copies.store(pipe_enabled && EmulatorSettings.IsBbMergedCopies(),
                                std::memory_order_relaxed);
        BB::merged_no_stream_sources.store(EmulatorSettings.IsBbMergedNoStreamSources(),
                                           std::memory_order_relaxed);
        BB::merged_large_batches.store(EmulatorSettings.IsBbMergedLargeBatches(),
                                       std::memory_order_relaxed);
        BB::readback_queue.store(pipe_enabled && EmulatorSettings.IsBbReadbackQueue(),
                                 std::memory_order_relaxed);
        buffer_cache.SetEarlyWriteback(pipe_enabled && EmulatorSettings.IsBbEarlyWriteback() &&
                                       EmulatorSettings.GetReadbacksMode() !=
                                           GpuReadbacksMode::Disabled);
        scheduler.SetGpuTiming(diag);
        GpuProfiler::OnFrame(instance, scheduler,
                             pipe_enabled && EmulatorSettings.IsThreadedRendererGpuProfile());
        if (diag) {
            const auto start = std::chrono::steady_clock::now();
            LogDrawPipeStats();
            diag_log_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::steady_clock::now() - start)
                                      .count(),
                                  std::memory_order_relaxed);
        }
    }
    auto lap = std::chrono::steady_clock::now();
    const auto time_step = [&lap](s64& out) {
        const auto now = std::chrono::steady_clock::now();
        out = std::chrono::duration_cast<std::chrono::nanoseconds>(now - lap).count();
        lap = now;
    };
    buffer_cache.TickFrame();
    time_step(housekeeping_ns[0]);
    texture_cache.ProcessDownloadImages();
    time_step(housekeeping_ns[1]);
    texture_cache.RunGarbageCollector();
    time_step(housekeeping_ns[2]);
    if (diagnostics.load(std::memory_order_relaxed)) {
        const auto start = std::chrono::steady_clock::now();
        LogVramReport();
        buffer_cache.LogReadbackReport();
        diag_log_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::steady_clock::now() - start)
                                  .count(),
                              std::memory_order_relaxed);
        lap = std::chrono::steady_clock::now();
    }
    runtime.TickFrame();
    time_step(housekeeping_ns[3]);
    for (size_t i = 0; i < housekeeping_ns.size(); ++i) {
        housekeeping_total[i].fetch_add(static_cast<u64>(housekeeping_ns[i]),
                                        std::memory_order_relaxed);
    }
}

void Rasterizer::OnGameFrame() {
    pipeline_cache.OnGameFrame();
    texture_cache.OnGameFrame();
    if (draw_pipe && diagnostics.load(std::memory_order_relaxed)) {
        ReportHitch();
    } else {
        hitch_last = {};
        hitch_frames = 0;
    }
}

void Rasterizer::LogVramReport() {
    const auto now = std::chrono::steady_clock::now();
    if (now - vram_report_time < std::chrono::seconds(10)) {
        return;
    }
    vram_report_time = now;
    const auto mib = [](u64 bytes) { return bytes >> 20; };

    const auto heaps = instance.GetMemoryHeapReport();
    const u64 buffers = buffer_cache.ResidentBytes();
    const auto images = texture_cache.TakeMemoryReport();

    // Memory the allocator holds in device-local heaps, and how much of it is in use.
    // vmaGetHeapBudgets is cheap (running totals); vmaCalculateStatistics walks every allocation.
    std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> vma_budgets{};
    vmaGetHeapBudgets(instance.GetAllocator(), vma_budgets.data());
    u64 vma_blocks = 0, vma_used = 0;
    const auto& memory_props = instance.GetMemoryProperties();
    for (u32 i = 0; i < memory_props.memoryHeapCount; ++i) {
        if (memory_props.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
            vma_blocks += vma_budgets[i].statistics.blockBytes;
            vma_used += vma_budgets[i].statistics.allocationBytes;
        }
    }
    const u64 known = buffers + vma_blocks;
    const s64 other = static_cast<s64>(heaps.local_usage) - static_cast<s64>(known);
    const u64 image_bytes = images.clean.bytes + images.written.bytes + images.stuck.bytes;

    LOG_INFO(Render,
             "VRAM: {} of {} MiB budget used, {} MiB in system memory | guest buffers {} MiB "
             "(never released) | allocator {} MiB held, {} MiB in use | outside both {} MiB",
             mib(heaps.local_usage), mib(heaps.local_budget), mib(heaps.host_usage), mib(buffers),
             mib(vma_blocks), mib(vma_used), other / (1 << 20));
    LOG_INFO(Render,
             "VRAM images: {} MiB in {} images ({} slots) | clean {} MiB / {} (old {} MiB / {}) | "
             "GPU-written {} MiB / {} (old {} MiB / {}) | GPU-written tiled, never freed {} MiB / "
             "{} (old {} MiB / {}) | extra backings {} MiB",
             mib(image_bytes), images.clean.count + images.written.count + images.stuck.count,
             images.allocated_images, mib(images.clean.bytes), images.clean.count,
             mib(images.clean.stale_bytes), images.clean.stale_count, mib(images.written.bytes),
             images.written.count, mib(images.written.stale_bytes), images.written.stale_count,
             mib(images.stuck.bytes), images.stuck.count, mib(images.stuck.stale_bytes),
             images.stuck.stale_count, mib(images.extra_backing_bytes));
    LOG_INFO(Render,
             "VRAM collector: {} runs, {} images freed ({} written back), skipped {} tiled and {} "
             "GPU-written | measures {} MiB, starts at {}, pressure {}, critical {} | {} area "
             "loads (freeing ahead)",
             images.gc_runs, images.evicted, images.written_back, images.skipped_stuck,
             images.skipped_written, mib(images.gc_used), mib(images.trigger), mib(images.pressure),
             mib(images.critical), images.loads);
    LOG_INFO(Render,
             "Spread loading: {} loads spread over frames, {} uploads put off to a later frame, "
             "{} new textures shown black meanwhile ({} MiB)",
             images.spread_loads, images.spread_deferred, images.spread_zero_filled,
             mib(images.spread_zero_bytes));
    {
        namespace GS = Common::GuestStats;
        const auto back = [](size_t i) {
            return GS::evicted_returned_bytes[i].load(std::memory_order_relaxed) >> 20;
        };
        const u64 evicted = GS::evicted_bytes_total.load(std::memory_order_relaxed);
        if (evicted != 0) {
            LOG_INFO(Render,
                     "Buffer eviction: {} MiB evicted so far; made resident again after under 10 "
                     "s {} MiB, 10-60 s {} MiB, 1-5 min {} MiB, later {} MiB",
                     mib(evicted), back(0), back(1), back(2), back(3));
        }
    }
    if (EmulatorSettings.IsBbLowerDetailFirst()) {
        namespace GS = Common::GuestStats;
        const u64 restored = GS::detail_restored.load(std::memory_order_relaxed);
        const u64 soft_ns = GS::detail_soft_ns.load(std::memory_order_relaxed);
        LOG_INFO(Render,
                 "Lower detail first: {} new textures sampled softer at first ({} MiB of largest "
                 "levels put off), {} got their largest level later ({} at once for another "
                 "use), softer for {:.0f} ms on average, longest {:.0f} ms",
                 GS::detail_deferred.load(std::memory_order_relaxed),
                 mib(GS::detail_deferred_bytes.load(std::memory_order_relaxed)), restored,
                 GS::detail_forced.load(std::memory_order_relaxed),
                 restored ? soft_ns / 1e6 / static_cast<double>(restored) : 0.0,
                 GS::detail_soft_max_ns.load(std::memory_order_relaxed) / 1e6);
    }
}

void Rasterizer::ReportHitch() {
    namespace GS = Common::GuestStats;
    HitchSnapshot now{
        .time = std::chrono::steady_clock::now(),
        .busy = draw_pipe->BusyTime(),
        .drain = draw_pipe->drain_time,
        .packets = draw_pipe->packets,
        .gpu_wait_ns = GS::gpu_wait_ns.load(std::memory_order_relaxed),
        .gpu_waits = GS::gpu_waits.load(std::memory_order_relaxed),
        .pipelines = GS::pipelines_compiled.load(std::memory_order_relaxed),
        .background_pipelines = GS::background_pipelines_started.load(std::memory_order_relaxed),
        .background_shaders = GS::background_shaders_started.load(std::memory_order_relaxed),
        .background_skips = GS::background_skipped_draws.load(std::memory_order_relaxed),
        .background_wait_ns = GS::background_wait_ns.load(std::memory_order_relaxed),
        .arena_binds = GS::arena_binds.load(std::memory_order_relaxed),
        .protect_ns = GS::protect_ns.load(std::memory_order_relaxed),
        .protect_calls = GS::protect_calls.load(std::memory_order_relaxed),
        .faults = write_faults.load(std::memory_order_relaxed),
        .mutex_waits = GS::mutex_waits.load(std::memory_order_relaxed),
        .sleeps = GS::sleeps.load(std::memory_order_relaxed),
        .sleep_ns = GS::sleep_actual_ns.load(std::memory_order_relaxed),
        .uploads = GS::image_uploads.load(std::memory_order_relaxed),
        .upload_bytes = GS::image_upload_bytes.load(std::memory_order_relaxed),
        .upload_ns = GS::image_upload_ns.load(std::memory_order_relaxed),
        .image_copy_ns = GS::image_copy_ns.load(std::memory_order_relaxed),
        .buffer_bytes = GS::buffer_upload_bytes.load(std::memory_order_relaxed),
        .buffer_ns = GS::buffer_upload_ns.load(std::memory_order_relaxed),
        .resident_ns = GS::resident_ns.load(std::memory_order_relaxed),
        .staging_count = GS::staging_allocations.load(std::memory_order_relaxed),
        .staging_bytes = GS::staging_allocation_bytes.load(std::memory_order_relaxed),
        .staging_ns = GS::staging_allocation_ns.load(std::memory_order_relaxed),
        .async_copies = GS::async_copies.load(std::memory_order_relaxed),
        .async_wait_ns = GS::async_copy_wait_ns.load(std::memory_order_relaxed),
        .file_opens = GS::file_opens.load(std::memory_order_relaxed),
        .resident_bytes = buffer_cache.ResidentBytes(),
        .diag_ns = diag_log_ns.load(std::memory_order_relaxed),
    };
    now.present_acquire_ns = GS::present_acquire_ns.load(std::memory_order_relaxed);
    now.present_queue_ns = GS::present_queue_ns.load(std::memory_order_relaxed);
    now.present_frame_wait_ns = GS::present_frame_wait_ns.load(std::memory_order_relaxed);
    now.presents = GS::presents.load(std::memory_order_relaxed);
    now.queue_ops_behind_present = GS::queue_ops_behind_present.load(std::memory_order_relaxed);
    now.skipped_bytes = GS::image_skipped_bytes.load(std::memory_order_relaxed);
    now.direct_bytes = GS::image_direct_bytes.load(std::memory_order_relaxed);
    now.allocs = GS::image_allocs.load(std::memory_order_relaxed);
    now.alloc_ns = GS::image_alloc_ns.load(std::memory_order_relaxed);
    now.slow_allocs = GS::image_slow_allocs.load(std::memory_order_relaxed);
    now.slow_alloc_ns = GS::image_slow_alloc_ns.load(std::memory_order_relaxed);
    now.detail_deferred = GS::detail_deferred.load(std::memory_order_relaxed);
    now.detail_deferred_bytes = GS::detail_deferred_bytes.load(std::memory_order_relaxed);
    now.detail_restored = GS::detail_restored.load(std::memory_order_relaxed);
    now.detail_restored_bytes = GS::detail_restored_bytes.load(std::memory_order_relaxed);
    now.detail_forced = GS::detail_forced.load(std::memory_order_relaxed);
    now.cmd_protect_calls = GS::cmd_protect_calls.load(std::memory_order_relaxed);
    now.cmd_protect_ns = GS::cmd_protect_ns.load(std::memory_order_relaxed);
    now.cmd_resident_ns = GS::cmd_resident_ns.load(std::memory_order_relaxed);
    now.cmd_buffer_upload_bytes = GS::cmd_buffer_upload_bytes.load(std::memory_order_relaxed);
    now.cmd_buffer_upload_ns = GS::cmd_buffer_upload_ns.load(std::memory_order_relaxed);
    now.a_hitch_draws = a_hitch_draws.load(std::memory_order_relaxed);
    for (size_t i = 0; i < now.protect_ns_by.size(); ++i) {
        now.protect_ns_by[i] = GS::recorder_protect_ns_by[i].load(std::memory_order_relaxed);
        now.protect_calls_by[i] = GS::recorder_protect_calls_by[i].load(std::memory_order_relaxed);
    }
    now.stream_copies = GS::stream_copies.load(std::memory_order_relaxed);
    now.stream_copy_bytes = GS::stream_copy_bytes.load(std::memory_order_relaxed);
    now.buffer_syncs = GS::buffer_syncs.load(std::memory_order_relaxed);
    now.buffer_sync_ranges = GS::buffer_sync_ranges.load(std::memory_order_relaxed);
    now.buffer_sync_inline_bytes = GS::buffer_sync_inline_bytes.load(std::memory_order_relaxed);
    now.buffer_mem_allocs = GS::buffer_mem_allocs.load(std::memory_order_relaxed);
    now.buffer_mem_alloc_ns = GS::buffer_mem_alloc_ns.load(std::memory_order_relaxed);
    now.ce_dumps = GS::ce_dumps.load(std::memory_order_relaxed);
    now.ce_dump_drains = GS::ce_dump_drains.load(std::memory_order_relaxed);
    now.ce_dump_drain_ns = GS::ce_dump_drain_ns.load(std::memory_order_relaxed);
    now.ce_dump_write_ns = GS::ce_dump_write_ns.load(std::memory_order_relaxed);
    now.ce_outside_ns = GS::ce_outside_ns.load(std::memory_order_relaxed);
    now.ce_wait_reads = GS::ce_wait_reads.load(std::memory_order_relaxed);
    now.ce_wait_writes = GS::ce_wait_writes.load(std::memory_order_relaxed);
    now.ce_wait_overflow = GS::ce_wait_overflow.load(std::memory_order_relaxed);
    now.ce_huge_write_skips = GS::ce_huge_write_skips.load(std::memory_order_relaxed);
    now.flips = GS::flips.load(std::memory_order_relaxed);
    now.flip_drain_ns = GS::flip_drain_ns.load(std::memory_order_relaxed);
    now.flip_signal_ns = GS::flip_signal_ns.load(std::memory_order_relaxed);
    now.evicted_refills = GS::evicted_refills.load(std::memory_order_relaxed);
    now.evicted_refill_bytes = GS::evicted_refill_bytes.load(std::memory_order_relaxed);
    now.helper_protect_calls = GS::helper_protect_calls.load(std::memory_order_relaxed);
    now.helper_protect_ns = GS::helper_protect_ns.load(std::memory_order_relaxed);
    now.evicted_bytes_total = GS::evicted_bytes_total.load(std::memory_order_relaxed);
    for (size_t i = 0; i < now.evicted_returned.size(); ++i) {
        now.evicted_returned[i] = GS::evicted_returned_bytes[i].load(std::memory_order_relaxed);
    }
    now.cmd_drain_ns = GS::cmd_drain_ns.load(std::memory_order_relaxed);
    now.cmd_compute_ns = GS::cmd_compute_ns.load(std::memory_order_relaxed);
    now.cmd_compute_packets = GS::cmd_compute_packets.load(std::memory_order_relaxed);
    for (size_t i = 0; i < now.cmd_opcode_ns.size(); ++i) {
        now.cmd_opcode_ns[i] = GS::cmd_opcode_ns[i].load(std::memory_order_relaxed);
        now.cmd_opcode_count[i] = GS::cmd_opcode_count[i].load(std::memory_order_relaxed);
    }
    static_assert(std::tuple_size_v<decltype(now.a_hitch_ns)> == ASectionNames.size());
    for (size_t i = 0; i < now.a_hitch_ns.size(); ++i) {
        now.a_hitch_ns[i] = a_hitch_ns[i].load(std::memory_order_relaxed);
    }
    for (size_t i = 0; i < now.upload_paths.size(); ++i) {
        now.upload_paths[i] = GS::upload_path_bytes[i].load(std::memory_order_relaxed);
    }
    for (size_t i = 0; i < now.upload_packets.size(); ++i) {
        now.upload_packets[i] = GS::upload_packet_bytes[i].load(std::memory_order_relaxed);
    }
    for (size_t i = 0; i < now.new_storage.size(); ++i) {
        now.new_storage[i] = GS::new_storage_bytes[i].load(std::memory_order_relaxed);
    }
    now.new_storage_zero = GS::new_storage_zero_bytes.load(std::memory_order_relaxed);
    now.new_storage_cleared = GS::new_storage_cleared_bytes.load(std::memory_order_relaxed);
    for (size_t i = 0; i < now.upload_ages.size(); ++i) {
        now.upload_ages[i] = GS::upload_age_bytes[i].load(std::memory_order_relaxed);
    }
    now.gpu_busy_ns = GS::gpu_timeline_busy_ns.load(std::memory_order_relaxed);
    now.gpu_idle_ns = GS::gpu_timeline_idle_ns.load(std::memory_order_relaxed);
    now.gpu_submits = GS::gpu_timeline_submits.load(std::memory_order_relaxed);
    now.queue_submit_ns = GS::queue_submit_ns.load(std::memory_order_relaxed);
    for (size_t i = 0; i < now.states.size(); ++i) {
        now.states[i] = state_samples[i].load(std::memory_order_relaxed);
    }
    for (size_t i = 0; i < now.housekeeping.size(); ++i) {
        now.housekeeping[i] = housekeeping_total[i].load(std::memory_order_relaxed);
    }
    static_assert(std::tuple_size_v<decltype(now.sections)> == SectionNames.size());
    for (size_t i = 0; i < now.sections.size(); ++i) {
        now.sections[i] = profile_ns[i].load(std::memory_order_relaxed);
    }
    for (size_t i = 0; i < now.packet_kinds.size(); ++i) {
        now.packet_kinds[i] = packet_kind_ns[i].load(std::memory_order_relaxed);
    }
    for (size_t i = 0; i < now.details.size(); ++i) {
        now.details[i] = GS::detail_ns[i].load(std::memory_order_relaxed);
    }
    now.recorder_protect_calls = GS::recorder_protect_calls.load(std::memory_order_relaxed);
    now.cleanup = VideoCore::TextureCache::ReadCleanupCounters();
    GS::detail_timing.store(HitchProfile() && diagnostics.load(std::memory_order_relaxed),
                            std::memory_order_relaxed);
    const auto last = std::exchange(hitch_last, now);
    const u64 slowest_ns = GS::slowest_wait_ns.exchange(0, std::memory_order_relaxed);
    if (last.time == std::chrono::steady_clock::time_point{}) {
        return;
    }
    // Counters the 10 s statistics reset count from zero again.
    const auto delta = [](u64 a, u64 b) { return a >= b ? a - b : a; };
    const auto ms_of = [](u64 ns) { return ns / 1e6; };
    const double ms = std::chrono::duration<double, std::milli>(now.time - last.time).count();
    // A game frame noticeably slower than the usual ones (average of non-stutter frames).
    const bool hitch = hitch_frames > 60 && ms > std::max(1.5 * hitch_avg_ms, hitch_avg_ms + 5.0);
    if (hitch) {
        // Where the GPU command thread's time went during the frame (~250 us samples).
        static constexpr std::array<const char*, 7> names{"waiting for the game",
                                                          "preparing draws",
                                                          "waiting for the draw recorder",
                                                          "waiting on game/GPU memory",
                                                          "waiting for a display buffer",
                                                          "running commands",
                                                          "waiting for the GPU"};
        // Sample counts as shares of the frame (the sampler's real period varies by OS timer).
        std::array<u64, names.size()> samples{};
        u64 total_samples = 0;
        for (size_t i = 0; i < names.size(); ++i) {
            samples[i] = delta(now.states[i * 2], last.states[i * 2]) +
                         delta(now.states[i * 2 + 1], last.states[i * 2 + 1]);
            total_samples += samples[i];
        }
        std::string states;
        for (size_t i = 0; i < names.size(); ++i) {
            if (samples[i] != 0) {
                states += fmt::format("{}{} {:.1f}", states.empty() ? "" : ", ", names[i],
                                      ms * samples[i] / total_samples);
            }
        }
        const char* file = GS::slowest_wait_file.load(std::memory_order_relaxed);
        std::string_view site = file ? file : "-";
        if (const auto slash = site.find_last_of("/\\"); slash != std::string_view::npos) {
            site = site.substr(slash + 1);
        }
        LOG_WARNING(
            Render_Vulkan,
            "Stutter: frame {:.1f} ms (usual {:.1f} ms) | GPU thread ms: {} | draw recorder busy "
            "{:.1f} ms, {} packets | GPU waits {} ({:.1f} ms, longest {:.1f} ms at {}:{}) | {} "
            "pipelines compiled ({} more and {} shaders in the background, {} draws skipped "
            "meanwhile, waited {:.1f} ms for them) | {} "
            "texture uploads ({:.1f} MiB, {:.1f} ms, copying {:.1f} ms, {:.1f} MiB of left-out "
            "levels not copied, {:.1f} MiB copied directly) | "
            "buffer uploads {:.1f} MiB, {:.1f} ms | {} buffer ranges made resident ({:.1f} MiB, "
            "{:.1f} ms) | {} staging allocations ({:.1f} MiB, {:.1f} ms) | {} background "
            "copies (1 MiB pieces), waited {:.1f} ms | page protection {} "
            "calls {:.1f} ms, {} write faults | "
            "game: {} files opened, {} mutex waits, {} sleeps ({:.1f} ms) | end of submissions: "
            "buffer tick {:.1f}, downloads {:.1f}, GC {:.1f}, runtime tick {:.1f} ms | "
            "diagnostics logging {:.1f} ms",
            ms, hitch_avg_ms, states.empty() ? "-" : states,
            std::chrono::duration<double, std::milli>(now.busy - last.busy).count(),
            now.packets - last.packets, delta(now.gpu_waits, last.gpu_waits),
            ms_of(delta(now.gpu_wait_ns, last.gpu_wait_ns)), ms_of(slowest_ns), site,
            GS::slowest_wait_line.load(std::memory_order_relaxed),
            delta(now.pipelines, last.pipelines),
            delta(now.background_pipelines, last.background_pipelines),
            delta(now.background_shaders, last.background_shaders),
            delta(now.background_skips, last.background_skips),
            ms_of(delta(now.background_wait_ns, last.background_wait_ns)),
            delta(now.uploads, last.uploads),
            delta(now.upload_bytes, last.upload_bytes) / 1048576.0,
            ms_of(delta(now.upload_ns, last.upload_ns)),
            ms_of(delta(now.image_copy_ns, last.image_copy_ns)),
            delta(now.skipped_bytes, last.skipped_bytes) / 1048576.0,
            delta(now.direct_bytes, last.direct_bytes) / 1048576.0,
            delta(now.buffer_bytes, last.buffer_bytes) / 1048576.0,
            ms_of(delta(now.buffer_ns, last.buffer_ns)), delta(now.arena_binds, last.arena_binds),
            delta(now.resident_bytes, last.resident_bytes) / 1048576.0,
            ms_of(delta(now.resident_ns, last.resident_ns)),
            delta(now.staging_count, last.staging_count),
            delta(now.staging_bytes, last.staging_bytes) / 1048576.0,
            ms_of(delta(now.staging_ns, last.staging_ns)),
            delta(now.async_copies, last.async_copies),
            ms_of(delta(now.async_wait_ns, last.async_wait_ns)),
            delta(now.protect_calls, last.protect_calls),
            ms_of(delta(now.protect_ns, last.protect_ns)), delta(now.faults, last.faults),
            delta(now.file_opens, last.file_opens), delta(now.mutex_waits, last.mutex_waits),
            delta(now.sleeps, last.sleeps), ms_of(delta(now.sleep_ns, last.sleep_ns)),
            ms_of(delta(now.housekeeping[0], last.housekeeping[0])),
            ms_of(delta(now.housekeeping[1], last.housekeeping[1])),
            ms_of(delta(now.housekeeping[2], last.housekeeping[2])),
            ms_of(delta(now.housekeeping[3], last.housekeeping[3])),
            ms_of(delta(now.diag_ns, last.diag_ns)));
        // Where a wait for the GPU goes: the GPU's own timeline (executing our submissions, or
        // idle between them) and the presentation thread (window image, present call, frames).
        LOG_WARNING(
            Render_Vulkan,
            "Stutter detail: GPU timeline {:.1f} ms executing, {:.1f} ms idle between "
            "submissions ({} submissions measured, a frame or two late) | queue submit "
            "calls {:.1f} ms | presentation: {} presents, acquiring a window image "
            "{:.1f} ms, present calls {:.1f} ms, waiting for a free frame {:.1f} ms, {} GPU "
            "submissions queued behind a present",
            ms_of(delta(now.gpu_busy_ns, last.gpu_busy_ns)),
            ms_of(delta(now.gpu_idle_ns, last.gpu_idle_ns)),
            delta(now.gpu_submits, last.gpu_submits),
            ms_of(delta(now.queue_submit_ns, last.queue_submit_ns)),
            delta(now.presents, last.presents),
            ms_of(delta(now.present_acquire_ns, last.present_acquire_ns)),
            ms_of(delta(now.present_queue_ns, last.present_queue_ns)),
            ms_of(delta(now.present_frame_wait_ns, last.present_frame_wait_ns)),
            delta(now.queue_ops_behind_present, last.queue_ops_behind_present));
        // Where this frame's texture uploads came from and how long before the upload their data
        // went stale: what copying earlier (when the data is written) could take off the frame.
        {
            static constexpr std::array<const char*, 3> sources{
                "written by the game", "written by the GPU", "maybe written (small)"};
            const auto age_mib = [&](GS::DirtySource source, size_t bucket) {
                const size_t i = static_cast<size_t>(source) * GS::UploadAgeBuckets + bucket;
                return delta(now.upload_ages[i], last.upload_ages[i]) / 1048576.0;
            };
            std::string by_source =
                fmt::format("new textures {:.1f}", age_mib(GS::DirtySource::New, 0));
            for (size_t s = 1; s < static_cast<size_t>(GS::DirtySource::Count); ++s) {
                const auto source = static_cast<GS::DirtySource>(s);
                by_source +=
                    fmt::format(" | {} {:.1f} / {:.1f} / {:.1f} / {:.1f} / {:.1f}", sources[s - 1],
                                age_mib(source, 0), age_mib(source, 1), age_mib(source, 2),
                                age_mib(source, 3), age_mib(source, 4));
            }
            LOG_WARNING(Render_Vulkan,
                        "Stutter uploads by source (MiB; for writes, how long before the upload: "
                        "under 1 / 1-2 / 2-6 / 6-30 / 30+ frames): {}",
                        by_source);
        }
        {
            const auto mib_of = [](u64 bytes) { return bytes / 1048576.0; };
            const auto path = [&](GS::UploadPath which) {
                const auto i = static_cast<size_t>(which);
                return mib_of(delta(now.upload_paths[i], last.upload_paths[i]));
            };
            const auto packet = [&](size_t i) {
                return mib_of(delta(now.upload_packets[i], last.upload_packets[i]));
            };
            LOG_WARNING(Render_Vulkan,
                        "Stutter uploads by path (MiB): textures drawn or dispatched with "
                        "{:.1f}, storage images {:.1f}, render and depth targets {:.1f}, merged "
                        "images {:.1f}, the displayed frame {:.1f}, other {:.1f} | by draw "
                        "recorder packet: draws {:.1f}, indirect draws {:.1f}, dispatches {:.1f}, "
                        "indirect dispatches {:.1f}, queued jobs (DMA and the like) {:.1f}, not on "
                        "the draw recorder {:.1f}",
                        path(GS::UploadPath::Texture), path(GS::UploadPath::Storage),
                        path(GS::UploadPath::Target), path(GS::UploadPath::Overlap),
                        path(GS::UploadPath::Display), path(GS::UploadPath::Other), packet(0),
                        packet(1), packet(2), packet(3), packet(4), packet(5));
        }
        {
            const auto use = [&](GS::NewStorageUse which) {
                const auto i = static_cast<size_t>(which);
                return delta(now.new_storage[i], last.new_storage[i]) / 1048576.0;
            };
            LOG_WARNING(Render_Vulkan,
                        "Stutter new images written by shaders (MiB uploaded before the write): "
                        "only written, whole image, enough threads {:.1f} | also read {:.1f} | "
                        "only part bound {:.1f} | coverage unclear {:.1f} | of them guest memory "
                        "looked all zero {:.1f} | cleared on the GPU instead {:.1f}",
                        use(GS::NewStorageUse::Whole), use(GS::NewStorageUse::AlsoRead),
                        use(GS::NewStorageUse::Part), use(GS::NewStorageUse::Unclear),
                        delta(now.new_storage_zero, last.new_storage_zero) / 1048576.0,
                        delta(now.new_storage_cleared, last.new_storage_cleared) / 1048576.0);
        }
        if (EmulatorSettings.IsBbLowerDetailFirst()) {
            LOG_WARNING(Render_Vulkan,
                        "Stutter lower detail first: {} new textures sampled softer for now "
                        "({:.1f} MiB of largest levels put off) | {} got their largest level "
                        "({:.1f} MiB uploaded), {} of them at once for another use",
                        delta(now.detail_deferred, last.detail_deferred),
                        delta(now.detail_deferred_bytes, last.detail_deferred_bytes) / 1048576.0,
                        delta(now.detail_restored, last.detail_restored),
                        delta(now.detail_restored_bytes, last.detail_restored_bytes) / 1048576.0,
                        delta(now.detail_forced, last.detail_forced));
        }
        const auto heaps = instance.GetMemoryHeapReport();
        const auto& cleanup_now = now.cleanup;
        const auto& cleanup_last = last.cleanup;
        LOG_WARNING(
            Render_Vulkan,
            "Stutter VRAM: image collector {} passes ({} under pressure, {} aggressive) {:.1f} ms "
            "| freed {} images ({:.0f} MiB), {} write-backs started, {:.0f} MiB copied to guest "
            "memory in {:.1f} ms | {} uploads at addresses freed by the cleanup ({:.0f} MiB) | "
            "buffer evictions {} ({:.0f} MiB, {:.1f} ms, unbind {:.1f} ms) | upload scratch "
            "released {:.0f} MiB, allocated {:.0f} MiB | reading memory use {:.1f} ms | VRAM {} of "
            "{} MiB, {} MiB in system memory",
            delta(cleanup_now.passes, cleanup_last.passes),
            delta(cleanup_now.pressure_passes, cleanup_last.pressure_passes),
            delta(cleanup_now.aggressive_passes, cleanup_last.aggressive_passes),
            ms_of(delta(cleanup_now.ns, cleanup_last.ns)),
            delta(cleanup_now.freed_images, cleanup_last.freed_images),
            delta(cleanup_now.freed_bytes, cleanup_last.freed_bytes) / 1048576.0,
            delta(cleanup_now.writebacks, cleanup_last.writebacks),
            delta(cleanup_now.written_bytes, cleanup_last.written_bytes) / 1048576.0,
            ms_of(delta(cleanup_now.writeback_done_ns, cleanup_last.writeback_done_ns)),
            delta(cleanup_now.reuploads, cleanup_last.reuploads),
            delta(cleanup_now.reupload_bytes, cleanup_last.reupload_bytes) / 1048576.0,
            delta(cleanup_now.buffer_evictions, cleanup_last.buffer_evictions),
            delta(cleanup_now.buffer_eviction_bytes, cleanup_last.buffer_eviction_bytes) /
                1048576.0,
            ms_of(delta(cleanup_now.buffer_eviction_ns, cleanup_last.buffer_eviction_ns)),
            ms_of(delta(cleanup_now.buffer_eviction_unbind_ns,
                        cleanup_last.buffer_eviction_unbind_ns)),
            delta(cleanup_now.scratch_released_bytes, cleanup_last.scratch_released_bytes) /
                1048576.0,
            delta(cleanup_now.scratch_allocation_bytes, cleanup_last.scratch_allocation_bytes) /
                1048576.0,
            ms_of(delta(cleanup_now.usage_query_ns, cleanup_last.usage_query_ns)),
            heaps.local_usage >> 20, heaps.local_budget >> 20, heaps.host_usage >> 20);
        if (HitchProfile()) {
            // Where the draw recorder's time went in this frame (every draw timed).
            std::array<std::pair<double, const char*>, SectionNames.size()> sections{};
            for (size_t i = 0; i < sections.size(); ++i) {
                sections[i] = {(now.sections[i] - last.sections[i]) / 1e6, SectionNames[i]};
            }
            std::ranges::sort(sections, std::greater{});
            std::string by_section;
            for (const auto& [section_ms, name] : sections) {
                if (section_ms >= 0.3) {
                    by_section += fmt::format("{}{} {:.1f}", by_section.empty() ? "" : ", ", name,
                                              section_ms);
                }
            }
            static constexpr std::array<const char*, 5> kinds{
                "draws", "indirect draws", "dispatches", "indirect dispatches", "other packets"};
            std::string by_kind;
            double packets_ms = 0.0;
            for (size_t i = 0; i < kinds.size(); ++i) {
                const double kind_ms = ms_of(delta(now.packet_kinds[i], last.packet_kinds[i]));
                packets_ms += kind_ms;
                if (kind_ms >= 0.1) {
                    by_kind +=
                        fmt::format("{}{} {:.1f}", by_kind.empty() ? "" : ", ", kinds[i], kind_ms);
                }
            }
            const double busy_ms =
                std::chrono::duration<double, std::milli>(now.busy - last.busy).count();
            LOG_WARNING(Render_Vulkan,
                        "Stutter detail: draw recorder by packet: {} (between packets {:.1f}) | "
                        "draws by section: {}",
                        by_kind.empty() ? "-" : by_kind, std::max(0.0, busy_ms - packets_ms),
                        by_section.empty() ? "-" : by_section);
            {
                // GPU command thread: the packet types it spent most time on, its draw sections
                // and the work it did itself.
                std::array<std::pair<u64, size_t>, 256> by_opcode{};
                for (size_t i = 0; i < by_opcode.size(); ++i) {
                    by_opcode[i] = {delta(now.cmd_opcode_ns[i], last.cmd_opcode_ns[i]), i};
                }
                std::ranges::sort(by_opcode, std::greater<>{});
                std::string opcodes;
                for (size_t i = 0; i < 6 && by_opcode[i].first >= 100'000; ++i) {
                    const size_t op = by_opcode[i].second;
                    const auto name = [&]() -> std::string_view {
                        // magic_enum names values up to 127 only.
                        switch (static_cast<AmdGpu::PM4ItOpcode>(op)) {
                        case AmdGpu::PM4ItOpcode::LoadConstRam:
                            return "LoadConstRam";
                        case AmdGpu::PM4ItOpcode::WriteConstRam:
                            return "WriteConstRam";
                        case AmdGpu::PM4ItOpcode::DumpConstRam:
                            return "DumpConstRam";
                        case AmdGpu::PM4ItOpcode::IncrementCeCounter:
                            return "IncrementCeCounter";
                        case AmdGpu::PM4ItOpcode::IncrementDeCounter:
                            return "IncrementDeCounter";
                        case AmdGpu::PM4ItOpcode::WaitOnCeCounter:
                            return "WaitOnCeCounter (constant engine, waits for the draw recorder "
                                   "when it dumps constants)";
                        case AmdGpu::PM4ItOpcode::WaitOnDeCounterDiff:
                            return "WaitOnDeCounterDiff";
                        case AmdGpu::PM4ItOpcode::GetLodStats:
                            return "GetLodStats";
                        case AmdGpu::PM4ItOpcode::DrawIndirectCountMulti:
                            return "DrawIndirectCountMulti";
                        case AmdGpu::PM4ItOpcode::DrawIndexIndirectCountMulti:
                            return "DrawIndexIndirectCountMulti";
                        default:
                            return magic_enum::enum_name(static_cast<AmdGpu::PM4ItOpcode>(op));
                        }
                    }();
                    opcodes +=
                        fmt::format("{}{} {:.1f} ({})", opcodes.empty() ? "" : ", ",
                                    name.empty() ? fmt::format("{:#x}", op) : std::string{name},
                                    ms_of(by_opcode[i].first),
                                    delta(now.cmd_opcode_count[op], last.cmd_opcode_count[op]));
                }
                std::string sections;
                for (size_t i = 1; i < now.a_hitch_ns.size(); ++i) {
                    sections += fmt::format(
                        "{}{} {:.1f}", sections.empty() ? "" : ", ", ASectionNames[i],
                        static_cast<double>(now.a_hitch_ns[i] - last.a_hitch_ns[i]) / 1e6);
                }
                LOG_WARNING(
                    Render_Vulkan,
                    "Stutter detail: GPU command thread by packet type (ms, count; waits for the "
                    "game included): {} | compute queues {:.1f} ms ({} packets) | waits for the "
                    "draw "
                    "recorder before packets {:.1f} ms | constant engine: {} dumps, {} of them "
                    "waiting for the draw recorder {:.1f} ms, writing {:.1f} ms, run at submission "
                    "start/end {:.1f} ms; dumps that had to wait because queued draws read the "
                    "memory "
                    "later {}, queued work writes it {}, too many queued reads {}; written at once "
                    "past "
                    "huge shader writes {} | flips {}: waiting for the draw recorder {:.1f} ms, "
                    "flip "
                    "handling {:.1f} ms | {} draws "
                    "queued: {} ms | on this thread: "
                    "page "
                    "protection {} calls {:.1f} ms, buffers made resident {:.1f} ms, buffer "
                    "uploads {:.1f} MiB {:.1f} ms",
                    opcodes.empty() ? "-" : opcodes,
                    ms_of(delta(now.cmd_compute_ns, last.cmd_compute_ns)),
                    delta(now.cmd_compute_packets, last.cmd_compute_packets),
                    ms_of(delta(now.cmd_drain_ns, last.cmd_drain_ns)),
                    delta(now.ce_dumps, last.ce_dumps),
                    delta(now.ce_dump_drains, last.ce_dump_drains),
                    ms_of(delta(now.ce_dump_drain_ns, last.ce_dump_drain_ns)),
                    ms_of(delta(now.ce_dump_write_ns, last.ce_dump_write_ns)),
                    ms_of(delta(now.ce_outside_ns, last.ce_outside_ns)),
                    delta(now.ce_wait_reads, last.ce_wait_reads),
                    delta(now.ce_wait_writes, last.ce_wait_writes),
                    delta(now.ce_wait_overflow, last.ce_wait_overflow),
                    delta(now.ce_huge_write_skips, last.ce_huge_write_skips),
                    delta(now.flips, last.flips),
                    ms_of(delta(now.flip_drain_ns, last.flip_drain_ns)),
                    ms_of(delta(now.flip_signal_ns, last.flip_signal_ns)),
                    delta(now.a_hitch_draws, last.a_hitch_draws), sections,
                    delta(now.cmd_protect_calls, last.cmd_protect_calls),
                    ms_of(delta(now.cmd_protect_ns, last.cmd_protect_ns)),
                    ms_of(delta(now.cmd_resident_ns, last.cmd_resident_ns)),
                    delta(now.cmd_buffer_upload_bytes, last.cmd_buffer_upload_bytes) / 1048576.0,
                    ms_of(delta(now.cmd_buffer_upload_ns, last.cmd_buffer_upload_ns)));
            }
            const auto detail = [&](GS::Detail which) {
                const auto i = static_cast<size_t>(which);
                return ms_of(delta(now.details[i], last.details[i]));
            };
            LOG_WARNING(
                Render_Vulkan,
                "Stutter detail: new textures: protect {:.1f}, data to staging {:.1f}, detile "
                "{:.1f}, upload record {:.1f}, views {:.1f}, create {:.1f} ms (texture memory "
                "{:.1f} ms for {}, {} of them new memory blocks taking {:.1f} ms) | buffers: find, "
                "mark and protect {:.1f}, staging and copy {:.1f} ms | page protection on the "
                "draw recorder: {} calls, {:.1f} ms",
                detail(GS::Detail::TexTrack), detail(GS::Detail::TexStaging),
                detail(GS::Detail::TexDetile), detail(GS::Detail::TexUpload),
                detail(GS::Detail::TexView), detail(GS::Detail::TexCreate),
                ms_of(delta(now.alloc_ns, last.alloc_ns)), delta(now.allocs, last.allocs),
                delta(now.slow_allocs, last.slow_allocs),
                ms_of(delta(now.slow_alloc_ns, last.slow_alloc_ns)), detail(GS::Detail::BufTrack),
                detail(GS::Detail::BufCopy),
                delta(now.recorder_protect_calls, last.recorder_protect_calls),
                detail(GS::Detail::RecorderProtect));
            const auto protect_by = [&](size_t i) {
                return fmt::format("{} calls {:.1f} ms",
                                   delta(now.protect_calls_by[i], last.protect_calls_by[i]),
                                   ms_of(delta(now.protect_ns_by[i], last.protect_ns_by[i])));
            };
            constexpr auto by = [](GS::Detail which) { return static_cast<size_t>(which); };
            LOG_WARNING(
                Render_Vulkan,
                "Stutter detail: buffers: {} small read-only copies ({:.1f} MiB, {:.1f} ms) | "
                "arena lookups {:.1f} ms | made resident {:.1f} ms (memory asked from the driver: "
                "{} "
                "blocks, {:.1f} ms; evicted data uploaded again as it was {} times, {:.1f} MiB) | "
                "{} "
                "buffers with data to "
                "upload in {} ranges ({:.1f} MiB copied right away, the rest on the copy "
                "threads) | page protection on the draw recorder by step: new textures {}, "
                "buffer uploads {}, elsewhere {} | on the protection helper thread {} calls {:.1f} "
                "ms | evicted data back within 10 s {:.1f} MiB, 10-60 s {:.1f} MiB",
                delta(now.stream_copies, last.stream_copies),
                delta(now.stream_copy_bytes, last.stream_copy_bytes) / 1048576.0,
                detail(GS::Detail::BufStream), detail(GS::Detail::BufArena),
                detail(GS::Detail::BufResident),
                delta(now.buffer_mem_allocs, last.buffer_mem_allocs),
                ms_of(delta(now.buffer_mem_alloc_ns, last.buffer_mem_alloc_ns)),
                delta(now.evicted_refills, last.evicted_refills),
                delta(now.evicted_refill_bytes, last.evicted_refill_bytes) / 1048576.0,
                delta(now.buffer_syncs, last.buffer_syncs),
                delta(now.buffer_sync_ranges, last.buffer_sync_ranges),
                delta(now.buffer_sync_inline_bytes, last.buffer_sync_inline_bytes) / 1048576.0,
                protect_by(by(GS::Detail::TexTrack)), protect_by(by(GS::Detail::BufTrack)),
                protect_by(by(GS::Detail::Count)),
                delta(now.helper_protect_calls, last.helper_protect_calls),
                ms_of(delta(now.helper_protect_ns, last.helper_protect_ns)),
                delta(now.evicted_returned[0], last.evicted_returned[0]) / 1048576.0,
                delta(now.evicted_returned[1], last.evicted_returned[1]) / 1048576.0);
        }
    } else if (ms < 200.0) {
        // Usual frame time, stutters (and long pauses such as loading) left out.
        hitch_avg_ms = hitch_frames == 0 ? ms : hitch_avg_ms * 0.95 + ms * 0.05;
    }
    // A heavier scene is not a run of stutters: after 10 slow frames in a row their time is
    // the new usual one (otherwise every frame of it was reported).
    hitch_run = hitch && ms < 200.0 ? hitch_run + 1 : 0;
    if (hitch_run >= 10) {
        hitch_avg_ms = ms;
        hitch_run = 0;
    }
    ++hitch_frames;
}

void Rasterizer::OnFence() {
    DrainDrawPipe();
    // The guest may reuse memory once it sees the fence: background copies reading it go first.
    {
        const auto start = std::chrono::steady_clock::now();
        if (Common::CopyPool::WaitAsync()) {
            Common::GuestStats::async_copy_wait_ns.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - start)
                    .count(),
                std::memory_order_relaxed);
        }
    }
    texture_cache.ProcessDownloadImages();
}

bool Rasterizer::BindResources(const Pipeline* pipeline) {
    if (IsComputeImageCopy(pipeline) || IsComputeMetaClear(pipeline) ||
        IsComputeImageClear(pipeline)) {
        return false;
    }

    set_write_index = 0;
    set_writes.clear();
    buffer_infos.clear();
    image_infos.clear();

    bool uses_dma = false;

    // Upscaling: the G-buffer's material textures (5+ targets) get the upscaler's LOD bias, so
    // they keep the detail of the output resolution (NVIDIA's DLSS programming guide). Shadows,
    // post-processing and UI keep the game's.
    scene_lod_bias = 0.0f;
    bool gbuffer_draw = false;
    if (!pipeline->IsCompute() &&
        std::popcount(static_cast<const GraphicsPipeline*>(pipeline)->GetGraphicsKey().mrt_mask &
                      0xffu) >= 5) {
        scene_lod_bias = BbSceneMipBias();
        gbuffer_draw = true;
    }
    // Better AO: the G-buffer pass holds the scene's depth, and its constants the main camera
    // (shadow passes bind the same constants with the light's camera).
    if (EmulatorSettings.IsBbVbao() && !vbao) {
        vbao = std::make_unique<BbVbao>(instance, scheduler, runtime, texture_cache);
    }
    if (ssr && gbuffer_draw) {
        ssr->OnGBufferDraw(db_desc.image_id);
    }
    if (vbao) {
        if (gbuffer_draw && EmulatorSettings.IsBbVbao()) {
            const auto& vp = Regs().viewports[0];
            vbao->OnGBufferDraw(db_desc.image_id, vp.xscale < 0.0f ? -1.0f : 1.0f,
                                vp.yscale < 0.0f ? -1.0f : 1.0f);
        } else {
            vbao->OnOtherDraw();
        }
    }

    if (!blur_readbacks.empty()) [[unlikely]] {
        ReadBackBlurTextures();
    }
    if (blur_watch_left != 0) [[unlikely]] {
        LogBlurWatchWriters(pipeline);
    }

    // Bind resource buffers and textures.
    Shader::Backend::Bindings binding{};
    push_data = MakeUserData(Regs());
    if (object_motion) {
        const u64 no_motion = object_motion->NoParam();
        push_data.motion_param = {u32(no_motion), u32(no_motion >> 32)};
    }
    bind_stage_ordinal = 0;
    prefetch_cursor = 0;
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) {
            continue;
        }
        set_writes.resize(set_writes.size() + stage->buffers.size() + stage->images.size() +
                          stage->samplers.size());
        ProfileLap(Section::Targets);
        BindBuffers(*stage, binding, push_data);
        ProfileLap(Section::Buffers);
        BindTextures(*stage, binding);
        ProfileLap(Section::Textures);
        uses_dma |= stage->uses_dma;
        ++bind_stage_ordinal;
    }

    if (uses_dma) {
        buffer_cache.SynchronizeDmaBuffers();
    }
    ProfileLap(Section::Buffers);

    if (Common::GuestStats::blur_dump_requested.load(std::memory_order_relaxed) ||
        blur_dump_left != 0) [[unlikely]] {
        DumpBlurInputs(pipeline);
    }
    return true;
}

void Rasterizer::LogBlurWatchWriters(const Pipeline* pipeline) {
    if (--blur_watch_left == 0) {
        LOG_INFO(Render_Vulkan, "[BB-BLUR] writer watch over");
        blur_watch.clear();
        blur_watch_image = {};
        blur_writer_checks_left = 0;
        return;
    }
    const auto stage_hashes = [&] {
        std::string out;
        for (const auto* stage : pipeline->GetStages()) {
            if (stage) {
                out += fmt::format("{}{} {:#x}", out.empty() ? "" : ", ",
                                   magic_enum::enum_name(stage->sw_stage), stage->pgm_hash);
            }
        }
        return out;
    };
    const auto check_after = [&](VAddr address) {
        if (blur_writer_checks_left == 0 || !blur_watch_image) {
            return;
        }
        --blur_writer_checks_left;
        std::string buffers;
        for (const auto* stage : pipeline->GetStages()) {
            if (stage) {
                buffers += fmt::format(" | {} {:#x}: {}", magic_enum::enum_name(stage->sw_stage),
                                       stage->pgm_hash, DescribeBufferNans(*stage));
            }
        }
        if (!pipeline->IsCompute()) {
            const auto& graphics = *static_cast<const GraphicsPipeline*>(pipeline);
            buffers += fmt::format(" | vertex attributes: {} | character motion attachment {}",
                                   DescribeVertexNans(graphics),
                                   bool(graphics.GetGraphicsKey().motion_slot));
        }
        LOG_INFO(Render_Vulkan, "[BB-BLUR] writer inputs of {:#x}{}", address, buffers);
        if (blur_writer_input_dumps_left != 0) {
            --blur_writer_input_dumps_left;
            // The textures it reads, as sampled and as cached (read back before the next draw).
            for (const auto* stage : pipeline->GetStages()) {
                for (u32 i = 0; stage && i < stage->images.size(); ++i) {
                    if (stage->images[i].is_written) {
                        continue;
                    }
                    const auto tsharp = stage->images[i].GetSharp(*stage);
                    std::string cached;
                    texture_cache.ForEachImageInRegion(
                        tsharp.Address(), 1, [&](VideoCore::ImageId id, auto& image) {
                            cached += fmt::format(
                                "{}{:#x} {}x{} {}{}", cached.empty() ? "" : ", ",
                                image.info.guest_address, image.info.size.width,
                                image.info.size.height, vk::to_string(image.info.pixel_format),
                                True(image.flags & VideoCore::ImageFlagBits::GpuModified)
                                    ? " GPU-written"
                                    : "");
                            if (std::ranges::find(blur_readbacks, id, &BlurReadback::id) ==
                                blur_readbacks.end()) {
                                blur_readbacks.push_back({id, image.image_uid, {}, false});
                            }
                        });
                    LOG_INFO(Render_Vulkan,
                             "[BB-BLUR]   it reads ({} {:#x}) texture {}: {:#x} {}x{} as {} {} | "
                             "cached as: {}",
                             magic_enum::enum_name(stage->sw_stage), stage->pgm_hash, i,
                             tsharp.Address(), tsharp.width + 1, tsharp.height + 1,
                             magic_enum::enum_name(tsharp.GetDataFmt()),
                             magic_enum::enum_name(tsharp.GetNumberFmt()),
                             cached.empty() ? std::string{"nothing"} : cached);
                }
            }
        }
        blur_readbacks.push_back({blur_watch_image, blur_watch_uid, stage_hashes(), false});
    };
    if (!pipeline->IsCompute()) {
        const auto& regs = Regs();
        for (u32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS; ++cb) {
            const auto& col_buf = regs.color_buffers[cb];
            if (!col_buf || !regs.color_target_mask.GetMask(cb) ||
                !BlurWatched(col_buf.Address())) {
                continue;
            }
            check_after(col_buf.Address());
            LOG_INFO(Render_Vulkan,
                     "[BB-BLUR] writer: draw into {:#x} (target {}, format {} {}) by {} | fast "
                     "clear {}, cmask {:#x}{} | blend {}",
                     col_buf.Address(), cb, magic_enum::enum_name(col_buf.GetDataFmt()),
                     magic_enum::enum_name(col_buf.GetNumberFmt()), stage_hashes(),
                     bool(col_buf.info.fast_clear), col_buf.CmaskAddress(),
                     col_buf.info.fast_clear && col_buf.CmaskAddress() &&
                             texture_cache.IsMetaCleared(col_buf.CmaskAddress(),
                                                         col_buf.view.slice_start)
                         ? " (fast-cleared, clear pending)"
                         : "",
                     bool(regs.blend_control[cb].enable));
        }
        return;
    }
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) {
            continue;
        }
        for (const auto& image : stage->images) {
            if (image.is_written && BlurWatched(image.GetSharp(*stage).Address())) {
                LOG_INFO(Render_Vulkan, "[BB-BLUR] writer: dispatch storage write to {:#x} by {}",
                         image.GetSharp(*stage).Address(), stage_hashes());
                check_after(image.GetSharp(*stage).Address());
            }
        }
    }
}

std::string Rasterizer::DescribeBufferNans(const Shader::Info& stage) {
    std::string out;
    u32 clean = 0;
    for (u32 i = 0; i < stage.buffers.size(); ++i) {
        const auto& desc = stage.buffers[i];
        if (desc.IsSpecial()) {
            continue;
        }
        const auto vsharp = desc.GetSharp(stage);
        const VAddr address = vsharp.base_address;
        const u64 bytes = std::min<u64>(vsharp.GetSize(), 4_MB) & ~3ull;
        const bool gpu_written =
            address && bytes && buffer_cache.IsRegionGpuModified(address, vsharp.GetSize());
        // GPU-written: the GPU's copy (the guest copy may be older); else guest memory.
        std::span<const u8> data;
        if (gpu_written) {
            data = ReadBackBuffer(address, std::min<u64>(bytes, 256_KB));
        } else if (address && bytes && memory->IsReadableMapping(address, bytes)) {
            data = {std::bit_cast<const u8*>(address), bytes};
        }
        if (data.empty()) {
            out += fmt::format("{}b{} {:#x} not readable", out.empty() ? "" : ", ", i, address);
            continue;
        }
        u64 nans = 0, infs = 0, first_bad = ~0ull;
        for (u64 offset = 0; offset + 4 <= data.size(); offset += 4) {
            float value;
            std::memcpy(&value, data.data() + offset, sizeof(value));
            if (!std::isfinite(value)) {
                (std::isnan(value) ? nans : infs) += 1;
                first_bad = std::min(first_bad, offset);
            }
        }
        if (nans + infs == 0 && !gpu_written) {
            ++clean;
            continue;
        }
        out += fmt::format("{}b{} {:#x} {} bytes{}{}", out.empty() ? "" : ", ", i, address,
                           vsharp.GetSize(), desc.is_written ? " written" : "",
                           gpu_written ? " GPU-written (GPU copy read back)" : "");
        if (nans + infs != 0) {
            // The 16 floats from the 64 bytes holding the first bad one (a matrix).
            const u64 row = first_bad & ~63ull;
            std::string floats;
            for (u64 offset = row; offset < std::min<u64>(row + 64, data.size()); offset += 4) {
                float value;
                std::memcpy(&value, data.data() + offset, sizeof(value));
                u32 raw;
                std::memcpy(&raw, &value, sizeof(raw));
                floats += std::isfinite(value) ? fmt::format(" {:.4g}", value)
                                               : fmt::format(" 0x{:08x}", raw);
            }
            out += fmt::format(": {} NaN, {} infinite, first at +{:#x} [+{:#x}:{}]", nans, infs,
                               first_bad, row, floats);
        } else {
            out += ": finite";
        }
    }
    out += fmt::format("{}{} buffers with only finite floats", out.empty() ? "" : ", ", clean);
    return out;
}

std::string Rasterizer::DescribeVertexNans(const GraphicsPipeline& pipeline) {
    VertexInputs<vk::VertexInputAttributeDescription2EXT> attributes;
    VertexInputs<vk::VertexInputBindingDescription2EXT> bindings;
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    VertexInputs<AmdGpu::Buffer> guest_buffers;
    pipeline.GetVertexInputs(attributes, bindings, divisors, guest_buffers,
                             Regs().vgt_instance_step_rate_0, Regs().vgt_instance_step_rate_1);
    std::string out;
    u32 other = 0;
    for (u32 i = 0; i < guest_buffers.size(); ++i) {
        const auto& buffer = guest_buffers[i];
        const auto format = attributes[i].format;
        u32 components = 0, component_bytes = 4;
        switch (format) {
        case vk::Format::eR32Sfloat:
            components = 1;
            break;
        case vk::Format::eR32G32Sfloat:
            components = 2;
            break;
        case vk::Format::eR32G32B32Sfloat:
            components = 3;
            break;
        case vk::Format::eR32G32B32A32Sfloat:
            components = 4;
            break;
        case vk::Format::eR16G16Sfloat:
            components = 2;
            component_bytes = 2;
            break;
        case vk::Format::eR16G16B16A16Sfloat:
            components = 4;
            component_bytes = 2;
            break;
        default:
            break;
        }
        const VAddr address = buffer.base_address;
        const u64 size = std::min<u64>(buffer.GetSize(), 1_MB);
        const u32 stride = buffer.GetStride();
        if (components == 0 || address == 0 || size == 0 || stride == 0) {
            ++other;
            continue;
        }
        const bool gpu_written = buffer_cache.IsRegionGpuModified(address, size);
        std::span<const u8> data;
        if (gpu_written) {
            data = ReadBackBuffer(address, size);
        } else if (memory->IsReadableMapping(address, size)) {
            data = {std::bit_cast<const u8*>(address), size};
        }
        if (data.empty()) {
            out += fmt::format("{}a{} {:#x} not readable", out.empty() ? "" : ", ", i, address);
            continue;
        }
        u64 bad = 0, first_bad = ~0ull, elements = 0;
        const u32 element_bytes = components * component_bytes;
        for (u64 at = 0; at + element_bytes <= data.size(); at += stride, ++elements) {
            for (u32 c = 0; c < components; ++c) {
                bool finite;
                if (component_bytes == 4) {
                    float value;
                    std::memcpy(&value, data.data() + at + c * 4, 4);
                    finite = std::isfinite(value);
                } else {
                    u16 half;
                    std::memcpy(&half, data.data() + at + c * 2, 2);
                    finite = ((half >> 10) & 0x1f) != 0x1f;
                }
                if (!finite) {
                    ++bad;
                    first_bad = std::min(first_bad, elements);
                }
            }
        }
        out += fmt::format("{}a{} {} {:#x}{}: ", out.empty() ? "" : ", ", i, vk::to_string(format),
                           address, gpu_written ? " GPU-written" : "");
        out += bad == 0 ? fmt::format("{} finite", elements)
                        : fmt::format("{} NaN/inf of {} (first in element {})", bad, elements,
                                      first_bad);
    }
    out += fmt::format("{}{} non-float", out.empty() ? "" : ", ", other);
    return out;
}

std::span<const u8> Rasterizer::ReadBackBuffer(VAddr address, u64 size) {
    const auto [buffer, offset] = buffer_cache.ObtainBuffer(address, static_cast<u32>(size));
    if (!buffer) {
        return {};
    }
    if (!blur_buffer_readback || blur_buffer_readback->SizeBytes() < size) {
        blur_buffer_readback = std::make_unique<VideoCore::Buffer>(
            instance, 0, std::max<u64>(size, 1_MB), VideoCore::MemoryType::HostCached,
            "BlurBufferReadback");
    }
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    const vk::MemoryBarrier2 to_copy = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
    };
    cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &to_copy});
    const vk::BufferCopy copy{.srcOffset = offset, .dstOffset = 0, .size = size};
    cmdbuf.copyBuffer(buffer->Handle(), blur_buffer_readback->Handle(), copy);
    const vk::MemoryBarrier2 to_host = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead,
    };
    cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &to_host});
    scheduler.Finish();
    blur_buffer_readback->Invalidate(0, size);
    return {blur_buffer_readback->mapped_data.data(), size};
}

void Rasterizer::ReadBackBlurTextures() {
    auto readbacks = std::move(blur_readbacks);
    blur_readbacks.clear();
    for (const auto& [image_id, uid, after_writer, read_as_bytes] : readbacks) {
        if (!texture_cache.IsImageAllocated(image_id)) {
            continue;
        }
        auto& image = texture_cache.GetImage(image_id);
        const auto& info = image.info;
        if (image.image_uid != uid || info.num_samples > 1 || info.props.is_block ||
            info.props.is_depth || info.num_bits == 0) {
            continue;
        }
        const u32 width = std::max(info.size.width >> image.dropped_levels, 1u);
        const u32 height = std::max(info.size.height >> image.dropped_levels, 1u);
        const u32 texel_bytes = info.num_bits / 8;
        const u64 bytes = u64(width) * height * texel_bytes;
        if (!blur_readback_buffer || blur_readback_buffer->SizeBytes() < bytes) {
            blur_readback_buffer = std::make_unique<VideoCore::Buffer>(
                instance, 0, std::max<u64>(bytes, 16_MB), VideoCore::MemoryType::HostCached,
                "BlurReadback");
        }
        const vk::BufferImageCopy copy{
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, 1},
        };
        runtime.DownloadImage(&image, blur_readback_buffer.get(), std::span{&copy, 1});
        scheduler.Finish();
        blur_readback_buffer->Invalidate(0, bytes);
        const u8* data = blur_readback_buffer->mapped_data.data();
        // Components as floats (16- or 32-bit float formats), else as unsigned bytes.
        const auto format = info.pixel_format;
        const bool half = format == vk::Format::eR16G16B16A16Sfloat ||
                          format == vk::Format::eR16G16Sfloat || format == vk::Format::eR16Sfloat;
        const bool single = format == vk::Format::eR32G32B32A32Sfloat ||
                            format == vk::Format::eR32G32Sfloat || format == vk::Format::eR32Sfloat;
        u64 nans = 0, infs = 0, count = 0;
        double sum = 0.0, max = 0.0;
        if (half) {
            for (u64 i = 0; i + 2 <= bytes; i += 2) {
                u16 h;
                std::memcpy(&h, data + i, 2);
                const u32 exponent = (h >> 10) & 0x1f;
                const u32 mantissa = h & 0x3ff;
                if (exponent == 0x1f) {
                    (mantissa ? nans : infs) += 1;
                    continue;
                }
                const double value = exponent == 0
                                         ? std::ldexp(double(mantissa), -24)
                                         : std::ldexp(double(mantissa | 0x400), int(exponent) - 25);
                sum += value;
                max = std::max(max, value);
                ++count;
            }
        } else if (single) {
            for (u64 i = 0; i + 4 <= bytes; i += 4) {
                float f;
                std::memcpy(&f, data + i, 4);
                if (std::isnan(f)) {
                    ++nans;
                } else if (std::isinf(f)) {
                    ++infs;
                } else {
                    sum += std::abs(f);
                    max = std::max(max, double(std::abs(f)));
                    ++count;
                }
            }
        } else {
            for (u64 i = 0; i < bytes; ++i) {
                sum += data[i] / 255.0;
                max = std::max(max, data[i] / 255.0);
                ++count;
            }
        }
        // 32-bit texels as four bytes (a shader may read them as 8_8_8_8 unorm): the average of
        // each byte (0-255) and how many are 0 and 255.
        std::string bytes8;
        if (texel_bytes == 4) {
            std::array<u64, 4> byte_sum{}, zeros{}, fulls{};
            for (u64 i = 0; i + 4 <= bytes; i += 4) {
                for (u32 c = 0; c < 4; ++c) {
                    const u8 value = data[i + c];
                    byte_sum[c] += value;
                    zeros[c] += value == 0;
                    fulls[c] += value == 255;
                }
            }
            const u64 texels = std::max<u64>(bytes / 4, 1);
            bytes8 = " | as 8-bit channels:";
            for (u32 c = 0; c < 4; ++c) {
                bytes8 += fmt::format(" {} avg {:.1f} ({:.1f}% 0, {:.1f}% 255)", "rgba"[c],
                                      double(byte_sum[c]) / texels, 100.0 * zeros[c] / texels,
                                      100.0 * fulls[c] / texels);
            }
        }
        if (!after_writer.empty()) {
            LOG_INFO(Render_Vulkan,
                     "[BB-BLUR] {:#x} after the writer {}: {} NaN (was {}){} | largest {:.4g}{}",
                     info.guest_address, after_writer, nans, blur_last_nans,
                     nans > blur_last_nans ? " <- this writer added NaN" : "", max, bytes8);
            blur_last_nans = nans;
            continue;
        }
        if ((nans + infs != 0 || read_as_bytes) && blur_watch.empty()) {
            blur_watch_image = image_id;
            blur_watch_uid = uid;
            blur_last_nans = nans;
            blur_writer_checks_left = 96;
            blur_writer_input_dumps_left = 6;
            blur_watch.push_back(info.guest_address);
            blur_watch_left = 20000; // about two frames of draws
            LOG_INFO(Render_Vulkan, "[BB-BLUR] following the writers of {:#x} for the next draws",
                     info.guest_address);
            // Every cached image over the same memory (formats differing from the one read).
            texture_cache.ForEachImageInRegion(
                info.guest_address, info.guest_size, [&](VideoCore::ImageId id, auto& other) {
                    LOG_INFO(Render_Vulkan,
                             "[BB-BLUR]   cached image over it: {:#x} {}x{} {} ({} KiB){}{}{}",
                             other.info.guest_address, other.info.size.width,
                             other.info.size.height, vk::to_string(other.info.pixel_format),
                             other.info.guest_size / 1024, id == image_id ? " (this one)" : "",
                             other.usage.render_target ? ", render target" : "",
                             True(other.flags & VideoCore::ImageFlagBits::GpuModified)
                                 ? ", GPU-written"
                                 : "");
                });
        }
        LOG_INFO(Render_Vulkan,
                 "[BB-BLUR] texture {:#x} {}x{} {} read back: {} NaN and {} infinite of {} "
                 "values | largest {:.4g}, average {:.4g}{}{}",
                 info.guest_address, width, height, vk::to_string(format), nans, infs,
                 nans + infs + count, max, count ? sum / double(count) : 0.0,
                 half || single ? " (magnitudes)" : " (bytes as 0-1)", bytes8);
    }
}

void Rasterizer::DumpBlurInputs(const Pipeline* pipeline) {
    // The game's motion blur passes and the pass whose constants hold its blur matrix.
    constexpr std::array<u64, 3> BlurShaders{0xe0305cef, 0x29e06868, 0xd3c8bb21};
    if (Common::GuestStats::blur_dump_requested.exchange(false, std::memory_order_relaxed)) {
        blur_dump_left = 12;
        LOG_INFO(Render_Vulkan, "[BB-BLUR] Motion blur inputs requested: {}",
                 temporal_dlss.DescribeBlur());
    }
    if (pipeline->IsCompute()) {
        return;
    }
    bool match = false;
    for (const auto* stage : pipeline->GetStages()) {
        match |= stage && std::ranges::find(BlurShaders, stage->pgm_hash) != BlurShaders.end();
    }
    if (!match) {
        return;
    }
    --blur_dump_left;
    // What it draws into, and its textures read back from the GPU before the next draw.
    {
        std::string targets;
        const auto& regs = Regs();
        for (u32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS; ++cb) {
            const auto& col_buf = regs.color_buffers[cb];
            if (col_buf && regs.color_target_mask.GetMask(cb)) {
                targets +=
                    fmt::format("{}{}: {:#x}", targets.empty() ? "" : ", ", cb, col_buf.Address());
            }
        }
        LOG_INFO(Render_Vulkan, "[BB-BLUR] draw: color targets {} | depth {:#x}",
                 targets.empty() ? std::string{"none"} : targets,
                 regs.depth_buffer.DepthValid() ? regs.depth_buffer.DepthAddress() : u64{0});
        for (const auto image_id : bound_images) {
            if (!image_id || std::ranges::find(blur_readbacks, image_id, &BlurReadback::id) !=
                                 blur_readbacks.end()) {
                continue;
            }
            const auto& image = texture_cache.GetImage(image_id);
            bool read_as_bytes = false;
            for (const auto* stage : pipeline->GetStages()) {
                for (u32 i = 0; stage && i < stage->images.size(); ++i) {
                    const auto tsharp = stage->images[i].GetSharp(*stage);
                    read_as_bytes |= tsharp.Address() == image.info.guest_address &&
                                     tsharp.GetDataFmt() == AmdGpu::DataFormat::Format8_8_8_8 &&
                                     image.info.pixel_format != vk::Format::eR8G8B8A8Unorm &&
                                     image.info.pixel_format != vk::Format::eB8G8R8A8Unorm &&
                                     image.info.pixel_format != vk::Format::eR8G8B8A8Srgb &&
                                     image.info.pixel_format != vk::Format::eB8G8R8A8Srgb;
                }
            }
            if (read_as_bytes) {
                LOG_INFO(Render_Vulkan,
                         "[BB-BLUR] {:#x} is read as 8_8_8_8 but is kept as {}: its values are "
                         "logged as both",
                         image.info.guest_address, vk::to_string(image.info.pixel_format));
            }
            blur_readbacks.push_back({image_id, image.image_uid, {}, read_as_bytes});
        }
    }
    const auto floats = [&](VAddr address, u64 size) -> std::string {
        const u64 bytes = std::min<u64>(size, 128) & ~3ull;
        if (address == 0 || bytes == 0 || !memory->IsReadableMapping(address, bytes)) {
            return "(not readable)";
        }
        std::string out;
        for (u64 i = 0; i < bytes; i += 4) {
            float value;
            std::memcpy(&value, std::bit_cast<const u8*>(address + i), sizeof(value));
            u32 bits;
            std::memcpy(&bits, &value, sizeof(bits));
            out += std::isfinite(value) ? fmt::format("{}{:.5g}", i ? " " : "", value)
                                        : fmt::format("{}0x{:08x}", i ? " " : "", bits);
        }
        return out;
    };
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) {
            continue;
        }
        std::string user_data;
        for (size_t i = 0; i < std::min<size_t>(stage->user_data.size(), 16); ++i) {
            user_data += fmt::format("{}{:08x}", i ? " " : "", stage->user_data[i]);
        }
        LOG_INFO(Render_Vulkan, "[BB-BLUR] input: stage {} shader {:#x} | user data {}",
                 magic_enum::enum_name(stage->sw_stage), stage->pgm_hash, user_data);
        for (u32 i = 0; i < stage->buffers.size(); ++i) {
            const auto& desc = stage->buffers[i];
            if (desc.IsSpecial()) {
                if (desc.buffer_type == Shader::BufferType::Flatbuf) {
                    const auto& flat = stage->flattened_ud_buf;
                    std::string words;
                    for (size_t w = 0; w < std::min<size_t>(flat.size(), 32); ++w) {
                        float value;
                        std::memcpy(&value, &flat[w], sizeof(value));
                        words += fmt::format("{}{:.5g}", w ? " " : "", value);
                    }
                    LOG_INFO(Render_Vulkan, "[BB-BLUR]   buffer {} (flattened user data): {}", i,
                             words);
                }
                continue;
            }
            const auto vsharp = desc.GetSharp(*stage);
            const VAddr address = vsharp.base_address;
            const u64 size = vsharp.GetSize();
            const bool gpu_written =
                address && size && buffer_cache.IsRegionGpuModified(address, size);
            LOG_INFO(Render_Vulkan,
                     "[BB-BLUR]   buffer {}: {:#x}, {} bytes, stride {}{}{} | first floats: {}", i,
                     address, size, vsharp.GetStride(), desc.is_written ? ", written" : "",
                     gpu_written ? ", GPU-written (guest memory may be older)" : "",
                     floats(address, size));
        }
        for (u32 i = 0; i < stage->images.size(); ++i) {
            const auto tsharp = stage->images[i].GetSharp(*stage);
            LOG_INFO(Render_Vulkan,
                     "[BB-BLUR]   texture {}: {:#x}, {}x{}, data format {}, number format {}{}", i,
                     tsharp.Address(), tsharp.width + 1, tsharp.height + 1,
                     static_cast<u32>(tsharp.GetDataFmt()), static_cast<u32>(tsharp.GetNumberFmt()),
                     stage->images[i].is_written ? ", written" : "");
        }
    }
}

bool Rasterizer::PrepareVertexInputs(const GraphicsPipeline* pipeline) {
    if (!instance.IsVertexInputDynamicState()) {
        return false;
    }
    const auto& regs = liverpool->regs;
    vtx_attributes.clear();
    vtx_bindings.clear();
    vtx_buffers.clear();
    vtx_range_index.clear();
    vtx_ranges.clear();
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    pipeline->GetVertexInputs(vtx_attributes, vtx_bindings, divisors, vtx_buffers,
                              regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1);

    // Stream memory sorted and merged into ranges, as BindVertexBuffers does.
    struct Span {
        VAddr base;
        VAddr end;
    };
    VertexInputs<Span> spans;
    for (const auto& buffer : vtx_buffers) {
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            spans.push_back({buffer.base_address, buffer.base_address + buffer.GetSize()});
        }
    }
    std::ranges::sort(spans, {}, &Span::base);
    VertexInputs<Span> merged;
    for (const auto& span : spans) {
        if (merged.empty() || merged.back().end < span.base) {
            merged.push_back(span);
        } else {
            merged.back().end = std::max(merged.back().end, span.end);
        }
    }
    for (const auto& range : merged) {
        const u64 size = memory->ClampRangeSize(range.base, range.end - range.base);
        vtx_ranges.push_back({range.base, size, RingCopyGeometry(range.base, size)});
    }
    for (const auto& buffer : vtx_buffers) {
        u8 index = NoRange;
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            for (u32 i = 0; i < merged.size(); ++i) {
                if (buffer.base_address >= merged[i].base && buffer.base_address < merged[i].end) {
                    index = static_cast<u8>(i);
                    break;
                }
            }
        }
        vtx_range_index.push_back(index);
    }
    return true;
}

void Rasterizer::BindVertexBuffers(const GraphicsPipeline* pipeline) {
    if (const auto* prepared = packet_vertex) {
        // Worked out on stage A: obtain the buffers and record.
        vertex_prepared.fetch_add(1, std::memory_order_relaxed);
        auto& v = last_vertex;
        v.dynamic = true;
        v.attributes.assign(prepared->attributes, prepared->attributes + prepared->count);
        v.bindings.assign(prepared->bindings, prepared->bindings + prepared->count);
        v.num_buffers = 0;
        if (prepared->count == 0) {
            RecordVertexBindings();
            return;
        }
        std::array<std::pair<const VideoCore::Buffer*, u64>, MaxVertexBufferCount> obtained{};
        for (u32 i = 0; i < prepared->num_ranges; ++i) {
            const auto& range = prepared->ranges[i];
            if (range.ring != NoRing) {
                // Copied into the constant ring by the GPU command thread.
                obtained[i] = {const_ring.get(), range.ring}; // not tracked: see BindBuffers
                continue;
            }
            const auto [buffer, offset] =
                buffer_cache.ObtainGeometryBuffer(range.base, u32(range.size));
            obtained[i] = {buffer, offset};
            needs_barrier |= runtime.IsBufferAccessed(buffer, offset, range.size);
        }
        VertexInputs<vk::Buffer> host_buffers;
        VertexInputs<vk::DeviceSize> host_offsets;
        for (u32 i = 0; i < prepared->count; ++i) {
            const u8 index = prepared->range_index[i];
            if (index == NoRange) {
                host_buffers.push_back(VK_NULL_HANDLE);
                host_offsets.push_back(0);
                continue;
            }
            const auto& [buffer, offset] = obtained[index];
            host_buffers.push_back(buffer->Handle());
            host_offsets.push_back(offset + prepared->buffers[i].base_address -
                                   prepared->ranges[index].base);
        }
        v.num_buffers = prepared->count;
        v.buffers = host_buffers;
        v.offsets = host_offsets;
        RecordVertexBindings();
        return;
    }
    if (DrawPipe::OnStageB()) {
        vertex_unprepared.fetch_add(1, std::memory_order_relaxed);
    }
    const auto& regs = Regs();
    VertexInputs<vk::VertexInputAttributeDescription2EXT> attributes;
    VertexInputs<vk::VertexInputBindingDescription2EXT> bindings;
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    VertexInputs<AmdGpu::Buffer> guest_buffers;
    pipeline->GetVertexInputs(attributes, bindings, divisors, guest_buffers,
                              regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1);

    auto& v = last_vertex;
    v.dynamic = instance.IsVertexInputDynamicState();
    v.attributes = attributes;
    v.bindings = bindings;
    v.num_buffers = 0;

    if (bindings.empty()) {
        // If there are no bindings, there is nothing further to do.
        RecordVertexBindings();
        return;
    }

    struct BufferRange {
        VAddr base_address;
        VAddr end_address;
        const VideoCore::Buffer* buffer;
        u64 offset;

        [[nodiscard]] size_t GetSize() const {
            return end_address - base_address;
        }
    };

    // Build list of ranges covering the requested buffers
    VertexInputs<BufferRange> ranges{};
    for (const auto& buffer : guest_buffers) {
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            ranges.emplace_back(buffer.base_address, buffer.base_address + buffer.GetSize());
        }
    }

    // Merge connecting ranges together
    VertexInputs<BufferRange> ranges_merged{};
    if (!ranges.empty()) {
        std::ranges::sort(ranges, [](const BufferRange& lhv, const BufferRange& rhv) {
            return lhv.base_address < rhv.base_address;
        });
        ranges_merged.emplace_back(ranges[0]);
        for (auto range : ranges) {
            auto& prev_range = ranges_merged.back();
            if (prev_range.end_address < range.base_address) {
                ranges_merged.emplace_back(range);
            } else {
                prev_range.end_address = std::max(prev_range.end_address, range.end_address);
            }
        }
    }

    // Map buffers for merged ranges
    for (auto& range : ranges_merged) {
        const u64 size = memory->ClampRangeSize(range.base_address, range.GetSize());
        std::tie(range.buffer, range.offset) =
            buffer_cache.ObtainGeometryBuffer(range.base_address, u32(size));
        needs_barrier |= runtime.IsBufferAccessed(range.buffer, range.offset, size);
        bound_buffers.emplace_back(range.buffer, range.offset, size,
                                   vk::AccessFlagBits2::eVertexAttributeRead);
    }

    // Bind vertex buffers
    VertexInputs<vk::Buffer> host_buffers;
    VertexInputs<vk::DeviceSize> host_offsets;
    VertexInputs<vk::DeviceSize> host_sizes;
    VertexInputs<vk::DeviceSize> host_strides;
    for (const auto& buffer : guest_buffers) {
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            const auto host_buffer_info =
                std::ranges::find_if(ranges_merged, [&](const BufferRange& range) {
                    return buffer.base_address >= range.base_address &&
                           buffer.base_address < range.end_address;
                });
            ASSERT(host_buffer_info != ranges_merged.cend());
            host_buffers.emplace_back(host_buffer_info->buffer->Handle());
            host_offsets.push_back(host_buffer_info->offset + buffer.base_address -
                                   host_buffer_info->base_address);
        } else {
            host_buffers.emplace_back(VK_NULL_HANDLE);
            host_offsets.push_back(0);
        }
        host_sizes.push_back(buffer.GetSize());
        host_strides.push_back(buffer.GetStride());
    }

    v.num_buffers = static_cast<u32>(guest_buffers.size());
    v.buffers = host_buffers;
    v.offsets = host_offsets;
    v.sizes = host_sizes;
    v.strides = host_strides;
    RecordVertexBindings();
}

void Rasterizer::BindIndexBuffer(u32 index_offset, bool is_indirect) {
    const auto& regs = Regs();

    // Figure out index type and size.
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const vk::IndexType index_type = is_index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
    const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
    const VAddr base_address = regs.index_base_address.Address<VAddr>();
    const VAddr index_address = base_address + u64(index_offset) * index_size;

    // Bind index buffer.
    const u32 num_indices =
        is_indirect ? regs.max_index_size : std::min(regs.num_indices, regs.max_index_size);
    const u64 requested_size = u64(num_indices) * index_size;
    const u64 mapped_size =
        base_address != 0 ? memory->ClampRangeSize(index_address, requested_size) : 0;
    u32 index_buffer_size = static_cast<u32>(Common::AlignDown(mapped_size, index_size));
    vk::Buffer index_buffer{};
    u64 buffer_offset{};
    if (index_buffer_size != 0 && !is_indirect && packet_index_ring.offset != NoRing &&
        packet_index_ring.address == index_address && packet_index_ring.size == index_buffer_size) {
        // Copied into the constant ring by the GPU command thread.
        // Not tracked for barriers: see BindBuffers.
        const auto* ring = const_ring.get();
        index_buffer = ring->Handle();
        buffer_offset = packet_index_ring.offset;
    } else if (index_buffer_size != 0) {
        const auto [buffer, offset] =
            buffer_cache.ObtainGeometryBuffer(index_address, index_buffer_size);
        needs_barrier |= runtime.IsBufferAccessed(buffer, offset, index_buffer_size);
        bound_buffers.emplace_back(buffer, offset, index_buffer_size,
                                   vk::AccessFlagBits2::eIndexRead);
        index_buffer = buffer->Handle();
        buffer_offset = offset;
    } else if (!instance.IsMaintenance6Supported()) {
        const auto* null_buffer = buffer_cache.GetNullIndexBuffer();
        index_buffer_size = static_cast<u32>(null_buffer->SizeBytes());
        needs_barrier |= runtime.IsBufferAccessed(null_buffer, 0, index_buffer_size);
        bound_buffers.emplace_back(null_buffer, 0, index_buffer_size,
                                   vk::AccessFlagBits2::eIndexRead);
        index_buffer = null_buffer->Handle();
    }
    last_index = {index_buffer, buffer_offset, index_buffer_size, index_type};
    RecordIndexBinding();
}

void Rasterizer::ResetBindings(bool is_compute) {
    for (auto& image_id : bound_images) {
        texture_cache.GetImage(image_id).binding = {};
    }
    for (const auto [buffer, offset, size, src_access] : bound_buffers) {
        const auto dst_stage = is_compute ? vk::PipelineStageFlagBits2::eComputeShader
                                          : vk::PipelineStageFlagBits2::eAllGraphics;
        runtime.AccessBuffer(buffer, offset, size, dst_stage, src_access);
    }
    bound_images.clear();
    first_texture_infos.clear();
    bound_buffers.clear();
    needs_barrier = false;
}

bool Rasterizer::IsComputeMetaClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Most of the time when a metadata is updated with a shader it gets cleared. It means
    // we can skip the whole dispatch and update the tracked state instead. Also, it is not
    // intended to be consumed and in such rare cases (e.g. HTile introspection, CRAA) we
    // will need its full emulation anyways.
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);

    // Assume if a shader reads metadata, it is a copy shader.
    for (const auto& desc : info.buffers) {
        const VAddr address = desc.GetSharp(info).base_address;
        if (!desc.IsSpecial() && !desc.is_written && texture_cache.IsMeta(address)) {
            return false;
        }
    }

    // Metadata surfaces are tiled and thus need address calculation to be written properly.
    // If a shader wants to encode HTILE, for example, from a depth image it will have to compute
    // proper tile address from dispatch invocation id. This address calculation contains an xor
    // operation so use it as a heuristic for metadata writes that are probably not clears.
    if (!info.has_bitwise_xor) {
        // Assume if a shader writes metadata without address calculation, it is a clear shader.
        for (const auto& desc : info.buffers) {
            const VAddr address = desc.GetSharp(info).base_address;
            if (!desc.IsSpecial() && desc.is_written && blur_watch_left != 0) {
                if (const auto meta = texture_cache.GetMetaInfo(address)) {
                    const auto& owner = texture_cache.GetImage(meta->owner_id);
                    if (BlurWatched(owner.info.guest_address)) {
                        LOG_INFO(Render_Vulkan,
                                 "[BB-BLUR] writer: metadata clear of {:#x} (meta {:#x}) by "
                                 "compute {:#x}",
                                 owner.info.guest_address, address, info.pgm_hash);
                    }
                }
            }
            if (!desc.IsSpecial() && desc.is_written && texture_cache.ClearMeta(address)) {
                // Assume all slices were updates
                LOG_TRACE(Render_Vulkan, "Metadata update skipped");
                return true;
            }
        }
    }
    return false;
}

// Diagnostics: why a compute shader that looks like a memory copy or clear of a render target
// is not turned into an image copy or clear (each reason once per shader).
static void NoteHleReject(const char* kind, u64 hash, const std::string& reason) {
    static std::mutex mutex;
    static std::unordered_set<u64> seen;
    std::scoped_lock lock{mutex};
    const u64 key =
        hash ^ (std::hash<std::string>{}(reason) * 31) ^ std::hash<std::string_view>{}(kind);
    if (seen.size() < 256 && seen.insert(key).second) {
        LOG_INFO(Render_Vulkan, "[HLE check] {} cs {:08x}: {}", kind, hash, reason);
    }
}

// The game's buffers of a shader, without the ones the recompiler adds (flattened user data,
// page table, ...). A compute copy or clear has exactly two; the extras made the image copy and
// clear detectors reject Bloodborne's render target copies (as bbport found).
static boost::container::small_vector<const Shader::BufferResource*, 4> GuestBuffers(
    const Shader::Info& info) {
    boost::container::small_vector<const Shader::BufferResource*, 4> buffers;
    for (const auto& desc : info.buffers) {
        if (!desc.IsSpecial()) {
            buffers.push_back(&desc);
        }
    }
    return buffers;
}

static std::string DescribeImage(const VideoCore::Image& image) {
    return fmt::format("{}x{} {} bits={} pitch={} guest_size={:#x} tile={} depth={} flags={:#x}",
                       image.info.size.width, image.info.size.height,
                       vk::to_string(image.info.pixel_format), image.info.num_bits,
                       image.info.pitch, image.info.guest_size, u32(image.info.tile_mode),
                       image.info.props.is_depth ? 1 : 0, u32(image.flags));
}

bool Rasterizer::IsComputeImageCopy(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = CsRegs();
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);
    const bool diag = diagnostics.load(std::memory_order_relaxed);
    const u64 hash = info.pgm_hash;
    const auto guest = GuestBuffers(info);
    if (cs_pgm.num_thread_x.full != 64 || guest.size() != 2 || !info.images.empty() ||
        info.uses_dma) {
        if (diag && (hash == 0x3d5ebf4e || hash == 0x8b355b5a || hash == 0xe2c27f0c)) {
            NoteHleReject("copy", hash,
                          fmt::format("threads {} buffers {} (game {}) images {} dma {}",
                                      cs_pgm.num_thread_x.full, info.buffers.size(), guest.size(),
                                      info.images.size(), info.uses_dma));
        }
        return false;
    }

    // Those 2 buffers must both be formatted. One must be source and another destination.
    const auto& desc0 = *guest[0];
    const auto& desc1 = *guest[1];
    if (!desc0.is_formatted || !desc1.is_formatted || desc0.is_written == desc1.is_written) {
        if (diag && hash == 0x3d5ebf4e) {
            NoteHleReject("copy", hash,
                          fmt::format("formatted {}/{} written {}/{}", desc0.is_formatted,
                                      desc1.is_formatted, desc0.is_written, desc1.is_written));
        }
        return false;
    }
    if (info.buffers.size() == 3 && info.buffers[2].buffer_type != Shader::BufferType::Flatbuf) {
        return false;
    }

    // Buffers must have the same size and each thread of the dispatch must copy 1 dword of data
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
    if (buf0.GetSize() != buf1.GetSize() || cs_pgm.dim_x != (buf0.GetSize() / 256)) {
        if (diag) {
            NoteHleReject("copy", hash,
                          fmt::format("sizes {:#x}/{:#x} groups {}", buf0.GetSize(), buf1.GetSize(),
                                      cs_pgm.dim_x));
        }
        return false;
    }

    // Find images the buffer alias
    const auto image0_id = texture_cache.FindImageFromRange(buf0.base_address, buf0.GetSize());
    if (!image0_id) {
        if (diag) {
            NoteHleReject("copy", hash,
                          fmt::format("no {} image at buffer 0 {:#x} size {:#x} ({})",
                                      texture_cache.FindImageFromRange(buf0.base_address,
                                                                       buf0.GetSize(), false)
                                          ? "downloadable"
                                          : "",
                                      buf0.base_address, buf0.GetSize(),
                                      desc0.is_written ? "destination" : "source"));
        }
        return false;
    }
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        if (diag) {
            NoteHleReject("copy", hash,
                          fmt::format("no image at buffer 1 {:#x} size {:#x} ({}); buffer 0: {}",
                                      buf1.base_address, buf1.GetSize(),
                                      desc1.is_written ? "destination" : "source",
                                      DescribeImage(texture_cache.GetImage(image0_id))));
        }
        return false;
    }

    // Image copy must be valid
    VideoCore::Image& image0 = texture_cache.GetImage(image0_id);
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    // Bloodborne copies its depth buffer (with the stencil after it) into an R32F texture: the
    // copy is larger than the images. The depth part is an image copy when no image lives in
    // the rest of the destination (nothing samples the copied stencil bytes as a texture).
    bool size_ok = image0.info.guest_size == buf0.GetSize();
    if (!size_ok && image0.info.guest_size < buf0.GetSize()) {
        const AmdGpu::Buffer& dst_buf = desc0.is_written ? buf0 : buf1;
        const VAddr tail = dst_buf.base_address + image0.info.guest_size;
        const u64 tail_size = buf0.GetSize() - image0.info.guest_size;
        bool tail_used = false;
        texture_cache.ForEachImageInRegion(tail, tail_size,
                                           [&](VideoCore::ImageId, auto&) { tail_used = true; });
        size_ok = !tail_used;
    }
    if (image0.info.guest_size != image1.info.guest_size ||
        image0.info.pitch != image1.info.pitch || !size_ok ||
        image0.info.num_bits != image1.info.num_bits) {
        if (diag) {
            NoteHleReject("copy", hash,
                          fmt::format("images differ (buffer size {:#x}): 0 = {}; 1 = {}",
                                      buf0.GetSize(), DescribeImage(image0),
                                      DescribeImage(image1)));
        }
        return false;
    }
    if (diag) {
        NoteHleReject("copy", hash,
                      fmt::format("done as an image copy: {} -> {}",
                                  DescribeImage(desc0.is_written ? image1 : image0),
                                  DescribeImage(desc0.is_written ? image0 : image1)));
    }

    // Perform image copy
    VideoCore::Image& src_image = desc0.is_written ? image1 : image0;
    VideoCore::Image& dst_image = desc0.is_written ? image0 : image1;
    texture_cache.EnsureFullDetail(src_image);
    runtime.CopyColorAndDepth(&src_image, &dst_image);
    return true;
}

bool Rasterizer::IsComputeImageClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = CsRegs();
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);
    const auto guest = GuestBuffers(info);
    if (cs_pgm.num_thread_x.full != 64 || guest.size() != 2 || !info.images.empty() ||
        info.uses_dma) {
        return false;
    }

    // From those 2 buffers, first must hold the clear vector and second the image being cleared
    const auto& desc0 = *guest[0];
    const auto& desc1 = *guest[1];
    if (desc0.is_formatted || !desc1.is_formatted || desc0.is_written || !desc1.is_written) {
        return false;
    }
    if (info.buffers.size() == 3 && info.buffers[2].buffer_type != Shader::BufferType::Flatbuf) {
        return false;
    }

    // First buffer must have size of vec4 and second the size of a single layer
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
    const u32 buf1_bpp = AmdGpu::NumBitsPerBlock(buf1.GetDataFmt());
    const bool diag = diagnostics.load(std::memory_order_relaxed);
    if (buf0.GetSize() != 16 || (cs_pgm.dim_x * 128ULL * (buf1_bpp / 8)) != buf1.GetSize()) {
        if (diag) {
            NoteHleReject("clear", info.pgm_hash,
                          fmt::format("sizes {}/{:#x} bpp {} groups {}", buf0.GetSize(),
                                      buf1.GetSize(), buf1_bpp, cs_pgm.dim_x));
        }
        return false;
    }

    // Find image the buffer alias: the whole image, or (Clear shadow cube faces) one layer of an
    // array or cube.
    auto image1_id = texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    u32 layer = 0;
    bool one_layer = false;
    if (EmulatorSettings.IsBbClearOneLayer() &&
        (!image1_id || texture_cache.GetImage(image1_id).info.guest_size != buf1.GetSize())) {
        if (const auto layer_id =
                texture_cache.FindImageLayer(buf1.base_address, buf1.GetSize(), layer)) {
            image1_id = layer_id;
            one_layer = true;
        }
    }
    if (!image1_id) {
        if (diag) {
            NoteHleReject(
                "clear", info.pgm_hash,
                fmt::format("no image at {:#x} size {:#x}", buf1.base_address, buf1.GetSize()));
        }
        return false;
    }

    // Image clear must be valid
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if ((!one_layer && image1.info.guest_size != buf1.GetSize()) ||
        image1.info.num_bits != buf1_bpp || image1.info.props.is_depth) {
        if (diag) {
            NoteHleReject("clear", info.pgm_hash,
                          fmt::format("image differs (size {:#x} bpp {}): {}", buf1.GetSize(),
                                      buf1_bpp, DescribeImage(image1)));
        }
        return false;
    }

    // Perform image clear
    const float* values = reinterpret_cast<float*>(buf0.base_address);
    const vk::ClearValue clear = {
        .color = {.float32 = std::array<float, 4>{values[0], values[1], values[2], values[3]}},
    };
    const VideoCore::SubresourceRange range =
        one_layer
            ? VideoCore::SubresourceRange{.base = {.level = 0, .layer = static_cast<u16>(layer)},
                                          .extent = {.levels = 1, .layers = 1}}
            : VideoCore::SubresourceRange{.base = {.level = 0, .layer = 0},
                                          .extent = image1.info.resources};
    if (one_layer) {
        static std::atomic<u32> layers_logged{};
        if (layers_logged.fetch_add(1, std::memory_order_relaxed) < 8) {
            LOG_INFO(Render_Vulkan,
                     "Clear shadow cube faces: layer {} of {:#x} ({} layers) cleared", layer,
                     image1.info.guest_address, image1.info.resources.layers);
        }
    }
    ClearImageAs(image1, range,
                 LiverpoolToVK::SurfaceFormat(buf1.GetDataFmt(), buf1.GetNumberFmt()), clear);
    return true;
}

void Rasterizer::ClearImageAs(VideoCore::Image& image, const VideoCore::SubresourceRange& range,
                              vk::Format format, const vk::ClearValue& value) {
    const auto image_format = image.info.pixel_format;
    if (format == image_format || format == vk::Format::eUndefined || image.info.props.is_depth ||
        !VideoCore::IsVulkanFormatCompatible(image_format, format)) {
        runtime.ClearImage(&image, range, value);
        return;
    }
    const bool as_game = bool(image.usage_flags & vk::ImageUsageFlagBits::eColorAttachment);
    static std::atomic<u32> logged{};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
        LOG_INFO(Render_Vulkan, "Clear of {:#x} {}x{} in {} (kept as {}) to {} {} {} {}: {}",
                 image.info.guest_address, image.info.size.width, image.info.size.height,
                 vk::to_string(format), vk::to_string(image_format), value.color.float32[0],
                 value.color.float32[1], value.color.float32[2], value.color.float32[3],
                 as_game ? "cleared through a view in the game's format"
                         : "cleared in the image's format (not a render target)");
    }
    if (!as_game) {
        runtime.ClearImage(&image, range, value);
        return;
    }
    scheduler.EndRendering();
    for (u32 level = range.base.level; level < range.base.level + range.extent.levels; ++level) {
        if (level < image.dropped_levels) {
            continue;
        }
        VideoCore::ImageViewInfo view_info{};
        view_info.type =
            range.extent.layers > 1 ? AmdGpu::ImageType::Color2DArray : AmdGpu::ImageType::Color2D;
        view_info.format = format;
        view_info.range = {
            .base = {.level = static_cast<u16>(level), .layer = range.base.layer},
            .extent = {.levels = 1, .layers = range.extent.layers},
        };
        const auto& view = image.FindView(view_info, false);
        if (runtime.Transit(&image, vk::ImageLayout::eColorAttachmentOptimal,
                            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                            vk::AccessFlagBits2::eColorAttachmentWrite, view_info.range)) {
            runtime.FlushBarriers();
        }
        RenderState state{};
        auto& attachment = state.color_attachments[0];
        attachment.image_view = *view.image_view;
        attachment.image_layout = vk::ImageLayout::eColorAttachmentOptimal;
        std::memcpy(attachment.clear_value.data(), value.color.float32.data(),
                    sizeof(attachment.clear_value));
        attachment.is_clear = 1;
        state.width = static_cast<u16>(std::max(image.info.size.width >> level, 1u));
        state.height = static_cast<u16>(std::max(image.info.size.height >> level, 1u));
        state.num_layers = range.extent.layers;
        state.num_color_attachments = 1;
        scheduler.BeginRendering(state);
        scheduler.EndRendering();
    }
    image.flags |= VideoCore::ImageFlagBits::GpuModified;
    image.flags &= ~VideoCore::ImageFlagBits::Dirty;
}

void Rasterizer::BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                             Shader::PushData& push_data) {
    const u64 alignment = instance.StorageMinAlignment();
    for (u32 buffer_index = 0; buffer_index < stage.buffers.size(); ++buffer_index) {
        const auto& desc = stage.buffers[buffer_index];
        const auto src_access =
            vk::AccessFlagBits2::eShaderRead |
            (desc.is_written ? vk::AccessFlagBits2::eShaderWrite : vk::AccessFlagBits2::eNone);
        if (const auto* prefetched = packet_num_prefetch ? FindPrefetched(buffer_index) : nullptr) {
            // Copied by the GPU command thread into the constant ring (PrefetchBuffers).
            // (The DLSS scene-constant scan reads the same data from ordinary memory: the ring
            // may be write-combined, slow to read from the CPU.)
            const auto* ring = const_ring.get();
            const u32 size = prefetched->size;
            buffer_infos.emplace_back(ring->Handle(), prefetched->offset, size);
            if (buffer_index == 0 && stage.pgm_hash == BbSsr::TraceShader) {
                ssr_constants = {ring->Handle(), prefetched->offset, size, 0};
                ssr_guest_constants =
                    reinterpret_cast<const void*>(desc.GetSharp(stage).base_address);
            }
            if (desc.IsSpecial()) {
                temporal_dlss.ObserveConstants(runtime, instance, ring, prefetched->offset, size,
                                               stage.pgm_hash, binding.unified);
                temporal_dlss.ObserveSceneConstants(stage.FlatUserData().data(), size);
                if (vbao) {
                    vbao->ObserveSceneConstants(stage.FlatUserData().data(), size);
                }
                temporal_dlss.ObserveBlurInputs(stage.FlatUserData().data(), size, stage.pgm_hash,
                                                binding.unified);
            } else {
                // No barrier tracking: the GPU never writes the ring, and the host writes of a
                // region are visible at submission (it is reused only after the GPU is done).
                if (vbao) {
                    vbao->ObserveSceneConstants(
                        reinterpret_cast<const void*>(desc.GetSharp(stage).base_address), size);
                }
                if (temporal_dlss.Requested()) {
                    temporal_dlss.ObserveConstants(runtime, instance, ring, prefetched->offset,
                                                   size, stage.pgm_hash, binding.unified);
                    temporal_dlss.ObserveSceneConstants(
                        reinterpret_cast<const void*>(desc.GetSharp(stage).base_address), size);
                    temporal_dlss.ObserveBlurInputs(
                        reinterpret_cast<const void*>(desc.GetSharp(stage).base_address), size,
                        stage.pgm_hash, binding.unified);
                }
            }
            auto& set_write = set_writes[set_write_index++];
            set_write.dstSet = VK_NULL_HANDLE;
            set_write.dstBinding = binding.unified++;
            set_write.dstArrayElement = 0;
            set_write.descriptorCount = 1;
            set_write.descriptorType = vk::DescriptorType::eStorageBuffer;
            set_write.pBufferInfo = &buffer_infos.back();
            ++binding.buffer;
            continue;
        }
        if (desc.IsSpecial()) {
            if (desc.buffer_type == Shader::BufferType::GdsBuffer) {
                const auto* gds_buf = buffer_cache.GetGdsBuffer();
                buffer_infos.emplace_back(gds_buf->Handle(), 0, gds_buf->SizeBytes());
                needs_barrier |=
                    runtime.IsBufferAccessed(gds_buf, 0, gds_buf->SizeBytes(), desc.is_written);
                bound_buffers.emplace_back(gds_buf, 0, gds_buf->SizeBytes(), src_access);
            } else if (desc.buffer_type == Shader::BufferType::Flatbuf) {
                auto& vk_buffer = buffer_cache.GetStreamBuffer();
                const auto flat_ud = stage.FlatUserData();
                const u32 ubo_size = flat_ud.size() * sizeof(u32);
                const u64 offset = vk_buffer.Copy(flat_ud.data(), ubo_size, alignment);
                buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
                temporal_dlss.ObserveConstants(runtime, instance, &vk_buffer, offset, ubo_size,
                                               stage.pgm_hash, binding.unified);
                temporal_dlss.ObserveSceneConstants(flat_ud.data(), ubo_size);
                if (vbao) {
                    vbao->ObserveSceneConstants(flat_ud.data(), ubo_size);
                }
                temporal_dlss.ObserveBlurInputs(flat_ud.data(), ubo_size, stage.pgm_hash,
                                                binding.unified);
            } else if (desc.buffer_type == Shader::BufferType::ClipPlanes) {
                // Permutations compiled without enabled planes never read the buffer, so the
                // declared binding is satisfied with a null descriptor instead of a copy.
                if (Regs().clipper_control.user_clip_plane_enable == 0) {
                    buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
                } else {
                    auto& vk_buffer = buffer_cache.GetStreamBuffer();
                    std::array<float, AmdGpu::NUM_CLIP_PLANES * 4> planes{};
                    for (u32 i = 0; i < AmdGpu::NUM_CLIP_PLANES; ++i) {
                        const auto& plane = Regs().clip_user_data[i];
                        planes[i * 4 + 0] = std::bit_cast<float>(plane.data_x);
                        planes[i * 4 + 1] = std::bit_cast<float>(plane.data_y);
                        planes[i * 4 + 2] = std::bit_cast<float>(plane.data_z);
                        planes[i * 4 + 3] = std::bit_cast<float>(plane.data_w);
                    }
                    const u32 ubo_size = static_cast<u32>(sizeof(planes));
                    const u64 offset = vk_buffer.Copy(planes.data(), ubo_size, alignment);
                    buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
                }
            } else if (desc.buffer_type == Shader::BufferType::BdaPagetable) {
                const auto* bda_buffer = buffer_cache.GetBdaPageTableBuffer();
                buffer_infos.emplace_back(bda_buffer->Handle(), 0, bda_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::FaultBuffer) {
                const auto* fault_buffer = buffer_cache.GetFaultBuffer();
                buffer_infos.emplace_back(fault_buffer->Handle(), 0, fault_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::SharedMemory) {
                auto& lds_buffer = buffer_cache.GetStreamBuffer();
                const auto& cs_program = CsRegs();
                const auto lds_size = cs_program.SharedMemSize() * cs_program.NumWorkgroups();
                const auto [data, offset] = lds_buffer.Map(lds_size, alignment);
                std::memset(data, 0, lds_size);
                lds_buffer.Commit();
                buffer_infos.emplace_back(lds_buffer.Handle(), offset, lds_size);
            } else {
                UNREACHABLE_MSG("Unexpected buffer type {}", u32(desc.buffer_type));
            }
        } else {
            const auto vsharp = desc.GetSharp(stage);
            if (vsharp.base_address == 0 || vsharp.GetSize() == 0) {
                buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
            } else {
                const u64 size = memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize());
                if (size != vsharp.GetSize()) {
                    LOG_ERROR(Render, "Clamped size from {} to {} for stage {:#x}",
                              vsharp.GetSize(), size, stage.pgm_hash);
                }
                VideoCore::ObtainBufferFlags flags = {};
                if (desc.is_written) {
                    flags |= VideoCore::ObtainBufferFlags::IsWritten;
                }
                if (desc.is_formatted) {
                    flags |= VideoCore::ObtainBufferFlags::IsTexelBuffer;
                }
                VideoCore::BufferCache::SetWriterShader(desc.is_written ? stage.pgm_hash : 0);
                const auto [buffer, offset] =
                    buffer_cache.ObtainBuffer(vsharp.base_address, size, flags);
                VideoCore::BufferCache::SetWriterShader(0);
                const u64 offset_aligned = Common::AlignDown(offset, alignment);
                const u64 adjust = offset - offset_aligned;
                if (adjust % 4 != 0) {
                    LOG_WARNING(Render_Vulkan, "Buffer binding in shader {:#x} isn't dword aligned",
                                stage.pgm_hash);
                }
                push_data.AddOffset(binding.buffer, adjust);
                buffer_infos.emplace_back(buffer->Handle(), offset_aligned, size + adjust);
                if (buffer_index == 0 && stage.pgm_hash == BbSsr::TraceShader) {
                    ssr_constants = {buffer->Handle(), offset_aligned, size + adjust,
                                     u32(adjust / 4)};
                    ssr_guest_constants = reinterpret_cast<const void*>(vsharp.base_address);
                }
                bound_buffers.emplace_back(buffer, offset, size, src_access);
                if (desc.is_written) {
                    // Raw storage-buffer writes can also make an aliased cached image stale.
                    texture_cache.InvalidateMemoryFromGPU(vsharp.base_address, size);
                }
                needs_barrier |= runtime.IsBufferAccessed(buffer, offset, size, desc.is_written);
                if (!desc.is_written && vbao) {
                    vbao->ObserveSceneConstants(reinterpret_cast<const void*>(vsharp.base_address),
                                                size);
                }
                if (!desc.is_written && temporal_dlss.Requested()) {
                    temporal_dlss.ObserveConstants(runtime, instance, buffer, offset, size,
                                                   stage.pgm_hash, binding.unified);
                    temporal_dlss.ObserveSceneConstants(
                        reinterpret_cast<const void*>(vsharp.base_address), size);
                    temporal_dlss.ObserveBlurInputs(
                        reinterpret_cast<const void*>(vsharp.base_address), size, stage.pgm_hash,
                        binding.unified);
                }
            }
        }

        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified++;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType = vk::DescriptorType::eStorageBuffer;
        set_write.pBufferInfo = &buffer_infos.back();
        ++binding.buffer;
    }
}

void Rasterizer::BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding) {
    // The pixel shader is bound first (SwStage::Fragment is stage 0), the vertex shader after.
    const bool pixel_shader = stage.sw_stage == Shader::SwStage::Fragment;
    if (pixel_shader) {
        stage_textures = {};
    }
    if (stage.images.empty()) {
        // Nothing to look up (most vertex shaders, shadow map pixel shaders without alpha test).
        ProfileLap(Section::Textures);
        BindSamplers(stage, binding);
        ProfileLap(Section::Samplers);
        return;
    }
    const u32 first_image_idx = image_infos.size();
    // Better reflections at full resolution: the lighting reads the full-size reflections
    // instead of the game's half-size target.
    const auto replace_reflections = [&] {
        if (!pixel_shader || !ssr || EmulatorSettings.GetBbSsrMode() != 1 ||
            stage.pgm_hash != BbSsr::LightingShader ||
            stage.images.size() <= BbSsr::LightingReflectionSlot ||
            image_infos.size() <= first_image_idx + BbSsr::LightingReflectionSlot) {
            return;
        }
        if (const auto view = ssr->LightingReflections()) {
            image_infos[first_image_idx + BbSsr::LightingReflectionSlot] =
                vk::DescriptorImageInfo{VK_NULL_HANDLE, view, vk::ImageLayout::eGeneral};
        }
    };
    // To emulate storing to explicit mip levels, build a descriptor array with each mip level.
    boost::container::small_vector<u32, 8> image_descriptor_array_sizes;

    const bool memo_on = texture_memo && target_memo_enabled.load(std::memory_order_relaxed);
    const u64 generation = texture_cache.RegistryGeneration();
    // Texture memo: the same T# with the same resource flags finds the same image while the
    // registry generation holds.
    const auto memo_slot = [&](const std::array<u64, 4>& sharp_words,
                               u32 memo_flags) -> TextureMemoEntry* {
        if (!memo_on) {
            return nullptr;
        }
        u64 hash = memo_flags * 0x9E3779B97F4A7C15ull;
        for (const u64 word : sharp_words) {
            hash = (hash ^ word) * 0xFF51AFD7ED558CCDull;
        }
        return &(*texture_memo)[(hash >> 32) % TextureMemoSize];
    };
    const auto memo_hit = [&](const TextureMemoEntry* memo, const std::array<u64, 4>& sharp_words,
                              u32 memo_flags) {
        return memo && memo->valid && memo->flags == memo_flags && memo->generation == generation &&
               memo->sharp == sharp_words;
    };
    const auto finish_binding = [&](VideoCore::ImageId& image_id,
                                    const Shader::ImageResource& image_desc) {
        auto* image = &texture_cache.GetImage(image_id);
        if (auto depth_image_id = texture_cache.GetAssociatedDepth(*image)) {
            // If this image has an associated depth image, it's a stencil attachment.
            // Redirect the access to the actual depth-stencil buffer.
            image_id = depth_image_id;
            image = &texture_cache.GetImage(image_id);
        }
        if (image->binding.is_bound) {
            // The image is already bound. In case if it is about to be used as storage we
            // need to force general layout on it.
            image->binding.force_general |= image_desc.is_written;
        }
        image->binding.is_bound = 1u;
        if (image_desc.exact_texel_reads && !image_desc.is_written) {
            texture_cache.NoteExactTexelReads(image_id, image_desc.exact_texel_reads,
                                              stage.pgm_hash);
        }
    };

    // Texture set memo: the whole stage's sampled textures as resolved last time.
    TextureSet* set_slot = nullptr;
    const u32 set_count = static_cast<u32>(stage.images.size());
    if (memo_on && texture_sets && EmulatorSettings.IsBbTextureSetMemo() && set_count != 0 &&
        set_count <= TextureSet::MaxImages &&
        std::ranges::none_of(stage.images, [](const Shader::ImageResource& image) {
            return image.is_written ||
                   image.mip_fallback_mode != Shader::MipStorageFallbackMode::None;
        })) {
        std::array<std::array<u64, 4>, TextureSet::MaxImages> sharps;
        for (u32 i = 0; i < set_count; ++i) {
            const auto tsharp = stage.images[i].GetSharp(stage);
            std::memcpy(sharps[i].data(), &tsharp, sizeof(sharps[i]));
        }
        const u64 key = XXH3_64bits_withSeed(sharps.data(), set_count * sizeof(sharps[0]),
                                             reinterpret_cast<u64>(&stage)) |
                        1;
        auto& set = (*texture_sets)[key % TextureSetCount];
        set_slot = &set;
        bool hit = set.key == key && set.stage == &stage && set.count == set_count &&
                   set.generation == generation &&
                   std::equal(sharps.begin(), sharps.begin() + set_count, set.sharps.begin());
        if (hit) {
            // New contents of several of the set's images are uploaded together.
            std::array<VideoCore::ImageId, TextureSet::MaxImages> refresh_ids;
            for (u32 i = 0; i < set_count; ++i) {
                refresh_ids[i] = set.entries[i].id;
            }
            if (EmulatorSettings.IsBbBatchedUploads()) {
                texture_cache.UpdateImages(std::span{refresh_ids.data(), set_count});
            }
            // Every image still has the backing its view belongs to.
            for (u32 i = 0; i < set_count && hit; ++i) {
                const auto& entry = set.entries[i];
                if (!entry.id) {
                    continue;
                }
                // What FindTexture does first: returns at once for an image already found
                // clean this GC tick, else refreshes it (as the full path would here).
                texture_cache.UpdateTexture(entry.id);
                const auto& image = texture_cache.GetImage(entry.id);
                hit = image.backing == entry.backing && !image.detail_pending &&
                      !image.binding.needs_rebind && !image.binding.is_target &&
                      !image.binding.force_general &&
                      set.generation == texture_cache.RegistryGeneration();
            }
        }
        if (!hit) {
            texture_set_misses.fetch_add(1, std::memory_order_relaxed);
            set.key = key;
            set.stage = &stage;
            set.count = 0;
            set.generation = ~0ull;
            std::copy_n(sharps.begin(), set_count, set.sharps.begin());
        } else {
            texture_set_hits.fetch_add(1, std::memory_order_relaxed);
            for (u32 i = 0; i < set_count; ++i) {
                const auto& entry = set.entries[i];
                if (!entry.id) {
                    image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE,
                                             vk::ImageLayout::eGeneral);
                } else {
                    texture_cache.TouchFoundImage(entry.id);
                    auto& image = texture_cache.GetImage(entry.id);
                    image.binding.is_bound = 1u;
                    bound_images.emplace_back(entry.id);
                    if (i < stage_textures.size() && pixel_shader) {
                        stage_textures[i] = entry.id;
                    }
                    const auto new_layout = image.info.props.is_depth
                                                ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                                : vk::ImageLayout::eShaderReadOnlyOptimal;
                    needs_barrier |= runtime.Transit(
                        &image, new_layout, vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eShaderRead, entry.view_info.range);
                    image.usage.texture = 1u;
                    temporal_dlss.ObserveTexture(image, entry.id, entry.view_info, stage.pgm_hash,
                                                 i);
                    if (i == 0 && first_texture_infos.size() < first_texture_infos.capacity()) {
                        first_texture_infos.emplace_back(stage.pgm_hash, u32(image_infos.size()));
                    }
                    image_infos.emplace_back(VK_NULL_HANDLE, entry.view,
                                             image.backing->state.layout);
                }
                auto& set_write = set_writes[set_write_index++];
                set_write.dstSet = VK_NULL_HANDLE;
                set_write.dstBinding = binding.unified++;
                set_write.dstArrayElement = 0;
                set_write.descriptorCount = 1;
                set_write.descriptorType = vk::DescriptorType::eSampledImage;
                set_write.pImageInfo = &image_infos[first_image_idx + i];
            }
            replace_reflections();
            ProfileLap(Section::TexSetHit);
            BindSamplers(stage, binding);
            ProfileLap(Section::Samplers);
            return;
        }
    }
    // What a remembered set will hold; only built when this set can be remembered.
    std::optional<std::array<TextureSetEntry, TextureSet::MaxImages>> resolved;
    if (set_slot) {
        resolved.emplace();
    }
    bool set_ok = set_slot != nullptr;

    u32 num_images{};
    // The shader's image resource each binding comes from (new storage image statistics).
    std::array<const Shader::ImageResource*, Shader::NUM_IMAGES> binding_res{};
    for (const auto& image_desc : stage.images) {
        const auto tsharp = image_desc.GetSharp(stage);
        const Shader::MipStorageFallbackMode mip_fallback_mode = image_desc.mip_fallback_mode;
        std::array<u64, 4> sharp_words{};
        static_assert(sizeof(tsharp) == sizeof(sharp_words));
        std::memcpy(sharp_words.data(), &tsharp, sizeof(sharp_words));
        const u32 base_flags = (u32(image_desc.is_depth) << 0) | (u32(image_desc.is_array) << 1) |
                               (u32(image_desc.is_written) << 2) |
                               (u32(image_desc.is_atomic) << 3) | (u32(image_desc.is_r128) << 4) |
                               (u32(mip_fallback_mode) << 5) |
                               (u32(image_desc.constant_mip_index) << 8) |
                               (u32(image_desc.post_op) << 24) | 0x80000000u;

        // A remembered single binding skips the checks below: they gave the same answer for
        // this T# when it was remembered.
        if (mip_fallback_mode != Shader::MipStorageFallbackMode::DynamicIndex) {
            const auto* memo = memo_slot(sharp_words, base_flags);
            if (memo_hit(memo, sharp_words, base_flags)) {
                image_memo_hits.fetch_add(1, std::memory_order_relaxed);
                binding_res[num_images] = &image_desc;
                auto& [image_id, desc] = image_bindings[num_images++];
                image_id = memo->image_id;
                desc = memo->desc;
                texture_cache.TouchFoundImage(image_id);
                finish_binding(image_id, image_desc);
                image_descriptor_array_sizes.push_back(1);
                continue;
            }
        }

        if (texture_cache.IsMeta(tsharp.Address())) {
            // Which texture and shader (each pair once, up to 32 lines), then the plain warning.
            static std::mutex meta_mutex;
            static std::vector<std::pair<VAddr, u64>> meta_seen;
            bool detail = false;
            {
                std::scoped_lock lock{meta_mutex};
                const std::pair key{tsharp.Address(), stage.pgm_hash};
                if (meta_seen.size() < 32 && std::ranges::find(meta_seen, key) == meta_seen.end()) {
                    meta_seen.push_back(key);
                    detail = true;
                }
            }
            if (detail) {
                const auto meta = texture_cache.GetMetaInfo(tsharp.Address());
                std::string owner = "unknown owner";
                if (meta) {
                    const auto& image = texture_cache.GetImage(meta->owner_id);
                    owner = fmt::format(
                        "{} of the {}x{} {} at {:#x}", magic_enum::enum_name(meta->type),
                        image.info.size.width, image.info.size.height,
                        vk::to_string(image.info.pixel_format), image.info.guest_address);
                }
                LOG_WARNING(Render_Vulkan,
                            "Unexpected metadata read by a shader (texture): {:#x} ({}x{}, data "
                            "format {}) read by shader {:#x} is {}",
                            tsharp.Address(), tsharp.width + 1, tsharp.height + 1,
                            static_cast<u32>(tsharp.GetDataFmt()), stage.pgm_hash, owner);
            }
            LOG_WARNING(Render_Vulkan, "Unexpected metadata read by a shader (texture)");
        }

        const auto data_fmt = tsharp.GetDataFmt();
        const auto num_fmt = tsharp.GetNumberFmt();
        if (tsharp.Address() == 0 || data_fmt == AmdGpu::DataFormat::FormatInvalid) {
            binding_res[num_images] = &image_desc;
            image_bindings[num_images++].image_id = {};
            image_descriptor_array_sizes.push_back(1);
            continue;
        }

        // magic_enum::enum_contains scans the enumerators: a table per format enum instead.
        static const auto valid_data_fmt = [] {
            std::array<bool, 256> table{};
            for (const auto value : magic_enum::enum_values<AmdGpu::DataFormat>()) {
                if (static_cast<u32>(value) < table.size()) {
                    table[static_cast<u32>(value)] = true;
                }
            }
            return table;
        }();
        static const auto valid_num_fmt = [] {
            std::array<bool, 256> table{};
            for (const auto value : magic_enum::enum_values<AmdGpu::NumberFormat>()) {
                if (static_cast<u32>(value) < table.size()) {
                    table[static_cast<u32>(value)] = true;
                }
            }
            return table;
        }();
        const auto known = [](const auto& table, auto value) {
            const auto index = static_cast<u32>(value);
            return index < table.size() && table[index];
        };
        if (!memory->IsValidGpuMapping(tsharp.Address(), 0) || !known(valid_data_fmt, data_fmt) ||
            !known(valid_num_fmt, num_fmt)) {
            LOG_WARNING(Render_Vulkan,
                        "Rejecting invalid T# address={:#x}, pitch={}, width={}, "
                        "data_format={}, num_format={}",
                        tsharp.Address(), tsharp.pitch, tsharp.width, static_cast<u32>(data_fmt),
                        static_cast<u32>(num_fmt));
            binding_res[num_images] = &image_desc;
            image_bindings[num_images++].image_id = {};
            image_descriptor_array_sizes.push_back(1);
            continue;
        }

        const u32 num_bindings = image_desc.NumBindings(stage);

        for (auto i = 0; i < num_bindings; i++) {
            const u32 memo_flags = base_flags | (u32(i) << 16);
            TextureMemoEntry* memo = memo_slot(sharp_words, memo_flags);
            const bool hit = memo_hit(memo, sharp_words, memo_flags);
            binding_res[num_images] = &image_desc;
            auto& [image_id, desc] = image_bindings[num_images++];
            if (hit) {
                image_id = memo->image_id;
                desc = memo->desc;
                image_memo_hits.fetch_add(1, std::memory_order_relaxed);
                texture_cache.TouchFoundImage(image_id);
            } else {
                image_id = {};
                std::construct_at(&desc, tsharp, image_desc);
                if (mip_fallback_mode == Shader::MipStorageFallbackMode::ConstantIndex) {
                    ASSERT(num_bindings == 1);
                    desc.view_info.range.base.level += image_desc.constant_mip_index;
                    desc.view_info.range.extent.levels = 1;
                } else if (mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex) {
                    desc.view_info.range.base.level += i;
                    desc.view_info.range.extent.levels = 1;
                }

                image_id = texture_cache.FindImage(desc);
                if (memo) {
                    image_memo_misses.fetch_add(1, std::memory_order_relaxed);
                    *memo = {sharp_words, memo_flags, true, generation, image_id, desc};
                }
            }
            finish_binding(image_id, image_desc);
        }

        image_descriptor_array_sizes.push_back(num_bindings);
    }

    ProfileLap(Section::TexLookup);
    // Sampled images whose contents changed are uploaded together (Batched texture uploads).
    {
        std::array<VideoCore::ImageId, 32> refresh_ids;
        u32 num_refresh = 0;
        for (u32 i = 0; i < num_images && num_refresh < refresh_ids.size(); ++i) {
            const auto& [image_id, desc] = image_bindings[i];
            if (image_id && desc.type != VideoCore::TextureCache::BindingType::Storage) {
                refresh_ids[num_refresh++] = image_id;
            }
        }
        if (num_refresh > 1 && EmulatorSettings.IsBbBatchedUploads()) {
            texture_cache.UpdateImages(std::span{refresh_ids.data(), num_refresh});
        }
    }
    // Second pass to re-bind images that were updated after binding
    for (u32 i = 0; i < num_images; ++i) {
        auto& [image_id, desc] = image_bindings[i];
        const u32 slot = i;
        bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        if (!image_id) {
            image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        } else {
            if (auto& old_image = texture_cache.GetImage(image_id);
                old_image.binding.needs_rebind) {
                old_image.binding = {};
                image_id = texture_cache.FindImage(desc);
                set_ok = false;
            }

            bound_images.emplace_back(image_id);
            if (i < stage_textures.size() && pixel_shader) {
                stage_textures[i] = image_id;
            }

            auto& image = texture_cache.GetImage(image_id);
            if (is_storage && binding_res[i] && image.backing &&
                image.backing->state.layout == vk::ImageLayout::eUndefined &&
                image.backing->subresource_states.empty() &&
                True(image.flags & VideoCore::ImageFlagBits::CpuDirty)) {
                // Uploaded from guest memory below, just before the shader writes it (or
                // cleared instead, when the shader overwrites all of it).
                if (NoteNewStorageImage(image, desc.view_info, *binding_res[i], stage) &&
                    EmulatorSettings.IsBbClearShaderWritten() && image.info.num_samples == 1 &&
                    !image.info.props.is_depth && !image.info.props.has_stencil &&
                    !image.info.props.is_block) {
                    texture_cache.ClearInsteadOfUpload(image);
                }
            }
            auto& image_view = texture_cache.FindTexture(image_id, desc);
            const auto binding = image.binding;
            ProfileLap(Section::TexView);

            // The image is either bound as storage in a separate descriptor or bound as render
            // target in feedback loop. Depth images are excluded because they can't be bound as
            // storage and feedback loop doesn't make sense for them
            if ((binding.force_general || binding.is_target) && !image.info.props.is_depth) {
                if (instance.IsAttachmentFeedbackLoopLayoutSupported() && image.binding.is_target) {
                    needs_barrier |= runtime.Transit(
                        &image, vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT,
                        vk::PipelineStageFlagBits2::eAllGraphics, vk::AccessFlagBits2::eShaderRead);
                } else {
                    needs_barrier |= runtime.Transit(
                        &image, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                        desc.view_info.range);
                }
            } else {
                if (is_storage) {
                    needs_barrier |= runtime.Transit(
                        &image, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                        desc.view_info.range);
                } else {
                    const auto new_layout = image.info.props.is_depth
                                                ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                                : vk::ImageLayout::eShaderReadOnlyOptimal;
                    needs_barrier |= runtime.Transit(
                        &image, new_layout, vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eShaderRead, desc.view_info.range);
                }
            }
            image.usage.storage |= is_storage;
            image.usage.texture |= !is_storage;
            if (!is_storage) {
                temporal_dlss.ObserveTexture(image, image_id, desc.view_info, stage.pgm_hash, slot);
            }
            if (slot == 0 && first_texture_infos.size() < first_texture_infos.capacity()) {
                first_texture_infos.emplace_back(stage.pgm_hash, u32(image_infos.size()));
            }

            image_infos.emplace_back(VK_NULL_HANDLE, *image_view.image_view,
                                     image.backing->state.layout);
            if (set_ok && slot < TextureSet::MaxImages) {
                (*resolved)[slot] = {image_id, *image_view.image_view, image.backing,
                                     desc.view_info};
                // (Not while sampled at lower detail: its full view comes later.)
                set_ok = !is_storage && !binding.force_general && !binding.is_target &&
                         !image.detail_pending;
            }
            ProfileLap(Section::TexLayout);
        }
    }
    if (set_slot) {
        if (set_ok && num_images == set_count) {
            // Remembered: the same T#s resolve the same way while no image is (un)registered.
            set_slot->generation = texture_cache.RegistryGeneration();
            set_slot->count = set_count;
            std::copy_n(resolved->begin(), set_count, set_slot->entries.begin());
        } else {
            set_slot->key = 0;
        }
    }

    u32 image_info_idx = first_image_idx;
    u32 image_binding_idx = 0;
    for (u32 array_size : image_descriptor_array_sizes) {
        const auto& [_, desc] = image_bindings[image_binding_idx];
        const bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = array_size;
        set_write.descriptorType =
            is_storage ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eSampledImage;
        set_write.pImageInfo = &image_infos[image_info_idx];

        image_info_idx += array_size;
        image_binding_idx += array_size;
        binding.unified += array_size;
    }
    replace_reflections();
    ProfileLap(Section::Textures);
    BindSamplers(stage, binding);
    ProfileLap(Section::Samplers);
}

void Rasterizer::BindSamplers(const Shader::Info& stage, Shader::Backend::Bindings& binding) {
    for (const auto& sampler : stage.samplers) {
        auto ssharp = sampler.GetSharp(stage);
        if (!ssharp.Valid() || (ssharp.border_color_type.Value() == AmdGpu::BorderColor::Custom &&
                                Regs().ta_bc_base.Address() == 0)) {
            LOG_WARNING(Render_Vulkan,
                        "Rejecting invalid S# max_aniso={}, filter_mode={}, mip_filter={}, "
                        "border_color_type={}, border_color_base={:#x}",
                        static_cast<u32>(ssharp.max_aniso.Value()),
                        static_cast<u32>(ssharp.filter_mode.Value()),
                        static_cast<u32>(ssharp.mip_filter.Value()),
                        static_cast<u32>(ssharp.border_color_type.Value()),
                        Regs().ta_bc_base.Address());
            ssharp = AmdGpu::Sampler{};
        }
        vk::Sampler vk_sampler{};
        // Upscaler LOD bias of this draw's G-buffer textures (not for compare samplers).
        const float lod_bias = sampler.is_depth ? 0.0f : scene_lod_bias;
        std::array<u64, 2> sharp_words{};
        static_assert(sizeof(ssharp) == sizeof(sharp_words));
        std::memcpy(sharp_words.data(), &ssharp, sizeof(sharp_words));
        SamplerMemoEntry* memo = nullptr;
        if (sampler_memo && target_memo_enabled.load(std::memory_order_relaxed)) {
            const u64 hash = ((sharp_words[0] ^ (sharp_words[1] * 0x9E3779B97F4A7C15ull)) ^
                              u64(sampler.is_depth) ^ (u64(std::bit_cast<u32>(lod_bias)) << 1)) *
                             0xFF51AFD7ED558CCDull;
            memo = &(*sampler_memo)[(hash >> 40) % SamplerMemoSize];
        }
        const u64 sampler_generation = texture_cache.SamplerGeneration();
        if (memo && memo->valid && memo->sharp == sharp_words &&
            memo->is_depth == sampler.is_depth && memo->lod_bias == lod_bias &&
            memo->generation == sampler_generation) {
            vk_sampler = memo->handle;
            // Keep it alive: GetSampler's LRU touch, once per garbage collection tick.
            if (memo->touched_gc_tick != texture_cache.GcTick()) {
                texture_cache.TouchSampler(ssharp, sampler.is_depth, lod_bias);
                memo->touched_gc_tick = texture_cache.GcTick();
            }
        } else {
            vk_sampler =
                texture_cache.GetSampler(ssharp, Regs().ta_bc_base, sampler.is_depth, lod_bias);
            if (memo) {
                *memo = {sharp_words,        sampler.is_depth,       lod_bias,  true,
                         sampler_generation, texture_cache.GcTick(), vk_sampler};
            }
        }
        image_infos.emplace_back(vk_sampler, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified++;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType = vk::DescriptorType::eSampler;
        set_write.pImageInfo = &image_infos.back();
    }
}

Rasterizer::TargetSignature Rasterizer::MakeTargetSignature(
    const GraphicsPipeline* pipeline) const {
    const auto& key = pipeline->GetGraphicsKey();
    const auto& regs = Regs();
    TargetSignature signature{};
    const u32 num_color = std::min<u32>(std::bit_width(key.mrt_mask), AmdGpu::NUM_COLOR_BUFFERS);
    for (u32 cb = 0; cb < num_color; ++cb) {
        signature.ids[cb] = cb_descs[cb].image_id;
        if (cb_descs[cb].image_id) {
            signature.views[cb] = cb_descs[cb].desc.view_info;
        }
    }
    signature.ids[AmdGpu::NUM_COLOR_BUFFERS] = db_desc.image_id;
    if (db_desc.image_id) {
        signature.views[AmdGpu::NUM_COLOR_BUFFERS] = db_desc.desc.view_info;
    }
    signature.color_samples = key.color_samples;
    signature.mrt_mask = key.mrt_mask;
    std::memcpy(&signature.depth_control, &regs.depth_control, sizeof(u32));
    signature.depth_valid = regs.depth_buffer.DepthValid();
    signature.stencil_valid = regs.depth_buffer.StencilValid();
    return signature;
}

bool Rasterizer::TargetMemoUsable(const TargetSignature& signature) {
    const auto& regs = Regs();
    if (!target_memo.valid || !scheduler.IsRenderingWith(target_memo.state) ||
        regs.depth_render_control.depth_clear_enable ||
        regs.depth_render_control.stencil_clear_enable || !(signature == target_memo.signature)) {
        return false;
    }
    // Nothing may have happened to the targets that the full path would act on: a pending
    // upload, a rebind, a feedback loop with this draw's textures, an unregistered image.
    for (const auto id : signature.ids) {
        if (!id) {
            continue;
        }
        const auto& image = texture_cache.GetImage(id);
        if (True(image.flags & VideoCore::ImageFlagBits::Dirty) ||
            False(image.flags & VideoCore::ImageFlagBits::Registered) || image.binding.is_bound ||
            image.binding.needs_rebind) {
            return false;
        }
    }
    return true;
}

RenderState Rasterizer::BeginRendering(const GraphicsPipeline* pipeline) {
    if (!target_memo_enabled.load(std::memory_order_relaxed)) {
        target_memo.valid = false;
        return BeginRenderingFull(pipeline);
    }
    const auto signature = MakeTargetSignature(pipeline);
    if (TargetMemoUsable(signature)) {
        target_memo_hits.fetch_add(1, std::memory_order_relaxed);
        attachment_feedback_loop = false;
        return target_memo.state;
    }
    target_memo_misses.fetch_add(1, std::memory_order_relaxed);
    target_memo.valid = false;
    RenderState state = BeginRenderingFull(pipeline);
    // States that clear a target differ for the next draw; feedback loops are not remembered.
    bool clears =
        state.depth_stencil_attachment.depth_clear || state.depth_stencil_attachment.stencil_clear;
    for (u32 i = 0; i < state.num_color_attachments; ++i) {
        clears |= state.color_attachments[i].is_clear != 0;
    }
    if (!clears && !attachment_feedback_loop) {
        target_memo = {true, signature, state};
    }
    return state;
}

RenderState Rasterizer::BeginRenderingFull(const GraphicsPipeline* pipeline) {
    attachment_feedback_loop = false;
    const auto& regs = Regs();
    const auto& key = pipeline->GetGraphicsKey();
    RenderState state;
    state.width = instance.GetMaxFramebufferWidth();
    state.height = instance.GetMaxFramebufferHeight();
    state.num_layers = std::numeric_limits<u16>::max();
    state.num_color_attachments = std::bit_width(key.mrt_mask);
    bool has_attachment = false;
    for (auto cb = 0u; cb < state.num_color_attachments; ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        if (!image_id) {
            state.color_attachments[cb] = {};
            continue;
        }
        has_attachment = true;
        auto* image = &texture_cache.GetImage(image_id);
        if (image->binding.needs_rebind) {
            image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
            image = &texture_cache.GetImage(image_id);
        }
        {
            Common::GuestStats::UploadPathScope path{Common::GuestStats::UploadPath::Target};
            texture_cache.UpdateImage(image_id);
        }
        runtime.SetBackingSamples(image, key.color_samples[cb]);
        const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
        const auto slice = image_view.info.range.base.layer;
        const auto mip = image_view.info.range.base.level;
        if (BlurWatched(image->info.guest_address)) [[unlikely]] {
            LOG_INFO(Render_Vulkan,
                     "[BB-BLUR] render pass into {:#x} (target {}): image {} (backing {}), "
                     "wanted view {}, view {} created as {} | pipeline {} {} | {}x{} levels {}-{} "
                     "layers {}-{}",
                     image->info.guest_address, cb, vk::to_string(image->info.pixel_format),
                     fmt::ptr(static_cast<VkImage>(image->GetImage())),
                     vk::to_string(desc.view_info.format), vk::to_string(image_view.info.format),
                     vk::to_string(instance.GetSupportedFormat(image_view.info.format,
                                                               image->format_features)),
                     magic_enum::enum_name(key.color_buffers[cb].data_format),
                     magic_enum::enum_name(key.color_buffers[cb].num_format),
                     image->info.size.width, image->info.size.height, mip,
                     mip + image_view.info.range.extent.levels - 1, slice,
                     slice + image_view.info.range.extent.layers - 1);
        }

        const auto& col_buf = regs.color_buffers[cb];
        const bool is_clear = texture_cache.IsMetaCleared(col_buf.CmaskAddress(), slice);
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);

        if (image->binding.is_bound) {
            ASSERT_MSG(!image->binding.force_general,
                       "Having image both as storage and render target is unsupported");
            runtime.FlushBarriers();
            needs_barrier |=
                runtime.Transit(image,
                                instance.IsAttachmentFeedbackLoopLayoutSupported()
                                    ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                                    : vk::ImageLayout::eGeneral,
                                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                vk::AccessFlagBits2::eColorAttachmentWrite |
                                    vk::AccessFlagBits2::eColorAttachmentRead);
            attachment_feedback_loop = true;
        } else {
            needs_barrier |= runtime.Transit(image, vk::ImageLayout::eColorAttachmentOptimal,
                                             vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                             vk::AccessFlagBits2::eColorAttachmentWrite |
                                                 vk::AccessFlagBits2::eColorAttachmentRead,
                                             desc.view_info.range);
        }

        state.width = std::min<u32>(state.width, std::max(image->info.size.width >> mip, 1u));
        state.height = std::min<u32>(state.height, std::max(image->info.size.height >> mip, 1u));
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        const auto clear_value =
            is_clear ? LiverpoolToVK::ColorBufferClearValue(col_buf) : vk::ClearValue{};
        auto& attachment = state.color_attachments[cb];
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image->backing->state.layout;
        attachment.clear_value = clear_value.color.uint32;
        attachment.is_clear = is_clear;

        image->usage.render_target = 1u;
    }
    for (u32 cb = state.num_color_attachments; cb < state.color_attachments.size(); ++cb) {
        state.color_attachments[cb] = {};
    }

    if (auto image_id = db_desc.image_id; image_id) {
        has_attachment = true;
        auto& desc = db_desc.desc;
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& image_view = texture_cache.FindDepthTarget(image_id, desc);
        auto& image = texture_cache.GetImage(image_id);

        const auto slice = image_view.info.range.base.layer;
        const bool is_depth_clear =
            (regs.depth_render_control.depth_clear_enable && regs.depth_control.depth_enable &&
             regs.depth_control.depth_write_enable) ||
            texture_cache.IsMetaCleared(htile_address, slice);
        const bool is_stencil_clear = regs.depth_render_control.stencil_clear_enable;
        texture_cache.TouchMeta(htile_address, slice, false);
        ASSERT(desc.view_info.range.extent.levels == 1 && !image.binding.needs_rebind);

        const bool has_stencil = image.info.props.has_stencil;
        // Stencil writes can be enabled while depth writes are off.
        const bool stencil_write =
            has_stencil && regs.depth_control.stencil_enable && !desc.view_info.is_storage;
        const auto new_layout = desc.view_info.is_storage
                                    ? has_stencil ? vk::ImageLayout::eDepthStencilAttachmentOptimal
                                                  : vk::ImageLayout::eDepthAttachmentOptimal
                                : stencil_write
                                    ? vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal
                                : has_stencil ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                              : vk::ImageLayout::eDepthReadOnlyOptimal;
        needs_barrier |= runtime.Transit(&image, new_layout,
                                         vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                                             vk::PipelineStageFlagBits2::eLateFragmentTests,
                                         vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
                                             vk::AccessFlagBits2::eDepthStencilAttachmentRead,
                                         desc.view_info.range);

        state.width = std::min<u32>(state.width, image.info.size.width);
        state.height = std::min<u32>(state.height, image.info.size.height);
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        auto& attachment = state.depth_stencil_attachment;
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image.backing->state.layout;
        attachment.clear_value = {};
        attachment.is_clear = 0;

        if (regs.depth_buffer.DepthValid()) {
            attachment.clear_value[0] = is_depth_clear ? std::bit_cast<u32>(regs.depth_clear) : 0u;
            attachment.has_depth = true;
            attachment.depth_clear = is_depth_clear;
        }
        if (regs.depth_buffer.StencilValid()) {
            attachment.clear_value[1] = is_stencil_clear ? regs.stencil_clear : 0u;
            attachment.has_stencil = true;
            attachment.stencil_clear = is_stencil_clear;
        }

        image.usage.depth_target = true;
    } else {
        state.depth_stencil_attachment = {};
    }

    // Without attachments the area stays at the max framebuffer size. Draws cannot pass the
    // scissors, so bound it by them.
    if (!has_attachment) {
        state.num_layers = 1;
        const AmdGpu::Scissor scissor = regs.GetCombinedScissor();
        if (scissor.bottom_right_x > 0 && scissor.bottom_right_y > 0) {
            state.width = std::min<u32>(state.width, scissor.bottom_right_x);
            state.height = std::min<u32>(state.height, scissor.bottom_right_y);
        }
    }
    // Character motion image (part of the remembered state, so the pass stays open).
    if (key.motion_slot && object_motion->Enabled()) {
        object_motion->Attach(state);
    }

    return state;
}

void Rasterizer::Resolve() {
    const auto& mrt0_hint = CbExtent(0);
    const auto& mrt1_hint = CbExtent(1);
    VideoCore::TextureCache::ImageDesc mrt0_desc{Regs().color_buffers[0], mrt0_hint};
    VideoCore::TextureCache::ImageDesc mrt1_desc{Regs().color_buffers[1], mrt1_hint};
    auto& mrt0_image = texture_cache.GetImage(texture_cache.FindImage(mrt0_desc, true));
    auto& mrt1_image = texture_cache.GetImage(texture_cache.FindImage(mrt1_desc, true));

    ScopeMarkerBegin(fmt::format("Resolve:MRT0={:#x}:MRT1={:#x}", Regs().color_buffers[0].Address(),
                                 Regs().color_buffers[1].Address()));
    runtime.ResolveImage(&mrt0_image, &mrt1_image, mrt0_desc.view_info.range,
                         mrt1_desc.view_info.range);
    ScopeMarkerEnd();
}

void Rasterizer::DepthStencilCopy(bool is_depth, bool is_stencil) {
    auto& regs = Regs();

    auto read_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), DbExtent(), false);
    auto write_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), DbExtent(), true);

    auto& read_image = texture_cache.GetImage(texture_cache.FindImage(read_desc));
    const auto write_id = texture_cache.FindImage(write_desc);
    auto& write_image = texture_cache.GetImage(write_id);
    // The copy destination is never bound as a target, so its stencil plane is linked here.
    texture_cache.AssociateStencil(write_id, write_desc.info);

    VideoCore::SubresourceRange sub_range;
    sub_range.base.layer = Regs().depth_view.slice_start;
    sub_range.extent.layers = Regs().depth_view.NumSlices() - sub_range.base.layer;

    ScopeMarkerBegin(fmt::format(
        "DepthStencilCopy:DR={:#x}:SR={:#x}:DW={:#x}:SW={:#x}", regs.depth_buffer.DepthAddress(),
        regs.depth_buffer.StencilAddress(), regs.depth_buffer.DepthWriteAddress(),
        regs.depth_buffer.StencilWriteAddress()));

    runtime.CopyDepthStencil(&read_image, &write_image, sub_range);

    ScopeMarkerEnd();
}

void Rasterizer::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    DrainDrawPipe();
    ASSERT_MSG(address % 4 == 0 && num_bytes % 4 == 0,
               "FillBuffer address and size must be a multiple of 4 bytes");
    if (!is_gds) {
        texture_cache.ClearMeta(address);
        if (!buffer_cache.IsRegionGpuModified(address, num_bytes)) {
            u32* buffer = std::bit_cast<u32*>(address);
            std::fill(buffer, buffer + (num_bytes / sizeof(u32)), value);
            return;
        }
    }
    const auto [buffer, offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
        if (is_gds) {
            return {buffer_cache.GetGdsBuffer(), address};
        }
        return buffer_cache.ObtainBuffer(address, num_bytes,
                                         VideoCore::ObtainBufferFlags::IsWritten);
    }();
    runtime.FillBuffer(buffer, offset, num_bytes, value);
}

void Rasterizer::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    DrainDrawPipe();
    if (!dst_gds && !buffer_cache.IsRegionGpuModified(dst, num_bytes)) {
        if (!src_gds && !buffer_cache.IsRegionGpuModified(src, num_bytes) &&
            !texture_cache.FindImageFromRange(src, num_bytes)) {
            // Both buffers were not transferred to GPU yet. Can safely copy in host memory.
            std::memcpy(std::bit_cast<void*>(dst), std::bit_cast<void*>(src), num_bytes);
            return;
        }
    }
    texture_cache.InvalidateMemoryFromGPU(dst, num_bytes);
    const auto* gds_buffer = buffer_cache.GetGdsBuffer();
    const auto [src_buffer, src_offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
        if (src_gds) {
            return {gds_buffer, src};
        }
        return buffer_cache.ObtainBuffer(src, num_bytes,
                                         VideoCore::ObtainBufferFlags::IsTexelBuffer);
    }();
    const auto [dst_buffer, dst_offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
        if (dst_gds) {
            return {gds_buffer, dst};
        }
        return buffer_cache.ObtainBuffer(dst, num_bytes,
                                         VideoCore::ObtainBufferFlags::IsWritten |
                                             VideoCore::ObtainBufferFlags::IsTexelBuffer);
    }();
    const vk::BufferCopy copy = {
        .srcOffset = src_offset,
        .dstOffset = dst_offset,
        .size = num_bytes,
    };
    runtime.CopyBuffer(src_buffer, dst_buffer, std::span{&copy, 1});
}

u32 Rasterizer::ReadDataFromGds(u32 gds_offset) {
    DrainDrawPipe();
    auto* gds_buf = buffer_cache.GetGdsBuffer();
    u32 value;
    std::memcpy(&value, gds_buf->mapped_data.data() + gds_offset, sizeof(u32));
    return value;
}

bool Rasterizer::OnWriteFault(VAddr addr, u64 size, bool assume_locks) {
    // A fault of the GPU command thread is handled inline, on state the draw recording thread
    // owns while it runs.
    DrainDrawPipe();
    write_faults.fetch_add(1, std::memory_order_relaxed);
    // (A wider unprotect window around the fault, as bbport does, measured slower here:
    // the extra pages are uploaded again on their next use.)
    return InvalidateMemory(addr, size, assume_locks);
}

bool Rasterizer::InvalidateMemory(VAddr addr, u64 size, bool assume_locks) {
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    {
        const Common::GameWaits::Scope wait{"write fault: buffers"};
        buffer_cache.InvalidateMemory(addr, size, assume_locks);
    }
    {
        const Common::GameWaits::Scope wait{"write fault: textures"};
        texture_cache.InvalidateMemory(addr, size);
    }
    return true;
}

bool Rasterizer::ReadMemory(VAddr addr, u64 size, bool assume_locks) {
    DrainDrawPipe();
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    buffer_cache.ReadMemory(addr, size, false, false, assume_locks);
    return true;
}

bool Rasterizer::IsMapped(VAddr addr, u64 size) {
    if (size == 0) {
        // There is no memory, so not mapped.
        return false;
    }
    if (static_cast<u64>(addr) > std::numeric_limits<u64>::max() - size) {
        // Memory range wrapped the address space, cannot be mapped.
        return false;
    }
    const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);

    Common::RecursiveSharedLock lock{mapped_ranges_mutex};
    return boost::icl::contains(mapped_ranges, range);
}

void Rasterizer::MapMemory(VAddr addr, u64 size) {
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges += decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
}

void Rasterizer::RegisterMemory(VAddr addr, u64 size) {
    page_manager.OnGpuMap(addr, size);
}

void Rasterizer::UnmapMemory(VAddr addr, u64 size) {
    // Background texture copies may still read this memory.
    Common::CopyPool::WaitAsync();
    buffer_cache.InvalidateMemory(addr, size);
    texture_cache.UnmapMemory(addr, size);
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges -= decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
}

void Rasterizer::UpdateDynamicState(const GraphicsPipeline* pipeline, const bool is_indexed) const {
    UpdateViewportScissorState();
    UpdateDepthStencilState();
    UpdatePrimitiveState(is_indexed);
    UpdateRasterizationState();
    UpdateColorBlendingState(pipeline);

    auto& dynamic_state = scheduler.GetDynamicState();
    // Threaded recording: the commands come from a copy; the flags clear here as Commit() would.
    // MarkCommitted() clears exactly the flags Commit() would act on: when it clears none,
    // there is nothing to record and the (~700 byte) copy is skipped.
    const auto dirty_before = dynamic_state.dirty_state;
    dynamic_state.MarkCommitted(instance);
    if (std::memcmp(&dirty_before, &dynamic_state.dirty_state, sizeof(dirty_before)) == 0) {
        return;
    }
    dynamic_state.dirty_state = dirty_before;
    scheduler.Record([snapshot = dynamic_state, &instance = instance](
                         vk::CommandBuffer cmdbuf) mutable { snapshot.Commit(instance, cmdbuf); });
    dynamic_state.MarkCommitted(instance);
}

void Rasterizer::UpdateViewportScissorState() const {
    const auto& regs = Regs();

    const AmdGpu::Scissor scsr = regs.GetCombinedScissor();

    boost::container::static_vector<vk::Viewport, AmdGpu::NUM_VIEWPORTS> viewports;
    boost::container::static_vector<vk::Rect2D, AmdGpu::NUM_VIEWPORTS> scissors;

    if (regs.polygon_control.enable_window_offset &&
        (regs.window_offset.window_x_offset != 0 || regs.window_offset.window_y_offset != 0)) {
        LOG_ERROR(Render_Vulkan,
                  "PA_SU_SC_MODE_CNTL.VTX_WINDOW_OFFSET_ENABLE support is not yet implemented.");
    }

    const auto& vp_ctl = regs.viewport_control;
    for (u32 i = 0; i < AmdGpu::NUM_VIEWPORTS; i++) {
        const auto& vp = regs.viewports[i];
        const auto& vp_d = regs.viewport_depths[i];
        if (vp.xscale == 0) {
            continue;
        }

        const auto zoffset = vp_ctl.zoffset_enable ? vp.zoffset : 0.f;
        const auto zscale = vp_ctl.zscale_enable ? vp.zscale : 1.f;

        vk::Viewport viewport{};

        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_pipeline_graphics.c#L688-689
        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_cmd_buffer.c#L3103-3109
        // When the clip space is ranged [-1...1], the zoffset is centered.
        // By reversing the above viewport calculations, we get the following:
        if (regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW) {
            viewport.minDepth = zoffset - zscale;
            viewport.maxDepth = zoffset + zscale;
        } else {
            viewport.minDepth = zoffset;
            viewport.maxDepth = zoffset + zscale;
        }

        if (!instance.IsDepthRangeUnrestrictedSupported()) {
            // Unrestricted depth range not supported by device. Restrict to valid range.
            viewport.minDepth = std::max(viewport.minDepth, 0.f);
            viewport.maxDepth = std::min(viewport.maxDepth, 1.f);
        }

        if (regs.IsClipDisabled()) {
            // In case if clipping is disabled we patch the shader to convert vertex position
            // from screen space coordinates to NDC by defining a render space as full hardware
            // window range [0..16383, 0..16383] and setting the viewport to its size.
            viewport.x = 0.f;
            viewport.y = 0.f;
            viewport.width = float(std::min<u32>(instance.GetMaxViewportWidth(), 16_KB));
            viewport.height = float(std::min<u32>(instance.GetMaxViewportHeight(), 16_KB));
        } else {
            const auto xoffset = vp_ctl.xoffset_enable ? vp.xoffset : 0.f;
            const auto xscale = vp_ctl.xscale_enable ? vp.xscale : 1.f;
            const auto yoffset = vp_ctl.yoffset_enable ? vp.yoffset : 0.f;
            const auto yscale = vp_ctl.yscale_enable ? vp.yscale : 1.f;

            // Temporal DLSS sub-pixel jitter (zero unless this is a jittered scene draw).
            viewport.x = xoffset - xscale + draw_jitter[0];
            viewport.y = yoffset - yscale + draw_jitter[1];
            viewport.width = xscale * 2.0f;
            viewport.height = yscale * 2.0f;
        }

        viewports.push_back(viewport);

        auto vp_scsr = scsr;
        if (regs.mode_control.vport_scissor_enable) {
            vp_scsr.top_left_x =
                std::max(vp_scsr.top_left_x, s16(regs.viewport_scissors[i].top_left_x));
            vp_scsr.top_left_y =
                std::max(vp_scsr.top_left_y, s16(regs.viewport_scissors[i].top_left_y));
            vp_scsr.bottom_right_x =
                std::min(vp_scsr.bottom_right_x, s16(regs.viewport_scissors[i].bottom_right_x));
            vp_scsr.bottom_right_y =
                std::min(vp_scsr.bottom_right_y, s16(regs.viewport_scissors[i].bottom_right_y));
        }
        scissors.push_back({
            .offset = {vp_scsr.top_left_x, vp_scsr.top_left_y},
            .extent = {vp_scsr.GetWidth(), vp_scsr.GetHeight()},
        });
    }

    if (viewports.empty()) {
        // Vulkan requires providing at least one viewport.
        constexpr vk::Viewport empty_viewport = {
            .x = -1.0f,
            .y = -1.0f,
            .width = 1.0f,
            .height = 1.0f,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        constexpr vk::Rect2D empty_scissor = {
            .offset = {0, 0},
            .extent = {1, 1},
        };
        viewports.push_back(empty_viewport);
        scissors.push_back(empty_scissor);
    }

    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetViewports(viewports);
    dynamic_state.SetScissors(scissors);
}

void Rasterizer::UpdateDepthStencilState() const {
    const auto& regs = Regs();
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto depth_test_enabled =
        regs.depth_control.depth_enable && regs.depth_buffer.DepthValid();
    dynamic_state.SetDepthTestEnabled(depth_test_enabled);
    if (depth_test_enabled) {
        dynamic_state.SetDepthWriteEnabled(regs.depth_control.depth_write_enable &&
                                           !regs.depth_render_control.depth_clear_enable);
        dynamic_state.SetDepthCompareOp(LiverpoolToVK::CompareOp(regs.depth_control.depth_func));
    }

    const auto depth_bounds_test_enabled = regs.depth_control.depth_bounds_enable;
    dynamic_state.SetDepthBoundsTestEnabled(depth_bounds_test_enabled);
    if (depth_bounds_test_enabled) {
        dynamic_state.SetDepthBounds(regs.depth_bounds_min, regs.depth_bounds_max);
    }

    const auto depth_bias_enabled = regs.polygon_control.NeedsBias();
    dynamic_state.SetDepthBiasEnabled(depth_bias_enabled);
    if (depth_bias_enabled) {
        const bool front = regs.polygon_control.enable_polygon_offset_front;
        dynamic_state.SetDepthBias(
            front ? regs.poly_offset.front_offset : regs.poly_offset.back_offset,
            regs.poly_offset.depth_bias,
            (front ? regs.poly_offset.front_scale : regs.poly_offset.back_scale) / 16.f);
    }

    const auto stencil_test_enabled =
        regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid();
    dynamic_state.SetStencilTestEnabled(stencil_test_enabled);
    if (stencil_test_enabled) {
        const StencilOps front_ops{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_front),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_front),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_front),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_ref_func),
        };
        const StencilOps back_ops = regs.depth_control.backface_enable ? StencilOps{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_back),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_back),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_back),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_bf_func),
        } : front_ops;
        dynamic_state.SetStencilOps(front_ops, back_ops);

        const bool stencil_clear = regs.depth_render_control.stencil_clear_enable;
        const auto front = regs.stencil_ref_front;
        const auto back =
            regs.depth_control.backface_enable ? regs.stencil_ref_back : regs.stencil_ref_front;
        // GCN REPLACE_OP writes DB_STENCILREFMASK.STENCILOPVAL, so a face whose stencil ops
        // include ReplaceOp takes its Vulkan reference from op_val.
        const auto& sc = regs.stencil_control;
        const auto uses_op_val = [](AmdGpu::StencilFunc fail, AmdGpu::StencilFunc zpass,
                                    AmdGpu::StencilFunc zfail) {
            return fail == AmdGpu::StencilFunc::ReplaceOp ||
                   zpass == AmdGpu::StencilFunc::ReplaceOp ||
                   zfail == AmdGpu::StencilFunc::ReplaceOp;
        };
        const bool front_op =
            uses_op_val(sc.stencil_fail_front, sc.stencil_zpass_front, sc.stencil_zfail_front);
        const bool back_op =
            regs.depth_control.backface_enable
                ? uses_op_val(sc.stencil_fail_back, sc.stencil_zpass_back, sc.stencil_zfail_back)
                : front_op;
        const auto ref_conflict = [](AmdGpu::CompareFunc func, const AmdGpu::StencilRefMask& ref) {
            return func != AmdGpu::CompareFunc::Always && func != AmdGpu::CompareFunc::Never &&
                   ref.stencil_test_val != ref.stencil_op_val;
        };
        if ((front_op && ref_conflict(regs.depth_control.stencil_ref_func, front)) ||
            (back_op && regs.depth_control.backface_enable &&
             ref_conflict(regs.depth_control.stencil_bf_func, back))) {
            LOG_WARNING(Render_Vulkan, "Stencil test requires test_val while ReplaceOp requires "
                                       "op_val; the stencil test will use op_val");
        }
        dynamic_state.SetStencilReferences(front_op ? front.stencil_op_val : front.stencil_test_val,
                                           back_op ? back.stencil_op_val : back.stencil_test_val);
        dynamic_state.SetStencilWriteMasks(!stencil_clear ? front.stencil_write_mask : 0U,
                                           !stencil_clear ? back.stencil_write_mask : 0U);
        dynamic_state.SetStencilCompareMasks(front.stencil_mask, back.stencil_mask);
    }
}

void Rasterizer::UpdatePrimitiveState(const bool is_indexed) const {
    const auto& regs = Regs();
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto is_list_topology = [](const AmdGpu::PrimitiveType type) {
        const auto topology = LiverpoolToVK::PrimitiveType(type);
        return topology == vk::PrimitiveTopology::ePointList ||
               topology == vk::PrimitiveTopology::eLineList ||
               topology == vk::PrimitiveTopology::eTriangleList ||
               topology == vk::PrimitiveTopology::eLineListWithAdjacency ||
               topology == vk::PrimitiveTopology::eTriangleListWithAdjacency;
    };
    const auto is_patch_list_topology = [](const AmdGpu::PrimitiveType type) {
        // Quad and rect lists are emulated using tessellation.
        return type == AmdGpu::PrimitiveType::PatchPrimitive ||
               type == AmdGpu::PrimitiveType::QuadList || type == AmdGpu::PrimitiveType::RectList;
    };

    const auto prim_restart =
        (regs.enable_primitive_restart & 1) != 0 &&
        (instance.IsListRestartSupported() || !is_list_topology(regs.primitive_type)) &&
        (instance.IsPatchListRestartSupported() || !is_patch_list_topology(regs.primitive_type));
    ASSERT_MSG(!is_indexed || !prim_restart || regs.primitive_restart_index == 0xFFFF ||
                   regs.primitive_restart_index == 0xFFFFFFFF,
               "Primitive restart index other than -1 is not supported yet");

    const auto cull_mode = LiverpoolToVK::IsPrimitiveCulled(regs.primitive_type)
                               ? LiverpoolToVK::CullMode(regs.polygon_control.CullingMode())
                               : vk::CullModeFlagBits::eNone;
    const auto front_face = LiverpoolToVK::FrontFace(regs.polygon_control.front_face);

    dynamic_state.SetPrimitiveRestartEnabled(prim_restart);
    dynamic_state.SetRasterizerDiscardEnabled(regs.clipper_control.dx_rasterization_kill);
    dynamic_state.SetCullMode(cull_mode);
    dynamic_state.SetFrontFace(front_face);
}

void Rasterizer::UpdateRasterizationState() const {
    const auto& regs = Regs();
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetLineWidth(regs.line_control.Width());
}

void Rasterizer::UpdateColorBlendingState(const GraphicsPipeline* pipeline) const {
    const auto& regs = Regs();
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetBlendConstants(regs.blend_constants);
    dynamic_state.SetColorWriteMasks(pipeline->GetGraphicsKey().write_masks);
    dynamic_state.SetAttachmentFeedbackLoopEnabled(attachment_feedback_loop);
}

void Rasterizer::ScopeMarkerBegin(const std::string_view& str, bool from_guest) {
    DrainDrawPipe();
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopeMarkerEnd(bool from_guest) {
    DrainDrawPipe();
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.endDebugUtilsLabelEXT();
}

void Rasterizer::ScopedMarkerInsert(const std::string_view& str, bool from_guest) {
    DrainDrawPipe();
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                         bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
        .color = std::array<f32, 4>(
            {(f32)((color >> 16) & 0xff) / 255.0f, (f32)((color >> 8) & 0xff) / 255.0f,
             (f32)(color & 0xff) / 255.0f, (f32)((color >> 24) & 0xff) / 255.0f})});
}

// ---------------------------------------------------------------------------------------------
// Threaded renderer (vk_draw_pipe.h)
// ---------------------------------------------------------------------------------------------

namespace {

constexpr u32 AlignPacket(u32 size) {
    return (size + 7) & ~7u;
}

struct PacketHeader {
    u8 kind;
    bool is_indexed;
    u64 copy_end; ///< copy thread: the jobs up to this one are done before the packet runs
    bool regs_reset;
    u8 num_stages;
    u32 num_blocks;
    u32 index_offset;
    u32 num_prefetch;
    u64 ring_end;
    const void* pipeline;
    VAddr indirect_address;
    VAddr indirect_count_address;
    u32 indirect_offset;
    u32 indirect_stride;
    u32 indirect_max_count;
    AmdGpu::CbDbExtent db_extent;
    std::array<AmdGpu::CbDbExtent, AmdGpu::NUM_COLOR_BUFFERS> cb_extent;
    AmdGpu::ComputeProgram cs;
    bool vertex_prepared;
    u8 num_vertex_inputs;
    u8 num_vertex_ranges;
    bool motion_prepared;
    bool ring_used; ///< the packet reads constant ring data (retirement bookkeeping)
    u32 index_ring_size;
    VAddr index_ring_address;
    u64 index_ring_offset;
};

/// Bytes of a prepared vertex input section with `count` streams and `ranges` ranges.
constexpr u32 VertexSectionSize(u32 count, u32 ranges) {
    return count * static_cast<u32>(sizeof(vk::VertexInputAttributeDescription2EXT) +
                                    sizeof(vk::VertexInputBindingDescription2EXT) +
                                    sizeof(AmdGpu::Buffer)) +
           AlignPacket(count) + ranges * 24;
}

struct TaskPacket {
    u8 kind;
    u8 pad[3];
    u32 size;
    Rasterizer::OrderedTask task;
    u64 copy_end; ///< as PacketHeader::copy_end
};

struct PacketStage {
    const Shader::Info* info;
    VAddr pgm_base;
    u32 user_data_size;
    u32 flat_size;
};

/// The draw recording thread's registers while it runs a packet.
thread_local const AmdGpu::Regs* stage_b_regs = nullptr;

} // Anonymous namespace

bool Rasterizer::UseDrawPipe() const {
    return draw_pipe && pipe_enabled && !host_markers_enabled && !guest_markers_enabled;
}

void Rasterizer::DrainDrawPipe(u32 reason, std::source_location where) {
    if (!draw_pipe || !DrawPipe::OnStageA() || draw_pipe->Idle()) {
        return;
    }
    const auto before = draw_pipe->drain_time;
    draw_pipe->Drain();
    // Statistics: where the GPU command thread waited for the draw recorder, and how long.
    const u64 key = (u64(where.line()) << 32) | reason;
    auto [it, is_new] = pipe_drain_sites.try_emplace(key);
    if (is_new) {
        it->second.function = where.function_name();
        it->second.line = where.line();
        it->second.reason = reason;
    }
    ++it->second.count;
    it->second.time += draw_pipe->drain_time - before;
}

bool Rasterizer::DrainForSpecialPackets() {
    static const bool drain = [] {
        const char* value = std::getenv("SHADPS4_PIPE_DRAINS");
        return value && value[0] == '1';
    }();
    return drain;
}

bool Rasterizer::IsGpuSideThread() const {
    return DrawPipe::OnStageA() || DrawPipe::OnStageB() ||
           std::this_thread::get_id() == liverpool->GetGpuCommandProcessorThread();
}

const AmdGpu::Regs& Rasterizer::Regs() const {
    if (DrawPipe::OnStageB() && stage_b_regs) [[unlikely]] {
        return *stage_b_regs;
    }
    return liverpool->regs;
}

AmdGpu::CbDbExtent Rasterizer::CbExtent(u32 cb) const {
    return DrawPipe::OnStageB() ? pipe_cb_extent[cb] : liverpool->last_cb_extent[cb];
}

AmdGpu::CbDbExtent Rasterizer::DbExtent() const {
    return DrawPipe::OnStageB() ? pipe_db_extent : liverpool->last_db_extent;
}

const AmdGpu::ComputeProgram& Rasterizer::CsRegs() const {
    return DrawPipe::OnStageB() ? pipe_cs : liverpool->GetCsRegs();
}

void Rasterizer::QueueDraw(PacketKind kind, const Pipeline* pipeline, bool is_indexed,
                           u32 index_offset, const IndirectArgs& args) {
    auto& dirty = liverpool->pipe_dirty;
    const auto& regs = liverpool->regs;
    if (!pipe_synced) {
        // First packet: stage B is idle; start its copy from the live registers.
        draw_pipe->Drain();
        *pipe_regs = regs;
        dirty.Clear();
        pipe_synced = true;
    }

    // Small read-only buffers copied here (constant ring); only for draws.
    prefetch_scratch.clear();
    if (const_ring && pipeline && (kind == PacketKind::Draw || kind == PacketKind::DrawIndirect)) {
        PrefetchBuffers(pipeline);
    }
    if (pipeline) {
        NoteQueuedReads(pipeline);
    }
    ALap(ASection::Prefetch);
    const u32 num_prefetch = static_cast<u32>(prefetch_scratch.size());
    const bool vertex_prepared =
        pipeline && (kind == PacketKind::Draw || kind == PacketKind::DrawIndirect) &&
        PrepareVertexInputs(static_cast<const GraphicsPipeline*>(pipeline));
    ALap(ASection::Vertex);
    const u32 num_vertex = vertex_prepared ? static_cast<u32>(vtx_attributes.size()) : 0;
    const u32 num_vertex_ranges = vertex_prepared ? static_cast<u32>(vtx_ranges.size()) : 0;
    // Character motion inputs of a skinned draw, off the draw recorder.
    const bool motion_prepared =
        pipeline && kind == PacketKind::Draw && object_motion && object_motion->Active() &&
        static_cast<const GraphicsPipeline*>(pipeline)->GetGraphicsKey().motion_vectors;
    // The index buffer, when stage B would copy it (BindIndexBuffer's range).
    RingIndex index_ring{};
    if (kind == PacketKind::Draw && pipeline && is_indexed && const_ring) {
        const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
        const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
        const VAddr base_address = regs.index_base_address.Address<VAddr>();
        const VAddr index_address = base_address + u64(index_offset) * index_size;
        const u64 requested = u64(std::min(regs.num_indices, regs.max_index_size)) * index_size;
        const u64 mapped = base_address != 0 ? memory->ClampRangeSize(index_address, requested) : 0;
        const u32 bytes = static_cast<u32>(Common::AlignDown(mapped, index_size));
        if (bytes != 0) {
            index_ring = {index_address, bytes, RingCopyGeometry(index_address, bytes)};
        }
    }
    bool geometry_in_ring = index_ring.offset != NoRing;
    if (vertex_prepared) {
        for (const auto& range : vtx_ranges) {
            geometry_in_ring |= range.ring != NoRing;
        }
    }
    MotionPrep motion_prep{};
    if (motion_prepared) {
        motion_prep = ComputeMotionPrep(static_cast<const GraphicsPipeline*>(pipeline), is_indexed,
                                        index_offset);
    }
    ALap(ASection::Motion);
    // Storage buffers this draw writes: later prefetches must not read them early.
    if (pipeline) {
        const u64 packet_no = draw_pipe->packets + 1;
        for (const auto* info : pipeline->GetStages()) {
            if (!info) {
                continue;
            }
            for (const auto& desc : info->buffers) {
                if (desc.IsSpecial() || !desc.is_written) {
                    continue;
                }
                const auto vsharp = desc.GetSharp(*info);
                if (vsharp.base_address != 0 && vsharp.GetSize() != 0) {
                    pending_writes.push_back({packet_no, vsharp.base_address,
                                              vsharp.base_address + vsharp.GetSize(),
                                              info->sw_stage == Shader::SwStage::Compute
                                                  ? "compute shader storage write"
                                                  : "graphics shader storage write",
                                              info->pgm_hash, true});
                }
            }
        }
    }

    // The register blocks written since the previous packet.
    pipe_blocks.clear();
    for (u32 word = 0; word < dirty.bits.size(); ++word) {
        for (u64 bits = dirty.bits[word]; bits != 0; bits &= bits - 1) {
            pipe_blocks.push_back(static_cast<u16>(word * 64 + std::countr_zero(bits)));
        }
    }
    const u32 num_blocks = static_cast<u32>(pipe_blocks.size());

    // Each stage's user data as the pipeline selection left it.
    std::array<const Shader::Info*, Shader::MaxStageTypes> infos{};
    u32 num_stages = 0;
    u32 stage_bytes = 0;
    for (const auto* info : pipeline ? pipeline->GetStages() : decltype(pipeline->GetStages()){}) {
        if (!info) {
            continue;
        }
        infos[num_stages++] = info;
        stage_bytes +=
            sizeof(PacketStage) +
            AlignPacket(static_cast<u32>(info->user_data.size() + info->flattened_ud_buf.size()) *
                        sizeof(u32));
    }
    ASSERT(num_stages <= Shader::Info::MaxUdSnapshots);

    constexpr u32 BlockBytes = AmdGpu::RegDirty::BlockWords * sizeof(u32);
    const u32 size = sizeof(PacketHeader) + AlignPacket(num_blocks * sizeof(u16)) +
                     num_blocks * BlockBytes + stage_bytes +
                     num_prefetch * static_cast<u32>(sizeof(PrefetchedBuffer)) +
                     (vertex_prepared ? VertexSectionSize(num_vertex, num_vertex_ranges) : 0) +
                     (motion_prepared ? AlignPacket(sizeof(MotionPrep)) : 0);
    u8* const out = draw_pipe->Begin(size);
    u8* at = out;

    auto& header = *reinterpret_cast<PacketHeader*>(at);
    header = PacketHeader{
        .kind = static_cast<u8>(kind),
        .is_indexed = is_indexed,
        .copy_end = copy_head,
        .regs_reset = dirty.reset,
        .num_stages = static_cast<u8>(num_stages),
        .num_blocks = num_blocks,
        .index_offset = index_offset,
        .num_prefetch = num_prefetch,
        .ring_end = ring_head,
        .pipeline = pipeline,
        .indirect_address = args.address,
        .indirect_count_address = args.count_address,
        .indirect_offset = args.offset,
        .indirect_stride = args.stride,
        .indirect_max_count = args.max_count,
        .db_extent = liverpool->last_db_extent,
        .cb_extent = liverpool->last_cb_extent,
        .cs = {},
        .vertex_prepared = vertex_prepared,
        .num_vertex_inputs = static_cast<u8>(num_vertex),
        .num_vertex_ranges = static_cast<u8>(num_vertex_ranges),
        .motion_prepared = motion_prepared,
        .ring_used = num_prefetch != 0 || geometry_in_ring,
        .index_ring_size = index_ring.size,
        .index_ring_address = index_ring.address,
        .index_ring_offset = index_ring.offset,
    };
    if (kind == PacketKind::Dispatch || kind == PacketKind::DispatchIndirect) {
        header.cs = liverpool->GetCsRegs();
    }
    at += sizeof(PacketHeader);

    std::memcpy(at, pipe_blocks.data(), num_blocks * sizeof(u16));
    at += AlignPacket(num_blocks * sizeof(u16));
    for (const u16 block : pipe_blocks) {
        std::memcpy(at, regs.reg_array.data() + block * AmdGpu::RegDirty::BlockWords, BlockBytes);
        at += BlockBytes;
    }
    dirty.Clear();

    for (u32 i = 0; i < num_stages; ++i) {
        const auto* info = infos[i];
        const auto ud_size = static_cast<u32>(info->user_data.size());
        const auto flat_size = static_cast<u32>(info->flattened_ud_buf.size());
        *reinterpret_cast<PacketStage*>(at) = {info, info->pgm_base, ud_size, flat_size};
        at += sizeof(PacketStage);
        auto* words = reinterpret_cast<u32*>(at);
        if (ud_size != 0) {
            std::memcpy(words, info->user_data.data(), ud_size * sizeof(u32));
        }
        if (flat_size != 0) {
            std::memcpy(words + ud_size, info->flattened_ud_buf.data(), flat_size * sizeof(u32));
        }
        at += AlignPacket((ud_size + flat_size) * sizeof(u32));
    }
    std::memcpy(at, prefetch_scratch.data(), num_prefetch * sizeof(PrefetchedBuffer));
    at += num_prefetch * sizeof(PrefetchedBuffer);
    if (vertex_prepared) {
        const auto put = [&at](const void* src, size_t bytes) {
            std::memcpy(at, src, bytes);
            at += bytes;
        };
        put(vtx_attributes.data(), num_vertex * sizeof(vk::VertexInputAttributeDescription2EXT));
        put(vtx_bindings.data(), num_vertex * sizeof(vk::VertexInputBindingDescription2EXT));
        put(vtx_buffers.data(), num_vertex * sizeof(AmdGpu::Buffer));
        std::memcpy(at, vtx_range_index.data(), num_vertex);
        at += AlignPacket(num_vertex);
        static_assert(sizeof(PreparedRange) == 24);
        put(vtx_ranges.data(), num_vertex_ranges * sizeof(PreparedRange));
    }
    if (motion_prepared) {
        std::memcpy(at, &motion_prep, sizeof(MotionPrep));
        at += AlignPacket(sizeof(MotionPrep));
    }
    ASSERT(static_cast<u32>(at - out) == size);
    draw_pipe->Commit(size);
    ALap(ASection::Packet);
}

void Rasterizer::ABeginDraw() {
    if (!diagnostics.load(std::memory_order_relaxed)) {
        a_profile = false;
        a_hitch = false;
        a_last = {};
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    a_profile = (a_profile_counter++ & 7) == 0;
    a_hitch = HitchProfile();
    if (a_hitch) {
        a_hitch_draws.fetch_add(1, std::memory_order_relaxed);
    }
    if (a_profile && a_last != std::chrono::steady_clock::time_point{}) {
        const auto decode = (now - a_last) - (draw_pipe->drain_time - a_drain_mark);
        if (decode < std::chrono::milliseconds(1)) {
            a_profile_ns[static_cast<size_t>(ASection::Decode)].fetch_add(
                decode.count(), std::memory_order_relaxed);
            a_profile_draws.fetch_add(1, std::memory_order_relaxed);
        }
    }
    a_lap = now;
}

void Rasterizer::AEndDraw() {
    if (!diagnostics.load(std::memory_order_relaxed)) {
        return;
    }
    a_profile = false;
    a_hitch = false;
    a_last = std::chrono::steady_clock::now();
    a_drain_mark = draw_pipe->drain_time;
}

u64 Rasterizer::RunInOrder(OrderedTask task, const void* data, u32 size, VAddr write_addr,
                           u64 write_size, const char* writer) {
    if (!UseDrawPipe() || !DrawPipe::OnStageA()) {
        DrainDrawPipe();
        task(*this, static_cast<const u8*>(data));
        return 0;
    }
    if (write_size != 0) {
        pending_writes.push_back(
            {draw_pipe->packets + 1, write_addr, write_addr + write_size, writer, 0});
    }
    const u32 total = sizeof(TaskPacket) + AlignPacket(size);
    u8* out = draw_pipe->Begin(total);
    *reinterpret_cast<TaskPacket*>(out) = {
        static_cast<u8>(PacketKind::Task), {}, size, task, copy_head};
    std::memcpy(out + sizeof(TaskPacket), data, size);
    draw_pipe->Commit(total);
    return draw_pipe->packets;
}

void Rasterizer::RingRetire() {
    // Space the GPU has finished with: the draw recorder reports, per command buffer, how far
    // into the ring its recorded draws read.
    const u64 written = ring_retire_w.load(std::memory_order_acquire);
    u64 read = ring_retire_r.load(std::memory_order_relaxed);
    while (read < written) {
        const auto& entry = ring_retire[read % RingRetireSize];
        if (!scheduler.IsFree(entry.tick)) {
            break;
        }
        ring_free_until = std::max(ring_free_until, entry.end);
        ++read;
    }
    ring_retire_r.store(read, std::memory_order_release);
}

u8* Rasterizer::RingAlloc(u64 size, u64 alignment, u64& offset) {
    u64 pos = Common::AlignUp(ring_head, alignment);
    if (pos % ConstRingSize + size > ConstRingSize) {
        pos = Common::AlignUp(pos, ConstRingSize);
    }
    if (pos + size > ring_free_until + ConstRingSize) {
        // Only when the space runs out: the retirement check reads the draw recorder's
        // counter and may ask the driver how far the GPU is (once per allocation it cost
        // ~0.3 us each, several times per draw).
        RingRetire();
        if (pos + size > ring_free_until + ConstRingSize) {
            return nullptr; // Full: the draw recorder copies this one as before.
        }
    }
    ring_head = pos + size;
    offset = pos % ConstRingSize;
    return const_ring->mapped_data.data() + offset;
}

u64 Rasterizer::RingCopyGeometry(VAddr address, u64 size) {
    // BufferCache::ObtainGeometryBuffer's copy path, decided here: data it would copy to the
    // stream buffer (under 1 KiB, or changed by the CPU) and not written by queued GPU work.
    if (!const_ring || size == 0 || size > VideoCore::BufferCache::StreamThreshold() ||
        !EmulatorSettings.IsBbRingGeometry() || buffer_cache.IsRegionGpuModified(address, size)) {
        return NoRing;
    }
    const bool copied = size < 1024 || !Common::BbTuning::Threaded() ||
                        !EmulatorSettings.IsBbGeometryInPlace() ||
                        buffer_cache.IsRegionCpuModified(address, size);
    if (!copied || std::ranges::any_of(pending_writes, [&](const auto& write) {
            return address < write.end && write.begin < address + size;
        })) {
        return NoRing;
    }
    u64 offset = 0;
    u8* dest = RingAlloc(size, instance.UniformMinAlignment(), offset);
    if (!dest) {
        ring_fallbacks.fetch_add(1, std::memory_order_relaxed);
        return NoRing;
    }
    RingCopy(address, dest, offset, size);
    geometry_ring_copies.fetch_add(1, std::memory_order_relaxed);
    return offset;
}

void Rasterizer::RingCopy(VAddr guest, u8* dest, u64 offset, u64 size) {
    if (!copy_thread_on) {
        memory->CopySparseMemory(guest, dest, size);
        const_ring->Flush(offset, size);
        return;
    }
    // Copy thread: made later, before the packet runs on the draw recorder.
    while (copy_head - copy_done.load(std::memory_order_acquire) >= CopyJobCount) {
        std::this_thread::yield();
    }
    copy_jobs[copy_head % CopyJobCount] = {guest, dest, offset, size};
    ++copy_head;
    copy_published.store(copy_head, std::memory_order_seq_cst);
    if (copy_sleeping.load(std::memory_order_seq_cst)) {
        copy_wake.fetch_add(1, std::memory_order_seq_cst);
        copy_wake.notify_one();
    }
}

void Rasterizer::RunCopyThread(std::stop_token stop) {
    Common::SetCurrentThreadName("shadPS4:GpuCopies");
    Common::SetCurrentThreadPriority(Common::ThreadPriority::High);
    const std::stop_callback wake_on_stop(stop, [this] {
        copy_wake.fetch_add(1, std::memory_order_seq_cst);
        copy_wake.notify_all();
    });
    u64 at = 0;
    while (true) {
        u64 available = copy_published.load(std::memory_order_acquire);
        if (available == at) {
            const auto spin_until =
                std::chrono::steady_clock::now() + std::chrono::microseconds(50);
            for (u32 spins = 1; available == at; ++spins) {
                if (stop.stop_requested()) {
                    return;
                }
                if ((spins & 255) == 0 && std::chrono::steady_clock::now() >= spin_until) {
                    const u32 seen = copy_wake.load(std::memory_order_seq_cst);
                    copy_sleeping.store(true, std::memory_order_seq_cst);
                    if (copy_published.load(std::memory_order_seq_cst) == at &&
                        !stop.stop_requested()) {
                        copy_wake.wait(seen, std::memory_order_seq_cst);
                    }
                    copy_sleeping.store(false, std::memory_order_relaxed);
                }
#if defined(__x86_64__) || defined(_M_X64)
                _mm_pause();
#endif
                available = copy_published.load(std::memory_order_acquire);
            }
        }
        const auto start = std::chrono::steady_clock::now();
        for (; at < available; ++at) {
            const auto& job = copy_jobs[at % CopyJobCount];
            memory->CopySparseMemory(job.guest, job.dest, job.size);
            const_ring->Flush(job.offset, job.size);
            copy_done.store(at + 1, std::memory_order_release);
        }
        copy_busy_ns.fetch_add((std::chrono::steady_clock::now() - start).count(),
                               std::memory_order_relaxed);
    }
}

void Rasterizer::WaitForCopies(u64 end) {
    const auto start = std::chrono::steady_clock::now();
    for (u32 spins = 0; copy_done.load(std::memory_order_acquire) < end; ++spins) {
        if (spins < 4096) {
#if defined(__x86_64__) || defined(_M_X64)
            _mm_pause();
#endif
        } else {
            std::this_thread::yield();
        }
    }
    copy_waits.fetch_add(1, std::memory_order_relaxed);
    copy_wait_ns.fetch_add((std::chrono::steady_clock::now() - start).count(),
                           std::memory_order_relaxed);
}

void Rasterizer::PrefetchBuffers(const Pipeline* pipeline) {
    // Writes of draws the recorder has finished are visible to IsRegionGpuModified.
    const u64 consumed = draw_pipe->ConsumedPackets();
    while (!pending_writes.empty() && pending_writes.front().packet <= consumed) {
        pending_writes.pop_front();
    }
    const u64 alignment = instance.StorageMinAlignment();
    u32 ordinal = 0;
    for (const auto* info : pipeline->GetStages()) {
        if (!info) {
            continue;
        }
        const u32 stage = ordinal++;
        for (u32 index = 0; index < info->buffers.size(); ++index) {
            const auto& desc = info->buffers[index];
            const u8* source = nullptr;
            u64 size = 0;
            VAddr guest = 0;
            if (desc.IsSpecial()) {
                if (desc.buffer_type != Shader::BufferType::Flatbuf) {
                    continue;
                }
                source = reinterpret_cast<const u8*>(info->flattened_ud_buf.data());
                size = info->flattened_ud_buf.size() * sizeof(u32);
            } else {
                if (desc.is_written) {
                    continue;
                }
                const auto vsharp = desc.GetSharp(*info);
                size = vsharp.GetSize();
                guest = vsharp.base_address;
                // BufferCache::ObtainBuffer's stream path, decided here.
                if (guest == 0 || size == 0 || size > VideoCore::BufferCache::StreamThreshold() ||
                    buffer_cache.IsRegionGpuModified(guest, size)) {
                    continue;
                }
                const bool pending = std::ranges::any_of(pending_writes, [&](const auto& write) {
                    return guest < write.end && write.begin < guest + size;
                });
                if (pending) {
                    continue;
                }
            }
            if (size == 0) {
                continue;
            }
            u64 offset = 0;
            u8* dest = RingAlloc(size, alignment, offset);
            if (!dest) {
                ring_fallbacks.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            if (source) {
                std::memcpy(dest, source, size);
                const_ring->Flush(offset, size);
            } else {
                RingCopy(guest, dest, offset, size);
            }
            prefetch_scratch.push_back({static_cast<u8>(stage), static_cast<u8>(index), 0,
                                        static_cast<u32>(size), offset});
        }
    }
    if (!prefetch_scratch.empty()) {
        ring_prefetches.fetch_add(prefetch_scratch.size(), std::memory_order_relaxed);
    }
}

void Rasterizer::NoteQueuedReads(const Pipeline* pipeline) {
    const u64 consumed = draw_pipe->ConsumedPackets();
    while (!queued_later_reads.empty() && queued_later_reads.front().packet <= consumed) {
        queued_later_reads.pop_front();
    }
    if (queued_later_reads.empty()) {
        queued_reads_overflow = false;
    }
    const u64 packet = draw_pipe->packets + 1;
    u32 ordinal = 0;
    for (const auto* info : pipeline->GetStages()) {
        if (!info) {
            continue;
        }
        const u32 stage = ordinal++;
        for (u32 index = 0; index < info->buffers.size(); ++index) {
            const auto& desc = info->buffers[index];
            if (desc.IsSpecial() || desc.is_written) {
                continue;
            }
            const auto vsharp = desc.GetSharp(*info);
            if (vsharp.base_address == 0 || vsharp.GetSize() == 0) {
                continue;
            }
            const bool copied = std::ranges::any_of(prefetch_scratch, [&](const auto& entry) {
                return entry.stage == stage && entry.buffer == index;
            });
            if (copied && !copy_thread_on) {
                continue; // read from the constant ring, not the guest's memory
            }
            auto& reads = queued_later_reads;
            if (reads.size() >= 16384) {
                queued_reads_overflow = true;
                continue;
            }
            reads.push_back({packet, vsharp.base_address, vsharp.base_address + vsharp.GetSize()});
        }
    }
}

Rasterizer::QueuedUse Rasterizer::CheckQueuedUse(VAddr address, u64 size) {
    if (!UseDrawPipe() || !draw_pipe) {
        return {true, true}; // not tracked: wait as before
    }
    const u64 consumed = draw_pipe->ConsumedPackets();
    const auto overlaps = [&](const std::deque<PendingWrite>& ranges) {
        return std::ranges::any_of(ranges, [&](const auto& range) {
            return range.packet > consumed && address < range.end && range.begin < address + size;
        });
    };
    QueuedUse use{
        .read_later = queued_reads_overflow || overlaps(queued_later_reads),
        .written = false,
        .overflow = queued_reads_overflow,
    };
    // Constant dumps past huge writes: a storage write declared over all of memory says nothing
    // about this range.
    constexpr u64 HugeWriteBytes = 256_MB;
    // Not with the copy shader on the CPU: its copies then land in game memory when the draw
    // recorder reaches them, after a dump written at once, and would overwrite the dump's newer
    // data (writing the GPU's copy, they lost to the dump's data at the next upload).
    const bool skip_huge =
        EmulatorSettings.IsBbCeSkipHugeWrites() && !EmulatorSettings.IsBbCpuCopyShader();
    for (const auto& write : pending_writes) {
        if (write.packet > consumed && address < write.end && write.begin < address + size) {
            if (skip_huge && write.shader_write && write.end - write.begin > HugeWriteBytes) {
                ++use.huge_writes_skipped;
                continue;
            }
            use.written = true;
            use.writer = write.writer ? write.writer : "queued task";
            use.writer_shader = write.shader;
            use.writer_bytes = write.end - write.begin;
            use.writer_packets_back = draw_pipe->packets - write.packet;
            break;
        }
    }
    return use;
}

// ---------------------------------------------------------------------------------------------
// Real occlusion tests (bb_real_occlusion)
// ---------------------------------------------------------------------------------------------
// The game tests whole areas: between two counter dumps (ZPASS_DONE, the second 8 bytes after
// the first) it draws a box against the scene depth, and skips the area while no pixel of the
// box is visible. The GPU counts the box's visible samples with a Vulkan occlusion query; the
// count goes into the game's result block when that submission has finished.

namespace {
constexpr u64 OcclusionValidBit = 0x8000000000000000ULL;
constexpr u64 OcclusionVisibleCount = 0x2FFFFFFULL;
} // Anonymous namespace

void Rasterizer::OcclusionDump(u64* results, u64 value, u32 pairs) {
    const auto address = reinterpret_cast<VAddr>(results);
    OcclusionTask task{};
    if (occlusion_a.open && address == occlusion_a.begin_address + sizeof(u64)) {
        task = {reinterpret_cast<u64*>(occlusion_a.begin_address), occlusion_a.begin_value, pairs,
                occlusion_a.draws, false};
        task.depth_address = occlusion_a.depth_address;
        task.depth_extent = occlusion_a.depth_extent;
        task.depth_control = occlusion_a.depth_control;
        task.index = occlusion_a.index;
        occlusion_a.open = false;
    } else {
        // A new test (one left open is answered "visible" by the draw recorder).
        occlusion_a = {address, value, 0, true};
        occlusion_a.pairs = pairs;
        if (const u64 frame = occlusion_frame.load(std::memory_order_relaxed);
            frame != occlusion_index_frame) {
            occlusion_index_frame = frame;
            occlusion_next_index = 0;
        }
        occlusion_a.index = occlusion_next_index++;
        task = {results, value, pairs, 0, true};
        task.index = occlusion_a.index;
    }
    task.frame = occlusion_frame.load(std::memory_order_relaxed);
    task.time = std::chrono::steady_clock::now();
    RunInOrder(
        [](Rasterizer& self, const u8* data) {
            self.OcclusionTaskRun(*reinterpret_cast<const OcclusionTask*>(data));
        },
        &task, sizeof(task));
}

bool Rasterizer::OcclusionRedirectBegin() {
    // Decoupled render resolution: the 3D scene is drawn at the render size into its own depth
    // buffer, but the game's occlusion test boxes still use the output-size depth buffer and
    // viewport. While a test is open, its box is drawn against the scene's depth instead, with
    // the viewport and scissors scaled to the render size.
    if (!EmulatorSettings.IsBbRealOcclusion()) {
        scene_depth.valid = false;
        if (occlusion_a.open) {
            // Turned off during a test: it ends here, answered "visible".
            const OcclusionTask task{reinterpret_cast<u64*>(occlusion_a.begin_address),
                                     occlusion_a.begin_value,
                                     0,
                                     ~0u,
                                     false,
                                     occlusion_frame.load(std::memory_order_relaxed),
                                     std::chrono::steady_clock::now()};
            occlusion_a.open = false;
            RunInOrder(
                [](Rasterizer& self, const u8* data) {
                    self.OcclusionTaskRun(*reinterpret_cast<const OcclusionTask*>(data));
                },
                &task, sizeof(task));
        }
        return false;
    }
    const auto& resolution = Core::BbResolution::Applied();
    if (!resolution.enabled || resolution.regular ||
        (resolution.render_width == resolution.output_width &&
         resolution.render_height == resolution.output_height)) {
        return false;
    }
    auto& regs = liverpool->regs;
    const auto extent = liverpool->last_db_extent;
    const bool depth = regs.depth_buffer.DepthValid() &&
                       (regs.depth_control.depth_enable || regs.depth_control.stencil_enable);
    if (!occlusion_a.open) {
        // Remember the scene's depth buffer: the last one drawn with at the render size.
        if (depth && extent.width == resolution.render_width &&
            extent.height == resolution.render_height) {
            const bool same = scene_depth.valid && scene_depth.depth_buffer.DepthAddress() ==
                                                       regs.depth_buffer.DepthAddress();
            const u64 frame = same ? scene_depth.frame : 0;
            const bool written = same && scene_depth.written;
            scene_depth = {true, regs.depth_buffer, regs.depth_view, regs.depth_htile_data_base,
                           extent};
            scene_depth.frame = frame;
            scene_depth.written = written;
            if (regs.depth_control.depth_write_enable) {
                scene_depth.frame = occlusion_frame.load(std::memory_order_relaxed);
                scene_depth.written = true;
            }
        }
        return false;
    }
    if (!depth || !scene_depth.valid || extent.width != resolution.output_width ||
        extent.height != resolution.output_height) {
        return false;
    }
    auto& saved = occlusion_saved;
    saved.depth_buffer = regs.depth_buffer;
    saved.depth_view = regs.depth_view;
    saved.htile = regs.depth_htile_data_base;
    saved.extent = extent;
    saved.viewports = regs.viewports;
    saved.screen_scissor = regs.screen_scissor;
    saved.window_scissor = regs.window_scissor;
    saved.generic_scissor = regs.generic_scissor;
    saved.viewport_scissors = regs.viewport_scissors;

    regs.depth_buffer = scene_depth.depth_buffer;
    regs.depth_view = scene_depth.depth_view;
    regs.depth_htile_data_base = scene_depth.htile;
    liverpool->last_db_extent = scene_depth.extent;
    const float sx = float(resolution.render_width) / float(resolution.output_width);
    const float sy = float(resolution.render_height) / float(resolution.output_height);
    for (auto& vp : regs.viewports) {
        vp.xscale *= sx;
        vp.xoffset *= sx;
        vp.yscale *= sy;
        vp.yoffset *= sy;
    }
    const auto scale = [](auto value, float factor) {
        return static_cast<decltype(value)>(std::ceil(float(value) * factor));
    };
    regs.screen_scissor.top_left_x = scale(regs.screen_scissor.top_left_x, sx);
    regs.screen_scissor.top_left_y = scale(regs.screen_scissor.top_left_y, sy);
    regs.screen_scissor.bottom_right_x = scale(regs.screen_scissor.bottom_right_x, sx);
    regs.screen_scissor.bottom_right_y = scale(regs.screen_scissor.bottom_right_y, sy);
    const auto scale_scissor = [&](AmdGpu::ViewportScissor& s) {
        s.top_left_x = scale(u16(s.top_left_x), sx);
        s.top_left_y = scale(u16(s.top_left_y), sy);
        s.bottom_right_x = scale(u16(s.bottom_right_x), sx);
        s.bottom_right_y = scale(u16(s.bottom_right_y), sy);
    };
    scale_scissor(regs.window_scissor);
    scale_scissor(regs.generic_scissor);
    for (auto& s : regs.viewport_scissors) {
        scale_scissor(s);
    }
    OcclusionMarkRedirected();
    occlusion_redirects.fetch_add(1, std::memory_order_relaxed);
    (scene_depth.written && scene_depth.frame == occlusion_frame.load(std::memory_order_relaxed)
         ? occlusion_depth_current
         : occlusion_depth_older)
        .fetch_add(1, std::memory_order_relaxed);
    return true;
}

void Rasterizer::OcclusionRedirectEnd() {
    auto& regs = liverpool->regs;
    const auto& saved = occlusion_saved;
    regs.depth_buffer = saved.depth_buffer;
    regs.depth_view = saved.depth_view;
    regs.depth_htile_data_base = saved.htile;
    liverpool->last_db_extent = saved.extent;
    regs.viewports = saved.viewports;
    regs.screen_scissor = saved.screen_scissor;
    regs.window_scissor = saved.window_scissor;
    regs.generic_scissor = saved.generic_scissor;
    regs.viewport_scissors = saved.viewport_scissors;
    OcclusionMarkRedirected();
}

void Rasterizer::OcclusionMarkRedirected() {
    // The draw recorder's copy of the registers follows the changes (and their undoing).
    auto& regs = liverpool->regs;
    liverpool->MarkRegWritten(regs.depth_buffer);
    liverpool->MarkRegWritten(regs.depth_view);
    liverpool->MarkRegWritten(regs.depth_htile_data_base);
    liverpool->MarkRegWritten(regs.viewports);
    liverpool->MarkRegWritten(regs.screen_scissor);
    liverpool->MarkRegWritten(regs.window_scissor);
    liverpool->MarkRegWritten(regs.generic_scissor);
    liverpool->MarkRegWritten(regs.viewport_scissors);
}

void Rasterizer::OcclusionTaskRun(const OcclusionTask& task) {
    auto& b = occlusion_b;
    if (task.begin) {
        if (b.open) {
            b.dropped = true;
            OcclusionClose(0); // never ended: answered "visible"
        }
        b = {};
        b.open = true;
        b.results = task.results;
        b.begin_value = task.begin_value;
        b.pairs = task.pairs;
        b.frame = task.frame;
        b.time = task.time;
        b.index = task.index;
        occlusion_tests.fetch_add(1, std::memory_order_relaxed);
        {
            // The game began this test again while the previous answer was still on its way.
            std::scoped_lock lock{occlusion_mutex};
            if (std::ranges::any_of(occlusion_pending, [&](const PendingOcclusion& pending) {
                    return pending.results == task.results;
                })) {
                occlusion_retested.fetch_add(1, std::memory_order_relaxed);
            }
        }
    } else if (b.open && b.results == task.results) {
        b.depth_address = task.depth_address;
        b.depth_extent = task.depth_extent;
        b.depth_control = task.depth_control;
        OcclusionClose(task.stage_a_draws);
    }
    PollOcclusion();
}

void Rasterizer::OcclusionClose(u32 stage_a_draws) {
    auto& b = occlusion_b;
    PendingOcclusion test{
        .results = b.results,
        .begin_value = b.begin_value,
        .pairs = b.pairs,
        .tick = scheduler.CurrentTick(),
        .queries = b.queries,
        // Every draw of the test recorded with a query, or the answer is "visible".
        .known = !b.dropped && b.draws_seen == stage_a_draws && b.queries.size() == b.draws_seen,
        .frame = b.frame,
        .time = b.time,
        .closed = std::chrono::steady_clock::now(),
        .depth_address = b.depth_address,
        .depth_extent = b.depth_extent,
        .depth_control = b.depth_control,
        .depth_view = b.depth_view,
        .render_size = b.render_size,
        .depth_clear = b.depth_clear,
        .index = b.index,
    };
    {
        std::scoped_lock lock{occlusion_mutex};
        occlusion_pending.push_back(std::move(test));
        occlusion_waiting.store(static_cast<u32>(occlusion_pending.size()),
                                std::memory_order_relaxed);
    }
    b.open = false;
}

u32 Rasterizer::OcclusionQueryBegin() {
    std::scoped_lock lock{occlusion_mutex};
    if (!occlusion_pool) {
        occlusion_pool = instance.GetDevice()
                             .createQueryPoolUnique(vk::QueryPoolCreateInfo{
                                 .queryType = vk::QueryType::eOcclusion,
                                 .queryCount = OcclusionPoolSize,
                             })
                             .value;
        instance.GetDevice().resetQueryPool(*occlusion_pool, 0, OcclusionPoolSize);
        occlusion_busy.assign(OcclusionPoolSize, false);
    }
    const u32 query = occlusion_next;
    if (occlusion_busy[query]) {
        occlusion_b.dropped = true; // every query still waiting for its answer
        return NoOcclusionQuery;
    }
    occlusion_next = (query + 1) % OcclusionPoolSize;
    occlusion_busy[query] = true;
    // Answered and read: free on the GPU, so it can be reset from here.
    instance.GetDevice().resetQueryPool(*occlusion_pool, query, 1);
    const vk::QueryControlFlags flags = instance.IsOcclusionQueryPrecise()
                                            ? vk::QueryControlFlagBits::ePrecise
                                            : vk::QueryControlFlags{};
    scheduler.Record([pool = *occlusion_pool, query, flags](vk::CommandBuffer cmdbuf) {
        cmdbuf.beginQuery(pool, query, flags);
    });
    return query;
}

void Rasterizer::OcclusionQueryEnd(u32 query) {
    if (query == NoOcclusionQuery) {
        return;
    }
    scheduler.Record([pool = *occlusion_pool, query](vk::CommandBuffer cmdbuf) {
        cmdbuf.endQuery(pool, query);
    });
    occlusion_b.queries.push_back(query);
}

void Rasterizer::PollOcclusion() {
    boost::container::small_vector<OcclusionAnswer, 16> answers;
    {
        std::scoped_lock lock{occlusion_mutex};
        while (!occlusion_pending.empty() && scheduler.IsFree(occlusion_pending.front().tick)) {
            answers.push_back(AnswerOcclusion(occlusion_pending.front()));
            occlusion_pending.pop_front();
        }
        occlusion_waiting.store(static_cast<u32>(occlusion_pending.size()),
                                std::memory_order_relaxed);
    }
    for (const auto& answer : answers) {
        WriteOcclusionAnswer(answer);
    }
}

void Rasterizer::WriteOcclusionAnswer(const OcclusionAnswer& answer) {
    // Begin and end of every counter pair, the whole count in the first pair; the game reads
    // "not ready" until then (the block was cleared before the test).
    boost::container::small_vector<u64, 32> data(u64(answer.pairs) * 2);
    for (u32 i = 0; i < answer.pairs; ++i) {
        data[i * 2] = answer.begin;
        data[i * 2 + 1] = i == 0 ? answer.begin + answer.count : answer.begin;
    }
    std::memcpy(answer.results, data.data(), data.size() * sizeof(u64));
}

Rasterizer::OcclusionAnswer Rasterizer::AnswerOcclusion(const PendingOcclusion& test) {
    u64 count = 0;
    bool known = test.known;
    for (const u32 query : test.queries) {
        u64 samples = 0;
        const auto result = instance.GetDevice().getQueryPoolResults(
            *occlusion_pool, query, 1, sizeof(samples), &samples, sizeof(samples),
            vk::QueryResultFlagBits::e64);
        known &= result == vk::Result::eSuccess;
        count += samples;
        occlusion_busy[query] = false;
    }
    if (!known) {
        occlusion_unknown.fetch_add(1, std::memory_order_relaxed);
        count = OcclusionVisibleCount;
    }
    {
        // One-frame hidden blips: visible, hidden, visible again in consecutive game frames.
        auto& slot = occlusion_slots[test.results];
        slot[0] = slot[1];
        slot[1] = slot[2];
        slot[2] = {test.frame,
                   count,
                   test.depth_address,
                   test.depth_extent,
                   test.depth_control,
                   test.depth_view,
                   test.render_size,
                   test.depth_clear,
                   known};
        {
            // By the test's place in its frame (n-th test), consecutive frames.
            auto& place = occlusion_indices[test.index];
            place[0] = place[1];
            place[1] = place[2];
            place[2] = {test.frame, count, test.results};
            if (place[0].frame != 0 && place[0].frame + 1 == place[1].frame &&
                place[1].frame + 1 == place[2].frame && place[0].count != 0 &&
                place[1].count == 0 && place[2].count != 0) {
                occlusion_index_blips.fetch_add(1, std::memory_order_relaxed);
                if (occlusion_index_blips_logged < 40) {
                    ++occlusion_index_blips_logged;
                    LOG_INFO(Render_Vulkan,
                             "Occlusion: hidden blip of the game's test #{} of its frame: frame "
                             "{} count {} ({}), frame {} count 0 ({}), frame {} count {} ({})",
                             test.index, place[0].frame, place[0].count, fmt::ptr(place[0].results),
                             place[1].frame, fmt::ptr(place[1].results), place[2].frame,
                             place[2].count, fmt::ptr(place[2].results));
                }
            }
        }
        if (slot[1].frame != 0) {
            occlusion_slot_gaps[std::min<u64>(slot[2].frame - slot[1].frame, 3)].fetch_add(
                1, std::memory_order_relaxed);
        }
        // The slot's last three answers (a slot may be used every other frame).
        const bool consecutive = slot[0].frame != 0 && slot[2].frame - slot[0].frame <= 6;
        if (consecutive && slot[0].count != 0 && slot[1].count == 0 && slot[2].count != 0) {
            occlusion_blips.fetch_add(1, std::memory_order_relaxed);
            if (occlusion_blips_logged < 40) {
                ++occlusion_blips_logged;
                const auto describe = [](const OcclusionSlotAnswer& a) {
                    return fmt::format("frame {} count {}{} | depth {:#x} hint {}x{} control "
                                       "{:#x} | view {:#x} pass {}x{}{}",
                                       a.frame, a.count, a.known ? "" : " (unknown)",
                                       a.depth_address, a.depth_extent & 0xffff,
                                       a.depth_extent >> 16, a.depth_control, a.depth_view,
                                       a.render_size & 0xffff, a.render_size >> 16,
                                       a.depth_clear ? " (clears depth)" : "");
                };
                LOG_INFO(Render_Vulkan,
                         "Occlusion: hidden blip of the test at {}:\n  before: {}\n  "
                         "hidden: {}\n  after:  {}",
                         fmt::ptr(test.results), describe(slot[0]), describe(slot[1]),
                         describe(slot[2]));
            }
        }
        if (occlusion_slots.size() > 4096) {
            occlusion_slots.clear();
        }
    }
    occlusion_answered.fetch_add(1, std::memory_order_relaxed);
    occlusion_hidden.fetch_add(count == 0 ? 1 : 0, std::memory_order_relaxed);
    const u64 frames = occlusion_frame.load(std::memory_order_relaxed) - test.frame;
    occlusion_late_frames[std::min<u64>(frames, 3)].fetch_add(1, std::memory_order_relaxed);
    const auto now = std::chrono::steady_clock::now();
    occlusion_late_ns.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now - test.time).count(),
        std::memory_order_relaxed);
    {
        const auto& entry = submit_times[test.tick % submit_times.size()];
        if (entry.tick.load(std::memory_order_acquire) == test.tick) {
            const s64 closed = test.closed.time_since_epoch().count();
            const s64 submitted = std::max(entry.ns.load(std::memory_order_relaxed), closed);
            const s64 begun = test.time.time_since_epoch().count();
            const s64 answered = now.time_since_epoch().count();
            occlusion_span_ns[0].fetch_add(u64(std::max<s64>(closed - begun, 0)),
                                           std::memory_order_relaxed);
            occlusion_span_ns[1].fetch_add(u64(submitted - closed), std::memory_order_relaxed);
            occlusion_span_ns[2].fetch_add(u64(std::max<s64>(answered - submitted, 0)),
                                           std::memory_order_relaxed);
            occlusion_span_count.fetch_add(1, std::memory_order_relaxed);
        }
    }
    return {test.results, test.begin_value | OcclusionValidBit, test.pairs, count};
}

const Rasterizer::PrefetchedBuffer* Rasterizer::FindPrefetched(u32 buffer_index) {
    // PrefetchBuffers lists them in binding order (stage, then buffer index): the next one
    // is the only candidate.
    if (prefetch_cursor < packet_num_prefetch) {
        const auto& entry = packet_prefetch[prefetch_cursor];
        if (entry.stage == bind_stage_ordinal && entry.buffer == buffer_index) {
            ++prefetch_cursor;
            return &entry;
        }
    }
    return nullptr;
}

bool Rasterizer::HitchProfile() {
    return EmulatorSettings.IsBbStutterDetail();
}

void Rasterizer::RunDrawPacket(void* context, const u8* data, u32 size) {
    auto& self = *static_cast<Rasterizer*>(context);
    // Copy thread: this packet's constant and geometry copies (and every earlier one) are done
    // before it runs, so the draw recorder and the GPU see them as if stage A had made them.
    const u64 copy_end = *data == static_cast<u8>(PacketKind::Task)
                             ? reinterpret_cast<const TaskPacket*>(data)->copy_end
                             : reinterpret_cast<const PacketHeader*>(data)->copy_end;
    if (self.copy_done.load(std::memory_order_acquire) < copy_end) [[unlikely]] {
        self.WaitForCopies(copy_end);
    }
    if (!HitchProfile() || !self.diagnostics.load(std::memory_order_relaxed)) [[likely]] {
        RunDrawPacketBody(self, data, size);
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    RunDrawPacketBody(self, data, size);
    const size_t kind = std::min<size_t>(*data, self.packet_kind_ns.size() - 1);
    self.packet_kind_ns[kind].fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                            std::chrono::steady_clock::now() - start)
                                            .count(),
                                        std::memory_order_relaxed);
}

void Rasterizer::RunDrawPacketBody(Rasterizer& self, const u8* data, u32 size) {
    // Upload statistics: which kind of packet the texture uploads below happen in.
    Common::GuestStats::UploadPacketScope upload_packet{*data};
    // Occlusion answers go to the game as soon as the GPU has them, not at the next test.
    if (self.occlusion_waiting.load(std::memory_order_relaxed) != 0 &&
        (++self.occlusion_poll_counter & 31) == 0) {
        self.PollOcclusion();
    }
    if (*data == static_cast<u8>(PacketKind::Task)) {
        const auto& task = *reinterpret_cast<const TaskPacket*>(data);
        task.task(self, data + sizeof(TaskPacket));
        return;
    }
    self.profile_draw = self.diagnostics.load(std::memory_order_relaxed) &&
                        ((self.profile_counter++ & 7) == 0 || HitchProfile()) &&
                        *data == static_cast<u8>(PacketKind::Draw);
    if (self.profile_draw) {
        self.profile_last = std::chrono::steady_clock::now();
    }
    const u8* in = data;
    const auto& header = *reinterpret_cast<const PacketHeader*>(in);
    in += sizeof(PacketHeader);

    auto& regs = *self.pipe_regs;
    if (header.regs_reset) {
        regs.SetDefaults();
    }
    const auto* blocks = reinterpret_cast<const u16*>(in);
    in += AlignPacket(header.num_blocks * sizeof(u16));
    constexpr u32 BlockWords = AmdGpu::RegDirty::BlockWords;
    for (u32 i = 0; i < header.num_blocks; ++i) {
        std::memcpy(regs.reg_array.data() + blocks[i] * BlockWords, in, BlockWords * sizeof(u32));
        in += BlockWords * sizeof(u32);
    }

    for (u32 i = 0; i < header.num_stages; ++i) {
        const auto& stage = *reinterpret_cast<const PacketStage*>(in);
        in += sizeof(PacketStage);
        const auto* words = reinterpret_cast<const u32*>(in);
        Shader::Info::ud_snapshots[i] = {stage.info,           words,
                                         stage.user_data_size, words + stage.user_data_size,
                                         stage.flat_size,      stage.pgm_base};
        in += AlignPacket((stage.user_data_size + stage.flat_size) * sizeof(u32));
    }
    self.packet_prefetch = reinterpret_cast<const PrefetchedBuffer*>(in);
    self.packet_num_prefetch = header.num_prefetch;
    in += header.num_prefetch * sizeof(PrefetchedBuffer);
    self.packet_vertex = nullptr;
    if (header.vertex_prepared) {
        const u32 count = header.num_vertex_inputs;
        auto& vertex = self.packet_vertex_storage;
        vertex.count = count;
        vertex.num_ranges = header.num_vertex_ranges;
        vertex.attributes = reinterpret_cast<const vk::VertexInputAttributeDescription2EXT*>(in);
        in += count * sizeof(vk::VertexInputAttributeDescription2EXT);
        vertex.bindings = reinterpret_cast<const vk::VertexInputBindingDescription2EXT*>(in);
        in += count * sizeof(vk::VertexInputBindingDescription2EXT);
        vertex.buffers = reinterpret_cast<const AmdGpu::Buffer*>(in);
        in += count * sizeof(AmdGpu::Buffer);
        vertex.range_index = in;
        in += AlignPacket(count);
        vertex.ranges = reinterpret_cast<const PreparedRange*>(in);
        in += vertex.num_ranges * sizeof(PreparedRange);
        self.packet_vertex = &vertex;
    }
    self.packet_index_ring = {header.index_ring_address, header.index_ring_size,
                              header.index_ring_offset};
    self.packet_motion = nullptr;
    if (header.motion_prepared) {
        std::memcpy(&self.packet_motion_storage, in, sizeof(MotionPrep));
        self.packet_motion = &self.packet_motion_storage;
        in += AlignPacket(sizeof(MotionPrep));
    }
    ASSERT(static_cast<u32>(in - data) == size);
    Shader::Info::num_ud_snapshots = header.num_stages;

    stage_b_regs = &regs;
    self.pipe_cb_extent = header.cb_extent;
    self.pipe_db_extent = header.db_extent;
    self.pipe_cs = header.cs;
    const IndirectArgs args{
        .address = header.indirect_address,
        .count_address = header.indirect_count_address,
        .offset = header.indirect_offset,
        .stride = header.indirect_stride,
        .max_count = header.indirect_max_count,
    };
    switch (static_cast<PacketKind>(header.kind)) {
    case PacketKind::Draw:
        if (!header.pipeline) {
            self.scheduler.PopPendingOperations();
            self.FilterDraw();
            break;
        }
        self.ProfileLap(Section::Packet);
        self.DrawRecord(static_cast<const GraphicsPipeline*>(header.pipeline), header.is_indexed,
                        header.index_offset);
        if (self.profile_draw) {
            self.profile_draws.fetch_add(1, std::memory_order_relaxed);
        }
        break;
    case PacketKind::DrawIndirect:
        self.DrawIndirectRecord(static_cast<const GraphicsPipeline*>(header.pipeline),
                                header.is_indexed, args);
        break;
    case PacketKind::Dispatch:
        self.DispatchRecord(static_cast<const ComputePipeline*>(header.pipeline));
        break;
    case PacketKind::DispatchIndirect:
        self.DispatchIndirectRecord(static_cast<const ComputePipeline*>(header.pipeline), args);
        break;
    }
    Shader::Info::num_ud_snapshots = 0;
    self.profile_draw = false;
    if (header.ring_used) {
        // The command buffer that now holds these ring bindings (read after recording: a flush
        // during the draw moves them to a later one). When it changes, the previous one's
        // ring reach is final and goes to stage A.
        const u64 tick = self.scheduler.CurrentTick();
        if (tick != self.ring_b_tick) {
            const u64 w = self.ring_retire_w.load(std::memory_order_relaxed);
            if (self.ring_b_end != 0 &&
                w - self.ring_retire_r.load(std::memory_order_acquire) < RingRetireSize) {
                self.ring_retire[w % RingRetireSize] = {self.ring_b_end, self.ring_b_tick};
                self.ring_retire_w.store(w + 1, std::memory_order_release);
            }
            self.ring_b_tick = tick;
        }
        self.ring_b_end = std::max(self.ring_b_end, header.ring_end);
    }
    self.packet_prefetch = nullptr;
    self.packet_num_prefetch = 0;
    self.packet_vertex = nullptr;
    self.packet_motion = nullptr;
    self.packet_index_ring = {};
}

void Rasterizer::LogDrawPipeStats() {
    const auto now = std::chrono::steady_clock::now();
    if (pipe_stats_time == std::chrono::steady_clock::time_point{}) {
        pipe_stats_time = now;
        return;
    }
    const auto elapsed = now - pipe_stats_time;
    if (elapsed < std::chrono::seconds(10)) {
        return;
    }
    const double seconds = std::chrono::duration<double>(elapsed).count();
    const auto busy = draw_pipe->BusyTime();
    const auto percent = [&](std::chrono::nanoseconds time) {
        return 100.0 * std::chrono::duration<double>(time).count() / seconds;
    };
    LOG_INFO(Render_Vulkan,
             "Threaded renderer: {:.0f} draws/s queued, {:.0f} waits/s, GPU command thread "
             "waited {:.1f}%, draw recorder busy {:.1f}%",
             (draw_pipe->packets - pipe_stats_packets) / seconds,
             (draw_pipe->drains - pipe_stats_drains) / seconds,
             percent(draw_pipe->drain_time - pipe_stats_drain_time),
             percent(busy - pipe_stats_busy));
    const u64 memo_hits = target_memo_hits.load(std::memory_order_relaxed);
    const u64 memo_misses = target_memo_misses.load(std::memory_order_relaxed);
    const u64 memo_draws =
        (memo_hits - pipe_stats_memo_hits) + (memo_misses - pipe_stats_memo_misses);
    if (memo_draws != 0) {
        LOG_INFO(Render_Vulkan, "  render state memo: {:.0f}% of draws reused the open pass",
                 100.0 * (memo_hits - pipe_stats_memo_hits) / memo_draws);
    }
    pipe_stats_memo_hits = memo_hits;
    pipe_stats_memo_misses = memo_misses;
    const u64 image_hits = image_memo_hits.load(std::memory_order_relaxed);
    const u64 image_misses = image_memo_misses.load(std::memory_order_relaxed);
    const u64 image_lookups =
        (image_hits - pipe_stats_image_hits) + (image_misses - pipe_stats_image_misses);
    if (image_lookups != 0) {
        LOG_INFO(Render_Vulkan, "  image lookup memo: {:.0f}% of {:.0f} lookups/s remembered",
                 100.0 * (image_hits - pipe_stats_image_hits) / image_lookups,
                 image_lookups / seconds);
    }
    pipe_stats_image_hits = image_hits;
    const u64 set_hits = texture_set_hits.load(std::memory_order_relaxed);
    const u64 set_misses = texture_set_misses.load(std::memory_order_relaxed);
    const u64 set_lookups = (set_hits - pipe_stats_set_hits) + (set_misses - pipe_stats_set_misses);
    if (set_lookups != 0) {
        LOG_INFO(Render_Vulkan, "  texture set memo: {:.0f}% of {:.0f} stage texture sets/s reused",
                 100.0 * (set_hits - pipe_stats_set_hits) / set_lookups, set_lookups / seconds);
    }
    pipe_stats_set_hits = set_hits;
    pipe_stats_set_misses = set_misses;
    pipe_stats_image_misses = image_misses;
    {
        const u64 prefetches = ring_prefetches.load(std::memory_order_relaxed);
        const u64 fallbacks = ring_fallbacks.load(std::memory_order_relaxed);
        if (prefetches != pipe_stats_ring_prefetches || fallbacks != pipe_stats_ring_fallbacks) {
            LOG_INFO(Render_Vulkan, "  constant ring: {:.0f} buffers/s copied ahead, {:.0f}/s full",
                     (prefetches - pipe_stats_ring_prefetches) / seconds,
                     (fallbacks - pipe_stats_ring_fallbacks) / seconds);
        }
        pipe_stats_ring_prefetches = prefetches;
        pipe_stats_ring_fallbacks = fallbacks;
        if (copy_thread_on) {
            const u64 waits = copy_waits.load(std::memory_order_relaxed);
            const s64 wait_ns = copy_wait_ns.load(std::memory_order_relaxed);
            const s64 busy = copy_busy_ns.load(std::memory_order_relaxed);
            LOG_INFO(Render_Vulkan,
                     "  copy thread: {:.0f} copies/s, busy {:.1f}%; the draw recorder waited for "
                     "it {:.0f} times/s, {:.2f} ms/s",
                     (copy_head - pipe_stats_copy_jobs) / seconds,
                     (busy - pipe_stats_copy_busy) / 1e7 / seconds,
                     (waits - pipe_stats_copy_waits) / seconds,
                     (wait_ns - pipe_stats_copy_wait_ns) / 1e6 / seconds);
            pipe_stats_copy_jobs = copy_head;
            pipe_stats_copy_busy = busy;
            pipe_stats_copy_waits = waits;
            pipe_stats_copy_wait_ns = wait_ns;
        }
        const u64 geometry = geometry_ring_copies.load(std::memory_order_relaxed);
        LOG_INFO(Render_Vulkan, "  vertex/index data copied ahead: {:.0f}/s",
                 (geometry - pipe_stats_geometry_ring_copies) / seconds);
        pipe_stats_geometry_ring_copies = geometry;
        LOG_INFO(
            Render_Vulkan,
            "  draws that write nothing (skipped when on): {:.0f}/s; depth-test-only draws without "
            "a pixel shader: {:.0f}/s",
            (empty_draws_skipped - pipe_stats_empty_draws_skipped) / seconds,
            (depth_test_only_draws - pipe_stats_depth_test_only_draws) / seconds);
        pipe_stats_empty_draws_skipped = empty_draws_skipped;
        pipe_stats_depth_test_only_draws = depth_test_only_draws;
        const u64 tests = occlusion_tests.load(std::memory_order_relaxed);
        const u64 answered = occlusion_answered.load(std::memory_order_relaxed);
        const u64 hidden = occlusion_hidden.load(std::memory_order_relaxed);
        const u64 unknown = occlusion_unknown.load(std::memory_order_relaxed);
        const u64 redirects = occlusion_redirects.load(std::memory_order_relaxed);
        if (tests != pipe_stats_occlusion[0]) {
            LOG_INFO(Render_Vulkan,
                     "  real occlusion tests: {:.0f}/s, answered {:.0f}/s ({:.0f}/s hidden, "
                     "{:.0f}/s visible as not every draw was counted), {:.0f}/s moved to the "
                     "render-size scene depth",
                     (tests - pipe_stats_occlusion[0]) / seconds,
                     (answered - pipe_stats_occlusion[1]) / seconds,
                     (hidden - pipe_stats_occlusion[2]) / seconds,
                     (unknown - pipe_stats_occlusion[3]) / seconds,
                     (redirects - pipe_stats_occlusion[4]) / seconds);
        }
        pipe_stats_occlusion = {tests, answered, hidden, unknown, redirects};
        const std::array<u64, 6> late{occlusion_late_frames[0].load(std::memory_order_relaxed),
                                      occlusion_late_frames[1].load(std::memory_order_relaxed),
                                      occlusion_late_frames[2].load(std::memory_order_relaxed),
                                      occlusion_late_frames[3].load(std::memory_order_relaxed),
                                      occlusion_late_ns.load(std::memory_order_relaxed),
                                      occlusion_retested.load(std::memory_order_relaxed)};
        const u64 late_answers =
            (late[0] - pipe_stats_occlusion_late[0]) + (late[1] - pipe_stats_occlusion_late[1]) +
            (late[2] - pipe_stats_occlusion_late[2]) + (late[3] - pipe_stats_occlusion_late[3]);
        if (late_answers != 0) {
            const auto pct = [&](u32 i) {
                return 100.0 * (late[i] - pipe_stats_occlusion_late[i]) / late_answers;
            };
            LOG_INFO(Render_Vulkan,
                     "  occlusion answers: reached the game {:.1f} ms after the test on "
                     "average | in the same game frame {:.0f}%, 1 frame later {:.0f}%, 2 "
                     "{:.0f}%, 3 or more {:.0f}% | tests begun again before the previous answer "
                     "arrived {:.0f}/s",
                     (late[4] - pipe_stats_occlusion_late[4]) / 1e6 / late_answers, pct(0), pct(1),
                     pct(2), pct(3), (late[5] - pipe_stats_occlusion_late[5]) / seconds);
        }
        pipe_stats_occlusion_late = late;
        const std::array<u64, 4> span{occlusion_span_ns[0].load(std::memory_order_relaxed),
                                      occlusion_span_ns[1].load(std::memory_order_relaxed),
                                      occlusion_span_ns[2].load(std::memory_order_relaxed),
                                      occlusion_span_count.load(std::memory_order_relaxed)};
        const u64 spans = span[3] - pipe_stats_occlusion_span[3];
        if (spans != 0) {
            const auto ms = [&](u32 i) {
                return (span[i] - pipe_stats_occlusion_span[i]) / 1e6 / spans;
            };
            LOG_INFO(Render_Vulkan,
                     "  occlusion answer time: {:.1f} ms to the draw recorder, {:.1f} ms waiting "
                     "for its submission, {:.1f} ms on the GPU",
                     ms(0), ms(1), ms(2));
        }
        pipe_stats_occlusion_span = span;
        const u64 blips = occlusion_blips.load(std::memory_order_relaxed);
        const u64 index_blips = occlusion_index_blips.load(std::memory_order_relaxed);
        LOG_INFO(Render_Vulkan,
                 "  occlusion hidden blips by the test's place in the frame: {:.1f}/s",
                 (index_blips - pipe_stats_occlusion_index_blips) / seconds);
        pipe_stats_occlusion_index_blips = index_blips;
        LOG_INFO(Render_Vulkan,
                 "  occlusion hidden blips (visible, hidden, visible): {:.1f}/s | a slot's next "
                 "answer came 0 frames later {}, 1 {}, 2 {}, 3+ {} (in all)",
                 (blips - pipe_stats_occlusion_blips) / seconds,
                 occlusion_slot_gaps[0].load(std::memory_order_relaxed),
                 occlusion_slot_gaps[1].load(std::memory_order_relaxed),
                 occlusion_slot_gaps[2].load(std::memory_order_relaxed),
                 occlusion_slot_gaps[3].load(std::memory_order_relaxed));
        pipe_stats_occlusion_blips = blips;
        const std::array<u64, 2> depth{occlusion_depth_current.load(std::memory_order_relaxed),
                                       occlusion_depth_older.load(std::memory_order_relaxed)};
        const u64 depth_current = depth[0] - pipe_stats_occlusion_depth[0];
        const u64 depth_older = depth[1] - pipe_stats_occlusion_depth[1];
        if (depth_current + depth_older != 0) {
            LOG_INFO(Render_Vulkan,
                     "  occlusion tests on the scene's depth: {:.0f}% against depth drawn in the "
                     "same game frame, {:.0f}% against depth from an earlier frame",
                     100.0 * depth_current / (depth_current + depth_older),
                     100.0 * depth_older / (depth_current + depth_older));
        }
        pipe_stats_occlusion_depth = depth;
    }
    {
        const u64 prepared = vertex_prepared.load(std::memory_order_relaxed);
        const u64 unprepared = vertex_unprepared.load(std::memory_order_relaxed);
        LOG_INFO(Render_Vulkan, "  vertex inputs prepared ahead: {:.0f}/s, {:.0f}/s not",
                 (prepared - pipe_stats_vertex_prepared) / seconds,
                 (unprepared - pipe_stats_vertex_unprepared) / seconds);
        pipe_stats_vertex_prepared = prepared;
        pipe_stats_vertex_unprepared = unprepared;
    }
    {
        const u64 frames = stat_frames - pipe_stats_frames;
        LOG_INFO(Render_Vulkan, "  game frames: {:.1f}/s, {:.0f} draws per frame", frames / seconds,
                 frames ? double(stat_draws - pipe_stats_draws) / frames : 0.0);
        // GPU time from timestamps around each submission of this scheduler (the presenter's
        // own work, e.g. frame generation, is not included).
        const auto [busy_ns, idle_ns, submits] = scheduler.TakeGpuTiming();
        if (frames != 0 && submits != 0) {
            LOG_INFO(Render_Vulkan,
                     "  GPU: {:.2f} ms/frame executing, {:.2f} ms/frame idle between "
                     "submissions ({:.1f} submissions/frame); frame time {:.2f} ms",
                     busy_ns / 1e6 / frames, idle_ns / 1e6 / frames, double(submits) / frames,
                     seconds * 1e3 / frames);
        }
        pipe_stats_frames = stat_frames;
        pipe_stats_draws = stat_draws;
    }
    LOG_INFO(Render_Vulkan, "  command recording (per 10 s): {}", scheduler.TakeRecordingStats());
    {
        // Where the time goes: the command thread's state, and whether the draw recorder was
        // working meanwhile (sampled every ~250 us).
        static constexpr std::array<const char*, 7> names{"waiting for the game to submit",
                                                          "decoding/preparing draws",
                                                          "waiting for the draw recorder",
                                                          "waiting on game/GPU memory (WaitRegMem)",
                                                          "waiting for a flip (display buffer)",
                                                          "running commands (flips, readbacks)",
                                                          "waiting for the GPU"};
        std::array<u64, 14> counts{};
        u64 total = 0;
        for (size_t i = 0; i < counts.size(); ++i) {
            counts[i] = state_samples[i].exchange(0, std::memory_order_relaxed);
            total += counts[i];
        }
        if (total != 0) {
            std::string line;
            u64 recorder_idle = 0;
            for (size_t i = 0; i < names.size(); ++i) {
                const u64 all = counts[i * 2] + counts[i * 2 + 1];
                recorder_idle += counts[i * 2];
                if (all * 200 < total) {
                    continue; // under 0.5%
                }
                line +=
                    fmt::format("{}{} {:.1f}% (recorder idle {:.1f}%)", line.empty() ? "" : "; ",
                                names[i], 100.0 * all / total, 100.0 * counts[i * 2] / total);
            }
            LOG_INFO(Render_Vulkan, "  command thread time: {}", line);
            LOG_INFO(Render_Vulkan, "  draw recorder idle {:.1f}% of the time",
                     100.0 * recorder_idle / total);
        }
    }
    {
        const u64 faults = write_faults.load(std::memory_order_relaxed);
        LOG_INFO(Render_Vulkan, "  guest write faults: {:.0f}/s",
                 (faults - pipe_stats_faults) / seconds);
        pipe_stats_faults = faults;
    }
    {
        // Guest-side waits: contended mutexes and sleeps (the game's own threads).
        namespace GS = Common::GuestStats;
        const u64 waits = GS::mutex_waits.exchange(0, std::memory_order_relaxed);
        const u64 sleeps = GS::sleeps.exchange(0, std::memory_order_relaxed);
        const u64 asked = GS::sleep_requested_ns.exchange(0, std::memory_order_relaxed);
        const u64 slept = GS::sleep_actual_ns.exchange(0, std::memory_order_relaxed);
        LOG_INFO(Render_Vulkan,
                 "  guest waits: {:.0f} mutex waits/s, {:.0f} sleeps/s (asked {:.0f} us, slept "
                 "{:.0f} us on average)",
                 waits / seconds, sleeps / seconds, sleeps ? asked / 1e3 / sleeps : 0.0,
                 sleeps ? slept / 1e3 / sleeps : 0.0);
        const u64 protects = GS::protect_calls.exchange(0, std::memory_order_relaxed);
        const u64 pages = GS::protect_pages.exchange(0, std::memory_order_relaxed);
        const u64 protect_ns = GS::protect_ns.exchange(0, std::memory_order_relaxed);
        const u64 skipped = GS::upload_checks_skipped.exchange(0, std::memory_order_relaxed);
        LOG_INFO(Render_Vulkan,
                 "  page protection: {:.0f} calls/s, {:.0f} pages/s, {:.1f}% of a core; {:.0f} "
                 "read-only uploads/s skipped without the lock",
                 protects / seconds, pages / seconds, protect_ns / seconds / 1e7,
                 skipped / seconds);
    }
    const u64 profiled = profile_draws.load(std::memory_order_relaxed);
    if (profiled != pipe_stats_profile_draws) {
        std::array<s64, static_cast<size_t>(Section::Count)> ns{};
        s64 total = 0;
        for (size_t i = 0; i < ns.size(); ++i) {
            const s64 now_ns = profile_ns[i].load(std::memory_order_relaxed);
            ns[i] = now_ns - pipe_stats_profile_ns[i];
            pipe_stats_profile_ns[i] = now_ns;
            total += ns[i];
        }
        const double draws = double(profiled - pipe_stats_profile_draws);
        std::string line;
        for (size_t i = 0; i < ns.size(); ++i) {
            line +=
                fmt::format(" {} {:.0f}%", SectionNames[i], total ? 100.0 * ns[i] / total : 0.0);
        }
        LOG_INFO(Render_Vulkan, "  draw recorder per draw: {:.2f} us;{}", total / draws / 1000.0,
                 line);
        const u64 a_draws = a_profile_draws.load(std::memory_order_relaxed);
        if (const u64 n = a_draws - pipe_stats_a_draws; n != 0) {
            std::string a_line;
            s64 a_total = 0;
            std::array<s64, static_cast<size_t>(ASection::Count)> a_ns{};
            for (size_t i = 0; i < a_ns.size(); ++i) {
                const s64 v = a_profile_ns[i].load(std::memory_order_relaxed);
                a_ns[i] = v - pipe_stats_a_ns[i];
                pipe_stats_a_ns[i] = v;
                a_total += a_ns[i];
            }
            for (size_t i = 0; i < a_ns.size(); ++i) {
                a_line += fmt::format(" {} {:.2f} us", ASectionNames[i], a_ns[i] / 1000.0 / n);
            }
            LOG_INFO(Render_Vulkan, "  command thread per draw: {:.2f} us;{}", a_total / 1000.0 / n,
                     a_line);
        }
        pipe_stats_a_draws = a_draws;
        pipe_stats_profile_draws = profiled;
    }
    std::vector<const DrainSite*> sites;
    for (const auto& [key, site] : pipe_drain_sites) {
        sites.push_back(&site);
    }
    std::ranges::sort(sites,
                      [](const DrainSite* a, const DrainSite* b) { return a->time > b->time; });
    for (size_t i = 0; i < std::min<size_t>(sites.size(), 8); ++i) {
        const auto* site = sites[i];
        std::string_view function{site->function};
        if (function.size() > 60) {
            function = function.substr(function.size() - 60);
        }
        LOG_INFO(Render_Vulkan,
                 "  waits at line {} opcode {:#x}: {:.0f}/s, {:.1f}% of the time ({})", site->line,
                 site->reason, site->count / seconds, percent(site->time), function);
    }
    pipe_drain_sites.clear();
    pipe_stats_time = now;
    pipe_stats_packets = draw_pipe->packets;
    pipe_stats_drains = draw_pipe->drains;
    pipe_stats_drain_time = draw_pipe->drain_time;
    pipe_stats_busy = busy;
}

std::thread::id Rasterizer::GetGpuCommandProcessorThread() {
    return liverpool->GetGpuCommandProcessorThread();
}

#ifdef __linux__
u32 Rasterizer::GetGpuCommandProcessorThreadId() {
    return liverpool->GetGpuCommandProcessorThreadId();
}
#endif

void Rasterizer::MarkPass(const GraphicsPipeline* pipeline, const RenderState& state) {
    auto* profiler = GpuProfiler::Get();
    if (!profiler || scheduler.IsRenderingWith(state)) {
        return;
    }
    const auto& vs = pipeline->GetStage(Shader::SwStage::Vertex);
    const auto* ps = pipeline->GetStages()[static_cast<u32>(Shader::SwStage::Fragment)];
    u64 key = (u64(state.width) | u64(state.height) << 32) * 0x9E3779B97F4A7C15ull;
    key ^= vs.pgm_hash * 0xFF51AFD7ED558CCDull;
    key ^= (ps ? ps->pgm_hash : 0) * 0xC4CEB9FE1A85EC53ull;
    key ^= u64(state.num_color_attachments) << 7;
    profiler->Mark(key, [&] {
        return fmt::format("pass {}x{} ({} color{}) vs {:08x} ps {:08x}", state.width, state.height,
                           state.num_color_attachments,
                           state.depth_stencil_attachment.has_depth ? " + depth" : "", vs.pgm_hash,
                           ps ? ps->pgm_hash : 0);
    });
}

bool Rasterizer::NoteNewStorageImage(const VideoCore::Image& image,
                                     const VideoCore::ImageViewInfo& view,
                                     const Shader::ImageResource& res, const Shader::Info& stage) {
    namespace GS = Common::GuestStats;
    const auto& info = image.info;
    const bool reads = res.is_atomic || (res.exact_texel_reads & Shader::ExactTexelFetch) != 0;
    const u32 levels = info.resources.levels;
    const u32 layers = info.resources.layers;
    const bool whole_view = view.range.base.level == 0 && view.range.extent.levels >= levels &&
                            view.range.base.layer == 0 && view.range.extent.layers >= layers;
    const bool compute = stage.sw_stage == Shader::SwStage::Compute;
    const bool indirect = GS::upload_packet == 3;
    u64 threads_x = 0, threads_y = 0, threads_z = 0;
    if (compute) {
        const auto& cs = CsRegs();
        threads_x = u64(cs.dim_x) * cs.num_thread_x.full;
        threads_y = u64(cs.dim_y) * cs.num_thread_y.full;
        threads_z = u64(cs.dim_z) * cs.num_thread_z.full;
    }
    const u32 mip = std::min<u32>(view.range.base.level, levels - 1);
    const u64 width = std::max(info.size.width >> mip, 1u);
    const u64 height = std::max(info.size.height >> mip, 1u);
    const u64 depth = info.props.is_volume ? std::max(info.size.depth >> mip, 1u)
                                           : std::max<u32>(view.range.extent.layers, 1u);
    const bool enough_threads =
        compute && !indirect && threads_x * threads_y * threads_z >= width * height * depth;
    const auto use = reads             ? GS::NewStorageUse::AlsoRead
                     : !whole_view     ? GS::NewStorageUse::Part
                     : !enough_threads ? GS::NewStorageUse::Unclear
                                       : GS::NewStorageUse::Whole;
    GS::new_storage_bytes[static_cast<size_t>(use)].fetch_add(info.guest_size,
                                                              std::memory_order_relaxed);
    // Guest memory all zero? 64 places of 64 bytes (a hint, not a proof).
    bool sampled = false, zero = true;
    if (info.guest_size >= 64) {
        for (u32 i = 0; i < 64 && zero; ++i) {
            const u64 offset = (u64(info.guest_size - 64) * i / 63) & ~63ull;
            const VAddr address = info.guest_address + offset;
            if (!memory->IsReadableMapping(address, 64)) {
                continue;
            }
            sampled = true;
            const auto* bytes = std::bit_cast<const u8*>(address);
            zero = std::all_of(bytes, bytes + 64, [](u8 b) { return b == 0; });
        }
    }
    if (sampled && zero) {
        GS::new_storage_zero_bytes.fetch_add(info.guest_size, std::memory_order_relaxed);
    }
    const bool whole = use == GS::NewStorageUse::Whole;
    // Each kind once (shader, format, size), 1 MiB or larger: up to 64 lines.
    if (info.guest_size < 1_MB) {
        return whole;
    }
    static std::mutex seen_mutex;
    static std::vector<u64> seen;
    const u64 key = HashCombine(HashCombine(stage.pgm_hash, u64(info.pixel_format)),
                                (u64(info.size.width) << 40) ^ (u64(info.size.height) << 20) ^
                                    (u64(levels) << 8) ^ layers);
    {
        std::scoped_lock lock{seen_mutex};
        if (seen.size() >= 64 || std::ranges::find(seen, key) != seen.end()) {
            return whole;
        }
        seen.push_back(key);
    }
    static constexpr std::array UseNames{"only written, whole image, enough threads", "also read",
                                         "only part bound", "coverage unclear"};
    LOG_INFO(Render_Vulkan,
             "New image written by {} shader {:#x} (uploaded first): {}x{}x{} {} ({} levels, {} "
             "layers, {:.2f} MiB, {}) at {:#x} | bound: levels {}-{}, layers {}-{} | shader also "
             "reads it: {} (image loads {}, atomics {}) | threads {}x{}x{}{} for {}x{}x{} texels | "
             "guest memory {} | {}",
             compute ? "compute" : "graphics", stage.pgm_hash, info.size.width, info.size.height,
             info.size.depth, vk::to_string(info.pixel_format), levels, layers,
             info.guest_size / 1048576.0, info.props.is_tiled ? "tiled" : "linear",
             info.guest_address, view.range.base.level,
             view.range.base.level + view.range.extent.levels - 1, view.range.base.layer,
             view.range.base.layer + view.range.extent.layers - 1, reads,
             (res.exact_texel_reads & Shader::ExactTexelFetch) != 0, res.is_atomic, threads_x,
             threads_y, threads_z, indirect ? " (indirect: not known)" : "", width, height, depth,
             !sampled ? "not sampled"
             : zero   ? "looks all zero"
                      : "has data",
             UseNames[static_cast<size_t>(use)]);
    return whole;
}

void Rasterizer::MarkDispatch(const ComputePipeline* pipeline) {
    auto* profiler = GpuProfiler::Get();
    if (!profiler) {
        return;
    }
    const auto& cs = pipeline->GetStage(Shader::SwStage::Compute);
    const auto& cs_program = CsRegs();
    profiler->Mark(cs.pgm_hash ^ 0xD15Aull, [&] {
        return fmt::format("dispatch cs {:08x} ({}x{}x{} groups)", cs.pgm_hash, cs_program.dim_x,
                           cs_program.dim_y, cs_program.dim_z);
    });
}

} // namespace Vulkan
