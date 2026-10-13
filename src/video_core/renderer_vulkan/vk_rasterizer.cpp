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
#include "common/debug.h"
#include "common/elf_info.h"
#include "common/guest_stats.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_gpu_profiler.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_hle.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/texture_cache.h"

#include <vk_mem_alloc.h>

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
    if (!EmulatorSettings.IsNullGPU()) {
        liverpool->BindRasterizer(this);
    }
    memory->SetRasterizer(this);

    scheduler.SetSubmitCallback([this](Vulkan::SubmitInfo& info) {
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
        scheduler.EnableThreadedRecording();
        sampler_memo = std::make_unique<std::array<SamplerMemoEntry, SamplerMemoSize>>();
        const_ring = std::make_unique<VideoCore::Buffer>(
            instance, 0, ConstRingSize, VideoCore::MemoryType::Stream, "Constant ring");
        if (const_ring->mapped_data.empty()) {
            const_ring.reset();
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
        db_desc.first = {};
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

    ScopeMarkerBegin(fmt::format("EliminateFastClear:MRT={:#x}:M={:#x}", col_buf.Address(),
                                 col_buf.CmaskAddress()));
    runtime.ClearImage(&image, desc.view_info.range, clear_value);
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
        .color = std::popcount(pipeline->GetGraphicsKey().mrt_mask) >= 1 ? cb_descs[0].first
                                                                         : VideoCore::ImageId{},
        .depth = db_desc.first,
        .num_indices = regs.num_indices,
        .num_instances = regs.num_instances.NumInstances(),
        .indirect = indirect,
    };
    draw_jitter =
        temporal_dlss.OnDraw(instance, runtime, scheduler, texture_cache, velocity_mirror, info);
}

// Fills color target 0 with a copy of another image of the same size and format, with the same
// bookkeeping a draw into it gets.
void Rasterizer::CopyInsteadOfDraw(const BbTemporalDlss::DrawReplacement& replacement) {
    auto& [target_id, desc] = cb_descs[0];
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
    auto* depth = db_desc.first ? &texture_cache.GetImage(db_desc.first) : nullptr;
    const auto layer = db_desc.first ? db_desc.second.view_info.range.base.layer : 0;
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
    scheduler.Record([handle = last_index.handle, offset = last_index.offset,
                      type = last_index.type](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindIndexBuffer(handle, offset, type);
    });
}

void Rasterizer::Draw(bool is_indexed, u32 index_offset) {
    RENDERER_TRACE;
    ++stat_draws;

    // Threaded renderer: stage A selects the pipeline and queues the rest of the draw. Draws
    // FilterDraw handles itself (fast clear elimination, resolve, depth copy, skips) run here.
    if (UseDrawPipe() && FilterDrawPasses()) {
        const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline();
        if (pipeline) {
            QueueDraw(PacketKind::Draw, pipeline, is_indexed, index_offset, {});
        }
        return;
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
    if (DrawPipe::OnStageB()) {
        scheduler.PopPendingOperations();
        if (target_memo_enabled.load(std::memory_order_relaxed)) {
            scheduler.KickRecording(); // nobody holds the command buffer here
        }
    }
    ProfileLap(Section::Pending);
    const auto& regs = Regs();

    PrepareRenderState(pipeline);
    ProfileLap(Section::Targets);
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
    const auto state = BeginRendering(pipeline);
    ProfileLap(Section::BeginRendering);

    BindVertexBuffers(pipeline);
    if (is_indexed) {
        BindIndexBuffer(index_offset);
    }
    ProfileLap(Section::Vertex);

    if (needs_barrier) {
        runtime.FlushBarriers();
    }
    ProfileLap(Section::Barriers);

    pipeline->BindResources(set_writes, push_data);
    ProfileLap(Section::Descriptors);
    UpdateDynamicState(pipeline, is_indexed);
    MarkPass(pipeline, state);
    scheduler.BeginRendering(state);
    ProfileLap(Section::Dynamic);

    const auto& vs_info = pipeline->GetStage(Shader::SwStage::Vertex);
    const auto& fetch_shader = pipeline->GetFetchShader();
    const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs_info, fetch_shader);

    const auto pipeline_handle = pipeline->Handle();
    scheduler.Record([pipeline_handle](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline_handle);
    });

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
    scheduler.Record(record_draw);
    ReplayDisplayCopy(pipeline, state, draw);
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

    const DrawIndirectParams params = {
        .vertex_sgpr_offset = vertex_sgpr_offset,
        .instance_sgpr_offset = instance_sgpr_offset,
    };
    const IndirectArgs args{
        .address = arg_address,
        .count_address = count_address,
        .offset = offset,
        .stride = stride,
        .max_count = max_count,
    };
    if (UseDrawPipe() && FilterDrawPasses()) {
        const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline(params);
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
    if (DrawPipe::OnStageB()) {
        scheduler.PopPendingOperations();
        if (target_memo_enabled.load(std::memory_order_relaxed)) {
            scheduler.KickRecording(); // nobody holds the command buffer here
        }
    }
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
        BindIndexBuffer();
    }

    const auto [buffer, base] =
        buffer_cache.ObtainBuffer(arg_address + offset, stride * max_count, false);
    needs_barrier |= runtime.IsBufferAccessed(buffer, base, stride * max_count);

    const VideoCore::Buffer* count_buffer;
    u64 count_offset;
    if (count_address != 0) {
        std::tie(count_buffer, count_offset) = buffer_cache.ObtainBuffer(count_address, 4, false);
        needs_barrier |= runtime.IsBufferAccessed(count_buffer, count_offset, 4);
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
    if (ExecuteShaderHLE(cs, Regs(), cs_program, *this)) {
        return;
    }

    if (!BindResources(pipeline)) {
        return;
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

    const auto [buffer, base] = buffer_cache.ObtainBuffer(address + offset, size, false);
    needs_barrier |= runtime.IsBufferAccessed(buffer, base, size);

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

void Rasterizer::Finish() {
    DrainDrawPipe();
    scheduler.Finish();
}

void Rasterizer::OnSubmit() {
    ++stat_frames;
    DrainDrawPipe();
    velocity_mirror.SetEnabled(EmulatorSettings.IsBbObjectMotion());
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
        BB::readback_queue.store(pipe_enabled && EmulatorSettings.IsBbReadbackQueue(),
                                 std::memory_order_relaxed);
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
             "GPU-written | measures {} MiB, starts at {}, pressure {}, critical {}",
             images.gc_runs, images.evicted, images.written_back, images.skipped_stuck,
             images.skipped_written, mib(images.gc_used), mib(images.trigger), mib(images.pressure),
             mib(images.critical));
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
    for (size_t i = 0; i < now.states.size(); ++i) {
        now.states[i] = state_samples[i].load(std::memory_order_relaxed);
    }
    for (size_t i = 0; i < now.housekeeping.size(); ++i) {
        now.housekeeping[i] = housekeeping_total[i].load(std::memory_order_relaxed);
    }
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
            "pipelines compiled | {} texture uploads ({:.1f} MiB, {:.1f} ms, copying {:.1f} ms) | "
            "buffer uploads {:.1f} MiB, {:.1f} ms | {} buffer ranges made resident ({:.1f} MiB, "
            "{:.1f} ms) | {} staging allocations ({:.1f} MiB, {:.1f} ms) | {} background "
            "texture copies, waited {:.1f} ms | page protection {} "
            "calls {:.1f} ms, {} write faults | "
            "game: {} files opened, {} mutex waits, {} sleeps ({:.1f} ms) | end of submissions: "
            "buffer tick {:.1f}, downloads {:.1f}, GC {:.1f}, runtime tick {:.1f} ms | "
            "diagnostics logging {:.1f} ms",
            ms, hitch_avg_ms, states.empty() ? "-" : states,
            std::chrono::duration<double, std::milli>(now.busy - last.busy).count(),
            now.packets - last.packets, delta(now.gpu_waits, last.gpu_waits),
            ms_of(delta(now.gpu_wait_ns, last.gpu_wait_ns)), ms_of(slowest_ns), site,
            GS::slowest_wait_line.load(std::memory_order_relaxed),
            delta(now.pipelines, last.pipelines), delta(now.uploads, last.uploads),
            delta(now.upload_bytes, last.upload_bytes) / 1048576.0,
            ms_of(delta(now.upload_ns, last.upload_ns)),
            ms_of(delta(now.image_copy_ns, last.image_copy_ns)),
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
    } else if (ms < 200.0) {
        // Usual frame time, stutters (and long pauses such as loading) left out.
        hitch_avg_ms = hitch_frames == 0 ? ms : hitch_avg_ms * 0.95 + ms * 0.05;
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

    // Bind resource buffers and textures.
    Shader::Backend::Bindings binding{};
    push_data = MakeUserData(Regs());
    bind_stage_ordinal = 0;
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

    return true;
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
        vtx_ranges.push_back(
            {range.base, memory->ClampRangeSize(range.base, range.end - range.base)});
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
            const auto [buffer, offset] = buffer_cache.ObtainBuffer(range.base, range.size, false);
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
            buffer_cache.ObtainBuffer(range.base_address, size, false);
        needs_barrier |= runtime.IsBufferAccessed(range.buffer, range.offset, size);
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

void Rasterizer::BindIndexBuffer(u32 index_offset) {
    const auto& regs = Regs();

    // Figure out index type and size.
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const vk::IndexType index_type = is_index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
    const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
    const VAddr index_address =
        regs.index_base_address.Address<VAddr>() + index_offset * index_size;

    // Bind index buffer.
    const u32 index_buffer_size = regs.num_indices * index_size;
    const auto [buffer, offset] =
        buffer_cache.ObtainBuffer(index_address, index_buffer_size, false);
    needs_barrier |= runtime.IsBufferAccessed(buffer, offset, index_buffer_size);
    last_index = {buffer->Handle(), offset, index_type};
    RecordIndexBinding();
}

void Rasterizer::ResetBindings(bool is_compute) {
    for (auto& image_id : bound_images) {
        texture_cache.GetImage(image_id).binding = {};
    }
    for (const auto [buffer, offset, size, is_written] : bound_buffers) {
        const auto dst_stage = is_compute ? vk::PipelineStageFlagBits2::eComputeShader
                                          : vk::PipelineStageFlagBits2::eAllGraphics;
        const auto write_flag =
            is_written ? vk::AccessFlagBits2::eShaderWrite : vk::AccessFlagBits2::eNone;
        runtime.AccessBuffer(buffer, offset, size, dst_stage,
                             vk::AccessFlagBits2::eShaderRead | write_flag);
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
            if (!desc.IsSpecial() && desc.is_written && texture_cache.ClearMeta(address)) {
                // Assume all slices were updates
                LOG_TRACE(Render_Vulkan, "Metadata update skipped");
                return true;
            }
        }
    }
    return false;
}

bool Rasterizer::IsComputeImageCopy(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = CsRegs();
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() != 2 || !info.images.empty()) {
        return false;
    }

    // Those 2 buffers must both be formatted. One must be source and another destination.
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (!desc0.is_formatted || !desc1.is_formatted || desc0.is_written == desc1.is_written) {
        return false;
    }

    // Buffers must have the same size and each thread of the dispatch must copy 1 dword of data
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
    if (buf0.GetSize() != buf1.GetSize() || cs_pgm.dim_x != (buf0.GetSize() / 256)) {
        return false;
    }

    // Find images the buffer alias
    const auto image0_id = texture_cache.FindImageFromRange(buf0.base_address, buf0.GetSize());
    if (!image0_id) {
        return false;
    }
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image copy must be valid
    VideoCore::Image& image0 = texture_cache.GetImage(image0_id);
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image0.info.guest_size != image1.info.guest_size ||
        image0.info.pitch != image1.info.pitch || image0.info.guest_size != buf0.GetSize() ||
        image0.info.num_bits != image1.info.num_bits) {
        return false;
    }

    // Perform image copy
    VideoCore::Image& src_image = desc0.is_written ? image1 : image0;
    VideoCore::Image& dst_image = desc0.is_written ? image0 : image1;
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
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() != 2 || !info.images.empty()) {
        return false;
    }

    // From those 2 buffers, first must hold the clear vector and second the image being cleared
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (desc0.is_formatted || !desc1.is_formatted || desc0.is_written || !desc1.is_written) {
        return false;
    }

    // First buffer must have size of vec4 and second the size of a single layer
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
    const u32 buf1_bpp = AmdGpu::NumBitsPerBlock(buf1.GetDataFmt());
    if (buf0.GetSize() != 16 || (cs_pgm.dim_x * 128ULL * (buf1_bpp / 8)) != buf1.GetSize()) {
        return false;
    }

    // Find image the buffer alias
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image clear must be valid
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image1.info.guest_size != buf1.GetSize() || image1.info.num_bits != buf1_bpp ||
        image1.info.props.is_depth) {
        return false;
    }

    // Perform image clear
    const float* values = reinterpret_cast<float*>(buf0.base_address);
    const vk::ClearValue clear = {
        .color = {.float32 = std::array<float, 4>{values[0], values[1], values[2], values[3]}},
    };
    const VideoCore::SubresourceRange range = {
        .base =
            {
                .level = 0,
                .layer = 0,
            },
        .extent = image1.info.resources,
    };
    runtime.ClearImage(&image1, range, clear);
    return true;
}

void Rasterizer::BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                             Shader::PushData& push_data) {
    const u64 alignment = instance.StorageMinAlignment();
    for (u32 buffer_index = 0; buffer_index < stage.buffers.size(); ++buffer_index) {
        const auto& desc = stage.buffers[buffer_index];
        if (const auto* prefetched = packet_num_prefetch ? FindPrefetched(buffer_index) : nullptr) {
            // Copied by the GPU command thread into the constant ring (PrefetchBuffers).
            // (The DLSS scene-constant scan reads the same data from ordinary memory: the ring
            // may be write-combined, slow to read from the CPU.)
            const auto* ring = const_ring.get();
            const u32 size = prefetched->size;
            buffer_infos.emplace_back(ring->Handle(), prefetched->offset, size);
            if (desc.IsSpecial()) {
                temporal_dlss.ObserveConstants(runtime, instance, ring, prefetched->offset, size,
                                               stage.pgm_hash, binding.unified);
                temporal_dlss.ObserveSceneConstants(stage.FlatUserData().data(), size);
            } else {
                bound_buffers.emplace_back(ring, prefetched->offset, size, false);
                needs_barrier |= runtime.IsBufferAccessed(ring, prefetched->offset, size, false);
                if (temporal_dlss.Requested()) {
                    temporal_dlss.ObserveConstants(runtime, instance, ring, prefetched->offset,
                                                   size, stage.pgm_hash, binding.unified);
                    temporal_dlss.ObserveSceneConstants(
                        reinterpret_cast<const void*>(desc.GetSharp(stage).base_address), size);
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
                bound_buffers.emplace_back(gds_buf, 0, gds_buf->SizeBytes(), desc.is_written);
            } else if (desc.buffer_type == Shader::BufferType::Flatbuf) {
                auto& vk_buffer = buffer_cache.GetStreamBuffer();
                const auto flat_ud = stage.FlatUserData();
                const u32 ubo_size = flat_ud.size() * sizeof(u32);
                const u64 offset = vk_buffer.Copy(flat_ud.data(), ubo_size, alignment);
                buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
                temporal_dlss.ObserveConstants(runtime, instance, &vk_buffer, offset, ubo_size,
                                               stage.pgm_hash, binding.unified);
                temporal_dlss.ObserveSceneConstants(flat_ud.data(), ubo_size);
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
                const auto [buffer, offset] = buffer_cache.ObtainBuffer(
                    vsharp.base_address, size, desc.is_written, desc.is_formatted);
                const u64 offset_aligned = Common::AlignDown(offset, alignment);
                const u64 adjust = offset - offset_aligned;
                if (adjust % 4 != 0) {
                    LOG_WARNING(Render_Vulkan, "Buffer binding in shader {:#x} isn't dword aligned",
                                stage.pgm_hash);
                }
                push_data.AddOffset(binding.buffer, adjust);
                buffer_infos.emplace_back(buffer->Handle(), offset_aligned, size + adjust);
                bound_buffers.emplace_back(buffer, offset, size, desc.is_written);
                if (desc.is_written) {
                    // Raw storage-buffer writes can also make an aliased cached image stale.
                    texture_cache.InvalidateMemoryFromGPU(vsharp.base_address, size);
                }
                needs_barrier |= runtime.IsBufferAccessed(buffer, offset, size, desc.is_written);
                if (!desc.is_written && temporal_dlss.Requested()) {
                    temporal_dlss.ObserveConstants(runtime, instance, buffer, offset, size,
                                                   stage.pgm_hash, binding.unified);
                    temporal_dlss.ObserveSceneConstants(
                        reinterpret_cast<const void*>(vsharp.base_address), size);
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
    image_bindings.clear();
    const u32 first_image_idx = image_infos.size();
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
    };

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
                auto& [image_id, desc] = image_bindings.emplace_back(memo->image_id, memo->desc);
                texture_cache.TouchFoundImage(image_id);
                finish_binding(image_id, image_desc);
                image_descriptor_array_sizes.push_back(1);
                continue;
            }
        }

        if (texture_cache.IsMeta(tsharp.Address())) {
            LOG_WARNING(Render_Vulkan, "Unexpected metadata read by a shader (texture)");
        }

        const auto data_fmt = tsharp.GetDataFmt();
        const auto num_fmt = tsharp.GetNumberFmt();
        if (tsharp.Address() == 0 || data_fmt == AmdGpu::DataFormat::FormatInvalid) {
            image_bindings.emplace_back(std::piecewise_construct, std::tuple{}, std::tuple{});
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
            image_bindings.emplace_back(std::piecewise_construct, std::tuple{}, std::tuple{});
            image_descriptor_array_sizes.push_back(1);
            continue;
        }

        const u32 num_bindings = image_desc.NumBindings(stage);

        for (auto i = 0; i < num_bindings; i++) {
            const u32 memo_flags = base_flags | (u32(i) << 16);
            TextureMemoEntry* memo = memo_slot(sharp_words, memo_flags);
            const bool hit = memo_hit(memo, sharp_words, memo_flags);
            auto& [image_id, desc] =
                hit ? image_bindings.emplace_back(memo->image_id, memo->desc)
                    : image_bindings.emplace_back(std::piecewise_construct, std::tuple{},
                                                  std::tuple{tsharp, image_desc});
            if (hit) {
                image_memo_hits.fetch_add(1, std::memory_order_relaxed);
                texture_cache.TouchFoundImage(image_id);
            } else {
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

    // Second pass to re-bind images that were updated after binding
    u32 resource_slot = 0;
    for (auto& [image_id, desc] : image_bindings) {
        const u32 slot = resource_slot++;
        bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        if (!image_id) {
            image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        } else {
            if (auto& old_image = texture_cache.GetImage(image_id);
                old_image.binding.needs_rebind) {
                old_image.binding = {};
                image_id = texture_cache.FindImage(desc);
            }

            bound_images.emplace_back(image_id);

            auto& image = texture_cache.GetImage(image_id);
            auto& image_view = texture_cache.FindTexture(image_id, desc);
            const auto binding = image.binding;

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
        std::array<u64, 2> sharp_words{};
        static_assert(sizeof(ssharp) == sizeof(sharp_words));
        std::memcpy(sharp_words.data(), &ssharp, sizeof(sharp_words));
        SamplerMemoEntry* memo = nullptr;
        if (sampler_memo && target_memo_enabled.load(std::memory_order_relaxed)) {
            const u64 hash = ((sharp_words[0] ^ (sharp_words[1] * 0x9E3779B97F4A7C15ull)) ^
                              u64(sampler.is_depth)) *
                             0xFF51AFD7ED558CCDull;
            memo = &(*sampler_memo)[(hash >> 40) % SamplerMemoSize];
        }
        const u64 sampler_generation = texture_cache.SamplerGeneration();
        if (memo && memo->valid && memo->sharp == sharp_words &&
            memo->is_depth == sampler.is_depth && memo->generation == sampler_generation) {
            vk_sampler = memo->handle;
            // Keep it alive: GetSampler's LRU touch, once per garbage collection tick.
            if (memo->touched_gc_tick != texture_cache.GcTick()) {
                texture_cache.TouchSampler(ssharp, sampler.is_depth);
                memo->touched_gc_tick = texture_cache.GcTick();
            }
        } else {
            vk_sampler = texture_cache.GetSampler(ssharp, Regs().ta_bc_base, sampler.is_depth);
            if (memo) {
                *memo = {sharp_words,        sampler.is_depth,       true,
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
        signature.ids[cb] = cb_descs[cb].first;
        if (cb_descs[cb].first) {
            signature.views[cb] = cb_descs[cb].second.view_info;
        }
    }
    signature.ids[AmdGpu::NUM_COLOR_BUFFERS] = db_desc.first;
    if (db_desc.first) {
        signature.views[AmdGpu::NUM_COLOR_BUFFERS] = db_desc.second.view_info;
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
    for (auto cb = 0u; cb < state.num_color_attachments; ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        if (!image_id) {
            state.color_attachments[cb] = {};
            continue;
        }
        auto* image = &texture_cache.GetImage(image_id);
        if (image->binding.needs_rebind) {
            image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
            image = &texture_cache.GetImage(image_id);
        }
        texture_cache.UpdateImage(image_id);
        runtime.SetBackingSamples(image, key.color_samples[cb]);
        const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
        const auto slice = image_view.info.range.base.layer;
        const auto mip = image_view.info.range.base.level;

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

    if (auto image_id = db_desc.first; image_id) {
        auto& desc = db_desc.second;
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

    if (state.num_layers == std::numeric_limits<u16>::max()) {
        state.num_layers = 1;
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
    auto& write_image = texture_cache.GetImage(texture_cache.FindImage(write_desc));

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
        return buffer_cache.ObtainBuffer(address, num_bytes, true);
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
        return buffer_cache.ObtainBuffer(src, num_bytes, false, true);
    }();
    const auto [dst_buffer, dst_offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
        if (dst_gds) {
            return {gds_buffer, dst};
        }
        return buffer_cache.ObtainBuffer(dst, num_bytes, true, true);
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
    buffer_cache.InvalidateMemory(addr, size, assume_locks);
    texture_cache.InvalidateMemory(addr, size);
    return true;
}

bool Rasterizer::ReadMemory(VAddr addr, u64 size, bool assume_locks) {
    DrainDrawPipe();
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    buffer_cache.ReadMemory(addr, size, false, assume_locks);
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

    const auto combined_scissor_value_tl = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::max({scr, s16(win + win_offset), s16(gen + win_offset)});
    };
    const auto combined_scissor_value_br = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::min({scr, s16(win + win_offset), s16(gen + win_offset)});
    };
    const bool enable_offset = !regs.window_scissor.window_offset_disable;

    AmdGpu::Scissor scsr{};
    scsr.top_left_x = combined_scissor_value_tl(
        regs.screen_scissor.top_left_x, s16(regs.window_scissor.top_left_x),
        s16(regs.generic_scissor.top_left_x),
        enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.top_left_y = combined_scissor_value_tl(
        regs.screen_scissor.top_left_y, s16(regs.window_scissor.top_left_y),
        s16(regs.generic_scissor.top_left_y),
        enable_offset ? regs.window_offset.window_y_offset : 0);
    scsr.bottom_right_x = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_x, regs.window_scissor.bottom_right_x,
        regs.generic_scissor.bottom_right_x,
        enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.bottom_right_y = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_y, regs.window_scissor.bottom_right_y,
        regs.generic_scissor.bottom_right_y,
        enable_offset ? regs.window_offset.window_y_offset : 0);

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
            vp_scsr.bottom_right_x = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_x),
                                              regs.viewport_scissors[i].bottom_right_x);
            vp_scsr.bottom_right_y = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_y),
                                              regs.viewport_scissors[i].bottom_right_y);
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
};

/// Bytes of a prepared vertex input section with `count` streams and `ranges` ranges.
constexpr u32 VertexSectionSize(u32 count, u32 ranges) {
    return count * static_cast<u32>(sizeof(vk::VertexInputAttributeDescription2EXT) +
                                    sizeof(vk::VertexInputBindingDescription2EXT) +
                                    sizeof(AmdGpu::Buffer)) +
           AlignPacket(count) + ranges * 16;
}

struct TaskPacket {
    u8 kind;
    u8 pad[3];
    u32 size;
    Rasterizer::OrderedTask task;
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
    if (const_ring && (kind == PacketKind::Draw || kind == PacketKind::DrawIndirect)) {
        PrefetchBuffers(pipeline);
    }
    const u32 num_prefetch = static_cast<u32>(prefetch_scratch.size());
    const bool vertex_prepared =
        (kind == PacketKind::Draw || kind == PacketKind::DrawIndirect) &&
        PrepareVertexInputs(static_cast<const GraphicsPipeline*>(pipeline));
    const u32 num_vertex = vertex_prepared ? static_cast<u32>(vtx_attributes.size()) : 0;
    const u32 num_vertex_ranges = vertex_prepared ? static_cast<u32>(vtx_ranges.size()) : 0;
    // Storage buffers this draw writes: later prefetches must not read them early.
    {
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
                    pending_writes.push_back(
                        {packet_no, vsharp.base_address, vsharp.base_address + vsharp.GetSize()});
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
    for (const auto* info : pipeline->GetStages()) {
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
                     (vertex_prepared ? VertexSectionSize(num_vertex, num_vertex_ranges) : 0);
    u8* const out = draw_pipe->Begin(size);
    u8* at = out;

    auto& header = *reinterpret_cast<PacketHeader*>(at);
    header = PacketHeader{
        .kind = static_cast<u8>(kind),
        .is_indexed = is_indexed,
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
        static_assert(sizeof(PreparedRange) == 16);
        put(vtx_ranges.data(), num_vertex_ranges * sizeof(PreparedRange));
    }
    ASSERT(static_cast<u32>(at - out) == size);
    draw_pipe->Commit(size);
}

u64 Rasterizer::RunInOrder(OrderedTask task, const void* data, u32 size, VAddr write_addr,
                           u64 write_size) {
    if (!UseDrawPipe() || !DrawPipe::OnStageA()) {
        DrainDrawPipe();
        task(*this, static_cast<const u8*>(data));
        return 0;
    }
    if (write_size != 0) {
        pending_writes.push_back({draw_pipe->packets + 1, write_addr, write_addr + write_size});
    }
    const u32 total = sizeof(TaskPacket) + AlignPacket(size);
    u8* out = draw_pipe->Begin(total);
    *reinterpret_cast<TaskPacket*>(out) = {static_cast<u8>(PacketKind::Task), {}, size, task};
    std::memcpy(out + sizeof(TaskPacket), data, size);
    draw_pipe->Commit(total);
    return draw_pipe->packets;
}

u8* Rasterizer::RingAlloc(u64 size, u64 alignment, u64& offset) {
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

    u64 pos = Common::AlignUp(ring_head, alignment);
    if (pos % ConstRingSize + size > ConstRingSize) {
        pos = Common::AlignUp(pos, ConstRingSize);
    }
    if (pos + size > ring_free_until + ConstRingSize) {
        return nullptr; // Full: the draw recorder copies this one as before.
    }
    ring_head = pos + size;
    offset = pos % ConstRingSize;
    return const_ring->mapped_data.data() + offset;
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
            } else {
                memory->CopySparseMemory(guest, dest, size);
            }
            const_ring->Flush(offset, size);
            prefetch_scratch.push_back({static_cast<u8>(stage), static_cast<u8>(index), 0,
                                        static_cast<u32>(size), offset});
            ring_prefetches.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

const Rasterizer::PrefetchedBuffer* Rasterizer::FindPrefetched(u32 buffer_index) const {
    for (u32 i = 0; i < packet_num_prefetch; ++i) {
        const auto& entry = packet_prefetch[i];
        if (entry.stage == bind_stage_ordinal && entry.buffer == buffer_index) {
            return &entry;
        }
    }
    return nullptr;
}

void Rasterizer::RunDrawPacket(void* context, const u8* data, u32 size) {
    auto& self = *static_cast<Rasterizer*>(context);
    if (*data == static_cast<u8>(PacketKind::Task)) {
        const auto& task = *reinterpret_cast<const TaskPacket*>(data);
        task.task(self, data + sizeof(TaskPacket));
        return;
    }
    self.profile_draw = self.diagnostics.load(std::memory_order_relaxed) &&
                        (self.profile_counter++ & 7) == 0 &&
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
    if (header.num_prefetch != 0) {
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
