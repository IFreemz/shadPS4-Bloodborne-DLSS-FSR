// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <bit>
#include <cstring>
#include <optional>
#include <ranges>

#include "common/elf_info.h"
#include "common/guest_stats.h"
#include "common/hash.h"
#include "common/io_file.h"
#include "common/path_util.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_bb_motion_history.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_serialization.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

namespace {
/// Time the GPU command thread waits for a background compile it needs now (stutter reports).
struct WaitTimer {
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~WaitTimer() {
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - start)
                            .count();
        Common::GuestStats::background_wait_ns.fetch_add(static_cast<u64>(ns),
                                                         std::memory_order_relaxed);
    }
};
} // Anonymous namespace

using Shader::HwStage;
using Shader::Output;
using Shader::SwStage;

constexpr static auto SpirvVersion1_6 = 0x00010600U;

constexpr static std::array DescriptorHeapSizes = {
    vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 512},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1024},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1024},
};

static u32 MapOutputs(std::span<Shader::OutputMap, 3> outputs, const AmdGpu::VsOutputControl& ctl) {
    u32 num_outputs = 0;

    if (ctl.vs_out_misc_enable) {
        auto& misc_vec = outputs[num_outputs++];
        misc_vec[0] = ctl.use_vtx_point_size ? Output::PointSize : Output::None;
        misc_vec[1] = ctl.use_vtx_edge_flag
                          ? Output::EdgeFlag
                          : (ctl.use_vtx_gs_cut_flag ? Output::GsCutFlag : Output::None);
        misc_vec[2] =
            ctl.use_vtx_kill_flag
                ? Output::KillFlag
                : (ctl.use_vtx_render_target_idx ? Output::RenderTargetIndex : Output::None);
        misc_vec[3] = ctl.use_vtx_viewport_idx ? Output::ViewportIndex : Output::None;
    }

    if (ctl.vs_out_ccdist0_enable) {
        auto& ccdist0 = outputs[num_outputs++];
        ccdist0[0] = ctl.IsClipDistEnabled(0)
                         ? Output::ClipDist0
                         : (ctl.IsCullDistEnabled(0) ? Output::CullDist0 : Output::None);
        ccdist0[1] = ctl.IsClipDistEnabled(1)
                         ? Output::ClipDist1
                         : (ctl.IsCullDistEnabled(1) ? Output::CullDist1 : Output::None);
        ccdist0[2] = ctl.IsClipDistEnabled(2)
                         ? Output::ClipDist2
                         : (ctl.IsCullDistEnabled(2) ? Output::CullDist2 : Output::None);
        ccdist0[3] = ctl.IsClipDistEnabled(3)
                         ? Output::ClipDist3
                         : (ctl.IsCullDistEnabled(3) ? Output::CullDist3 : Output::None);
    }

    if (ctl.vs_out_ccdist1_enable) {
        auto& ccdist1 = outputs[num_outputs++];
        ccdist1[0] = ctl.IsClipDistEnabled(4)
                         ? Output::ClipDist4
                         : (ctl.IsCullDistEnabled(4) ? Output::CullDist4 : Output::None);
        ccdist1[1] = ctl.IsClipDistEnabled(5)
                         ? Output::ClipDist5
                         : (ctl.IsCullDistEnabled(5) ? Output::CullDist5 : Output::None);
        ccdist1[2] = ctl.IsClipDistEnabled(6)
                         ? Output::ClipDist6
                         : (ctl.IsCullDistEnabled(6) ? Output::CullDist6 : Output::None);
        ccdist1[3] = ctl.IsClipDistEnabled(7)
                         ? Output::ClipDist7
                         : (ctl.IsCullDistEnabled(7) ? Output::CullDist7 : Output::None);
    }

    return num_outputs;
}

const Shader::RuntimeInfo& PipelineCache::BuildRuntimeInfo(HwStage stage, SwStage l_stage) {
    auto& info = runtime_infos[u32(l_stage)];
    const auto& regs = liverpool->regs;
    const auto BuildCommon = [&](const auto& program) {
        info.props.fp_denorm_mode32 = program.settings.fp_denorm_mode32;
        info.props.fp_denorm_mode16_64 = program.settings.fp_denorm_mode64;
        info.props.fp_round_mode32 = program.settings.fp_round_mode32;
        info.props.fp_round_mode16_64 = program.settings.fp_round_mode64;
        info.props.num_allocated_vgprs = program.NumVgprs();
        info.props.num_user_data = program.settings.num_user_regs;
        info.props.num_input_vgprs = program.settings.vgpr_comp_cnt;
        info.props.dx10_clamp = program.settings.dx10_clamp;
    };
    info.Initialize(stage, l_stage);
    switch (stage) {
    case HwStage::Local: {
        BuildCommon(regs.ls_program);
        Shader::TessellationDataConstantBuffer tess_constants{};
        const auto* hull_info = infos[u32(SwStage::TessellationControl)];
        hull_info->ReadTessConstantBuffer(tess_constants);
        info.hw.ls.ls_stride = tess_constants.ls_stride;
        break;
    }
    case HwStage::Hull:
        BuildCommon(regs.hs_program);
        break;
    case HwStage::Export:
        BuildCommon(regs.es_program);
        info.hw.es.vertex_data_size = regs.vgt_esgs_ring_itemsize;
        break;
    case HwStage::Geometry: {
        BuildCommon(regs.gs_program);
        info.hw.gs.num_outputs = MapOutputs(info.hw.gs.outputs, regs.vs_output_control);
        info.hw.gs.output_vertices = regs.vgt_gs_max_vert_out;
        info.hw.gs.num_invocations =
            regs.vgt_gs_instance_cnt.IsEnabled() ? regs.vgt_gs_instance_cnt.count : 1;
        if (regs.stage_enable.raw == AmdGpu::ShaderStageEnable::LsHsEsGs) {
            info.hw.gs.in_primitive = [&]() {
                switch (regs.tess_config.topology) {
                case AmdGpu::TessellationTopology::Point:
                    return AmdGpu::PrimitiveType::PointList;
                case AmdGpu::TessellationTopology::Line:
                    return AmdGpu::PrimitiveType::LineList;
                case AmdGpu::TessellationTopology::TriangleCw:
                case AmdGpu::TessellationTopology::TriangleCcw:
                    return AmdGpu::PrimitiveType::TriangleList;
                default:
                    UNREACHABLE();
                }
            }();
        } else {
            info.hw.gs.in_primitive = regs.primitive_type;
        }
        for (u32 stream_id = 0; stream_id < Shader::GsMaxOutputStreams; ++stream_id) {
            info.hw.gs.out_primitive[stream_id] =
                regs.vgt_gs_out_prim_type.GetPrimitiveType(stream_id);
        }
        info.hw.gs.in_vertex_data_size = regs.vgt_esgs_ring_itemsize;
        info.hw.gs.out_vertex_data_size = regs.vgt_gs_vert_itemsize[0];
        info.hw.gs.mode = regs.vgt_gs_mode.mode;
        const auto params_vc = AmdGpu::GetParams(regs.vs_program);
        info.hw.gs.vs_copy = params_vc.code;
        info.hw.gs.vs_copy_hash = params_vc.hash;
        DumpShader(info.hw.gs.vs_copy, info.hw.gs.vs_copy_hash, Shader::HwStage::Vertex, 0,
                   "copy.bin");
        break;
    }
    case HwStage::Vertex: {
        BuildCommon(regs.vs_program);
        info.hw.vs.user_clip_plane_mask = regs.clipper_control.user_clip_plane_enable;
        info.hw.vs.num_outputs = MapOutputs(info.hw.vs.outputs, regs.vs_output_control);
        info.hw.vs.emulate_depth_negative_one_to_one =
            !instance.IsDepthClipControlSupported() &&
            regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW;
        info.hw.vs.clip_disable = regs.IsClipDisabled();
        info.hw.vs.motion_vectors = motion_select ? Shader::MotionVectors::VertexVersion : 0;
        break;
    }
    case HwStage::Fragment: {
        BuildCommon(regs.ps_program);
        info.hw.fs.en_flags = regs.ps_input_ena;
        info.hw.fs.addr_flags = regs.ps_input_addr;
        info.hw.fs.num_inputs = regs.num_interp;
        info.hw.fs.front_face_all_bits = regs.barycentric_control.front_face_all_bits;
        info.hw.fs.depth_before_shader = regs.depth_shader_control.depth_before_shader;
        info.hw.fs.num_samples =
            regs.ps_input_addr.sample_coverage_ena && regs.ps_input_ena.sample_coverage_ena
                ? regs.aa_config.NumSamples()
                : 1;
        info.hw.fs.z_export_format = regs.z_export_format;
        u8 stencil_ref_export_enable = regs.depth_shader_control.stencil_op_val_export_enable |
                                       regs.depth_shader_control.stencil_test_val_export_enable;
        info.hw.fs.mrtz_mask = regs.depth_shader_control.z_export_enable |
                               (stencil_ref_export_enable << 1) |
                               (regs.depth_shader_control.mask_export_enable << 2) |
                               (regs.depth_shader_control.coverage_to_mask_enable << 3);
        const auto& cb0_blend = regs.blend_control[0];
        if (cb0_blend.enable) {
            info.hw.fs.dual_source_blending =
                LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.color_dst_factor) ||
                LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.color_src_factor);
            if (cb0_blend.separate_alpha_blend) {
                info.hw.fs.dual_source_blending |=
                    LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.alpha_dst_factor) ||
                    LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.alpha_src_factor);
            }
        } else {
            info.hw.fs.dual_source_blending = false;
        }
        const auto& ps_inputs = regs.ps_inputs;
        for (u32 i = 0; i < regs.num_interp; i++) {
            info.hw.fs.inputs[i] = {
                .param_index = u16(ps_inputs[i].input_offset),
                .is_default = u16(ps_inputs[i].use_default),
                .is_flat = u16(ps_inputs[i].flat_shade),
                .default_value = u16(ps_inputs[i].default_value),
            };
        }
        for (u32 i = 0; i < Shader::MaxColorBuffers; i++) {
            info.hw.fs.color_buffers[i] = graphics_key.color_buffers[i];
        }
        // Lowered user clip planes ride the same emulation path as guest-exported distances, so
        // the fragment side arms whenever the hardware vertex stage lowers them, keeping its input
        // locations in sync with the shifted vertex outputs.
        const bool lowers_user_clip_planes =
            regs.clipper_control.user_clip_plane_enable &&
            !regs.stage_enable.IsStageEnabled(static_cast<u32>(HwStage::Geometry));
        info.hw.fs.clip_distance_emulation =
            ((regs.vs_output_control.clip_distance_enable &&
              !regs.stage_enable.IsStageEnabled(static_cast<u32>(HwStage::Local))) ||
             lowers_user_clip_planes) &&
            profile.needs_clip_distance_emulation;
        info.hw.fs.motion_vectors = motion_select;
        break;
    }
    case HwStage::Compute: {
        const auto& cs_pgm = liverpool->GetCsRegs();
        info.props.num_user_data = cs_pgm.settings.num_user_regs;
        info.props.num_allocated_vgprs = cs_pgm.settings.num_vgprs * 4;
        info.props.fp_denorm_mode32 = cs_pgm.settings.fp_denorm_mode32;
        info.props.fp_denorm_mode16_64 = cs_pgm.settings.fp_denorm_mode64;
        info.props.fp_round_mode32 = cs_pgm.settings.fp_round_mode32;
        info.props.fp_round_mode16_64 = cs_pgm.settings.fp_round_mode64;
        info.props.dx10_clamp = cs_pgm.settings.dx10_clamp;
        info.hw.cs.workgroup_size = {cs_pgm.num_thread_x.full, cs_pgm.num_thread_y.full,
                                     cs_pgm.num_thread_z.full};
        info.hw.cs.tgid_enable = {cs_pgm.IsTgidEnabled(0), cs_pgm.IsTgidEnabled(1),
                                  cs_pgm.IsTgidEnabled(2)};
        info.hw.cs.shared_memory_size = cs_pgm.SharedMemSize();
        break;
    }
    default:
        break;
    }
    switch (l_stage) {
    case SwStage::Vertex:
        info.sw.vs.step_rate_0 = regs.vgt_instance_step_rate_0;
        info.sw.vs.step_rate_1 = regs.vgt_instance_step_rate_1;
        info.sw.vs.vertex_sgpr_offset = draw_indirect_params.vertex_sgpr_offset;
        info.sw.vs.instance_sgpr_offset = draw_indirect_params.instance_sgpr_offset;
        info.sw.vs.is_indirect = draw_indirect_params.is_indirect;
        info.sw.vs.tess_emulated_primitive =
            regs.primitive_type == AmdGpu::PrimitiveType::RectList ||
            regs.primitive_type == AmdGpu::PrimitiveType::QuadList;
        break;
    case SwStage::TessellationControl: {
        info.sw.tcs.num_input_control_points = regs.ls_hs_config.hs_input_control_points;
        info.sw.tcs.num_threads = regs.ls_hs_config.hs_output_control_points;
        info.sw.tcs.tess_type = regs.tess_config.type;
        info.sw.tcs.offchip_lds_enable = regs.hs_program.settings.oc_lds_en;
        break;
    }
    case SwStage::TessellationEval: {
        info.sw.tes.tess_type = regs.tess_config.type;
        info.sw.tes.tess_topology = regs.tess_config.topology;
        info.sw.tes.tess_partitioning = regs.tess_config.partitioning;
        break;
    }
    default:
        break;
    }
    return info;
}

PipelineCache::PipelineCache(const Instance& instance_, Scheduler& scheduler_,
                             AmdGpu::Liverpool* liverpool_, u32 sparse_page_shift)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      desc_heap{instance, scheduler.GetWorkSemaphore(), DescriptorHeapSizes} {
    const auto& vk12_props = instance.GetVk12Properties();
    profile = Shader::Profile{
        .max_viewport_width = instance.GetMaxViewportWidth(),
        .max_viewport_height = instance.GetMaxViewportHeight(),
        .max_shared_memory_size = instance.MaxComputeSharedMemorySize(),
        .supported_spirv = SpirvVersion1_6,
        .subgroup_size = instance.SubgroupSize(),
        .sparse_page_shift = sparse_page_shift,
        .support_int8 = instance.IsShaderInt8Supported(),
        .support_int16 = instance.IsShaderInt16Supported(),
        .support_int64 = instance.IsShaderInt64Supported(),
        .support_float16 = instance.IsShaderFloat16Supported(),
        .support_float64 = instance.IsShaderFloat64Supported(),
        .supports_denorm_behavior_independence =
            vk12_props.denormBehaviorIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .supports_rounding_mode_independence =
            vk12_props.roundingModeIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .support_fp16_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat16),
        .support_fp16_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat16),
        .support_fp16_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat16),
        .support_fp32_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat32),
        .support_fp32_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat32),
        .support_fp32_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat32),
        .support_fp64_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat64),
        .support_fp64_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat64),
        .support_fp64_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat64),
        .support_fp16_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat16),
        .support_fp32_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat32),
        .support_fp64_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat64),
        .supports_image_load_store_lod = instance_.IsImageLoadStoreLodSupported(),
        .supports_native_cube_calc = instance_.IsAmdGcnShaderSupported(),
        .supports_trinary_minmax = instance_.IsAmdShaderTrinaryMinMaxSupported(),
        .supports_buffer_fp32_atomic_min_max =
            instance_.IsShaderAtomicFloatBuffer32MinMaxSupported(),
        .supports_image_fp32_atomic_min_max = instance_.IsShaderAtomicFloatImage32MinMaxSupported(),
        .supports_buffer_int64_atomics = instance_.IsBufferInt64AtomicsSupported(),
        .supports_shared_int64_atomics = instance_.IsSharedInt64AtomicsSupported(),
        .supports_workgroup_explicit_memory_layout =
            instance_.IsWorkgroupMemoryExplicitLayoutSupported(),
        .supports_amd_shader_explicit_vertex_parameter =
            instance_.IsAmdShaderExplicitVertexParameterSupported(),
        .supports_fragment_shader_barycentric = instance_.IsFragmentShaderBarycentricSupported(),
        .supports_shader_subgroup_clock = instance_.IsShaderSubgroupClockSupported(),
        .needs_manual_interpolation = instance.IsFragmentShaderBarycentricSupported() &&
                                      instance.GetDriverID() == vk::DriverId::eNvidiaProprietary,
        .needs_lds_barriers = instance.GetDriverID() == vk::DriverId::eNvidiaProprietary ||
                              instance.GetDriverID() == vk::DriverId::eMesaKosmickrisp,
        .needs_buffer_offsets = instance.StorageMinAlignment() > 4,
        .needs_unorm_fixup = instance.GetDriverID() == vk::DriverId::eMesaKosmickrisp,
        .needs_clip_distance_emulation = instance.GetDriverID() == vk::DriverId::eNvidiaProprietary,
        .supports_shader_stencil_export = instance_.IsShaderStencilExportSupported(),
    };
    // The driver's own cache from the last session first: the warm-up builds every cached
    // pipeline with it, much faster when the driver has compiled them before.
    pipeline_cache = CreateDriverCache();
    WarmUp();

    // Background pipeline compiles, and saves of the driver's cache.
    const u32 hardware = std::max(1u, std::thread::hardware_concurrency());
    const u32 workers = std::clamp(hardware / 4, 1u, 4u);
    for (u32 i = 0; i < workers; ++i) {
        compile_threads.emplace_back([this](std::stop_token stop) { CompileWorker(stop); });
    }
    SaveDriverCacheInBackground();
}

PipelineCache::~PipelineCache() {
    // Workers finish their current job first, then what was compiled this session is kept.
    for (auto& thread : compile_threads) {
        thread.request_stop();
    }
    const bool had_workers = !compile_threads.empty();
    compile_threads.clear();
    if (had_workers && EmulatorSettings.IsPipelineCacheEnabled() &&
        num_new_pipelines != driver_saved_pipelines) {
        SaveDriverCache();
    }
}

void PipelineCache::SaveDriverCache() {
    auto [result, data] = instance.GetDevice().getPipelineCacheData(*pipeline_cache);
    if (result == vk::Result::eSuccess && !data.empty()) {
        const auto path = DriverCachePath();
        auto temp = path;
        temp += ".tmp";
        bool written = false;
        {
            const Common::FS::IOFile file{temp, Common::FS::FileAccessMode::Create};
            written = file.IsOpen() && file.WriteSpan(std::span<const u8>{data}) == data.size();
        }
        std::error_code ec;
        if (written) {
            std::filesystem::rename(temp, path, ec);
        }
        if (written && !ec) {
            LOG_INFO(Render_Vulkan, "Driver pipeline cache: saved {} KiB", data.size() >> 10);
        }
    }
}

std::filesystem::path PipelineCache::DriverCachePath() const {
    // With the game's shader cache: its folder, or beside its archive.
    const auto cache_dir = Common::FS::GetUserPath(Common::FS::PathType::CacheDir);
    const auto serial = std::string{Common::ElfInfo::Instance().GameSerial()};
    if (EmulatorSettings.IsPipelineCacheArchived()) {
        return cache_dir / fmt::format("{}_driver_pipelines.bin", serial);
    }
    return cache_dir / serial / "driver_pipelines.bin";
}

vk::UniquePipelineCache PipelineCache::CreateDriverCache() {
    const auto device = instance.GetDevice();
    std::vector<u8> data;
    if (EmulatorSettings.IsPipelineCacheEnabled()) {
        const auto path = DriverCachePath();
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
        // Moved from beside the game's cache folder by an earlier build.
        const auto old_path =
            Common::FS::GetUserPath(Common::FS::PathType::CacheDir) /
            fmt::format("{}_driver_pipelines.bin", Common::ElfInfo::Instance().GameSerial());
        if (old_path != path && std::filesystem::exists(old_path, ec)) {
            std::filesystem::rename(old_path, path, ec);
        }
        const Common::FS::IOFile file{path, Common::FS::FileAccessMode::Read};
        if (file.IsOpen()) {
            data.resize(file.GetSize());
            if (file.ReadSpan(std::span<u8>{data}) != data.size()) {
                data.clear();
            }
        }
    }
    // Only the driver and GPU that wrote it can use it (the header says which).
    struct Header {
        u32 size;
        u32 version;
        u32 vendor;
        u32 device;
        std::array<u8, VK_UUID_SIZE> uuid;
    };
    if (data.size() >= sizeof(Header)) {
        Header header;
        std::memcpy(&header, data.data(), sizeof(header));
        const auto props = instance.GetPhysicalDevice().getProperties();
        if (header.vendor != props.vendorID || header.device != props.deviceID ||
            std::memcmp(header.uuid.data(), props.pipelineCacheUUID.data(), VK_UUID_SIZE) != 0) {
            LOG_INFO(Render_Vulkan, "Driver pipeline cache: from another GPU or driver, not used");
            data.clear();
        }
    }
    if (!data.empty()) {
        auto [result, cache] = device.createPipelineCacheUnique(
            {.initialDataSize = data.size(), .pInitialData = data.data()});
        if (result == vk::Result::eSuccess) {
            LOG_INFO(Render_Vulkan, "Driver pipeline cache: loaded {} KiB", data.size() >> 10);
            return std::move(cache);
        }
    }
    auto [cache_result, cache] = device.createPipelineCacheUnique({});
    ASSERT_MSG(cache_result == vk::Result::eSuccess, "Failed to create pipeline cache: {}",
               vk::to_string(cache_result));
    return std::move(cache);
}

void PipelineCache::SaveDriverCacheInBackground() {
    if (!EmulatorSettings.IsPipelineCacheEnabled() || compile_threads.empty() ||
        driver_save_queued.exchange(true)) {
        return;
    }
    driver_saved_pipelines = num_new_pipelines;
    driver_saved_at = std::chrono::steady_clock::now();
    {
        std::scoped_lock lock{compile_mutex};
        compile_jobs.emplace_back([this] {
            SaveDriverCache();
            driver_save_queued.store(false);
        });
    }
    compile_cv.notify_one();
}

void PipelineCache::CompileWorker(std::stop_token stop) {
    Common::SetCurrentThreadName("shadPS4:PipelineCompile");
    // Below the game and the renderer threads: compiles only use otherwise idle cores.
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
    while (true) {
        std::function<void()> job;
        {
            std::unique_lock lock{compile_mutex};
            if (!compile_cv.wait(lock, stop, [this] { return !compile_jobs.empty(); })) {
                return;
            }
            job = std::move(compile_jobs.front());
            compile_jobs.pop_front();
        }
        job();
    }
}

void PipelineCache::NoteDepthTarget() {
    const auto& regs = liverpool->regs;
    const u64 frame = game_frames.load(std::memory_order_relaxed);
    if (frame != depth_target_frame) {
        const u64 shift = std::min<u64>(frame - depth_target_frame, 32);
        for (auto& target : depth_targets) {
            target.frames = shift >= 32 ? 0 : target.frames << shift;
        }
        depth_target_frame = frame;
        last_depth_target = 0;
    }
    if (!regs.depth_control.depth_enable || !regs.depth_buffer.DepthValid()) {
        return;
    }
    const VAddr address = regs.depth_buffer.DepthAddress();
    if (address == last_depth_target) {
        return;
    }
    last_depth_target = address;
    auto it = std::ranges::find(depth_targets, address, &DepthTargetUse::address);
    if (it == depth_targets.end()) {
        it = std::ranges::min_element(depth_targets, {}, [](const DepthTargetUse& target) {
            return std::popcount(target.frames);
        });
        *it = {address, 0};
    }
    it->frames |= 1;
}

bool PipelineCache::MayDefer() const {
    if (!EmulatorSettings.IsBbBackgroundPipelines() || compile_threads.empty() ||
        !instance.IsVertexInputDynamicState() || EmulatorSettings.IsShaderCollect() ||
        EmulatorSettings.IsDumpShaders()) {
        return false;
    }
    // Scene geometry only: depth tested, into a depth target used in each of the last 8 game
    // frames. Full-screen passes, one-off passes and loading screens are compiled at once.
    const auto& regs = liverpool->regs;
    if (!regs.depth_control.depth_enable || !regs.depth_buffer.DepthValid()) {
        return false;
    }
    const auto it = std::ranges::find(depth_targets, regs.depth_buffer.DepthAddress(),
                                      &DepthTargetUse::address);
    return it != depth_targets.end() && ((it->frames >> 1) & 0xFFu) == 0xFFu;
}

void PipelineCache::StartBackgroundCompile() {
    pending_pipelines.emplace(graphics_key, u8{0});
    const u64 hash = std::hash<GraphicsPipelineKey>{}(graphics_key);
    LOG_INFO(Render_Vulkan, "Compiling graphics pipeline {:#x} in the background", hash);
    std::optional<Shader::Gcn::FetchShaderData> fetch;
    if (fetch_shader) {
        fetch = *fetch_shader;
    }
    {
        std::scoped_lock lock{compile_mutex};
        compile_jobs.emplace_back([this, key = graphics_key, hash, stage_infos = infos,
                                   stage_runtime = runtime_infos, stage_modules = modules,
                                   fetch = std::move(fetch)]() mutable {
            const auto start = std::chrono::steady_clock::now();
            GraphicsPipeline::SerializationSupport sdata{};
            auto pipeline = std::make_unique<GraphicsPipeline>(
                instance, scheduler, desc_heap, profile, key, *pipeline_cache, stage_infos,
                stage_runtime, fetch ? &*fetch : nullptr, stage_modules, sdata, false);
            const double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count();
            std::scoped_lock done{finished_mutex};
            finished_pipelines.push_back({key, hash, std::move(pipeline), std::move(sdata), ms});
            num_finished.fetch_add(1, std::memory_order_release);
        });
    }
    compile_cv.notify_one();
    Common::GuestStats::background_pipelines_started.fetch_add(1, std::memory_order_relaxed);
}

void PipelineCache::CollectBackgroundPipelines(bool shaders) {
    std::vector<FinishedPipeline> done;
    std::vector<FinishedShader> done_shaders;
    {
        std::scoped_lock lock{finished_mutex};
        done.swap(finished_pipelines);
        if (shaders) {
            done_shaders.swap(finished_shaders);
        }
        num_finished.store(static_cast<u32>(finished_shaders.size()), std::memory_order_relaxed);
    }
    for (auto& finished : done_shaders) {
        CollectBackgroundShader(finished);
    }
    for (auto& finished : done) {
        pending_pipelines.erase(finished.key);
        const auto [it, is_new] = graphics_pipelines.try_emplace(finished.key);
        if (!is_new) {
            continue;
        }
        it.value() = std::move(finished.pipeline);
        RegisterPipelineData(finished.key, finished.hash, finished.sdata);
        ++num_new_pipelines;
        LOG_INFO(Render_Vulkan, "Graphics pipeline {:#x} ready ({:.1f} ms in the background)",
                 finished.hash, finished.ms);
    }
    if (num_new_pipelines != driver_saved_pipelines &&
        std::chrono::steady_clock::now() - driver_saved_at > std::chrono::seconds{30}) {
        SaveDriverCacheInBackground();
    }
}

const GraphicsPipeline* PipelineCache::GetGraphicsPipeline(const DrawIndirectParams params,
                                                           bool may_defer) {
    draw_indirect_params = params;
    // Finished background work first: shader variants must not be added while this draw's
    // stages point into them.
    if (num_finished.load(std::memory_order_acquire) != 0) {
        CollectBackgroundPipelines();
    }
    if (may_defer) {
        NoteDepthTarget();
    }
    translate_in_background = may_defer && MayDefer();
    stage_deferred = false;
    const bool refreshed = RefreshGraphicsKey();
    translate_in_background = false;
    if (!refreshed) {
        if (std::exchange(stage_deferred, false)) {
            Common::GuestStats::background_skipped_draws.fetch_add(1, std::memory_order_relaxed);
        }
        return nullptr;
    }
    if (const auto found = graphics_pipelines.find(graphics_key);
        found != graphics_pipelines.end()) {
        return found->second.get();
    }
    if (pending_pipelines.contains(graphics_key)) {
        if (may_defer && MayDefer()) {
            Common::GuestStats::background_skipped_draws.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }
        // Needed now (not a skippable draw): wait for the background compile.
        const WaitTimer wait;
        while (pending_pipelines.contains(graphics_key)) {
            std::this_thread::yield();
            CollectBackgroundPipelines(false);
        }
        return graphics_pipelines.find(graphics_key)->second.get();
    }
    if (may_defer && MayDefer()) {
        StartBackgroundCompile();
        Common::GuestStats::background_skipped_draws.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    if (num_new_pipelines != driver_saved_pipelines &&
        std::chrono::steady_clock::now() - driver_saved_at > std::chrono::seconds{30}) {
        SaveDriverCacheInBackground();
    }
    const auto [it, is_new] = graphics_pipelines.try_emplace(graphics_key);
    if (is_new) {
        const auto pipeline_hash = std::hash<GraphicsPipelineKey>{}(graphics_key);
        LOG_INFO(Render_Vulkan, "Compiling graphics pipeline {:#x}", pipeline_hash);
        Common::GuestStats::pipelines_compiled.fetch_add(1, std::memory_order_relaxed);

        GraphicsPipeline::SerializationSupport sdata{};
        it.value() = std::make_unique<GraphicsPipeline>(
            instance, scheduler, desc_heap, profile, graphics_key, *pipeline_cache, infos,
            runtime_infos, fetch_shader, modules, sdata, false);

        RegisterPipelineData(graphics_key, pipeline_hash, sdata);
        ++num_new_pipelines;

        if (EmulatorSettings.IsShaderCollect()) {
            for (auto stage = 0; stage < MaxShaderStages; ++stage) {
                if (infos[stage]) {
                    auto& m = modules[stage];
                    module_related_pipelines[m].emplace_back(graphics_key);
                }
            }
        }
    }
    return it->second.get();
}

const ComputePipeline* PipelineCache::GetComputePipeline() {
    if (!RefreshComputeKey()) {
        return nullptr;
    }
    const auto [it, is_new] = compute_pipelines.try_emplace(compute_key);
    if (is_new) {
        const auto pipeline_hash = std::hash<ComputePipelineKey>{}(compute_key);
        LOG_INFO(Render_Vulkan, "Compiling compute pipeline {:#x}", pipeline_hash);
        Common::GuestStats::pipelines_compiled.fetch_add(1, std::memory_order_relaxed);

        ComputePipeline::SerializationSupport sdata{};
        it.value() = std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile,
                                                       *pipeline_cache, compute_key, *infos[0],
                                                       modules[0], sdata, false);
        RegisterPipelineData(compute_key, sdata);
        ++num_new_pipelines;

        if (EmulatorSettings.IsShaderCollect()) {
            auto& m = modules[0];
            module_related_pipelines[m].emplace_back(compute_key);
        }
    }
    return it->second.get();
}

bool PipelineCache::RefreshGraphicsKey() {
    std::memset(&graphics_key, 0, sizeof(GraphicsPipelineKey));
    const auto& regs = liverpool->regs;
    auto& key = graphics_key;

    const bool db_enabled = regs.depth_buffer.DepthValid() || regs.depth_buffer.StencilValid();

    key.z_format = regs.depth_buffer.DepthValid() ? regs.depth_buffer.z_info.format
                                                  : AmdGpu::DepthBuffer::ZFormat::Invalid;
    key.stencil_format = regs.depth_buffer.StencilValid()
                             ? regs.depth_buffer.stencil_info.format
                             : AmdGpu::DepthBuffer::StencilFormat::Invalid;
    key.depth_clamp_enable = !regs.depth_render_override.disable_viewport_clamp;
    key.depth_clip_enable = regs.clipper_control.ZclipEnable();
    key.clip_space = regs.clipper_control.clip_space;
    key.provoking_vtx_last = regs.polygon_control.provoking_vtx_last;
    key.prim_type = regs.primitive_type;
    key.polygon_mode = regs.polygon_control.PolyMode();
    key.patch_control_points =
        regs.stage_enable.hs_en ? regs.ls_hs_config.hs_input_control_points : 0;
    key.logic_op = regs.color_control.rop3;
    key.depth_samples = db_enabled ? regs.depth_buffer.NumSamples() : 1;
    key.num_samples = key.depth_samples;
    key.cb_shader_mask = regs.color_shader_mask;

    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;

    // Character motion vectors: only G-buffer draws (5+ targets, single-sampled depth) whose
    // vertex shader has a bone palette get the extra attachment.
    bool motion_possible = false;
    if (Shader::MotionVectors::enabled && !skip_cb_binding) {
        u32 bound = 0;
        for (s32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS; ++cb) {
            bound += (regs.color_buffers[cb] && regs.color_target_mask.GetMask(cb)) ? 1 : 0;
        }
        motion_possible = bound >= 5 && !regs.color_buffers[Shader::MotionVectors::Output] &&
                          regs.depth_buffer.DepthValid() && regs.depth_buffer.NumSamples() == 1 &&
                          !regs.IsClipDisabled() &&
                          regs.stage_enable.raw == AmdGpu::ShaderStageEnable::VgtStages::Vs;
    }
    motion_select = false;

    // First pass to fill render target information needed by shader recompiler
    for (s32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS && !skip_cb_binding; ++cb) {
        const auto& col_buf = regs.color_buffers[cb];
        if (!col_buf || !regs.color_target_mask.GetMask(cb)) {
            // No attachment bound or writing to it is disabled.
            continue;
        }

        // Fill color target information
        auto& color_buffer = key.color_buffers[cb];
        color_buffer.data_format = col_buf.GetDataFmt();
        color_buffer.num_format = col_buf.GetNumberFmt();
        color_buffer.num_conversion = col_buf.GetNumberConversion();
        color_buffer.export_format = regs.color_export_format.GetFormat(cb);
        color_buffer.swizzle = col_buf.Swizzle();

        const auto& bc = regs.blend_control[cb];
        color_buffer.blend_self_scale =
            bc.enable && !col_buf.info.blend_bypass &&
            (bc.color_func == AmdGpu::BlendControl::BlendFunc::Min ||
             bc.color_func == AmdGpu::BlendControl::BlendFunc::Max) &&
            bc.color_src_factor == AmdGpu::BlendControl::BlendFactor::SrcColor &&
            bc.color_dst_factor == AmdGpu::BlendControl::BlendFactor::DstColor;
    }

    // Character motion: start from this vertex shader's last choice, so the stages are
    // normally built once; they are rebuilt only when the choice turns out different.
    const u64 motion_vs = motion_possible ? regs.vs_program.Address<u64>() : 0;
    if (motion_possible) {
        const auto it = motion_choices.find(motion_vs);
        motion_select = it != motion_choices.end() && it->second;
    }

    // Compile and bind shader stages
    if (!RefreshGraphicsStages()) {
        motion_select = false;
        return false;
    }
    if (motion_possible) {
        bool wanted = false;
        const auto* vs = infos[static_cast<u32>(Shader::SwStage::Vertex)];
        const auto* fs = infos[static_cast<u32>(Shader::SwStage::Fragment)];
        if (vs && fs && !(fs->mrt_mask & (1u << Shader::MotionVectors::Output))) {
            for (const auto& resource : vs->buffers) {
                if (resource.IsSpecial()) {
                    continue;
                }
                const auto buffer = resource.GetSharp(*vs);
                if (buffer.Valid() && buffer.GetStride() == 16 &&
                    Motion::ClassifyBuffer(u32(buffer.GetSize())) != Motion::BufferRole::Other) {
                    wanted = true;
                    break;
                }
            }
        }
        motion_choices[motion_vs] = wanted;
        if (wanted != motion_select) {
            motion_select = wanted;
            if (!RefreshGraphicsStages()) {
                motion_select = false;
                return false;
            }
        }
    }

    // Second pass to mask out render targets not written by shader and fill remaining info
    u8 color_samples = 0;
    bool all_color_samples_same = true;
    for (s32 cb = 0; cb < key.num_color_attachments && !skip_cb_binding; ++cb) {
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (!col_buf || !target_mask) {
            continue;
        }
        if ((key.mrt_mask & (1u << cb)) == 0) {
            std::memset(&key.color_buffers[cb], 0, sizeof(Shader::PsColorBuffer));
            continue;
        }

        // Fill color blending information
        if (regs.blend_control[cb].enable && !col_buf.info.blend_bypass) {
            key.blend_controls[cb] = regs.blend_control[cb];
        }

        // Apply swizzle to target mask
        key.write_masks[cb] =
            vk::ColorComponentFlags{key.color_buffers[cb].swizzle.ApplyMask(target_mask)};

        // Fill color samples
        const u8 prev_color_samples = std::exchange(color_samples, col_buf.NumSamples());
        all_color_samples_same &= color_samples == prev_color_samples || prev_color_samples == 0;
        key.color_samples[cb] = color_samples;
        key.num_samples = std::max(key.num_samples, color_samples);
    }

    // Every G-buffer pipeline gets the motion attachment (written only by skinned draws), so
    // the pass is not restarted each time the game switches between skinned and static draws.
    if (motion_possible && !(key.mrt_mask & (1u << Shader::MotionVectors::Output))) {
        constexpr u32 mv = Shader::MotionVectors::Output;
        key.motion_vectors = motion_select ? 1 : 0;
        key.motion_slot = 1;
        key.mrt_mask |= 1u << mv;
        key.num_color_attachments = mv + 1;
        auto& color_buffer = key.color_buffers[mv];
        color_buffer = {};
        color_buffer.data_format = AmdGpu::DataFormat::Format32_32_32_32;
        color_buffer.num_format = AmdGpu::NumberFormat::Float;
        color_buffer.swizzle = AmdGpu::IdentityMapping;
        key.blend_controls[mv] = {};
        key.write_masks[mv] =
            motion_select ? vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                                vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
                          : vk::ColorComponentFlags{};
        key.color_samples[mv] = 1;
    }
    motion_select = false;

    // Force all color samples to match depth samples to avoid unsupported MSAA configuration
    if (color_samples != 0) {
        const bool depth_mismatch = db_enabled && color_samples != key.depth_samples;
        if (!all_color_samples_same && !instance.IsMixedAnySamplesSupported() ||
            all_color_samples_same && depth_mismatch && !instance.IsMixedDepthSamplesSupported()) {
            key.color_samples.fill(key.depth_samples);
            key.num_samples = key.depth_samples;
        }
    }

    return true;
}

bool PipelineCache::RefreshGraphicsStages() {
    const auto& regs = liverpool->regs;
    auto& key = graphics_key;
    fetch_shader = nullptr;

    Shader::Backend::Bindings binding{};
    const auto bind_stage = [&](HwStage stage_in, SwStage stage_out) -> bool {
        const auto stage_in_idx = static_cast<u32>(stage_in);
        const auto stage_out_idx = static_cast<u32>(stage_out);
        if (!regs.stage_enable.IsStageEnabled(stage_in_idx)) {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            return false;
        }

        const auto* pgm = regs.ProgramForStage(stage_in_idx);
        if (!pgm || !pgm->Address<u32*>()) {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            return false;
        }

        const auto params = AmdGpu::GetParams(*pgm);
        std::tie(infos[stage_out_idx], modules[stage_out_idx], key.stage_hashes[stage_out_idx]) =
            GetProgram(stage_in, stage_out, params, binding);
        // A shader being translated in the background: this draw is skipped.
        return !stage_deferred;
    };

    infos.fill(nullptr);
    modules.fill(nullptr);
    fetch_program = nullptr;

    bind_stage(HwStage::Fragment, SwStage::Fragment);
    if (stage_deferred) {
        return false;
    }

    const auto* fs_info = infos[static_cast<u32>(SwStage::Fragment)];
    key.mrt_mask = fs_info ? fs_info->mrt_mask : 0u;
    key.num_color_attachments = std::bit_width(key.mrt_mask);

    switch (regs.stage_enable.raw) {
    case AmdGpu::ShaderStageEnable::VgtStages::EsGs:
        if (!instance.IsGeometryStageSupported()) {
            LOG_WARNING(Render_Vulkan, "Geometry shader stage unsupported, skipping");
            return false;
        }
        if (regs.vgt_gs_mode.onchip || regs.vgt_strmout_config.raw) {
            LOG_WARNING(Render_Vulkan, "Geometry shader features unsupported, skipping");
            return false;
        }
        if (!bind_stage(HwStage::Export, SwStage::Vertex)) {
            return false;
        }
        if (!bind_stage(HwStage::Geometry, SwStage::Geometry)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::LsHs:
        if (!instance.IsTessellationSupported()) {
            return false;
        }
        if (!bind_stage(HwStage::Hull, SwStage::TessellationControl)) {
            return false;
        }
        if (!bind_stage(HwStage::Vertex, SwStage::TessellationEval)) {
            return false;
        }
        if (!bind_stage(HwStage::Local, SwStage::Vertex)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::LsHsEsGs:
        if (!instance.IsTessellationSupported()) {
            return false;
        }
        if (!instance.IsGeometryStageSupported()) {
            LOG_WARNING(Render_Vulkan, "Geometry shader stage unsupported, skipping");
            return false;
        }
        if (regs.vgt_gs_mode.onchip || regs.vgt_strmout_config.raw) {
            LOG_WARNING(Render_Vulkan, "Geometry shader features unsupported, skipping");
            return false;
        }
        if (!bind_stage(HwStage::Hull, SwStage::TessellationControl)) {
            return false;
        }
        if (!bind_stage(HwStage::Export, SwStage::TessellationEval)) {
            return false;
        }
        if (!bind_stage(HwStage::Local, SwStage::Vertex)) {
            return false;
        }
        if (!bind_stage(HwStage::Geometry, SwStage::Geometry)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::Vs:
        bind_stage(HwStage::Vertex, SwStage::Vertex);
        break;
    default:
        LOG_WARNING(Render_Vulkan, "unimplemented shader stage {}", (u32)regs.stage_enable.raw);
        return false;
    }
    if (stage_deferred) {
        return false;
    }
    // Taken from the program once all stages are bound (a variant finished in the background
    // and added meanwhile can move the list it is in).
    if (fetch_program) {
        auto& fetch = fetch_program->modules[fetch_perm].spec.fetch_shader_data;
        fetch_shader = fetch.Empty() ? nullptr : &fetch;
    }

    const auto* vs_info = infos[static_cast<u32>(SwStage::Vertex)];
    if (vs_info && fetch_shader && !instance.IsVertexInputDynamicState()) {
        // Without vertex input dynamic state, the pipeline needs to specialize on format.
        // Stride will still be handled outside the pipeline using dynamic state.
        u32 vertex_binding = 0;
        for (const auto& attrib : fetch_shader->attributes) {
            const auto& buffer = attrib.GetSharp(*vs_info);
            ASSERT_MSG(vertex_binding < MaxVertexBufferCount,
                       "Vertex attribute binding count exceeded limit: {} >= {}", vertex_binding,
                       MaxVertexBufferCount);
            key.vertex_buffer_formats[vertex_binding++] =
                Vulkan::LiverpoolToVK::SurfaceFormat(buffer.GetDataFmt(), buffer.GetNumberFmt());
        }
    }

    return true;
}

bool PipelineCache::RefreshComputeKey() {
    Shader::Backend::Bindings binding{};
    const auto& cs_pgm = liverpool->GetCsRegs();
    const auto cs_params = AmdGpu::GetParams(cs_pgm);
    std::tie(infos[0], modules[0], compute_key.value) =
        GetProgram(HwStage::Compute, SwStage::Compute, cs_params, binding);
    return true;
}

vk::ShaderModule PipelineCache::CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                              const std::span<const u32>& code, size_t perm_idx,
                                              Shader::Backend::Bindings& binding) {
    LOG_INFO(Render_Vulkan, "Compiling {} shader {:#x} {}", info.hw_stage, info.pgm_hash,
             perm_idx != 0 ? "(permutation)" : "");
    DumpShader(code, info.pgm_hash, info.hw_stage, perm_idx, "bin");

    const auto ir_program = Shader::TranslateProgram(code, pools, info, runtime_info, profile);
    auto spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir_program, binding);
    DumpShader(spv, info.pgm_hash, info.hw_stage, perm_idx, "spv");

    vk::ShaderModule module;

    auto patch = GetShaderPatch(info.pgm_hash, info.hw_stage, perm_idx, "spv");
    const bool is_patched = patch && EmulatorSettings.IsPatchShaders();
    if (is_patched) {
        LOG_INFO(Loader, "Loaded patch for {} shader {:#x}", info.hw_stage, info.pgm_hash);
        module = CompileSPV(*patch, instance.GetDevice());
    } else {
        module = CompileSPV(spv, instance.GetDevice());
    }

    RegisterShaderBinary(std::move(spv), info.pgm_hash, perm_idx);

    const auto name = GetShaderName(info.hw_stage, info.pgm_hash, perm_idx);
    Vulkan::SetObjectName(instance.GetDevice(), module, name);
    if (EmulatorSettings.IsShaderCollect()) {
        DebugState.CollectShader(name, info.sw_stage, module, spv, code,
                                 patch ? *patch : std::span<const u32>{}, is_patched);
    }
    return module;
}

PipelineCache::Result PipelineCache::GetProgram(HwStage hw_stage, SwStage sw_stage,
                                                const Shader::ShaderParams& params,
                                                Shader::Backend::Bindings& binding) {
    auto runtime_info = BuildRuntimeInfo(hw_stage, sw_stage);
    // A new shader for a draw that may be skipped: translated in the background.
    if (IsShaderPending(params.hash)) {
        if (translate_in_background) {
            stage_deferred = true;
            return {};
        }
        const WaitTimer wait;
        while (IsShaderPending(params.hash)) {
            std::this_thread::yield();
            CollectBackgroundPipelines();
        }
    } else if (translate_in_background && !program_cache.contains(params.hash)) {
        StartBackgroundShader(hw_stage, sw_stage, params, runtime_info, binding, std::nullopt);
        stage_deferred = true;
        return {};
    }

    auto [it_pgm, new_program] = program_cache.try_emplace(params.hash);
    if (new_program) {
        it_pgm.value() = std::make_unique<Program>(hw_stage, sw_stage, params);
        auto& program = it_pgm.value();
        auto start = binding;
        const auto module = CompileModule(program->info, runtime_info, params.code, 0, binding);
        auto spec = Shader::StageSpecialization(program->info, runtime_info, profile, start);
        const auto perm_hash = HashCombine(params.hash, 0);

        RegisterShaderMeta(program->info, spec.fetch_shader_data, spec, perm_hash, 0);
        program->AddPermut(module, std::move(spec));
        if (!program->modules[0].spec.fetch_shader_data.Empty()) {
            fetch_program = program.get();
            fetch_perm = 0;
        }
        return std::make_tuple(&program->info, module, perm_hash);
    }

    // Not a reference into the map: finished background shaders can be added below.
    Program* const program = it_pgm.value().get();
    auto& info = program->info;
    info.pgm_base = params.Base(); // Needs to be actualized for inline cbuffer address fixup
    info.user_data = params.user_data;
    info.RefreshFlatBuf();
    auto spec = Shader::StageSpecialization(info, runtime_info, profile, binding);

    size_t perm_idx = program->modules.size();
    u64 perm_hash = HashCombine(params.hash, perm_idx);

    vk::ShaderModule module{};

    auto it = std::ranges::find(program->modules, spec, &Program::Module::spec);
    if (it == program->modules.end() && IsShaderPending(params.hash, &spec)) {
        if (translate_in_background) {
            stage_deferred = true;
            return {};
        }
        const WaitTimer wait;
        while (IsShaderPending(params.hash, &spec)) {
            std::this_thread::yield();
            CollectBackgroundPipelines();
        }
        it = std::ranges::find(program->modules, spec, &Program::Module::spec);
        perm_idx = program->modules.size();
        perm_hash = HashCombine(params.hash, perm_idx);
    } else if (it == program->modules.end() && translate_in_background) {
        StartBackgroundShader(hw_stage, sw_stage, params, runtime_info, binding, spec);
        stage_deferred = true;
        return {};
    }
    if (it == program->modules.end()) {
        // Diagnostics: a known shader in a new variant. Empty slots are variants the cache
        // held for other pipelines that could not be preloaded.
        const auto empty = std::ranges::count_if(
            program->modules, [](const Program::Module& m) { return !m.module; });
        LOG_INFO(Render_Vulkan, "New variant {} of {} shader {:#x} ({} known, {} not preloaded)",
                 perm_idx, hw_stage, params.hash, program->modules.size() - empty, empty);
        auto new_info = Shader::Info(hw_stage, sw_stage, params);
        module = CompileModule(new_info, runtime_info, params.code, perm_idx, binding);

        RegisterShaderMeta(info, spec.fetch_shader_data, spec, perm_hash, perm_idx);
        program->AddPermut(module, std::move(spec));
    } else {
        info.AddBindings(binding);
        module = it->module;
        perm_idx = std::distance(program->modules.begin(), it);
        perm_hash = HashCombine(params.hash, perm_idx);
    }
    if (!program->modules[perm_idx].spec.fetch_shader_data.Empty()) {
        fetch_program = program;
        fetch_perm = perm_idx;
    }
    return std::make_tuple(&program->info, module, perm_hash);
}

bool PipelineCache::IsShaderPending(u64 hash, const Shader::StageSpecialization* spec) const {
    const auto it = pending_shaders.find(hash);
    if (it == pending_shaders.end()) {
        return false;
    }
    if (!spec) {
        return it->second.new_program;
    }
    return std::ranges::any_of(it->second.variants,
                               [spec](const Shader::StageSpecialization& s) { return s == *spec; });
}

void PipelineCache::StartBackgroundShader(HwStage hw_stage, SwStage sw_stage,
                                          const Shader::ShaderParams& params,
                                          const Shader::RuntimeInfo& runtime_info,
                                          const Shader::Backend::Bindings& binding,
                                          std::optional<Shader::StageSpecialization> spec) {
    auto& pending = pending_shaders[params.hash];
    if (spec) {
        pending.variants.push_back(*spec);
    } else {
        pending.new_program = true;
    }
    LOG_INFO(Render_Vulkan, "Translating {} shader {:#x} in the background{}", hw_stage,
             params.hash, spec ? " (permutation)" : "");
    // The user data is copied: the registers change with the next draw.
    std::array<u32, Shader::ShaderParams::NumShaderUserData> user_data{};
    std::ranges::copy(params.user_data, user_data.begin());
    {
        std::scoped_lock lock{compile_mutex};
        compile_jobs.emplace_back([this, hw_stage, sw_stage, code = params.code, hash = params.hash,
                                   user_data, stage_runtime = runtime_info, start = binding,
                                   stage_spec = std::move(spec)]() mutable {
            const auto begin = std::chrono::steady_clock::now();
            const Shader::ShaderParams job_params{
                .user_data =
                    std::span<const u32, Shader::ShaderParams::NumShaderUserData>{user_data},
                .code = code,
                .hash = hash,
            };
            FinishedShader done{.hash = hash, .hw_stage = hw_stage};
            std::optional<Shader::Info> variant_info;
            Shader::Info* info;
            if (stage_spec) {
                info = &variant_info.emplace(hw_stage, sw_stage, job_params);
            } else {
                done.program = std::make_unique<Program>(hw_stage, sw_stage, job_params);
                info = &done.program->info;
            }
            // Each compile thread has its own IR memory.
            thread_local Shader::Pools job_pools;
            auto bindings = start;
            const auto ir_program =
                Shader::TranslateProgram(code, job_pools, *info, stage_runtime, profile);
            done.spv =
                Shader::Backend::SPIRV::EmitSPIRV(profile, stage_runtime, ir_program, bindings);
            done.module = CompileSPV(done.spv, instance.GetDevice());
            if (stage_spec) {
                done.spec = std::move(stage_spec);
            } else {
                done.spec.emplace(done.program->info, stage_runtime, profile, start);
            }
            done.ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
                    .count();
            std::scoped_lock finished{finished_mutex};
            finished_shaders.push_back(std::move(done));
            num_finished.fetch_add(1, std::memory_order_release);
        });
    }
    compile_cv.notify_one();
    Common::GuestStats::background_shaders_started.fetch_add(1, std::memory_order_relaxed);
}

void PipelineCache::CollectBackgroundShader(FinishedShader& done) {
    const auto device = instance.GetDevice();
    const auto pending = pending_shaders.find(done.hash);
    Program* program{};
    size_t perm_idx{};
    if (done.program) {
        const auto [it, is_new] = program_cache.try_emplace(done.hash);
        if (is_new) {
            it.value() = std::move(done.program);
            program = it.value().get();
            // The copy it was translated with is gone; every draw sets its own.
            program->info.user_data = {};
        }
        if (pending != pending_shaders.end()) {
            pending.value().new_program = false;
        }
    } else {
        const auto it = program_cache.find(done.hash);
        if (it != program_cache.end()) {
            program = it->second.get();
            perm_idx = program->modules.size();
        }
        if (pending != pending_shaders.end()) {
            auto& variants = pending.value().variants;
            const auto match = std::ranges::find_if(
                variants, [&](const Shader::StageSpecialization& s) { return s == *done.spec; });
            if (match != variants.end()) {
                variants.erase(match);
            }
        }
    }
    if (pending != pending_shaders.end() && !pending->second.new_program &&
        pending->second.variants.empty()) {
        pending_shaders.erase(pending);
    }
    if (!program) {
        device.destroyShaderModule(done.module);
        return;
    }

    auto patch = GetShaderPatch(done.hash, done.hw_stage, perm_idx, "spv");
    if (patch && EmulatorSettings.IsPatchShaders()) {
        LOG_INFO(Loader, "Loaded patch for {} shader {:#x}", done.hw_stage, done.hash);
        device.destroyShaderModule(done.module);
        done.module = CompileSPV(*patch, device);
    }
    const auto name = GetShaderName(done.hw_stage, done.hash, perm_idx);
    Vulkan::SetObjectName(device, done.module, name);
    const u64 perm_hash = HashCombine(done.hash, perm_idx);
    RegisterShaderBinary(std::move(done.spv), done.hash, perm_idx);
    RegisterShaderMeta(program->info, done.spec->fetch_shader_data, *done.spec, perm_hash,
                       perm_idx);
    program->AddPermut(done.module, std::move(*done.spec));
    LOG_INFO(Render_Vulkan, "{} shader {:#x} variant {} ready ({:.1f} ms in the background)",
             done.hw_stage, done.hash, perm_idx, done.ms);
}

std::optional<vk::ShaderModule> PipelineCache::ReplaceShader(vk::ShaderModule module,
                                                             std::span<const u32> spv_code) {
    std::optional<vk::ShaderModule> new_module{};
    for (const auto& [_, program] : program_cache) {
        for (auto& m : program->modules) {
            if (m.module == module) {
                const auto& d = instance.GetDevice();
                d.destroyShaderModule(m.module);
                m.module = CompileSPV(spv_code, d);
                new_module = m.module;
            }
        }
    }
    if (module_related_pipelines.contains(module)) {
        auto& pipeline_keys = module_related_pipelines[module];
        for (auto& key : pipeline_keys) {
            if (std::holds_alternative<GraphicsPipelineKey>(key)) {
                auto& graphics_key = std::get<GraphicsPipelineKey>(key);
                graphics_pipelines.erase(graphics_key);
            } else if (std::holds_alternative<ComputePipelineKey>(key)) {
                auto& compute_key = std::get<ComputePipelineKey>(key);
                compute_pipelines.erase(compute_key);
            }
        }
    }
    return new_module;
}

std::string PipelineCache::GetShaderName(Shader::HwStage stage, u64 hash,
                                         std::optional<size_t> perm) {
    if (perm) {
        return fmt::format("{}_{:#018x}_{}", stage, hash, *perm);
    }
    return fmt::format("{}_{:#018x}", stage, hash);
}

void PipelineCache::DumpShader(std::span<const u32> code, u64 hash, Shader::HwStage stage,
                               size_t perm_idx, std::string_view ext) {
    if (!EmulatorSettings.IsDumpShaders()) {
        return;
    }

    using namespace Common::FS;
    const auto dump_dir = GetUserPath(PathType::ShaderDir) / "dumps";
    if (!std::filesystem::exists(dump_dir)) {
        std::filesystem::create_directories(dump_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, perm_idx), ext);
    const auto file = IOFile{dump_dir / filename, FileAccessMode::Create};
    file.WriteSpan(code);
}

std::optional<std::vector<u32>> PipelineCache::GetShaderPatch(u64 hash, Shader::HwStage stage,
                                                              size_t perm_idx,
                                                              std::string_view ext) {

    using namespace Common::FS;
    const auto patch_dir = GetUserPath(PathType::ShaderDir) / "patch";
    if (!std::filesystem::exists(patch_dir)) {
        std::filesystem::create_directories(patch_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, perm_idx), ext);
    const auto filepath = patch_dir / filename;
    if (!std::filesystem::exists(filepath)) {
        return {};
    }
    const auto file = IOFile{patch_dir / filename, FileAccessMode::Read};
    std::vector<u32> code(file.GetSize() / sizeof(u32));
    file.Read(code);
    return code;
}
} // namespace Vulkan
