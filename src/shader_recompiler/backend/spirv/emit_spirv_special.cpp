// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"
#include "shader_recompiler/ir/debug_print.h"
#include "shader_recompiler/ir/microinstruction.h"

namespace Shader::Backend::SPIRV {

void EmitPrologue(EmitContext& ctx) {
    if (ctx.hw_stage == HwStage::Fragment) {
        ctx.DefineAmdPerVertexAttribs();
    }
    if (ctx.info.loads.Get(IR::Attribute::WorkgroupIndex)) {
        ctx.DefineWorkgroupIndex();
    }
    ctx.DefineBufferProperties();
}

void ConvertDepthMode(EmitContext& ctx) {
    const Id type{ctx.F32[1]};
    const Id position{ctx.OpLoad(ctx.F32[4], ctx.output_position)};
    const Id z{ctx.OpCompositeExtract(type, position, 2u)};
    const Id w{ctx.OpCompositeExtract(type, position, 3u)};
    const Id screen_depth{ctx.OpFMul(type, ctx.OpFAdd(type, z, w), ctx.Constant(type, 0.5f))};
    const Id vector{ctx.OpCompositeInsert(ctx.F32[4], screen_depth, position, 2u)};
    ctx.OpStore(ctx.output_position, vector);
}

void ConvertPositionToClipSpace(EmitContext& ctx) {
    ASSERT_MSG(!ctx.info.stores.GetAny(IR::Attribute::ViewportIndex),
               "Multi-viewport with shader clip space conversion not yet implemented.");

    const Id type{ctx.F32[1]};
    Id position{ctx.OpLoad(ctx.F32[4], ctx.output_position)};
    const Id x{ctx.OpCompositeExtract(type, position, 0u)};
    const Id y{ctx.OpCompositeExtract(type, position, 1u)};
    const Id z{ctx.OpCompositeExtract(type, position, 2u)};
    const Id w{ctx.OpCompositeExtract(type, position, 3u)};
    const Id xoffset_ptr{ctx.OpAccessChain(ctx.TypePointer(spv::StorageClass::PushConstant, type),
                                           ctx.push_data_block,
                                           ctx.ConstU32(PushData::XOffsetIndex))};
    const Id xoffset{ctx.OpLoad(type, xoffset_ptr)};
    const Id yoffset_ptr{ctx.OpAccessChain(ctx.TypePointer(spv::StorageClass::PushConstant, type),
                                           ctx.push_data_block,
                                           ctx.ConstU32(PushData::YOffsetIndex))};
    const Id yoffset{ctx.OpLoad(type, yoffset_ptr)};
    const Id xscale_ptr{ctx.OpAccessChain(ctx.TypePointer(spv::StorageClass::PushConstant, type),
                                          ctx.push_data_block,
                                          ctx.ConstU32(PushData::XScaleIndex))};
    const Id xscale{ctx.OpLoad(type, xscale_ptr)};
    const Id yscale_ptr{ctx.OpAccessChain(ctx.TypePointer(spv::StorageClass::PushConstant, type),
                                          ctx.push_data_block,
                                          ctx.ConstU32(PushData::YScaleIndex))};
    const Id yscale{ctx.OpLoad(type, yscale_ptr)};
    const Id vport_w =
        ctx.Constant(type, float(std::min<u32>(ctx.profile.max_viewport_width / 2, 8_KB)));
    const Id wnd_x = ctx.OpFAdd(type, ctx.OpFMul(type, x, xscale), xoffset);
    const Id ndc_x = ctx.OpFSub(type, ctx.OpFDiv(type, wnd_x, vport_w), ctx.Constant(type, 1.f));
    const Id vport_h =
        ctx.Constant(type, float(std::min<u32>(ctx.profile.max_viewport_height / 2, 8_KB)));
    const Id wnd_y = ctx.OpFAdd(type, ctx.OpFMul(type, y, yscale), yoffset);
    const Id ndc_y = ctx.OpFSub(type, ctx.OpFDiv(type, wnd_y, vport_h), ctx.Constant(type, 1.f));
    const Id vector{ctx.OpCompositeConstruct(ctx.F32[4], std::array<Id, 4>({ndc_x, ndc_y, z, w}))};
    ctx.OpStore(ctx.output_position, vector);
}

// Character motion vectors (runtime_info.h, MotionVectors; ported from bbport). The vertex shader
// stores its clip position for this frame and loads the previous frame's (same draw, same
// vertex). The draw's parameters are found by the device address in push data:
//   u32x4: store address (lo, hi), load address (lo, hi); 0 = none
//   u32x4: vertices per instance, first vertex, first instance, instances
// Disabled accesses are branched around.
static void EmitVertexMotion(EmitContext& ctx) {
    const Id u32_type = ctx.U32[1];
    const Id bool_type = ctx.U1[1];
    const Id position = ctx.OpLoad(ctx.F32[4], ctx.output_position);
    const Id param_ptr =
        ctx.OpAccessChain(ctx.TypePointer(spv::StorageClass::PushConstant, ctx.U32[2]),
                          ctx.push_data_block, ctx.ConstU32(PushData::MotionParamIndex));
    const Id param_words = ctx.OpLoad(ctx.U32[2], param_ptr);
    const auto to_u64 = [&](Id lo, Id hi) {
        return ctx.OpBitwiseOr(
            ctx.U64, ctx.OpUConvert(ctx.U64, lo),
            ctx.OpShiftLeftLogical(ctx.U64, ctx.OpUConvert(ctx.U64, hi), ctx.ConstU32(32U)));
    };
    const Id param_address = to_u64(ctx.OpCompositeExtract(u32_type, param_words, 0u),
                                    ctx.OpCompositeExtract(u32_type, param_words, 1u));
    const Id u32x4_ptr = ctx.TypePointer(spv::StorageClass::PhysicalStorageBuffer, ctx.U32[4]);
    const Id f32x4_ptr = ctx.TypePointer(spv::StorageClass::PhysicalStorageBuffer, ctx.F32[4]);
    const Id addresses = ctx.OpLoad(ctx.U32[4], ctx.OpConvertUToPtr(u32x4_ptr, param_address),
                                    spv::MemoryAccessMask::Aligned, 16u);
    const Id ranges =
        ctx.OpLoad(ctx.U32[4],
                   ctx.OpConvertUToPtr(u32x4_ptr, ctx.OpIAdd(ctx.U64, param_address,
                                                             ctx.Constant(ctx.U64, u64(16)))),
                   spv::MemoryAccessMask::Aligned, 16u);
    const Id store_address = to_u64(ctx.OpCompositeExtract(u32_type, addresses, 0u),
                                    ctx.OpCompositeExtract(u32_type, addresses, 1u));
    const Id load_address = to_u64(ctx.OpCompositeExtract(u32_type, addresses, 2u),
                                   ctx.OpCompositeExtract(u32_type, addresses, 3u));
    const Id vertices = ctx.OpCompositeExtract(u32_type, ranges, 0u);
    const Id first_vertex = ctx.OpCompositeExtract(u32_type, ranges, 1u);
    const Id first_instance = ctx.OpCompositeExtract(u32_type, ranges, 2u);
    const Id instances = ctx.OpCompositeExtract(u32_type, ranges, 3u);
    const Id vertex = ctx.OpISub(u32_type, ctx.OpLoad(u32_type, ctx.vertex_index), first_vertex);
    const Id instance = ctx.OpISub(u32_type, ctx.OpLoad(u32_type, ctx.instance_id), first_instance);
    const Id in_range = ctx.OpLogicalAnd(bool_type, ctx.OpULessThan(bool_type, vertex, vertices),
                                         ctx.OpULessThan(bool_type, instance, instances));
    const Id slot = ctx.OpIAdd(u32_type, vertex, ctx.OpIMul(u32_type, instance, vertices));
    const Id slot_offset =
        ctx.OpIMul(ctx.U64, ctx.OpUConvert(ctx.U64, slot), ctx.Constant(ctx.U64, u64(16)));
    const Id zero64 = ctx.Constant(ctx.U64, u64(0));
    const Id do_store =
        ctx.OpLogicalAnd(bool_type, in_range, ctx.OpINotEqual(bool_type, store_address, zero64));
    const Id do_load =
        ctx.OpLogicalAnd(bool_type, in_range, ctx.OpINotEqual(bool_type, load_address, zero64));

    const Id store_label = ctx.OpLabel();
    const Id store_merge = ctx.OpLabel();
    ctx.OpSelectionMerge(store_merge, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(do_store, store_label, store_merge);
    ctx.AddLabel(store_label);
    {
        // One 16-byte store. Indexed draws may run the same vertex more than once, but every
        // such invocation writes the same clip position, so the overlap is harmless (and much
        // cheaper than bbport's four atomics per vertex).
        const Id target = ctx.OpIAdd(ctx.U64, store_address, slot_offset);
        ctx.OpStore(ctx.OpConvertUToPtr(f32x4_ptr, target), position,
                    spv::MemoryAccessMask::Aligned, 16u);
    }
    ctx.OpBranch(store_merge);
    ctx.AddLabel(store_merge);

    const Id load_label = ctx.OpLabel();
    const Id load_merge = ctx.OpLabel();
    ctx.OpSelectionMerge(load_merge, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(do_load, load_label, load_merge);
    ctx.AddLabel(load_label);
    const Id loaded = ctx.OpLoad(
        ctx.F32[4], ctx.OpConvertUToPtr(f32x4_ptr, ctx.OpIAdd(ctx.U64, load_address, slot_offset)),
        spv::MemoryAccessMask::Aligned, 16u);
    ctx.OpBranch(load_merge);
    ctx.AddLabel(load_merge);
    Id previous = ctx.OpPhi(ctx.F32[4], position, store_merge, loaded, load_label);
    const Id valid = ctx.OpSelect(ctx.F32[1], do_load, ctx.Constant(ctx.F32[1], 1.0f),
                                  ctx.Constant(ctx.F32[1], 0.0f));
    previous = ctx.OpCompositeInsert(ctx.F32[4], valid, previous, 2u);
    ctx.OpStore(ctx.motion_out_cur, position);
    ctx.OpStore(ctx.motion_out_prev, previous);
}

// Screen-space motion (previous minus current, render pixels, y down) through the viewport
// scale; b = valid, a = this fragment's depth (the upscaler rejects vectors of covered pixels).
static void EmitFragmentMotion(EmitContext& ctx) {
    const Id f32_type = ctx.F32[1];
    const Id bool_type = ctx.U1[1];
    const Id current = ctx.OpLoad(ctx.F32[4], ctx.motion_in_cur);
    const Id previous = ctx.OpLoad(ctx.F32[4], ctx.motion_in_prev);
    const auto push = [&](u32 index) {
        return ctx.OpLoad(
            f32_type, ctx.OpAccessChain(ctx.TypePointer(spv::StorageClass::PushConstant, f32_type),
                                        ctx.push_data_block, ctx.ConstU32(index)));
    };
    const Id xscale = push(PushData::XScaleIndex);
    const Id yscale = push(PushData::YScaleIndex);
    const auto ndc = [&](Id clip, u32 component) {
        return ctx.OpFDiv(f32_type, ctx.OpCompositeExtract(f32_type, clip, component),
                          ctx.OpCompositeExtract(f32_type, clip, 3u));
    };
    const Id dx =
        ctx.OpFMul(f32_type, ctx.OpFSub(f32_type, ndc(previous, 0u), ndc(current, 0u)), xscale);
    const Id dy =
        ctx.OpFMul(f32_type, ctx.OpFSub(f32_type, ndc(previous, 1u), ndc(current, 1u)), yscale);
    const Id zero = ctx.Constant(f32_type, 0.0f);
    const Id epsilon = ctx.Constant(f32_type, 1e-5f);
    const Id front = ctx.OpLogicalAnd(
        bool_type,
        ctx.OpFOrdGreaterThan(bool_type, ctx.OpCompositeExtract(f32_type, previous, 3u), epsilon),
        ctx.OpFOrdGreaterThan(bool_type, ctx.OpCompositeExtract(f32_type, current, 3u), epsilon));
    const Id valid =
        ctx.OpSelect(f32_type, front, ctx.OpCompositeExtract(f32_type, previous, 2u), zero);
    const Id depth = ctx.OpCompositeExtract(f32_type, ctx.OpLoad(ctx.F32[4], ctx.frag_coord), 2u);
    ctx.OpStore(ctx.motion_frag_out,
                ctx.OpCompositeConstruct(ctx.F32[4],
                                         std::array<Id, 4>{ctx.OpSelect(f32_type, front, dx, zero),
                                                           ctx.OpSelect(f32_type, front, dy, zero),
                                                           valid, depth}));
}

void EmitEpilogue(EmitContext& ctx) {
    if (Sirit::ValidId(ctx.motion_out_cur)) {
        EmitVertexMotion(ctx);
    }
    if (Sirit::ValidId(ctx.motion_frag_out)) {
        EmitFragmentMotion(ctx);
    }
    if (ctx.hw_stage == HwStage::Vertex &&
        ctx.runtime_info.hw.vs.emulate_depth_negative_one_to_one) {
        ConvertDepthMode(ctx);
    }
    if (ctx.hw_stage == HwStage::Vertex && ctx.runtime_info.hw.vs.clip_disable) {
        ConvertPositionToClipSpace(ctx);
    }
}

void EmitDiscard(EmitContext& ctx) {
    ctx.OpDemoteToHelperInvocationEXT();
}

void EmitDiscardCond(EmitContext& ctx, Id condition) {
    const Id kill_label{ctx.OpLabel()};
    const Id merge_label{ctx.OpLabel()};
    ctx.OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(condition, kill_label, merge_label);
    ctx.AddLabel(kill_label);
    ctx.OpDemoteToHelperInvocationEXT();
    ctx.OpBranch(merge_label);
    ctx.AddLabel(merge_label);
}

void EmitEmitVertex(EmitContext& ctx) {
    ctx.OpEmitVertex();
}

void EmitEmitPrimitive(EmitContext& ctx) {
    ctx.OpEndPrimitive();
}

void EmitEmitVertex(EmitContext& ctx, const IR::Value& stream) {
    UNREACHABLE_MSG("Geometry streams");
}

void EmitEndPrimitive(EmitContext& ctx, const IR::Value& stream) {
    UNREACHABLE_MSG("Geometry streams");
}

void EmitDebugPrint(EmitContext& ctx, IR::Inst* inst, Id fmt, Id arg0, Id arg1, Id arg2, Id arg3) {
    IR::DebugPrintFlags flags = inst->Flags<IR::DebugPrintFlags>();
    std::array<Id, IR::DEBUGPRINT_NUM_FORMAT_ARGS> fmt_args = {arg0, arg1, arg2, arg3};
    auto fmt_args_span = std::span<Id>(fmt_args.begin(), fmt_args.begin() + flags.num_args);
    ctx.OpDebugPrintf(fmt, fmt_args_span);
}

Id EmitMemtime(EmitContext& ctx) {
    if (ctx.profile.supports_shader_subgroup_clock) {
        return ctx.OpReadClockKHR(ctx.U64, ctx.ConstU32(std::to_underlying(spv::Scope::Subgroup)));
    } else {
        return ctx.Constant(ctx.U64, 1U);
    }
}

} // namespace Shader::Backend::SPIRV
