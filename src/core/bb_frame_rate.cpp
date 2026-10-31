// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-FileCopyrightText: Copyright 2026 bbport contributors (deadinside28/bloodborne_pc)
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Ported from bbport 0.7 (gpu/shim/bbport_frame_rate.cpp and gpu/shim/bbport_guest_code.h,
// https://github.com/deadinside28/bloodborne_pc). Its notes, shortened:
//
// The game's main loop (0x2018d20) runs its frame pacer (0x2034770: it waits for the frame time
// its mode asks, 1/30 s, and measures the frame at [pacer + 0x264]) and then steps the game by a
// fixed 1/30 s (0x2018e36). The hooks keep the pacer's mode at one vblank and no frame time to
// wait for beyond the shortest frame, and step the game by the measured frame time (1/480 to
// 1/30 s: below 30 FPS the game slows down, as it always did).
//
// What the game ties to a 30 FPS frame besides: subsystems stepped by a fixed 1/30 s, per-frame
// steps and approaches, per-frame damping. Their instructions read the constant from a page next
// to the image instead (the instruction's own address operand moved; the game's code is otherwise
// as it was), where FrameStep writes the value for this frame's time: a step of 1/30 s becomes
// the frame's time, a per-frame factor f becomes f^frames for the frame's length in 30 FPS frames.
// Durations the game converts from 30 FPS frames into seconds stay as they are: what counts them
// down uses the real time.

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <map>
#include <new>
#include <set>
#include <string>

#include "common/bb_tuning.h"
#include "common/logging/log.h"
#include "common/memory_patcher.h"
#include "common/singleton.h"
#include "common/types.h"
#include "core/bb_frame_rate.h"
#include "core/cpu_patches.h"
#include "core/emulator_settings.h"
#include "core/file_format/psf.h"

namespace Core::BbFrameRate {
namespace {

/// A stub's bytes, built in place (bbport's BbGuestCode::Emitter).
struct Emitter {
    std::array<u8, 512> code{};
    std::size_t size = 0;
    bool overflow = false;
    void Bytes(std::initializer_list<u8> bytes) {
        for (const u8 b : bytes) {
            if (size < code.size()) {
                code[size] = b;
            }
            ++size;
        }
        overflow = overflow || size > code.size();
    }
    void U32(u32 v) {
        u8 b[4];
        std::memcpy(b, &v, 4);
        Bytes({b[0], b[1], b[2], b[3]});
    }
    void U64(u64 v) {
        U32(u32(v));
        U32(u32(v >> 32));
    }
    /// Saves the caller-saved general registers (the handler may change them), aligns the stack.
    void SaveVolatile() {
        Bytes({0x50, 0x51, 0x52, 0x56, 0x57, 0x41, 0x50, 0x41, 0x51, 0x41, 0x52, 0x41, 0x53});
        Bytes({0x41, 0x54});             // push r12 (holds the unaligned stack pointer)
        Bytes({0x49, 0x89, 0xe4});       // mov r12, rsp
        Bytes({0x48, 0x83, 0xe4, 0xf0}); // and rsp, -16
    }
    void RestoreVolatile() {
        Bytes({0x4c, 0x89, 0xe4}); // mov rsp, r12
        Bytes({0x41, 0x5c});       // pop r12
        Bytes({0x41, 0x5b, 0x41, 0x5a, 0x41, 0x59, 0x41, 0x58, 0x5f, 0x5e, 0x5a, 0x59, 0x58});
    }
    void CallAbs(const void* fn) {
        Bytes({0x48, 0xb8}); // movabs rax, fn
        U64(reinterpret_cast<u64>(fn));
        Bytes({0xff, 0xd0}); // call rax
    }
    /// jmp [rip]: to `target` anywhere.
    void JmpAbs(u64 target) {
        Bytes({0xff, 0x25, 0x00, 0x00, 0x00, 0x00});
        U64(target);
    }
    /// The 32-bit value at `address` into the stack frame's [rbp + offset] (rax kept).
    void StoreToFrame(const void* address, s8 offset) {
        Bytes({0x50, 0x48, 0xb8}); // push rax; movabs rax, address
        U64(reinterpret_cast<u64>(address));
        Bytes({0x8b, 0x00});                   // mov eax, [rax]
        Bytes({0x89, 0x45, u8(offset), 0x58}); // mov [rbp + offset], eax; pop rax
    }
};

std::atomic<bool> active{false};
u64 image_base = 0;

constexpr float Frame30 = 1.0f / 30.0f;
/// The highest frame rate the pacer allows. bbport: the game itself breaks above it (on loading
/// screens at 880-1000 FPS its main loop outran its worker threads; at 480 all loads went through).
constexpr int MaxRate = 480;

/// What the redirected instructions read, rewritten by FrameStep for each frame.
struct alignas(64) Live {
    float dt = Frame30;              ///< the frame's time: the game steps by it
    float shortest = 1.0f / MaxRate; ///< the pacer's frame time target
    float pad[2] = {};
    float decay95[4] = {0.95f, 0.95f, 0.95f, 0.95f}; ///< 0.95 per 30 FPS frame (read as four)
    float approach30 = Frame30;                      ///< a 1/30 approach per 30 FPS frame
    float approach10 = 0.1f;                         ///< a 0.1 approach per 30 FPS frame
    float step_down = -0.1f;                         ///< -0.1 per 30 FPS frame
    float step_up = 0.1f;                            ///< +0.1 per 30 FPS frame
    float decay08 = 0.8f;                            ///< 0.8 per 30 FPS frame
    float cloth = 1.0f;                              ///< cloth constraints' stiffness
    float model_turn = 0.0f; ///< the model preview's turn per 30 FPS frame (per stick unit)
};
Live* live = nullptr;
float model_turn_base = 0.0f; ///< the game's own model_turn (its constant, read at start)

/// The pacer at the main loop's step (rdi: it): this frame's values. On the game's thread.
void PS4_SYSV_ABI FrameStep(u64 pacer) {
    const float measured = *reinterpret_cast<const float*>(pacer + 0x264);
    const float dt =
        std::clamp(std::isfinite(measured) ? measured : Frame30, live->shortest, Frame30);
    const float frames = dt * 30.0f; // the frame's length in 30 FPS frames
    live->dt = dt;
    const float decay95 = std::pow(0.95f, frames);
    for (float& d : live->decay95) {
        d = decay95;
    }
    live->approach30 = 1.0f - std::pow(1.0f - Frame30, frames);
    live->approach10 = 1.0f - std::pow(0.9f, frames);
    live->step_down = -0.1f * frames;
    live->step_up = 0.1f * frames;
    live->decay08 = std::pow(0.8f, frames);
    live->model_turn = model_turn_base * frames;
    // The cloth setups' range checks and a few uses of 1.0 next to them: the game's 1.0. The cloth
    // follows the frame rate by its constraints' stiffness factor instead (ClothStiffnessFactor).
    live->cloth = 1.0f;
}

/// The game's exponent of Havok's stiffness adaptation (0x48d7bbc: -1.725), read at start.
float cloth_exponent = -1.725f;

/// Havok's stiffness factor of a constraint set for one substep (0x1167a80, replaced): `mode` 2
/// adapts the constraints' stiffness to how often they are solved, (substeps x a x b)^-1.725, for a
/// set of type 8 the iteration's share instead and for types 5 and 10 only the last substep. The
/// substeps are per frame: at 191 FPS the cloth is solved 6.4 times as often as at 30 FPS by the
/// same factor, and stiffer by as much (it flutters). The factor here is the game's x frames (the
/// frame's length in 30 FPS frames): the cloth is pulled as far per second as at 30 FPS. At 30 FPS
/// it is the game's.
float PS4_SYSV_ABI ClothStiffnessFactor(const u8* set, int mode, int iteration, int substeps,
                                        float a, float b) {
    const u32 type = *reinterpret_cast<const u32*>(set + 0x18);
    const bool adapts = mode == 2 && !(substeps == 1 && a == 1.0f && b == 1.0f);
    if (adapts && type == 8) {
        return float(iteration + 1) / float(substeps);
    }
    if (adapts && (type == 5 || type == 10)) {
        return substeps - 1 == iteration ? 1.0f : 0.0f;
    }
    const float game = adapts ? std::pow(float(substeps) * a * b, cloth_exponent) : 1.0f;
    if (type == 5 || type == 8 || type == 10) {
        return game;
    }
    // A constraint solved with stiffness k pulls its particles k of the way per solve: at 30 FPS
    // k a frame, at any frame rate ~k x frames. x0.5 at 60 FPS, the community's 60 FPS value.
    return game * (live->dt * 30.0f);
}

// ---- Redirected constants: instruction (its RIP-relative operand), the constant, the value ----
struct Redirect {
    u64 site;
    u64 constant;
    std::size_t value; ///< offsetof(Live, ..)
};
#define LIVE(field) offsetof(Live, field)
constexpr Redirect Redirects[] = {
    // A fixed 1/30 s step or "one frame":
    {0xab9f0d, 0x47efdcc, LIVE(dt)}, // a time step, one frame when unset
    {0xdca3ed, 0x4888688, LIVE(dt)}, // a per-frame budget
    {0xee596a, 0x48ad65c, LIVE(dt)}, // effects: a fade of at least one frame (three releases)
    {0xee5a8a, 0x48ad660, LIVE(dt)},
    {0xee5baa, 0x48ad664, LIVE(dt)},
    {0xef06f1, 0x48adc08, LIVE(dt)}, // at least one frame
    {0xef6992, 0x48add28, LIVE(dt)},
    {0x1044943, 0x48cc3f4, LIVE(dt)}, // an update by one frame
    {0x12a4416, 0x492577c, LIVE(dt)}, // a timer advanced by one frame each frame
    {0x15dc1f7, 0x4926c9c, LIVE(dt)},
    {0x173456e, 0x49270e4, LIVE(dt)}, // an update by one frame
    {0x17f9d26, 0x4927554, LIVE(dt)},
    {0x180f647, 0x49275f4, LIVE(dt)}, // a timer advanced by one frame each frame
    {0x18bc4f7, 0x4927c4c, LIVE(dt)}, // an update by one frame (twice)
    {0x18bc50d, 0x4927c4c, LIVE(dt)},
    {0x1dbc308, 0x4928e98, LIVE(dt)}, // frames counted at one frame's time
    {0x1f77d4a, 0x4929170, LIVE(dt)}, // an update by one frame (0 after a reset)
    {0x227784b, 0x4929b80, LIVE(dt)}, // timers advanced by one frame (0xebbda0)
    {0x2277892, 0x4929b80, LIVE(dt)},
    {0x2313dad, 0x4929e70, LIVE(dt)}, // a simulation step: 1/60 in the game's 60 Hz mode,
    {0x2313db7, 0x4929e74, LIVE(dt)}, // 1/30 otherwise
    {0x2315d71, 0x4929e74, LIVE(dt)},
    // Body parts the physics moves (0x1515ec0, each part every frame): their timers advanced by
    // the step they are given, 1/30 s.
    {0x1515fab, 0x4926364, LIVE(dt)},
    {0x1516030, 0x4926364, LIVE(dt)},
    // Steps and approaches per frame:
    {0x19084f5, 0x4927fd4, LIVE(step_down)}, // a value ramped by 0.1 a frame
    {0x1908522, 0x4927fd4, LIVE(step_down)},
    {0x1908518, 0x4927fe0, LIVE(step_up)},
    {0x190854c, 0x4927fe0, LIVE(step_up)},
    {0x143e2b2, 0x4925e94, LIVE(approach30)}, // the follow camera's approaches
    {0x143af8c, 0x4925e68, LIVE(approach10)},
    // Damping per frame:
    {0x1a4abca, 0x490b5b0, LIVE(decay95)}, // a character's motion (0x1a4a730, x0.95 a frame)
    {0x1514c4d, 0x4926334, LIVE(decay08)}, // the run's wall check (x0.8 a frame while blocked)
    // Cloth constraints' stiffness:
    {0x47c6dc, 0x47693f0, LIVE(cloth)},
    {0x7e45be, 0x47b63e4, LIVE(cloth)},
    {0x7e4642, 0x47b63e4, LIVE(cloth)},
    {0x7e6239, 0x47b63e4, LIVE(cloth)},
    {0x7f706d, 0x47b8474, LIVE(cloth)},
    {0x7f70ad, 0x47b8474, LIVE(cloth)},
    {0x7f79e5, 0x47b8474, LIVE(cloth)},
    {0x7f7be8, 0x47b8474, LIVE(cloth)},
    {0x7f7dbd, 0x47b8474, LIVE(cloth)},
    {0x805898, 0x47b91c0, LIVE(cloth)},
    {0x805d18, 0x47b91c0, LIVE(cloth)},
    {0x806269, 0x47b91c0, LIVE(cloth)},
    {0x8062f0, 0x47b91c0, LIVE(cloth)},
    {0x809d62, 0x47b98b8, LIVE(cloth)},
    {0x80a4c9, 0x47b98b8, LIVE(cloth)},
    {0x80a876, 0x47b98b8, LIVE(cloth)},
    // The model preview (0x1db2780: the character creator, the stick turns the model): yaw and
    // pitch += stick x the constant, per frame.
    {0x1db27b0, 0x4928dfc, LIVE(model_turn)},
};
#undef LIVE

/// The farthest the hooks read or write in the image (the last constant, with room).
constexpr u64 ImageReach = 0x4929f00;

u8* At(u64 offset) {
    return reinterpret_cast<u8*>(image_base + offset);
}

/// The instruction at `r.site` reads `r.constant` RIP-relative (a VEX instruction, no immediate
/// after the operand): it reads the live value instead.
bool RedirectOperand(const Redirect& r) {
    u8* code = At(r.site);
    std::size_t disp_at = 0, length = 0;
    if (code[0] == 0xc5 && (code[3] & 0xc7) == 0x05) {
        disp_at = 4, length = 8;
    } else if (code[0] == 0xc4 && (code[4] & 0xc7) == 0x05) {
        disp_at = 5, length = 9;
    } else {
        return false;
    }
    s32 disp = 0;
    std::memcpy(&disp, code + disp_at, 4);
    const u64 next = image_base + r.site + length;
    if (next + s64(disp) != image_base + r.constant) {
        return false;
    }
    const s64 to = s64(reinterpret_cast<u64>(live) + r.value) - s64(next);
    if (to != s32(to)) {
        return false;
    }
    const s32 to32 = s32(to);
    std::memcpy(code + disp_at, &to32, 4);
    return true;
}

// ---- Detours: the instructions at a site replaced by a jump to a stub ----
u8* stub_page = nullptr;
std::size_t stub_used = 0;
constexpr std::size_t StubPageSize = 4096;

/// `e` into the stub page; a jump to it at `site` over `length` bytes (the rest int3) if the site
/// holds `expected`.
bool Detour(u64 site, const u8* expected, std::size_t length, const Emitter& e) {
    u8* code = At(site);
    if (length < 5 || std::memcmp(code, expected, length) != 0 || e.overflow ||
        stub_used + e.size > StubPageSize) {
        return false;
    }
    u8* stub = stub_page + stub_used;
    const s64 rel = s64(reinterpret_cast<u64>(stub)) - s64(image_base + site + 5);
    if (rel != s32(rel)) {
        return false;
    }
    std::memcpy(stub, e.code.data(), e.size);
    stub_used += (e.size + 15) & ~std::size_t(15);
    const s32 rel32 = s32(rel);
    code[0] = 0xe9;
    std::memcpy(code + 1, &rel32, 4);
    for (std::size_t i = 5; i < length; ++i) {
        code[i] = 0xcc;
    }
    return true;
}

/// The pacer's mode, after the game chose it (0x203493d, `lea r15, [r12 + 0x268]`; r12 the
/// pacer): one vblank (+0x10), no flip skipping (+0x14), the shortest frame as the frame time
/// target (+0x18), no vblank counts to wait for (+0x268, +0x26c).
bool HookPacerMode() {
    constexpr u8 code[] = {0x4d, 0x8d, 0xbc, 0x24, 0x68, 0x02, 0x00, 0x00};
    Emitter e;
    e.Bytes({0x41, 0xc7, 0x44, 0x24, 0x10, 0x01, 0x00, 0x00, 0x00}); // mov dword [r12 + 0x10], 1
    e.Bytes({0x41, 0xc6, 0x44, 0x24, 0x14, 0x00});                   // mov byte [r12 + 0x14], 0
    e.Bytes({0x50, 0x48, 0xb8});                                     // push rax; movabs rax,
    e.U64(reinterpret_cast<u64>(&live->shortest));                   //   &shortest
    e.Bytes({0x8b, 0x00});                                           // mov eax, [rax]
    e.Bytes({0x41, 0x89, 0x44, 0x24, 0x18, 0x58}); // mov [r12 + 0x18], eax; pop rax
    e.Bytes({0x49, 0xc7, 0x84, 0x24, 0x68, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
             0x00}); // mov qword [r12 + 0x268], 0
    for (const u8 b : code) {
        e.Bytes({b});
    }
    e.JmpAbs(image_base + 0x203493d + sizeof(code));
    return Detour(0x203493d, code, sizeof(code), e);
}

/// The pacer keeps its one vblank when a frame came late (0x2034d96: `test dl, dl; je; mov eax,
/// [r12 + 0x10]`, then eax is stored back: 0 when late).
bool HookPacerLate() {
    constexpr u8 code[] = {0x84, 0xd2, 0x74, 0x05, 0x41, 0x8b, 0x44, 0x24, 0x10};
    Emitter e;
    e.Bytes({0x84, 0xd2});                   // test dl, dl
    e.Bytes({0x41, 0x8b, 0x44, 0x24, 0x10}); // mov eax, [r12 + 0x10]
    e.JmpAbs(image_base + 0x2034d9f);
    return Detour(0x2034d96, code, sizeof(code), e);
}

/// The main loop's step (0x2018e36: `mov dword [rbp - 0x38], 1/30`, its time; rbx the pacer):
/// FrameStep, then the frame's time.
bool HookStep() {
    constexpr u8 code[] = {0xc7, 0x45, 0xc8, 0x89, 0x88, 0x08, 0x3d};
    Emitter e;
    e.SaveVolatile();
    e.Bytes({0x48, 0x89, 0xdf}); // mov rdi, rbx
    e.CallAbs(reinterpret_cast<const void*>(&FrameStep));
    e.RestoreVolatile();
    e.StoreToFrame(&live->dt, -0x38);
    e.JmpAbs(image_base + 0x2018e36 + sizeof(code));
    return Detour(0x2018e36, code, sizeof(code), e);
}

/// A subsystem stepped by a fixed 1/30 s (`mov dword [rbp + offset], 1/30` at `site`): by the
/// frame's time.
bool HookFixedStep(u64 site, s8 offset) {
    const u8 code[] = {0xc7, 0x45, u8(offset), 0x89, 0x88, 0x08, 0x3d};
    Emitter e;
    e.StoreToFrame(&live->dt, offset);
    e.JmpAbs(image_base + site + sizeof(code));
    return Detour(site, code, sizeof(code), e);
}

/// The run's wall check (0x1514c24: the distance moved this frame x 30 = a speed at 30 FPS): the
/// distance divided by the frame's time ([rbp - 0x74], the character update's), a speed at any
/// frame rate. Above ~90 FPS the per-frame distance fell under the check's threshold and the run
/// dropped to half speed. 1: done by the hook, 2: a Sprint Fix patch already did it, 0: other code.
int HookRunSpeed() {
    constexpr u64 site = 0x1514c24;
    const u8* code = At(site);
    constexpr u8 patched[] = {0xc5, 0xfa, 0x5e, 0x45, 0x8c, 0x0f, 0x1f, 0x00}; // vdivss; nop
    if (std::memcmp(code, patched, sizeof(patched)) == 0) {
        return 2;
    }
    s32 disp = 0;
    std::memcpy(&disp, code + 4, 4);
    if (code[0] != 0xc5 || code[1] != 0xfa || code[2] != 0x59 || code[3] != 0x05 ||
        image_base + site + 8 + s64(disp) != image_base + 0x492632c) {
        return 0;
    }
    u8 expected[8];
    std::memcpy(expected, code, 8);
    Emitter e;
    e.Bytes({0xc5, 0xfa, 0x5e, 0x45, 0x8c}); // vdivss xmm0, xmm0, [rbp - 0x74]
    e.JmpAbs(image_base + site + 8);
    return Detour(site, expected, 8, e) ? 1 : 0;
}

/// Havok's stiffness factor (0x1167a80: `push rbp; mov rbp, rsp; push r15`) by
/// ClothStiffnessFactor (its arguments as they are, a tail call).
bool HookClothStiffness() {
    constexpr u64 site = 0x1167a80;
    constexpr u8 code[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57};
    std::memcpy(&cloth_exponent, At(0x48d7bbc), sizeof(float));
    if (!(cloth_exponent < 0.0f && cloth_exponent > -4.0f)) {
        return false;
    }
    Emitter e;
    e.JmpAbs(reinterpret_cast<u64>(&ClothStiffnessFactor));
    return Detour(site, code, sizeof(code), e);
}

/// The three core sites hold the expected code (checked before any is changed, so that the game
/// is never left with only some of them).
bool CoreCodeMatches() {
    constexpr u8 mode[] = {0x4d, 0x8d, 0xbc, 0x24, 0x68, 0x02, 0x00, 0x00};
    constexpr u8 late[] = {0x84, 0xd2, 0x74, 0x05, 0x41, 0x8b, 0x44, 0x24, 0x10};
    constexpr u8 step[] = {0xc7, 0x45, 0xc8, 0x89, 0x88, 0x08, 0x3d};
    return std::memcmp(At(0x203493d), mode, sizeof(mode)) == 0 &&
           std::memcmp(At(0x2034d96), late, sizeof(late)) == 0 &&
           std::memcmp(At(0x2018e36), step, sizeof(step)) == 0;
}

// ---- Frame rate patches ----
/// The code the core hooks replace (image offsets, sizes).
constexpr std::pair<u64, u64> CoreSites[] = {{0x203493d, 8}, {0x2034d96, 9}, {0x2018e36, 7}};

bool deciding = false;
struct PatchWrites {
    bool core = false;         ///< writes the code the core hooks replace
    std::set<u64> constants{}; ///< the redirected constants it writes
};
std::map<std::string, PatchWrites, std::less<>> patch_writes;

/// A frame rate patch: it changes the frame pacer the hooks replace, or more than one of the frame
/// time constants they redirect (the 30 to 960 FPS++ and Uncap FPS++ patches, and Faster loading
/// screens, which needs one of them). Others that write one constant (Smaller Hunter) stay, and
/// that constant keeps the patch's value.
bool IsFrameRatePatch(const PatchWrites& writes) {
    return writes.core || writes.constants.size() >= 2;
}

/// A constant a kept patch writes: its instructions keep reading the patch's value.
bool KeptPatchWrites(u64 constant) {
    for (const auto& [name, writes] : patch_writes) {
        if (!IsFrameRatePatch(writes) && writes.constants.contains(constant)) {
            return true;
        }
    }
    return false;
}

/// The live limit as the pacer's shortest frame time.
float ShortestFrame(int fps) {
    return 1.0f / float(fps > 0 ? std::clamp(fps, 30, MaxRate) : MaxRate);
}

} // namespace

void Install() {
    if (!EmulatorSettings.IsBbFrameHooks() || !MemoryPatcher::g_eboot_address) {
        return;
    }
    image_base = MemoryPatcher::g_eboot_address;
    if (MemoryPatcher::g_eboot_image_size < ImageReach) {
        LOG_WARNING(Loader, "Frame rate hooks: not installed (not Bloodborne 1.09's image)");
        return;
    }
    if (!CoreCodeMatches()) {
        LOG_WARNING(Loader, "Frame rate hooks: not installed (the game's frame pacer is not the "
                            "expected code)");
        return;
    }
    // Both within reach of the game's code: the module's trampoline area, right after its image.
    void* data = AllocatePatchTrampolineSpace(At(0), sizeof(Live));
    stub_page = static_cast<u8*>(AllocatePatchTrampolineSpace(At(0), StubPageSize));
    if (!data || !stub_page) {
        LOG_WARNING(Loader, "Frame rate hooks: not installed (no memory next to the game's code)");
        return;
    }
    live = new (data) Live{};
    std::memcpy(&model_turn_base, At(0x4928dfc), sizeof(model_turn_base));
    live->model_turn = model_turn_base;
    live->shortest = ShortestFrame(EmulatorSettings.GetBbFpsLimit());
    // The core first: without all three the game keeps its own frame rate.
    if (!(HookPacerMode() && HookPacerLate() && HookStep())) {
        LOG_ERROR(Loader, "Frame rate hooks: the frame pacer's hooks did not fit");
        return;
    }
    active.store(true, std::memory_order_release);
    const int steps = int(HookFixedStep(0x17f9b71, -0x48)) + int(HookFixedStep(0x1572a7d, -0x30));
    const int run = HookRunSpeed();
    const bool cloth = HookClothStiffness();
    int redirected = 0;
    for (const Redirect& r : Redirects) {
        if (KeptPatchWrites(r.constant)) {
            LOG_INFO(Loader, "Frame rate hooks: {:#x} keeps the value an enabled patch sets",
                     r.site);
        } else if (RedirectOperand(r)) {
            ++redirected;
        } else {
            LOG_WARNING(Loader, "Frame rate hooks: {:#x} is not the expected code, left as it is",
                        r.site);
        }
    }
    LOG_INFO(Loader,
             "Frame rate hooks: on, limit {:.0f} FPS. {} of 2 fixed steps, {} of {} "
             "per-frame values follow the frame's time, the run's wall check {}, the cloth's "
             "stiffness {}",
             1.0f / live->shortest, steps, redirected, std::size(Redirects),
             run == 1   ? "by speed"
             : run == 2 ? "by speed (a Sprint Fix patch did it already)"
                        : "unchanged (not found)",
             cloth ? "by solves per second" : "per frame (not found)");
}

void Prepare() {
    patch_writes.clear();
    auto* param_sfo = Common::Singleton<PSF>::Instance();
    const auto version = param_sfo->GetString("APP_VER").value_or("");
    deciding = EmulatorSettings.IsBbFrameHooks() &&
               Common::BbTuning::IsBloodborne(MemoryPatcher::g_game_serial) && version == "01.09" &&
               MemoryPatcher::g_eboot_name == "eboot.bin";
}

void CheckPatch(std::string_view name, u64 address, u64 size) {
    if (!deciding || address < 0x400000) {
        return;
    }
    const u64 begin = address - 0x400000;
    const u64 end = begin + size;
    auto it = patch_writes.find(name);
    if (it == patch_writes.end()) {
        it = patch_writes.emplace(std::string{name}, PatchWrites{}).first;
    }
    PatchWrites& writes = it->second;
    const bool was = IsFrameRatePatch(writes);
    for (const auto& [site, length] : CoreSites) {
        writes.core |= begin < site + length && end > site;
    }
    for (const Redirect& r : Redirects) {
        if (begin < r.constant + 4 && end > r.constant) {
            writes.constants.insert(r.constant);
        }
    }
    if (!was && IsFrameRatePatch(writes)) {
        LOG_INFO(Loader, "Frame rate hooks: patch {} skipped (a frame rate patch)", name);
    }
}

bool ShouldSkipPatch(std::string_view name) {
    if (!deciding) {
        return false;
    }
    const auto it = patch_writes.find(name);
    return it != patch_writes.end() && IsFrameRatePatch(it->second);
}

bool Active() {
    return active.load(std::memory_order_acquire);
}

float CapPeriod() {
    if (!Active() || live->shortest <= 1.01f / MaxRate) {
        return 0.0f;
    }
    return live->shortest;
}

void SetLimit(int fps) {
    if (!Active()) {
        return;
    }
    live->shortest = ShortestFrame(fps);
    LOG_INFO(Loader, "Frame rate hooks: limit {:.0f} FPS", 1.0f / live->shortest);
}

} // namespace Core::BbFrameRate
