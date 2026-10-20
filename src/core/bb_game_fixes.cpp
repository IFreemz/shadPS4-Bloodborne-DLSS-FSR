// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <atomic>
#include <cstring>

#include "common/crash_crumbs.h"
#include "common/logging/log.h"
#include "common/memory_patcher.h"
#include "common/types.h"
#include "core/bb_game_fixes.h"

namespace Core::BbGameFixes {
namespace {

// Offsets in the game's image (patch-file address - 0x400000).
u8* At(u64 offset) {
    return reinterpret_cast<u8*>(MemoryPatcher::g_eboot_address) + offset;
}

bool Fits(u64 offset, u64 size) {
    return offset + size <= MemoryPatcher::g_eboot_image_size;
}

// Save-load crash ("Unimplemented PM4 type 0"): while an effect sets up, the
// game's main thread clears three small history targets (0xebb960) through the same command
// buffer its render thread is filling at that moment. The two threads advance one cursor, and the
// buffer gets holes of zeros the command processor cannot decode. These three clears are done here
// instead, by writing the zeros straight into the targets' memory; every other clear stays the
// game's own.
constexpr u64 NativeClear = 0x2168630; // the game's clear (context, state, render target, colour)
constexpr std::array<u64, 3> ClearSites{0xebbc86, 0xebbca4, 0xebbcc2}; // call NativeClear
// Room for the jump to FxInitClear, in the page of the FPS++ patches' code caves (patch address
// 0x054A0000; theirs end at 0x054A00F3).
constexpr u64 Trampoline = 0x50a0f00;
constexpr u64 TrampolineSize = 16;

std::atomic<u64> fx_cleared{0};
std::atomic<u64> fx_passed{0};

// The game's render target descriptor (at +0x10 of its object) when it is one of those targets:
// 2x2 RG16 float, tiled, no compression or fast clear, at most 32 KiB, cleared to (0, 0, 0, 1)
// (only red and green are stored, so all zeros is the same picture in every tiling).
bool PlanZeroClear(const u32* target, const float* color, u64& address, u32& bytes) {
    if (color[0] != 0.0f || color[1] != 0.0f || color[2] != 0.0f || color[3] != 1.0f ||
        target[0] == 0 || target[3] != 0 || target[6] != 0x00020002 || target[7] != 0 ||
        target[8] != 0 || target[9] != 0 || target[10] != 0) {
        return false;
    }
    const u32 info = target[4], attrib = target[5];
    if (((info >> 2) & 31) != 5 || ((info >> 8) & 7) != 7 ||
        (info & (3u | (1u << 7) | (3u << 11) | (3u << 13) | (1u << 28) | (1u << 31))) != 0 ||
        (attrib & 31) != 14 || (attrib & (31u << 12)) != 0) {
        return false;
    }
    const u64 pitch = u64((target[1] & 0x7ff) + 1) * 8;
    const u64 texels = u64((target[2] & 0x3fffff) + 1) * 64;
    address = u64(target[0]) << 8;
    bytes = u32(texels * 4);
    return pitch >= 2 && texels / pitch >= 2 && texels % pitch == 0 && texels * 4 <= 32768 &&
           address + bytes <= (1ull << 40);
}

void PS4_SYSV_ABI FxInitClear(void* context, void* state, void* render_target, const float* color) {
    if (render_target && color) {
        u32 target[11];
        std::memcpy(target, static_cast<const u8*>(render_target) + 0x10, sizeof(target));
        u64 address = 0;
        u32 bytes = 0;
        if (PlanZeroClear(target, color, address, bytes)) {
            // A CPU write: the texture cache sees it like any other write by the game.
            // Breadcrumb: a crash here shows up as this step in the crash log.
            Common::Crumbs::Add("save-load fix: target cleared (address, size)", address, bytes);
            std::memset(reinterpret_cast<void*>(address), 0, bytes);
            const u64 n = fx_cleared.fetch_add(1, std::memory_order_relaxed) + 1;
            if (n <= 3 || (n & (n - 1)) == 0) {
                LOG_INFO(Loader, "Game fix: effect history target cleared directly ({} so far)", n);
            }
            return;
        }
    }
    const u64 n = fx_passed.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n <= 3 || (n & (n - 1)) == 0) {
        LOG_INFO(Loader, "Game fix: effect clear left to the game (other target, {} so far)", n);
    }
    using Clear = void(PS4_SYSV_ABI*)(void*, void*, void*, const float*);
    Common::Crumbs::Add("save-load fix: the game's own clear");
    reinterpret_cast<Clear>(At(NativeClear))(context, state, render_target, color);
}

bool InstallFxInitClear() {
    if (!Fits(Trampoline, TrampolineSize) || !Fits(NativeClear, 1)) {
        return false;
    }
    for (const u64 site : ClearSites) {
        if (!Fits(site, 5) || At(site)[0] != 0xe8) {
            return false;
        }
        s32 rel;
        std::memcpy(&rel, At(site) + 1, sizeof(rel));
        if (site + 5 + s64(rel) != NativeClear) {
            return false;
        }
    }
    // The trampoline's room must be unused (zeros).
    for (u64 i = 0; i < TrampolineSize; ++i) {
        if (At(Trampoline)[i] != 0) {
            return false;
        }
    }
    // movabs rax, FxInitClear; jmp rax
    u8* code = At(Trampoline);
    const u64 target = reinterpret_cast<u64>(&FxInitClear);
    code[0] = 0x48;
    code[1] = 0xb8;
    std::memcpy(code + 2, &target, sizeof(target));
    code[10] = 0xff;
    code[11] = 0xe0;
    for (const u64 site : ClearSites) {
        const s32 rel = s32(s64(Trampoline) - s64(site + 5));
        std::memcpy(At(site) + 1, &rel, sizeof(rel));
    }
    return true;
}

} // namespace

void Install() {
    if (!MemoryPatcher::g_eboot_address) {
        return;
    }
    LOG_INFO(Loader, "Game fix: save-load command buffer race {}",
             InstallFxInitClear() ? "on" : "not installed (other code there)");
}

} // namespace Core::BbGameFixes
