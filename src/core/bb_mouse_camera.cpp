// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <new>
#include <span>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "common/logging/log.h"
#include "common/types.h"
#include "common/memory_patcher.h"
#include "common/path_util.h"
#include "core/bb_mouse_camera.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace Core::BbMouseCamera {
namespace {

// The follow camera's update (NS_SPRJ::ChrExFollowCam) stores the free camera's angles after the
// stick has turned them: vmovss [r13+0x140], xmm0 (pitch); vextractps [r13+0x144], xmm1, 1 (yaw).
// 19 bytes, unique in the 1.09 eboot. The camera object keeps its pitch limits at +0x1ec
// (highest) and +0x1f0 (lowest); [rbp-0x3c4] tells the update the camera had input this frame, so
// it doesn't start turning back on its own, and then eases the pitch to xmm0's.
constexpr std::array<u8, 19> TurnStore{0xc4, 0xc1, 0x7a, 0x11, 0x85, 0x40, 0x01, 0x00, 0x00, 0xc4,
                                       0xc3, 0x79, 0x17, 0x8d, 0x44, 0x01, 0x00, 0x00, 0x01};

// The camera turning by itself as the character moves: 9-byte stores in the same update, relative
// to TurnStore (the yaw read back from where the camera is, the turn away from walls). With the
// mouse they fight every turn, so they are held. The pitch return's weight is left alone: held,
// the pitch is pulled back to level every frame.
struct HoldSite {
    std::ptrdiff_t offset;
    std::array<u8, 9> bytes;
};
constexpr std::array<HoldSite, 2> HoldSites{{
    {-0x76c, {0xc4, 0xc1, 0x7a, 0x11, 0x85, 0x44, 0x01, 0x00, 0x00}},
    {+0xf92, {0xc4, 0xc1, 0x7a, 0x11, 0x85, 0x94, 0x02, 0x00, 0x00}},
}};

struct Data {
    std::atomic<u64> pending; // radians to add: pitch (low 32 bits), yaw (high), as floats
    float pi;
    float minus_pi;
    float two_pi;
    float unused;
    u8* camera; // the camera object of the last update (diagnostics)
};
static_assert(offsetof(Data, camera) == 24);

Data* data = nullptr;
bool installed = false;

struct Settings {
    float radians_per_count = 0.0f;
    bool invert_x = false, invert_y = false;
};
std::atomic<u64> settings_ms{0};
Settings settings;

std::string Setting(const std::vector<std::string>& lines, const std::string& key,
                    const std::string& fallback) {
    for (const auto& line : lines)
        if (line.rfind(key + "=", 0) == 0)
            return line.substr(key.size() + 1);
    return fallback;
}

std::vector<std::string> ReadSettings() {
    std::vector<std::string> lines;
    std::ifstream file{Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "dlss.ini"};
    for (std::string line; std::getline(file, line);)
        lines.push_back(line);
    return lines;
}

// Sensitivity 1.5 (F1's default) is 0.225 degrees a count.
void LoadSettings() {
    const auto lines = ReadSettings();
    float sensitivity = static_cast<float>(std::atof(Setting(lines, "mouse_sensitivity", "1.5").c_str()));
    if (!(sensitivity > 0.0f))
        sensitivity = 1.5f;
    settings.radians_per_count = sensitivity * 0.15f * 0.0174532924f;
    settings.invert_x = Setting(lines, "mouse_invert_x", "0") == "1";
    settings.invert_y = Setting(lines, "mouse_invert_y", "0") == "1";
}

u64 NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

u64 Pack(float pitch, float yaw) {
    u32 a, b;
    std::memcpy(&a, &pitch, 4);
    std::memcpy(&b, &yaw, 4);
    return u64(a) | (u64(b) << 32);
}

void Unpack(u64 value, float& pitch, float& yaw) {
    const u32 a = u32(value), b = u32(value >> 32);
    std::memcpy(&pitch, &a, 4);
    std::memcpy(&yaw, &b, 4);
}

// The store redirected here: the game's two stores, then this frame's mouse turn added to the
// angles in the camera object, the pitch kept within its limits and the yaw within a turn.
std::vector<u8> BuildStub(const u8* store, const Data* d) {
    std::vector<u8> code;
    const auto emit = [&](std::initializer_list<u8> bytes) { code.insert(code.end(), bytes); };
    const auto emit64 = [&](u64 value) {
        for (int i = 0; i < 8; ++i)
            code.push_back(u8(value >> (i * 8)));
    };
    code.insert(code.end(), TurnStore.begin(), TurnStore.end()); // the game's stores
    emit({0x48, 0x8d, 0x64, 0x24, 0x80});                         // lea rsp, [rsp-0x80] (red zone)
    emit({0x9c, 0x50, 0x51});                                     // pushfq; push rax; push rcx
    emit({0x48, 0x83, 0xec, 0x60});                               // sub rsp, 0x60
    emit({0xc5, 0xfe, 0x7f, 0x14, 0x24});                         // vmovdqu [rsp], ymm2
    emit({0xc5, 0xfe, 0x7f, 0x5c, 0x24, 0x20});                   // vmovdqu [rsp+0x20], ymm3
    emit({0xc5, 0xfe, 0x7f, 0x64, 0x24, 0x40});                   // vmovdqu [rsp+0x40], ymm4
    emit({0x48, 0xb8});                                           // mov rax, data
    emit64(reinterpret_cast<u64>(d));
    emit({0x4c, 0x89, 0x68, 0x18}); // mov [rax+24], r13
    emit({0x31, 0xc9});             // xor ecx, ecx
    emit({0x48, 0x87, 0x08});       // xchg [rax], rcx: take the pending turn
    emit({0x48, 0x85, 0xc9});       // test rcx, rcx
    emit({0x0f, 0x84});             // jz done
    const size_t jz_at = code.size();
    emit({0, 0, 0, 0});
    emit({0xc4, 0xe1, 0xf9, 0x6e, 0xd1}); // vmovq xmm2, rcx
    // pitch
    emit({0xc4, 0xc1, 0x6a, 0x58, 0x9d, 0x40, 0x01, 0x00, 0x00}); // vaddss xmm3, xmm2, [r13+0x140]
    // Within the two limits, whichever of them is the higher.
    emit({0xc4, 0xc1, 0x7a, 0x10, 0xa5, 0xec, 0x01, 0x00, 0x00}); // vmovss xmm4, [r13+0x1ec]
    emit({0xc4, 0xc1, 0x5a, 0x5f, 0xa5, 0xf0, 0x01, 0x00, 0x00}); // vmaxss xmm4, xmm4, [r13+0x1f0]
    emit({0xc5, 0xe2, 0x5d, 0xdc});                               // vminss xmm3, xmm3, xmm4
    emit({0xc4, 0xc1, 0x7a, 0x10, 0xa5, 0xec, 0x01, 0x00, 0x00}); // vmovss xmm4, [r13+0x1ec]
    emit({0xc4, 0xc1, 0x5a, 0x5d, 0xa5, 0xf0, 0x01, 0x00, 0x00}); // vminss xmm4, xmm4, [r13+0x1f0]
    emit({0xc5, 0xe2, 0x5f, 0xdc});                               // vmaxss xmm3, xmm3, xmm4
    emit({0xc4, 0xc1, 0x7a, 0x11, 0x9d, 0x40, 0x01, 0x00, 0x00}); // vmovss [r13+0x140], xmm3
    // The update goes on with the pitch in xmm0: with input this frame it becomes the pitch the
    // camera eases to (+0x150), so it must be the turned one.
    emit({0xc5, 0xfa, 0x10, 0xc3}); // vmovss xmm0, xmm0, xmm3
    // yaw
    emit({0xc5, 0xe9, 0x73, 0xd2, 0x20});                         // vpsrlq xmm2, xmm2, 32
    emit({0xc4, 0xc1, 0x6a, 0x58, 0x9d, 0x44, 0x01, 0x00, 0x00}); // vaddss xmm3, xmm2, [r13+0x144]
    emit({0xc5, 0xf8, 0x2e, 0x58, 0x08});                         // vucomiss xmm3, [rax+8] (pi)
    emit({0x76, 0x05});                                           // jbe +5
    emit({0xc5, 0xe2, 0x5c, 0x58, 0x10});                         // vsubss xmm3, xmm3, [rax+16]
    emit({0xc5, 0xf8, 0x2e, 0x58, 0x0c});                         // vucomiss xmm3, [rax+12] (-pi)
    emit({0x73, 0x05});                                           // jae +5
    emit({0xc5, 0xe2, 0x58, 0x58, 0x10});                         // vaddss xmm3, xmm3, [rax+16]
    emit({0xc4, 0xc1, 0x7a, 0x11, 0x9d, 0x44, 0x01, 0x00, 0x00}); // vmovss [r13+0x144], xmm3
    emit({0xc6, 0x85, 0x3c, 0xfc, 0xff, 0xff, 0x01});             // mov byte [rbp-0x3c4], 1
    // done:
    const u32 rel = u32(code.size() - (jz_at + 4));
    std::memcpy(&code[jz_at], &rel, 4);
    emit({0xc5, 0xfe, 0x6f, 0x14, 0x24});                   // vmovdqu ymm2, [rsp]
    emit({0xc5, 0xfe, 0x6f, 0x5c, 0x24, 0x20});             // vmovdqu ymm3, [rsp+0x20]
    emit({0xc5, 0xfe, 0x6f, 0x64, 0x24, 0x40});             // vmovdqu ymm4, [rsp+0x40]
    emit({0x48, 0x83, 0xc4, 0x60});                         // add rsp, 0x60
    emit({0x59, 0x58, 0x9d});                               // pop rcx; pop rax; popfq
    emit({0x48, 0x8d, 0xa4, 0x24, 0x80, 0x00, 0x00, 0x00}); // lea rsp, [rsp+0x80]
    emit({0xff, 0x25, 0x00, 0x00, 0x00, 0x00});             // jmp [rip+0]
    emit64(reinterpret_cast<u64>(store + TurnStore.size()));
    return code;
}

} // namespace

void Install() {
#ifdef _WIN32
    if (installed)
        return;
    auto* image = reinterpret_cast<u8*>(MemoryPatcher::g_eboot_address);
    const size_t size = MemoryPatcher::g_eboot_image_size;
    if (!image || size < TurnStore.size())
        return;
    u8* store = nullptr;
    for (size_t i = 0; i + TurnStore.size() <= size; ++i) {
        if (image[i] != TurnStore[0] || std::memcmp(image + i, TurnStore.data(), TurnStore.size()))
            continue;
        if (store) {
            LOG_WARNING(Loader, "Mouse camera: the camera code appears twice, not installed");
            return;
        }
        store = image + i;
    }
    if (!store) {
        LOG_INFO(Loader, "Mouse camera: camera code not found (not Bloodborne 1.09)");
        return;
    }

    auto* block = static_cast<u8*>(
        VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
    if (!block)
        return;
    data = new (block) Data{};
    data->pi = 3.14159274f;
    data->minus_pi = -3.14159274f;
    data->two_pi = 6.28318548f;
    u8* stub = block + 64;
    const auto code = BuildStub(store, data);
    std::memcpy(stub, code.data(), code.size());

    DWORD old{};
    if (!VirtualProtect(store - 0x800, 0x1800, PAGE_EXECUTE_READWRITE, &old))
        return;
    // jmp [rip+0] to the stub over the store, the rest int3.
    std::array<u8, 19> jump{0xff, 0x25, 0x00, 0x00, 0x00, 0x00};
    const u64 to = reinterpret_cast<u64>(stub);
    std::memcpy(jump.data() + 6, &to, 8);
    std::memset(jump.data() + 14, 0xcc, jump.size() - 14);
    std::memcpy(store, jump.data(), jump.size());

    const auto lines = ReadSettings();
    const bool hold = Setting(lines, "camera_auto_rotation", "0") != "1";
    int held = 0;
    for (const auto& site : HoldSites) {
        u8* at = store + site.offset;
        if (hold && !std::memcmp(at, site.bytes.data(), site.bytes.size())) {
            std::memset(at, 0x90, site.bytes.size());
            ++held;
        }
    }
    FlushInstructionCache(GetCurrentProcess(), store - 0x800, 0x1800);
    LoadSettings();
    settings_ms = NowMs();
    installed = true;
    LOG_INFO(Loader, "Mouse camera: installed (auto-rotation {})",
             hold ? fmt::format("held at {} of {} sites", held, HoldSites.size()) : std::string{"kept"});
#endif
}

bool Installed() {
    return installed;
}

void AddMotion(float dx, float dy) {
    if (!installed || (dx == 0.0f && dy == 0.0f))
        return;
    const u64 now = NowMs();
    if (now - settings_ms.load(std::memory_order_relaxed) > 500) {
        settings_ms = now;
        LoadSettings();
    }
    static u64 logged_ms = 0;
    if (data->camera && now - logged_ms > 500) {
        logged_ms = now;
        float pitch_now, yaw_now, limit_a, limit_b;
        std::memcpy(&pitch_now, data->camera + 0x140, 4);
        std::memcpy(&yaw_now, data->camera + 0x144, 4);
        std::memcpy(&limit_a, data->camera + 0x1ec, 4);
        std::memcpy(&limit_b, data->camera + 0x1f0, 4);
        LOG_DEBUG(Loader, "Mouse camera: dx {} dy {} pitch {:.4f} yaw {:.4f} limits {:.4f} {:.4f}",
                 dx, dy, pitch_now, yaw_now, limit_a, limit_b);
    }
    const float pitch = dy * settings.radians_per_count * (settings.invert_y ? -1.0f : 1.0f);
    const float yaw = dx * settings.radians_per_count * (settings.invert_x ? -1.0f : 1.0f);
    u64 was = data->pending.load(std::memory_order_relaxed);
    for (;;) {
        float p, y;
        Unpack(was, p, y);
        if (data->pending.compare_exchange_weak(was, Pack(p + pitch, y + yaw),
                                                std::memory_order_acq_rel,
                                                std::memory_order_relaxed))
            return;
    }
}

} // namespace Core::BbMouseCamera
