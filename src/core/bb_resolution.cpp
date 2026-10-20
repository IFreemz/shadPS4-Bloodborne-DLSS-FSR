// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include <SDL3/SDL_properties.h>
#include <SDL3/SDL_video.h>
#include "common/bb_tuning.h"
#include "common/logging/log.h"
#include "common/memory_patcher.h"
#include "common/path_util.h"
#include "common/singleton.h"
#include "core/bb_resolution.h"
#include "core/emulator_settings.h"
#include "core/file_format/psf.h"
#include "core/file_sys/fs.h"
#include "emulator.h"

#include "sdl_window.h"

extern Frontend::WindowSDL* g_window;

namespace Core::BbResolution {
namespace {

struct PatchLine {
    u32 address; // as in the patch file (eboot address + 0x400000)
    const char* bytes;
};

// Kyo's Resolution Patch 2560x1440 (16:9) without its graphics heap lines: the same bytes as every
// 16:9 Resolution Patch, the output size written in four places.
constexpr PatchLine ResolutionLines[] = {
    {0x00E73A60, "31C0C3"},
    {0x00E73AA0, "31C0C3"},
    {0x00FC96ED, "EB13"},
    {0x00FC96FD, "9090909090"},
    {0x00FE93D1, "9090909090"},
    {0x00FF2952, "9090909090"},
    {0x00FF2975, "9090909090"},
    {0x00FF35C2, "9090909090"},
    {0x00FF35E9, "9090909090"},
    {0x00FF37A2, "9090909090"},
    {0x00FF37C9, "9090909090"},
    {0x0183A35D, "398EE33F"},
    {0x019E83AF, "B8800700000F1F4000"},
    {0x01A44357, "B8800700000F1F4000"},
    {0x01A44365, "B8380400000F1F4000"},
    {0x01A44C55, "B8800700000F1F4000"},
    {0x01A44C63, "B8380400000F1F4000"},
    {0x01A452C7, "B8800700000F1F4000"},
    {0x01A452D5, "B8380400000F1F4000"},
    {0x01FFD491, "B8800700000F1F4000"},
    {0x01FFD4D1, "B8380400000F1F4000"},
    {0x0212C674, "BE800700000F1F4000"},
    {0x0212C6A4, "BE380400000F1F4000"},
    {0x02196A6B, "B8000A00000F1F4000"},
    {0x02196A7A, "B8A00500000F1F4000"},
    {0x02358554, "B8000A00000F1F4000B9A00500000F1F4000"},
    {0x0241848F, "B8800700000F1F4000B9380400000F1F4000"},
    {0x02438BB9, "B8800700000F1F4000"},
    {0x02438BC7, "B8380400000F1F4000"},
    {0x02438F59, "B8800700000F1F4000"},
    {0x02438F67, "B8380400000F1F4000"},
    {0x024B56D2, "9090909090909090909090"},
    {0x02594AA6,
     "418D4DFF83F97F7718C78424A0000000000A0000C78424A4000000A0050000EB1E418B8CC480010000898C24A0000"
     "000418B8CC484010000898C24A40000004D8DB4C484010000C68424B000000057C78424A80000003C000000C68424"
     "AC00000001498B8CC48801000048898C24BC000000C68424C400000003C68424C80000000390"},
    {0x026C3DE4, "B8A0000000"},
    {0x026C3DF0, "B85A00000090"},
    {0x026C45BA, "48B8A00000005A0000004889842400020000909090"},
    {0x055289F8, "A00000005A000000"},
};

// The Increased Graphics Heap Sizes patch (auser1337), part of Kyo's Resolution Patches above
// 1080p: a scene above 1080p doesn't fit the game's own graphics heap.
constexpr PatchLine HeapLines[] = {
    {0x04B36DDA, "00F7"},
    {0x04B36DE2, "00A0"},
};

// Kyo's Decoupled 1080p UI (1280x720 render res): the 3D scene's size, written in nine places,
// the buffers that follow it derived from it.
constexpr PatchLine DecoupledLines[] = {
    {0x01258F2C, "E8BF2BAE030F1F440000"},
    {0x0125B1A3, "8B03D1E88904248B4304D1E889442404"},
    {0x0125B1D6, "8B03C1E8028904248B4304C1E80289442404"},
    {0x025D818D,
     "488B0534853603488B80900000008B8880000000448B8084000000C5FA1005E415750231F631D2C5F057C96690"},
    {0x02697AEA, "B90005000090BAD002000090"},
    {0x0269C7D5, "E9FAF269029090909090909090"},
    {0x026C3AF3, "EB"},
    {0x026C3DE4, "418B07D1E8"},
    {0x026C3DF0, "418B4704D1E8"},
    {0x026C65F6, "EB"},
    {0x026C6624, "BA000500009090"},
    {0x026C662E, "B8D00200009090"},
    {0x026C6735, "EB"},
    {0x026C6760, "BA000500009090"},
    {0x026C676A, "B8D00200009090"},
    {0x026F8CE1, "B90005000090BED002000090"},
    {0x026F93D8, "BB000500009090BED00200009090"},
    {0x026F99DB, "BE000500009090B9D00200009090"},
    {0x04D3BAD4, "C744242000050000C7442424D0020000E9F90C96FD"},
    {0x04D3BAF0, "488B05D14BC000488B80900000008B88840000008B8080000000C3"},
};

// The sizes in the templates above.
constexpr u32 TemplateOutputWidth = 2560, TemplateOutputHeight = 1440;
constexpr u32 TemplateRenderWidth = 1280, TemplateRenderHeight = 720;
// Kyo's patches go up to 4K, and above a 1080p scene the game needs a bigger graphics heap.
constexpr u32 MaxOutputHeight = 2160, MinOutputHeight = 720;

// Only the usual 16:9 sizes, as in Kyo's patches: the game halves and quarters the scene's size
// for some of its buffers, and other sizes (1706x960) don't halve evenly, which breaks them.
constexpr std::array<u32, 8> StandardHeights{360, 540, 720, 810, 900, 1080, 1440, 2160};

// The old preset names: the nearest usual size to that share of the output height.
constexpr std::array<std::pair<const char*, float>, 4> RenderScales{{
    {"quality", 2.0f / 3.0f},
    {"balanced", 0.58f},
    {"performance", 0.5f},
    {"ultra", 1.0f / 3.0f},
}};

Choice applied;
bool deciding{};
std::vector<std::string> conflicting; // enabled patches that set sizes F1 sets

// What a patch must not change while F1 sets the resolution, from the templates: the bytes
// holding a size, and the code bytes around them (the same in every size). The cave holding the
// frame size (also written by the Performance Patch) is rewritten after
// the patches instead, so a patch touching them still applies.
struct Owned {
    std::unordered_map<u32, u8> code;  // address -> byte F1 writes
    std::unordered_set<u32> sizes;     // bytes holding a size
    std::unordered_set<u32> rewritten; // frame-size cave
};

template <size_t N>
void AddOwned(Owned& owned, const PatchLine (&lines)[N], u32 w, u32 h, bool rewritten);

const Owned& OwnedBytes();

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

u32 WidthFor(u32 height) {
    return height * 16 / 9;
}

// The largest usual 16:9 size inside the window, between 720p and 4K.
u32 FitHeight(u32 width, u32 height) {
    const u32 fit = (u64{width} * 9 >= u64{height} * 16) ? height : u32(u64{width} * 9 / 16);
    u32 best = 720;
    for (const u32 h : StandardHeights)
        if (h >= MinOutputHeight && h <= MaxOutputHeight && h <= fit)
            best = std::max(best, h);
    return best;
}

std::pair<u32, u32> WindowSize() {
    if (g_window && g_window->GetWidth() > 0 && g_window->GetHeight() > 0)
        return {u32(g_window->GetWidth()), u32(g_window->GetHeight())};
    return {u32(EmulatorSettings.GetWindowWidth()), u32(EmulatorSettings.GetWindowHeight())};
}

std::vector<u8> Bytes(const char* hex) {
    std::vector<u8> out;
    for (size_t i = 0; hex[i] && hex[i + 1]; i += 2)
        out.push_back(u8(std::stoul(std::string{hex + i, 2}, nullptr, 16)));
    return out;
}

// Each place the template's width or height is written gets the new one. Places are found
// first, so a new width equal to the template's height (1440x810) isn't replaced again.
std::vector<u8> WithSize(const char* hex, u32 from_w, u32 from_h, u32 w, u32 h) {
    auto bytes = Bytes(hex);
    const auto original = bytes;
    const auto put = [&](u32 from, u32 to) {
        u8 find[4], value[4];
        std::memcpy(find, &from, 4);
        std::memcpy(value, &to, 4);
        for (size_t i = 0; i + 4 <= original.size(); ++i)
            if (!std::memcmp(original.data() + i, find, 4))
                std::memcpy(bytes.data() + i, value, 4);
    };
    put(from_w, w);
    put(from_h, h);
    return bytes;
}

template <size_t N>
int Write(const PatchLine (&lines)[N], u32 from_w, u32 from_h, u32 w, u32 h) {
    auto* image = reinterpret_cast<u8*>(MemoryPatcher::g_eboot_address);
    const u64 size = MemoryPatcher::g_eboot_image_size;
    int written = 0;
    for (const auto& line : lines) {
        const auto bytes = WithSize(line.bytes, from_w, from_h, w, h);
        const u64 offset = u64{line.address} - 0x400000;
        if (line.address < 0x400000 || offset + bytes.size() > size)
            continue;
        std::memcpy(image + offset, bytes.data(), bytes.size());
        ++written;
    }
    return written;
}

template <size_t N>
void AddOwned(Owned& owned, const PatchLine (&lines)[N], u32 w, u32 h, bool rewritten) {
    for (const auto& line : lines) {
        const auto bytes = Bytes(line.bytes);
        std::vector<bool> size(bytes.size());
        for (const u32 value : {w, h}) {
            u8 find[4];
            std::memcpy(find, &value, 4);
            for (size_t i = 0; i + 4 <= bytes.size(); ++i)
                if (!std::memcmp(bytes.data() + i, find, 4))
                    std::fill_n(size.begin() + i, 4, true);
        }
        const bool cave = line.address == 0x02594AA6; // the frame-size cave
        for (size_t i = 0; i < bytes.size(); ++i) {
            const u32 at = line.address + u32(i);
            if (rewritten || cave)
                owned.rewritten.insert(at);
            else if (size[i])
                owned.sizes.insert(at);
            else
                owned.code.insert_or_assign(at, bytes[i]); // decoupled, written last, wins
        }
    }
}

const Owned& OwnedBytes() {
    static const Owned owned = [] {
        Owned o;
        AddOwned(o, ResolutionLines, TemplateOutputWidth, TemplateOutputHeight, false);
        AddOwned(o, DecoupledLines, TemplateRenderWidth, TemplateRenderHeight, false);
        return o;
    }();
    return owned;
}

} // namespace

std::vector<u32> RenderHeights(u32 output_height) {
    // Below the output, up to 1080p, and no more than 3x below it (the upscalers' limit).
    std::vector<u32> heights;
    for (const u32 h : StandardHeights)
        if (h < output_height && output_height <= 3 * h)
            heights.push_back(h);
    return heights;
}

Choice Resolve(const std::string& enabled, const std::string& output, const std::string& render,
               u32 window_width, u32 window_height) {
    Choice choice;
    choice.enabled = enabled == "1";
    u32 w = 0, h = 0;
    if (std::sscanf(output.c_str(), "%ux%u", &w, &h) != 2 || !w || !h) {
        w = window_width;
        h = window_height;
    }
    const u32 oh = FitHeight(w, h);
    choice.output_width = WidthFor(oh);
    choice.output_height = oh;

    u32 rh = oh; // native
    if (render != "native") {
        u32 wanted = u32(std::atoi(render.c_str()));
        for (const auto& [name, scale] : RenderScales)
            if (render == name)
                wanted = u32(std::lround(oh * double(scale)));
        const auto heights = RenderHeights(oh);
        if (wanted < oh && !heights.empty()) {
            rh = heights.front();
            for (const u32 option : heights) // nearest, the larger one on a tie
                if (std::abs(int(option) - int(wanted)) <= std::abs(int(rh) - int(wanted)))
                    rh = option;
        }
    }
    choice.render_height = rh;
    choice.render_width = WidthFor(rh);
    return choice;
}

Choice Preview() {
    const auto lines = ReadSettings();
    const auto [w, h] = WindowSize();
    auto choice = Resolve(Setting(lines, "res_control", "0"), Setting(lines, "res_output", "auto"),
                          Setting(lines, "res_render", "quality"), w, h);
    choice.heap = Setting(lines, "res_heap", "auto");
    if (choice.heap == "on")
        choice.heap = "full";
    if (choice.heap != "full" && choice.heap != "first" && choice.heap != "smaller" &&
        choice.heap != "off")
        choice.heap = "auto";
    return choice;
}

void Prepare() {
    applied = Preview();
    deciding = applied.enabled;
    conflicting.clear();
    if (!applied.enabled)
        return;
    auto* param_sfo = Common::Singleton<PSF>::Instance();
    const auto version = param_sfo->GetString("APP_VER").value_or("");
    if (!Common::BbTuning::IsBloodborne(MemoryPatcher::g_game_serial) || version != "01.09" ||
        MemoryPatcher::g_eboot_name != "eboot.bin") {
        applied.problem = "Needs Bloodborne 1.09: the game's resolution patches are used instead.";
        applied.enabled = deciding = false;
        LOG_INFO(Loader, "Resolution: not set from F1 ({} {})", MemoryPatcher::g_game_serial,
                 version);
    }
}

void CheckPatch(std::string_view name, u64 address, std::span<const u8> bytes) {
    if (!deciding)
        return;
    const auto& owned = OwnedBytes();
    bool conflict = false;
    for (size_t i = 0; i < bytes.size(); ++i) {
        const u32 at = u32(address + i);
        if (owned.sizes.contains(at))
            conflict = true;
        // The heap sizes too, unless F1 leaves the heap to the patch file (Off).
        else if (applied.heap != "off" && std::ranges::any_of(HeapLines, [at](const auto& line) {
                     return at >= line.address && at < line.address + 2;
                 }))
            conflict = true;
        else if (const auto it = owned.code.find(at);
                 it != owned.code.end() && it->second != bytes[i])
            conflict = true;
    }
    if (conflict && std::ranges::find(conflicting, name) == conflicting.end()) {
        conflicting.emplace_back(name);
        LOG_INFO(Loader, "Resolution: patch {} sets sizes F1 sets, skipped", name);
    }
}

bool ShouldSkipPatch(std::string_view name) {
    return deciding && std::ranges::find(conflicting, name) != conflicting.end();
}

const std::vector<std::string>& SkippedPatches() {
    return conflicting;
}

void Install() {
    if (!applied.enabled)
        return;
    const u32 ow = applied.output_width, oh = applied.output_height;
    const u32 rw = applied.render_width, rh = applied.render_height;
    int lines = 0;
    // A scene above 1080p (Native at 1440p, 1440p at 4K) needs a bigger graphics heap. The
    // game's two heaps are 0x8EF00000 and 0x37600000 bytes; the patch makes them 0xF7000000 and
    // 0xA0000000, and the second then no longer fits below the GPU driver's page at 0xFE0000000:
    // it goes right after it, where rendering breaks (missing textures, flicker) on some cards.
    // Auto raises the second to 0x90000000 instead, which fits. F1 can also pick the full patch,
    // the first heap only, the smaller second heap at any size, or off: the game's own heaps,
    // unless a heap patch in the patch file changes them.
    const auto& mode = applied.heap;
    const bool heap = mode == "full" || mode == "first" || mode == "smaller" ||
                      (mode == "auto" && rh > HeapRenderHeight);
    if (heap) {
        constexpr PatchLine first[] = {HeapLines[0]};
        lines += Write(first, 0, 0, 0, 0);
        if (mode == "auto" || mode == "smaller") {
            constexpr PatchLine second[] = {{0x04B36DE2, "0090"}};
            lines += Write(second, 0, 0, 0, 0);
        } else if (mode == "full") {
            constexpr PatchLine second[] = {HeapLines[1]};
            lines += Write(second, 0, 0, 0, 0);
        }
    }
    LOG_INFO(Loader, "Resolution: graphics heap {} ({})",
             heap            ? "raised"
             : mode == "off" ? "left to the patch file"
                             : "the game's own",
             mode);
    // 1080p is the game's own frame size: nothing to change there.
    if (ow != 1920 || oh != 1080)
        lines += Write(ResolutionLines, TemplateOutputWidth, TemplateOutputHeight, ow, oh);
    // Last: two of its places are also in the Resolution Patch, and the decoupled ones win.
    // The game sets up part of its rendering once, at boot, from this size: changing it needs
    // a restart (a switch from the main menu left that part at the old size).
    if (rw != ow || rh != oh)
        lines += Write(DecoupledLines, TemplateRenderWidth, TemplateRenderHeight, rw, rh);
    LOG_INFO(Loader, "Resolution: {}x{} render, {}x{} output ({} patch runs)", rw, rh, ow, oh,
             lines);
}

const Choice& Applied() {
    return applied;
}

void RestartGame() {
    restarting = true;
#ifdef _WIN32
    // Gone from the screen straight away: the old process still has to wind down.
    if (g_window)
        if (auto* hwnd = static_cast<HWND>(
                SDL_GetPointerProperty(SDL_GetWindowProperties(g_window->GetSDLWindow()),
                                       SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr)))
            ShowWindowAsync(hwnd, SW_HIDE);
#endif
    // Its own thread: the relaunch waits for the GPU thread to release DLSS, which may in turn
    // wait for the thread that draws F1.
    std::thread([] {
        LOG_INFO(Loader, "Resolution: restarting the game to apply it");
        auto* mnt = Common::Singleton<FileSys::MntPoints>::Instance();
        std::filesystem::path eboot = mnt->GetHostPath("/app0/eboot.bin");
        if (auto file = mnt->Open("/app0/eboot.bin", false))
            if (auto host = file->GetHostPath())
                eboot = *host;
        // Standalone under BBLauncher: its restart (IPC RESTART) marks the game closed before
        // relaunching, then reports "No running game to restart".
        Common::Singleton<Core::Emulator>::Instance()->Restart(eboot, {}, true);
    }).detach();
}

} // namespace Core::BbResolution
