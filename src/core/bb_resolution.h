// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common/types.h"

namespace Core::BbResolution {

// Bloodborne's output and render resolution set from F1 instead of the patch file. The bytes are
// those of Kyo's Resolution Patch and Decoupled UI patches, with the sizes written in:
// the output size (the game's own frame, its post-processing and UI) and a lower render size for
// the 3D scene, which the upscaler then brings back to the output size.

struct Choice {
    bool enabled{};                      // F1's "Set the resolution here"
    u32 output_width{}, output_height{}; // the game's frame
    u32 render_width{}, render_height{}; // the 3D scene
    std::string problem;                 // why nothing was applied, if nothing was
    std::string heap{"auto"};            // graphics heap size: auto, full, first, smaller or off
    // The whole frame (post-processing and UI too) at the render size, as a Resolution Patch:
    // faster, a softer UI. Off: Decoupled, the frame at the output size and the scene lower.
    bool regular{};
};

// Above this render height the graphics heap is raised (more VRAM, extra dmem needed).
inline constexpr u32 HeapRenderHeight = 1080;
// The render heights F1 offers below this output height (16:9, usual sizes only, no more than
// 3x below it). Above 1080p they need the bigger graphics heap: more VRAM and extra dmem.
std::vector<u32> RenderHeights(u32 output_height);
// From the dlss.ini settings and a window size: what a launch with them applies.
Choice Resolve(const std::string& enabled, const std::string& output, const std::string& render,
               u32 window_width, u32 window_height);
// Same, from dlss.ini and the window right now (F1's preview of the next launch).
Choice Preview();

// At game load, before the patch file: decides whether F1 sets the resolution.
void Prepare();
// Before the patches apply, each enabled patch's lines (patch-file addresses): a patch that sets
// a size F1 sets, or changes the code around it, is skipped as a whole. Found by address, so
// renamed patches and other patch repositories are covered too.
void CheckPatch(std::string_view name, u64 address, std::span<const u8> bytes);
// While F1 sets the resolution: whether this patch is skipped.
bool ShouldSkipPatch(std::string_view name);
// The enabled patches skipped this launch, for F1.
const std::vector<std::string>& SkippedPatches();
// After the patch file: writes the resolution into the game's code.
void Install();
// What this launch applied (enabled is false when F1 doesn't set the resolution).
const Choice& Applied();
// Relaunches the game straight away (back to the last autosave), so a new resolution applies.
void RestartGame();
// Set by RestartGame: the exit that follows ends the process at once (emulator.cpp).
inline bool restarting = false;

} // namespace Core::BbResolution
