// SPDX-FileCopyrightText: Copyright 2026 Robert Bopko
// SPDX-License-Identifier: GPL-2.0-or-later
// written by Robert Bopko (github.com/zeroznet) with Boba Bott (Claude Opus 5.5)
#pragma once

#include <string_view>
#include "common/types.h"

// SDR to HDR the way Special K retrofits SDR games: Perceptual Boost on the finished frame in the
// present pass, shown through an scRGB swapchain. Settings live in dlss.ini next to the
// upscaler's.
namespace Vulkan::RetroHdr {

enum class State {
    Disabled,
    Active,
    RenoDx,
    NotWindows,
    DisplayUnknown,
    WindowsHdrOff,
    NoScRgbFormat,
};

// Decided once, when the swapchain picks its format.
struct Session {
    State state{State::Disabled};
    float windows_peak_nits{};
    float sdr_white_nits{203.0f};
};

// Swapchain construction: whether the swapchain should be scRGB for RetroHDR.
bool Start(void* native_window, bool scrgb_supported);
const Session& GetSession();
std::string_view Describe(State state);
// A dlss.ini value as a float; `fallback` unless the whole value is a finite number.
float ParseFloat(std::string_view value, float fallback);

constexpr u32 ModeNone = 0;
constexpr u32 ModeOff = 1;
constexpr u32 ModeOn = 2;

struct PassParams {
    u32 mode{ModeNone};
    float brightness{1.0f};
    float sdr_white{1.0f}; // Windows SDR white level / 80 nits
};

// GPU thread, every frame: re-reads dlss.ini when it changed.
PassParams Poll();

constexpr float DefaultBrightness = 2.0f;
constexpr float MinBrightness = 0.30f;

float PeakNits(float override_nits);
// Brightness at which SDR white reaches `peak_nits`.
float MaxBrightness(float peak_nits);
// The boost's output for SDR linear `value` at brightness 1.0, in nits.
float BoostedNits(float value);

} // namespace Vulkan::RetroHdr
