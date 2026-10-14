// SPDX-FileCopyrightText: Copyright 2026 Robert Bopko
// SPDX-License-Identifier: GPL-2.0-or-later
// written by Robert Bopko (github.com/zeroznet) with Boba Bott (Claude Opus 5.5)
#include "video_core/renderer_vulkan/vk_retro_hdr.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_bb_temporal_dlss.h"
#include "video_core/renderer_vulkan/vk_hdr_mod.h"

#ifdef _WIN32
#include <vector>
#include <dxgi1_6.h>
#include <windows.h>
#endif

namespace Vulkan::RetroHdr {
namespace {

constexpr float MinPeakNits = 100.0f;
constexpr float FallbackPeakNits = 1000.0f;
constexpr float FallbackSdrWhiteNits = 203.0f;

// SMPTE ST 2084 as Special K's LinearToPQ and PQToLinear: `max_value` encodes to 1.0.
constexpr float PqM1 = 2610.0f / 16384.0f;
constexpr float PqM2 = 2523.0f / 4096.0f * 128.0f;
constexpr float PqC1 = 3424.0f / 4096.0f;
constexpr float PqC2 = 2413.0f / 4096.0f * 32.0f;
constexpr float PqC3 = 2392.0f / 4096.0f * 32.0f;

// Special K's SK_HDR_PQBoost_v1 (PerceptualBoost0..3 in widgets/hdr.cpp).
constexpr float BoostInputMax = 1.0f;
constexpr float BoostOutputMax = 0.1f;
constexpr float BoostGain = 1.273f;
constexpr float BoostDivisor = 0.5f;

float LinearToPq(float value, float max_value) {
    const float x = std::pow(std::max(value, 0.0f) / max_value, PqM1);
    return std::pow((PqC1 + PqC2 * x) / (1.0f + PqC3 * x), PqM2);
}

float PqToLinear(float code, float max_value) {
    const float x = std::pow(std::max(code, 0.0f), 1.0f / PqM2);
    return std::pow(std::max(x - PqC1, 0.0f) / (PqC2 - PqC3 * x), 1.0f / PqM1) * max_value;
}

struct Settings {
    bool enabled{};
    float brightness{DefaultBrightness};
    float peak_override{};
};

Settings ReadSettings() {
    Settings settings;
    std::ifstream file{BbDlssSettingsPath()};
    for (std::string line; std::getline(file, line);) {
        const auto eq = line.find('=');
        if (eq == std::string::npos)
            continue;
        const auto key = line.substr(0, eq);
        const auto value = line.substr(eq + 1);
        if (key == "retro_hdr")
            settings.enabled = std::atoi(value.c_str()) != 0;
        else if (key == "retro_hdr_brightness")
            settings.brightness = ParseFloat(value, DefaultBrightness);
        else if (key == "retro_hdr_peak")
            settings.peak_override = ParseFloat(value, 0.0f);
    }
    return settings;
}

Session session;
Settings polled;
std::filesystem::file_time_type polled_time{};
std::chrono::steady_clock::time_point next_poll{};

#ifdef _WIN32
struct Display {
    bool hdr_on;
    float peak_nits;
    float sdr_white_nits;
};

float SdrWhiteNits(HMONITOR monitor) {
    MONITORINFOEXW monitor_info{};
    monitor_info.cbSize = sizeof(monitor_info);
    UINT32 path_count{}, mode_count{};
    if (!GetMonitorInfoW(monitor, &monitor_info) ||
        GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &path_count, &mode_count) !=
            ERROR_SUCCESS)
        return FallbackSdrWhiteNits;
    std::vector<DISPLAYCONFIG_PATH_INFO> paths(path_count);
    std::vector<DISPLAYCONFIG_MODE_INFO> modes(mode_count);
    if (QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &path_count, paths.data(), &mode_count,
                           modes.data(), nullptr) != ERROR_SUCCESS)
        return FallbackSdrWhiteNits;
    for (UINT32 i = 0; i < path_count; ++i) {
        const auto& path = paths[i];
        DISPLAYCONFIG_SOURCE_DEVICE_NAME source{};
        source.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
        source.header.size = sizeof(source);
        source.header.adapterId = path.sourceInfo.adapterId;
        source.header.id = path.sourceInfo.id;
        if (DisplayConfigGetDeviceInfo(&source.header) != ERROR_SUCCESS ||
            std::wstring_view{source.viewGdiDeviceName} != std::wstring_view{monitor_info.szDevice})
            continue;
        DISPLAYCONFIG_SDR_WHITE_LEVEL white{};
        white.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL;
        white.header.size = sizeof(white);
        white.header.adapterId = path.targetInfo.adapterId;
        white.header.id = path.targetInfo.id;
        if (DisplayConfigGetDeviceInfo(&white.header) == ERROR_SUCCESS)
            return float(white.SDRWhiteLevel) / 1000.0f * 80.0f;
    }
    return FallbackSdrWhiteNits;
}

// The display that holds the window, which need not be the primary one.
std::optional<Display> QueryDisplay(HWND window) {
    const HMONITOR monitor = MonitorFromWindow(window, MONITOR_DEFAULTTONEAREST);
    IDXGIFactory1* factory{};
    if (!monitor ||
        FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory))))
        return std::nullopt;
    std::optional<Display> display;
    IDXGIAdapter1* adapter{};
    for (UINT a = 0; !display && SUCCEEDED(factory->EnumAdapters1(a, &adapter)); ++a) {
        IDXGIOutput* output{};
        for (UINT o = 0; !display && SUCCEEDED(adapter->EnumOutputs(o, &output)); ++o) {
            IDXGIOutput6* output6{};
            DXGI_OUTPUT_DESC1 desc{};
            if (SUCCEEDED(output->QueryInterface(__uuidof(IDXGIOutput6),
                                                 reinterpret_cast<void**>(&output6))) &&
                SUCCEEDED(output6->GetDesc1(&desc)) && desc.Monitor == monitor)
                display = Display{
                    .hdr_on = desc.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020,
                    .peak_nits = desc.MaxLuminance,
                    .sdr_white_nits = SdrWhiteNits(monitor),
                };
            if (output6)
                output6->Release();
            output->Release();
        }
        adapter->Release();
    }
    factory->Release();
    return display;
}
#endif

State ResolveState([[maybe_unused]] void* native_window, [[maybe_unused]] bool scrgb_supported) {
    if (RenoDxLoaded())
        return State::RenoDx;
#ifdef _WIN32
    std::optional<Display> display;
    if (native_window)
        display = QueryDisplay(static_cast<HWND>(native_window));
    if (!display)
        return State::DisplayUnknown;
    session.windows_peak_nits = display->peak_nits;
    session.sdr_white_nits = display->sdr_white_nits;
    if (!display->hdr_on)
        return State::WindowsHdrOff;
    if (!scrgb_supported)
        return State::NoScRgbFormat;
    return polled.enabled ? State::Active : State::Disabled;
#else
    return State::NotWindows;
#endif
}

} // namespace

bool Start(void* native_window, bool scrgb_supported) {
    std::error_code error;
    polled_time = std::filesystem::last_write_time(BbDlssSettingsPath(), error);
    polled = ReadSettings();
    session.state = ResolveState(native_window, scrgb_supported);
    LOG_INFO(Render_Vulkan, "RetroHDR: {} (Windows peak {:.0f} nits, SDR white {:.0f} nits)",
             Describe(session.state), session.windows_peak_nits, session.sdr_white_nits);
    return session.state == State::Active;
}

float ParseFloat(std::string_view value, float fallback) {
    const char* const end = value.data() + value.size();
    float parsed{};
    const auto [stop, error] = std::from_chars(value.data(), end, parsed);
    if (error != std::errc{} || !std::isfinite(parsed))
        return fallback;
    for (const char* c = stop; c != end; ++c) {
        if (*c != ' ' && *c != '\t' && *c != '\r')
            return fallback;
    }
    return parsed;
}

const Session& GetSession() {
    return session;
}

std::string_view Describe(State state) {
    switch (state) {
    case State::Disabled:
        return "off";
    case State::Active:
        return "active";
    case State::RenoDx:
        return "RenoDX HDR is loaded";
    case State::NotWindows:
        return "needs Windows";
    case State::DisplayUnknown:
        return "Windows did not report this display";
    case State::WindowsHdrOff:
        return "turn on Use HDR in the Windows display settings, then restart the game";
    case State::NoScRgbFormat:
        return "the GPU driver offers no scRGB swapchain";
    }
    return {};
}

PassParams Poll() {
    if (session.state != State::Active)
        return {};
    const auto now = std::chrono::steady_clock::now();
    if (now >= next_poll) {
        next_poll = now + std::chrono::milliseconds{250};
        std::error_code error;
        const auto time = std::filesystem::last_write_time(BbDlssSettingsPath(), error);
        if (!error && time != polled_time) {
            polled_time = time;
            polled = ReadSettings();
        }
    }
    return {
        .mode = polled.enabled ? ModeOn : ModeOff,
        .brightness = std::clamp(polled.brightness, MinBrightness,
                                 MaxBrightness(PeakNits(polled.peak_override))),
        .sdr_white = session.sdr_white_nits / 80.0f,
    };
}

float PeakNits(float override_nits) {
    if (override_nits >= MinPeakNits)
        return override_nits;
    return session.windows_peak_nits >= MinPeakNits ? session.windows_peak_nits : FallbackPeakNits;
}

float MaxBrightness(float peak_nits) {
    return std::max(MinBrightness, peak_nits / BoostedNits(1.0f));
}

float BoostedNits(float value) {
    return PqToLinear(LinearToPq(value, BoostInputMax) * BoostGain, BoostOutputMax) / BoostDivisor *
           80.0f;
}

} // namespace Vulkan::RetroHdr
