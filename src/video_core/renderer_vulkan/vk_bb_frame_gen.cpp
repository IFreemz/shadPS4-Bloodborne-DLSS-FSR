// SPDX-FileCopyrightText: Copyright 2026 IFreemz
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include <fmt/format.h>

#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_bb_frame_gen.h"
#include "video_core/renderer_vulkan/vk_bb_fsr_frame_gen.h"
#include "video_core/renderer_vulkan/vk_bb_temporal_dlss.h"
#include "video_core/renderer_vulkan/vk_hdr_mod.h"

#if defined(SHADPS4_BB_FRAME_GEN) && defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <dxgi1_6.h>
#include <windows.h>
// Streamline's headers expect the Vulkan C types.
#include <sl.h>
#include <sl_consts.h>
#include <sl_dlss_g.h>
#include <sl_security.h>
#include <vulkan/vulkan.h>

// sl_pcl.h and sl_reflex.h do not compile as C++23 (`using to_underlying = std::to_underlying`);
// the few declarations used here, as in those headers.
namespace sl {
enum class PCLMarker : uint32_t {
    eSimulationStart = 0,
    eSimulationEnd = 1,
    eRenderSubmitStart = 2,
    eRenderSubmitEnd = 3,
    ePresentStart = 4,
    ePresentEnd = 5,
};
enum ReflexMode { eOff, eLowLatency, eLowLatencyWithBoost, ReflexMode_eCount };
// {F03AF81A-6D0B-4902-A651-C4965E215434}
SL_STRUCT_BEGIN(
    ReflexOptions,
    StructType({0xf03af81a, 0x6d0b, 0x4902, {0xa6, 0x51, 0xc4, 0x96, 0x5e, 0x21, 0x54, 0x34}}),
    kStructVersion1)
ReflexMode mode = ReflexMode::eOff;
uint32_t frameLimitUs = 0;
bool useMarkersToOptimize = false;
uint16_t virtualKey = 0;
uint32_t idThread = 0;
SL_STRUCT_END()
} // namespace sl
using PFun_slPCLSetMarker = sl::Result(sl::PCLMarker marker, const sl::FrameToken& frame);
using PFun_slReflexSleep = sl::Result(const sl::FrameToken& frame);
using PFun_slReflexSetOptions = sl::Result(const sl::ReflexOptions& options);
#define BB_FRAME_GEN 1
#endif

namespace Vulkan::FrameGen {

namespace {
std::atomic<bool> counter_visible{};
std::string problem;
// Per-second counts of emulator presents and of frames shown for them.
std::chrono::steady_clock::time_point window_start{};
u32 window_presents{}, window_shown{}, windows{};
std::atomic<float> base_fps{}, output_fps{};
std::atomic<bool> generating{};
// Frames per rendered frame the user asked for, and the most this card and runtime allow.
std::atomic<u32> multiplier{2}, max_multiplier{2};

void CountPresent(u32 shown) {
    const auto now = std::chrono::steady_clock::now();
    ++window_presents;
    window_shown += shown;
    const float seconds = std::chrono::duration<float>(now - window_start).count();
    if (seconds >= 1.0f) {
        base_fps = float(window_presents) / seconds;
        output_fps = float(window_shown) / seconds;
        if (generating && ++windows % 5 == 0)
            LOG_INFO(Render_Vulkan, "[FRAME-GEN] base {:.1f} fps, shown {:.1f} fps",
                     base_fps.load(), output_fps.load());
        window_presents = window_shown = 0;
        window_start = now;
    }
}
} // namespace

void SetCounterVisible(bool visible) {
    counter_visible = visible;
}

bool CounterVisible() {
    return counter_visible;
}

Stats GetStats() {
    return {generating, base_fps, output_fps};
}

void SetMultiplier(u32 value) {
    value = std::clamp<u32>(value, 2, 6);
    if (multiplier.exchange(value) != value)
        RecordTiming("multiplier", 0, 0, value);
    FsrFrameGen::SetMultiplier(value);
}

u32 MaxMultiplier() {
    return FsrFrameGen::Active() ? FsrFrameGen::MaxMultiplier() : max_multiplier.load();
}

std::string Problem() {
    return FsrFrameGen::Problem().empty() ? problem : FsrFrameGen::Problem();
}

const char* BackendName() {
    if (FsrFrameGen::Active())
        return FsrFrameGen::UsingDlss() ? "DLSS" : "FSR";
    return Active() ? "DLSS" : "";
}

namespace {
// A dlss.ini value as saved at launch.
std::optional<std::string> LaunchSetting(std::string_view key) {
    std::ifstream file{BbDlssSettingsPath()};
    std::string line;
    while (std::getline(file, line)) {
        const auto start = line.find_first_not_of(" \t");
        if (start == std::string::npos)
            continue;
        line.erase(0, start);
        const auto equals = line.find('=');
        if (equals == std::string::npos || !line.starts_with(key) ||
            line.find_first_not_of(" \t", key.size()) != equals)
            continue;
        auto value = line.substr(equals + 1);
        std::erase_if(value, [](char c) { return c == ' ' || c == '\t' || c == '\r'; });
        return value;
    }
    return {};
}

// dlss.ini fg_backend: auto (DLSS-G when the card supports it, else FSR), dlss or fsr.
std::string Backend() {
    static const std::string backend = LaunchSetting("fg_backend").value_or("");
    return backend.empty() ? "auto" : backend;
}
} // namespace

bool Requested() {
    static const bool requested =
        LaunchSetting("frame_gen").value_or("").find('1') != std::string::npos;
    return requested;
}

bool TimingEnabled() {
    static const bool enabled = [] {
        if (!Requested())
            return false;
        if (const auto setting = LaunchSetting("fg_timing_log"))
            return *setting == "1";
#ifdef SHADPS4_FG_TIMING_DEFAULT_ON
        const auto path = BbDlssSettingsPath();
        std::ofstream file{path, std::ios::app};
        file << "\nfg_timing_log=1\n";
        file.close();
        if (!file) {
            LOG_WARNING(Render_Vulkan, "[FRAME-GEN] Could not enable timing trace in {}",
                        path.string());
            return false;
        }
        return true;
#else
        return false;
#endif
    }();
    return enabled;
}

void RecordTiming(std::string_view event, u32 frame, u64 duration_us, u64 value) {
    if (!TimingEnabled())
        return;
    static std::mutex mutex;
    std::scoped_lock lock{mutex};
    static std::ofstream file = [] {
        const auto stamp = std::chrono::system_clock::now().time_since_epoch().count();
        const auto directory = BbDlssSettingsPath().parent_path() / "log";
        std::error_code error;
        std::filesystem::create_directories(directory, error);
        const auto path = directory / fmt::format("frame_gen_timing_{}.csv", stamp);
        std::ofstream stream{path};
        if (!stream) {
            LOG_WARNING(Render_Vulkan, "[FRAME-GEN] Could not open timing trace {}", path.string());
            return stream;
        }
        stream << "time_us,event,frame,duration_us,value\n";
        LOG_INFO(Render_Vulkan, "[FRAME-GEN] Timing trace {}", path.string());
        return stream;
    }();
    if (!file)
        return;
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    file << std::chrono::duration_cast<std::chrono::microseconds>(now).count() << ',' << event
         << ',' << frame << ',' << duration_us << ',' << value << '\n';
    static u32 records{};
    if (++records % 300 == 0)
        file.flush();
}

TimingScope::TimingScope(std::string_view event, u32 frame, u64 value)
    : event{event}, frame{frame}, value{value} {
    if (TimingEnabled())
        start = std::chrono::steady_clock::now();
}

TimingScope::~TimingScope() {
    if (start != std::chrono::steady_clock::time_point{}) {
        const auto elapsed = std::chrono::steady_clock::now() - start;
        RecordTiming(event, frame,
                     std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count(), value);
    }
}

void ReplayPresentStall(bool inside_submit_lock) {
#ifdef SHADPS4_FG_STALL_REPLAY
    if (!TimingEnabled() || !Active() || FsrFrameGen::Active() || !generating)
        return;
    static auto last_read = std::chrono::steady_clock::time_point{};
    static u32 stall_us{};
    static bool outside_lock{};
    const auto now = std::chrono::steady_clock::now();
    if (now - last_read >= std::chrono::seconds{1}) {
        last_read = now;
        u32 next{};
        if (const auto setting = LaunchSetting("fg_test_present_stall_us")) {
            const auto [end, error] =
                std::from_chars(setting->data(), setting->data() + setting->size(), next);
            if (error != std::errc{} || end != setting->data() + setting->size())
                next = 0;
        }
        next = std::min(next, 70000u);
        const bool next_outside =
            LaunchSetting("fg_test_present_stall_outside_lock").value_or("") == "1";
        if (next != stall_us || next_outside != outside_lock) {
            stall_us = next;
            outside_lock = next_outside;
            RecordTiming("stall_replay_options", 0, 0, stall_us);
            RecordTiming("stall_replay_outside_lock", 0, 0, outside_lock);
            LOG_WARNING(Render_Vulkan,
                        "[FRAME-GEN] Local test: injected present stall {} us, outside lock {}",
                        stall_us, outside_lock);
        }
    }
    if (stall_us && inside_submit_lock != outside_lock) {
        TimingScope timing{"injected_present_stall", 0, stall_us};
        std::this_thread::sleep_for(std::chrono::microseconds{stall_us});
    }
#else
    (void)inside_submit_lock;
#endif
}

#ifdef BB_FRAME_GEN

namespace {
std::filesystem::path ExecutableDirectory() {
    std::wstring path(MAX_PATH, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, path.data(), DWORD(path.size()));
    path.resize(length);
    return std::filesystem::path{path}.parent_path();
}

struct Api {
    HMODULE module{};
    PFun_slInit* init{};
    PFun_slShutdown* shutdown{};
    PFun_slGetNewFrameToken* new_frame{};
    PFun_slSetConstants* set_constants{};
    PFun_slSetTagForFrame* set_tags{};
    PFun_slGetFeatureFunction* feature_function{};
    PFun_slIsFeatureSupported* feature_supported{};
    PFun_slDLSSGSetOptions* dlssg_options{};
    PFun_slDLSSGGetState* dlssg_state{};
    PFun_slReflexSetOptions* reflex_options{};
    PFun_slReflexSleep* reflex_sleep{};
    PFun_slPCLSetMarker* marker{};
    bool feature_functions{};
};
Api api;
bool active{};
std::atomic<bool> enabled{true};
std::atomic<u32> paused_frames{};
// Reflex's sleep (F1 "Lower input delay (Reflex)"): the game's thread waits before each frame and
// marks it for Reflex (GameSubmitEnd). Low-latency mode itself stays on: DLSS-G needs it.
std::atomic<bool> reflex_low_latency{true}, reflex_sleep{false};
std::chrono::steady_clock::time_point last_generated{};
std::atomic<bool> reflex_applied{};
bool reflex_applied_mode{};
bool presented_from_game{}; // presenter thread: this frame's markers came from the game

// The game's thread opens frames by explicit index: the frame in
// progress there, and the last one it submitted, which the presenter presents.
std::atomic<bool> game_frames_ready{};
std::atomic<u32> game_frame{}, presented_game_frame{};
// Every frame gets the next number, whichever thread opens it, so numbers only go up when the
// sleep is switched while playing.
std::atomic<u32> next_frame_index{1};

sl::FrameToken* TokenFor(u32 index) {
    sl::FrameToken* frame_token{};
    if (api.new_frame(frame_token, &index) != sl::Result::eOk)
        return nullptr;
    return frame_token;
}

void MarkerFor(sl::PCLMarker marker, sl::FrameToken* frame_token) {
    if (frame_token && api.marker)
        api.marker(marker, *frame_token);
}

// Presenter thread state.
sl::FrameToken* token{};
bool frame_open{};
sl::DLSSGMode last_mode{sl::DLSSGMode::eOff};
u64 frames{};
u32 status_logged{};

const sl::ViewportHandle Viewport{0u};

void LogMessage(sl::LogType type, const char* message) {
    std::string text{message};
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r'))
        text.pop_back();
    if (type == sl::LogType::eError)
        LOG_ERROR(Render_Vulkan, "[SL] {}", text);
    else if (type == sl::LogType::eWarn)
        LOG_WARNING(Render_Vulkan, "[SL] {}", text);
    else
        LOG_INFO(Render_Vulkan, "[SL] {}", text);
}

template <typename T>
bool Export(T*& function, const char* name) {
    function = reinterpret_cast<T*>(GetProcAddress(api.module, name));
    return function != nullptr;
}

// The DLSS-G, Reflex and PCL entry points exist once Streamline has a device, which it gets
// when the emulator creates its device through Streamline's vkCreateDevice.
bool FeatureFunctions() {
    if (api.feature_functions)
        return true;
    const auto get = [](sl::Feature feature, const char* name, auto*& function) {
        void* address{};
        if (api.feature_function(feature, name, address) != sl::Result::eOk || !address)
            return false;
        function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(address);
        return true;
    };
    if (!get(sl::kFeatureDLSS_G, "slDLSSGSetOptions", api.dlssg_options) ||
        !get(sl::kFeatureDLSS_G, "slDLSSGGetState", api.dlssg_state) ||
        !get(sl::kFeatureReflex, "slReflexSetOptions", api.reflex_options) ||
        !get(sl::kFeatureReflex, "slReflexSleep", api.reflex_sleep) ||
        !get(sl::kFeaturePCL, "slPCLSetMarker", api.marker)) {
        LOG_ERROR(Render_Vulkan, "[FRAME-GEN] Streamline feature functions are missing");
        active = false;
        return false;
    }
    api.feature_functions = true;
    return true;
}

void Marker(sl::PCLMarker marker) {
    if (frame_open && api.marker)
        api.marker(marker, *token);
}

sl::float4x4 Matrix(const std::array<float, 16>& m) {
    sl::float4x4 result{};
    for (u32 r = 0; r < 4; ++r)
        result.setRow(r, sl::float4(m[r * 4], m[r * 4 + 1], m[r * 4 + 2], m[r * 4 + 3]));
    return result;
}

sl::Resource ImageResource(vk::Image image, vk::ImageView view, vk::Format format,
                           vk::Extent2D extent) {
    sl::Resource resource{sl::ResourceType::eTex2d, static_cast<VkImage>(image), nullptr,
                          static_cast<VkImageView>(view),
                          static_cast<u32>(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)};
    resource.width = extent.width;
    resource.height = extent.height;
    resource.nativeFormat = static_cast<u32>(format);
    resource.mipLevels = 1;
    resource.arrayLayers = 1;
    resource.flags = 0;
    resource.usage = static_cast<u32>(VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    return resource;
}
} // namespace

// The first NVIDIA GPU's LUID, which Streamline checks DLSS-G support for.
std::optional<LUID> NvidiaAdapter() {
    IDXGIFactory1* factory{};
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory))))
        return std::nullopt;
    std::optional<LUID> luid;
    IDXGIAdapter1* adapter{};
    for (UINT i = 0; !luid && factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc{};
        if (SUCCEEDED(adapter->GetDesc1(&desc)) && desc.VendorId == 0x10DE &&
            !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE))
            luid = desc.AdapterLuid;
        adapter->Release();
    }
    factory->Release();
    return luid;
}

namespace {
// ---- The frame generation model, in the driver's profile for shadPS4.exe ----
// nvngx_dlssg.dll reads its model from the driver profile setting "Override DLSS-FG preset" when
// it starts, the setting the NVIDIA app's model override writes (NVIDIA's NvApiDriverSettings.h:
// 0 the default, 1 A, 2 B, 3 C; C from DLSS-G 310.9.16, older ones ignore it). Written through
// NVAPI's public driver settings calls before Streamline loads, so it applies at this start.
namespace Nv {
using Status = s32;
using Handle = void*;
constexpr Status Ok = 0;
constexpr Status SettingNotFound = -160;
constexpr Status ExecutableNotFound = -166;
constexpr u32 FgPresetSetting = 0x10E41DF1;
constexpr u32 StringMax = 2048;
using UnicodeString = u16[StringMax];

#pragma pack(push, 8)
struct Profile {
    u32 version;
    UnicodeString name;
    u32 gpu_support;
    u32 is_predefined;
    u32 apps;
    u32 settings;
};
struct Application {
    u32 version;
    u32 is_predefined;
    UnicodeString app_name;
    UnicodeString friendly_name;
    UnicodeString launcher;
    UnicodeString file_in_folder;
    u32 flags;
    UnicodeString command_line;
};
#pragma pack(pop)
#pragma pack(push, 4)
struct Setting {
    u32 version;
    UnicodeString name;
    u32 id;
    u32 type; // 0: a DWORD
    u32 location;
    u32 is_current_predefined;
    u32 is_predefined_valid;
    union {
        u32 u32_value;
        u8 binary[4 + 4096];
        UnicodeString text;
        u64 u64_value;
    } predefined;
    union {
        u32 u32_value;
        u8 binary[4 + 4096];
        UnicodeString text;
        u64 u64_value;
    } current;
};
#pragma pack(pop)
static_assert(sizeof(Profile) == 4116 && sizeof(Application) == 20492 && sizeof(Setting) == 12320);
constexpr u32 Version(u32 size, u32 version) {
    return size | (version << 16);
}

void Copy(UnicodeString& to, const std::wstring& from) {
    const size_t n = std::min<size_t>(from.size(), StringMax - 1);
    for (size_t i = 0; i < n; ++i)
        to[i] = u16(from[i]);
    to[n] = 0;
}
} // namespace Nv

/// dlss.ini fg_model: default (the driver's choice), a, b or c.
u32 FgModelPreset() {
    const auto model = LaunchSetting("fg_model").value_or("default");
    return model == "a" ? 1 : model == "b" ? 2 : model == "c" ? 3 : 0;
}

void ApplyFgModel() {
    const u32 preset = FgModelPreset();
    HMODULE nvapi = LoadLibraryW(L"nvapi64.dll");
    if (!nvapi) {
        if (preset)
            LOG_WARNING(Render_Vulkan,
                        "[FRAME-GEN] Model: nvapi64.dll not found, the driver's model");
        return;
    }
    using QueryInterface = void* (*)(u32);
    const auto query =
        reinterpret_cast<QueryInterface>(GetProcAddress(nvapi, "nvapi_QueryInterface"));
    const auto get = [&]<typename F>(F*& out, u32 id) {
        out = query ? reinterpret_cast<F*>(query(id)) : nullptr;
        return out != nullptr;
    };
    using Initialize = Nv::Status();
    using CreateSession = Nv::Status(Nv::Handle*);
    using SessionCall = Nv::Status(Nv::Handle);
    using FindApplication = Nv::Status(Nv::Handle, const u16*, Nv::Handle*, Nv::Application*);
    using FindProfile = Nv::Status(Nv::Handle, const u16*, Nv::Handle*);
    using CreateProfile = Nv::Status(Nv::Handle, Nv::Profile*, Nv::Handle*);
    using CreateApplication = Nv::Status(Nv::Handle, Nv::Handle, Nv::Application*);
    using GetSetting = Nv::Status(Nv::Handle, Nv::Handle, u32, Nv::Setting*);
    using SetSetting = Nv::Status(Nv::Handle, Nv::Handle, Nv::Setting*);
    using DeleteSetting = Nv::Status(Nv::Handle, Nv::Handle, u32);
    Initialize* initialize{};
    CreateSession* create_session{};
    SessionCall *destroy_session{}, *load_settings{}, *save_settings{};
    FindApplication* find_application{};
    FindProfile* find_profile{};
    CreateProfile* create_profile{};
    CreateApplication* create_application{};
    GetSetting* get_setting{};
    SetSetting* set_setting{};
    DeleteSetting* delete_setting{};
    // IDs from NVIDIA's nvapi_interface.h.
    if (!get(initialize, 0x0150E828) || !get(create_session, 0x0694D52E) ||
        !get(destroy_session, 0xDAD9CFF8) || !get(load_settings, 0x375DBD6B) ||
        !get(save_settings, 0xFCBC7E14) || !get(find_application, 0xEEE566B2) ||
        !get(find_profile, 0x7E4A9A0B) || !get(create_profile, 0xCC176068) ||
        !get(create_application, 0x4347A9DE) || !get(get_setting, 0x73BF8338) ||
        !get(set_setting, 0x577DD202) || !get(delete_setting, 0xE4A26362) ||
        initialize() != Nv::Ok) {
        LOG_WARNING(Render_Vulkan, "[FRAME-GEN] Model: NVAPI's driver settings are not available");
        return;
    }
    Nv::Handle session{};
    if (create_session(&session) != Nv::Ok)
        return;
    std::wstring module(MAX_PATH, L'\0');
    module.resize(GetModuleFileNameW(nullptr, module.data(), DWORD(module.size())));
    const std::wstring exe = std::filesystem::path{module}.filename().wstring();
    static constexpr const char* Names[] = {"the driver's", "A", "B", "C"};
    const auto done = [&](Nv::Status status, const char* what) {
        if (status == Nv::Ok)
            LOG_INFO(Render_Vulkan, "[FRAME-GEN] Model: {} ({})", Names[preset], what);
        else
            LOG_WARNING(Render_Vulkan, "[FRAME-GEN] Model: could not set {} ({}: NVAPI {})",
                        Names[preset], what, status);
        destroy_session(session);
    };
    Nv::Status status = load_settings(session);
    if (status != Nv::Ok)
        return done(status, "loading the driver settings");
    Nv::UnicodeString name{};
    Nv::Copy(name, exe);
    auto application = std::make_unique<Nv::Application>();
    application->version = Nv::Version(sizeof(Nv::Application), 4);
    Nv::Handle profile{};
    status = find_application(session, name, &profile, application.get());
    if (status == Nv::ExecutableNotFound) {
        if (!preset)
            return done(Nv::Ok, "no profile for shadPS4");
        // A profile of our own for shadPS4.exe (found again by name if it exists already).
        auto created = std::make_unique<Nv::Profile>();
        created->version = Nv::Version(sizeof(Nv::Profile), 1);
        Nv::Copy(created->name, L"shadPS4 Bloodborne");
        status = find_profile(session, created->name, &profile);
        if (status != Nv::Ok)
            status = create_profile(session, created.get(), &profile);
        if (status != Nv::Ok)
            return done(status, "creating the profile");
        *application = {};
        application->version = Nv::Version(sizeof(Nv::Application), 4);
        Nv::Copy(application->app_name, exe);
        status = create_application(session, profile, application.get());
        if (status != Nv::Ok)
            return done(status, "adding shadPS4 to the profile");
    } else if (status != Nv::Ok) {
        return done(status, "finding shadPS4's profile");
    }
    auto setting = std::make_unique<Nv::Setting>();
    setting->version = Nv::Version(sizeof(Nv::Setting), 1);
    status = get_setting(session, profile, Nv::FgPresetSetting, setting.get());
    const bool present = status == Nv::Ok;
    if (!preset) {
        if (!present)
            return done(Nv::Ok, "not set in shadPS4's profile");
        status = delete_setting(session, profile, Nv::FgPresetSetting);
    } else {
        if (present && setting->current.u32_value == preset)
            return done(Nv::Ok, "already set");
        *setting = {};
        setting->version = Nv::Version(sizeof(Nv::Setting), 1);
        setting->id = Nv::FgPresetSetting;
        setting->type = 0;
        setting->current.u32_value = preset;
        status = set_setting(session, profile, setting.get());
    }
    if (status == Nv::Ok)
        status = save_settings(session);
    done(status, "set in the driver profile for shadPS4");
}
} // namespace

PFN_vkGetInstanceProcAddr Load() {
    if (!Requested())
        return nullptr;
    if (Backend() == "fsr") {
        FsrFrameGen::Select();
        return nullptr;
    }
    // RenoDX makes the swapchain scRGB, where DLSS-G does not generate frames.
    if (RenoDxInstalled()) {
        LOG_INFO(Render_Vulkan, "[FRAME-GEN] RenoDX found: FSR frame generation");
        FsrFrameGen::Select();
        return nullptr;
    }
    // Without DLSS-G, FSR frame generation unless dlss.ini asks for DLSS-G only.
    const auto fail = [](std::string reason) -> PFN_vkGetInstanceProcAddr {
        LOG_WARNING(Render_Vulkan, "[FRAME-GEN] DLSS-G off: {}", reason);
        if (Backend() == "dlss")
            problem = std::move(reason);
        else
            FsrFrameGen::Select();
        return nullptr;
    };
    auto luid = NvidiaAdapter();
    if (!luid)
        return fail("DLSS Frame generation needs an NVIDIA RTX 40 or 50 series GPU.");
    const auto interposer = ExecutableDirectory() / "sl.interposer.dll";
    if (!std::filesystem::is_regular_file(interposer))
        return fail("The frame generation files (sl.*.dll) are missing next to shadPS4.exe.");
    if (!sl::security::verifyEmbeddedSignature(interposer.c_str()))
        return fail("sl.interposer.dll is not signed by NVIDIA.");
    ApplyFgModel();
    api.module = LoadLibraryW(interposer.c_str());
    if (!api.module) {
        LOG_ERROR(Render_Vulkan, "[FRAME-GEN] sl.interposer.dll could not be loaded");
        return nullptr;
    }
    PFN_vkGetInstanceProcAddr get_instance_proc_addr{};
    get_instance_proc_addr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        GetProcAddress(api.module, "vkGetInstanceProcAddr"));
    if (!Export(api.init, "slInit") || !Export(api.shutdown, "slShutdown") ||
        !Export(api.new_frame, "slGetNewFrameToken") ||
        !Export(api.set_constants, "slSetConstants") || !Export(api.set_tags, "slSetTagForFrame") ||
        !Export(api.feature_function, "slGetFeatureFunction") ||
        !Export(api.feature_supported, "slIsFeatureSupported") || !get_instance_proc_addr) {
        LOG_ERROR(Render_Vulkan, "[FRAME-GEN] sl.interposer.dll is missing entry points");
        FreeLibrary(api.module);
        api = {};
        return nullptr;
    }

    static const std::wstring plugin_path = ExecutableDirectory().wstring();
    static const std::wstring log_path = (BbDlssSettingsPath().parent_path() / "log").wstring();
    static const wchar_t* plugin_paths[] = {plugin_path.c_str()};
    static const sl::Feature features[] = {sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL};
    sl::Preferences preferences{};
    preferences.showConsole = false;
    preferences.logLevel = sl::LogLevel::eDefault;
    preferences.pathsToPlugins = plugin_paths;
    preferences.numPathsToPlugins = 1;
    preferences.pathToLogsAndData = log_path.c_str();
    preferences.logMessageCallback = LogMessage;
    // No over-the-air downloads; command buffer state is restored by the emulator.
    preferences.flags = sl::PreferenceFlags::eDisableCLStateTracking |
                        sl::PreferenceFlags::eUseManualHooking |
                        sl::PreferenceFlags::eUseFrameBasedResourceTagging;
    preferences.featuresToLoad = features;
    preferences.numFeaturesToLoad = u32(std::size(features));
    preferences.engine = sl::EngineType::eCustom;
    preferences.engineVersion = "shadPS4";
    preferences.projectId = "6a1d7c55-3b8e-4f0a-9d62-bb0f5e1c7a21";
    preferences.renderAPI = sl::RenderAPI::eVulkan;
    const auto result = api.init(preferences, sl::kSDKVersion);
    if (result != sl::Result::eOk) {
        FreeLibrary(api.module);
        api = {};
        return fail(fmt::format("NVIDIA Streamline did not start (error {}).", int(result)));
    }
    sl::AdapterInfo adapter{};
    adapter.deviceLUID = reinterpret_cast<uint8_t*>(&*luid);
    adapter.deviceLUIDSizeInBytes = sizeof(LUID);
    const auto supported = api.feature_supported(sl::kFeatureDLSS_G, adapter);
    if (supported != sl::Result::eOk) {
        api.shutdown();
        FreeLibrary(api.module);
        api = {};
        switch (supported) {
        case sl::Result::eErrorDriverOutOfDate:
            return fail("DLSS Frame generation needs a newer NVIDIA driver.");
        case sl::Result::eErrorOSOutOfDate:
        case sl::Result::eErrorOSDisabledHWS:
            return fail("DLSS Frame generation needs Windows 10 20H1 or newer with "
                        "Hardware-accelerated GPU scheduling turned on.");
        default:
            return fail("DLSS Frame generation needs an NVIDIA RTX 40 or 50 series GPU.");
        }
    }
    active = true;
    LOG_INFO(Render_Vulkan, "[FRAME-GEN] Streamline loaded");
    return get_instance_proc_addr;
}

bool Active() {
    return active || FsrFrameGen::Active();
}

void Shutdown() {
    FsrFrameGen::Shutdown();
    if (!active)
        return;
    active = false;
    api.shutdown();
}

void BeginFrame() {
    if (!active || !FeatureFunctions())
        return;
    const u32 submitted = presented_game_frame.exchange(0);
    if (submitted) {
        token = TokenFor(submitted);
        if (!token)
            return;
    } else if (game_frames_ready && game_frame.load() != 0) {
        // A repeated or blank frame while the game numbers its frames: no frame of its own.
        return;
    } else if (token = TokenFor(next_frame_index++); !token) {
        return;
    }
    frame_open = true;
    const bool low_latency = reflex_low_latency.load();
    if (!reflex_applied.exchange(true) || reflex_applied_mode != low_latency) {
        sl::ReflexOptions reflex{};
        reflex.mode = low_latency ? sl::ReflexMode::eLowLatency : sl::ReflexMode::eOff;
        if (api.reflex_options(reflex) != sl::Result::eOk) {
            reflex_applied = false;
            LOG_WARNING(Render_Vulkan, "[FRAME-GEN] Reflex options were not accepted");
        }
        reflex_applied_mode = low_latency;
        LOG_INFO(Render_Vulkan, "[FRAME-GEN] Reflex low latency {}, sleep {}", low_latency,
                 reflex_sleep.load());
    }
    game_frames_ready = true;
    presented_from_game = submitted != 0;
    RecordTiming("present_frame", submitted, 0, reflex_sleep.load());
    if (submitted)
        return;
    Marker(sl::PCLMarker::eSimulationStart);
    Marker(sl::PCLMarker::eSimulationEnd);
    Marker(sl::PCLMarker::eRenderSubmitStart);
}

void GameSubmitBegin() {
    if (!active || !game_frames_ready)
        return;
    if (const u32 frame = game_frame.load()) {
        auto* frame_token = TokenFor(frame);
        MarkerFor(sl::PCLMarker::eSimulationEnd, frame_token);
        MarkerFor(sl::PCLMarker::eRenderSubmitStart, frame_token);
    }
}

void GameSubmitEnd() {
    if (!active || !game_frames_ready) {
        game_frame = 0;
        return;
    }
    if (const u32 frame = game_frame.load())
        MarkerFor(sl::PCLMarker::eRenderSubmitEnd, TokenFor(frame));
    // Keep frame tracking unchanged when the user switches the sleep live.
    const u32 next = next_frame_index++;
    auto* frame_token = TokenFor(next);
    if (frame_token) {
        if (reflex_sleep) {
            TimingScope timing{"reflex_sleep", next};
            const auto result = api.reflex_sleep(*frame_token);
            if (result != sl::Result::eOk)
                LOG_WARNING(Render_Vulkan, "[FRAME-GEN] Reflex sleep failed ({})", int(result));
        }
        MarkerFor(sl::PCLMarker::eSimulationStart, frame_token);
    }
    game_frame = next;
}

u32 CurrentGameFrame() {
    return game_frame.load();
}

void SetPresentedGameFrame(u32 frame) {
    presented_game_frame = frame;
}

void SetFrame(const FrameInputs* inputs, const OverlayInputs* overlays, vk::CommandBuffer command,
              vk::Extent2D backbuffer, vk::Rect2D game_area) {
    if (FsrFrameGen::Active()) {
        if (!enabled || paused_frames > 0)
            inputs = nullptr;
        if (paused_frames > 0)
            --paused_frames;
        FsrFrameGen::SetFrame(inputs, overlays, command, backbuffer, game_area);
        generating = inputs != nullptr;
        return;
    }
    if (!frame_open)
        return;
    const auto cmd = static_cast<VkCommandBuffer>(command);
    sl::DLSSGOptions options{};
    if (!enabled)
        inputs = nullptr;
    if (paused_frames > 0) {
        --paused_frames;
        inputs = nullptr;
    }
    options.mode = inputs ? sl::DLSSGMode::eOn : sl::DLSSGMode::eOff;
    // Up to what Streamline reports for this card: 1 below RTX 50 series.
    options.numFramesToGenerate = std::min(multiplier.load(), max_multiplier.load()) - 1;
    options.flags = sl::DLSSGFlags::eRetainResourcesWhenOff;
    // All inputs are volatile: SL copies them in this command buffer before asynchronous FG.
    options.queueParallelismMode = sl::DLSSGQueueParallelismMode::eBlockNoClientQueues;
    if (inputs) {
        // The formats and sizes of everything DLSS-G gets.
        options.mvecDepthWidth = inputs->render.width;
        options.mvecDepthHeight = inputs->render.height;
        options.colorWidth = backbuffer.width;
        options.colorHeight = backbuffer.height;
        options.mvecBufferFormat = u32(vk::Format::eR16G16Sfloat);
        options.depthBufferFormat = u32(vk::Format::eR32Sfloat);
        if (overlays) {
            options.colorBufferFormat = u32(overlays->hudless_format);
            options.hudLessBufferFormat = u32(overlays->hudless_format);
            options.uiBufferFormat = u32(vk::Format::eR16Sfloat);
        }
        sl::Constants constants{};
        constants.cameraViewToClip = Matrix(inputs->view_to_clip);
        constants.clipToCameraView = Matrix(inputs->clip_to_view);
        constants.clipToLensClip = Matrix({1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1});
        constants.clipToPrevClip = Matrix(inputs->clip_to_prev_clip);
        constants.prevClipToClip = Matrix(inputs->prev_clip_to_clip);
        constants.jitterOffset = {inputs->jitter[0], inputs->jitter[1]};
        // Motion is in render pixels.
        constants.mvecScale = {1.0f / float(inputs->render.width),
                               1.0f / float(inputs->render.height)};
        constants.cameraPinholeOffset = {0.0f, 0.0f};
        constants.cameraPos = {inputs->position[0], inputs->position[1], inputs->position[2]};
        constants.cameraUp = {inputs->up[0], inputs->up[1], inputs->up[2]};
        constants.cameraRight = {inputs->right[0], inputs->right[1], inputs->right[2]};
        constants.cameraFwd = {inputs->forward[0], inputs->forward[1], inputs->forward[2]};
        constants.cameraNear = inputs->near_plane;
        constants.cameraFar = inputs->far_plane;
        constants.cameraFOV = inputs->fov_y;
        constants.cameraAspectRatio = inputs->aspect;
        constants.depthInverted = inputs->depth_inverted ? sl::Boolean::eTrue : sl::Boolean::eFalse;
        constants.cameraMotionIncluded = sl::Boolean::eTrue;
        constants.motionVectors3D = sl::Boolean::eFalse;
        constants.reset = inputs->reset ? sl::Boolean::eTrue : sl::Boolean::eFalse;
        constants.orthographicProjection = sl::Boolean::eFalse;
        constants.motionVectorsDilated = sl::Boolean::eFalse;
        constants.motionVectorsJittered = sl::Boolean::eFalse;
        api.set_constants(constants, *token, Viewport);

        auto depth = ImageResource(inputs->depth, inputs->depth_view, vk::Format::eR32Sfloat,
                                   inputs->render);
        auto motion = ImageResource(inputs->motion, inputs->motion_view, vk::Format::eR16G16Sfloat,
                                    inputs->render);
        const sl::Extent render_extent{0, 0, inputs->render.width, inputs->render.height};
        // Where the game image sits in the window (letterboxing, overlays aside).
        const sl::Extent game_extent{u32(game_area.offset.y), u32(game_area.offset.x),
                                     game_area.extent.width, game_area.extent.height};
        const bool subrect =
            game_area.offset.x != 0 || game_area.offset.y != 0 || game_area.extent != backbuffer;
        sl::ResourceTag tags[] = {
            sl::ResourceTag{&depth, sl::kBufferTypeDepth, sl::ResourceLifecycle::eOnlyValidNow,
                            &render_extent},
            sl::ResourceTag{&motion, sl::kBufferTypeMotionVectors,
                            sl::ResourceLifecycle::eOnlyValidNow, &render_extent},
            sl::ResourceTag{nullptr, sl::kBufferTypeBackbuffer, sl::ResourceLifecycle{},
                            &game_extent},
        };
        api.set_tags(*token, Viewport, tags, subrect ? 3u : 2u,
                     reinterpret_cast<sl::CommandBuffer*>(cmd));
        if (overlays) {
            // Copied now: the frame image and the mask are reused before DLSS-G is done.
            auto hudless = ImageResource(overlays->hudless, overlays->hudless_view,
                                         overlays->hudless_format, overlays->extent);
            auto alpha = ImageResource(overlays->alpha, overlays->alpha_view,
                                       vk::Format::eR16Sfloat, overlays->extent);
            const sl::Extent full{0, 0, overlays->extent.width, overlays->extent.height};
            sl::ResourceTag overlay_tags[] = {
                sl::ResourceTag{&hudless, sl::kBufferTypeHUDLessColor,
                                sl::ResourceLifecycle::eOnlyValidNow, &full},
                sl::ResourceTag{&alpha, sl::kBufferTypeUIAlpha,
                                sl::ResourceLifecycle::eOnlyValidNow, &full},
            };
            api.set_tags(*token, Viewport, overlay_tags, 2u,
                         reinterpret_cast<sl::CommandBuffer*>(cmd));
            options.enableUserInterfaceRecomposition = sl::Boolean::eTrue;
        }
    } else {
        sl::ResourceTag tags[] = {
            sl::ResourceTag{nullptr, sl::kBufferTypeDepth,
                            sl::ResourceLifecycle::eValidUntilPresent},
            sl::ResourceTag{nullptr, sl::kBufferTypeMotionVectors,
                            sl::ResourceLifecycle::eValidUntilPresent},
            sl::ResourceTag{nullptr, sl::kBufferTypeHUDLessColor,
                            sl::ResourceLifecycle::eOnlyValidNow},
            sl::ResourceTag{nullptr, sl::kBufferTypeUIAlpha, sl::ResourceLifecycle::eOnlyValidNow},
        };
        api.set_tags(*token, Viewport, tags, 4u, reinterpret_cast<sl::CommandBuffer*>(cmd));
    }
    if (options.mode != last_mode) {
        LOG_INFO(Render_Vulkan, "[FRAME-GEN] {}", inputs ? "On" : "Off");
        reflex_applied = false;
    }
    last_mode = options.mode;
    api.dlssg_options(Viewport, options);
    generating = inputs != nullptr;
}

void EndSubmit() {
    if (!presented_from_game)
        Marker(sl::PCLMarker::eRenderSubmitEnd);
}

void BeginPresent() {
    Marker(sl::PCLMarker::ePresentStart);
}

void EndPresent() {
    if (FsrFrameGen::Active()) {
        if (generating)
            last_generated = std::chrono::steady_clock::now();
        CountPresent(FsrFrameGen::TakeShown());
        return;
    }
    Marker(sl::PCLMarker::ePresentEnd);
    if (!frame_open) {
        generating = false;
        CountPresent(1);
        return;
    }
    frame_open = false;
    // Frames shown since the last query, generated ones included.
    u32 shown = 1;
    sl::DLSSGState state{};
    sl::Result result;
    {
        TimingScope timing{"dlssg_state"};
        result = api.dlssg_state(Viewport, state, nullptr);
    }
    if (result == sl::Result::eOk) {
        const u32 max = std::clamp<u32>(state.numFramesToGenerateMax, 1, 5) + 1;
        if (max != max_multiplier.exchange(max))
            LOG_INFO(Render_Vulkan, "[FRAME-GEN] Up to {}x", max);
        if (last_mode == sl::DLSSGMode::eOn) {
            shown = state.numFramesActuallyPresented;
            if ((state.status != sl::DLSSGStatus::eOk && ++frames % 600 == 1) ||
                status_logged < 3) {
                LOG_INFO(Render_Vulkan,
                         "[FRAME-GEN] status={} presented={} maxGenerated={} minSize={}",
                         u32(state.status), state.numFramesActuallyPresented,
                         state.numFramesToGenerateMax, state.minWidthOrHeight);
                ++status_logged;
            }
        }
    }
    if (generating)
        last_generated = std::chrono::steady_clock::now();
    CountPresent(shown);
}

void SetEnabled(bool value) {
    enabled = value;
}

void Pause() {
    paused_frames = 30;
    reflex_applied = false;
}

bool SkipRepeatedFrame() {
    return generating &&
           std::chrono::steady_clock::now() - last_generated < std::chrono::milliseconds{100};
}

void SetReflex(bool low_latency, bool sleep) {
    const bool previous_low_latency = reflex_low_latency.exchange(low_latency);
    const bool previous_sleep = reflex_sleep.exchange(sleep);
    if (low_latency != previous_low_latency || sleep != previous_sleep) {
        reflex_applied = false;
        RecordTiming("reflex_options", CurrentGameFrame(), 0, sleep);
    }
}

#else

PFN_vkGetInstanceProcAddr Load() {
    // FSR's swapchain presents the frames, generated by DLSS-G or FSR.
    if (Requested()) {
        FsrFrameGen::SetGenerator(Backend());
        FsrFrameGen::Select();
    }
    return nullptr;
}
bool Active() {
    return FsrFrameGen::Active();
}
void Shutdown() {
    FsrFrameGen::Shutdown();
}
void GameSubmitBegin() {}
void GameSubmitEnd() {}
u32 CurrentGameFrame() {
    return 0;
}
void SetPresentedGameFrame(u32) {}
void BeginFrame() {}
void SetFrame(const FrameInputs* inputs, const OverlayInputs* overlays, vk::CommandBuffer command,
              vk::Extent2D backbuffer, vk::Rect2D game_area) {
    FsrFrameGen::SetFrame(inputs, overlays, command, backbuffer, game_area);
    generating = FsrFrameGen::Active() && inputs != nullptr;
}
void EndSubmit() {}
void BeginPresent() {}
void EndPresent() {
    CountPresent(FsrFrameGen::TakeShown());
}
void SetEnabled(bool) {}
void Pause() {}
void SetReflex(bool, bool) {}
bool SkipRepeatedFrame() {
    return false;
}

#endif

} // namespace Vulkan::FrameGen
