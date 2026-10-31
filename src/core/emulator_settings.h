// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "common/logging/log.h"
#include "common/types.h"

#define EmulatorSettings (*EmulatorSettingsImpl::GetInstance())

enum HideCursorState : int {
    Never,
    Idle,
    Always,
};

enum UsbBackendType : int {
    Real,
    SkylandersPortal,
    InfinityBase,
    DimensionsToypad,
};

enum GpuReadbacksMode : int {
    Disabled,
    Relaxed,
    Precise,
};

enum class ConfigMode {
    Default,
    Global,
    Clean,
};

enum AudioBackend : int {
    SDL,
    OpenAL,
    // Add more backends as needed
};

enum OpenALHrtfMode : int {
    HrtfAuto, // Let OpenAL Soft decide (on for headphone-like stereo outputs)
    HrtfOn,   // Force HRTF binaural rendering
    HrtfOff,  // Never use HRTF
};

enum OpenALOutputMode : int {
    OutputAuto,       // Let OpenAL Soft negotiate with the device
    OutputStereo,     // Force stereo output
    OutputQuad,       // Force quadraphonic output
    OutputSurround51, // Force 5.1 surround output
    OutputSurround71, // Force 7.1 surround output
};

template <typename T>
struct Setting {
    T default_value{};
    T value{};
    std::optional<T> game_specific_value{};

    Setting() = default;
    // Single-argument ctor: initialises both default_value and value so
    // that CleanMode can always recover the intended factory default.
    /*implicit*/ Setting(T init) : default_value(std::move(init)), value(default_value) {}

    /// Return the active value under the given mode.
    T get(ConfigMode mode = ConfigMode::Default) const {
        switch (mode) {
        case ConfigMode::Default:
            return game_specific_value.value_or(value);
        case ConfigMode::Global:
            return value;
        case ConfigMode::Clean:
            return default_value;
        }
        return value;
    }

    /// Write v to the base layer.
    /// Set proper value as base or game_specific
    void set(const T& v, bool game_specific = false) {
        if (game_specific) {
            game_specific_value = v;
        } else {
            value = v;
        }
    }

    /// Discard the game-specific override; subsequent get(Default) will
    /// fall back to the base value.
    void reset_game_specific() {
        game_specific_value = std::nullopt;
    }
};

template <typename T>
void to_json(nlohmann::json& j, const Setting<T>& s) {
    j = s.value;
}

template <typename T>
void from_json(const nlohmann::json& j, Setting<T>& s) {
    s.value = j.get<T>();
}

struct OverrideItem {
    const char* key;
    std::function<void(void* group_ptr, const nlohmann::json& entry,
                       std::vector<std::string>& changed)>
        apply;
    /// Return the value that should be written to the per-game config file.
    /// Falls back to base value if no game-specific override is set.
    std::function<nlohmann::json(const void* group_ptr)> get_for_save;

    /// Clear game_specific_value for this field.
    std::function<void(void* group_ptr)> reset_game_specific;
};

template <typename Struct, typename T>
inline OverrideItem make_override(const char* key, Setting<T> Struct::* member) {
    return OverrideItem{
        key,
        [member, key](void* base, const nlohmann::json& entry, std::vector<std::string>& changed) {
            Struct* obj = reinterpret_cast<Struct*>(base);
            Setting<T>& dst = obj->*member;
            try {
                T newValue = entry.get<T>();
                if (dst.value != newValue) {
                    std::ostringstream oss;
                    oss << key << " ( " << dst.value << " -> " << newValue << " )";
                    changed.push_back(oss.str());
                }
                dst.game_specific_value = newValue;
            } catch (const std::exception& e) {
                LOG_ERROR(Config, "[make_override] error parsing {}: {}", key, e.what());
                LOG_ERROR(Config, "[make_override] Entry was: {}", entry.dump());
                LOG_ERROR(Config, "[make_override] Type name: {}", entry.type_name());
            }
        },

        // --- get_for_save -------------------------------------------
        // Returns game_specific_value when present, otherwise base value.
        // This means a freshly-opened game-specific dialog still shows
        // useful (current-global) values rather than empty entries.
        [member](const void* base) -> nlohmann::json {
            const Struct* obj = reinterpret_cast<const Struct*>(base);
            const Setting<T>& src = obj->*member;
            return nlohmann::json(src.game_specific_value.value_or(src.value));
        },

        // --- reset_game_specific ------------------------------------
        [member](void* base) {
            Struct* obj = reinterpret_cast<Struct*>(base);
            (obj->*member).reset_game_specific();
        }};
}

// -------------------------------
// Support types
// -------------------------------
struct GameInstallDir {
    std::filesystem::path path;
    bool enabled;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GameInstallDir, path, enabled)

// -------------------------------
// General settings
// -------------------------------
struct GeneralSettings {
    Setting<std::vector<GameInstallDir>> install_dirs;
    Setting<std::filesystem::path> addon_install_dir;
    Setting<std::filesystem::path> home_dir;
    Setting<std::filesystem::path> sys_modules_dir;
    Setting<std::filesystem::path> font_dir;

    Setting<int> volume_slider{100};
    Setting<bool> neo_mode{false};
    Setting<bool> dev_kit_mode{false};
    Setting<int> extra_dmem_in_mbytes{0};
    Setting<int> extra_fmem_in_mbytes{0};
    Setting<bool> trophy_popup_disabled{false};
    Setting<double> trophy_notification_duration{6.0};
    Setting<std::string> trophy_notification_side{"right"};
    Setting<bool> show_splash{false};
    Setting<bool> discord_rpc_enabled{false};
    Setting<bool> show_fps_counter{false};
    Setting<int> console_language{1};
    Setting<int> big_picture_scale{1000};
    Setting<int> big_picture_folder_depth{2};
    Setting<bool> redzone_patches{true};
    // return a vector of override descriptors (runtime, but tiny)
    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{
            make_override<GeneralSettings>("volume_slider", &GeneralSettings::volume_slider),
            make_override<GeneralSettings>("neo_mode", &GeneralSettings::neo_mode),
            make_override<GeneralSettings>("dev_kit_mode", &GeneralSettings::dev_kit_mode),
            make_override<GeneralSettings>("extra_dmem_in_mbytes",
                                           &GeneralSettings::extra_dmem_in_mbytes),
            make_override<GeneralSettings>("extra_fmem_in_mbytes",
                                           &GeneralSettings::extra_fmem_in_mbytes),
            make_override<GeneralSettings>("trophy_popup_disabled",
                                           &GeneralSettings::trophy_popup_disabled),
            make_override<GeneralSettings>("trophy_notification_duration",
                                           &GeneralSettings::trophy_notification_duration),
            make_override<GeneralSettings>("show_splash", &GeneralSettings::show_splash),
            make_override<GeneralSettings>("trophy_notification_side",
                                           &GeneralSettings::trophy_notification_side),
            make_override<GeneralSettings>("console_language", &GeneralSettings::console_language),
            make_override<GeneralSettings>("redzone_patches", &GeneralSettings::redzone_patches)};
    }
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GeneralSettings, install_dirs, addon_install_dir, home_dir,
                                   sys_modules_dir, font_dir, volume_slider, neo_mode, dev_kit_mode,
                                   extra_dmem_in_mbytes, extra_fmem_in_mbytes,
                                   trophy_popup_disabled, trophy_notification_duration, show_splash,
                                   trophy_notification_side, discord_rpc_enabled, show_fps_counter,
                                   console_language, big_picture_scale, big_picture_folder_depth,
                                   redzone_patches)

// -------------------------------
// Network settings
// -------------------------------
struct NetworkSettings {
    Setting<bool> connected_to_network{false};
    Setting<bool> shad_net_enabled{false};
    Setting<std::string> shadnet_server{"srv.shadps4.net:31313"};
    Setting<std::string> shadnet_webapi_server{"http://srv.shadps4.net:31315"};
    Setting<bool> enable_upnp{true};
    Setting<int> p2p_port{3658}; // 0 is any free port, 3658 is the default PS4 port. With UPnP
                                 // enabled it is forwarded on the router.
    Setting<bool> disable_https{false};

    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{
            make_override<NetworkSettings>("connected_to_network",
                                           &NetworkSettings::connected_to_network),
            make_override<NetworkSettings>("shad_net_enabled", &NetworkSettings::shad_net_enabled),
            make_override<NetworkSettings>("shadnet_server", &NetworkSettings::shadnet_server),
            make_override<NetworkSettings>("shadnet_webapi_server",
                                           &NetworkSettings::shadnet_webapi_server),
            make_override<NetworkSettings>("enable_upnp", &NetworkSettings::enable_upnp),
            make_override<NetworkSettings>("p2p_port", &NetworkSettings::p2p_port),
            make_override<NetworkSettings>("disable_https", &NetworkSettings::disable_https)};
    }
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NetworkSettings, connected_to_network, shad_net_enabled,
                                   shadnet_server, shadnet_webapi_server, enable_upnp, p2p_port,
                                   disable_https)

// -------------------------------
// Log settings
// -------------------------------
struct LogSettings {
    Setting<bool> append{false}; // specific
    Setting<bool> enable{true};  // specific
    Setting<std::string> filter{""};
    Setting<std::string> flush_level{""};
    Setting<u32> max_skip_duration{5'000};
    Setting<bool> separate{false}; // specific
    Setting<unsigned long long> size_limit{100_MB};
    Setting<bool> skip_duplicate{true};
    Setting<bool> sync{true};
#ifdef _WIN32
    Setting<std::string> type{"wincolor"};
#endif

    // return a vector of override descriptors (runtime, but tiny)
    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{
            make_override<LogSettings>("append", &LogSettings::append),
            make_override<LogSettings>("enable", &LogSettings::enable),
            make_override<LogSettings>("filter", &LogSettings::filter),
            make_override<LogSettings>("flush_level", &LogSettings::flush_level),
            make_override<LogSettings>("max_skip_duration", &LogSettings::max_skip_duration),
            make_override<LogSettings>("separate", &LogSettings::separate),
            make_override<LogSettings>("size_limit", &LogSettings::size_limit),
            make_override<LogSettings>("skip_duplicate", &LogSettings::skip_duplicate),
            make_override<LogSettings>("sync", &LogSettings::sync),
#ifdef _WIN32
            make_override<LogSettings>("type", &LogSettings::type),
#endif
        };
    }
};
#ifdef _WIN32
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LogSettings, append, enable, filter, flush_level,
                                   max_skip_duration, separate, size_limit, skip_duplicate, sync,
                                   type)
#else
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LogSettings, append, enable, filter, flush_level,
                                   max_skip_duration, separate, size_limit, skip_duplicate, sync)
#endif

// -------------------------------
// Debug settings
// -------------------------------
struct DebugSettings {
    Setting<bool> debug_dump{false};         // specific
    Setting<bool> shader_collect{false};     // specific
    Setting<std::string> config_version{""}; // specific

    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{
            make_override<DebugSettings>("debug_dump", &DebugSettings::debug_dump),
            make_override<DebugSettings>("shader_collect", &DebugSettings::shader_collect)};
    }
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DebugSettings, debug_dump, shader_collect, config_version)

// -------------------------------
// Input settings
// -------------------------------

struct InputSettings {
    Setting<int> cursor_state{HideCursorState::Idle};      // specific
    Setting<int> cursor_hide_timeout{5};                   // specific
    Setting<int> usb_device_backend{UsbBackendType::Real}; // specific
    Setting<bool> use_special_pad{false};
    Setting<int> special_pad_class{1};
    Setting<bool> motion_controls_enabled{true}; // specific
    Setting<bool> use_unified_input_config{true};
    Setting<std::string> default_controller_id{""};
    Setting<bool> background_controller_input{false}; // specific
    Setting<bool> ime_accessibility_enabled{false};   // specific
    Setting<bool> ime_url_mail_short_panel{false};    // specific
    Setting<bool> is_circle_enter{false};             // specific
    Setting<s32> camera_id{-1};
    Setting<bool> use_mice_as_mice{false};
    Setting<bool> use_keyboard_as_keyboard{false};

    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{
            make_override<InputSettings>("cursor_state", &InputSettings::cursor_state),
            make_override<InputSettings>("cursor_hide_timeout",
                                         &InputSettings::cursor_hide_timeout),
            make_override<InputSettings>("usb_device_backend", &InputSettings::usb_device_backend),
            make_override<InputSettings>("motion_controls_enabled",
                                         &InputSettings::motion_controls_enabled),
            make_override<InputSettings>("background_controller_input",
                                         &InputSettings::background_controller_input),
            make_override<InputSettings>("ime_accessibility_enabled",
                                         &InputSettings::ime_accessibility_enabled),
            make_override<InputSettings>("ime_url_mail_short_panel",
                                         &InputSettings::ime_url_mail_short_panel),
            make_override<InputSettings>("is_circle_enter", &InputSettings::is_circle_enter),
            make_override<InputSettings>("camera_id", &InputSettings::camera_id),
            make_override<InputSettings>("use_mice_as_mice", &InputSettings::use_mice_as_mice),
            make_override<InputSettings>("use_keyboard_as_keyboard",
                                         &InputSettings::use_keyboard_as_keyboard)};
    }
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(InputSettings, cursor_state, cursor_hide_timeout,
                                   usb_device_backend, use_special_pad, special_pad_class,
                                   motion_controls_enabled, use_unified_input_config,
                                   default_controller_id, background_controller_input,
                                   ime_accessibility_enabled, ime_url_mail_short_panel, camera_id,
                                   is_circle_enter, use_mice_as_mice, use_keyboard_as_keyboard)
// -------------------------------
// Audio settings
// -------------------------------
struct AudioSettings {
    Setting<u32> audio_backend{AudioBackend::SDL};
    Setting<std::string> sdl_mic_device{"Default Device"};
    Setting<std::string> sdl_main_output_device{"Default Device"};
    Setting<std::string> sdl_padSpk_output_device{"Default Device"};
    Setting<std::string> openal_mic_device{"Default Device"};
    Setting<std::string> openal_main_output_device{"Default Device"};
    Setting<std::string> openal_padSpk_output_device{"Default Device"};
    Setting<u32> openal_hrtf{OpenALHrtfMode::HrtfAuto};
    Setting<u32> openal_output_mode{OpenALOutputMode::OutputAuto};

    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{
            make_override<AudioSettings>("audio_backend", &AudioSettings::audio_backend),
            make_override<AudioSettings>("sdl_mic_device", &AudioSettings::sdl_mic_device),
            make_override<AudioSettings>("sdl_main_output_device",
                                         &AudioSettings::sdl_main_output_device),
            make_override<AudioSettings>("sdl_padSpk_output_device",
                                         &AudioSettings::sdl_padSpk_output_device),
            make_override<AudioSettings>("openal_mic_device", &AudioSettings::openal_mic_device),
            make_override<AudioSettings>("openal_main_output_device",
                                         &AudioSettings::openal_main_output_device),
            make_override<AudioSettings>("openal_padSpk_output_device",
                                         &AudioSettings::openal_padSpk_output_device),
            make_override<AudioSettings>("openal_hrtf", &AudioSettings::openal_hrtf),
            make_override<AudioSettings>("openal_output_mode", &AudioSettings::openal_output_mode)};
    }
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(AudioSettings, audio_backend, sdl_mic_device,
                                   sdl_main_output_device, sdl_padSpk_output_device,
                                   openal_mic_device, openal_main_output_device,
                                   openal_padSpk_output_device, openal_hrtf, openal_output_mode)

// -------------------------------
// GPU settings
// -------------------------------
// Bloodborne settings whose default changed to on are switched on once in configs saved before
// (each saved config holds every value, so a new default never reached them). 1: "Also free idle
// buffers" and "Background texture copies". 2: "Real occlusion tests". 3: "Use more VRAM",
// "Overlap game and GPU frames", "Early GPU data write-back", "Keep rendering while presenting",
// "Copy only kept texture levels" and "Direct texture copies". 4: the threaded renderer's settings
// that F3 lists only as advanced on, "Real occlusion tests" off (white flicker), the
// screen-space reflections back to the game's (hidden from F1 for now), "Windows Guest Red Zone
// Protection" on (crashes without it) and "Enable Shader Cache" on.
constexpr int BbDefaultsVersion = 4;

struct GPUSettings {
    Setting<u32> window_width{1280};
    Setting<u32> window_height{720};
    Setting<u32> internal_screen_width{1280};
    Setting<u32> internal_screen_height{720};
    Setting<bool> null_gpu{false};
    Setting<bool> copy_gpu_buffers{false};
    Setting<u32> readbacks_mode{GpuReadbacksMode::Disabled};
    Setting<bool> readback_linear_images_enabled{false};
    Setting<bool> direct_memory_access_enabled{false};
    Setting<bool> dump_shaders{false};
    Setting<bool> patch_shaders{false};
    Setting<u32> vblank_frequency{60};
    Setting<bool> full_screen{false};
    Setting<std::string> full_screen_mode{"Windowed"};
    Setting<std::string> present_mode{"Mailbox"};
    Setting<bool> hdr_allowed{false};
    Setting<bool> fsr_enabled{false};
    Setting<bool> rcas_enabled{true};
    Setting<int> rcas_attenuation{250};
    Setting<bool> userfaultfd{false};
    Setting<bool> inline_fetch_shader{false};
    // Bloodborne: records draws on a second thread (vk_draw_pipe.h).
    Setting<bool> threaded_renderer{true};
    // Bloodborne threaded renderer: statistics, time breakdown and hitch reports in the log.
    Setting<bool> threaded_renderer_diagnostics{false};
    // Bloodborne threaded renderer: GPU time per pass in the log (vk_gpu_profiler.h).
    Setting<bool> threaded_renderer_gpu_profile{false};
    // No longer used: F1's Object motion vectors (dlss.ini object_motion) switches the velocity
    // mirror. Kept so saved configs still load.
    Setting<bool> bb_object_motion{true};
    // Under the threaded renderer (each off: the emulator's original behaviour for that part).
    // GPU readbacks on a second queue (copy engine), waiting only for the GPU work that wrote
    // the data instead of all queued work.
    Setting<bool> bb_readback_queue{true};
    // Streaming: parallel copies, resident buffer memory in 64 MiB slabs, staging memory kept.
    Setting<bool> bb_streaming{true};
    // Copy shader batches in one compute dispatch, merged within 64 KiB.
    Setting<bool> bb_merged_copies{true};
    // Faster Windows locks and sleeps (applies at game start).
    Setting<bool> bb_fast_locks{true};
    // Flip a frame as soon as it's ready (between vblanks), evenly paced, instead of on the
    // next vblank tick.
    Setting<bool> bb_flip_when_ready{true};
    // VRAM cleanup (applies at game start): frees idle textures when VRAM runs low.
    // Off: the emulator's original cache behaviour.
    Setting<bool> bb_vram_reclaim{true};
    // With VRAM cleanup: also free guest buffer memory unused for 30 s (applies at start).
    Setting<bool> bb_buffer_eviction{true};
    // With VRAM cleanup: at each loading screen, free the textures and buffers not used since it
    // started, so play continues with more free VRAM (the next area uploads what it needs).
    Setting<bool> bb_load_screen_cleanup{true};
    // With VRAM cleanup (testing): marks from the driver's current VRAM budget, a small margin
    // below it, instead of an eighth of the card kept free. Off: the usual marks.
    Setting<bool> bb_more_vram{true};
    // With "Use more VRAM" (testing, applies at start): up to the card's size less a reserve for
    // Windows, past the driver's budget (Windows then makes room by moving other programs' memory
    // to system RAM), with memory priorities so texture copies are the first to go if it moves any
    // of the emulator's.
    Setting<bool> bb_vram_past_budget{false};
    // Texture quality: 0 = full, 1 = High (textures over 2048 stored at half size), 2 = Medium
    // (over 1024). Only block-compressed textures with mip levels that the GPU never writes: their
    // largest level is left out. Applies to textures loaded after a change.
    Setting<int> bb_texture_quality{0};
    // shadPS4's in-game windows (F3, F1, pop-ups) sized by screen pixels, as games size their
    // menus, instead of also by Windows' display scaling (150% made them half as big again).
    Setting<bool> bb_ignore_display_scale{true};
    // Testing: the VRAM budget the caches assume, in MiB (0 = the GPU's real budget; at start).
    Setting<int> bb_vram_budget_test_mb{0};
    // Threaded renderer: textures and large buffer uploads copied from guest memory on background
    // threads; the copies are waited for before the GPU submission that reads them and before the
    // guest sees a fence.
    Setting<bool> bb_background_texture_copies{true};
    // Threaded renderer: clean vertex/index data of 1-16 KiB read where it is in GPU memory
    // instead of copied for every draw. Off: copied as before.
    Setting<bool> bb_geometry_in_place{true};
    // Threaded renderer: draws that cannot write anything (no pixel shader, colour, depth or
    // stencil target, no memory writes; occlusion query geometry) skipped. Off: drawn.
    Setting<bool> bb_skip_empty_draws{true};
    // Threaded renderer: new scene pipelines compiled on background threads, their draws skipped
    // until ready (a frame or two) instead of the game waiting. Off: compiled at once.
    Setting<bool> bb_background_pipelines{true};
    // Anisotropic filtering for the game's mipmapped textures (surfaces at an angle stay sharp
    // without shimmering): 2, 4, 8 or 16 samples. 0: the game's own filtering.
    Setting<int> bb_anisotropy{16};
    // Where the per-draw data the CPU writes every frame lives: 0 auto (graphics card memory
    // with Resizable BAR / Smart Access Memory, system RAM without), 1 graphics card, 2 system.
    Setting<int> bb_draw_data_memory{0};
    // Page protection: only the pages whose protection changes, not the already protected pages
    // between them (Windows' cost is per page). Off: one call over the whole stretch.
    Setting<bool> bb_exact_protection{true};
    // New textures a draw needs are unshuffled and copied together: one barrier and one render
    // pass break for the group instead of per texture. Off: one texture at a time.
    Setting<bool> bb_batched_uploads{true};
    // Threaded renderer: a draw's textures reused as found for the same textures last time
    // (texture set memo). Off: looked up for every draw.
    Setting<bool> bb_texture_set_memo{true};
    // Threaded renderer: small or changed vertex/index data copied ahead into the draw data ring
    // by the decoding thread. Off: copied by the draw recorder as before.
    Setting<bool> bb_ring_geometry{true};
    // Threaded renderer: the next GPU submission is decoded while the draw recorder still
    // records the last one. Off: waits for the recorder after each submission.
    Setting<bool> bb_submit_overlap{true};
    // Threaded renderer (applies at start): the game's next frame is handed to the GPU side before
    // the game waits for the last one, and its first two submissions do not wait at all (after
    // Lantern Engine). Off: every submission after the end of a frame waits for the GPU side to be
    // idle, as before.
    Setting<bool> bb_overlap_frames{true};
    // Threaded renderer: memory a game thread had to wait for a GPU readback of is sent to the GPU
    // as soon as GPU work writes it again, and copied back once the GPU has finished, so the
    // game's next write does not wait. Off: read back only when the game touches it, as before.
    Setting<bool> bb_early_writeback{true};
    // Threaded renderer: occlusion query results written in order with the queued draws. Off:
    // the decoding thread waits for every queued draw first, as before.
    Setting<bool> bb_occlusion_in_order{true};
    // Pixel shader float math the driver may fuse (less shader code); applies after a restart
    // and rebuilds the shader cache when changed. Off: unfused, as the PS4.
    Setting<bool> bb_fast_pixel_math{true};
    // Threaded renderer: the game's constant copies (constant RAM dumps) written at once when no
    // queued draw reads that memory later. Off: every one waits for the queued draws first.
    Setting<bool> bb_constants_no_wait{true};
    // NVIDIA: pixel shader inputs interpolated by the GPU where the shader can't tell the
    // difference (less shader code); applies after a restart and rebuilds the shader cache when
    // changed. Off: every input by GCN's exact formula.
    Setting<bool> bb_hardware_interpolation{true};
    // Threaded renderer: the game's occlusion tests (one box per area behind walls) counted by
    // the GPU, so the game skips areas that are hidden. Off: every test reads "visible".
    Setting<bool> bb_real_occlusion{false};
    // Threaded renderer: stage A's copies of game memory (constants, small vertex and index
    // data) made on a thread of their own; applies after a restart. Off: made by stage A.
    Setting<bool> bb_copy_thread{true};
    // Threaded renderer: while a frame is presented (a frame limiter such as RTSS may sleep in
    // the present), the GPU threads' submissions queue up and go to the GPU right after it
    // returns, instead of the threads waiting for the queue. Off: they wait for the present.
    Setting<bool> bb_render_during_present{true};
    // Texture quality High/Medium: the guest data of a texture's left-out largest level is not
    // copied for its uploads (only the levels kept), which is most of a traversal burst's bytes
    // for the textures it reduces. Off: the whole texture is copied and the level dropped after.
    Setting<bool> bb_skip_left_out_levels{true};
    // Threaded renderer: texture data is copied from the game's own addresses when the whole
    // texture is mapped and readable, not through the view of the backing memory (whose pages
    // the process has not touched yet: the first read of each 4 KiB page costs a fault, which
    // made traversal copies run at ~5 GB/s). Off: always through the backing view.
    Setting<bool> bb_direct_texture_copies{true};
    // Threaded renderer: when an area loads (hundreds of textures in a second), texture uploads
    // past 64 MiB a frame wait for the next frames (new ones show black meanwhile), so the load
    // is spread over several frames. Off: everything uploaded at once.
    Setting<bool> bb_spread_loads{true};
    // Diagnostics: every draw timed, and where the draw recorder's time went shown for each
    // stutter (slower: for finding stutter causes).
    Setting<bool> bb_stutter_detail{false};
    // Saved with the settings: the Bloodborne defaults they were saved with (BbDefaultsVersion).
    Setting<int> bb_defaults{BbDefaultsVersion};

    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{
            make_override<GPUSettings>("null_gpu", &GPUSettings::null_gpu),
            make_override<GPUSettings>("copy_gpu_buffers", &GPUSettings::copy_gpu_buffers),
            make_override<GPUSettings>("full_screen", &GPUSettings::full_screen),
            make_override<GPUSettings>("full_screen_mode", &GPUSettings::full_screen_mode),
            make_override<GPUSettings>("present_mode", &GPUSettings::present_mode),
            make_override<GPUSettings>("window_height", &GPUSettings::window_height),
            make_override<GPUSettings>("window_width", &GPUSettings::window_width),
            make_override<GPUSettings>("hdr_allowed", &GPUSettings::hdr_allowed),
            make_override<GPUSettings>("fsr_enabled", &GPUSettings::fsr_enabled),
            make_override<GPUSettings>("rcas_enabled", &GPUSettings::rcas_enabled),
            make_override<GPUSettings>("rcas_attenuation", &GPUSettings::rcas_attenuation),
            make_override<GPUSettings>("dump_shaders", &GPUSettings::dump_shaders),
            make_override<GPUSettings>("patch_shaders", &GPUSettings::patch_shaders),
            make_override<GPUSettings>("readbacks_mode", &GPUSettings::readbacks_mode),
            make_override<GPUSettings>("readback_linear_images_enabled",
                                       &GPUSettings::readback_linear_images_enabled),
            make_override<GPUSettings>("direct_memory_access_enabled",
                                       &GPUSettings::direct_memory_access_enabled),
            make_override<GPUSettings>("vblank_frequency", &GPUSettings::vblank_frequency),
            make_override<GPUSettings>("userfaultfd", &GPUSettings::userfaultfd),
            make_override<GPUSettings>("inline_fetch_shader", &GPUSettings::inline_fetch_shader),
            make_override<GPUSettings>("threaded_renderer", &GPUSettings::threaded_renderer),
            make_override<GPUSettings>("threaded_renderer_diagnostics",
                                       &GPUSettings::threaded_renderer_diagnostics),
            make_override<GPUSettings>("threaded_renderer_gpu_profile",
                                       &GPUSettings::threaded_renderer_gpu_profile),
            make_override<GPUSettings>("bb_object_motion", &GPUSettings::bb_object_motion),
            make_override<GPUSettings>("bb_readback_queue", &GPUSettings::bb_readback_queue),
            make_override<GPUSettings>("bb_streaming", &GPUSettings::bb_streaming),
            make_override<GPUSettings>("bb_merged_copies", &GPUSettings::bb_merged_copies),
            make_override<GPUSettings>("bb_fast_locks", &GPUSettings::bb_fast_locks),
            make_override<GPUSettings>("bb_flip_when_ready", &GPUSettings::bb_flip_when_ready),
            make_override<GPUSettings>("bb_vram_reclaim", &GPUSettings::bb_vram_reclaim),
            make_override<GPUSettings>("bb_buffer_eviction", &GPUSettings::bb_buffer_eviction),
            make_override<GPUSettings>("bb_load_screen_cleanup",
                                       &GPUSettings::bb_load_screen_cleanup),
            make_override<GPUSettings>("bb_more_vram", &GPUSettings::bb_more_vram),
            make_override<GPUSettings>("bb_vram_past_budget", &GPUSettings::bb_vram_past_budget),
            make_override<GPUSettings>("bb_texture_quality", &GPUSettings::bb_texture_quality),
            make_override<GPUSettings>("bb_ignore_display_scale",
                                       &GPUSettings::bb_ignore_display_scale),
            make_override<GPUSettings>("bb_background_texture_copies",
                                       &GPUSettings::bb_background_texture_copies),
            make_override<GPUSettings>("bb_vram_budget_test_mb",
                                       &GPUSettings::bb_vram_budget_test_mb),
            make_override<GPUSettings>("bb_geometry_in_place", &GPUSettings::bb_geometry_in_place),
            make_override<GPUSettings>("bb_skip_empty_draws", &GPUSettings::bb_skip_empty_draws),
            make_override<GPUSettings>("bb_background_pipelines",
                                       &GPUSettings::bb_background_pipelines),
            make_override<GPUSettings>("bb_anisotropy", &GPUSettings::bb_anisotropy),
            make_override<GPUSettings>("bb_draw_data_memory", &GPUSettings::bb_draw_data_memory),
            make_override<GPUSettings>("bb_exact_protection", &GPUSettings::bb_exact_protection),
            make_override<GPUSettings>("bb_batched_uploads", &GPUSettings::bb_batched_uploads),
            make_override<GPUSettings>("bb_texture_set_memo", &GPUSettings::bb_texture_set_memo),
            make_override<GPUSettings>("bb_ring_geometry", &GPUSettings::bb_ring_geometry),
            make_override<GPUSettings>("bb_submit_overlap", &GPUSettings::bb_submit_overlap),
            make_override<GPUSettings>("bb_overlap_frames", &GPUSettings::bb_overlap_frames),
            make_override<GPUSettings>("bb_early_writeback", &GPUSettings::bb_early_writeback),
            make_override<GPUSettings>("bb_occlusion_in_order",
                                       &GPUSettings::bb_occlusion_in_order),
            make_override<GPUSettings>("bb_fast_pixel_math", &GPUSettings::bb_fast_pixel_math),
            make_override<GPUSettings>("bb_constants_no_wait", &GPUSettings::bb_constants_no_wait),
            make_override<GPUSettings>("bb_hardware_interpolation",
                                       &GPUSettings::bb_hardware_interpolation),
            make_override<GPUSettings>("bb_real_occlusion", &GPUSettings::bb_real_occlusion),
            make_override<GPUSettings>("bb_copy_thread", &GPUSettings::bb_copy_thread),
            make_override<GPUSettings>("bb_render_during_present",
                                       &GPUSettings::bb_render_during_present),
            make_override<GPUSettings>("bb_skip_left_out_levels",
                                       &GPUSettings::bb_skip_left_out_levels),
            make_override<GPUSettings>("bb_direct_texture_copies",
                                       &GPUSettings::bb_direct_texture_copies),
            make_override<GPUSettings>("bb_spread_loads", &GPUSettings::bb_spread_loads),
            make_override<GPUSettings>("bb_stutter_detail", &GPUSettings::bb_stutter_detail),
            make_override<GPUSettings>("bb_defaults", &GPUSettings::bb_defaults),
        };
    }
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    GPUSettings, window_width, window_height, internal_screen_width, internal_screen_height,
    null_gpu, copy_gpu_buffers, readbacks_mode, readback_linear_images_enabled,
    direct_memory_access_enabled, dump_shaders, patch_shaders, vblank_frequency, full_screen,
    full_screen_mode, present_mode, hdr_allowed, fsr_enabled, rcas_enabled, rcas_attenuation,
    userfaultfd, inline_fetch_shader, threaded_renderer, threaded_renderer_diagnostics,
    threaded_renderer_gpu_profile, bb_object_motion, bb_readback_queue, bb_streaming,
    bb_merged_copies, bb_fast_locks, bb_flip_when_ready, bb_vram_reclaim, bb_buffer_eviction,
    bb_load_screen_cleanup, bb_more_vram, bb_vram_past_budget, bb_texture_quality,
    bb_ignore_display_scale, bb_vram_budget_test_mb, bb_background_texture_copies,
    bb_geometry_in_place, bb_skip_empty_draws, bb_background_pipelines, bb_anisotropy,
    bb_exact_protection, bb_batched_uploads, bb_texture_set_memo, bb_ring_geometry,
    bb_submit_overlap, bb_overlap_frames, bb_early_writeback, bb_stutter_detail, bb_defaults,
    bb_occlusion_in_order, bb_fast_pixel_math, bb_constants_no_wait, bb_hardware_interpolation,
    bb_real_occlusion, bb_copy_thread, bb_spread_loads, bb_render_during_present,
    bb_skip_left_out_levels, bb_direct_texture_copies)

// -------------------------------
// Vulkan settings
// -------------------------------
struct VulkanSettings {
    Setting<s32> gpu_id{-1};
    Setting<bool> renderdoc_enabled{false};
    Setting<bool> vkvalidation_enabled{false};
    Setting<bool> vkvalidation_core_enabled{true};
    Setting<bool> vkvalidation_sync_enabled{false};
    Setting<bool> vkvalidation_gpu_enabled{false};
    Setting<bool> vkcrash_diagnostic_enabled{false};
    Setting<bool> vkhost_markers{false};
    Setting<bool> vkguest_markers{false};
    Setting<bool> pipeline_cache_enabled{true};
    Setting<bool> pipeline_cache_archived{false};
    // Bloodborne threaded renderer (here: GPUSettings' JSON field list is at its 64 limit).
    // Copy threads for texture and buffer data (0 = auto: half the logical cores, 2 to 8;
    // otherwise this many, leaving two cores); applies after a restart. Traversal bursts of new
    // textures copy 200-440 MiB in one frame: ~7 GB/s on 4 threads, ~10 GB/s on 8.
    Setting<int> bb_copy_threads{0};
    // RenderDoc (diagnostics): frames one F12 capture takes, one after another in one file.
    Setting<int> bb_rdoc_capture_frames{1};
    // Texture uploads staged in cached system memory instead of write-combined (uncached)
    // memory: faster for the CPU's copy, possibly slower for the GPU's read. Off: uncached.
    Setting<bool> bb_cached_texture_staging{true};
    // Lower detail first: in a burst of new textures (a traversal load) their largest level
    // waits and arrives over the next frames within a per-frame budget; meanwhile shaders
    // sample the next level down (softer for a few frames). Needs VK_EXT_image_view_min_lod.
    Setting<bool> bb_lower_detail_first{true};
    // Clear new shader-written images: a new image a compute shader only writes, bound whole
    // with a thread for every texel, is cleared on the GPU instead of uploaded from guest memory
    // first (the shader overwrites it). Traversal bursts uploaded 180-410 MiB of them.
    Setting<bool> bb_clear_shader_written{true};
    // Constant dumps past huge shader writes: a queued shader's storage write declared over
    // 256 MiB (a descriptor over all of memory, ~3.7 GiB in Bloodborne) does not hold up a
    // constant dump to memory inside it; the dump is written at once, as with nothing queued
    // (the command thread waited 20-40 ms for the draw recorder in traversal bursts).
    Setting<bool> bb_ce_skip_huge_writes{true};
    // Cheaper buffer refill (buffer eviction): evicted buffers keep their page protection and
    // are uploaded straight from guest memory when made resident again, instead of being marked
    // changed (unprotected) at eviction and protected again, page run by page run, at their next
    // use (~12-15 ms per traversal burst).
    Setting<bool> bb_evict_keep_protection{true};
    // Keep returning buffers (buffer eviction): a loading screen evicts only what VRAM needs
    // (down to 1 GiB below the pressure limit) instead of every buffer idle for 2 s (~700 MiB
    // per respawn, ~170 MiB of it uploaded again in the next area's first frames), and buffers
    // that came back after an eviction are evicted last.
    Setting<bool> bb_evict_keep_returning{true};
    // Protection on a helper thread: a new texture's memory protection (noticing the game's
    // writes) is done by one helper thread instead of the draw recorder (~6-9 ms of a traversal
    // burst); the texture's background copies wait for it. Needs background texture copies.
    Setting<bool> bb_protect_helper_thread{true};
    // Copy shader on the CPU: the game's GPU copy shader (thousands of small copies a second)
    // is done in game memory when neither side holds data only the GPU has. Its destination
    // was marked GPU-written, and the game's next write near it waited for the GPU to finish
    // (3-16 ms each, the slow in-play frames when running into new parts of an area).
    Setting<bool> bb_cpu_copy_shader{true};
    // With "Copy shader on the CPU": a copy whose destination already holds the source's bytes is
    // left out. Writing them again still marked the memory changed, so the buffers over it were
    // uploaded again.
    Setting<bool> bb_copy_skip_same{true};
    // Clear one face of a cube or one layer of an array: the game clears its point lights'
    // shadow cube a face at a time with a compute shader. Only whole images were cleared as
    // images; a face was left to the shader, which wrote memory only, and the face kept its
    // older shadows (wedges of light in lamp shadows).
    Setting<bool> bb_clear_one_layer{true};
    // Better ambient occlusion: the game's screen-space AO replaced with GT-VBAO (ground-truth
    // AO with a visibility bitmask) from the scene's depth and normals: contact shadows in
    // corners, under objects and in carvings, without the game's halos.
    Setting<bool> bb_vbao{false};
    // Its strength (exponent on the visibility, x100) and reach (metres, x100). Set in the F1
    // window, global only: a game's own settings file kept an old value that the F1 window's
    // change did not replace.
    Setting<int> bb_vbao_strength{150};
    Setting<int> bb_vbao_radius{150};
    // The AO computed at half the scene's size and upscaled guided by the depth (about a
    // quarter of the cost; slightly softer fine detail). Off: at the full scene size.
    Setting<bool> bb_vbao_half_res{true};
    // Screen-space reflections (with the game's "Enable Screen Space Reflections" patch):
    // 0 the game's, 1 better (traced through a depth pyramid with the game's own inputs, and
    // denoised), 2 off (none at all). Global only.
    Setting<int> bb_ssr_mode{0};
    // Better reflections at the scene's full size (not the game's half size), from a full-size
    // copy of the last frame, read by the lighting directly: a little sharper, at more GPU time.
    Setting<bool> bb_ssr_full_res{false};
    // Settings window: the threaded renderer's individual settings are listed (remembered).
    Setting<bool> bb_show_threaded_settings{false};
    // Bloodborne's frame rate set free by hooks in the game's code instead of the
    // frame rate patches, which must be off then. Applies at the next start. Global only.
    Setting<bool> bb_frame_hooks{true};
    // Tests (issue #25, white dots), with merged copy shaders: the copies' sources never
    // snapshotted from game memory into the stream buffer (read from the buffer cache, as with
    // merged copies off), and the old 64 MiB batches instead of 64 KiB ones.
    Setting<bool> bb_merged_no_stream_sources{false};
    Setting<bool> bb_merged_large_batches{false};
    // The frame rate limit with the frame rate hooks (30..480; 0: none), changed while the game
    // runs. Global only.
    Setting<int> bb_fps_limit{0};
    // With an FPS cap ("Show frames when ready"): frames are shown one cap period apart, an early
    // one held until its turn, instead of each as soon as it is ready. Global only.
    Setting<bool> bb_even_capped_flips{true};

    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{
            make_override<VulkanSettings>("gpu_id", &VulkanSettings::gpu_id),
            make_override<VulkanSettings>("renderdoc_enabled", &VulkanSettings::renderdoc_enabled),
            make_override<VulkanSettings>("vkvalidation_enabled",
                                          &VulkanSettings::vkvalidation_enabled),
            make_override<VulkanSettings>("vkvalidation_core_enabled",
                                          &VulkanSettings::vkvalidation_core_enabled),
            make_override<VulkanSettings>("vkvalidation_sync_enabled",
                                          &VulkanSettings::vkvalidation_sync_enabled),
            make_override<VulkanSettings>("vkvalidation_gpu_enabled",
                                          &VulkanSettings::vkvalidation_gpu_enabled),
            make_override<VulkanSettings>("vkcrash_diagnostic_enabled",
                                          &VulkanSettings::vkcrash_diagnostic_enabled),
            make_override<VulkanSettings>("vkhost_markers", &VulkanSettings::vkhost_markers),
            make_override<VulkanSettings>("vkguest_markers", &VulkanSettings::vkguest_markers),
            make_override<VulkanSettings>("pipeline_cache_enabled",
                                          &VulkanSettings::pipeline_cache_enabled),
            make_override<VulkanSettings>("pipeline_cache_archived",
                                          &VulkanSettings::pipeline_cache_archived),
            make_override<VulkanSettings>("bb_copy_threads", &VulkanSettings::bb_copy_threads),
            make_override<VulkanSettings>("bb_rdoc_capture_frames",
                                          &VulkanSettings::bb_rdoc_capture_frames),
            make_override<VulkanSettings>("bb_cached_texture_staging",
                                          &VulkanSettings::bb_cached_texture_staging),
            make_override<VulkanSettings>("bb_lower_detail_first",
                                          &VulkanSettings::bb_lower_detail_first),
            make_override<VulkanSettings>("bb_clear_shader_written",
                                          &VulkanSettings::bb_clear_shader_written),
            make_override<VulkanSettings>("bb_ce_skip_huge_writes",
                                          &VulkanSettings::bb_ce_skip_huge_writes),
            make_override<VulkanSettings>("bb_evict_keep_protection",
                                          &VulkanSettings::bb_evict_keep_protection),
            make_override<VulkanSettings>("bb_evict_keep_returning",
                                          &VulkanSettings::bb_evict_keep_returning),
            make_override<VulkanSettings>("bb_protect_helper_thread",
                                          &VulkanSettings::bb_protect_helper_thread),
            make_override<VulkanSettings>("bb_cpu_copy_shader",
                                          &VulkanSettings::bb_cpu_copy_shader),
            make_override<VulkanSettings>("bb_copy_skip_same", &VulkanSettings::bb_copy_skip_same),
            make_override<VulkanSettings>("bb_clear_one_layer",
                                          &VulkanSettings::bb_clear_one_layer),
            make_override<VulkanSettings>("bb_merged_no_stream_sources",
                                          &VulkanSettings::bb_merged_no_stream_sources),
            make_override<VulkanSettings>("bb_merged_large_batches",
                                          &VulkanSettings::bb_merged_large_batches),
        };
    }
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    VulkanSettings, gpu_id, renderdoc_enabled, vkvalidation_enabled, vkvalidation_core_enabled,
    vkvalidation_sync_enabled, vkvalidation_gpu_enabled, vkcrash_diagnostic_enabled, vkhost_markers,
    vkguest_markers, pipeline_cache_enabled, pipeline_cache_archived, bb_copy_threads,
    bb_cached_texture_staging, bb_lower_detail_first, bb_clear_shader_written,
    bb_ce_skip_huge_writes, bb_evict_keep_protection, bb_evict_keep_returning,
    bb_protect_helper_thread, bb_cpu_copy_shader, bb_copy_skip_same, bb_clear_one_layer,
    bb_rdoc_capture_frames, bb_vbao, bb_vbao_strength, bb_vbao_radius, bb_vbao_half_res,
    bb_ssr_mode, bb_ssr_full_res, bb_show_threaded_settings, bb_frame_hooks, bb_fps_limit,
    bb_even_capped_flips, bb_merged_no_stream_sources, bb_merged_large_batches)

// -------------------------------
// Main manager
// -------------------------------
class EmulatorSettingsImpl {
public:
    EmulatorSettingsImpl();
    ~EmulatorSettingsImpl();

    static std::shared_ptr<EmulatorSettingsImpl> GetInstance();
    static void SetInstance(std::shared_ptr<EmulatorSettingsImpl> instance);

    bool Save(const std::string& serial = "");
    bool Load(const std::string& serial = "");
    void SetDefaultValues();
    bool TransferSettings();

    // Config mode
    ConfigMode GetConfigMode() const {
        return m_configMode;
    }
    void SetConfigMode(ConfigMode mode) {
        m_configMode = mode;
    }

    //
    // Game-specific override management
    /// Clears all per-game overrides.  Call this when a game exits so
    /// the emulator reverts to global settings.
    void ClearGameSpecificOverrides();

    // general accessors
    bool AddGameInstallDir(const std::filesystem::path& dir, bool enabled = true);
    std::vector<std::filesystem::path> GetGameInstallDirs() const;
    void SetAllGameInstallDirs(const std::vector<GameInstallDir>& dirs);
    void RemoveGameInstallDir(const std::filesystem::path& dir);
    void SetGameInstallDirEnabled(const std::filesystem::path& dir, bool enabled);
    void SetGameInstallDirs(const std::vector<std::filesystem::path>& dirs_config);
    const std::vector<bool> GetGameInstallDirsEnabled();
    const std::vector<GameInstallDir>& GetAllGameInstallDirs() const;

    std::filesystem::path GetHomeDir();
    void SetHomeDir(const std::filesystem::path& dir);
    std::filesystem::path GetSysModulesDir();
    void SetSysModulesDir(const std::filesystem::path& dir);
    std::filesystem::path GetFontsDir();
    void SetFontsDir(const std::filesystem::path& dir);
    std::filesystem::path GetAddonInstallDir();
    void SetAddonInstallDir(const std::filesystem::path& dir);

private:
    /// Switches on Bloodborne settings whose default changed, in a config saved before
    /// (BbDefaultsVersion). Returns whether it changed anything.
    bool UpdateBbDefaults(const nlohmann::json& file, bool game_specific);
    GeneralSettings m_general{};
    NetworkSettings m_network{};
    LogSettings m_log{};
    DebugSettings m_debug{};
    InputSettings m_input{};
    AudioSettings m_audio{};
    GPUSettings m_gpu{};
    VulkanSettings m_vulkan{};
    ConfigMode m_configMode{ConfigMode::Default};

    // Runtime-only override: when true, IsShadNetEnabled() reports false for the
    // rest of this run regardless of the persisted setting
    std::atomic<bool> m_shadnet_session_disabled{false};

    static std::shared_ptr<EmulatorSettingsImpl> s_instance;
    static std::mutex s_mutex;

    /// Apply overrideable fields from groupJson into group.game_specific_value.
    template <typename Group>
    void ApplyGroupOverrides(Group& group, const nlohmann::json& groupJson,
                             std::vector<std::string>& changed) {
        for (auto& item : group.GetOverrideableFields()) {
            if (!groupJson.contains(item.key))
                continue;
            item.apply(&group, groupJson.at(item.key), changed);
        }
    }

    // Write all overrideable fields from group into out (for game-specific save).
    template <typename Group>
    static void SaveGroupGameSpecific(const Group& group, nlohmann::json& out) {
        for (auto& item : group.GetOverrideableFields())
            out[item.key] = item.get_for_save(&group);
    }

    // Discard every game-specific override in group.
    template <typename Group>
    static void ClearGroupOverrides(Group& group) {
        for (auto& item : group.GetOverrideableFields())
            item.reset_game_specific(&group);
    }

    static void PrintChangedSummary(const std::vector<std::string>& changed);
    void MigrateNetworkKeys(const nlohmann::json& general, const nlohmann::json& network);

public:
    // Used by the launcher to back up / restore settings (e.g. per-game settings dialog)
    EmulatorSettingsImpl& operator=(const EmulatorSettingsImpl& other) {
        if (this != &other) {
            m_shadnet_session_disabled.store(other.m_shadnet_session_disabled.load());
            m_general = other.m_general;
            m_network = other.m_network;
            m_log = other.m_log;
            m_debug = other.m_debug;
            m_input = other.m_input;
            m_audio = other.m_audio;
            m_gpu = other.m_gpu;
            m_vulkan = other.m_vulkan;
            m_configMode = other.m_configMode;
        }
        return *this;
    }

    // Add these getters to access overrideable fields
    std::vector<OverrideItem> GetGeneralOverrideableFields() const {
        return m_general.GetOverrideableFields();
    }
    std::vector<OverrideItem> GetNetworkOverrideableFields() const {
        return m_network.GetOverrideableFields();
    }
    std::vector<OverrideItem> GetDebugOverrideableFields() const {
        return m_debug.GetOverrideableFields();
    }
    std::vector<OverrideItem> GetInputOverrideableFields() const {
        return m_input.GetOverrideableFields();
    }
    std::vector<OverrideItem> GetAudioOverrideableFields() const {
        return m_audio.GetOverrideableFields();
    }
    std::vector<OverrideItem> GetGPUOverrideableFields() const {
        return m_gpu.GetOverrideableFields();
    }
    std::vector<OverrideItem> GetVulkanOverrideableFields() const {
        return m_vulkan.GetOverrideableFields();
    }
    std::vector<std::string> GetAllOverrideableKeys() const;

#define SETTING_FORWARD(group, Name, field)                                                        \
    auto Get##Name() const {                                                                       \
        return (group).field.get(m_configMode);                                                    \
    }                                                                                              \
    void Set##Name(const decltype((group).field.value)& v, bool specific = false) {                \
        (group).field.set(v, specific);                                                            \
    }
#define SETTING_FORWARD_BOOL(group, Name, field)                                                   \
    bool Is##Name() const {                                                                        \
        return (group).field.get(m_configMode);                                                    \
    }                                                                                              \
    void Set##Name(bool v, bool specific = false) {                                                \
        (group).field.set(v, specific);                                                            \
    }
#define SETTING_FORWARD_BOOL_READONLY(group, Name, field)                                          \
    bool Is##Name() const {                                                                        \
        return (group).field.get(m_configMode);                                                    \
    }

    // General settings
    SETTING_FORWARD(m_general, VolumeSlider, volume_slider)
    SETTING_FORWARD_BOOL(m_general, Neo, neo_mode)
    SETTING_FORWARD_BOOL(m_general, DevKit, dev_kit_mode)
    SETTING_FORWARD(m_general, ExtraDmemInMBytes, extra_dmem_in_mbytes)
    SETTING_FORWARD(m_general, ExtraFmemInMBytes, extra_fmem_in_mbytes)
    bool IsShadNetEnabled() const {
        return m_network.shad_net_enabled.get(m_configMode) &&
               !m_shadnet_session_disabled.load(std::memory_order_relaxed);
    }
    void SetShadNetEnabled(bool v, bool specific = false) {
        m_network.shad_net_enabled.set(v, specific);
    }
    bool IsShadNetEnabledSetting() const {
        return m_network.shad_net_enabled.get(m_configMode);
    }
    void SetShadNetSessionDisabled(bool v) {
        m_shadnet_session_disabled.store(v, std::memory_order_relaxed);
    }
    bool IsShadNetSessionDisabled() const {
        return m_shadnet_session_disabled.load(std::memory_order_relaxed);
    }
    SETTING_FORWARD_BOOL(m_general, TrophyPopupDisabled, trophy_popup_disabled)
    SETTING_FORWARD(m_general, TrophyNotificationDuration, trophy_notification_duration)
    SETTING_FORWARD(m_general, TrophyNotificationSide, trophy_notification_side)
    SETTING_FORWARD_BOOL(m_general, ShowSplash, show_splash)
    SETTING_FORWARD_BOOL(m_general, DiscordRPCEnabled, discord_rpc_enabled)
    SETTING_FORWARD_BOOL(m_general, ShowFpsCounter, show_fps_counter)
    SETTING_FORWARD(m_general, ConsoleLanguage, console_language)
    SETTING_FORWARD(m_general, BigPictureScale, big_picture_scale)
    SETTING_FORWARD(m_general, BigPictureFolderDepth, big_picture_folder_depth)
    SETTING_FORWARD_BOOL(m_general, RedZonePatchingEnabled, redzone_patches)

    // Network settings
    SETTING_FORWARD_BOOL(m_network, ConnectedToNetwork, connected_to_network)
    SETTING_FORWARD(m_network, ShadNetServer, shadnet_server)
    SETTING_FORWARD(m_network, ShadNetWebApiServer, shadnet_webapi_server)
    SETTING_FORWARD_BOOL(m_network, UPnPEnabled, enable_upnp)
    SETTING_FORWARD(m_network, P2PPort, p2p_port)
    SETTING_FORWARD_BOOL(m_network, ForcedHttpsDisabled, disable_https)

    // Log settings
    SETTING_FORWARD_BOOL(m_log, LogAppend, append)
    SETTING_FORWARD_BOOL(m_log, LogEnable, enable)
    SETTING_FORWARD(m_log, LogFilter, filter)
    SETTING_FORWARD(m_log, LogFlushLevel, flush_level)
    SETTING_FORWARD(m_log, LogMaxSkipDuration, max_skip_duration)
    SETTING_FORWARD_BOOL(m_log, LogSeparate, separate)
    SETTING_FORWARD(m_log, LogSizeLimit, size_limit)
    SETTING_FORWARD_BOOL(m_log, LogSkipDuplicate, skip_duplicate)
    SETTING_FORWARD_BOOL(m_log, LogSync, sync)
#ifdef _WIN32
    SETTING_FORWARD(m_log, LogType, type)
#endif

    // Audio settings
    SETTING_FORWARD(m_audio, AudioBackend, audio_backend)
    SETTING_FORWARD(m_audio, SDLMicDevice, sdl_mic_device)
    SETTING_FORWARD(m_audio, SDLMainOutputDevice, sdl_main_output_device)
    SETTING_FORWARD(m_audio, SDLPadSpkOutputDevice, sdl_padSpk_output_device)
    SETTING_FORWARD(m_audio, OpenALMicDevice, openal_mic_device)
    SETTING_FORWARD(m_audio, OpenALMainOutputDevice, openal_main_output_device)
    SETTING_FORWARD(m_audio, OpenALPadSpkOutputDevice, openal_padSpk_output_device)
    SETTING_FORWARD(m_audio, OpenALHrtf, openal_hrtf)
    SETTING_FORWARD(m_audio, OpenALOutputMode, openal_output_mode)

    // Debug settings
    SETTING_FORWARD_BOOL(m_debug, DebugDump, debug_dump)
    SETTING_FORWARD_BOOL(m_debug, ShaderCollect, shader_collect)
    SETTING_FORWARD(m_debug, ConfigVersion, config_version)

    // GPU Settings
    SETTING_FORWARD_BOOL(m_gpu, NullGPU, null_gpu)
    SETTING_FORWARD_BOOL(m_gpu, DumpShaders, dump_shaders)
    SETTING_FORWARD_BOOL(m_gpu, CopyGpuBuffers, copy_gpu_buffers)
    SETTING_FORWARD_BOOL(m_gpu, FullScreen, full_screen)
    SETTING_FORWARD(m_gpu, FullScreenMode, full_screen_mode)
    SETTING_FORWARD(m_gpu, PresentMode, present_mode)
    SETTING_FORWARD(m_gpu, WindowHeight, window_height)
    SETTING_FORWARD(m_gpu, WindowWidth, window_width)
    SETTING_FORWARD(m_gpu, InternalScreenHeight, internal_screen_height)
    SETTING_FORWARD(m_gpu, InternalScreenWidth, internal_screen_width)
    SETTING_FORWARD_BOOL(m_gpu, HdrAllowed, hdr_allowed)
    SETTING_FORWARD_BOOL(m_gpu, FsrEnabled, fsr_enabled)
    SETTING_FORWARD_BOOL(m_gpu, RcasEnabled, rcas_enabled)
    SETTING_FORWARD(m_gpu, RcasAttenuation, rcas_attenuation)
    SETTING_FORWARD(m_gpu, ReadbacksMode, readbacks_mode)
    SETTING_FORWARD_BOOL(m_gpu, ReadbackLinearImagesEnabled, readback_linear_images_enabled)
    SETTING_FORWARD_BOOL(m_gpu, DirectMemoryAccessEnabled, direct_memory_access_enabled)
    SETTING_FORWARD_BOOL_READONLY(m_gpu, PatchShaders, patch_shaders)
    SETTING_FORWARD_BOOL(m_gpu, UserfaultfdTracking, userfaultfd)
    SETTING_FORWARD_BOOL(m_gpu, ThreadedRenderer, threaded_renderer)
    SETTING_FORWARD_BOOL(m_gpu, ThreadedRendererDiagnostics, threaded_renderer_diagnostics)
    SETTING_FORWARD_BOOL(m_gpu, ThreadedRendererGpuProfile, threaded_renderer_gpu_profile)
    SETTING_FORWARD_BOOL(m_gpu, BbObjectMotion, bb_object_motion)
    SETTING_FORWARD_BOOL(m_gpu, BbReadbackQueue, bb_readback_queue)
    SETTING_FORWARD_BOOL(m_gpu, BbStreaming, bb_streaming)
    SETTING_FORWARD_BOOL(m_gpu, BbMergedCopies, bb_merged_copies)
    SETTING_FORWARD_BOOL(m_gpu, BbFastLocks, bb_fast_locks)
    SETTING_FORWARD_BOOL(m_gpu, BbFlipWhenReady, bb_flip_when_ready)
    SETTING_FORWARD_BOOL(m_gpu, BbVramReclaim, bb_vram_reclaim)
    SETTING_FORWARD_BOOL(m_gpu, BbBufferEviction, bb_buffer_eviction)
    SETTING_FORWARD_BOOL(m_gpu, BbLoadScreenCleanup, bb_load_screen_cleanup)
    SETTING_FORWARD_BOOL(m_gpu, BbMoreVram, bb_more_vram)
    SETTING_FORWARD_BOOL(m_gpu, BbVramPastBudget, bb_vram_past_budget)
    SETTING_FORWARD_BOOL(m_gpu, BbBackgroundTextureCopies, bb_background_texture_copies)
    SETTING_FORWARD_BOOL(m_gpu, BbGeometryInPlace, bb_geometry_in_place)
    SETTING_FORWARD_BOOL(m_gpu, BbSkipEmptyDraws, bb_skip_empty_draws)
    SETTING_FORWARD_BOOL(m_gpu, BbBackgroundPipelines, bb_background_pipelines)
    SETTING_FORWARD_BOOL(m_gpu, BbExactProtection, bb_exact_protection)
    SETTING_FORWARD_BOOL(m_gpu, BbBatchedUploads, bb_batched_uploads)
    SETTING_FORWARD_BOOL(m_gpu, BbTextureSetMemo, bb_texture_set_memo)
    SETTING_FORWARD_BOOL(m_gpu, BbRingGeometry, bb_ring_geometry)
    SETTING_FORWARD_BOOL(m_gpu, BbSubmitOverlap, bb_submit_overlap)
    SETTING_FORWARD_BOOL(m_gpu, BbOverlapFrames, bb_overlap_frames)
    SETTING_FORWARD_BOOL(m_gpu, BbEarlyWriteback, bb_early_writeback)
    SETTING_FORWARD_BOOL(m_gpu, BbOcclusionInOrder, bb_occlusion_in_order)
    SETTING_FORWARD_BOOL(m_gpu, BbFastPixelMath, bb_fast_pixel_math)
    SETTING_FORWARD_BOOL(m_gpu, BbConstantsNoWait, bb_constants_no_wait)
    SETTING_FORWARD_BOOL(m_gpu, BbHardwareInterpolation, bb_hardware_interpolation)
    SETTING_FORWARD_BOOL(m_gpu, BbRealOcclusion, bb_real_occlusion)
    SETTING_FORWARD_BOOL(m_gpu, BbCopyThread, bb_copy_thread)
    SETTING_FORWARD_BOOL(m_gpu, BbRenderDuringPresent, bb_render_during_present)
    SETTING_FORWARD_BOOL(m_gpu, BbSkipLeftOutLevels, bb_skip_left_out_levels)
    SETTING_FORWARD_BOOL(m_gpu, BbDirectTextureCopies, bb_direct_texture_copies)
    SETTING_FORWARD(m_vulkan, BbCopyThreads, bb_copy_threads)
    SETTING_FORWARD(m_vulkan, BbRdocCaptureFrames, bb_rdoc_capture_frames)
    SETTING_FORWARD_BOOL(m_vulkan, BbCachedTextureStaging, bb_cached_texture_staging)
    SETTING_FORWARD_BOOL(m_vulkan, BbLowerDetailFirst, bb_lower_detail_first)
    SETTING_FORWARD_BOOL(m_vulkan, BbClearShaderWritten, bb_clear_shader_written)
    SETTING_FORWARD_BOOL(m_vulkan, BbCeSkipHugeWrites, bb_ce_skip_huge_writes)
    SETTING_FORWARD_BOOL(m_vulkan, BbEvictKeepProtection, bb_evict_keep_protection)
    SETTING_FORWARD_BOOL(m_vulkan, BbEvictKeepReturning, bb_evict_keep_returning)
    SETTING_FORWARD_BOOL(m_vulkan, BbProtectHelperThread, bb_protect_helper_thread)
    SETTING_FORWARD_BOOL(m_vulkan, BbCpuCopyShader, bb_cpu_copy_shader)
    SETTING_FORWARD_BOOL(m_vulkan, BbCopySkipSame, bb_copy_skip_same)
    SETTING_FORWARD_BOOL(m_vulkan, BbClearOneLayer, bb_clear_one_layer)
    SETTING_FORWARD_BOOL(m_vulkan, BbVbao, bb_vbao)
    SETTING_FORWARD(m_vulkan, BbVbaoStrength, bb_vbao_strength)
    SETTING_FORWARD(m_vulkan, BbVbaoRadius, bb_vbao_radius)
    SETTING_FORWARD_BOOL(m_vulkan, BbVbaoHalfRes, bb_vbao_half_res)
    SETTING_FORWARD(m_vulkan, BbSsrMode, bb_ssr_mode)
    SETTING_FORWARD_BOOL(m_vulkan, BbSsrFullRes, bb_ssr_full_res)
    SETTING_FORWARD_BOOL(m_vulkan, BbShowThreadedSettings, bb_show_threaded_settings)
    SETTING_FORWARD_BOOL(m_vulkan, BbFrameHooks, bb_frame_hooks)
    SETTING_FORWARD(m_vulkan, BbFpsLimit, bb_fps_limit)
    SETTING_FORWARD_BOOL(m_vulkan, BbEvenCappedFlips, bb_even_capped_flips)
    SETTING_FORWARD_BOOL(m_vulkan, BbMergedNoStreamSources, bb_merged_no_stream_sources)
    SETTING_FORWARD_BOOL(m_vulkan, BbMergedLargeBatches, bb_merged_large_batches)
    SETTING_FORWARD_BOOL(m_gpu, BbSpreadLoads, bb_spread_loads)
    SETTING_FORWARD_BOOL(m_gpu, BbStutterDetail, bb_stutter_detail)
    SETTING_FORWARD(m_gpu, BbVramBudgetTestMb, bb_vram_budget_test_mb)
    SETTING_FORWARD(m_gpu, BbTextureQuality, bb_texture_quality)
    SETTING_FORWARD_BOOL(m_gpu, BbIgnoreDisplayScale, bb_ignore_display_scale)
    SETTING_FORWARD(m_gpu, BbAnisotropy, bb_anisotropy)
    SETTING_FORWARD(m_gpu, BbDrawDataMemory, bb_draw_data_memory)
    SETTING_FORWARD_BOOL_READONLY(m_gpu, InlineFetchShader, inline_fetch_shader)

    u32 GetVblankFrequency() {
        if (m_gpu.vblank_frequency.value < 30) {
            return 30;
        }
        return m_gpu.vblank_frequency.get();
    }
    void SetVblankFrequency(const u32& v, bool is_specific = false) {
        u32 val = v < 30 ? 30 : v;
        if (is_specific) {
            m_gpu.vblank_frequency.game_specific_value = val;
        } else {
            m_gpu.vblank_frequency.value = val;
        }
    }

    // Input Settings
    SETTING_FORWARD(m_input, CursorState, cursor_state)
    SETTING_FORWARD(m_input, CursorHideTimeout, cursor_hide_timeout)
    SETTING_FORWARD(m_input, UsbDeviceBackend, usb_device_backend)
    SETTING_FORWARD_BOOL(m_input, MotionControlsEnabled, motion_controls_enabled)
    SETTING_FORWARD_BOOL(m_input, BackgroundControllerInput, background_controller_input)
    SETTING_FORWARD_BOOL(m_input, ImeAccessibilityEnabled, ime_accessibility_enabled)
    SETTING_FORWARD_BOOL(m_input, ImeUrlMailShortPanel, ime_url_mail_short_panel)
    SETTING_FORWARD(m_input, DefaultControllerId, default_controller_id)
    SETTING_FORWARD_BOOL(m_input, UsingSpecialPad, use_special_pad)
    SETTING_FORWARD(m_input, SpecialPadClass, special_pad_class)
    SETTING_FORWARD_BOOL(m_input, UseUnifiedInputConfig, use_unified_input_config)
    SETTING_FORWARD(m_input, CameraId, camera_id)
    SETTING_FORWARD_BOOL(m_input, CircleEnter, is_circle_enter)
    SETTING_FORWARD_BOOL(m_input, MiceUsedAsMice, use_mice_as_mice)
    SETTING_FORWARD_BOOL(m_input, KeyboardUsedAsKeyboard, use_keyboard_as_keyboard)

    // Vulkan settings
    SETTING_FORWARD(m_vulkan, GpuId, gpu_id)
    SETTING_FORWARD_BOOL(m_vulkan, RenderdocEnabled, renderdoc_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, VkValidationEnabled, vkvalidation_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, VkValidationCoreEnabled, vkvalidation_core_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, VkValidationSyncEnabled, vkvalidation_sync_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, VkValidationGpuEnabled, vkvalidation_gpu_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, VkCrashDiagnosticEnabled, vkcrash_diagnostic_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, VkHostMarkersEnabled, vkhost_markers)
    SETTING_FORWARD_BOOL(m_vulkan, VkGuestMarkersEnabled, vkguest_markers)
    SETTING_FORWARD_BOOL(m_vulkan, PipelineCacheEnabled, pipeline_cache_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, PipelineCacheArchived, pipeline_cache_archived)

#undef SETTING_FORWARD
#undef SETTING_FORWARD_BOOL
#undef SETTING_FORWARD_BOOL_READONLY
};
