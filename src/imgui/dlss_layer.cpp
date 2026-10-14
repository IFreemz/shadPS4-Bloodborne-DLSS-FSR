// SPDX-FileCopyrightText: Copyright 2026 IFreemz
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <imgui.h>

#include "common/path_util.h"
#include "core/emulator_settings.h"
#include "core/file_sys/bb_hdr_compat.h"
#include "imgui/dlss_layer.h"
#include "imgui/imgui_layer.h"
#include "imgui/renderer/imgui_core.h"
#include "video_core/renderer_vulkan/vk_bb_frame_gen.h"
#include "video_core/renderer_vulkan/vk_bb_temporal_dlss.h"
#include "video_core/renderer_vulkan/vk_hdr_meter.h"
#include "video_core/renderer_vulkan/vk_hdr_mod.h"
#include "video_core/renderer_vulkan/vk_retro_hdr.h"

namespace ImGui::Dlss {
namespace {

// The panel edits the same settings file the renderer polls, so it needs no renderer access.
class SettingsFile {
public:
    void Load() {
        lines.clear();
        std::ifstream file{Path()};
        for (std::string line; std::getline(file, line);)
            lines.push_back(line);
    }
    std::string Get(const std::string& key, const std::string& fallback) const {
        for (const auto& line : lines)
            if (line.rfind(key + "=", 0) == 0)
                return line.substr(key.size() + 1);
        return fallback;
    }
    void Set(const std::string& key, const std::string& value) {
        Load(); // keep edits made to the file while the panel was open
        for (auto& line : lines)
            if (line.rfind(key + "=", 0) == 0) {
                line = key + "=" + value;
                return Save();
            }
        lines.push_back(key + "=" + value);
        Save();
    }

private:
    static std::filesystem::path Path() {
        return Vulkan::BbDlssSettingsPath();
    }
    void Save() const {
        std::ofstream file{Path(), std::ios::trunc};
        for (const auto& line : lines)
            file << line << '\n';
    }
    std::vector<std::string> lines;
};

constexpr std::array RecommendedPatches{"Disable AA", "Disable Chromatic Aberration",
                                        "Disable DoF"};
// Removes the depth and motion data the upscaler needs; the full name ends in "(Perf Increase)".
constexpr char MotionBlurPatch[] = "Disable Motion Blur[^\"]*";

// With RenoDX HDR, Disable AA turns skin blue: its HDR output expects what the game's AA pass
// leaves in the image.
bool Wanted(size_t i) {
    return !(Vulkan::RenoDxLoaded() && i < 2);
}

std::filesystem::path PatchFile() {
    return Common::FS::GetUserPath(Common::FS::PathType::PatchesDir) / "shadPS4" / "Bloodborne.xml";
}

std::string ReadAll(const std::filesystem::path& path) {
    std::ifstream file{path, std::ios::binary};
    std::ostringstream text;
    text << file.rdbuf();
    return text.str();
}

std::regex PatchPattern(const char* name) {
    return std::regex{std::string{"(<Metadata [^>]*Name=\""} + name +
                      "\"[^>]*isEnabled=\")(true|false)(\")"};
}

// Patches in the shadPS4 patch file, for the recommended-visuals check.
struct PatchState {
    bool file_found{};
    std::array<bool, RecommendedPatches.size()> enabled{};
    bool motion_blur_disabled{};
};

PatchState ReadPatches() {
    PatchState state;
    const auto text = ReadAll(PatchFile());
    state.file_found = !text.empty();
    for (size_t i = 0; i < RecommendedPatches.size(); ++i) {
        std::smatch match;
        state.enabled[i] = std::regex_search(text, match, PatchPattern(RecommendedPatches[i])) &&
                           match[2] == "true";
    }
    std::smatch match;
    state.motion_blur_disabled =
        std::regex_search(text, match, PatchPattern(MotionBlurPatch)) && match[2] == "true";
    return state;
}

bool EnableRecommendedPatches() {
    auto text = ReadAll(PatchFile());
    if (text.empty())
        return false;
    for (size_t i = 0; i < RecommendedPatches.size(); ++i)
        text = std::regex_replace(text, PatchPattern(RecommendedPatches[i]),
                                  Wanted(i) ? "$1true$3" : "$1false$3");
    text = std::regex_replace(text, PatchPattern(MotionBlurPatch), "$1false$3");
    std::ofstream file{PatchFile(), std::ios::binary | std::ios::trunc};
    file << text;
    return bool(file);
}

constexpr std::array UpscalerValues{"auto", "dlss", "fsr", "fsr4"};
constexpr std::array UpscalerNames{"Automatic", "DLSS", "FSR 3.1", "FSR 4 (add-on)"};
constexpr std::array PresetValues{13, 11, 10, 12, 0};
constexpr std::array PresetNames{"M (recommended)", "K", "J", "L", "NVIDIA default"};

class DlssLayer final : public ImGui::Layer {
public:
    void Draw() override;
    bool ShouldKeepDrawing() override {
        return open || toggle_requested;
    }
    void Open() {
        settings.Load();
        patches = ReadPatches();
        patch_message.clear();
        open = true;
        focus = true;
        // Like shadPS4's other prompts: the game stops reading the controller while the panel
        // is open, and the mouse cursor is shown.
        ImGui::Core::AcquireGamepadInputCapture();
        GetIO().MouseDrawCursor = true;
    }
    void Close() {
        if (!open)
            return;
        open = false;
        ImGui::Core::ReleaseGamepadInputCapture();
        GetIO().MouseDrawCursor = false;
    }
    bool open{};
    bool focus{};
    bool registered{};
    std::atomic<bool> toggle_requested{}; // set by the input thread, handled while drawing

private:
    SettingsFile settings;
    PatchState patches;
    std::string patch_message;
};

DlssLayer layer;
std::atomic<float> game_x{}, game_y{};

// The game's frame rate in a corner, and the displayed one while frame generation runs. Drawn
// into the game image, so it does not stop the game reaching the screen directly the way
// overlay windows do.
class CounterLayer final : public ImGui::Layer {
public:
    void Draw() override {
        if (const auto hdr = Vulkan::HdrMeter::Get(); hdr.valid) {
            // Debug readout (SHADPS4_HDR_METER=1), below the fps counter.
            SetNextWindowPos({game_x + 12.0f, game_y + 60.0f});
            SetNextWindowBgAlpha(1.0f);
            if (Begin("##bb_hdr_meter", nullptr,
                      ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
                          ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoNav |
                          ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoSavedSettings |
                          ImGuiWindowFlags_NoDocking))
                Text("HDR nits: peak %.0f  99.9%% %.0f  99%% %.0f  median %.0f", hdr.peak, hdr.p999,
                     hdr.p99, hdr.median);
            End();
        }
        if (!Vulkan::FrameGen::CounterVisible())
            return;
        const auto stats = Vulkan::FrameGen::GetStats();
        // Inside the game image: frame generation leaves the bars around it alone.
        SetNextWindowPos({game_x + 12.0f, game_y + 12.0f});
        // Opaque: frame generation keeps the whole box still, so no scene may show through it.
        SetNextWindowBgAlpha(1.0f);
        if (Begin("##bb_fps_counter", nullptr,
                  ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
                      ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoNav |
                      ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoSavedSettings |
                      ImGuiWindowFlags_NoDocking)) {
            Text("%.0f fps", stats.base_fps);
            if (stats.generating) {
                SameLine();
                TextColored({0.4f, 1.0f, 0.4f, 1.0f}, "-> %.0f fps", stats.output_fps);
            }
        }
        End();
    }
    // It changes with the game's frames, so it never asks for repeated presents: those would
    // switch frame generation off at 30 fps.
    bool ShouldKeepDrawing() override {
        return false;
    }
};

CounterLayer counter;

void DlssLayer::Draw() {
    if (toggle_requested.exchange(false)) {
        if (open)
            Close();
        else
            Open();
    }
    if (!open)
        return;
    if (focus) {
        SetNextWindowFocus();
        focus = false;
    }
    SetNextWindowSize({460, 0}, ImGuiCond_Appearing);
    SetNextWindowPos(GetMainViewport()->GetCenter(), ImGuiCond_Appearing, {0.5f, 0.5f});
    bool keep_open = true;
    if (!Begin("Upscaling  (F1 / Esc to close)", &keep_open,
               ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDocking |
                   ImGuiWindowFlags_AlwaysAutoResize)) {
        End();
        return;
    }
    if (!keep_open || (IsWindowFocused() && IsKeyPressed(ImGuiKey_Escape, false))) {
        End();
        Close();
        return;
    }
    const auto status = Vulkan::BbTemporalDlssStatus();
    if (status.active) {
        TextColored({0.4f, 1.0f, 0.4f, 1.0f}, "Active");
        SameLine();
        Text("%s  %ux%u -> %ux%u", status.backend.c_str(), status.render_width,
             status.render_height, status.output_width, status.output_height);
    } else {
        TextColored({1.0f, 0.75f, 0.3f, 1.0f}, "Not active");
        if (!status.reason.empty())
            TextWrapped("%s", status.reason.c_str());
    }
    if (!status.gpu.empty())
        TextDisabled("GPU: %s", status.gpu.c_str());
    if (patches.motion_blur_disabled) {
        PushStyleColor(ImGuiCol_Text, ImVec4{1.0f, 0.4f, 0.4f, 1.0f});
        TextWrapped("The Disable Motion Blur patch is on. It removes the depth and motion data "
                    "the upscaler needs: turn it off below and restart the game. To get rid of "
                    "motion blur, untick Game motion blur instead.");
        PopStyleColor();
    }

    if (!Wanted(0) && patches.enabled[0] &&
        !::Core::FileSys::BbHdrCompat::ShouldSkipPatch(RecommendedPatches[0])) {
        PushStyleColor(ImGuiCol_Text, ImVec4{1.0f, 0.4f, 0.4f, 1.0f});
        TextWrapped("The Disable AA patch is on. With RenoDX HDR it makes skin look blue: turn it "
                    "off below and restart the game.");
        PopStyleColor();
    }

    if (EmulatorSettings.GetReadbacksMode() == 0) {
        PushStyleColor(ImGuiCol_Text, ImVec4{1.0f, 0.4f, 0.4f, 1.0f});
        TextWrapped("GPU readbacks are off. Bloodborne then shows stretched polygons and missing "
                    "parts of the world at random (with or without upscaling). Press F3, go to "
                    "Experimental and set Readbacks Mode to Relaxed, then restart the game.");
        PopStyleColor();
    }

    SeparatorText("Settings");
    bool enabled = settings.Get("enabled", "1") != "0";
    if (Checkbox("Upscaling enabled", &enabled))
        settings.Set("enabled", enabled ? "1" : "0");

    const auto upscaler = settings.Get("upscaler", "auto");
    int upscaler_index = 0;
    for (size_t i = 0; i < UpscalerValues.size(); ++i)
        if (upscaler.starts_with(UpscalerValues[i]))
            upscaler_index = int(i); // "fsr4" also starts with "fsr" and comes later
    // FSR 4 is listed only when its add-on is installed.
    const int upscaler_count = int(UpscalerNames.size()) - (status.fsr4_installed ? 0 : 1);
    if (Combo("Upscaler", &upscaler_index, UpscalerNames.data(), upscaler_count))
        settings.Set("upscaler", UpscalerValues[upscaler_index]);
    if (!status.dlss_problem.empty() && upscaler_index != 2)
        TextDisabled("%s", status.dlss_problem.c_str());

    const int preset = std::atoi(settings.Get("preset", "13").c_str());
    int preset_index = 0;
    for (size_t i = 0; i < PresetValues.size(); ++i)
        if (PresetValues[i] == preset)
            preset_index = int(i);
    BeginDisabled(status.backend != "DLSS" || upscaler_index >= 2);
    if (Combo("DLSS model", &preset_index, PresetNames.data(), int(PresetNames.size())))
        settings.Set("preset", std::to_string(PresetValues[preset_index]));
    EndDisabled();

    float sharpness = float(std::atof(settings.Get("sharpness", "0.6").c_str()));
    if (SliderFloat("Sharpness", &sharpness, 0.0f, 1.0f, "%.2f"))
        settings.Set("sharpness", fmt::format("{:.2f}", sharpness));

    if (Vulkan::RenoDxLoaded()) {
        bool hdr_compat = settings.Get("hdr_mod_compat", "1") != "0";
        if (Checkbox("Remaster HDR compatibility (restart)", &hdr_compat))
            settings.Set("hdr_mod_compat", hdr_compat ? "1" : "0");
        BeginDisabled(!hdr_compat);
        bool moon_restore = settings.Get("hdr_moon_restore", "0") == "1";
        if (Checkbox("Restore original moon lighting (restart)", &moon_restore))
            settings.Set("hdr_moon_restore", moon_restore ? "1" : "0");
        EndDisabled();
    }

    bool frame_gen = settings.Get("frame_gen", "0") != "0";
    if (Checkbox("Frame generation", &frame_gen))
        settings.Set("frame_gen", frame_gen ? "1" : "0");
    if (!Vulkan::FrameGen::Active()) {
        const auto problem = Vulkan::FrameGen::Problem();
        if (frame_gen && !problem.empty()) {
            TextDisabled("%s", problem.c_str());
        } else {
            SameLine();
            TextDisabled(frame_gen ? "(restart the game to turn it on)"
                                   : "(DLSS on RTX 40/50, FSR on other GPUs)");
        }
    } else {
        const auto stats = Vulkan::FrameGen::GetStats();
        SameLine();
        if (stats.generating)
            TextColored({0.4f, 1.0f, 0.4f, 1.0f}, "%s  %.0f -> %.0f fps",
                        Vulkan::FrameGen::BackendName(), stats.base_fps, stats.output_fps);
        else
            TextDisabled("%s  %.0f fps, not generating", Vulkan::FrameGen::BackendName(),
                         stats.base_fps);
    }

    // Which frame generation runs; Streamline or FSR's swapchain is set up at launch.
    {
        static constexpr std::array BackendNames{"Automatic", "DLSS (RTX 40/50)", "FSR"};
        static constexpr std::array BackendValues{"auto", "dlss", "fsr"};
        const auto backend = settings.Get("fg_backend", "auto");
        int index = 0;
        for (size_t i = 0; i < BackendValues.size(); ++i)
            if (backend == BackendValues[i])
                index = int(i);
        SetNextItemWidth(200.0f);
        if (Combo("Frame gen type", &index, BackendNames.data(), int(BackendNames.size())))
            settings.Set("fg_backend", BackendValues[index]);
        if (index != 2 && std::string_view{Vulkan::FrameGen::BackendName()} == "FSR" &&
            Vulkan::RenoDxLoaded())
            TextDisabled("RenoDX HDR is loaded: DLSS frame generation does not work with its HDR "
                         "output, FSR is used.");
        SameLine();
        TextDisabled("(restart to apply)");
    }
    if (Vulkan::FrameGen::Active()) {
        // 3x and up only where Streamline offers them (RTX 50 series).
        const int max = int(Vulkan::FrameGen::MaxMultiplier());
        const int chosen = std::clamp(std::atoi(settings.Get("fg_multiplier", "2").c_str()), 2, 6);
        if (max > 2 || chosen > 2) {
            int index = std::min(chosen, max) - 2;
            static constexpr std::array Names{"2x", "3x", "4x", "5x", "6x"};
            SetNextItemWidth(120.0f);
            if (Combo("Frame gen multiplier", &index, Names.data(), max - 1))
                settings.Set("fg_multiplier", std::to_string(index + 2));
            if (chosen > max) {
                SameLine();
                TextDisabled("(%dx needs an RTX 50 card)", chosen);
            }
        }
    }

    // NVIDIA Reflex's wait before each frame: less input delay with frame generation, but on
    // some setups a lower base frame rate. Takes effect within a second.
    bool reflex_sleep = settings.Get("fg_reflex_sleep", "0") != "0";
    BeginDisabled(std::string_view{Vulkan::FrameGen::BackendName()} != "DLSS");
    if (Checkbox("Lower input delay (Reflex)", &reflex_sleep))
        settings.Set("fg_reflex_sleep", reflex_sleep ? "1" : "0");
    EndDisabled();
    SameLine();
    TextDisabled("(less input delay, may lower fps)");

    bool fps_counter = settings.Get("fps_counter", "0") != "0";
    if (Checkbox("Frame gen FPS counter", &fps_counter))
        settings.Set("fps_counter", fps_counter ? "1" : "0");
    SameLine();
    TextDisabled("(top-left corner)");

    bool motion_blur = settings.Get("motion_blur", "0") != "0";
    if (Checkbox("Game motion blur", &motion_blur))
        settings.Set("motion_blur", motion_blur ? "1" : "0");
    SameLine();
    TextDisabled("(off = sharper in motion)");

    bool menu_fix = settings.Get("menu_fix", "1") != "0";
    if (Checkbox("Menu fix", &menu_fix))
        settings.Set("menu_fix", menu_fix ? "1" : "0");
    if (IsItemHovered())
        SetTooltip("While a menu covers most of the screen (inventory, pause menu), show the "
                   "game's own image\nso the scene behind it doesn't shimmer. Large popups can "
                   "trigger it too.\nThe game's performance is the same either way.");
    SameLine();
    TextDisabled("(steady menu backgrounds)");

    SeparatorText("SDR to HDR");
    namespace RetroHdr = ::Vulkan::RetroHdr;
    const auto& retro_hdr = RetroHdr::GetSession();
    BeginDisabled(retro_hdr.state != RetroHdr::State::Active &&
                  retro_hdr.state != RetroHdr::State::Disabled);
    bool retro_hdr_on = settings.Get("retro_hdr", "0") != "0";
    if (Checkbox("SDR to HDR", &retro_hdr_on))
        settings.Set("retro_hdr", retro_hdr_on ? "1" : "0");
    SameLine();
    if (retro_hdr.state == RetroHdr::State::Active)
        TextDisabled(retro_hdr_on ? "(untick to compare with SDR)" : "(showing SDR)");
    else if (retro_hdr.state == RetroHdr::State::Disabled)
        TextDisabled(retro_hdr_on ? "(restart the game to turn it on)"
                                  : "(Special K style HDR from the SDR image)");
    else
        TextDisabled("(%s)", RetroHdr::Describe(retro_hdr.state).data());

    int peak_override = std::atoi(settings.Get("retro_hdr_peak", "0").c_str());
    const float peak = RetroHdr::PeakNits(float(peak_override));
    const float max_brightness = RetroHdr::MaxBrightness(peak);
    float brightness = RetroHdr::ParseFloat(settings.Get("retro_hdr_brightness", "2.00"),
                                            RetroHdr::DefaultBrightness);
    brightness = std::clamp(brightness, RetroHdr::MinBrightness, max_brightness);
    if (SliderFloat("HDR brightness", &brightness, RetroHdr::MinBrightness, max_brightness,
                    "%.2fx"))
        settings.Set("retro_hdr_brightness", fmt::format("{:.2f}", brightness));
    TextDisabled("white %.0f nits, 18%% gray %.0f nits", brightness * RetroHdr::BoostedNits(1.0f),
                 brightness * RetroHdr::BoostedNits(0.18f));
    if (SliderInt("HDR peak", &peak_override, 0, 4000, peak_override < 100 ? "auto" : "%d nits"))
        settings.Set("retro_hdr_peak", std::to_string(peak_override < 100 ? 0 : peak_override));
    SameLine();
    if (retro_hdr.windows_peak_nits >= 100.0f)
        TextDisabled("(Windows reports %.0f nits)", retro_hdr.windows_peak_nits);
    else
        TextDisabled("(no Windows value, %.0f nits used)", peak);
    EndDisabled();

    SeparatorText("Resolution");
    TextWrapped("The game's render resolution is upscaled to your shadPS4 window size. The "
                "render resolution comes from the Resolution Patch you enable for Bloodborne "
                "(shadPS4 patches or BBLauncher): for a 4K window, the 1440p patch is Quality "
                "mode and the 1080p patch is Performance mode (faster, softer). A render "
                "resolution equal to the window only anti-aliases (DLAA / FSR Native AA).");

    SeparatorText("Recommended game patches");
    if (!patches.file_found) {
        TextWrapped("Bloodborne patch file not found. Download patches in shadPS4 first.");
    } else {
        bool all = true;
        for (size_t i = 0; i < RecommendedPatches.size(); ++i) {
            const bool skipped =
                patches.enabled[i] &&
                ::Core::FileSys::BbHdrCompat::ShouldSkipPatch(RecommendedPatches[i]);
            const bool good = (patches.enabled[i] && !skipped) == Wanted(i);
            all &= good;
            TextColored(good ? ImVec4{0.4f, 1.0f, 0.4f, 1.0f} : ImVec4{1.0f, 0.75f, 0.3f, 1.0f},
                        "%s  %s%s", patches.enabled[i] && !skipped ? "on " : "off",
                        RecommendedPatches[i],
                        skipped     ? " (ignored for HDR)"
                        : Wanted(i) ? ""
                                    : " (must be off with RenoDX HDR)");
        }
        if (patches.motion_blur_disabled) {
            all = false;
            TextColored({1.0f, 0.4f, 0.4f, 1.0f}, "on   Disable Motion Blur (must be off)");
        }
        if (!all) {
            TextWrapped("The game's own anti-aliasing, chromatic aberration and depth of field "
                        "blur the image before the upscaler sees it.");
            if (Button("Apply recommended patches")) {
                patch_message = EnableRecommendedPatches() ? "Done. Restart the game to apply."
                                                           : "Could not write the patch file.";
                patches = ReadPatches();
            }
        }
        if (!patch_message.empty())
            TextWrapped("%s", patch_message.c_str());
    }
    End();
}

} // namespace

void Register() {
    if (!layer.registered) {
        ImGui::Layer::AddLayer(&layer);
        ImGui::Layer::AddLayer(&counter);
        layer.registered = true;
    }
}

void Unregister() {
    if (layer.registered) {
        ImGui::Layer::RemoveLayer(&counter);
        ImGui::Layer::RemoveLayer(&layer);
        layer.registered = false;
    }
}

void Toggle() {
    layer.toggle_requested = true;
}

void SetGameArea(float x, float y) {
    game_x = x;
    game_y = y;
}

} // namespace ImGui::Dlss
