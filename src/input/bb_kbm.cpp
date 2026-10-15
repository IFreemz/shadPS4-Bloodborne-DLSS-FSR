// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <fmt/format.h>

#include "common/elf_info.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "core/emulator_settings.h"
#include "input/bb_kbm.h"
#include "input/input_handler.h"

namespace Input::BbKbm {
namespace {

constexpr char Marker[] = "# Bloodborne keyboard and mouse";

// Souls-style PC layout. Every action is a PS4 button or stick direction the game already uses.
constexpr std::array ActionTable{
    Action{"move_forward", "Move forward", "axis_left_y_minus", "w", ""},
    Action{"move_back", "Move back", "axis_left_y_plus", "s", ""},
    Action{"move_left", "Move left", "axis_left_x_minus", "a", ""},
    Action{"move_right", "Move right", "axis_left_x_plus", "d", ""},
    Action{"walk", "Walk (hold)", "leftjoystick_halfmode", "lalt", ""},
    Action{"dodge", "Dodge / sprint", "circle", "space", ""},
    Action{"l3", "L3", "l3", "lctrl", ""},
    Action{"attack", "Attack", "r1", "", "leftbutton"},
    Action{"strong_attack", "Strong attack", "r2", "v", ""},
    Action{"strong_modifier", "Strong attack modifier (with Attack)", "", "lshift", ""},
    Action{"firearm", "Firearm / parry", "l2", "", "rightbutton"},
    Action{"transform", "Transform weapon", "l1", "c", "middlebutton"},
    Action{"lock_on", "Lock on / reset camera", "r3", "q", ""},
    Action{"blood_vial", "Blood Vial", "triangle", "r", ""},
    Action{"use_item", "Use item", "square", "f", ""},
    Action{"interact", "Interact", "cross", "e", ""},
    Action{"confirm", "Confirm (menus)", "cross", "enter", ""},
    Action{"back", "Back (menus)", "circle", "backspace", ""},
    Action{"pad_up", "D-pad up (menus, spells)", "pad_up", "up", ""},
    Action{"pad_down", "D-pad down (quick item)", "pad_down", "down", ""},
    Action{"pad_left", "D-pad left (left hand)", "pad_left", "left", "mousewheelup"},
    Action{"pad_right", "D-pad right (right hand)", "pad_right", "right", "mousewheeldown"},
    Action{"gestures", "Gestures", "touchpad_center", "g", ""},
    Action{"menu", "Menu", "options", "escape", ""},
    Action{"camera_up", "Camera up (keys)", "axis_right_y_minus", "i", ""},
    Action{"camera_down", "Camera down (keys)", "axis_right_y_plus", "k", ""},
    Action{"camera_left", "Camera left (keys)", "axis_right_x_minus", "j", ""},
    Action{"camera_right", "Camera right (keys)", "axis_right_x_plus", "l", ""},
};

constexpr char Controller[] = R"(
# Controller
triangle = triangle
cross = cross
square = square
circle = circle
l1 = l1
l2 = l2
l3 = l3
r1 = r1
r2 = r2
r3 = r3
options = options
touchpad_center = back
pad_up = pad_up
pad_down = pad_down
pad_left = pad_left
pad_right = pad_right
axis_left_x = axis_left_x
axis_left_y = axis_left_y
axis_right_x = axis_right_x
axis_right_y = axis_right_y
analog_deadzone = leftjoystick, 5, 127
analog_deadzone = rightjoystick, 5, 127
)";

std::filesystem::path SettingsPath() {
    return Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "dlss.ini";
}

std::vector<std::string> ReadLines(const std::filesystem::path& path) {
    std::vector<std::string> lines;
    std::ifstream file{path};
    for (std::string line; std::getline(file, line);)
        lines.push_back(line);
    return lines;
}

std::string GetSetting(const std::string& key, const std::string& fallback) {
    for (const auto& line : ReadLines(SettingsPath()))
        if (line.rfind(key + "=", 0) == 0)
            return line.substr(key.size() + 1);
    return fallback;
}

void SetSetting(const std::string& key, const std::string& value, bool remove = false) {
    auto lines = ReadLines(SettingsPath());
    const auto it = std::ranges::find_if(lines, [&](const std::string& line) {
        return line.rfind(key + "=", 0) == 0;
    });
    if (remove) {
        if (it != lines.end())
            lines.erase(it);
    } else if (it != lines.end()) {
        *it = key + "=" + value;
    } else {
        lines.push_back(key + "=" + value);
    }
    std::ofstream file{SettingsPath(), std::ios::trunc};
    for (const auto& line : lines)
        file << line << '\n';
}

std::filesystem::path ConfigPath() {
    const std::string id = EmulatorSettings.IsUseUnifiedInputConfig()
                               ? std::string{"default"}
                               : std::string(Common::ElfInfo::Instance().GameSerial());
    return GetInputConfigFile(id);
}

std::string Read(const std::filesystem::path& path) {
    std::ifstream file{path};
    std::stringstream text;
    text << file.rdbuf();
    return text.str();
}

float Sensitivity() {
    const std::string value = GetSetting("mouse_sensitivity", "");
    float sensitivity = DefaultSensitivity;
    if (!value.empty())
        sensitivity = static_cast<float>(std::atof(value.c_str()));
    if (!(sensitivity > 0.0f))
        sensitivity = DefaultSensitivity;
    return std::clamp(sensitivity, MinSensitivity, MaxSensitivity);
}

const Action* Find(std::string_view id) {
    for (const auto& action : ActionTable)
        if (id == action.id)
            return &action;
    return nullptr;
}

void Reload() {
    SDL_Event event{};
    event.type = SDL_EVENT_RELOAD_INPUTS;
    SDL_PushEvent(&event);
}

} // namespace

std::span<const Action> Actions() {
    return ActionTable;
}

Binding GetBinding(const Action& action) {
    const std::string value = GetSetting(fmt::format("kbm_{}", action.id), "\x01");
    if (value == "\x01")
        return {action.default_key, action.default_mouse};
    const auto comma = value.find(',');
    if (comma == std::string::npos)
        return {value, ""};
    return {value.substr(0, comma), value.substr(comma + 1)};
}

void SetBinding(const Action& action, const Binding& binding) {
    SetSetting(fmt::format("kbm_{}", action.id), binding.key + "," + binding.mouse);
}

void ResetBindings() {
    for (const auto& action : ActionTable)
        SetSetting(fmt::format("kbm_{}", action.id), "", true);
}

bool Apply() {
    const auto path = ConfigPath();
    std::error_code error;
    if (std::filesystem::exists(path, error) && !Active()) {
        auto backup = path;
        backup += ".bak";
        std::filesystem::copy_file(path, backup, std::filesystem::copy_options::overwrite_existing,
                                   error);
        if (error) {
            LOG_ERROR(Input, "Keyboard and mouse: could not back up {}: {}", path.string(),
                      error.message());
            return false;
        }
    }

    std::string text = fmt::format("{} (F1 > Keyboard & mouse; rebind there, changes here are "
                                   "replaced). Your previous file is {}.bak\n",
                                   Marker, path.filename().string());
    // The mouse as the right stick: a small floor so slow turns still move the camera.
    text += fmt::format("mouse_to_joystick = right\nmouse_movement_params = 0.15, {:.2f}, 0.05\n\n",
                        Sensitivity());
    for (const auto& action : ActionTable) {
        if (!*action.output)
            continue;
        const Binding binding = GetBinding(action);
        if (!binding.key.empty())
            text += fmt::format("{} = {}\n", action.output, binding.key);
        if (!binding.mouse.empty())
            text += fmt::format("{} = {}\n", action.output, binding.mouse);
    }
    // Strong attack as modifier + Attack (more keys win, so Attack alone stays a light attack).
    const Binding modifier = GetBinding(*Find("strong_modifier"));
    const Binding attack = GetBinding(*Find("attack"));
    if (!modifier.key.empty()) {
        if (!attack.key.empty())
            text += fmt::format("r2 = {}, {}\n", modifier.key, attack.key);
        if (!attack.mouse.empty())
            text += fmt::format("r2 = {}, {}\n", modifier.key, attack.mouse);
    }
    text += Controller;

    {
        std::ofstream file{path, std::ios::trunc};
        file << text;
        if (!file)
            return false;
    }
    LOG_INFO(Input, "Keyboard and mouse controls written to {}", path.string());
    Reload();
    return true;
}

bool Restore() {
    const auto path = ConfigPath();
    auto backup = path;
    backup += ".bak";
    std::error_code error;
    if (std::filesystem::exists(backup, error))
        std::filesystem::copy_file(backup, path, std::filesystem::copy_options::overwrite_existing,
                                   error);
    else
        std::filesystem::remove(path, error); // regenerated from shadPS4's defaults on reload
    if (error) {
        LOG_ERROR(Input, "Keyboard and mouse: could not restore {}: {}", path.string(),
                  error.message());
        return false;
    }
    Reload();
    RequestMouseLook(false);
    return true;
}

bool Active() {
    return Read(ConfigPath()).rfind(Marker, 0) == 0;
}

bool MouseLookWanted() {
    return GetSetting("mouse_look", "1").rfind('0', 0) != 0 && Active();
}

void RequestMouseLook(bool on) {
    SDL_Event event{};
    event.type = SDL_EVENT_BB_MOUSE_LOOK;
    event.user.data1 = reinterpret_cast<void*>(static_cast<intptr_t>(on ? 1 : 0));
    SDL_PushEvent(&event);
}

std::string DisplayName(std::string_view name) {
    static constexpr std::pair<std::string_view, std::string_view> Names[] = {
        {"leftbutton", "Left click"},       {"rightbutton", "Right click"},
        {"middlebutton", "Middle click"},   {"sidebuttonback", "Mouse 4"},
        {"sidebuttonforward", "Mouse 5"},   {"mousewheelup", "Wheel up"},
        {"mousewheeldown", "Wheel down"},   {"lshift", "Left Shift"},
        {"rshift", "Right Shift"},          {"lctrl", "Left Ctrl"},
        {"rctrl", "Right Ctrl"},            {"lalt", "Left Alt"},
        {"ralt", "Right Alt"},              {"space", "Space"},
        {"escape", "Esc"},                  {"enter", "Enter"},
        {"tab", "Tab"},                     {"backspace", "Backspace"},
        {"capslock", "Caps Lock"},          {"up", "Up"},
        {"down", "Down"},                   {"left", "Left"},
        {"right", "Right"},
    };
    if (name.empty())
        return "-";
    for (const auto& [id, label] : Names)
        if (name == id)
            return std::string(label);
    std::string upper(name);
    for (auto& c : upper)
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return upper;
}

} // namespace Input::BbKbm
