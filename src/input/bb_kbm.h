// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>
#include <string>
#include <string_view>

namespace Input::BbKbm {

// Bloodborne keyboard and mouse controls: a generated input config (default.ini with the unified
// input config, otherwise the game's) with the mouse as the camera. Bindings live in dlss.ini as
// kbm_<action>=<key>,<mouse> and are rebindable in F1.

struct Action {
    const char* id;    // dlss.ini key: kbm_<id>
    const char* label; // F1
    const char* output; // shadPS4 output (PS4 button or stick direction)
    const char* default_key;
    const char* default_mouse;
};
std::span<const Action> Actions();

struct Binding {
    std::string key;   // shadPS4 key name, empty: none
    std::string mouse; // leftbutton, rightbutton, middlebutton, sidebuttonback,
                       // sidebuttonforward, mousewheelup, mousewheeldown; empty: none
};
// The action's binding from dlss.ini, or its default.
Binding GetBinding(const Action& action);
void SetBinding(const Action& action, const Binding& binding);
void ResetBindings();

constexpr float DefaultSensitivity = 1.5f;
constexpr float MinSensitivity = 0.3f;
constexpr float MaxSensitivity = 5.0f;

// Writes the controls (the previous file is kept as <name>.bak) and reloads the input config.
bool Apply();
// Puts the file from before Apply back (or shadPS4's default bindings) and reloads.
bool Restore();
// Whether the active input config is ours.
bool Active();
// Whether mouse look should be on while playing (ours active and not turned off in F1 or F7).
bool MouseLookWanted();
// F1's Mouse look switch and F7: remembered in dlss.ini.
void SetMouseLook(bool on);
// Any thread: captures the mouse as the camera, or releases it.
void RequestMouseLook(bool on);

// Display name of a key or mouse input ("Left Shift", "Mouse 4").
std::string DisplayName(std::string_view name);

} // namespace Input::BbKbm
