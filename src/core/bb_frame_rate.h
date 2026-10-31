// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-FileCopyrightText: Copyright 2026 bbport contributors (deadinside28/bloodborne_pc)
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string_view>

#include "common/types.h"

namespace Core::BbFrameRate {

// Bloodborne 1.09's frame rate set free by hooks in its own code instead of the community's frame
// rate patches (from bbport 0.7, https://github.com/deadinside28/bloodborne_pc): the game's frame
// pacer no longer waits for a 30 FPS frame, the game steps by each frame's real time, and what it
// ties to a 30 FPS frame (fixed steps, per-frame smoothing and damping, the run's speed check,
// the cloth's stiffness) follows that time. Installed at game load when the setting is on; left
// out when the game's code differs (another version).
void Install();

// Before the patch file applies: whether the hooks will be used (the setting, Bloodborne 1.09).
void Prepare();

// Every write of every enabled patch, before any applies (patch-file address). A frame rate
// patch (one that writes the frame pacer's code the hooks replace, or the frame time constants
// they redirect) is skipped as a whole while the hooks are used, as the resolution patches are
// while F1 sets the resolution.
void CheckPatch(std::string_view name, u64 address, u64 size);
bool ShouldSkipPatch(std::string_view name);

// True when the hooks run the game's frame rate. The video out then shows a frame as soon as
// the game submits it (the game itself asks for every second vblank).
bool Active();

// The frame rate limit while the game runs (30..480; 0: none, 480). The game's own frame pacer
// waits for it, before the frame's simulation.
void SetLimit(int fps);

/// The FPS cap's frame time in seconds, 0 without a cap (or without the hooks).
float CapPeriod();

} // namespace Core::BbFrameRate
