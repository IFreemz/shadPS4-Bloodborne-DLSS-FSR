// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace Core::BbMouseCamera {

// Native mouse camera for Bloodborne: mouse movement turns the follow camera by an angle per count,
// added to the camera's pitch and yaw where the game stores them after the stick's turn, instead
// of the mouse pretending to be the right stick.

// Game thread, after the eboot is loaded and patched: finds the camera's store and redirects it.
void Install();
// Whether the camera code was found and redirected.
bool Installed();
// Input thread: relative mouse motion while the mouse turns the camera.
void AddMotion(float dx, float dy);

} // namespace Core::BbMouseCamera
