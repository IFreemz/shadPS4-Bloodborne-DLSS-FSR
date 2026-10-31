// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Libraries::Pad {
struct OrbisPadData;
}

namespace Input::BbRoute {

// Benchmark routes: the controller state (keyboard included) and the mouse camera's turns are
// recorded once and played back for runs that repeat the same movement, with the frame rate of
// each replay measured. The clock stops on loading screens and while F1 or another window holds
// the controller, so loads of another length do not shift the inputs after them. Idea from bbport
// (BB_PAD_RECORD / BB_PAD_REPLAY, https://github.com/deadinside28/bloodborne_pc).

enum class Mode { Idle, Recording, Armed, Playing };

struct Result {
    bool valid{};
    double seconds{};
    u32 frames{};
    double average_fps{};
    double low_fps{}; // the slowest 1% of frames
    double worst_ms{};
    u32 stutters{}; // frames over twice the median
};

void StartRecording();
void StopRecording();
// Plays the saved route from the next moment the game has the controller (F1 closed).
void Play();
void Stop();

Mode GetMode();
bool HasRoute();
// The route's length, and how far the recording or replay is, in seconds.
double RouteSeconds();
double ElapsedSeconds();
Result LastResult();

// Every pad read, after the state is filled (the game's thread).
void OnPadRead(Libraries::Pad::OrbisPadData* data, int num, bool intercepted);
// The mouse camera's motion (the input thread). False while a replay turns the camera: the
// motion is dropped.
bool OnMouseMotion(float dx, float dy);
// Every frame the game submits.
void OnGameFrame();

} // namespace Input::BbRoute
