// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#include <fmt/format.h>

#include "common/logging/log.h"
#include "common/path_util.h"
#include "core/bb_game_fixes.h"
#include "core/bb_mouse_camera.h"
#include "core/libraries/pad/pad.h"
#include "input/bb_route.h"

namespace Input::BbRoute {
namespace {

using Clock = std::chrono::steady_clock;
using Libraries::Pad::OrbisPadButtonDataOffset;
using Libraries::Pad::OrbisPadData;

struct PadSample {
    double ms;
    u32 buttons;
    u8 lx, ly, rx, ry, l2, r2;
    bool operator==(const PadSample& other) const {
        return buttons == other.buttons && lx == other.lx && ly == other.ly && rx == other.rx &&
               ry == other.ry && l2 == other.l2 && r2 == other.r2;
    }
};
struct Motion {
    double ms;
    float dx, dy;
};

std::mutex mutex;
std::atomic<Mode> mode{Mode::Idle};
std::vector<PadSample> pads;
std::vector<Motion> motions;
bool route_loaded{};

// The route clock: milliseconds while the game shows its world and has the controller.
double world_ms{};
Clock::time_point last_read{};
bool paused{true};

size_t next_pad{};
size_t next_motion{};
double route_ms{};

std::vector<float> frame_ms;
Clock::time_point last_frame{};
bool frame_counted{};
Result result;

std::filesystem::path RouteDir() {
    return Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "routes";
}
std::filesystem::path RoutePath() {
    return RouteDir() / "route.txt";
}

bool Neutral(const PadSample& s) {
    const auto centred = [](u8 v) { return std::abs(int(v) - 128) < 24; };
    return s.buttons == 0 && s.l2 < 8 && s.r2 < 8 && centred(s.lx) && centred(s.ly) &&
           centred(s.rx) && centred(s.ry);
}

void Advance(bool hold) {
    const auto now = Clock::now();
    if (!paused && !hold)
        world_ms += std::chrono::duration<double, std::milli>(now - last_read).count();
    paused = hold;
    last_read = now;
}

void Save() {
    std::error_code ec;
    std::filesystem::create_directories(RouteDir(), ec);
    std::ofstream file{RoutePath(), std::ios::trunc};
    for (const auto& s : pads)
        file << fmt::format("p {:.1f} {} {} {} {} {} {} {}\n", s.ms, s.buttons, s.lx, s.ly, s.rx,
                            s.ry, s.l2, s.r2);
    for (const auto& m : motions)
        file << fmt::format("m {:.1f} {} {}\n", m.ms, m.dx, m.dy);
    file << fmt::format("e {:.1f}\n", world_ms);
    LOG_INFO(Input, "Route: recorded {:.1f} s ({} controller states, {} mouse turns) to {}",
             world_ms / 1000.0, pads.size(), motions.size(), RoutePath().string());
}

// The saved route, from its first input that is not neutral (the recording started while the
// player stood) to where the recording was stopped.
bool Load() {
    pads.clear();
    motions.clear();
    double end_ms = 0.0;
    std::ifstream file{RoutePath()};
    for (std::string line; std::getline(file, line);) {
        std::istringstream in{line};
        char kind{};
        in >> kind;
        if (kind == 'p') {
            PadSample s{};
            unsigned v[7]{};
            in >> s.ms >> v[0] >> v[1] >> v[2] >> v[3] >> v[4] >> v[5] >> v[6];
            if (!in)
                continue;
            s.buttons = v[0];
            s.lx = u8(v[1]), s.ly = u8(v[2]), s.rx = u8(v[3]), s.ry = u8(v[4]);
            s.l2 = u8(v[5]), s.r2 = u8(v[6]);
            pads.push_back(s);
        } else if (kind == 'm') {
            Motion m{};
            in >> m.ms >> m.dx >> m.dy;
            if (in)
                motions.push_back(m);
        } else if (kind == 'e') {
            in >> end_ms;
        }
    }
    std::ranges::sort(motions, {}, &Motion::ms);
    double start = 1e300;
    for (const auto& s : pads)
        if (!Neutral(s)) {
            start = s.ms;
            break;
        }
    if (!motions.empty())
        start = std::min(start, motions.front().ms);
    if (start == 1e300)
        start = 0.0;
    // The state before the first input stays in force until then.
    std::erase_if(pads, [&](const PadSample& s) { return s.ms < start && Neutral(s); });
    for (auto& s : pads)
        s.ms = std::max(0.0, s.ms - start);
    for (auto& m : motions)
        m.ms -= start;
    route_ms = std::max(0.0, end_ms - start);
    if (!pads.empty())
        route_ms = std::max(route_ms, pads.back().ms);
    if (!motions.empty())
        route_ms = std::max(route_ms, motions.back().ms);
    route_loaded = !pads.empty() || !motions.empty();
    return route_loaded;
}

void Finish() {
    Result r{};
    if (frame_ms.size() >= 10) {
        std::vector<float> sorted = frame_ms;
        std::ranges::sort(sorted);
        double total = 0.0;
        for (const float ms : frame_ms)
            total += ms;
        const size_t slowest = std::max<size_t>(1, sorted.size() / 100);
        double slow_total = 0.0;
        for (size_t i = sorted.size() - slowest; i < sorted.size(); ++i)
            slow_total += sorted[i];
        const float median = sorted[sorted.size() / 2];
        r.valid = true;
        r.frames = u32(frame_ms.size());
        r.seconds = total / 1000.0;
        r.average_fps = 1000.0 * double(frame_ms.size()) / total;
        r.low_fps = 1000.0 * double(slowest) / slow_total;
        r.worst_ms = sorted.back();
        r.stutters =
            u32(std::ranges::count_if(frame_ms, [&](float ms) { return ms > 2 * median; }));
        LOG_INFO(Input,
                 "Route replay: {:.1f} s, {} frames, average {:.1f} FPS, 1% low {:.1f} FPS, worst "
                 "frame {:.1f} ms, {} stutters (frames over twice the median)",
                 r.seconds, r.frames, r.average_fps, r.low_fps, r.worst_ms, r.stutters);
        std::error_code ec;
        std::filesystem::create_directories(RouteDir(), ec);
        const auto path = RouteDir() / "results.csv";
        const bool fresh = !std::filesystem::exists(path);
        std::ofstream csv{path, std::ios::app};
        if (fresh)
            csv << "time,seconds,frames,average_fps,low_1pct_fps,worst_ms,stutters\n";
        const std::time_t now = std::time(nullptr);
        char stamp[32]{};
        std::strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
        csv << fmt::format("{},{:.1f},{},{:.1f},{:.1f},{:.1f},{}\n", stamp, r.seconds, r.frames,
                           r.average_fps, r.low_fps, r.worst_ms, r.stutters);
    } else {
        LOG_INFO(Input, "Route replay: stopped before enough frames to measure");
    }
    result = r;
    mode = Mode::Idle;
}

} // namespace

void StartRecording() {
    std::scoped_lock lock{mutex};
    if (mode != Mode::Idle)
        return;
    pads.clear();
    motions.clear();
    world_ms = 0.0;
    paused = true;
    route_loaded = false;
    mode = Mode::Recording;
    LOG_INFO(Input, "Route: recording (starts when F1 is closed)");
}

void StopRecording() {
    std::scoped_lock lock{mutex};
    if (mode != Mode::Recording)
        return;
    Save();
    mode = Mode::Idle;
    Load();
}

void Play() {
    std::scoped_lock lock{mutex};
    if (mode != Mode::Idle || !Load())
        return;
    mode = Mode::Armed;
    LOG_INFO(Input, "Route: replay of {:.1f} s armed (starts when F1 is closed)",
             route_ms / 1000.0);
}

void Stop() {
    std::scoped_lock lock{mutex};
    if (mode == Mode::Playing)
        Finish();
    else if (mode == Mode::Armed)
        mode = Mode::Idle;
}

Mode GetMode() {
    return mode.load(std::memory_order_relaxed);
}

bool HasRoute() {
    std::scoped_lock lock{mutex};
    return route_loaded || std::filesystem::exists(RoutePath());
}

double RouteSeconds() {
    std::scoped_lock lock{mutex};
    if (!route_loaded && mode == Mode::Idle && std::filesystem::exists(RoutePath()))
        Load();
    return route_ms / 1000.0;
}

double ElapsedSeconds() {
    std::scoped_lock lock{mutex};
    return world_ms / 1000.0;
}

Result LastResult() {
    std::scoped_lock lock{mutex};
    return result;
}

void OnPadRead(OrbisPadData* data, int num, bool intercepted) {
    if (num <= 0 || GetMode() == Mode::Idle)
        return;
    std::vector<Motion> turns;
    {
        std::scoped_lock lock{mutex};
        const bool hold = intercepted || Core::BbGameFixes::IsLoadingScreen();
        if (mode == Mode::Armed) {
            if (hold)
                return;
            mode = Mode::Playing;
            world_ms = 0.0;
            paused = true;
            next_pad = next_motion = 0;
            frame_ms.clear();
            frame_counted = false;
            LOG_INFO(Input, "Route: replay started");
        }
        Advance(hold);
        if (mode == Mode::Recording) {
            if (hold)
                return;
            const auto& d = data[num - 1];
            const PadSample s{world_ms,           u32(d.buttons),    d.leftStick.x,
                              d.leftStick.y,      d.rightStick.x,    d.rightStick.y,
                              d.analogButtons.l2, d.analogButtons.r2};
            if (pads.empty() || !(pads.back() == s))
                pads.push_back(s);
            return;
        }
        if (mode != Mode::Playing || hold)
            return;
        while (next_pad < pads.size() && pads[next_pad].ms <= world_ms)
            ++next_pad;
        while (next_motion < motions.size() && motions[next_motion].ms <= world_ms)
            turns.push_back(motions[next_motion++]);
        if (next_pad > 0) {
            const auto& s = pads[next_pad - 1];
            for (int i = 0; i < num; ++i) {
                data[i].buttons = OrbisPadButtonDataOffset(s.buttons);
                data[i].leftStick = {s.lx, s.ly};
                data[i].rightStick = {s.rx, s.ry};
                data[i].analogButtons.l2 = s.l2;
                data[i].analogButtons.r2 = s.r2;
            }
        }
        if (world_ms >= route_ms)
            Finish();
    }
    for (const auto& t : turns)
        Core::BbMouseCamera::AddMotion(t.dx, t.dy);
}

bool OnMouseMotion(float dx, float dy) {
    const Mode now = GetMode();
    if (now == Mode::Playing)
        return false;
    if (now == Mode::Recording) {
        std::scoped_lock lock{mutex};
        if (!paused)
            motions.push_back({world_ms, dx, dy});
    }
    return true;
}

void OnGameFrame() {
    if (GetMode() != Mode::Playing)
        return;
    std::scoped_lock lock{mutex};
    const auto now = Clock::now();
    if (paused || Core::BbGameFixes::IsLoadingScreen()) {
        frame_counted = false;
        return;
    }
    if (frame_counted)
        frame_ms.push_back(std::chrono::duration<float, std::milli>(now - last_frame).count());
    last_frame = now;
    frame_counted = true;
}

} // namespace Input::BbRoute
