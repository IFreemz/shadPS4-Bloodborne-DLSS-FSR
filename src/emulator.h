// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>
#include <optional>
#include <thread>

#include "common/singleton.h"
#include "core/linker.h"
#include "input/controller.h"
#include "sdl_window.h"

namespace Vulkan {
class Instance;
}

namespace Core {

using HLEInitDef = void (*)(Core::Loader::SymbolsResolver* sym);

struct SysModules {
    std::string_view module_name;
    HLEInitDef callback;
};

class Emulator {
public:
    Emulator();
    ~Emulator();

    void Run(std::filesystem::path file, std::vector<std::string> args = {},
             std::optional<std::filesystem::path> game_folder = {},
             std::vector<std::pair<std::filesystem::path, std::string>> mounts = {},
             std::vector<std::string> const& env_vars = {}, bool append_log = false);
    void UpdatePlayTime(const std::string_view serial);
    void Shutdown();
    std::unique_ptr<Vulkan::Instance> TakeEarlyVulkanInstance();

    /**
     * This will kill the current process and launch a new process with the same configuration
     * (using CLI args) but replacing the eboot image and guest arguments
     */
    void Restart(std::filesystem::path eboot_path, const std::vector<std::string>& guest_args = {},
                 bool standalone = false);

    /**
     * Launches a new emulator process with the supplied CLI arguments, then terminates this
     * process. The new process waits for this one to exit before initializing. Standalone: the
     * new process starts itself even under a launcher (IPC), without the launcher's link.
     */
    [[noreturn]] void Relaunch(std::vector<std::string> args, bool standalone = false);

    const char* executableName;
    bool waitForDebuggerBeforeRun{false};

private:
    void LoadSystemModules(const std::string& game_serial);

    Core::MemoryManager* memory;
    Input::GameControllers* controllers;
    Core::Linker* linker;
    std::unique_ptr<Frontend::WindowSDL> window;
    std::unique_ptr<Vulkan::Instance> early_vk_instance;
    std::chrono::steady_clock::time_point start_time;
    std::jthread play_time_thread;
};

} // namespace Core
