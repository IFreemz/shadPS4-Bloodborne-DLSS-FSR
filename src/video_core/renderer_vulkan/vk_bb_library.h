// SPDX-FileCopyrightText: Copyright 2026 IFreemz
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>

namespace Vulkan::BbLibrary {

#ifdef _WIN32
#define BB_LIBRARY_NAME(windows, linux) windows
#else
#define BB_LIBRARY_NAME(windows, linux) linux
#endif

std::filesystem::path ExecutableDirectory();

// Libraries that depend on others next to them resolve those from their own directory.
void* Load(const std::filesystem::path& path);
void* Find(void* library, const char* name);
void Free(void* library);

template <typename T>
T Find(void* library, const char* name) {
    return reinterpret_cast<T>(Find(library, name));
}

} // namespace Vulkan::BbLibrary
