// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <source_location>
#include <fmt/format.h>

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/string_literal.h"
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Frontend {
enum class WindowSystemType : u8;
class WindowSDL;
} // namespace Frontend

namespace Vulkan {

class DlssNgx;

constexpr u32 TargetVulkanApiVersion = VK_API_VERSION_1_3;

vk::SurfaceKHR CreateSurface(vk::Instance instance, const Frontend::WindowSDL& emu_window);

vk::UniqueInstance CreateInstance(Frontend::WindowSystemType window_type, bool enable_validation,
                                  bool enable_crash_diagnostic, DlssNgx* dlss_ngx = nullptr);

vk::UniqueDebugUtilsMessengerEXT CreateDebugCallback(vk::Instance instance);

template <typename T>
concept VulkanHandleType = vk::isVulkanHandleType<T>::value;

template <VulkanHandleType HandleType>
void SetObjectName(vk::Device device, const HandleType& handle, std::string_view debug_name) {
    const vk::DebugUtilsObjectNameInfoEXT name_info = {
        .objectType = HandleType::objectType,
        .objectHandle = reinterpret_cast<u64>(static_cast<typename HandleType::NativeType>(handle)),
        .pObjectName = debug_name.data(),
    };
    auto result = device.setDebugUtilsObjectNameEXT(name_info);
    if (result != vk::Result::eSuccess) {
        LOG_DEBUG(Render_Vulkan, "Could not set object debug name: {}", vk::to_string(result));
    }
}

template <VulkanHandleType HandleType, typename... Args>
void SetObjectName(vk::Device device, const HandleType& handle, const char* format,
                   const Args&... args) {
    const std::string debug_name = fmt::vformat(format, fmt::make_format_args(args...));
    SetObjectName(device, handle, debug_name);
}

template <StringLiteral msg = "">
static void Check(vk::Result r, std::source_location caller = std::source_location::current()) {
    if constexpr (msg.len <= 1) {
        ASSERT_MSG(r == vk::Result::eSuccess, "vk::Result={} (checked at {}:{} in {})",
                   vk::to_string(r), caller.file_name(), caller.line(), caller.function_name());
    } else {
        ASSERT_MSG(r == vk::Result::eSuccess,
                   "Failed to {}: vk::Result={} (checked at {}:{} in {})", msg.value,
                   vk::to_string(r), caller.file_name(), caller.line(), caller.function_name());
    }
}

template <StringLiteral msg = "", typename T>
static T Check(vk::ResultValue<T> r,
               std::source_location caller = std::source_location::current()) {
    if constexpr (msg.len <= 1) {
        ASSERT_MSG(r.result == vk::Result::eSuccess, "vk::Result={} (checked at {}:{} in {})",
                   vk::to_string(r.result), caller.file_name(), caller.line(),
                   caller.function_name());
    } else {
        ASSERT_MSG(r.result == vk::Result::eSuccess,
                   "Failed to {}: vk::Result={} (checked at {}:{} in {})", msg.value,
                   vk::to_string(r.result), caller.file_name(), caller.line(),
                   caller.function_name());
    }
    return std::move(r.value);
}

} // namespace Vulkan
