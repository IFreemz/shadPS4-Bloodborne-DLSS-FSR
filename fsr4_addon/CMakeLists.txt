# SPDX-FileCopyrightText: Copyright 2026 IFreemz
# SPDX-License-Identifier: MIT
#
# shadps4_fsr4.dll (Windows) / shadps4_fsr4.so (Linux), the optional FSR 4 add-on. It links
# FireBurn's FSR-Vulkan (MIT):
#   cmake -S fsr4_addon -B build-fsr4 -DFSR_VULKAN_ROOT=<path to FSR-Vulkan>
#   cmake --build build-fsr4 --config Release
# The Vulkan loader import library comes from the Vulkan SDK, or VULKAN_LIBRARY.

cmake_minimum_required(VERSION 3.20)
project(shadps4_fsr4 LANGUAGES C CXX)

set(FSR_VULKAN_ROOT "" CACHE PATH "FireBurn/FSR-Vulkan source tree")
set(VULKAN_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/../externals/vulkan-headers/include"
    CACHE PATH "Vulkan-Headers include directory")
set(VULKAN_LIBRARY "" CACHE FILEPATH "vulkan-1 import library (default: Vulkan SDK)")

if (NOT CMAKE_SIZEOF_VOID_P EQUAL 8 OR NOT (WIN32 OR CMAKE_SYSTEM_NAME STREQUAL "Linux"))
    message(FATAL_ERROR "The FSR 4 add-on is Windows x64 and Linux x64 only")
endif()
if (NOT EXISTS "${FSR_VULKAN_ROOT}/include/ffx_vk_fsr4_v07.h")
    message(FATAL_ERROR "Set FSR_VULKAN_ROOT to FSR-Vulkan (github.com/FireBurn/FSR-Vulkan)")
endif()
if (NOT VULKAN_LIBRARY)
    find_package(Vulkan REQUIRED)
    set(VULKAN_LIBRARY Vulkan::Vulkan)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")

add_library(shadps4_fsr4 SHARED
    shadps4_fsr4.cpp
    shadps4_fsr4.h
    "${FSR_VULKAN_ROOT}/src/ffx_vk_fsr4_v07.c"
    "${FSR_VULKAN_ROOT}/src/ffx_vk_fsr4_v07_assets.c"
    "${FSR_VULKAN_ROOT}/src/ffx_vk_fsr4_v07_provider.c"
    "${FSR_VULKAN_ROOT}/src/ffx_vk_fsr4_v07_schedule.c")
target_include_directories(shadps4_fsr4 PRIVATE "${FSR_VULKAN_ROOT}/include" "${VULKAN_HEADERS}")
target_link_libraries(shadps4_fsr4 PRIVATE "${VULKAN_LIBRARY}")
target_compile_definitions(shadps4_fsr4 PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN
                           _CRT_SECURE_NO_WARNINGS)
if (NOT WIN32)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/exports.map" "{ global: ShadFsr4GetApi; local: *; };")
    set_target_properties(shadps4_fsr4 PROPERTIES PREFIX "" C_VISIBILITY_PRESET hidden
                          CXX_VISIBILITY_PRESET hidden)
    target_link_options(shadps4_fsr4 PRIVATE -Wl,--no-undefined -Wl,--exclude-libs,ALL
                        "-Wl,--version-script=${CMAKE_CURRENT_BINARY_DIR}/exports.map"
                        -static-libstdc++ -static-libgcc)
endif()
