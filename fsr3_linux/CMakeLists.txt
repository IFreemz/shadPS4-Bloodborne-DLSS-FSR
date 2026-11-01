# SPDX-FileCopyrightText: Copyright 2026 IFreemz
# SPDX-License-Identifier: MIT
#
# libamd_fidelityfx_vk.so: FSR 3.1 upscaling for the Linux build, the counterpart of AMD's
# amd_fidelityfx_vk.dll. AMD only ships that DLL for Windows, so this builds the same ffx-api
# from source: AMD's API layer (FidelityFX SDK v1.1.4) over the FSR 3.1.4 upscaler and Vulkan
# backend as made portable by FireBurn's FSR-Vulkan, with its pre-compiled SPIR-V.
#   cmake -S fsr3_linux -B build-fsr3 -DFFX_SDK_ROOT=<FidelityFX SDK v1.1.4> \
#         -DFSR_VULKAN_ROOT=<FSR-Vulkan>
#   cmake --build build-fsr3
# Upscaling only: AMD's frame generation swapchain is Windows code.

cmake_minimum_required(VERSION 3.20)
project(amd_fidelityfx_vk LANGUAGES CXX)

set(FFX_SDK_ROOT "" CACHE PATH "AMD FidelityFX SDK v1.1.4 (contains ffx-api/)")
set(FSR_VULKAN_ROOT "" CACHE PATH "FireBurn/FSR-Vulkan source tree")

if (WIN32)
    message(FATAL_ERROR "On Windows, use AMD's signed amd_fidelityfx_vk.dll")
endif()
if (NOT EXISTS "${FFX_SDK_ROOT}/ffx-api/src/ffx_api.cpp")
    message(FATAL_ERROR "Set FFX_SDK_ROOT to the FidelityFX SDK v1.1.4")
endif()
set(SDK "${FSR_VULKAN_ROOT}/upstream/ffx-1.1.4/sdk")
if (NOT EXISTS "${SDK}/src/backends/vk/ffx_vk.cpp")
    message(FATAL_ERROR "Set FSR_VULKAN_ROOT to FSR-Vulkan (github.com/FireBurn/FSR-Vulkan)")
endif()
if (NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "Build with Clang: AMD's sources use MSVC integer suffixes (-fms-extensions)")
endif()
find_package(Vulkan REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

# ffx-api includes the SDK headers both as FidelityFX/ and FidelityFx/.
set(CASE_LINKS "${CMAKE_CURRENT_BINARY_DIR}/case_links")
file(MAKE_DIRECTORY "${CASE_LINKS}")
file(CREATE_LINK "${SDK}/include/FidelityFX" "${CASE_LINKS}/FidelityFx" SYMBOLIC)

# AMD's backend skips host-visible types when it wants device-local memory, and fails on GPUs
# whose device-local memory is all host-visible (lavapipe, some integrated GPUs): fall back to it.
set(PATCHED_FFX_VK "${CMAKE_CURRENT_BINARY_DIR}/ffx_vk.cpp")
file(READ "${SDK}/src/backends/vk/ffx_vk.cpp" FFX_VK)
set(SKIP [[            if (requestedProperties == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT && (memoryProperties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
                continue;]])
set(SKIP_WITH_FALLBACK [[            if (requestedProperties == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT && (memoryProperties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                if (hostVisibleFallback == UINT32_MAX) {
                    hostVisibleFallback = i;
                    hostVisibleFallbackProperties = memoryProperties;
                }
                continue;
            }]])
set(BEST [[    uint32_t bestCandidate = UINT32_MAX;
]])
set(BEST_WITH_FALLBACK [[    uint32_t bestCandidate = UINT32_MAX;
    uint32_t hostVisibleFallback = UINT32_MAX;
    VkMemoryPropertyFlags hostVisibleFallbackProperties = 0;
]])
set(RETURN [[    return bestCandidate;
}]])
set(RETURN_WITH_FALLBACK [[    if (bestCandidate == UINT32_MAX && hostVisibleFallback != UINT32_MAX) {
        outProperties = hostVisibleFallbackProperties;
        return hostVisibleFallback;
    }
    return bestCandidate;
}]])
foreach(pair "SKIP;SKIP_WITH_FALLBACK" "BEST;BEST_WITH_FALLBACK" "RETURN;RETURN_WITH_FALLBACK")
    list(GET pair 0 from)
    list(GET pair 1 to)
    string(FIND "${FFX_VK}" "${${from}}" found)
    if (found EQUAL -1)
        message(FATAL_ERROR "ffx_vk.cpp changed: update the memory type fallback")
    endif()
    string(REPLACE "${${from}}" "${${to}}" FFX_VK "${FFX_VK}")
endforeach()
file(WRITE "${PATCHED_FFX_VK}.new" "${FFX_VK}")
file(COPY_FILE "${PATCHED_FFX_VK}.new" "${PATCHED_FFX_VK}" ONLY_IF_DIFFERENT)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${SDK}/src/backends/vk/ffx_vk.cpp")

set(API "${FFX_SDK_ROOT}/ffx-api")
add_library(amd_fidelityfx_vk SHARED
    ffx_provider_linux.cpp
    "${API}/src/ffx_api.cpp"
    "${API}/src/backends.cpp"
    "${API}/src/validation.cpp"
    "${API}/src/ffx_provider_fsr3upscale.cpp"
    "${SDK}/src/shared/ffx_assert.cpp"
    "${SDK}/src/shared/ffx_message.cpp"
    "${SDK}/src/shared/ffx_object_management.cpp"
    "${SDK}/src/shared/ffx_breadcrumbs_list.cpp"
    "${SDK}/src/components/fsr3upscaler/ffx_fsr3upscaler.cpp"
    "${PATCHED_FFX_VK}"
    "${SDK}/src/backends/shared/ffx_shader_blobs.cpp"
    "${SDK}/src/backends/shared/blob_accessors/ffx_fsr3upscaler_shaderblobs.cpp")
target_include_directories(amd_fidelityfx_vk PRIVATE
    "${API}/include" "${API}/src" "${SDK}/include" "${CASE_LINKS}"
    "${SDK}/src/shared" "${SDK}/src/components" "${SDK}/src/backends/shared"
    "${SDK}/src/backends/shared/blob_accessors"
    "${FSR_VULKAN_ROOT}/generated/ffx-1.1.4/vk/fsr3upscaler")
target_compile_definitions(amd_fidelityfx_vk PRIVATE
    FFX_BACKEND_VK FFX_FSR3UPSCALER=1 FFXDLL_VERSION_MAJOR=1 FFXDLL_VERSION_MINOR=0
    FFXDLL_VERSION_PATCH=1 CI_PIPELINE_ID=0)
target_compile_options(amd_fidelityfx_vk PRIVATE
    "SHELL:-include \"${FSR_VULKAN_ROOT}/src/ffx_1_1_4_portability.hpp\""
    "SHELL:-include \"${CMAKE_CURRENT_SOURCE_DIR}/linux_compat.h\"" -fms-extensions -w)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/exports.map"
     "{ global: ffxCreateContext; ffxDestroyContext; ffxConfigure; ffxQuery; ffxDispatch; local: *; };")
target_link_libraries(amd_fidelityfx_vk PRIVATE Vulkan::Vulkan)
target_link_options(amd_fidelityfx_vk PRIVATE -Wl,--no-undefined -Wl,--exclude-libs,ALL
                    "-Wl,--version-script=${CMAKE_CURRENT_BINARY_DIR}/exports.map"
                    -static-libstdc++ -static-libgcc)
set_target_properties(amd_fidelityfx_vk PROPERTIES PREFIX "lib")
