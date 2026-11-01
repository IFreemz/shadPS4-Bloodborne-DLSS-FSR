# SPDX-FileCopyrightText: Copyright 2026 IFreemz
# SPDX-License-Identifier: MIT
#
# libamd_fidelityfx_vk.so: FSR 3.1 upscaling and frame generation for the Linux build, the
# counterpart of AMD's amd_fidelityfx_vk.dll. AMD only ships that DLL for Windows, so this builds
# the same ffx-api from source: AMD's API layer and frame generation swapchain (FidelityFX SDK
# v1.1.4) over the FSR 3.1.4 effects and Vulkan backend as made portable by FireBurn's
# FSR-Vulkan, with its pre-compiled SPIR-V.
#   cmake -S fsr3_linux -B build-fsr3 -DFFX_SDK_ROOT=<FidelityFX SDK v1.1.4> \
#         -DFSR_VULKAN_ROOT=<FSR-Vulkan> -DCMAKE_CXX_COMPILER=clang++
#   cmake --build build-fsr3

cmake_minimum_required(VERSION 3.20)
project(amd_fidelityfx_vk LANGUAGES CXX)

set(FFX_SDK_ROOT "" CACHE PATH "AMD FidelityFX SDK v1.1.4 (contains ffx-api/ and sdk/)")
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
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

set(API "${FFX_SDK_ROOT}/ffx-api")
set(SWAPCHAIN "${FFX_SDK_ROOT}/sdk/src/backends/vk/FrameInterpolationSwapchain")

# ffx-api includes the SDK headers both as FidelityFX/ and FidelityFx/.
set(CASE_LINKS "${CMAKE_CURRENT_BINARY_DIR}/case_links")
file(MAKE_DIRECTORY "${CASE_LINKS}")
file(CREATE_LINK "${SDK}/include/FidelityFX" "${CASE_LINKS}/FidelityFx" SYMBOLIC)

# Copies `source` to `output` with each FROM;TO pair of variables replaced, failing loudly when
# the source no longer contains what is replaced.
function(patch_source source output)
    file(READ "${source}" text)
    foreach(pair ${ARGN})
        string(REPLACE ":" ";" pair "${pair}")
        list(GET pair 0 from)
        list(GET pair 1 to)
        string(FIND "${text}" "${${from}}" found)
        if (found EQUAL -1)
            message(FATAL_ERROR "${source} changed: update the ${from} patch")
        endif()
        string(REPLACE "${${from}}" "${${to}}" text "${text}")
    endforeach()
    file(WRITE "${output}.new" "${text}")
    file(COPY_FILE "${output}.new" "${output}" ONLY_IF_DIFFERENT)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${source}")
endfunction()

# The backend: FSR-Vulkan leaves out AMD's frame generation swapchain, which this library builds,
# so hand it the swapchain again. And AMD's backend skips host-visible types when it wants
# device-local memory, failing on GPUs whose device-local memory is all host-visible (lavapipe,
# some integrated GPUs): fall back to those.
set(NO_SWAPCHAIN [[backendInterface->fpSwapChainConfigureFrameGeneration = nullptr;]])
set(SWAPCHAIN_CONFIG [[backendInterface->fpSwapChainConfigureFrameGeneration = ffxSetFrameGenerationConfigToSwapchainVK;]])
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
patch_source("${SDK}/src/backends/vk/ffx_vk.cpp" "${CMAKE_CURRENT_BINARY_DIR}/ffx_vk.cpp"
             NO_SWAPCHAIN:SWAPCHAIN_CONFIG SKIP:SKIP_WITH_FALLBACK BEST:BEST_WITH_FALLBACK
             RETURN:RETURN_WITH_FALLBACK)

# ffx-api passes only the device to the backend: report the features shadPS4 enables.
set(DEVICE_CONTEXT [[VkDeviceContext deviceContext = { backendDesc->vkDevice, backendDesc->vkPhysicalDevice, backendDesc->vkDeviceProcAddr };]])
set(DEVICE_CONTEXT_WITH_FEATURES [[VkDeviceContext deviceContext = { backendDesc->vkDevice, backendDesc->vkPhysicalDevice, backendDesc->vkDeviceProcAddr };
            ffxLinuxEnabledFeatures(&deviceContext);]])
set(VK_HEADER [[#include <FidelityFX/host/backends/vk/ffx_vk.h>]])
set(VK_HEADER_WITH_FEATURES [[#include <FidelityFX/host/backends/vk/ffx_vk.h>
void ffxLinuxEnabledFeatures(VkDeviceContext* context);]])
patch_source("${API}/src/backends.cpp" "${CMAKE_CURRENT_BINARY_DIR}/backends.cpp"
             DEVICE_CONTEXT:DEVICE_CONTEXT_WITH_FEATURES VK_HEADER:VK_HEADER_WITH_FEATURES)

# AMD's swapchain: two constructs only MSVC accepts.
set(WAIT_CALLBACK_CAST [[static_cast<FfxWaitCallbackFunc>(valuePtr)]])
set(WAIT_CALLBACK_REINTERPRET [[reinterpret_cast<FfxWaitCallbackFunc>(valuePtr)]])
set(TEMPORARY_SEMAPHORES [[toWait, SubmissionSemaphores());]])
set(NAMED_SEMAPHORES [[toWait, *std::make_unique<SubmissionSemaphores>());]])
patch_source("${SWAPCHAIN}/FrameInterpolationSwapchainVK.cpp"
             "${CMAKE_CURRENT_BINARY_DIR}/FrameInterpolationSwapchainVK.cpp"
             WAIT_CALLBACK_CAST:WAIT_CALLBACK_REINTERPRET TEMPORARY_SEMAPHORES:NAMED_SEMAPHORES)

add_library(amd_fidelityfx_vk SHARED
    ffx_provider_linux.cpp
    ffx_linux_device.cpp
    "${CMAKE_CURRENT_BINARY_DIR}/backends.cpp"
    "${CMAKE_CURRENT_BINARY_DIR}/ffx_vk.cpp"
    "${API}/src/ffx_api.cpp"
    "${API}/src/validation.cpp"
    "${API}/src/ffx_provider_fsr3upscale.cpp"
    "${API}/src/ffx_provider_framegeneration.cpp"
    "${API}/src/vk/ffx_provider_framegenerationswapchain_vk.cpp"
    "${SDK}/src/shared/ffx_assert.cpp"
    "${SDK}/src/shared/ffx_message.cpp"
    "${SDK}/src/shared/ffx_object_management.cpp"
    "${SDK}/src/shared/ffx_breadcrumbs_list.cpp"
    "${SDK}/src/components/fsr3upscaler/ffx_fsr3upscaler.cpp"
    "${SDK}/src/components/frameinterpolation/ffx_frameinterpolation.cpp"
    "${SDK}/src/components/opticalflow/ffx_opticalflow.cpp"
    "${SDK}/src/backends/shared/ffx_shader_blobs.cpp"
    "${SDK}/src/backends/shared/blob_accessors/ffx_fsr3upscaler_shaderblobs.cpp"
    "${SDK}/src/backends/shared/blob_accessors/ffx_frameinterpolation_shaderblobs.cpp"
    "${SDK}/src/backends/shared/blob_accessors/ffx_opticalflow_shaderblobs.cpp"
    "${CMAKE_CURRENT_BINARY_DIR}/FrameInterpolationSwapchainVK.cpp"
    "${SWAPCHAIN}/FrameInterpolationSwapchainVK_Helpers.cpp"
    "${SWAPCHAIN}/FrameInterpolationSwapchainVK_UiComposition.cpp"
    "${SWAPCHAIN}/FrameInterpolationSwapchainVK_DebugPacing.cpp")
target_include_directories(amd_fidelityfx_vk PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/compat" "${API}/include" "${API}/src" "${SDK}/include"
    "${CASE_LINKS}" "${SDK}/src/shared" "${SDK}/src/components" "${SDK}/src/backends/shared"
    "${SDK}/src/backends/shared/blob_accessors" "${SWAPCHAIN}"
    "${FSR_VULKAN_ROOT}/generated/ffx-1.1.4/vk/fsr3upscaler"
    "${FSR_VULKAN_ROOT}/generated/ffx-1.1.4/vk/fsr3framegeneration")
target_compile_definitions(amd_fidelityfx_vk PRIVATE
    FFX_BACKEND_VK FFX_FSR3UPSCALER=1 FFX_FI=1 FFX_OF=1 FFXDLL_VERSION_MAJOR=1
    FFXDLL_VERSION_MINOR=0 FFXDLL_VERSION_PATCH=1 CI_PIPELINE_ID=0)
target_compile_options(amd_fidelityfx_vk PRIVATE
    "SHELL:-include \"${FSR_VULKAN_ROOT}/src/ffx_1_1_4_portability.hpp\""
    "SHELL:-include \"${CMAKE_CURRENT_SOURCE_DIR}/linux_compat.h\"" -fms-extensions -w)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/exports.map"
     "{ global: ffxCreateContext; ffxDestroyContext; ffxConfigure; ffxQuery; ffxDispatch; local: *; };")
target_link_libraries(amd_fidelityfx_vk PRIVATE Vulkan::Vulkan Threads::Threads)
target_link_options(amd_fidelityfx_vk PRIVATE -Wl,--no-undefined -Wl,--exclude-libs,ALL
                    "-Wl,--version-script=${CMAKE_CURRENT_BINARY_DIR}/exports.map"
                    -static-libstdc++ -static-libgcc)
set_target_properties(amd_fidelityfx_vk PROPERTIES PREFIX "lib")
