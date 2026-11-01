// SPDX-FileCopyrightText: Copyright 2026 IFreemz
// SPDX-License-Identifier: MIT
//
// C interface between shadPS4 and shadps4_dlss.dll (shadps4_dlss.so on Linux). The emulator
// contains no NVIDIA code: it loads the bridge at runtime when present, and keeps its normal
// rendering when it is absent. DLSS Frame Generation follows bbport's (deadinside28/bloodborne_pc)
// extension of this bridge.

#pragma once

#include <stdint.h>
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SHADPS4_DLSS_BRIDGE_ABI 3

typedef void (*ShadDlssLogFn)(int warning, const char* message);

typedef struct ShadDlssImage {
    VkImage image;
    VkImageView view;
    VkImageSubresourceRange range;
    VkFormat format;
    uint32_t width;
    uint32_t height;
} ShadDlssImage;

typedef struct ShadDlssFeature {
    uint32_t input_width;
    uint32_t input_height;
    uint32_t output_width;
    uint32_t output_height;
    int32_t quality; // 0 DLAA, 1 Quality, 2 Balanced, 3 Performance, 4 Ultra Performance
    int32_t depth_inverted;
    uint32_t preset; // NVSDK_NGX_DLSS_Hint_Render_Preset value, 0 = driver default
    int32_t hdr;     // 1: linear HDR colour with automatic exposure, 0: LDR
} ShadDlssFeature;

typedef struct ShadDlssEvaluate {
    ShadDlssImage color;  // sampled, LDR
    ShadDlssImage depth;  // sampled, hardware depth
    ShadDlssImage motion; // sampled, render-resolution pixels, current to previous
    ShadDlssImage output; // storage, general layout
    float jitter_x;
    float jitter_y;
    int32_t reset;
    float frame_ms;
} ShadDlssEvaluate;

// DLSS Frame Generation called directly (Linux; Streamline drives it on Windows): frames between
// two finished frames.
typedef struct ShadDlssgFeature {
    uint32_t width; // the finished frame: input and generated colour
    uint32_t height;
    VkFormat format;
    uint32_t render_width; // depth and motion vectors
    uint32_t render_height;
} ShadDlssgFeature;

typedef struct ShadDlssgEvaluate {
    ShadDlssImage backbuffer; // sampled, general layout: the finished frame
    ShadDlssImage depth;      // sampled, general layout, R32F hardware depth
    ShadDlssImage motion;     // sampled, general layout, render pixels, current to previous
    ShadDlssImage output;     // storage, general layout: the generated frame
    ShadDlssImage hudless;    // sampled, general layout: without the overlays; image may be null
    float jitter_x;           // render pixels
    float jitter_y;
    float mvec_scale_x; // motion vector scale DLSS-G applies
    float mvec_scale_y;
    float camera_near;
    float camera_far;
    float camera_fov; // vertical, radians
    float camera_aspect;
    float camera_position[3];
    float camera_up[3];
    float camera_right[3];
    float camera_forward[3];
    // Row-major, row vectors, as Streamline takes them.
    float view_to_clip[4][4];
    float clip_to_view[4][4];
    float clip_to_prev_clip[4][4];
    float prev_clip_to_clip[4][4];
    int32_t depth_inverted;
    int32_t reset;
    uint32_t multi_frame_count; // frames generated for this pair (1: 2x)
    uint32_t multi_frame_index; // 1 .. multi_frame_count
} ShadDlssgEvaluate;

typedef struct ShadDlssApi {
    uint32_t abi;
    // Directories are UTF-16 paths: where nvngx_dlss.dll lives, and a writable data directory.
    int32_t (*Configure)(const wchar_t* dll_directory, const wchar_t* data_directory,
                         ShadDlssLogFn log);
    // Extensions NGX needs. The arrays stay valid until Shutdown.
    int32_t (*InstanceExtensions)(uint32_t* count, const VkExtensionProperties** extensions);
    int32_t (*DeviceExtensions)(VkInstance instance, VkPhysicalDevice physical, uint32_t* count,
                                const VkExtensionProperties** extensions);
    int32_t (*Initialize)(VkInstance instance, VkPhysicalDevice physical, VkDevice device,
                          PFN_vkGetInstanceProcAddr get_instance_proc,
                          PFN_vkGetDeviceProcAddr get_device_proc);
    int32_t (*CreateFeature)(VkCommandBuffer command, const ShadDlssFeature* feature);
    int32_t (*Evaluate)(VkCommandBuffer command, const ShadDlssEvaluate* evaluate);
    // The caller must have drained the GPU.
    void (*ReleaseFeature)(void);
    void (*Shutdown)(void);
    // Frames DLSS Frame Generation makes per rendered frame after Initialize: 0 when unavailable
    // (no nvngx_dlssg library, RTX 30 or older, an old driver), up to 3 on RTX 50. Always 0 on
    // Windows, where Streamline runs it.
    int32_t (*FrameGenMaxCount)(void);
    int32_t (*FrameGenCreate)(VkCommandBuffer command, const ShadDlssgFeature* feature);
    int32_t (*FrameGenEvaluate)(VkCommandBuffer command, const ShadDlssgEvaluate* evaluate);
    // The caller must have drained the GPU.
    void (*FrameGenRelease)(void);
} ShadDlssApi;

typedef const ShadDlssApi* (*ShadDlssGetApiFn)(void);

#ifdef __cplusplus
}
#endif
