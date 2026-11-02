// SPDX-FileCopyrightText: Copyright 2026 IFreemz
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

namespace FrameGen {
struct FrameInputs;
}

// Loads the optional shadps4_dlss.dll bridge next to the executable. shadPS4 itself contains no
// NVIDIA code; without the bridge (or on unsupported GPUs) rendering stays unchanged.
// Callers own synchronization and must drain recorded work before releasing a feature.
class DlssNgx {
public:
    // True when the bridge and nvngx_dlss.dll are present and DLSS is not disabled.
    static bool Present();
    static std::unique_ptr<DlssNgx> Create();
    ~DlssNgx();

    void AppendInstanceExtensions(std::vector<const char*>& enabled);
    void AppendDeviceExtensions(vk::Instance instance, vk::PhysicalDevice physical,
                                std::vector<const char*>& enabled);
    void Initialize(vk::Instance instance, vk::PhysicalDevice physical, vk::Device device);
    void Shutdown();
    bool IsAvailable() const;

    struct Resource {
        vk::Image image;
        vk::ImageView view;
        vk::ImageSubresourceRange range;
        vk::Format format;
        vk::Extent2D extent;
    };

    // 0 DLAA, 1 Quality, 2 Balanced, 3 Performance, 4 Ultra Performance.
    static int QualityForScale(float output_over_input);
    struct FeatureDesc {
        u32 input_width, input_height, output_width, output_height;
        int quality;
        bool depth_inverted;
        u32 preset; // NVSDK_NGX_DLSS_Hint_Render_Preset value, 0 = driver default
        bool hdr;   // linear HDR colour (automatic exposure)
    };
    // Releases an existing feature first; the caller must have drained the GPU if one exists.
    bool CreateFeature(vk::CommandBuffer command, const FeatureDesc& desc);
    bool HasFeature(const FeatureDesc& desc) const;
    struct EvalDesc {
        float jitter_x, jitter_y; // render pixels, same space as the motion vectors
        bool reset;
        float frame_ms;
    };
    bool Evaluate(vk::CommandBuffer command, const Resource& color, const Resource& depth,
                  const Resource& motion, const Resource& output, const EvalDesc& eval);
    void ReleaseFeatureAfterGpuDrain();

    // DLSS Frame Generation called directly (Linux; Streamline runs it on Windows): frames it
    // makes per rendered frame, 0 when unavailable.
    u32 FrameGenMaxCount() const;
    struct FrameGenDesc {
        vk::Extent2D display, render;
        vk::Format format;
    };
    // Releases an existing feature first; the caller must have drained the GPU if one exists.
    bool FrameGenCreate(vk::CommandBuffer command, const FrameGenDesc& desc);
    // Everything in the general layout. `hudless` may be null. Frame `index` (1 ..
    // `count`) of the `count` generated between two rendered frames.
    bool FrameGenEvaluate(vk::CommandBuffer command, const Resource& backbuffer,
                          const Resource& depth, const Resource& motion, const Resource& output,
                          const Resource* hudless, const FrameGen::FrameInputs& inputs,
                          std::array<float, 2> mvec_scale, bool reset, u32 count, u32 index);
    void FrameGenReleaseAfterGpuDrain();

private:
    DlssNgx();
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Vulkan
