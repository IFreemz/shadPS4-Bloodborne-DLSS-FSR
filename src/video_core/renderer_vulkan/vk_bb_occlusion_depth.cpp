// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <stdexcept>
#include <vector>
#include "common/logging/log.h"
#include "video_core/host_shaders/bb_occlusion_dilate_frag.h"
#include "video_core/host_shaders/fs_tri_vert.h"
#include "video_core/renderer_vulkan/vk_bb_occlusion_depth.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/texture_cache/image.h"

namespace Vulkan {

namespace {
vk::ImageSubresourceRange AttachmentRange(vk::Format format) {
    return {.aspectMask = BbOcclusionDepth::HasStencil(format)
                              ? vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil
                              : vk::ImageAspectFlagBits::eDepth,
            .levelCount = 1,
            .layerCount = 1};
}

template <typename T>
T CheckOrThrow(vk::ResultValue<T> result) {
    if (result.result != vk::Result::eSuccess) {
        throw std::runtime_error("occlusion test depth unavailable");
    }
    return std::move(result.value);
}

struct Push {
    u32 farthest_is_max;
    s32 radius;
};
} // Anonymous namespace

bool BbOcclusionDepth::HasStencil(vk::Format format) {
    return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint ||
           format == vk::Format::eD16UnormS8Uint;
}

struct BbOcclusionDepth::Impl {
    const Instance& instance;
    Runtime& runtime;
    Scheduler& scheduler;
    vk::Device device;
    bool failed{};

    vk::ShaderModule vertex{}, fragment{};
    vk::UniqueDescriptorSetLayout set_layout;
    vk::UniquePipelineLayout layout;
    struct FormatPipeline {
        vk::Format format;
        vk::UniquePipeline pipeline;
    };
    std::vector<FormatPipeline> pipelines;

    // The copy (sized and formatted as the game's depth image).
    std::unique_ptr<VideoCore::UniqueImage> image;
    vk::UniqueImageView view;
    vk::Format format{vk::Format::eUndefined};
    u32 width{}, height{};

    // The view the copy reads the game's depth through (its image may be replaced).
    vk::Image source_image{};
    vk::ImageView source_view{};

    Impl(const Instance& instance_, Runtime& runtime_, Scheduler& scheduler_)
        : instance{instance_}, runtime{runtime_}, scheduler{scheduler_},
          device{instance_.GetDevice()} {}

    ~Impl() {
        if (source_view) {
            device.destroyImageView(source_view);
        }
        if (vertex) {
            device.destroy(vertex);
        }
        if (fragment) {
            device.destroy(fragment);
        }
    }

    void CreateShared() {
        vertex = CompileSPV(FS_TRI_VERT, device);
        fragment = CompileSPV(BB_OCCLUSION_DILATE_FRAG, device);
        const vk::DescriptorSetLayoutBinding binding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eSampledImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment};
        set_layout = CheckOrThrow(device.createDescriptorSetLayoutUnique(
            {.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
             .bindingCount = 1,
             .pBindings = &binding}));
        const vk::PushConstantRange push{.stageFlags = vk::ShaderStageFlagBits::eFragment,
                                         .size = sizeof(Push)};
        const auto set = *set_layout;
        layout = CheckOrThrow(device.createPipelineLayoutUnique({.setLayoutCount = 1,
                                                          .pSetLayouts = &set,
                                                          .pushConstantRangeCount = 1,
                                                          .pPushConstantRanges = &push}));
    }

    vk::Pipeline PipelineFor(vk::Format depth_format) {
        for (const auto& entry : pipelines) {
            if (entry.format == depth_format) {
                return *entry.pipeline;
            }
        }
        const vk::PipelineInputAssemblyStateCreateInfo input_assembly{
            .topology = vk::PrimitiveTopology::eTriangleList};
        const vk::PipelineMultisampleStateCreateInfo multisampling{
            .rasterizationSamples = vk::SampleCountFlagBits::e1};
        const vk::PipelineDepthStencilStateCreateInfo depth_state{
            .depthTestEnable = true,
            .depthWriteEnable = true,
            .depthCompareOp = vk::CompareOp::eAlways,
        };
        const std::array dynamic_states{vk::DynamicState::eViewportWithCount,
                                        vk::DynamicState::eScissorWithCount};
        const vk::PipelineDynamicStateCreateInfo dynamic_info{
            .dynamicStateCount = u32(dynamic_states.size()),
            .pDynamicStates = dynamic_states.data()};
        const std::array stages{
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eVertex, .module = vertex, .pName = "main"},
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eFragment, .module = fragment, .pName = "main"}};
        const vk::PipelineRenderingCreateInfo rendering{
            .depthAttachmentFormat = depth_format,
            .stencilAttachmentFormat =
                HasStencil(depth_format) ? depth_format : vk::Format::eUndefined};
        const vk::PipelineColorBlendStateCreateInfo color_blending{};
        const vk::PipelineViewportStateCreateInfo viewport_info{};
        const vk::PipelineVertexInputStateCreateInfo vertex_input{};
        const vk::PipelineRasterizationStateCreateInfo raster{.lineWidth = 1.0f};
        auto pipeline = CheckOrThrow(device.createGraphicsPipelineUnique(
            {}, vk::GraphicsPipelineCreateInfo{.pNext = &rendering,
                                               .stageCount = u32(stages.size()),
                                               .pStages = stages.data(),
                                               .pVertexInputState = &vertex_input,
                                               .pInputAssemblyState = &input_assembly,
                                               .pViewportState = &viewport_info,
                                               .pRasterizationState = &raster,
                                               .pMultisampleState = &multisampling,
                                               .pDepthStencilState = &depth_state,
                                               .pColorBlendState = &color_blending,
                                               .pDynamicState = &dynamic_info,
                                               .layout = *layout}));
        const vk::Pipeline handle = *pipeline;
        pipelines.push_back({depth_format, std::move(pipeline)});
        return handle;
    }

    void EnsureImage(vk::Format depth_format, u32 image_width, u32 image_height) {
        if (image && format == depth_format && width == image_width && height == image_height) {
            return;
        }
        if (image) {
            // The old one may still be in use by the GPU.
            scheduler.DeferOperation([old_image = std::move(image), old_view = std::move(view)] {});
        }
        image = std::make_unique<VideoCore::UniqueImage>(device, instance.GetAllocator());
        image->Create(vk::ImageCreateInfo{
            .imageType = vk::ImageType::e2D,
            .format = depth_format,
            .extent = {image_width, image_height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
        });
        view = CheckOrThrow(device.createImageViewUnique(
            vk::ImageViewCreateInfo{.image = image->image,
                                    .viewType = vk::ImageViewType::e2D,
                                    .format = depth_format,
                                    .subresourceRange = AttachmentRange(depth_format)}));
        format = depth_format;
        width = image_width;
        height = image_height;
    }

    void EnsureSourceView(vk::Image source, vk::Format depth_format) {
        if (source == source_image && source_view) {
            return;
        }
        if (source_view) {
            scheduler.DeferOperation(
                [device = device, old = source_view] { device.destroyImageView(old); });
        }
        const vk::ImageViewUsageCreateInfo usage{.usage = vk::ImageUsageFlagBits::eSampled};
        source_view = CheckOrThrow(device.createImageView(vk::ImageViewCreateInfo{
            .pNext = &usage,
            .image = source,
            .viewType = vk::ImageViewType::e2D,
            .format = depth_format,
            .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eDepth,
                                 .levelCount = 1,
                                 .layerCount = 1}}));
        source_image = source;
    }

    void Barrier(vk::ImageLayout old_layout, vk::PipelineStageFlags2 src_stage,
                 vk::AccessFlags2 src_access, vk::PipelineStageFlags2 dst_stage,
                 vk::AccessFlags2 dst_access) {
        const vk::ImageMemoryBarrier2 barrier{
            .srcStageMask = src_stage,
            .srcAccessMask = src_access,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_access,
            .oldLayout = old_layout,
            .newLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image->image,
            .subresourceRange = AttachmentRange(format)};
        scheduler.Record([barrier](vk::CommandBuffer command) {
            command.pipelineBarrier2(
                vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
        });
    }

    vk::ImageView Build(VideoCore::Image& depth, u32 pass_width, u32 pass_height,
                        bool farthest_is_max) {
        const vk::Format depth_format = depth.info.pixel_format;
        const u32 image_width = depth.info.size.width;
        const u32 image_height = depth.info.size.height;
        if (depth.info.num_samples != 1 || pass_width == 0 || pass_height == 0 ||
            pass_width > image_width || pass_height > image_height) {
            return {};
        }
        if (!layout) {
            CreateShared();
        }
        EnsureImage(depth_format, image_width, image_height);
        EnsureSourceView(depth.GetImage(), depth_format);
        const vk::Pipeline pipeline = PipelineFor(depth_format);

        scheduler.EndRendering();
        runtime.Transit(&depth, vk::ImageLayout::eDepthStencilReadOnlyOptimal,
                        vk::PipelineStageFlagBits2::eFragmentShader,
                        vk::AccessFlagBits2::eShaderSampledRead);
        runtime.FlushBarriers();
        // Its old contents are not needed: rewritten everywhere the test pass reads.
        Barrier(vk::ImageLayout::eUndefined, vk::PipelineStageFlagBits2::eAllCommands,
                vk::AccessFlagBits2::eNone,
                vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                    vk::PipelineStageFlagBits2::eLateFragmentTests,
                vk::AccessFlagBits2::eDepthStencilAttachmentRead |
                    vk::AccessFlagBits2::eDepthStencilAttachmentWrite);

        RenderState pass{};
        pass.width = u16(pass_width);
        pass.height = u16(pass_height);
        pass.num_layers = 1;
        pass.depth_stencil_attachment.image_view = *view;
        pass.depth_stencil_attachment.image_layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
        pass.depth_stencil_attachment.has_depth = true;
        pass.depth_stencil_attachment.has_stencil = HasStencil(depth_format);
        pass.depth_stencil_attachment.stencil_clear = HasStencil(depth_format);
        scheduler.BeginRendering(pass);
        const Push push{.farthest_is_max = farthest_is_max ? 1u : 0u, .radius = 1};
        scheduler.Record([pipeline, layout = *layout, source = source_view, pass_width,
                          pass_height, push](vk::CommandBuffer command) {
            const vk::Viewport viewport{.width = float(pass_width),
                                        .height = float(pass_height),
                                        .maxDepth = 1.0f};
            const vk::Rect2D scissor{.extent = {pass_width, pass_height}};
            command.setViewportWithCount(viewport);
            command.setScissorWithCount(scissor);
            command.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
            const vk::DescriptorImageInfo image_info{
                .imageView = source, .imageLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal};
            const vk::WriteDescriptorSet write{.dstBinding = 0,
                                               .descriptorCount = 1,
                                               .descriptorType = vk::DescriptorType::eSampledImage,
                                               .pImageInfo = &image_info};
            command.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, layout, 0, write);
            command.pushConstants(layout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(push),
                                  &push);
            command.draw(3, 1, 0, 0);
        });
        scheduler.EndRendering();
        Barrier(vk::ImageLayout::eDepthStencilAttachmentOptimal,
                vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                    vk::PipelineStageFlagBits2::eLateFragmentTests,
                vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                    vk::PipelineStageFlagBits2::eLateFragmentTests,
                vk::AccessFlagBits2::eDepthStencilAttachmentRead |
                    vk::AccessFlagBits2::eDepthStencilAttachmentWrite);
        // The pass replaced the game's dynamic state, pipeline, descriptors and push constants.
        scheduler.GetDynamicState().Invalidate();
        return *view;
    }
};

BbOcclusionDepth::BbOcclusionDepth(const Instance& instance, Runtime& runtime,
                                   Scheduler& scheduler)
    : impl{std::make_unique<Impl>(instance, runtime, scheduler)} {}

BbOcclusionDepth::~BbOcclusionDepth() = default;

vk::ImageView BbOcclusionDepth::Build(VideoCore::Image& depth, u32 width, u32 height,
                                      bool farthest_is_max) {
    if (impl->failed) {
        return {};
    }
    try {
        return impl->Build(depth, width, height, farthest_is_max);
    } catch (const std::exception& e) {
        impl->failed = true;
        LOG_ERROR(Render_Vulkan, "Occlusion tests tolerant of a pixel: unavailable ({})",
                  e.what());
        return {};
    }
}

} // namespace Vulkan
