// SPDX-FileCopyrightText: Copyright 2026 IFreemz
// SPDX-License-Identifier: GPL-2.0-or-later
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>
#include "common/logging/log.h"
#include "video_core/host_shaders/bb_mirror_seed_depth_frag.h"
#include "video_core/host_shaders/bb_mirror_seed_stencil_frag.h"
#include "video_core/host_shaders/fs_tri_vert.h"
#include "video_core/renderer_vulkan/vk_bb_temporal_dlss.h"
#include "video_core/renderer_vulkan/vk_bb_velocity_mirror.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/texture_cache/image.h"

namespace Vulkan {
namespace {
constexpr vk::ImageSubresourceRange Range{
    .aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1};
// The game's depth is D32S8; the "Performance Patch" switches it to D16 without stencil.
bool HasStencil(vk::Format format) {
    return format == vk::Format::eD32SfloatS8Uint;
}
vk::ImageSubresourceRange DepthRange(vk::Format format) {
    return {.aspectMask = HasStencil(format)
                              ? vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil
                              : vk::ImageAspectFlagBits::eDepth,
            .levelCount = 1,
            .layerCount = 1};
}
// The game renders object velocity into a coarse target, 160x90 by default; patches such as
// "lower specific renders" and "HD Motion Blur" change its size.
constexpr u32 MinSourceWidth = 16, MinSourceHeight = 9;

// Seeds the mirror's depth and stencil with draws, for GPUs whose depth formats cannot be blit
// destinations (AMD). SHADPS4_BB_MIRROR_DRAW_SEED=1 forces it on other GPUs for testing.
class DepthSeeder {
public:
    explicit DepthSeeder(vk::Device device_) : device{device_} {
        vertex = CompileSPV(FS_TRI_VERT, device);
        depth_fragment = CompileSPV(BB_MIRROR_SEED_DEPTH_FRAG, device);
        stencil_fragment = CompileSPV(BB_MIRROR_SEED_STENCIL_FRAG, device);
        const vk::DescriptorSetLayoutBinding binding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eSampledImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment};
        set_layout = Check(device.createDescriptorSetLayoutUnique(
            {.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
             .bindingCount = 1,
             .pBindings = &binding}));
        const vk::PushConstantRange push{.stageFlags = vk::ShaderStageFlagBits::eFragment,
                                         .size = sizeof(Push)};
        const auto set = *set_layout;
        layout = Check(device.createPipelineLayoutUnique({.setLayoutCount = 1,
                                                          .pSetLayouts = &set,
                                                          .pushConstantRangeCount = 1,
                                                          .pPushConstantRanges = &push}));
    }
    ~DepthSeeder() {
        device.destroy(vertex);
        device.destroy(depth_fragment);
        device.destroy(stencil_fragment);
    }

    // Records the copy into the current rendering pass: depth, then stencil when stencil_view is
    // set. The source views must be in DEPTH_STENCIL_READ_ONLY_OPTIMAL.
    void Record(vk::CommandBuffer command, vk::Format format, vk::ImageView depth_view,
                vk::ImageView stencil_view, vk::Extent2D source, vk::Extent2D target) {
        auto& pipelines = PipelinesFor(format);
        const vk::Viewport viewport{
            .width = float(target.width), .height = float(target.height), .maxDepth = 1.0f};
        const vk::Rect2D scissor{.extent = target};
        command.setViewportWithCount(viewport);
        command.setScissorWithCount(scissor);
        Push push{.scale = {float(source.width) / float(target.width),
                            float(source.height) / float(target.height)}};
        const auto bind = [&](vk::Pipeline pipeline, vk::ImageView view) {
            command.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
            const vk::DescriptorImageInfo image{
                .imageView = view, .imageLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal};
            const vk::WriteDescriptorSet write{.dstBinding = 0,
                                               .descriptorCount = 1,
                                               .descriptorType = vk::DescriptorType::eSampledImage,
                                               .pImageInfo = &image};
            command.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, *layout, 0, write);
        };
        bind(*pipelines.depth, depth_view);
        command.pushConstants(*layout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(push), &push);
        command.draw(3, 1, 0, 0);
        if (!stencil_view)
            return;
        // Without VK_EXT_shader_stencil_export a shader cannot write stencil: write each bit in
        // its own pass, replacing through a one-bit write mask.
        bind(*pipelines.stencil, stencil_view);
        for (u32 bit = 0; bit < 8; ++bit) {
            push.bit = bit;
            command.setStencilWriteMask(vk::StencilFaceFlagBits::eFrontAndBack, 1u << bit);
            command.pushConstants(*layout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(push),
                                  &push);
            command.draw(3, 1, 0, 0);
        }
    }

private:
    struct Push {
        std::array<float, 2> scale;
        u32 bit;
    };
    struct Pipelines {
        vk::Format format;
        vk::UniquePipeline depth, stencil;
    };

    template <typename T>
    static T Check(vk::ResultValue<T> result) {
        if (result.result != vk::Result::eSuccess)
            throw std::runtime_error("velocity mirror depth seeding unavailable");
        return std::move(result.value);
    }

    Pipelines& PipelinesFor(vk::Format format) {
        for (auto& pipelines : cache)
            if (pipelines.format == format)
                return pipelines;
        Pipelines pipelines{.format = format};
        pipelines.depth = Create(format, depth_fragment, false);
        if (HasStencil(format))
            pipelines.stencil = Create(format, stencil_fragment, true);
        return cache.emplace_back(std::move(pipelines));
    }

    vk::UniquePipeline Create(vk::Format format, vk::ShaderModule fragment, bool stencil) {
        const vk::PipelineInputAssemblyStateCreateInfo input_assembly{
            .topology = vk::PrimitiveTopology::eTriangleList};
        const vk::PipelineMultisampleStateCreateInfo multisampling{.rasterizationSamples =
                                                                       vk::SampleCountFlagBits::e1};
        const vk::StencilOpState stencil_op{.failOp = vk::StencilOp::eKeep,
                                            .passOp = vk::StencilOp::eReplace,
                                            .depthFailOp = vk::StencilOp::eReplace,
                                            .compareOp = vk::CompareOp::eAlways,
                                            .compareMask = 0xFF,
                                            .reference = 0xFF};
        const vk::PipelineDepthStencilStateCreateInfo depth_state{.depthTestEnable = !stencil,
                                                                  .depthWriteEnable = !stencil,
                                                                  .depthCompareOp =
                                                                      vk::CompareOp::eAlways,
                                                                  .stencilTestEnable = stencil,
                                                                  .front = stencil_op,
                                                                  .back = stencil_op};
        std::vector dynamic_states{vk::DynamicState::eViewportWithCount,
                                   vk::DynamicState::eScissorWithCount};
        if (stencil)
            dynamic_states.push_back(vk::DynamicState::eStencilWriteMask);
        const vk::PipelineDynamicStateCreateInfo dynamic_info{
            .dynamicStateCount = u32(dynamic_states.size()),
            .pDynamicStates = dynamic_states.data()};
        const std::array stages{
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eVertex, .module = vertex, .pName = "main"},
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eFragment, .module = fragment, .pName = "main"}};
        const vk::PipelineRenderingCreateInfo rendering{
            .depthAttachmentFormat = format,
            .stencilAttachmentFormat = HasStencil(format) ? format : vk::Format::eUndefined};
        const vk::PipelineColorBlendStateCreateInfo color_blending{};
        const vk::PipelineViewportStateCreateInfo viewport_info{};
        const vk::PipelineVertexInputStateCreateInfo vertex_input{};
        const vk::PipelineRasterizationStateCreateInfo raster{.lineWidth = 1.0f};
        return Check(device.createGraphicsPipelineUnique(
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
    }

    vk::Device device;
    vk::ShaderModule vertex, depth_fragment, stencil_fragment;
    vk::UniqueDescriptorSetLayout set_layout;
    vk::UniquePipelineLayout layout;
    std::vector<Pipelines> cache;
};

bool ForceDrawSeed() {
    // Read once: getenv is slow on Windows and this runs for every mirrored draw.
    static const bool force = [] {
        const char* value = std::getenv("SHADPS4_BB_MIRROR_DRAW_SEED");
        return value && std::string_view{value} == "1";
    }();
    return force;
}
} // namespace
struct BbVelocityMirror::Impl {
    bool requested{}, stopped{}, failed{}, frame_drawn{};
    // Deferred replays of the current run of mirrored draws (one mirror pass, see Defer).
    bool batch_active{};
    RenderState batch_state{};
    BbVelocityMirror::Commands batch_commands;
    u64 flushes{};
    Scheduler* scheduler{};
    bool enabled{true};
    u64 draws{};
    std::unique_ptr<VideoCore::UniqueImage> image;
    std::unique_ptr<VideoCore::UniqueImage> depth;
    vk::UniqueImageView view;
    vk::UniqueImageView depth_view;
    vk::ImageLayout layout{vk::ImageLayout::eUndefined};
    vk::ImageLayout depth_layout{vk::ImageLayout::eUndefined};
    u64 source_depth_uid{};
    vk::Format depth_format{vk::Format::eUndefined};
    std::unique_ptr<DepthSeeder> seeder;
    bool seed_logged{};
    vk::Extent2D size{2560, 1440}; // replay resolution (the scene's render size)
    vk::Extent2D target{2560, 1440};
    Impl() {
        requested = BbTemporalDlssRequested();
    }
    void Transit(vk::CommandBuffer command, vk::ImageLayout next, vk::PipelineStageFlags2 stage,
                 vk::AccessFlags2 access) {
        const vk::ImageMemoryBarrier2 barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask =
                layout == vk::ImageLayout::eUndefined
                    ? vk::AccessFlagBits2::eNone
                    : vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = stage,
            .dstAccessMask = access,
            .oldLayout = layout,
            .newLayout = next,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image->image,
            .subresourceRange = Range};
        command.pipelineBarrier2(
            vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
        layout = next;
    }
    /// Same as Transit/TransitDepth, recorded in order through the scheduler (threaded
    /// recording keeps the command buffer with its recording thread).
    template <bool depth_image>
    void TransitRecorded(Scheduler& scheduler, vk::ImageLayout next, vk::PipelineStageFlags2 stage,
                         vk::AccessFlags2 access) {
        auto& current = depth_image ? depth_layout : layout;
        const vk::ImageMemoryBarrier2 barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask =
                current == vk::ImageLayout::eUndefined
                    ? vk::AccessFlagBits2::eNone
                    : vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = stage,
            .dstAccessMask = access,
            .oldLayout = current,
            .newLayout = next,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = depth_image ? depth->image : image->image,
            .subresourceRange = depth_image ? DepthRange(depth_format) : Range};
        scheduler.Record([barrier](vk::CommandBuffer command) {
            command.pipelineBarrier2(
                vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
        });
        current = next;
    }
    void TransitDepth(vk::CommandBuffer command, vk::ImageLayout next,
                      vk::PipelineStageFlags2 stage, vk::AccessFlags2 access) {
        const vk::ImageMemoryBarrier2 barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask =
                depth_layout == vk::ImageLayout::eUndefined
                    ? vk::AccessFlagBits2::eNone
                    : vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = stage,
            .dstAccessMask = access,
            .oldLayout = depth_layout,
            .newLayout = next,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = depth->image,
            .subresourceRange = DepthRange(depth_format)};
        command.pipelineBarrier2(
            vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
        depth_layout = next;
    }
};
BbVelocityMirror::BbVelocityMirror() : impl{std::make_unique<Impl>()} {}
BbVelocityMirror::~BbVelocityMirror() = default;
bool BbVelocityMirror::Requested() const {
    return impl->requested && impl->enabled;
}
void BbVelocityMirror::SetEnabled(bool enabled) {
    impl->enabled = enabled;
}
bool BbVelocityMirror::Defer(const Instance& instance, Runtime& runtime, Scheduler& scheduler,
                             const GraphicsPipeline& pipeline, const RenderState& guest_state,
                             VideoCore::Image* guest_depth, u32 depth_layer, const Capture& capture,
                             bool& guest_pass_ended) {
    guest_pass_ended = false;
    if (!impl->requested || impl->stopped || impl->failed || !pipeline.VelocityMirrorHandle() ||
        guest_state.width < MinSourceWidth || guest_state.height < MinSourceHeight ||
        guest_state.num_layers != 1 || guest_state.num_color_attachments != 1 ||
        !guest_state.depth_stencil_attachment.has_depth || !guest_depth || !guest_depth->backing ||
        depth_layer != 0)
        return false;
    const auto& depth_ci = guest_depth->backing->image.image_ci;
    if ((depth_ci.format != vk::Format::eD32SfloatS8Uint &&
         depth_ci.format != vk::Format::eD16Unorm) ||
        guest_state.depth_stencil_attachment.has_stencil != HasStencil(depth_ci.format) ||
        depth_ci.samples != vk::SampleCountFlagBits::e1 ||
        depth_ci.extent.width < guest_state.width || depth_ci.extent.height < guest_state.height ||
        !instance.IsFormatSupported(depth_ci.format,
                                    vk::FormatFeatureFlagBits2::eDepthStencilAttachment))
        return false;
    // The guest depth is scaled into the mirror's with a blit where the GPU can blit depth, and
    // with draws that sample it otherwise.
    const bool blit =
        !ForceDrawSeed() && (depth_ci.usage & vk::ImageUsageFlagBits::eTransferSrc) &&
        instance.IsFormatSupported(depth_ci.format, vk::FormatFeatureFlagBits2::eBlitSrc |
                                                        vk::FormatFeatureFlagBits2::eBlitDst);
    if (!blit &&
        (!(depth_ci.usage & vk::ImageUsageFlagBits::eSampled) ||
         !instance.IsFormatSupported(depth_ci.format, vk::FormatFeatureFlagBits2::eSampledImage))) {
        static bool logged{};
        if (!std::exchange(logged, true))
            LOG_WARNING(Render_Vulkan,
                        "[BB-VELOCITY-MIRROR] {} depth can be neither blit nor "
                        "sampled; object motion is off",
                        vk::to_string(depth_ci.format));
        return false;
    }
    const auto& state = scheduler.GetDynamicState();
    if (state.viewports.size() != 1 || state.scissors.size() != 1 ||
        state.rasterizer_discard_enable)
        return false;
    auto viewports = state.viewports;
    auto scissors = state.scissors;
    auto& viewport = viewports[0];
    static u32 viewport_diagnostics{};
    if (viewport_diagnostics++ < 4)
        LOG_INFO(Render_Vulkan, "[BB-VELOCITY-MIRROR] Viewport {}x{}, xy={},{}; target={}x{}",
                 viewport.width, viewport.height, viewport.x, viewport.y, guest_state.width,
                 guest_state.height);
    // Screen-space/clipping-disabled fullscreen draws are intentionally unsupported.
    const u32 SourceWidth = guest_state.width, SourceHeight = guest_state.height;
    if (std::abs(viewport.width) != float(SourceWidth) ||
        std::abs(viewport.height) != float(SourceHeight))
        return false;
    if (impl->image && (impl->size != impl->target || impl->depth_format != depth_ci.format)) {
        scheduler.Finish(); // the old images may still be in flight (pending draws flushed)
        guest_pass_ended = true;
        impl->image.reset();
        impl->depth.reset();
        impl->view.reset();
        impl->depth_view.reset();
        impl->layout = impl->depth_layout = vk::ImageLayout::eUndefined;
        impl->frame_drawn = false;
    }
    impl->size = impl->target;
    impl->depth_format = depth_ci.format;
    const u32 Width = impl->size.width, Height = impl->size.height;
    const float fx = float(Width) / SourceWidth, fy = float(Height) / SourceHeight;
    viewport.x *= fx;
    viewport.y *= fy;
    viewport.width *= fx;
    viewport.height *= fy;
    scissors[0].offset.x = s32(std::lround(scissors[0].offset.x * fx));
    scissors[0].offset.y = s32(std::lround(scissors[0].offset.y * fy));
    scissors[0].extent.width = u32(std::lround(scissors[0].extent.width * fx));
    scissors[0].extent.height = u32(std::lround(scissors[0].extent.height * fy));
    if (!impl->frame_drawn) {
        // Seeding the frame's mirror depth runs outside the guest pass (once per frame).
        scheduler.EndRendering();
        guest_pass_ended = true;
    }
    impl->scheduler = &scheduler;
    try {
        if (!impl->image) {
            impl->image = std::make_unique<VideoCore::UniqueImage>(instance.GetDevice(),
                                                                   instance.GetAllocator());
            impl->image->Create(vk::ImageCreateInfo{
                .imageType = vk::ImageType::e2D,
                .format = vk::Format::eR16G16B16A16Sfloat,
                .extent = {Width, Height, 1},
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = vk::SampleCountFlagBits::e1,
                .tiling = vk::ImageTiling::eOptimal,
                .usage = vk::ImageUsageFlagBits::eColorAttachment |
                         vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc});
            auto result = instance.GetDevice().createImageViewUnique(
                vk::ImageViewCreateInfo{.image = impl->image->image,
                                        .viewType = vk::ImageViewType::e2D,
                                        .format = vk::Format::eR16G16B16A16Sfloat,
                                        .subresourceRange = Range});
            if (result.result != vk::Result::eSuccess)
                throw std::runtime_error("velocity mirror image view unavailable");
            impl->view = std::move(result.value);
            impl->depth = std::make_unique<VideoCore::UniqueImage>(instance.GetDevice(),
                                                                   instance.GetAllocator());
            impl->depth->Create(
                vk::ImageCreateInfo{.imageType = vk::ImageType::e2D,
                                    .format = depth_ci.format,
                                    .extent = {Width, Height, 1},
                                    .mipLevels = 1,
                                    .arrayLayers = 1,
                                    .samples = vk::SampleCountFlagBits::e1,
                                    .tiling = vk::ImageTiling::eOptimal,
                                    .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment |
                                             vk::ImageUsageFlagBits::eTransferDst});
            auto result_depth = instance.GetDevice().createImageViewUnique(
                vk::ImageViewCreateInfo{.image = impl->depth->image,
                                        .viewType = vk::ImageViewType::e2D,
                                        .format = depth_ci.format,
                                        .subresourceRange = DepthRange(depth_ci.format)});
            if (result_depth.result != vk::Result::eSuccess)
                throw std::runtime_error("velocity mirror depth view unavailable");
            impl->depth_view = std::move(result_depth.value);
        }
        // Seeding (first mirrored draw of a frame) records directly; the per-draw path below
        // goes through Record() and leaves the command buffer with the recording thread.
        const auto command = impl->frame_drawn ? vk::CommandBuffer{} : scheduler.CommandBuffer();
        if (!impl->seed_logged) {
            impl->seed_logged = true;
            LOG_INFO(Render_Vulkan, "[BB-VELOCITY-MIRROR] Depth seeded by {}",
                     blit ? "blit" : "draws");
        }
        if (!impl->frame_drawn && !blit) {
            // Same copy as the blit below, as draws that sample the guest depth and stencil.
            if (!impl->seeder)
                impl->seeder = std::make_unique<DepthSeeder>(instance.GetDevice());
            const auto device = instance.GetDevice();
            const auto view_for = [&](vk::ImageAspectFlagBits aspect) {
                const vk::ImageViewUsageCreateInfo usage{.usage = vk::ImageUsageFlagBits::eSampled};
                auto view = device.createImageView(
                    vk::ImageViewCreateInfo{.pNext = &usage,
                                            .image = guest_depth->GetImage(),
                                            .viewType = vk::ImageViewType::e2D,
                                            .format = depth_ci.format,
                                            .subresourceRange = {.aspectMask = aspect,
                                                                 .levelCount = 1,
                                                                 .baseArrayLayer = depth_layer,
                                                                 .layerCount = 1}});
                if (view.result != vk::Result::eSuccess)
                    throw std::runtime_error("velocity mirror source view unavailable");
                return view.value;
            };
            const bool stencil = HasStencil(depth_ci.format);
            const auto depth_source = view_for(vk::ImageAspectFlagBits::eDepth);
            const auto stencil_source =
                stencil ? view_for(vk::ImageAspectFlagBits::eStencil) : vk::ImageView{};
            scheduler.DeferOperation([device, depth_source, stencil_source] {
                device.destroyImageView(depth_source);
                if (stencil_source)
                    device.destroyImageView(stencil_source);
            });
            runtime.Transit(guest_depth, vk::ImageLayout::eDepthStencilReadOnlyOptimal,
                            vk::PipelineStageFlagBits2::eFragmentShader,
                            vk::AccessFlagBits2::eShaderSampledRead);
            runtime.FlushBarriers();
            impl->TransitDepth(command, vk::ImageLayout::eDepthStencilAttachmentOptimal,
                               vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                                   vk::PipelineStageFlagBits2::eLateFragmentTests,
                               vk::AccessFlagBits2::eDepthStencilAttachmentRead |
                                   vk::AccessFlagBits2::eDepthStencilAttachmentWrite);
            RenderState seed{};
            seed.width = Width;
            seed.height = Height;
            seed.num_layers = 1;
            seed.depth_stencil_attachment.image_view = *impl->depth_view;
            seed.depth_stencil_attachment.image_layout =
                vk::ImageLayout::eDepthStencilAttachmentOptimal;
            seed.depth_stencil_attachment.has_depth = true;
            seed.depth_stencil_attachment.has_stencil = stencil;
            seed.depth_stencil_attachment.stencil_clear = stencil; // bits are only ever set
            scheduler.BeginRendering(seed);
            impl->seeder->Record(command, depth_ci.format, depth_source, stencil_source,
                                 {guest_state.width, guest_state.height}, {Width, Height});
            scheduler.EndRendering();
            runtime.Transit(guest_depth, guest_state.depth_stencil_attachment.image_layout,
                            vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                                vk::PipelineStageFlagBits2::eLateFragmentTests,
                            vk::AccessFlagBits2::eDepthStencilAttachmentRead |
                                vk::AccessFlagBits2::eDepthStencilAttachmentWrite);
            runtime.FlushBarriers();
            impl->source_depth_uid = guest_depth->image_uid;
            // The seeding draws replaced the guest's dynamic state, descriptors and push
            // constants; restore them for the replay and the guest draw after it.
            auto& dynamic = scheduler.GetDynamicState();
            dynamic.Invalidate();
            dynamic.Commit(instance, command);
        }
        if (!impl->frame_drawn && blit) {
            // Seed the exact original render-area depth/stencil BEFORE the first guest draw.
            // Nearest scaling preserves its visibility tests; the original image is read only.
            runtime.Transit(guest_depth, vk::ImageLayout::eTransferSrcOptimal,
                            vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferRead);
            runtime.FlushBarriers();
            impl->TransitDepth(command, vk::ImageLayout::eTransferDstOptimal,
                               vk::PipelineStageFlagBits2::eBlit,
                               vk::AccessFlagBits2::eTransferWrite);
            std::array<vk::ImageBlit, 2> blits;
            const u32 blit_count = HasStencil(depth_ci.format) ? 2 : 1;
            for (u32 i = 0; i < blit_count; ++i) {
                const auto aspect =
                    i == 0 ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eStencil;
                blits[i] = vk::ImageBlit{
                    .srcSubresource = {.aspectMask = aspect,
                                       .baseArrayLayer = depth_layer,
                                       .layerCount = 1},
                    .srcOffsets = std::array{vk::Offset3D{}, vk::Offset3D{guest_state.width,
                                                                          guest_state.height, 1}},
                    .dstSubresource = {.aspectMask = aspect, .layerCount = 1},
                    .dstOffsets =
                        std::array{vk::Offset3D{}, vk::Offset3D{s32(Width), s32(Height), 1}}};
            }
            command.blitImage(guest_depth->GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                              impl->depth->image, vk::ImageLayout::eTransferDstOptimal,
                              vk::ArrayProxy<const vk::ImageBlit>{blit_count, blits.data()},
                              vk::Filter::eNearest);
            runtime.Transit(guest_depth, guest_state.depth_stencil_attachment.image_layout,
                            vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                                vk::PipelineStageFlagBits2::eLateFragmentTests,
                            vk::AccessFlagBits2::eDepthStencilAttachmentRead |
                                vk::AccessFlagBits2::eDepthStencilAttachmentWrite);
            runtime.FlushBarriers();
            impl->source_depth_uid = guest_depth->image_uid;
        }
        RenderState mirror{};
        mirror.width = Width;
        mirror.height = Height;
        mirror.num_layers = 1;
        mirror.num_color_attachments = 1;
        mirror.depth_stencil_attachment = guest_state.depth_stencil_attachment;
        mirror.depth_stencil_attachment.image_view = *impl->depth_view;
        mirror.depth_stencil_attachment.image_layout =
            vk::ImageLayout::eDepthStencilAttachmentOptimal;
        mirror.depth_stencil_attachment.depth_clear = false;
        mirror.depth_stencil_attachment.stencil_clear = false;
        mirror.color_attachments[0] = {
            .image_view = *impl->view,
            .image_layout = vk::ImageLayout::eColorAttachmentOptimal,
            .clear_value = {std::bit_cast<u32>(.5f), std::bit_cast<u32>(.5f), 0, 0},
            .is_clear = impl->frame_drawn ? 0u : 1u};
        if (impl->batch_active) {
            // Same mirror pass (whether it clears is decided by the batch's first draw)?
            RenderState probe = mirror;
            probe.color_attachments[0].is_clear = impl->batch_state.color_attachments[0].is_clear;
            if (probe == impl->batch_state) {
                mirror = impl->batch_state;
            } else {
                scheduler.EndRendering(); // the pass-end hook flushes the batch
                guest_pass_ended = true;
                FlushBatch(scheduler);
            }
        }
        if (!impl->batch_active) {
            impl->batch_state = mirror;
            impl->batch_active = true;
            impl->batch_commands.clear();
        }
        impl->batch_commands.emplace_back(
            [handle = pipeline.VelocityMirrorHandle()](vk::CommandBuffer command) {
                command.bindPipeline(vk::PipelineBindPoint::eGraphics, handle);
            });
        capture(impl->batch_commands, viewports, scissors);
        impl->frame_drawn = true;
        ++impl->draws;
        return true;
    } catch (const std::exception& error) {
        impl->failed = true;
        scheduler.EndRendering();
        scheduler.GetDynamicState().Invalidate();
        LOG_WARNING(Render_Vulkan, "[BB-VELOCITY-MIRROR] Disabled: {}", error.what());
        return false;
    }
}
bool BbVelocityMirror::HasBatch() const {
    return impl->batch_active;
}
u64 BbVelocityMirror::Flushes() const {
    return impl->flushes;
}
void BbVelocityMirror::FlushBatch(Scheduler& scheduler) {
    if (!impl->batch_active) {
        return;
    }
    impl->batch_active = false;
    ++impl->flushes;
    auto commands = std::move(impl->batch_commands);
    impl->batch_commands.clear();
    impl->TransitRecorded<true>(scheduler, vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                                    vk::PipelineStageFlagBits2::eLateFragmentTests,
                                vk::AccessFlagBits2::eDepthStencilAttachmentRead |
                                    vk::AccessFlagBits2::eDepthStencilAttachmentWrite);
    impl->TransitRecorded<false>(scheduler, vk::ImageLayout::eColorAttachmentOptimal,
                                 vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                 vk::AccessFlagBits2::eColorAttachmentRead |
                                     vk::AccessFlagBits2::eColorAttachmentWrite);
    scheduler.BeginRendering(impl->batch_state);
    scheduler.Record([commands = std::move(commands)](vk::CommandBuffer command) mutable {
        for (auto& replay : commands) {
            replay(vk::CommandBuffer{command});
        }
    });
    scheduler.EndRendering();
    // The replays changed the bound pipeline, descriptors and dynamic state.
    scheduler.GetDynamicState().Invalidate();
}
std::optional<BbVelocityMirror::Frame> BbVelocityMirror::ConsumeFrame(vk::CommandBuffer command) {
    if (!impl->requested || impl->stopped || impl->failed || !impl->frame_drawn || !impl->image)
        return {};
    if (impl->batch_active && impl->scheduler) {
        // Normally flushed when the guest pass ended; the caller records directly here.
        impl->scheduler->EndRendering();
        FlushBatch(*impl->scheduler);
    }
    impl->frame_drawn = false;
    impl->Transit(command, vk::ImageLayout::eShaderReadOnlyOptimal,
                  vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead);
    return Frame{*impl->view, impl->size};
}
void BbVelocityMirror::SetTargetSize(vk::Extent2D size) {
    impl->target = size;
}
void BbVelocityMirror::Shutdown(Scheduler& scheduler) {
    if (!impl->requested || impl->stopped)
        return;
    impl->stopped = true;
    impl->batch_active = false;
    impl->batch_commands.clear();
    scheduler.Finish();
    impl->view.reset();
    impl->depth_view.reset();
    impl->image.reset();
    impl->depth.reset();
    LOG_INFO(Render_Vulkan,
             "[BB-VELOCITY-MIRROR] Orderly GPU-thread teardown after{} mirrored draws",
             impl->draws);
}
} // namespace Vulkan
