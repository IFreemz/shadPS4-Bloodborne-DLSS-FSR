// SPDX-FileCopyrightText: Copyright 2026 IFreemz
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include "common/logging/log.h"
#include "imgui/renderer/imgui_core.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/host_shaders/hdr_meter_comp.h"
#include "video_core/renderer_vulkan/vk_hdr_meter.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/texture_cache/image.h"

namespace Vulkan::HdrMeter {
namespace {

constexpr u32 Bins = 256;
constexpr u64 HistogramBytes = Bins * sizeof(u32);
constexpr u32 Slots = 3; // readbacks in flight
constexpr vk::ImageSubresourceRange Range{
    .aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1};

// The upper edge of a bin, in nits.
float BinNits(u32 bin) {
    return std::exp2(float(bin + 1) / 14.0f - 4.0f);
}

struct State {
    vk::Device device;
    VideoCore::UniqueImage image;
    vk::UniqueImageView view;
    vk::Extent2D extent{};
    vk::Format format{};
    vk::UniqueSampler sampler;
    vk::UniqueDescriptorSetLayout descriptors;
    vk::UniquePipelineLayout layout;
    vk::UniquePipeline pipeline;
    std::unique_ptr<VideoCore::Buffer> histogram, readback;
    u64 frame{};
    std::array<u64, Bins> second{};
    std::chrono::steady_clock::time_point second_start{};
    std::mutex mutex;
    Reading reading{};
};

std::unique_ptr<State> state;

void CreatePipeline(State& s) {
    const std::array bindings{
        vk::DescriptorSetLayoutBinding{.binding = 0,
                                       .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                       .descriptorCount = 1,
                                       .stageFlags = vk::ShaderStageFlagBits::eCompute},
        vk::DescriptorSetLayoutBinding{.binding = 1,
                                       .descriptorType = vk::DescriptorType::eStorageBuffer,
                                       .descriptorCount = 1,
                                       .stageFlags = vk::ShaderStageFlagBits::eCompute}};
    s.descriptors = Check<"HDR meter descriptors">(s.device.createDescriptorSetLayoutUnique(
        {.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptor,
         .bindingCount = u32(bindings.size()),
         .pBindings = bindings.data()}));
    s.layout = Check<"HDR meter layout">(s.device.createPipelineLayoutUnique(
        {.setLayoutCount = 1, .pSetLayouts = &s.descriptors.get()}));
    const auto module = CompileSPV(HDR_METER_COMP, s.device);
    s.pipeline = Check<"HDR meter pipeline">(s.device.createComputePipelineUnique(
        {}, vk::ComputePipelineCreateInfo{.stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                                                    .module = module,
                                                    .pName = "main"},
                                          .layout = *s.layout}));
    s.device.destroyShaderModule(module);
    s.sampler = Check<"HDR meter sampler">(s.device.createSamplerUnique(
        {.magFilter = vk::Filter::eNearest, .minFilter = vk::Filter::eNearest}));
}

} // namespace

bool Enabled() {
    static const bool enabled = std::getenv("SHADPS4_HDR_METER") != nullptr;
    return enabled;
}

void Record(const Instance& instance, vk::CommandBuffer command, vk::Extent2D extent,
            vk::Format format) {
    if (!state) {
        state = std::make_unique<State>();
        state->device = instance.GetDevice();
        CreatePipeline(*state);
        state->histogram = std::make_unique<VideoCore::Buffer>(instance, 0, HistogramBytes,
                                                               VideoCore::MemoryType::DeviceLocal);
        state->readback = std::make_unique<VideoCore::Buffer>(instance, 0, Slots * HistogramBytes,
                                                              VideoCore::MemoryType::HostCached);
        state->second_start = std::chrono::steady_clock::now();
        LOG_INFO(Render_Vulkan, "[HDR-METER] On");
    }
    auto& s = *state;
    if (s.extent != extent || s.format != format) {
        (void)s.device.waitIdle();
        s.image = VideoCore::UniqueImage{s.device, instance.GetAllocator()};
        s.image.Create(vk::ImageCreateInfo{.imageType = vk::ImageType::e2D,
                                           .format = format,
                                           .extent = {extent.width, extent.height, 1},
                                           .mipLevels = 1,
                                           .arrayLayers = 1,
                                           .samples = vk::SampleCountFlagBits::e1,
                                           .tiling = vk::ImageTiling::eOptimal,
                                           .usage = vk::ImageUsageFlagBits::eColorAttachment |
                                                    vk::ImageUsageFlagBits::eSampled});
        s.view = Check<"HDR meter view">(
            s.device.createImageViewUnique({.image = s.image.image,
                                            .viewType = vk::ImageViewType::e2D,
                                            .format = format,
                                            .subresourceRange = Range}));
        s.extent = extent;
        s.format = format;
    }

    // The slot written three frames ago holds a finished histogram by now.
    const u64 slot = (s.frame % Slots) * HistogramBytes;
    if (s.frame >= Slots) {
        std::array<u32, Bins> bins{};
        std::memcpy(bins.data(), s.readback->mapped_data.data() + slot, HistogramBytes);
        for (u32 i = 0; i < Bins; ++i)
            s.second[i] += bins[i];
    }
    ++s.frame;
    const auto now = std::chrono::steady_clock::now();
    if (now - s.second_start >= std::chrono::seconds{1}) {
        u64 total{};
        for (const auto count : s.second)
            total += count;
        Reading reading{.valid = total > 0};
        if (total > 0) {
            // Percentiles from the top: the bin where the brightest share starts.
            const auto from_top = [&](double share) {
                const u64 wanted = std::max<u64>(1, u64(double(total) * share));
                u64 seen{};
                for (u32 i = Bins; i-- > 0;) {
                    seen += s.second[i];
                    if (seen >= wanted)
                        return BinNits(i);
                }
                return BinNits(0);
            };
            reading.peak = from_top(0.0);
            reading.p999 = from_top(0.001);
            reading.p99 = from_top(0.01);
            reading.median = from_top(0.5);
            LOG_INFO(Render_Vulkan,
                     "[HDR-METER] peak {:.0f} nits, 99.9% {:.0f}, 99% {:.0f}, median {:.0f}",
                     reading.peak, reading.p999, reading.p99, reading.median);
        }
        {
            std::scoped_lock lock{s.mutex};
            s.reading = reading;
        }
        s.second = {};
        s.second_start = now;
    }

    const auto image_barrier = [&](vk::ImageLayout from, vk::ImageLayout to,
                                   vk::PipelineStageFlags2 src, vk::AccessFlags2 src_access,
                                   vk::PipelineStageFlags2 dst, vk::AccessFlags2 dst_access) {
        const vk::ImageMemoryBarrier2 barrier{.srcStageMask = src,
                                              .srcAccessMask = src_access,
                                              .dstStageMask = dst,
                                              .dstAccessMask = dst_access,
                                              .oldLayout = from,
                                              .newLayout = to,
                                              .image = s.image.image,
                                              .subresourceRange = Range};
        command.pipelineBarrier2(
            vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
    };
    const auto buffer_barrier = [&](vk::Buffer buffer, vk::PipelineStageFlags2 src,
                                    vk::AccessFlags2 src_access, vk::PipelineStageFlags2 dst,
                                    vk::AccessFlags2 dst_access) {
        const vk::BufferMemoryBarrier2 barrier{.srcStageMask = src,
                                               .srcAccessMask = src_access,
                                               .dstStageMask = dst,
                                               .dstAccessMask = dst_access,
                                               .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                               .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                               .buffer = buffer,
                                               .size = VK_WHOLE_SIZE};
        command.pipelineBarrier2(
            vk::DependencyInfo{.bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &barrier});
    };
    using Stage = vk::PipelineStageFlagBits2;
    using Access = vk::AccessFlagBits2;

    // The game image as RenoDX's version of the ImGui shader draws it into the window.
    image_barrier(vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
                  Stage::eAllCommands, Access::eMemoryRead, Stage::eColorAttachmentOutput,
                  Access::eColorAttachmentWrite);
    ImGui::Core::RenderGameFrame(command, *s.view, extent);
    image_barrier(vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
                  Stage::eColorAttachmentOutput, Access::eColorAttachmentWrite,
                  Stage::eComputeShader, Access::eShaderRead);

    buffer_barrier(s.histogram->Handle(), Stage::eAllCommands,
                   Access::eMemoryRead | Access::eMemoryWrite, Stage::eTransfer,
                   Access::eTransferWrite);
    command.fillBuffer(s.histogram->Handle(), 0, HistogramBytes, 0);
    buffer_barrier(s.histogram->Handle(), Stage::eTransfer, Access::eTransferWrite,
                   Stage::eComputeShader, Access::eShaderRead | Access::eShaderWrite);

    const vk::DescriptorImageInfo image_info{*s.sampler, *s.view,
                                             vk::ImageLayout::eShaderReadOnlyOptimal};
    const vk::DescriptorBufferInfo buffer_info{s.histogram->Handle(), 0, HistogramBytes};
    const std::array writes{
        vk::WriteDescriptorSet{.dstBinding = 0,
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                               .pImageInfo = &image_info},
        vk::WriteDescriptorSet{.dstBinding = 1,
                               .descriptorCount = 1,
                               .descriptorType = vk::DescriptorType::eStorageBuffer,
                               .pBufferInfo = &buffer_info}};
    command.bindPipeline(vk::PipelineBindPoint::eCompute, *s.pipeline);
    command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *s.layout, 0, writes);
    command.dispatch((extent.width + 15) / 16, (extent.height + 15) / 16, 1);

    buffer_barrier(s.histogram->Handle(), Stage::eComputeShader, Access::eShaderWrite,
                   Stage::eTransfer, Access::eTransferRead);
    command.copyBuffer(s.histogram->Handle(), s.readback->Handle(),
                       vk::BufferCopy{.srcOffset = 0, .dstOffset = slot, .size = HistogramBytes});
    buffer_barrier(s.readback->Handle(), Stage::eTransfer, Access::eTransferWrite, Stage::eHost,
                   Access::eHostRead);
}

Reading Get() {
    if (!state)
        return {};
    std::scoped_lock lock{state->mutex};
    return state->reading;
}

void Shutdown() {
    state.reset();
}

} // namespace Vulkan::HdrMeter
