// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <boost/container/static_vector.hpp>

#include <boost/container/small_vector.hpp>
#include "shader_recompiler/resource.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_pipeline_common.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

Pipeline::Pipeline(const Instance& instance_, Scheduler& scheduler_, DescriptorHeap& desc_heap_,
                   const Shader::Profile& profile_, vk::PipelineCache pipeline_cache,
                   bool is_compute_ /*= false*/)
    : instance{instance_}, scheduler{scheduler_}, desc_heap{desc_heap_}, profile{profile_},
      is_compute{is_compute_} {}

Pipeline::~Pipeline() = default;

namespace {
/// A copy of descriptor writes and the infos they point to, for deferred recording.
struct PushedWrites {
    boost::container::small_vector<vk::WriteDescriptorSet, 24> writes;
    boost::container::small_vector<vk::DescriptorImageInfo, 24> images;
    boost::container::small_vector<vk::DescriptorBufferInfo, 16> buffers;
    boost::container::small_vector<vk::BufferView, 4> texel_views;

    explicit PushedWrites(const Pipeline::DescriptorWrites& source) {
        writes.assign(source.begin(), source.end());
        for (const auto& write : source) {
            if (write.pImageInfo) {
                images.insert(images.end(), write.pImageInfo,
                              write.pImageInfo + write.descriptorCount);
            }
            if (write.pBufferInfo) {
                buffers.insert(buffers.end(), write.pBufferInfo,
                               write.pBufferInfo + write.descriptorCount);
            }
            if (write.pTexelBufferView) {
                texel_views.insert(texel_views.end(), write.pTexelBufferView,
                                   write.pTexelBufferView + write.descriptorCount);
            }
        }
    }

    /// Points the writes at this copy's infos (after it reached its final address).
    void Fixup() {
        u32 image = 0, buffer = 0, texel = 0;
        for (auto& write : writes) {
            if (write.pImageInfo) {
                write.pImageInfo = images.data() + image;
                image += write.descriptorCount;
            }
            if (write.pBufferInfo) {
                write.pBufferInfo = buffers.data() + buffer;
                buffer += write.descriptorCount;
            }
            if (write.pTexelBufferView) {
                write.pTexelBufferView = texel_views.data() + texel;
                texel += write.descriptorCount;
            }
        }
    }
};
} // Anonymous namespace

void Pipeline::BindResources(DescriptorWrites& set_writes,
                             const Shader::PushData& push_data) const {
    const auto bind_point =
        IsCompute() ? vk::PipelineBindPoint::eCompute : vk::PipelineBindPoint::eGraphics;
    const auto layout = *pipeline_layout;

    const auto stage_flags = IsCompute() ? vk::ShaderStageFlagBits::eCompute : AllGraphicsStageBits;
    scheduler.Record([layout, stage_flags, push_data](vk::CommandBuffer cmdbuf) {
        cmdbuf.pushConstants(layout, stage_flags, 0u, sizeof(push_data), &push_data);
    });

    // Bind descriptor set.
    if (set_writes.empty()) {
        return;
    }

    if (uses_push_descriptors) {
        scheduler.Record([bind_point, layout,
                          pushed = PushedWrites{set_writes}](vk::CommandBuffer cmdbuf) mutable {
            pushed.Fixup();
            cmdbuf.pushDescriptorSetKHR(bind_point, layout, 0, pushed.writes);
        });
        return;
    }

    const auto desc_set = desc_heap.Commit(*desc_layout);
    for (auto& set_write : set_writes) {
        set_write.dstSet = desc_set;
    }
    instance.GetDevice().updateDescriptorSets(set_writes, {});
    scheduler.Record([bind_point, layout, desc_set](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindDescriptorSets(bind_point, layout, 0, desc_set, {});
    });
}

std::string Pipeline::GetDebugString() const {
    std::string stage_desc;
    for (const auto& stage : stages) {
        if (stage) {
            const auto shader_name = PipelineCache::GetShaderName(stage->hw_stage, stage->pgm_hash);
            if (stage_desc.empty()) {
                stage_desc = shader_name;
            } else {
                stage_desc = fmt::format("{},{}", stage_desc, shader_name);
            }
        }
    }
    return stage_desc;
}

} // namespace Vulkan
