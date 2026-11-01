// SPDX-FileCopyrightText: Copyright 2026 IFreemz
// SPDX-License-Identifier: MIT
//
// ffx-api only receives the device, so FSR-Vulkan's backend treats optional features as off.
// shadPS4 enables FP16, subgroup size control and synchronization2 whenever the GPU supports
// them, as AMD's Windows DLL assumes: report those, so FSR uses its fast shader variants.

#include <FidelityFX/host/backends/vk/ffx_vk.h>

void ffxLinuxEnabledFeatures(VkDeviceContext* context)
{
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context->vkPhysicalDevice, &properties);
    if (properties.apiVersion < VK_API_VERSION_1_3)
        return;
    VkPhysicalDeviceVulkan13Features features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features features12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &features13};
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &features12};
    vkGetPhysicalDeviceFeatures2(context->vkPhysicalDevice, &features);
    context->shaderFloat16Enabled = features12.shaderFloat16;
    context->subgroupSizeControlEnabled = features13.subgroupSizeControl;
    context->synchronization2Enabled = features13.synchronization2;
}
