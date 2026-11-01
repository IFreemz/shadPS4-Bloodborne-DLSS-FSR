// SPDX-FileCopyrightText: Copyright 2026 IFreemz
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string>
#include <string_view>
#include <vector>
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_bb_frame_gen.h"
#include "video_core/renderer_vulkan/vk_common.h"

// AMD FSR 3.1 frame generation through amd_fidelityfx_vk.dll (libamd_fidelityfx_vk.so on Linux),
// for GPUs without DLSS-G. Like DLSS-G it replaces the swapchain: the emulator renders into its
// images, and it presents a generated frame before each rendered one from its own threads, on
// queues of its own.
namespace Vulkan {
class DlssNgx;
}

namespace Vulkan::FsrFrameGen {

// Used instead of DLSS-G (chosen in FrameGen::Load, before the device is created).
void Select();
bool Selected();
// Without Streamline: what generates the frames this swapchain presents, from dlss.ini
// fg_backend. "auto" uses DLSS-G when the GPU has it, "dlss" only DLSS-G, "fsr" FSR.
void SetGenerator(std::string_view generator);

// The queues it needs besides the emulator's: one to present from, one to acquire images on
// and, if the GPU has one, a compute queue. A family index of ~0u means none.
struct QueuePlan {
    struct Slot {
        u32 family = ~0u;
        u32 index = 0;
    };
    Slot present, acquire, compute;
};
// Picks them next to the emulator's queue (index 0 of `graphics_family`). False when the GPU
// has no other queue that can present; frame generation then stays off. `surface` is the
// window's, for asking which queues present to it (unused on Windows).
bool PlanQueues(vk::Instance instance, vk::PhysicalDevice physical, u32 graphics_family,
                vk::SurfaceKHR surface, QueuePlan& plan);
// Once the device exists (and DLSS is initialized). Loads the DLL.
void SetDevice(vk::PhysicalDevice physical, vk::Device device, vk::Queue game_queue,
               u32 game_family, const QueuePlan& plan, DlssNgx* dlss);

// Selected, loaded and the device has its queues.
bool Active();
// Active, with DLSS-G generating the frames.
bool UsingDlss();
// Why it is not active although it was selected (empty otherwise).
std::string Problem();
// Before the device is destroyed, after the swapchain.
void Shutdown();

// The swapchain functions, while Active().
vk::Result CreateSwapchain(const vk::SwapchainCreateInfoKHR& info, vk::SwapchainKHR& swapchain);
void DestroySwapchain(vk::SwapchainKHR swapchain);
vk::Result GetSwapchainImages(vk::SwapchainKHR swapchain, std::vector<vk::Image>& images);
vk::Result AcquireNextImage(vk::SwapchainKHR swapchain, vk::Semaphore semaphore, u32& index);
// With the emulator's queue, under its submit lock.
vk::Result Present(vk::Queue queue, const vk::PresentInfoKHR& info);

// Presenter thread, before the frame's commands are submitted. `inputs` is null for frames
// without them; `overlays` gives the window without the emulator's overlays.
void SetFrame(const FrameGen::FrameInputs* inputs, const FrameGen::OverlayInputs* overlays,
              vk::CommandBuffer command, vk::Extent2D backbuffer, vk::Rect2D game_area);
// After presenting: frames shown since the last call, generated ones included.
u32 TakeShown();

} // namespace Vulkan::FsrFrameGen
