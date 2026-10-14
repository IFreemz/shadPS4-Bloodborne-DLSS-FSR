// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#ifdef _WIN64
// For exclusive fullscreen (VK_EXT_full_screen_exclusive).
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#include <algorithm>
#include <limits>
#include "common/assert.h"
#include "common/logging/log.h"
#include "core/emulator_settings.h"
#include "imgui/renderer/imgui_core.h"
#include "sdl_window.h"
#include "video_core/renderer_vulkan/vk_bb_frame_gen.h"
#include "video_core/renderer_vulkan/vk_bb_fsr_frame_gen.h"
#include "video_core/renderer_vulkan/vk_hdr_mod.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_retro_hdr.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"

namespace Vulkan {

static constexpr vk::SurfaceFormatKHR SURFACE_FORMAT_HDR = {
    .format = vk::Format::eA2B10G10R10UnormPack32,
    .colorSpace = vk::ColorSpaceKHR::eHdr10St2084EXT,
};

Swapchain::Swapchain(const Instance& instance_, const Frontend::WindowSDL& window_)
    : instance{instance_}, window{window_}, surface{CreateSurface(instance.GetInstance(), window)} {
    FindPresentFormat();
    FindPresentMode();

    Create(window.GetWidth(), window.GetHeight());
    ImGui::Core::Initialize(instance, window, image_count, surface_format.format);
}

Swapchain::~Swapchain() {
    Destroy();
    instance.GetInstance().destroySurfaceKHR(surface);
}

void Swapchain::Create(u32 width_, u32 height_) {
    width = width_;
    height = height_;
    needs_recreation = false;

    Destroy();

    SetSurfaceProperties();

    const std::array queue_family_indices = {
        instance.GetGraphicsQueueFamilyIndex(),
        instance.GetPresentQueueFamilyIndex(),
    };

    const bool exclusive = queue_family_indices[0] == queue_family_indices[1];
    const u32 queue_family_indices_count = exclusive ? 1u : 2u;
    const vk::SharingMode sharing_mode =
        exclusive ? vk::SharingMode::eExclusive : vk::SharingMode::eConcurrent;
    const auto format = needs_hdr ? SURFACE_FORMAT_HDR : surface_format;
    const vk::SwapchainCreateInfoKHR swapchain_info = {
        .surface = surface,
        .minImageCount = image_count,
        .imageFormat = format.format,
        .imageColorSpace = format.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment |
                      vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
        .imageSharingMode = sharing_mode,
        .queueFamilyIndexCount = queue_family_indices_count,
        .pQueueFamilyIndices = queue_family_indices.data(),
        .preTransform = transform,
        .compositeAlpha = composite_alpha,
        .presentMode = present_mode,
        .clipped = true,
        .oldSwapchain = nullptr,
    };

#ifdef _WIN64
    // With frame generation, take the display exclusively: an overlay window on top of a
    // borderless one makes Windows compose the desktop, which drops the generated frames.
    exclusive_fullscreen = !FsrFrameGen::Active() && instance.IsFullScreenExclusiveSupported() &&
                           EmulatorSettings.IsFullScreen() &&
                           EmulatorSettings.GetFullScreenMode() == "Fullscreen";
    vk::SurfaceFullScreenExclusiveWin32InfoEXT exclusive_monitor{
        .hmonitor = MonitorFromWindow(static_cast<HWND>(window.GetWindowInfo().render_surface),
                                      MONITOR_DEFAULTTONEAREST)};
    vk::SurfaceFullScreenExclusiveInfoEXT exclusive_info{
        .pNext = &exclusive_monitor,
        .fullScreenExclusive = vk::FullScreenExclusiveEXT::eApplicationControlled};
    auto exclusive_swapchain_info = swapchain_info;
    if (exclusive_fullscreen)
        exclusive_swapchain_info.pNext = &exclusive_info;
    vk::SwapchainKHR chain{};
    const auto swapchain_result =
        FsrFrameGen::Active()
            ? FsrFrameGen::CreateSwapchain(swapchain_info, chain)
            : instance.GetDevice().createSwapchainKHR(&exclusive_swapchain_info, nullptr, &chain);
#else
    auto [swapchain_result, chain] = instance.GetDevice().createSwapchainKHR(swapchain_info);
#endif
    ASSERT_MSG(swapchain_result == vk::Result::eSuccess, "Failed to create swapchain: {}",
               vk::to_string(swapchain_result));
    swapchain = chain;
    exclusive_held = false;
    exclusive_retry = 0;
    AcquireExclusive();

    SetupImages();
    RefreshSemaphores();
}

void Swapchain::AcquireExclusive() {
#ifdef _WIN64
    if (!exclusive_fullscreen || exclusive_held)
        return;
    // Retried about once a second while it fails (the window is not in front).
    if (exclusive_retry++ % 60 != 0)
        return;
    const auto device = instance.GetDevice();
    const auto acquire = reinterpret_cast<PFN_vkAcquireFullScreenExclusiveModeEXT>(
        device.getProcAddr("vkAcquireFullScreenExclusiveModeEXT"));
    if (!acquire) {
        LOG_WARNING(Render_Vulkan,
                    "Exclusive fullscreen: vkAcquireFullScreenExclusiveModeEXT missing");
        exclusive_fullscreen = false;
        return;
    }
    const auto result = acquire(device, swapchain);
    exclusive_held = result == VK_SUCCESS;
    LOG_INFO(Render_Vulkan, "Exclusive fullscreen: {}", vk::to_string(vk::Result{result}));
#endif
}

void Swapchain::Recreate(u32 width_, u32 height_) {
    FrameGen::Pause();
    LOG_DEBUG(Render_Vulkan, "Recreate the swapchain: width={} height={} HDR={}", width_, height_,
              needs_hdr);
    Create(width_, height_);
}

void Swapchain::SetHDR(bool hdr) {
    if (needs_hdr == hdr) {
        return;
    }

    auto result = instance.GetDevice().waitIdle();
    if (result != vk::Result::eSuccess) {
        LOG_WARNING(ImGui, "Failed to wait for Vulkan device idle on mode change: {}",
                    vk::to_string(result));
    }

    needs_hdr = hdr;
    Recreate(width, height);
    ImGui::Core::OnSurfaceFormatChange(needs_hdr ? SURFACE_FORMAT_HDR.format
                                                 : surface_format.format);
}

bool Swapchain::AcquireNextImage() {
    AcquireExclusive();
    vk::Device device = instance.GetDevice();
    vk::Result result =
        FsrFrameGen::Active()
            ? FsrFrameGen::AcquireNextImage(swapchain, image_acquired[frame_index], image_index)
            : device.acquireNextImageKHR(swapchain, std::numeric_limits<u64>::max(),
                                         image_acquired[frame_index], VK_NULL_HANDLE, &image_index);

    switch (result) {
    case vk::Result::eSuccess:
        break;
    case vk::Result::eErrorFullScreenExclusiveModeLostEXT:
        exclusive_held = false;
        needs_recreation = true;
        break;
    case vk::Result::eSuboptimalKHR:
    case vk::Result::eErrorSurfaceLostKHR:
    case vk::Result::eErrorOutOfDateKHR:
    case vk::Result::eErrorUnknown:
        needs_recreation = true;
        break;
    default:
        // Streamline presents asynchronously and reports its own errors here.
        if (FrameGen::Active()) {
            LOG_WARNING(Render_Vulkan, "Swapchain acquire returned {}", vk::to_string(result));
            needs_recreation = true;
            break;
        }
        LOG_CRITICAL(Render_Vulkan, "Swapchain acquire returned unknown result {}",
                     vk::to_string(result));
        UNREACHABLE();
        break;
    }

    return !needs_recreation;
}

bool Swapchain::Present() {
    const vk::PresentInfoKHR present_info = {
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &present_ready[image_index],
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = &image_index,
    };

    auto result = FsrFrameGen::Active()
                      ? FsrFrameGen::Present(instance.GetPresentQueue(), present_info)
                      : instance.GetPresentQueue().presentKHR(present_info);
    if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR) {
        needs_recreation = true;
    } else if (result == vk::Result::eErrorFullScreenExclusiveModeLostEXT) {
        exclusive_held = false;
        needs_recreation = true;
    } else if (result != vk::Result::eSuccess && FrameGen::Active()) {
        // Streamline presents asynchronously and reports a failed present on a later one.
        LOG_WARNING(Render_Vulkan, "Swapchain presentation returned {}", vk::to_string(result));
        needs_recreation = true;
    } else {
        ASSERT_MSG(result == vk::Result::eSuccess, "Swapchain presentation failed: {}",
                   vk::to_string(result));
    }

    frame_index = (frame_index + 1) % image_count;

    return !needs_recreation;
}

void Swapchain::FindPresentFormat() {
    const auto [formats_result, formats] =
        instance.GetPhysicalDevice().getSurfaceFormatsKHR(surface);
    ASSERT_MSG(formats_result == vk::Result::eSuccess, "Failed to query surface formats: {}",
               vk::to_string(formats_result));

    // Check if the device supports HDR formats. Here we care of Rec.2020 PQ only as it is expected
    // game output. Other variants as e.g. linear Rec.2020 will require additional color space
    // rotation
    supports_hdr =
        std::find_if(formats.begin(), formats.end(), [](const vk::SurfaceFormatKHR& format) {
            return format == SURFACE_FORMAT_HDR;
        }) != formats.end();
    // Also make sure that user allowed us to use HDR
    supports_hdr &= EmulatorSettings.IsHdrAllowed();

    // If there is a single undefined surface format, the device doesn't care, so we'll just use
    // RGBA sRGB.
    if (formats[0].format == vk::Format::eUndefined) {
        surface_format.format = vk::Format::eR8G8B8A8Unorm;
        surface_format.colorSpace = vk::ColorSpaceKHR::eSrgbNonlinear;
        return;
    }

    // RenoDX turns the swapchain into scRGB below us. Asking for it here instead keeps the
    // format known to everything above the layers, frame generation included. RetroHDR draws
    // its SDR to HDR output into the same format.
    constexpr vk::SurfaceFormatKHR scrgb{.format = vk::Format::eR16G16B16A16Sfloat,
                                         .colorSpace = vk::ColorSpaceKHR::eExtendedSrgbLinearEXT};
    const bool scrgb_supported = std::ranges::find(formats, scrgb) != formats.end();
    const bool retro_hdr = RetroHdr::Start(window.GetWindowInfo().render_surface, scrgb_supported);
    if (RenoDxLoaded() && scrgb_supported) {
        LOG_INFO(Render_Vulkan, "RenoDX loaded: scRGB swapchain");
        surface_format = scrgb;
        return;
    }
    if (retro_hdr) {
        surface_format = scrgb;
        return;
    }

    // Try to find a suitable format.
    for (const vk::SurfaceFormatKHR& sformat : formats) {
        vk::Format format = sformat.format;
        if (format != vk::Format::eR8G8B8A8Unorm && format != vk::Format::eB8G8R8A8Unorm) {
            continue;
        }

        surface_format.format = format;
        surface_format.colorSpace = sformat.colorSpace;
        return;
    }

    UNREACHABLE_MSG("Unable to find required swapchain format!");
}

void Swapchain::FindPresentMode() {
    const auto [modes_result, modes] =
        instance.GetPhysicalDevice().getSurfacePresentModesKHR(surface);
    if (modes_result != vk::Result::eSuccess) {
        LOG_ERROR(Render, "Failed to query available present modes, falling back to Fifo as "
                          "guaranteed supported option.");
        present_mode = vk::PresentModeKHR::eFifo;
        return;
    }

    const auto requested_mode = EmulatorSettings.GetPresentMode();
    // Streamline's Vulkan DLSS-G path does not support VSync. Keep the guest vblank
    // clock, but do not add a display-refresh pacer to the generated host frames.
    if (FrameGen::Active() && !FsrFrameGen::Active() &&
        std::ranges::find(modes, vk::PresentModeKHR::eImmediate) != modes.cend()) {
        present_mode = vk::PresentModeKHR::eImmediate;
        LOG_INFO(Render_Vulkan, "[FRAME-GEN] DLSS-G uses Immediate presentation (configured {})",
                 requested_mode);
        return;
    }
    if (requested_mode == "Mailbox") {
        present_mode = vk::PresentModeKHR::eMailbox;
    } else if (requested_mode == "Fifo") {
        present_mode = vk::PresentModeKHR::eFifo;
    } else if (requested_mode == "Immediate") {
        present_mode = vk::PresentModeKHR::eImmediate;
    } else {
        LOG_ERROR(Render_Vulkan, "Unknown present mode {}, defaulting to Mailbox.",
                  EmulatorSettings.GetPresentMode());
        present_mode = vk::PresentModeKHR::eMailbox;
    }

    if (std::ranges::find(modes, present_mode) == modes.cend()) {
        // FIFO is guaranteed to be supported by the Vulkan spec.
        constexpr auto fallback = vk::PresentModeKHR::eFifo;
        LOG_WARNING(Render, "Requested present mode {} is not supported, falling back to {}.",
                    vk::to_string(present_mode), vk::to_string(fallback));
        present_mode = fallback;
    }
}

void Swapchain::SetSurfaceProperties() {
    const auto [capabilities_result, capabilities] =
        instance.GetPhysicalDevice().getSurfaceCapabilitiesKHR(surface);
    ASSERT_MSG(capabilities_result == vk::Result::eSuccess,
               "Failed to query surface capabilities: {}", vk::to_string(capabilities_result));

    extent = capabilities.currentExtent;
    if (capabilities.currentExtent.width == std::numeric_limits<u32>::max()) {
        extent.width = std::max(capabilities.minImageExtent.width,
                                std::min(capabilities.maxImageExtent.width, width));
        extent.height = std::max(capabilities.minImageExtent.height,
                                 std::min(capabilities.maxImageExtent.height, height));
    }

    // Select number of images in swap chain, we prefer one buffer in the background to work on
    image_count = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0) {
        image_count = std::min(image_count, capabilities.maxImageCount);
    }

    // Prefer identity transform if possible
    transform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
    if (!(capabilities.supportedTransforms & transform)) {
        transform = capabilities.currentTransform;
    }

    // Opaque is not supported everywhere.
    composite_alpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    if (!(capabilities.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::eOpaque)) {
        composite_alpha = vk::CompositeAlphaFlagBitsKHR::eInherit;
    }
}

void Swapchain::Destroy() {
    vk::Device device = instance.GetDevice();
    const auto wait_result = device.waitIdle();
    if (wait_result != vk::Result::eSuccess) {
        LOG_WARNING(Render_Vulkan, "Failed to wait for device to become idle: {}",
                    vk::to_string(wait_result));
    }

    for (auto& image_view : images_view) {
        device.destroyImageView(image_view);
    }
    images_view.clear();

    if (swapchain) {
        if (FsrFrameGen::Active())
            FsrFrameGen::DestroySwapchain(swapchain);
        else
            device.destroySwapchainKHR(swapchain);
        swapchain = nullptr;
    }

    for (const auto& sem : image_acquired) {
        device.destroySemaphore(sem);
    }
    for (const auto& sem : present_ready) {
        device.destroySemaphore(sem);
    }

    image_acquired.clear();
    present_ready.clear();
}

void Swapchain::RefreshSemaphores() {
    const vk::Device device = instance.GetDevice();
    image_acquired.resize(image_count);
    present_ready.resize(image_count);

    for (vk::Semaphore& semaphore : image_acquired) {
        auto [semaphore_result, sem] = device.createSemaphore({});
        ASSERT_MSG(semaphore_result == vk::Result::eSuccess,
                   "Failed to create image acquired semaphore: {}",
                   vk::to_string(semaphore_result));
        semaphore = sem;
    }
    for (vk::Semaphore& semaphore : present_ready) {
        auto [semaphore_result, sem] = device.createSemaphore({});
        ASSERT_MSG(semaphore_result == vk::Result::eSuccess,
                   "Failed to create present ready semaphore: {}", vk::to_string(semaphore_result));
        semaphore = sem;
    }

    for (u32 i = 0; i < image_count; ++i) {
        SetObjectName(device, image_acquired[i], "Swapchain Semaphore: image_acquired {}", i);
        SetObjectName(device, present_ready[i], "Swapchain Semaphore: present_ready {}", i);
    }
}

void Swapchain::SetupImages() {
    vk::Device device = instance.GetDevice();
    std::vector<vk::Image> imgs;
    vk::Result images_result;
    if (FsrFrameGen::Active()) {
        images_result = FsrFrameGen::GetSwapchainImages(swapchain, imgs);
    } else {
        auto [result, chain_images] = device.getSwapchainImagesKHR(swapchain);
        images_result = result;
        imgs = std::move(chain_images);
    }
    ASSERT_MSG(images_result == vk::Result::eSuccess, "Failed to create swapchain images: {}",
               vk::to_string(images_result));
    images = std::move(imgs);
    image_count = static_cast<u32>(images.size());
    images_view.resize(image_count);
    for (u32 i = 0; i < image_count; ++i) {
        if (images_view[i]) {
            device.destroyImageView(images_view[i]);
        }
        auto [im_view_result, im_view] = device.createImageView(vk::ImageViewCreateInfo{
            .image = images[i],
            .viewType = vk::ImageViewType::e2D,
            .format = needs_hdr ? SURFACE_FORMAT_HDR.format : surface_format.format,
            .subresourceRange =
                {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .levelCount = 1,
                    .layerCount = 1,
                },
        });
        ASSERT_MSG(im_view_result == vk::Result::eSuccess, "Failed to create image view: {}",
                   vk::to_string(im_view_result));
        images_view[i] = im_view;
    }

    for (u32 i = 0; i < image_count; ++i) {
        SetObjectName(device, images[i], "Swapchain Image {}", i);
        SetObjectName(device, images_view[i], "Swapchain ImageView {}", i);
    }
}

} // namespace Vulkan
