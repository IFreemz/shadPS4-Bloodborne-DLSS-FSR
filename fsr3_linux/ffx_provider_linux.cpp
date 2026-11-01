// This file is part of the FidelityFX SDK.
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

//
// Linux build of ffx_provider.cpp (FidelityFX SDK v1.1.4): the FSR 3.1 upscaler and frame
// generation, without the AMD driver's external providers (a Windows D3D12 extension).

#include "ffx_provider.h"

#include "ffx_provider_framegeneration.h"
#include "ffx_provider_fsr3upscale.h"
#include "vk/ffx_provider_framegenerationswapchain_vk.h"

#include <iterator>

static constexpr ffxProvider* providers[] = {
    &ffxProvider_FSR3Upscale::Instance,
    &ffxProvider_FrameGeneration::Instance,
    &ffxProvider_FrameGenerationSwapChain_VK::Instance,
};

const ffxProvider* GetffxProvider(ffxStructType_t descType, uint64_t overrideId, void*)
{
    for (const ffxProvider* provider : providers)
    {
        if ((!overrideId || provider->GetId() == overrideId) && provider->CanProvide(descType))
            return provider;
    }
    return nullptr;
}

const ffxProvider* GetAssociatedProvider(ffxContext* context)
{
    const InternalContextHeader* hdr = (const InternalContextHeader*)(*context);
    return hdr->provider;
}

uint64_t GetProviderVersions(ffxStructType_t descType, void*, uint64_t capacity, uint64_t* versionIds,
                             const char** versionNames)
{
    uint64_t count = 0;
    for (const ffxProvider* provider : providers)
    {
        if (count >= capacity)
            break;
        if (provider->CanProvide(descType))
        {
            if (versionIds)
                versionIds[count] = provider->GetId();
            if (versionNames)
                versionNames[count] = provider->GetVersionName();
            count++;
        }
    }
    return count;
}

uint64_t GetProviderCount(ffxStructType_t descType, void* device)
{
    return GetProviderVersions(descType, device, UINT64_MAX, nullptr, nullptr);
}
