// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
#version 450
#extension GL_EXT_samplerless_texture_functions : require

// Occlusion tests tolerant of a pixel: the scene's depth with each pixel set to the farthest
// depth of itself and its neighbours, so a gap of about a pixel stays open whatever the
// sub-pixel camera jitter or movement does from one frame to the next.
layout(binding = 0) uniform texture2D source;

layout(push_constant) uniform Push {
    uint farthest_is_max; // the test passes for smaller depths (LESS / LESS_EQUAL)
    int radius;
} pc;

void main() {
    const ivec2 size = textureSize(source, 0);
    const ivec2 center = ivec2(gl_FragCoord.xy);
    float depth = texelFetch(source, clamp(center, ivec2(0), size - 1), 0).r;
    for (int y = -pc.radius; y <= pc.radius; ++y) {
        for (int x = -pc.radius; x <= pc.radius; ++x) {
            const float d = texelFetch(source, clamp(center + ivec2(x, y), ivec2(0), size - 1), 0).r;
            depth = pc.farthest_is_max != 0u ? max(depth, d) : min(depth, d);
        }
    }
    gl_FragDepth = depth;
}
