// SPDX-FileCopyrightText: Copyright 2026 Robert Bopko
// SPDX-License-Identifier: GPL-2.0-or-later
// written by Robert Bopko (github.com/zeroznet) with Boba Bott (Claude Opus 5.5)

#version 450 core

layout(location = 0) out vec4 fColor;
layout(set = 0, binding = 0) uniform sampler2D sTexture;
layout(location = 0) in struct { vec4 Color; vec2 UV; } In;

// Windows SDR white level / 80 nits: ImGui's sRGB colors at the brightness of the SDR desktop.
layout(constant_id = 0) const float sdr_white = 2.5375;

vec3 srgb_to_linear(vec3 rgb) {
    return mix(pow(max(rgb + 0.055, 0.0) / 1.055, vec3(2.4)), rgb / 12.92,
               lessThan(rgb, vec3(0.04045)));
}

void main() {
    vec4 color = In.Color * texture(sTexture, In.UV.st);
    fColor = vec4(srgb_to_linear(color.rgb) * sdr_white, color.a);
}
