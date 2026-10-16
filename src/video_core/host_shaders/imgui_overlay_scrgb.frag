// SPDX-FileCopyrightText: Copyright 2026 Robert Bopko
// SPDX-License-Identifier: GPL-2.0-or-later
// written by Robert Bopko (github.com/zeroznet) with Boba Bott (Claude Opus 5.5)

#version 450 core

layout(location = 0) out vec4 fColor;
layout(set = 0, binding = 0) uniform sampler2D sTexture;
layout(location = 0) in struct { vec4 Color; vec2 UV; } In;

// Windows SDR white level / 80 nits: ImGui's sRGB colors at the brightness of the SDR desktop.
layout(constant_id = 0) const float sdr_white = 2.5375;
// HDR10 swapchain: write Rec.2020 PQ instead of scRGB.
layout(constant_id = 1) const bool pq = false;

const mat3 rec709_to_rec2020 = mat3(
    0.6274040, 0.0690970, 0.0163916,
    0.3292820, 0.9195400, 0.0880132,
    0.0433136, 0.0113612, 0.8955950);

// SMPTE ST 2084 of nits / 10000.
vec3 nits_to_pq(vec3 nits) {
    const float m1 = 2610.0 / 16384.0, m2 = 2523.0 / 4096.0 * 128.0;
    const float c1 = 3424.0 / 4096.0, c2 = 2413.0 / 4096.0 * 32.0, c3 = 2392.0 / 4096.0 * 32.0;
    vec3 x = pow(max(nits, 0.0) / 10000.0, vec3(m1));
    return pow((c1 + c2 * x) / (1.0 + c3 * x), vec3(m2));
}

vec3 srgb_to_linear(vec3 rgb) {
    return mix(pow(max(rgb + 0.055, 0.0) / 1.055, vec3(2.4)), rgb / 12.92,
               lessThan(rgb, vec3(0.04045)));
}

void main() {
    vec4 color = In.Color * texture(sTexture, In.UV.st);
    vec3 scrgb = srgb_to_linear(color.rgb) * sdr_white;
    fColor = vec4(pq ? nits_to_pq(rec709_to_rec2020 * scrgb * 80.0) : scrgb, color.a);
}
