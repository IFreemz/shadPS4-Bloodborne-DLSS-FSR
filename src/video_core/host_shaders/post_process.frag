// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#version 450

layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 color;

layout (binding = 0) uniform sampler2D texSampler;

layout (push_constant) uniform settings {
    float gamma;
    bool hdr;
    bool srgb_input;
    uint retro_mode;
    float retro_brightness;
    float retro_sdr_white;
} pp;

const uint RETRO_OFF = 1u, RETRO_ON = 2u;

const float cutoff = 0.0031308, a = 1.055, b = 0.055, d = 12.92;
vec3 gamma(vec3 rgb) {
    return mix(
        a * pow(rgb, vec3(1.0 / (2.4 + 1.0 - pp.gamma))) - b,
        d * rgb / pp.gamma,
        lessThan(rgb, vec3(cutoff))
    );
}

// Exact inverse of gamma() at unit gamma, for buffers that are sRGB encoded but must be sampled
// through a UNORM view because Vulkan has no sRGB variant of the 10-bit format.
vec3 degamma(vec3 rgb) {
    return mix(
        pow(max(rgb + b, 0.0) / a, vec3(2.4)),
        rgb / d,
        lessThan(rgb, vec3(d * cutoff))
    );
}

// SMPTE ST 2084 as Special K's LinearToPQ / PQToLinear: max_value encodes to 1.0.
const float pq_m1 = 2610.0 / 16384.0, pq_m2 = 2523.0 / 4096.0 * 128.0;
const float pq_c1 = 3424.0 / 4096.0, pq_c2 = 2413.0 / 4096.0 * 32.0, pq_c3 = 2392.0 / 4096.0 * 32.0;

vec4 linear_to_pq(vec4 v, float max_value) {
    vec4 x = pow(max(v, 0.0) / max_value, vec4(pq_m1));
    return pow((pq_c1 + pq_c2 * x) / (1.0 + pq_c3 * x), vec4(pq_m2));
}

vec4 pq_to_linear(vec4 code, float max_value) {
    vec4 x = pow(max(code, 0.0), vec4(1.0 / pq_m2));
    return pow(max(x - pq_c1, 0.0) / (pq_c2 - pq_c3 * x), vec4(1.0 / pq_m1)) * max_value;
}

// BT.709 primaries, D65, as Special K's Rec709_to_XYZ / XYZ_to_Rec709 (column-major).
const mat3 rec709_to_xyz = mat3(
    0.4123908, 0.2126390, 0.0193308,
    0.3575843, 0.7151687, 0.1191948,
    0.1804808, 0.0721923, 0.9505321);
const mat3 xyz_to_rec709 = mat3(
    3.2409699, -0.9692436, 0.0556301,
    -1.5373832, 1.8759675, -0.2039770,
    -0.4986108, 0.0415551, 1.0569715);

// Special K Perceptual Boost v1 with 33% Color Boost. SDR input keeps XYZ below 1.09, far from
// where the PQ inverse diverges (code 1.98).
vec3 perceptual_boost(vec3 rgb) {
    vec3 xyz = max(rec709_to_xyz * rgb, 0.0);
    if (xyz.y <= 0.0) {
        return vec3(0.0);
    }
    vec4 boosted = pq_to_linear(linear_to_pq(vec4(xyz, xyz.y), 1.0) * 1.273, 0.1) / 0.5;
    return xyz_to_rec709 * mix(xyz * boosted.w / xyz.y, boosted.xyz, 0.333);
}

void main() {
    vec4 color_linear = texture(texSampler, uv);
    if (pp.hdr) {
        color = color_linear;
    } else {
        if (pp.srgb_input) color_linear.rgb = degamma(color_linear.rgb);
        vec3 encoded = gamma(color_linear.rgb);
        if (pp.retro_mode == RETRO_ON) {
            vec3 sdr = pow(clamp(encoded, 0.0, 1.0), vec3(2.2));
            color = vec4(perceptual_boost(sdr) * pp.retro_brightness, color_linear.a);
        } else if (pp.retro_mode == RETRO_OFF) {
            color = vec4(degamma(clamp(encoded, 0.0, 1.0)) * pp.retro_sdr_white, color_linear.a);
        } else {
            color = vec4(encoded, color_linear.a);
        }
    }
}
