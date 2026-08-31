#pragma once

// Complex-IOR (eta, k) preset table for the Conductor material.
//
// RGB triples sampled at hero wavelengths R=630nm, G=530nm, B=460nm from
// PBRT v4's piecewise-linear measured spectra (src/pbrt/util/spectrum.cpp).
// Values frozen at compile time — no runtime spectral sampling.
//
// Companion to `MaterialData::attenuation` (overloaded as conductor_eta when
// `conductor_k != 0`) and `MaterialData::conductor_k`. Use `make_conductor_metal`
// or `make_conductor_ior` to populate both fields.
//
// Cross-checked against published normal-incidence reflectance values:
//   Au  R(0) ≈ 0.93 R / 0.71 G / 0.42 B  (computed: 0.93 / 0.76 / 0.39)
//   Ag  R(0) ≈ 0.97 R / 0.95 G / 0.93 B  (computed: 0.97 / 0.95 / 0.93)
//   Cu  R(0) ≈ 0.92 R / 0.61 G / 0.56 B  (computed: 0.92 / 0.61 / 0.56)
//   Al  R(0) ≈ 0.91 R / 0.92 G / 0.92 B  (computed: 0.91 / 0.92 / 0.92)
//   CuZn R(0) ≈ 0.88 R / 0.75 G / 0.47 B (computed: 0.88 / 0.75 / 0.47)

#include "newtype/render/Material.h"

namespace newtype::render {

enum class MetalPreset : uint {
    Custom   = 0,  // no preset — keep current (eta, k) values
    Gold     = 1,  // Au
    Silver   = 2,  // Ag
    Copper   = 3,  // Cu
    Aluminum = 4,  // Al
    Brass    = 5,  // CuZn
};

struct MetalSample {
    luisa::float3 eta;
    luisa::float3 k;
};

// Frozen RGB (eta, k) at hero wavelengths (630 / 530 / 460 nm).
// Source: PBRT v4 PiecewiseLinearSpectrum::Sample on {Au,Ag,Cu,Al,CuZn}_eta/k.
inline constexpr MetalSample kMetalPresets[] = {
    {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}},                                       // Custom (unused placeholder)
    {{0.183696f, 0.472909f, 1.418608f}, {3.066341f, 2.371123f, 1.843105f}},    // Gold (Au)
    {{0.134250f, 0.129840f, 0.143383f}, {3.962828f, 3.178400f, 2.567381f}},    // Silver (Ag)
    {{0.245884f, 1.081429f, 1.159567f}, {3.378350f, 2.592925f, 2.436339f}},    // Copper (Cu)
    {{1.357068f, 0.876447f, 0.646272f}, {7.587795f, 6.447291f, 5.590148f}},    // Aluminum (Al)
    {{0.445000f, 0.573000f, 0.994000f}, {3.522000f, 2.568000f, 1.883000f}},    // Brass (CuZn)
};

[[nodiscard]] inline MetalSample lookup_metal_preset(MetalPreset preset) noexcept {
    return kMetalPresets[static_cast<uint>(preset)];
}

/// Build a Conductor material from a preset.
[[nodiscard]] inline MaterialData make_conductor_metal(
    MetalPreset preset,
    float roughness = 0.3f,
    int albedoTexIdx = -1,
    int normalTexIdx = -1,
    int rmaTexIdx = -1) noexcept {
    ConductorParams p;
    const MetalSample s = lookup_metal_preset(preset);
    p.albedo       = luisa::float3{1.f, 1.f, 1.f};  // placeholder; complex IOR drives Fresnel
    p.roughness    = roughness;
    p.albedoTexIdx = albedoTexIdx;
    p.normalTexIdx = normalTexIdx;
    p.rmaTexIdx    = rmaTexIdx;
    p.eta          = s.eta;
    p.k            = s.k;
    return p.to_data();
}

/// Build a Conductor material from raw complex-IOR RGB triples.
[[nodiscard]] inline MaterialData make_conductor_ior(
    const luisa::float3& eta,
    const luisa::float3& k,
    float roughness = 0.3f,
    int albedoTexIdx = -1,
    int normalTexIdx = -1,
    int rmaTexIdx = -1) noexcept {
    ConductorParams p;
    p.albedo       = luisa::float3{1.f, 1.f, 1.f};  // placeholder; complex IOR drives Fresnel
    p.roughness    = roughness;
    p.albedoTexIdx = albedoTexIdx;
    p.normalTexIdx = normalTexIdx;
    p.rmaTexIdx    = rmaTexIdx;
    p.eta          = eta;
    p.k            = k;
    return p.to_data();
}

} // namespace newtype::render
