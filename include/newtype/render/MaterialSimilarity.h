#pragma once

// Material-similarity test for ReSTIR GI reuse.
//
// Mirrors RTXDI RAB_AreMaterialsSimilar (RAB_Material.hlsli:112-128):
//   - roughness:  relative difference via RTXDI_CompareRelativeDifference (Math.hlsli:20-23)
//   - F0:         absolute luminance delta; F0 derived inline via Schlick
//                 (dielectric 0.04 lerped to albedo by metallic)
//   - albedo:     absolute luminance delta
//
// Takes two Var<MaterialData> so the caller can hoist the current pixel's
// material read out of the candidate loop (1 read per pixel, not N reads).
//
// This callable performs NO MaterialBSDF method calls — only plain struct
// field reads + DSL arithmetic. Honors feedback-bsdf-pdf-perturbs-reuse-loops.md.

#include <luisa/dsl/sugar.h>
#include "newtype/render/Material.h"
#include "newtype/render/Shading.h"

namespace newtype::render {

// Host-side packed key for the material-similarity gate: the three scalar
// quantities are_materials_similar_keys compares. F0 luminance mirrors the
// device formula in are_materials_similar_gi — complex-IOR normal-incidence
// reflectance when conductor_k != 0, Schlick lerp otherwise — so key-based
// gating matches full-MaterialData gating up to float evaluation order.
//
// Stored in MaterialPool's sim-key buffer so ReSTIR neighbor gates read 12
// bytes instead of a 176-byte MaterialData per comparison.
[[nodiscard]] inline luisa::float3 material_similarity_key(const MaterialData& m) noexcept {
    const auto lum = [](const luisa::float3& c) noexcept {
        return luisa::dot(c, luisa::make_float3(0.2126f, 0.7152f, 0.0722f));
    };
    const bool has_k = m.conductor_k.x != 0.0f || m.conductor_k.y != 0.0f || m.conductor_k.z != 0.0f;
    luisa::float3 f0;
    if (has_k) {
        const auto sep = (m.attenuation - 1.0f) * (m.attenuation - 1.0f) + m.conductor_k * m.conductor_k;
        const auto den = (m.attenuation + 1.0f) * (m.attenuation + 1.0f) + m.conductor_k * m.conductor_k;
        f0 = sep / den;
    } else {
        f0 = (1.0f - m.metallic) * luisa::make_float3(0.04f) + m.metallic * m.albedo.xyz();
    }
    return luisa::make_float3(m.roughness, lum(f0), lum(m.albedo.xyz()));
}

// Key-based comparison — identical threshold semantics to
// are_materials_similar_gi (relative roughness, absolute F0 / albedo deltas).
[[nodiscard]] inline luisa::compute::Bool are_materials_similar_keys(
    luisa::compute::Float3 key_a,
    luisa::compute::Float3 key_b,
    luisa::compute::Float roughThresh,
    luisa::compute::Float f0Thresh,
    luisa::compute::Float albedoThresh
) noexcept {
    using namespace luisa::compute;

    // Roughness — RTXDI relative-difference form:
    //   abs(a - b) <= thresh * max(a, b)
    // thresh <= 0 is RTXDI's "always pass" sentinel (Math.hlsli:22).
    Float rmax = max(key_a.x, key_b.x);
    Bool rough_ok = (roughThresh <= 0.0f)
                  | (abs(key_a.x - key_b.x) <= roughThresh * rmax);

    Bool f0_ok = abs(key_a.y - key_b.y) <= f0Thresh;
    Bool albedo_ok = abs(key_a.z - key_b.z) <= albedoThresh;

    return rough_ok & f0_ok & albedo_ok;
}

[[nodiscard]] inline luisa::compute::Bool are_materials_similar_gi(
    luisa::compute::Var<MaterialData> a,
    luisa::compute::Var<MaterialData> b,
    luisa::compute::Float roughThresh,
    luisa::compute::Float f0Thresh,
    luisa::compute::Float albedoThresh
) noexcept {
    using namespace luisa::compute;

    // Roughness — RTXDI relative-difference form:
    //   abs(a - b) <= thresh * max(a, b)
    // thresh <= 0 is RTXDI's "always pass" sentinel (Math.hlsli:22).
    Float rmax = max(a.roughness, b.roughness);
    Bool rough_ok = (roughThresh <= 0.0f)
                  | (abs(a.roughness - b.roughness) <= roughThresh * rmax);

    // F0 via Schlick approximation OR complex-IOR normal-incidence reflectance.
    // Complex F0 at normal incidence: ((eta-1)^2 + k^2) / ((eta+1)^2 + k^2)
    // where eta = attenuation (overloaded storage) and k = conductor_k.
    // Sentinel: k != 0 marks a Conductor material (complex-IOR-only). k = 0
    // occurs only on non-Conductor types, which use the Schlick F0 path.
    Float3 eta_a = a.attenuation;
    Float3 k_a   = a.conductor_k;
    Float3 eta_b = b.attenuation;
    Float3 k_b   = b.conductor_k;

    Float3 sep_a   = sqr(eta_a - 1.0f) + sqr(k_a);
    Float3 den_a   = sqr(eta_a + 1.0f) + sqr(k_a);
    Float3 sep_b   = sqr(eta_b - 1.0f) + sqr(k_b);
    Float3 den_b   = sqr(eta_b + 1.0f) + sqr(k_b);
    Float3 f0_complex_a = sep_a / den_a;
    Float3 f0_complex_b = sep_b / den_b;

    Float3 f0_schlick_a = lerp(make_float3(0.04f), a.albedo.xyz(), a.metallic);
    Float3 f0_schlick_b = lerp(make_float3(0.04f), b.albedo.xyz(), b.metallic);

    Bool use_complex_a = any(k_a != 0.0f);
    Bool use_complex_b = any(k_b != 0.0f);

    Float3 f0_a = ite(use_complex_a, f0_complex_a, f0_schlick_a);
    Float3 f0_b = ite(use_complex_b, f0_complex_b, f0_schlick_b);

    Bool f0_ok = abs(luminance(f0_a) - luminance(f0_b)) <= f0Thresh;

    // Albedo luminance
    Bool albedo_ok = abs(luminance(a.albedo.xyz()) - luminance(b.albedo.xyz())) <= albedoThresh;

    return rough_ok & f0_ok & albedo_ok;
}

} // namespace newtype::render
