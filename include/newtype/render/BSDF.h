#pragma once

#include "newtype/core/Config.h"
#include "newtype/render/Sampling.h"
#include "newtype/render/Lobe.h"
#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

namespace newtype::render {

using namespace luisa;
using namespace luisa::compute;

//==============================================================================
// Fresnel Terms
//==============================================================================

/**
 * @brief Dielectric Fresnel (Schlick approximation)
 *
 * Fast approximation for glass/water materials.
 * R0 = ((eta1 - eta2) / (eta1 + eta2))^2
 * F = R0 + (1 - R0) * (1 - cos(theta))^5
 */
[[nodiscard]] Float fresnel_schlick(
    Expr<float> R0,
    Expr<float> cos_theta) noexcept;

/**
 * @brief Exact dielectric Fresnel
 *
 * For accurate glass refraction/reflection.
 */
[[nodiscard]] Float fresnel_dielectric(
    Expr<float> cos_theta_i,
    Expr<float> eta_i,
    Expr<float> eta_t) noexcept;

/**
 * @brief Thin-film interference Fresnel (3-wavelength Airy approximation)
 *
 * Computes wavelength-dependent reflectance from a thin dielectric film.
 * Uses wavelengths R=630nm, G=530nm, B=460nm.
 * Legacy path — kept for A/B against the Belcour-Barla model
 * (NT_ENABLE_BELCOUR_IRIDESCENCE, default on).
 */
[[nodiscard]] Float3 eval_thin_film_iridescence(
    Expr<float> cos_theta_d,
    Expr<float> thin_film_ior,
    Expr<float> thickness) noexcept;

#if NT_ENABLE_BELCOUR_IRIDESCENCE

/**
 * @brief Thin-film interference Fresnel — Belcour & Barla 2017
 *        (port of the KHR_materials_iridescence reference implementation)
 *
 * Substrate-aware two-interface model with analytic spectral integration:
 * the film->substrate Fresnel is derived from `base_f0`, and the CIE XYZ
 * sensitivity is folded in as Gaussian fits so the result is RGB. `cos_theta1`
 * is the cosine of the direction against the interface normal (microfacet
 * half-vector at the BSDF call site); `thickness_nm` is the film thickness
 * in nanometers. Returns white on total internal reflection.
 */
[[nodiscard]] Float3 eval_iridescence_fresnel(
    Expr<float> cos_theta1,
    Expr<float> outside_ior,
    Expr<float> film_ior,
    Expr<float> thickness_nm,
    Expr<luisa::float3> base_f0) noexcept;

#endif

/**
 * @brief Conductor Fresnel for a complex refractive index eta + i*k (per RGB channel)
 *
 * Closed-form Cook-Torrance conductor Fresnel, ported from PBRT v4 FrComplex
 * (scattering.h:81-92). `eta` and `k` are Float3 — each channel evaluated
 * independently. Use `MaterialData::attenuation` for eta (overloaded storage)
 * and `MaterialData::conductor_k` for k.
 */
[[nodiscard]] Float3 fresnel_conductor(
    Expr<float> cos_theta_i,
    Expr<luisa::float3> eta,
    Expr<luisa::float3> k) noexcept;

//==============================================================================
// Dispersion (Abbe number) — KHR_materials_dispersion
//==============================================================================

/// Channel wavelengths in micrometers for the RGB delta-wavelength convention
/// (R=630nm, G=530nm, B=460nm — the same triple MetalData.h presets and the
/// iridescence models use).
inline constexpr float kDispersionChannelUm[3] = { 0.630f, 0.530f, 0.460f };
/// KHR_materials_dispersion Cauchy constants (spec closed form
/// n(λ) = n_d + (n_d−1)/V · (523655/λ²[nm] − 1.5168), λ_d = 587.56nm).
/// kDispersionCauchyUm2 = 1/(λF⁻² − λC⁻²) over the Fraunhofer F (486.13nm) and
/// C (656.27nm) lines, so the modelled n_F − n_C equals (n_d−1)/V exactly —
/// the Abbe definition. A bare (1/λ² − 1/λd²) form would over-disperse by
/// 1/(λF⁻²−λC⁻²) ≈ 1.91× at the same V.
inline constexpr float kDispersionCauchyUm2  = 523.655e-3f; // µm² (= 523655 nm²)
inline constexpr float kDispersionCauchyAnchor = 1.5168f;  // 523655/λ_d², zero at the d-line

/**
 * @brief Per-channel IOR from the Abbe number (Cauchy first-order fit)
 *
 * KHR_materials_dispersion spec closed form: n(λ) = n_d + (n_d−1)/V ·
 * (523655/λ² − 1.5168), anchored so n(λ_d) = n_d (`ior` is the d-line
 * index) and n_F − n_C = (n_d−1)/V exactly (the Abbe definition — same V
 * gives the same spread as the KHR reference's ior ∓ (ior−1)/(2V) linear
 * form). Blue (460nm) gets the higher index, red (630nm) the lower — normal
 * dispersion. `channel`: 0=R, 1=G, 2=B. `abbe` <= 0 → n_d (feature off).
 * The Abbe value is floored at 5 inside as an art-direction clamp against
 * junk inputs.
 */
[[nodiscard]] inline Float dispersed_ior(
    Expr<float> n_d, Expr<float> abbe, Expr<uint> channel) noexcept {
    Float lam2 = ite(channel == 0u, sqr(kDispersionChannelUm[0]),
                ite(channel == 1u, sqr(kDispersionChannelUm[1]),
                                  sqr(kDispersionChannelUm[2])));
    Float b = (n_d - 1.0f) / max(abbe, 5.0f);
    Float delta = b * (kDispersionCauchyUm2 / lam2 - kDispersionCauchyAnchor);
    return ite(abbe > 0.0f, max(n_d + delta, 1.0001f), n_d);
}

/// Stochastic channel pick from a uniform variate: 0=R, 1=G, 2=B (1/3 each).
[[nodiscard]] inline UInt pick_dispersion_channel(Expr<float> u) noexcept {
    return min(cast<uint>(u * 3.0f), 2u);
}

/**
 * @brief Throughput weight for one stochastic dispersion pick: 3·e_c
 *
 * Same strobe pattern as the SSS probe (PassSSS): picking channel c with
 * probability 1/3 and multiplying the path throughput by 3·e_c makes the
 * multi-frame average converge to the per-channel integrated RGB estimate.
 * One weight per path (the wavelength is constant along a transport chain),
 * applied at the first dispersive refraction only.
 */
[[nodiscard]] inline Float3 dispersion_channel_weight(Expr<uint> channel) noexcept {
    return 3.0f * ite(channel == 0u, make_float3(1.f, 0.f, 0.f),
                 ite(channel == 1u, make_float3(0.f, 1.f, 0.f),
                                    make_float3(0.f, 0.f, 1.f)));
}

//==============================================================================
// Microfacet Distribution (GGX/Trowbridge-Reitz)
//==============================================================================

/**
 * @brief GGX (Trowbridge-Reitz) microfacet distribution
 *
 * D(wh) = alpha^2 / (pi * (cos(theta_h)^4 * (alpha^2 + tan(theta_h)^2)^2))
 *
 * Commonly used for PBR roughness modeling.
 */
[[nodiscard]] Float ggx_distribution(
    Expr<luisa::float3> wh,
    Expr<luisa::float2> alpha) noexcept;

/**
 * @brief GGX lambda function for shadowing-masking
 *
 * Lambda(w) = (-1 + sqrt(1 + 1/(a * tan(theta))^2)) / 2
 */
[[nodiscard]] Float ggx_lambda(
    Expr<luisa::float3> w,
    Expr<luisa::float2> alpha) noexcept;

/**
 * @brief GGX geometric term G1
 *
 * G1(w) = 1 / (1 + Lambda(w))
 */
[[nodiscard]] Float ggx_G1(
    Expr<luisa::float3> w,
    Expr<luisa::float2> alpha) noexcept;

/**
 * @brief GGX geometric term G (Smith correlated)
 *
 * G(wo, wi) = 1 / (1 + Lambda(wo) + Lambda(wi))
 */
[[nodiscard]] Float ggx_G(
    Expr<luisa::float3> wo,
    Expr<luisa::float3> wi,
    Expr<luisa::float2> alpha) noexcept;

/**
 * @brief Sample GGX microfacet normal
 *
 * Importance sampling of GGX distribution for BSDF sampling.
 */
[[nodiscard]] Float3 sample_ggx_wh(
    Expr<luisa::float3> wo,
    Expr<luisa::float2> alpha,
    Expr<luisa::float2> u) noexcept;

/**
 * @brief PDF of sampled GGX microfacet normal
 */
[[nodiscard]] Float ggx_pdf(
    Expr<luisa::float3> wo,
    Expr<luisa::float3> wh,
    Expr<luisa::float2> alpha) noexcept;

/**
 * @brief Minimum perceptual roughness, matching RTXDI's kMinRoughness.
 *
 * RTXDI clamps roughness to this value before squaring into alpha in every
 * BRDF call (FullSample RAB_Material.hlsli:19, ShadingHelpers.hlsli:33/48,
 * RAB_Surface.hlsli:235, RAB_LightSampling.hlsli:53, BrdfRayTracing.hlsl:127).
 * Without this floor, very-low-roughness surfaces collapse the GGX lobe to a
 * near-delta (alpha^2 ~ 1e-8 vs RTXDI's 8.1e-7), producing fireflies on rare
 * in-lobe samples and a dim average elsewhere because most light-sampled
 * candidates miss the lobe.
 */
inline constexpr float kMinRoughness = 0.03f;

/**
 * @brief Convert roughness to GGX alpha
 *
 * alpha = max(roughness, kMinRoughness)^2 — clamps at the roughness level
 * (not the alpha level) to match RTXDI's kMinRoughness convention.
 */
[[nodiscard]] inline Float roughness_to_alpha(Float roughness) noexcept {
    return sqr(max(roughness, kMinRoughness));
}

[[nodiscard]] inline Float2 roughness_to_alpha(Float2 roughness) noexcept {
    return sqr(max(roughness, make_float2(kMinRoughness)));
}

//==============================================================================
// Kulla-Conty Multiple-Scattering Energy Compensation (gated: NT_ENABLE_MS_GGX)
//==============================================================================

/**
 * @brief Single-scattering directional albedo E(mu, alpha) for GGX + Smith, F == 1
 *
 * Chebyshev tensor fit (6x6) of G = 1 - E over mu in [0,1], r = sqrt(alpha) in
 * [0.03, 1]. Fit constants + error table in docs/ms_ggx_compensation.md
 * (max abs err 0.022, mean 0.004; generated by tools/ms_ggx_fit.py).
 */
[[nodiscard]] Float ggx_directional_albedo_fit(
    Expr<float> mu, Expr<float> alpha) noexcept;

/**
 * @brief Average albedo E_avg(alpha) = 2*int_0^1 E(mu,alpha) mu dmu
 *
 * Deg-5 polynomial in alpha (max abs err 8e-4, tools/ms_ggx_fit.py [3]).
 */
[[nodiscard]] Float ggx_avg_albedo_fit(Expr<float> alpha) noexcept;

/**
 * @brief Cosine-weighted average Fresnel of a Schlick lobe with the given F0
 *
 * Closed form: F_avg = F0 + (1 - F0)/21. For complex-IOR conductors pass
 * F0 = fresnel_conductor(1, eta, k) — verified within 2% of the exact
 * conductor-Fresnel average for all MetalData presets (docs/ms_ggx_compensation.md).
 */
[[nodiscard]] Float3 schlick_f_avg(Expr<float3> F0) noexcept;

//==============================================================================
// BSDF Evaluations
//==============================================================================
// All BSDF structs below (and MaterialBSDF / LobeList) are NON-OWNING views:
// every field is an Expr<T> reference into an existing expression DAG
// (SurfaceData members, computed expressions, or literal constants).
// Constructing, copying, and passing them by value emits ZERO IR statements —
// no local variables, no assignments (contrast Var/Float members, which emit
// a local + copy per field per construction). This is what removes the
// per-shade "translate SurfaceData into a BSDF" cost: MaterialBSDF built from
// a resolved SurfaceData is free, so per-candidate / per-neighbor
// constructions in the ReSTIR loops no longer materialize anything.
//
// View discipline (enforced by Expr itself — no default ctor, no assignment):
//  - build the full view in ONE aggregate initialization (factories below);
//  - never assign members after construction — derive a new view instead;
//  - view only values whose scope dominates every use (SurfaceData outlives
//    its BSDFs at all call sites).

/**
 * @brief Lambertian diffuse BRDF
 *
 * f(wo, wi) = albedo / pi
 * PDF(wi) = cos(theta) / pi
 */
struct LambertianBSDF {
    Expr<luisa::float3> albedo;

    /** @brief Evaluate BRDF */
    [[nodiscard]] Float3 evaluate(
        Expr<luisa::float3> wo,
        Expr<luisa::float3> wi,
        Expr<luisa::float3> normal) const noexcept;

    /** @brief Sample BRDF (cosine-weighted) */
    [[nodiscard]] Float3 sample(
        Expr<luisa::float3> wo,
        Expr<luisa::float3> normal,
        Expr<luisa::float2> u,
        Float& out_pdf) const noexcept;

    /** @brief PDF of sampling */
    [[nodiscard]] Float pdf(
        Expr<luisa::float3> wo,
        Expr<luisa::float3> wi,
        Expr<luisa::float3> normal) const noexcept;
};

/**
 * @brief GGX microfacet reflection (for roughness/metallic)
 *
 * Uses Cook-Torrance microfacet model:
 * f = D * F * G / (4 * cos(wo) * cos(wi))
 */
struct MicrofacetBSDF {
    Expr<luisa::float3> albedo;
    Expr<luisa::float2> alpha;         // Roughness^2 (alpha.x for tangent, alpha.y for bitangent)
    Expr<float> metallic;       // 0 = dielectric, 1 = conductor
    Expr<float> ior;            // Index of refraction (for dielectric)
    Expr<float> iridescence;
    Expr<float> iridescence_ior;
    Expr<float> iridescence_thickness;
    Expr<luisa::float3> tangent_dir;  // world-space tangent
    Expr<float> bitangent_sign;                        // handedness
    // Complex-IOR conductor Fresnel — active when metallic > 0.5 AND conductor_k != 0.
    // attenuation slot is overloaded as conductor_eta_re.
    Expr<luisa::float3> conductor_eta;
    Expr<luisa::float3> conductor_k;

    /** @brief Evaluate BRDF */
    [[nodiscard]] Float3 evaluate(
        Expr<luisa::float3> wo,
        Expr<luisa::float3> wi,
        Expr<luisa::float3> normal) const noexcept;

    /** @brief Sample BRDF (GGX importance sampling) */
    [[nodiscard]] Float3 sample(
        Expr<luisa::float3> wo,
        Expr<luisa::float3> normal,
        Expr<luisa::float2> u,
        Float& out_pdf) const noexcept;

    /** @brief PDF of sampling */
    [[nodiscard]] Float pdf(
        Expr<luisa::float3> wo,
        Expr<luisa::float3> wi,
        Expr<luisa::float3> normal) const noexcept;
};

//==============================================================================
// Microfacet Transmission BSDF (Walter 2007)
//==============================================================================

/**
 * @brief GGX microfacet transmission (refraction)
 *
 * Walter et al. 2007 microfacet transmission model for dielectric surfaces.
 * Handles rough refraction through a microfacet surface using the GGX distribution.
 * For smooth surfaces (roughness=0), collapses to delta (perfect specular refraction).
 */
struct MicrofacetTransmissionBSDF {
    Expr<luisa::float3> albedo;       // absorption tint (attenuation color)
    Expr<luisa::float2> alpha;        // roughness^2
    Expr<float> eta;          // IOR ratio: eta_i / eta_t (e.g., 1.0/ior for entering)
    Expr<float> ior;          // absolute IOR of dielectric material

    /** @brief Evaluate BTDF */
    [[nodiscard]] Float3 evaluate(
        Expr<luisa::float3> wo,
        Expr<luisa::float3> wi,
        Expr<luisa::float3> normal) const noexcept;

    /** @brief Sample BTDF (Fresnel-weighted reflection or transmission) */
    [[nodiscard]] Float3 sample(
        Expr<luisa::float3> wo,
        Expr<luisa::float3> normal,
        Expr<luisa::float2> u,
        Float& out_pdf,
        Bool& out_is_transmission) const noexcept;

    /**
     * @brief Dispersion-aware sampling (KHR_materials_dispersion)
     *
     * Stochastically picks R/G/B from `u_disp` (dispersion > 0 only) and
     * refracts with that channel's IOR instead of the d-line `ior` member.
     * `out_dispersion_tint` returns 3·e_c on a dispersive transmission pick
     * (multiply the path throughput by it; for the exact estimator take the
     * picked channel's component of f, i.e. f·tint/3), and (1,1,1) otherwise
     * (reflection branch, TIR, or dispersion off — bit-identical to the
     * monochromatic sample() above). The Fresnel branch split and the
     * reflection lobe stay at the d-line IOR (dispersion in reflection is
     * negligible); only the refraction direction, its eta-Jacobian pdf, and
     * the transmission Fresnel factor use the channel IOR.
     */
    [[nodiscard]] Float3 sample(
        Expr<luisa::float3> wo,
        Expr<luisa::float3> normal,
        Expr<luisa::float2> u,
        Expr<float> u_disp,
        Expr<float> dispersion,
        Float& out_pdf,
        Bool& out_is_transmission,
        Float3& out_dispersion_tint) const noexcept;

    /** @brief PDF of sampling */
    [[nodiscard]] Float pdf(
        Expr<luisa::float3> wo,
        Expr<luisa::float3> wi,
        Expr<luisa::float3> normal) const noexcept;
};

//==============================================================================
// Dielectric BSDF (Combined Reflection + Transmission)
//==============================================================================

/**
 * @brief Combined dielectric BSDF for glass materials
 *
 * Wraps GGX microfacet reflection + microfacet transmission with Fresnel gating.
 * Fresnel determines the probability of reflection vs transmission.
 * Supports both smooth (delta) and rough (GGX) dielectrics.
 */
struct DielectricBSDF {
    Expr<luisa::float3> albedo;       // absorption tint
    Expr<float> roughness;
    Expr<float> ior;

    /** @brief Evaluate dielectric BSDF */
    [[nodiscard]] Float3 evaluate(
        Expr<luisa::float3> wo,
        Expr<luisa::float3> wi,
        Expr<luisa::float3> normal) const noexcept;

    /** @brief Sample dielectric BSDF */
    [[nodiscard]] Float3 sample(
        Expr<luisa::float3> wo,
        Expr<luisa::float3> normal,
        Expr<luisa::float2> u,
        Float& out_pdf,
        Bool& out_is_transmission) const noexcept;

    /** @brief Sample dielectric BSDF with stochastic dispersion (see
     *         MicrofacetTransmissionBSDF::sample — forwarded). */
    [[nodiscard]] Float3 sample(
        Expr<luisa::float3> wo,
        Expr<luisa::float3> normal,
        Expr<luisa::float2> u,
        Expr<float> u_disp,
        Expr<float> dispersion,
        Float& out_pdf,
        Bool& out_is_transmission,
        Float3& out_dispersion_tint) const noexcept;

    /** @brief PDF of sampling */
    [[nodiscard]] Float pdf(
        Expr<luisa::float3> wo,
        Expr<luisa::float3> wi,
        Expr<luisa::float3> normal) const noexcept;
};

//==============================================================================
// Thin Dielectric BSDF
//==============================================================================

/**
 * @brief Thin-walled dielectric BSDF (soap bubble, leaf, paper)
 *
 * Models a surface where entry and exit happen at the same point — no spatial
 * separation. Fresnel determines reflection vs transmission probability.
 * Transmission is diffuse (cosine-weighted) on the exit side, not refracted.
 *
 * For roughness > 0: GGX microfacet reflection instead of delta specular.
 * Transmission always uses diffuse lobe regardless of roughness.
 */
struct ThinDielectricBSDF {
    Expr<luisa::float3> albedo;     // transmission tint (attenuation color)
    Expr<float> roughness;
    Expr<float> ior;

    /** @brief Evaluate thin dielectric BSDF */
    [[nodiscard]] Float3 evaluate(
        Expr<luisa::float3> wo,
        Expr<luisa::float3> wi,
        Expr<luisa::float3> normal) const noexcept;

    /** @brief Sample thin dielectric BSDF */
    [[nodiscard]] Float3 sample(
        Expr<luisa::float3> wo,
        Expr<luisa::float3> normal,
        Expr<luisa::float2> u,
        Float& out_pdf) const noexcept;

    /** @brief PDF of sampling */
    [[nodiscard]] Float pdf(
        Expr<luisa::float3> wo,
        Expr<luisa::float3> wi,
        Expr<luisa::float3> normal) const noexcept;
};

//==============================================================================
// Sheen BSDF (Disney Principled)
//==============================================================================

/**
 * @brief Sheen lobe — Charlie microfacet distribution (KHR_materials_sheen)
 *
 * Velvet / peach-fuzz rim at grazing angles. Replaces the old overlay
 * sheen * (1 - cos_theta_d^5), which read ~0.97 already at theta_d = 60 deg
 * and whitened the entire material (unbounded additive energy on top of the
 * base).
 *
 * f_sheen = strength * D_charlie(theta_h) * G_charlie * sheen_color
 *           / (4 * NoV * NoL)
 *   alpha      = max(roughness^2, 1e-3),   inv_alpha = 1 / alpha
 *   D_charlie  = (2 + inv_alpha) / (2*pi) * sin(theta_h)^inv_alpha
 *   G_charlie  = 1 / (1 + lambda(NoV) + lambda(NoL))   (Smith, rational fit)
 *   sheen_color = mix(white, albedo, sheen_tint)
 * The Disney 0.25 * sheen scale lives in the lobe weight (build_lobe_list /
 * resolve_surface_layered), matching the clearcoat slot convention; the same
 * weight is subtracted from the base diffuse-family budget.
 *
 * References: Conty Estevez & Kulla, "Production Friendly Microfacet Sheen
 * BRDF" (SIGGRAPH 2017); KHR_materials_sheen reference implementation.
 *
 * Sampling: cosine-weighted hemisphere (the lobe is broad), pdf = cos/pi.
 * In the BRDF sampling pool under NT_ENABLE_SHEEN_SAMPLING.
 */
struct SheenBSDF {
    Expr<float> sheen_strength;
    Expr<float> sheen_tint;    // scalar: 0=white, 1=toward albedo
    Expr<luisa::float3> albedo;       // base color for tinting
    Expr<float> roughness;     // Charlie exponent: inv_alpha = 1 / max(roughness^2, 1e-3)

    /** @brief Evaluate sheen BRDF (additive term) */
    [[nodiscard]] Float3 evaluate(
        Expr<luisa::float3> wo,
        Expr<luisa::float3> wi,
        Expr<luisa::float3> normal) const noexcept;

    /** @brief Sample sheen lobe (cosine-weighted hemisphere) */
    [[nodiscard]] Float3 sample(
        Expr<luisa::float3> wo,
        Expr<luisa::float3> normal,
        Expr<luisa::float2> u,
        Float& out_pdf) const noexcept;

    /** @brief Sheen sampling pdf (cosine-weighted, cos/pi) */
    [[nodiscard]] Float pdf(
        Expr<luisa::float3> wo,
        Expr<luisa::float3> wi,
        Expr<luisa::float3> normal) const noexcept;
};

//==============================================================================
// Clearcoat BSDF (Disney Principled)
//==============================================================================

/**
 * @brief Clearcoat additive lobe (Disney Principled BSDF)
 *
 * Glossy dielectric coating (varnish, car paint, lacquer).
 * Uses GGX microfacet with Schlick Fresnel from coat_ior (Disney's default
 * IOR=1.5 → R0=0.04; the float-exact identity sqr((1.5-1)/(1.5+1)) == 0.04f
 * keeps old scenes bit-identical).
 * Additive in evaluate(); also directly sampleable (NT_ENABLE_COAT_SAMPLING
 * puts the coat lobe in the composed BRDF sampling pool).
 *
 * roughness_cc = mix(0.1, 0.001, clearcoat_gloss)
 * f_clearcoat = clearcoat * F_schlick * D_ggx * G_ggx / (4 * cos_o * cos_i)
 */
struct ClearcoatBSDF {
    Expr<float> clearcoat_strength;
    Expr<float> clearcoat_gloss;   // 0=rough clearcoat, 1=smooth clearcoat
    Expr<float> coat_ior;          // coat interface IOR (F0 = ((n-1)/(n+1))^2)

    /** @brief Evaluate clearcoat BRDF (additive term) */
    [[nodiscard]] Float3 evaluate(
        Expr<luisa::float3> wo,
        Expr<luisa::float3> wi,
        Expr<luisa::float3> normal) const noexcept;

    /** @brief Sample coat reflection (GGX NDF sampling with alpha_cc) */
    [[nodiscard]] Float3 sample(
        Expr<luisa::float3> wo,
        Expr<luisa::float3> normal,
        Expr<luisa::float2> u,
        Float& out_pdf) const noexcept;

    /** @brief Coat reflection pdf (GGX reflection pdf with alpha_cc) */
    [[nodiscard]] Float pdf(
        Expr<luisa::float3> wo,
        Expr<luisa::float3> wi,
        Expr<luisa::float3> normal) const noexcept;
};

//==============================================================================
// Subsurface BSDF (Hanrahan-Krueger + Thin Transmission)
//==============================================================================

/**
 * @brief Hanrahan-Krueger SSS with diffuse transmission
 *
 * Same hemisphere: HK approximation, energy-coupled to the transmission mix
 *   f = (1 - p_trans) * albedo / (pi * (cos_i + cos_o))
 *   with p_trans = diffuse_trans * (1 - F(cos_o)) — the same branch
 *   probability sample()/pdf() use, so reflect + transmit share one budget.
 * Opposite hemisphere: diffuse transmission tinted by Beer's law absorption
 *   f_trans = (1-F) * transmission_color * diffuse_trans / (pi * (cos_i + cos_o))
 *
 * diffuse_trans controls energy split: 0=pure HK reflection, 1=full transmission.
 * transmission_color = albedo * exp(-attenuation / attenuation_distance).
 */
struct SubsurfaceBSDF {
    Expr<luisa::float3> albedo;
    Expr<luisa::float3> transmission_color;  // precomputed Beer's law tint
    Expr<float> diffuse_trans;       // 0=HK only, 1=full transmission
    Expr<float> ior;                 // for Fresnel at surface

    /** @brief Evaluate SSS BRDF (both hemispheres) */
    [[nodiscard]] Float3 evaluate(
        Expr<luisa::float3> wo,
        Expr<luisa::float3> wi,
        Expr<luisa::float3> normal) const noexcept;

    /** @brief Sample SSS BRDF (bidirectional) */
    [[nodiscard]] Float3 sample(
        Expr<luisa::float3> wo,
        Expr<luisa::float3> normal,
        Expr<luisa::float2> u,
        Float& out_pdf) const noexcept;

    /** @brief PDF of sampling */
    [[nodiscard]] Float pdf(
        Expr<luisa::float3> wo,
        Expr<luisa::float3> wi,
        Expr<luisa::float3> normal) const noexcept;
};

//==============================================================================
// Fabric Diffuse BSDF (Ashikhmin-Premoze / Disney Fabric)
//==============================================================================

/**
 * @brief Fabric diffuse model with grazing-angle enhancement
 *
 * f(wo, wi) = (21 / (20*pi)) * albedo * (1 - (1-cos_o)^5) * (1 - (1-cos_i)^5)
 *
 * Stronger response at grazing angles compared to Lambertian,
 * producing a characteristic fabric/cloth appearance.
 * Uses cosine-weighted hemisphere sampling (same as Lambertian).
 */
struct FabricDiffuseBSDF {
    Expr<luisa::float3> albedo;

    /** @brief Evaluate fabric diffuse BRDF */
    [[nodiscard]] Float3 evaluate(
        Expr<luisa::float3> wo,
        Expr<luisa::float3> wi,
        Expr<luisa::float3> normal) const noexcept;

    /** @brief Sample fabric diffuse BRDF (cosine-weighted hemisphere) */
    [[nodiscard]] Float3 sample(
        Expr<luisa::float3> wo,
        Expr<luisa::float3> normal,
        Expr<luisa::float2> u,
        Float& out_pdf) const noexcept;

    /** @brief PDF of sampling */
    [[nodiscard]] Float pdf(
        Expr<luisa::float3> wo,
        Expr<luisa::float3> wi,
        Expr<luisa::float3> normal) const noexcept;
};

//==============================================================================
// Material-Based BSDF Selection
//==============================================================================

/**
 * @brief Combined BSDF for MaterialPool integration — NON-OWNING view
 *
 * Every field is an Expr view (see the view-discipline comment at the top of
 * the BSDF Evaluations section). A MaterialBSDF is built exclusively through
 * the factories:
 *   - SurfaceData::make_bsdf() / make_bsdf(probe) / make_bsdf_roughened()
 *     — the per-shade path; views a resolved SurfaceData.
 *   - make_material_bsdf(...) — the scalar-parameter path for call sites
 *     without a SurfaceData (legacy scalar p_hat/direct-illuminance helpers,
 *     gi_bounce x2 NEE). Pins the layered/MS fields to their historical
 *     defaults, bit-identical to the previous inline aggregate constructions.
 *
 * Selects between diffuse and specular based on material properties.
 * - Low metallic: Lambertian diffuse + GGX specular (layered)
 * - High metallic: Pure GGX conductor
 * - Sheen: Additive lobe for fabric-like appearance (sheen > 0)
 */
struct MaterialBSDF {
    Expr<luisa::float3> albedo;
    Expr<float> roughness;
    Expr<float> metallic;
    Expr<float> ior;
    Expr<float> sheen_val;
    Expr<float> sheen_tint_val;
    Expr<float> clearcoat_val;
    Expr<float> clearcoat_gloss_val;
    Expr<float> iridescence_val;
    Expr<float> iridescence_ior_val;
    Expr<float> iridescence_thickness_val;
    Expr<float> anisotropic_val;
    Expr<float> anisotropic_rot_val;
    Expr<luisa::float3> tangent_dir;
    Expr<float> bitangent_sign;
    Expr<uint> bsdf_type;             // Material BSDF type tag for dispatch (e.g., 11 = thin dielectric)
    Expr<float> flatness_val;         // Blend: 0=Lambertian, 1=Hanrahan-Krueger SSS
    Expr<float> fabric_val;           // Blend: 0=Lambertian, 1=full Ashikhmin-Premoze fabric diffuse
    Expr<float> specular_tint_val;    // 0=white specular, 1=albedo chromaticity tint
    Expr<float> specular_trans_val;   // Dielectric transmission energy scale
    Expr<luisa::float3> attenuation_val;    // SSS/glass absorption tint (also: conductor eta_re)
    Expr<float> diffuse_trans_val;    // SSS transmission fraction (0=HK only, 1=full transmission)
    Expr<float> attenuation_distance_val;   // SSS absorption distance for Beer's law
    // Complex-IOR conductor Fresnel — set by make_bsdf() from SurfaceData.
    Expr<luisa::float3> conductor_k_val;

    // Anisotropy-rotated tangent (item 10 hoist): the cos/sin/normalize
    // rotation depends only on the tangent frame + anisotropic_rot + the
    // surface normal — not on (wo, wi) — so it is computed once per
    // construction (a pure expression node in the factories) instead of being
    // rebuilt at every evaluate/evaluate_split/sample/pdf call site (was 10
    // mirrored copies). Default = no rotation (tangent passed through).
    Expr<luisa::float3> t_rot_val;

    // Lobe-list metadata. DAG view built by build_standard_lobe_list() and,
    // for layered surfaces, merged with SurfaceData's composed list via
    // per-slot ite() in SurfaceData::make_bsdf() — zero IR statements.
    // Consumed by lobe-list dispatch in BSDF.cpp.
    LobeList lobe_list;

    // vertical layering — composed-list flag + coat/fuzz layer params.
    // Active only when has_composed_lobe_list is true. The base layer's params
    // reuse the existing fields above (albedo, roughness, metallic, etc).
    // The coat layer (slot 0-1) and fuzz layer (slot 9) read from these
    // dedicated fields so the type-dispatch path can find them.
    Expr<bool> has_composed_lobe_list;
    Expr<uint> coat_bsdf_type;        // 7=Clearcoat, 3=Dielectric
    Expr<float> coat_clearcoat_val;
    Expr<float> coat_clearcoat_gloss_val;
    Expr<float> coat_ior;
    Expr<float> coat_roughness;
    Expr<luisa::float3> coat_attenuation;
    // Coat shading normal (SurfaceData::coat_ns) — the slot-0 coat dispatch
    // evaluates against this, not the base `normal` param. Equals ns for
    // coats without their own normal map.
    Expr<luisa::float3> coat_normal;
    Expr<luisa::float3> fuzz_albedo;
    Expr<float> fuzz_sheen_val;
    Expr<float> fuzz_sheen_tint_val;
    Expr<float> fuzz_sheen_roughness_val;
    // Forwarded F12 (air→coat) and F23 (coat→base) Fresnel terms, evaluated
    // once in resolve_surface_layered. Consumed in evaluate/evaluate_split to
    // skip the per-evaluate fresnel_dielectric recompute.
    Expr<float> coat_F12;
    Expr<float> coat_F23;

    // Kulla-Conty MS-GGX invariants, forwarded from SurfaceData::make_bsdf()
    // (computed once in resolve_surface — material + wo only). Defaults =
    // no compensation. NT_ENABLE_MS_GGX spec blocks consume these + one E_i
    // fit per evaluate call. See docs/ms_ggx_compensation.md.
    Expr<float> ms_e_o;
    Expr<float> ms_e_avg;
    Expr<luisa::float3> ms_f_avg;

    /** @brief Evaluate material BSDF */
    [[nodiscard]] Float3 evaluate(
        Expr<luisa::float3> wo,
        Expr<luisa::float3> wi,
        Expr<luisa::float3> normal) const noexcept;

    /** @brief Evaluate material BSDF with diffuse/specular separated */
    void evaluate_split(
        Expr<luisa::float3> wo,
        Expr<luisa::float3> wi,
        Expr<luisa::float3> normal,
        Float3& out_diffuse,
        Float3& out_specular) const noexcept;

    /** @brief Sample material BSDF */
    [[nodiscard]] Float3 sample(
        Expr<luisa::float3> wo,
        Expr<luisa::float3> normal,
        Expr<luisa::float2> u,
        Float& out_pdf) const noexcept;

    /** @brief PDF of sampling */
    [[nodiscard]] Float pdf(
        Expr<luisa::float3> wo,
        Expr<luisa::float3> wi,
        Expr<luisa::float3> normal) const noexcept;
};

/// Same view with the roughness floored (RTXDI kMinRoughness parity — see
/// SurfaceData::make_bsdf_roughened). For sites that only hold a MaterialBSDF
/// view (GI reuse kernels), where no SurfaceData is available. Lobe weights
/// and MS-GGX invariants have no roughness dependence, so a field-by-field
/// copy with the overridden roughness is exact.
[[nodiscard]] inline MaterialBSDF make_bsdf_roughened(
    const MaterialBSDF& bsdf, Expr<float> roughness_floor) noexcept {
    return MaterialBSDF{
        bsdf.albedo, max(bsdf.roughness, roughness_floor), bsdf.metallic, bsdf.ior,
        bsdf.sheen_val, bsdf.sheen_tint_val,
        bsdf.clearcoat_val, bsdf.clearcoat_gloss_val,
        bsdf.iridescence_val, bsdf.iridescence_ior_val, bsdf.iridescence_thickness_val,
        bsdf.anisotropic_val, bsdf.anisotropic_rot_val,
        bsdf.tangent_dir, bsdf.bitangent_sign,
        bsdf.bsdf_type,
        bsdf.flatness_val, bsdf.fabric_val,
        bsdf.specular_tint_val, bsdf.specular_trans_val, bsdf.attenuation_val,
        bsdf.diffuse_trans_val, bsdf.attenuation_distance_val,
        bsdf.conductor_k_val,
        bsdf.t_rot_val,
        bsdf.lobe_list,
        bsdf.has_composed_lobe_list,
        bsdf.coat_bsdf_type, bsdf.coat_clearcoat_val, bsdf.coat_clearcoat_gloss_val,
        bsdf.coat_ior, bsdf.coat_roughness, bsdf.coat_attenuation, bsdf.coat_normal,
        bsdf.fuzz_albedo, bsdf.fuzz_sheen_val, bsdf.fuzz_sheen_tint_val, bsdf.fuzz_sheen_roughness_val,
        bsdf.coat_F12, bsdf.coat_F23,
        bsdf.ms_e_o, bsdf.ms_e_avg, bsdf.ms_f_avg};
}

//==============================================================================
// View factories — Phase 2C replacement for build_lobe_list() /
// precompute_tangent_rotation() mutation-after-construction
//==============================================================================

/// Build the standard-layered (count==7) / delta-only (count==1) LobeList as
/// a pure expression DAG — no local variables, no stores. Slot layout and
/// arithmetic are identical to the former build_lobe_list_for_layer writes
/// (branch writes became per-slot ite selects of the same values):
///   slot 0: SpecularMetal w=metallic  (or DeltaDielectric w=1 when
///           bsdf_type is 3/11, count=1)
///   slot 1: Diffuse     w = diffuse_pool_budget * (1-fabric)
///   slot 2: Fabric      w = diffuse_pool_budget * fabric
///   slot 3: Subsurface  w = (1-m)(1-st)*flat*(1-sheen_w)
///   slot 4: Transmission w = (1-m)*st
///   slot 5: Sheen       w = 0.25*(1-m)*sheen
///   slot 6: Clearcoat (add.) w = 0.25*clearcoat
/// Slots 1-6 evaluate to 0 on the delta-only path (same as the old reset +
/// guarded writes). Slots 7-9 stay at the null reset defaults.
[[nodiscard]] LobeList build_standard_lobe_list(
    Expr<float> metallic,
    Expr<float> specular_trans,
    Expr<float> flatness,
    Expr<float> sheen,
    Expr<float> fabric,
    Expr<float> clearcoat,
    Expr<uint> bsdf_type) noexcept;

/// Scalar-parameter MaterialBSDF view (non-composed single layer).
/// Replaces the legacy "24-field aggregate init + build_lobe_list() +
/// precompute_tangent_rotation(ns)" sequence at sites without a SurfaceData.
/// bsdf_type is pinned to 0 and the layered fields to their historical
/// defaults, so the lobe weights and t_rot are bit-identical to those legacy
/// constructions. attenuation is the overloaded conductor-eta slot.
/// normal must be the same normal the evaluate/sample/pdf entry points
/// receive (t_rot depends on it).
[[nodiscard]] inline MaterialBSDF make_material_bsdf(
    Expr<luisa::float3> albedo,
    Expr<float> roughness,
    Expr<float> metallic,
    Expr<float> ior,
    Expr<float> sheen,
    Expr<float> sheen_tint,
    Expr<float> clearcoat,
    Expr<float> clearcoat_gloss,
    Expr<float> iridescence,
    Expr<float> iridescence_ior,
    Expr<float> iridescence_thickness,
    Expr<float> anisotropic,
    Expr<float> anisotropic_rot,
    Expr<luisa::float3> tangent,
    Expr<float> tangent_sign,
    Expr<luisa::float3> attenuation,
    Expr<luisa::float3> conductor_k,
    Expr<luisa::float3> normal) noexcept {

    // t_rot — same expression the former precompute_tangent_rotation computed.
    Expr<luisa::float3> bitangent = cross(normal, tangent) * tangent_sign;
    Expr<luisa::float3> t_rot = normalize(
        tangent * cos(anisotropic_rot) + bitangent * sin(anisotropic_rot));

    LobeList lobes = build_standard_lobe_list(
        metallic, 0.f, 0.f, sheen, 0.f, clearcoat, 0u);

    return MaterialBSDF{
        albedo, roughness, metallic, ior,
        sheen, sheen_tint,
        clearcoat, clearcoat_gloss,
        iridescence, iridescence_ior, iridescence_thickness,
        anisotropic, anisotropic_rot,
        tangent, tangent_sign,
        0u,                 // bsdf_type — pinned, as before
        0.f,                // flatness_val
        0.f,                // fabric_val
        0.f,                // specular_tint_val
        0.f,                // specular_trans_val
        attenuation,        // attenuation_val (overloaded as conductor_eta_re)
        0.f,                // diffuse_trans_val
        1.f,                // attenuation_distance_val
        conductor_k,
        t_rot,
        lobes,
        false,              // has_composed_lobe_list
        0u,                 // coat_bsdf_type
        0.f,                // coat_clearcoat_val
        0.5f,               // coat_clearcoat_gloss_val
        1.5f,               // coat_ior
        0.f,                // coat_roughness
        luisa::float3(1.f), // coat_attenuation
        normal,             // coat_normal (no layered coat — base normal)
        luisa::float3(1.f), // fuzz_albedo
        0.f,                // fuzz_sheen_val
        0.f,                // fuzz_sheen_tint_val
        0.5f,               // fuzz_sheen_roughness_val
        0.f,                // coat_F12
        0.f,                // coat_F23
        1.f,                // ms_e_o
        1.f,                // ms_e_avg
        luisa::float3(0.f)};// ms_f_avg
}

} // namespace newtype::render
