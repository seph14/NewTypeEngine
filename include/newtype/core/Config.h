#pragma once

// ============================================================================
// NewTypeEngine Build Configuration
// ============================================================================
// Preprocessor macros (0/1) — used with #if in source files.
// Do NOT use #ifdef — always compare against 0 so typos are caught.
//
// C++ code should prefer nt::config::kXxx with if-constexpr for
// guaranteed dead-code elimination. Use the #define form only where
// preprocessor visibility is required (#include guards, DSL generation).
// ============================================================================

// --- Production ---
#define NT_PRODUCTION              0

// --- Debug ---
#define NT_DEBUG_VIZ               1
#define NT_ASSERT_LOG              1
#define NT_PROFILING               1

// --- Features ---
#define NT_ALLOW_RASTER_FEATURES   0
#define NT_ENABLE_PROCEDURAL       0
#define NT_ENABLE_DENOISER         1
#define NT_ENABLE_GI               1
#define NT_ENABLE_ENVMAP           1
#define NT_ENABLE_CHECKERBOARD     1
#define NT_ENABLE_SPECULAR_DENOSING 1
#define NT_ENABLE_TIMELINE_EDITOR  1
// Delta NEE branch enables mirror metal GI
#define NT_ENABLE_DELTA_BRANCH_NEE 1

// --- BSSRDF (Burley 2015) ---
// Master gate for the ray-traced SSS probe pass + BSSRDF math. Default OFF
// to keep main bit-identical. See plan `resilient-squishing-avalanche.md`.
#define NT_ENABLE_BSSRDF         1

// --- SSS direct model: Burley probe replaces the HK lobe in DI ---
// When on (requires NT_ENABLE_BSSRDF), direct-lighting BSDFs (PassDI
// candidates/reuse + the shade shader's DI evaluate_split) zero the HK
// subsurface lobe — the additive Burley probe is the single direct SSS
// model, ending the double count that made SSS pale/over-bright. The HK
// lobe stays active for GI throughput, and point primitives keep HK
// everywhere (the probe pass skips them). OFF = legacy additive stack.
#ifndef NT_ENABLE_SSS_PROBE_DIRECT
#define NT_ENABLE_SSS_PROBE_DIRECT 1
#endif

// --- Two-interface Fresnel for vertical layering (F12 + F23) ---
// Adds the (1-F23) coat→base attenuation term to resolve_surface_layered,
// giving the proper R = F12 + (1-F12)(1-F23)·R3 composite for car paint /
// lacquer-over-wood. OFF = bit-identical to single-interface path.
// See docs/coat_fresnel_f23.md.
#ifndef NT_ENABLE_TWO_INTERFACE_FRESNEL
#define NT_ENABLE_TWO_INTERFACE_FRESNEL 1
#endif

// --- Coat lobe BRDF sampling (vertical layering noise fix) ---
// Adds the coat lobe (composed LobeList slot 0) to the BRDF sampling pool and
// the mixture pdf, and renormalizes the pool-picker thresholds by the pool
// sum. Fixes fireflies on diffuse-base+coat: evaluate() contains the
// near-delta coat GGX lobe but the sampling pdf was diffuse-only, so
// p_hat/brdf_pdf exploded when a diffuse-sampled direction landed on the coat
// spike. Also stops the sub-1 pool sum (base_scale) from dispatching dead
// transmission-lobe fallbacks. The slot-0 pool weight is Fresnel-gated
// (weights[0] * coat_F12) in BOTH sample() and pdf() so the pick probability
// tracks the coat lobe's directional energy (normal incidence → base lobes,
// grazing → coat) instead of spending ~half the single BRDF candidate on a
// ~4% lobe. OFF = bit-identical. See docs/clearcoat_sampling.md.
#ifndef NT_ENABLE_COAT_SAMPLING
#define NT_ENABLE_COAT_SAMPLING 1
#endif

// --- Sheen lobe BRDF sampling (fuzz/grazing noise + consistency fix) ---
// Adds the sheen lobes (single-layer slot 5; composed base slot 6 + fuzz
// slot 8) to the BRDF sampling pool, the mixture pdf, and pdf(), sampled
// cosine-hemisphere (the Charlie lobe is broad). Keeps evaluate() and the
// sampling pdf consistent for the ReSTIR target_pdf / MIS blends — same
// rationale as NT_ENABLE_COAT_SAMPLING — and closes the support hole on
// transmission-dominated materials, whose pool never sampled the reflection
// directions that evaluate() still credited sheen for.
// OFF = sheen stays additive eval-only (bounded by the 0.25 lobe weight).
#ifndef NT_ENABLE_SHEEN_SAMPLING
#define NT_ENABLE_SHEEN_SAMPLING 1
#endif

// --- GGX multiple-scattering energy compensation (Kulla-Conty) ---
// Adds an f_ms lobe to MicrofacetBSDF::evaluate so rough GGX surfaces
// (especially complex-IOR conductors) don't darken as roughness rises.
// Analytic fits (Chebyshev 8x8 for E, deg-5 poly for E_avg) — no LUT.
// OFF = bit-identical. See docs/ms_ggx_compensation.md + tools/ms_ggx_fit.py.
#ifndef NT_ENABLE_MS_GGX
#define NT_ENABLE_MS_GGX 1
#endif

// --- Media ---
#define NT_ENABLE_VIDEO_RECORDER   1
#define NT_ENABLE_SCREENSHOT       1
#define NT_ENABLE_AUDIO            1
#define NT_ENABLE_MEDIA_PLAYER     0   // MF + D3D11 video decode → Luisa image. Off by default; flip to 1 to enable. See [[session-2026-07-01-mf-d3d11-video-pipeline]].

// --- Backend ---
#define NT_ALLOW_CUDA			   0

// ============================================================================
// Derived: NT_PRODUCTION overrides
// ============================================================================
#if NT_PRODUCTION
#undef NT_DEBUG_VIZ
#define NT_DEBUG_VIZ               0
#undef NT_ASSERT_LOG
#define NT_ASSERT_LOG              0
#undef NT_PROFILING
#define NT_PROFILING               0
#endif

// ============================================================================
// constexpr mirrors for C++ usage
// ============================================================================
namespace newtype::config {

constexpr bool kProduction             = NT_PRODUCTION;
constexpr bool kDebugViz               = NT_DEBUG_VIZ;
constexpr bool kAssert                 = NT_ASSERT_LOG;
constexpr bool kProfiling              = NT_PROFILING;

constexpr bool kAllowRasterFeatures    = NT_ALLOW_RASTER_FEATURES;
constexpr bool kEnableProcedural       = NT_ENABLE_PROCEDURAL;
constexpr bool kEnableDenoiser         = NT_ENABLE_DENOISER;
constexpr bool kEnableGI               = NT_ENABLE_GI;
constexpr bool kEnableEnvmap           = NT_ENABLE_ENVMAP;
constexpr bool kEnableCheckerboard     = NT_ENABLE_CHECKERBOARD;
constexpr bool kEnableSpecularDenoising = NT_ENABLE_SPECULAR_DENOSING;
constexpr bool kEnableTimelineEditor   = NT_ENABLE_TIMELINE_EDITOR;

constexpr bool kEnableBSSRDF           = NT_ENABLE_BSSRDF;

constexpr bool kEnableVideoRecorder    = NT_ENABLE_VIDEO_RECORDER;
constexpr bool kEnableScreenshot       = NT_ENABLE_SCREENSHOT;
constexpr bool kEnableAudio            = NT_ENABLE_AUDIO;
constexpr bool kEnableMediaPlayer      = NT_ENABLE_MEDIA_PLAYER;

constexpr bool kAllowCuda              = NT_ALLOW_CUDA;

} // namespace nt::config
