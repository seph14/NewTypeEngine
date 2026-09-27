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

// --- LuisaCompute CPU-side validation layer (device-usage checking that
// wraps the DeviceInterface). Off by default: the layer adds per-call
// overhead that skews perf numbers, and the compile profiler does NOT
// require it. Flip to 1 only when debugging device usage.
#ifndef NT_ENABLE_VALIDATION
#define NT_ENABLE_VALIDATION       0
#endif

// --- GPU vertex layout (track A2, docs/vertex-packing-instancing-plan.md §5)
// 0 = fp32 48 B (the legacy layout)
// 1 = packed32 — fp32 pos + snorm10-10-10-2 normal + snorm10 tangent (32 B)
//     [DEFAULT since A2 measured +0.7% cornell fps (interleaved A/B, n=3,
//     non-overlapping), capture A/B within same-exe noise on every smoke
//     scene, and -33% vertex VRAM on mesh-heavy scenes]
// 2 = packed40 — fp32 pos + snorm10 normal + fp32 tangent + fp32 UV (40 B)
//     (fallback rung if a scene shows snorm tangent artifacts)
// Compile-time on purpose: the DSL capture types are compile-time, so a
// runtime knob would force templating every kernel on the vertex type; the
// A/B measurement protocol rebuilds per side anyway. Override at build time
// (/DNT_VERTEX_LAYOUT=0) or edit here. Layout switches do NOT bump the
// callable-DLL ABI (no vertex type crosses the exe<->DLL boundary).
#ifndef NT_VERTEX_LAYOUT
#define NT_VERTEX_LAYOUT           1
#endif

// --- Features ---
#define NT_ALLOW_RASTER_FEATURES   0
// Validation harness for the tet-cage prototype (docs/tetrahedral-cage-
// prototype.md Stage 2) flips this to 1 — the tetcage test scene renders its
// procedural type-3 reference under it. Default build: off.
#define NT_ENABLE_PROCEDURAL       0
#define NT_ENABLE_DENOISER         1
#define NT_ENABLE_GI               1
#define NT_ENABLE_ENVMAP           1
#define NT_ENABLE_SPECULAR_DENOSING 1
#define NT_ENABLE_TIMELINE_EDITOR  1
// Editor gizmo overlay (feature::Gizmo): world-space lines/points/cubes
// composited onto the final image after tonemap. Push sites keep compiling
// when off (no-op Gizmo); auto-off in production.
#define NT_ENABLE_EDITOR           1
// Delta NEE branch enables mirror metal GI
#define NT_ENABLE_DELTA_BRANCH_NEE 1

// --- SSS direct model: Burley probe replaces the HK lobe in DI ---
// When on, direct-lighting BSDFs (PassDI
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

// --- Belcour-Barla thin-film iridescence (KHR_materials_iridescence port) ---
// Substrate-aware thin-film interference with analytic spectral integration
// (Gaussian-fit CIE XYZ) replaces the legacy 3-wavelength Airy model whose
// film->substrate Fresnel ignored the base material. OFF = legacy Airy path,
// bit-identical. See docs/iridescence.md.
#ifndef NT_ENABLE_BELCOUR_IRIDESCENCE
#define NT_ENABLE_BELCOUR_IRIDESCENCE 1
#endif

// --- Abbe-number dispersion (KHR_materials_dispersion) ---
// Per-channel (R/G/B) IOR picks at single-sample estimator sites only: the
// rough-glass gather taps (PipelineInit — deterministic balanced (k+frame)%3
// allocation, x3 channel-basis weight), the smooth-glass post-denoise
// replay (3 deterministic delta-refracted taps in the tint pass; speckle
// plan Phase 3), MicrofacetTransmissionBSDF::sample, and the dispersive
// shadow walk (trace_shadow: first dispersive interface fans out to 3
// deterministic per-channel refracted sub-walks behind a deviation-angle
// gate; docs/Dispersive-Glass-Shadows.md). The PSR glass chain (PassDI
// G-buffer) stays d-line: a per-frame channel pick there broke the
// stable-G-buffer contract (ReLAX history resets = speckle;
// docs/dispersion_speckle_fix_plan.md). Reflection Fresnel and
// evaluate()/pdf() stay monochromatic (d-line) so ReSTIR p_hat stays
// RGB-stable. OFF = bit-identical. See docs/dispersion.md.
#ifndef NT_ENABLE_DISPERSION
#define NT_ENABLE_DISPERSION       1
#endif

// --- GGX multiple-scattering energy compensation (Kulla-Conty) ---
// Adds an f_ms lobe to MicrofacetBSDF::evaluate so rough GGX surfaces
// (especially complex-IOR conductors) don't darken as roughness rises.
// Analytic fits (Chebyshev 8x8 for E, deg-5 poly for E_avg) — no LUT.
// OFF = bit-identical. See docs/ms_ggx_compensation.md + tools/ms_ggx_fit.py.
#ifndef NT_ENABLE_MS_GGX
#define NT_ENABLE_MS_GGX 1
#endif

// --- SHARC world-space radiance cache (docs/sharc_rough_glass_plan.md) ---
// Master gate for the SHARC integration (Update/Resolve/Query passes and
// the rough-glass gather). Phase 0 ports the DSL core
// (include/newtype/render/Sharc.h + tools/sharc_harness.cpp parity gate)
// without touching the render pipeline; flipping to 1 lands the passes.
// Capacity is fixed per scene (owner decision, plan §8-1): dev default
// 2^20 entries (40 MiB; halved from 2^21 by Phase-2 hit-rate tuning),
// scenes override.
#ifndef NT_ENABLE_SHARC
#define NT_ENABLE_SHARC          1
#endif
// Key layout for the integrated passes: 0 = 64-bit keys (default; 8 B
// entries), 1 = compact 32-bit keys. The parity harness exercises both.
#ifndef NT_SHARC_COMPACT
#define NT_SHARC_COMPACT         1
#endif
// Insert route: 1 = native compare-exchange on the key (perf; 64-bit keys
// make the kernel SM6.6 — the pass calls set_warp_size), 0 = upstream's
// lock-buffer route. Owner decision: take the perf win (plan §8-2).
#ifndef NT_SHARC_64_BIT_ATOMICS
#define NT_SHARC_64_BIT_ATOMICS  1
#endif

// --- Media ---
#define NT_ENABLE_VIDEO_RECORDER   1
#define NT_ENABLE_SCREENSHOT       1
#define NT_ENABLE_AUDIO            1
// HOA volumetric audio: Steam Audio phonon ambisonics encode + HRTF binaural
// decode (newtype/audio). Requires NT_ENABLE_AUDIO; off with it.
#ifndef NT_ENABLE_SPATIAL_AUDIO
#define NT_ENABLE_SPATIAL_AUDIO    1
#endif
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
#define NT_PROFILING              0
#undef NT_ENABLE_EDITOR
#define NT_ENABLE_EDITOR           0
#endif

// ============================================================================
// constexpr mirrors for C++ usage
// ============================================================================
namespace newtype::config {

constexpr bool kProduction             = NT_PRODUCTION;
constexpr bool kDebugViz               = NT_DEBUG_VIZ;
constexpr bool kAssert                 = NT_ASSERT_LOG;
constexpr bool kProfiling              = NT_PROFILING;
constexpr bool kEnableValidation       = NT_ENABLE_VALIDATION;

constexpr bool kAllowRasterFeatures    = NT_ALLOW_RASTER_FEATURES;
constexpr bool kEnableProcedural       = NT_ENABLE_PROCEDURAL;
constexpr bool kEnableDenoiser         = NT_ENABLE_DENOISER;
constexpr bool kEnableGI               = NT_ENABLE_GI;
constexpr bool kEnableEnvmap           = NT_ENABLE_ENVMAP;
constexpr bool kEnableSpecularDenoising = NT_ENABLE_SPECULAR_DENOSING;
constexpr bool kEnableTimelineEditor   = NT_ENABLE_TIMELINE_EDITOR;
constexpr bool kEnableEditor           = NT_ENABLE_EDITOR;

constexpr bool kEnableSharc            = NT_ENABLE_SHARC;
constexpr bool kSharcCompact           = NT_SHARC_COMPACT;
constexpr bool kSharc64BitAtomics      = NT_SHARC_64_BIT_ATOMICS;

constexpr bool kEnableVideoRecorder    = NT_ENABLE_VIDEO_RECORDER;
constexpr bool kEnableScreenshot       = NT_ENABLE_SCREENSHOT;
constexpr bool kEnableAudio            = NT_ENABLE_AUDIO;
constexpr bool kEnableMediaPlayer      = NT_ENABLE_MEDIA_PLAYER;
constexpr bool kEnableSpatialAudio     = NT_ENABLE_SPATIAL_AUDIO;

constexpr bool kAllowCuda              = NT_ALLOW_CUDA;

} // namespace nt::config
