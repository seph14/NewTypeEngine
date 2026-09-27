/**
 * @brief Custom Material Shader — Example Callables
 *
 * EDIT THIS CODE FOR HOT-RELOAD!
 *
 * Each callable receives surface data after standard texture sampling
 * and can modify it before shading. The callable runs inside device.compile()
 * and builds DSL AST nodes — it is NOT runtime GPU code.
 *
 * Parameters:
 *   SurfaceData s    — resolved surface (albedo, roughness, metallic, etc.)
 *                      UV is in s.uv (set by resolve_surface / build_procedural_surface_base).
 *   Var<MaterialData> mat — raw material data (type, albedo, etc.)
 *   Float2 screen_uv — pixel position / resolution [0,1]
 *   Float3 wo        — view direction (outgoing)
 *   Float time       — frame_count as float
 *   const BindlessVar& tex — texture bindless array
 *   UInt w, UInt h   — screen dimensions
 *
 * Return: modified SurfaceData
 */

#include "CustomMaterialShaderAPI.h"
#include "newtype/render/Shading.h"
#include <luisa/dsl/sugar.h>
#include <cstdio>

using namespace luisa;
using namespace luisa::compute;
using namespace newtype::render;

//==============================================================================
// Example 1: Checkerboard — Procedural albedo from UV tiling
//==============================================================================
static SurfaceResolveFn checkerboard_resolver =
    [](SurfaceData s, Var<MaterialData> mat,
       Float2 screen_uv, Float3 wo, Float time,
       const BindlessVar& tex, UInt w, UInt h) -> SurfaceData {
    Float2 uv = s.uv;
    Float scale = 10.0f;
    Float checker = step(0.5f, fract(uv.x * scale))
                  * step(0.5f, fract(uv.y * scale))
                  + step(0.5f, fract((uv.x + 0.5f / scale) * scale))
                  * step(0.5f, fract((uv.y + 0.5f / scale) * scale));
    checker = saturate(checker);
    Float3 dark = make_float3(1.0f, 0.5f, 0.2f);
    s.albedo = lerp(dark, s.albedo, checker);
    return s;
};

//==============================================================================
// Example 2: Animated Roughness — Time-based sine wave
//==============================================================================
static SurfaceResolveFn animated_roughness_resolver =
    [](SurfaceData s, Var<MaterialData> mat,
       Float2 screen_uv, Float3 wo, Float time,
       const BindlessVar& tex, UInt w, UInt h) -> SurfaceData {
    Float2 uv = s.uv;
    // Slowly oscillate roughness between 0.05 and 0.6
    Float phase = time * 0.02f;
    Float wave  = sin(uv.x * 6.2832f + phase) * 0.5f + 0.5f;
    s.roughness = lerp(0.05f, 0.6f, wave);
    return s;
};

//==============================================================================
// Example 3: Screen-Space Outline — Edge detection via screen_uv → emission
//==============================================================================
static SurfaceResolveFn screenspace_outline_resolver =
    [](SurfaceData s, Var<MaterialData> mat,
       Float2 screen_uv, Float3 wo, Float time,
       const BindlessVar& tex, UInt w, UInt h) -> SurfaceData {
    // Detect edges via UV discontinuity (approximate screen-space derivative)
    Float edge_u = min(fract(screen_uv.x * cast<float>(w)),
                       1.0f - fract(screen_uv.x * cast<float>(w)));
    Float edge_v = min(fract(screen_uv.y * cast<float>(h)),
                       1.0f - fract(screen_uv.y * cast<float>(h)));
    Float edge   = min(edge_u, edge_v);

    // Glow near edges (within ~2 pixels)
    Float glow = saturate(1.0f - edge * 50.0f);

    // Add cyan emission along edges
    s.emission = s.emission + make_float3(0.0f, 1.0f, 1.0f) * glow * 5.0f;
    return s;
};

//==============================================================================
// Example 4: Iridescent Tint — View-dependent thin-film color shift
//==============================================================================
static SurfaceResolveFn iridescent_tint_resolver =
    [](SurfaceData s, Var<MaterialData> mat,
       Float2 screen_uv, Float3 wo, Float time,
       const BindlessVar& tex, UInt w, UInt h) -> SurfaceData {
    // Fresnel-like view-dependent factor
    Float cos_theta = abs(dot(normalize(wo), make_float3(0.0f, 1.0f, 0.0f)));
    Float fresnel = pow(1.0f - cos_theta, 3.0f);

    // Thin-film interference approximation: shift hue based on view angle
    Float hue_shift = fresnel * 6.2832f;
    Float r = sin(hue_shift) * 0.5f + 0.5f;
    Float g = sin(hue_shift + 2.094f) * 0.5f + 0.5f;
    Float b = sin(hue_shift + 4.189f) * 0.5f + 0.5f;
    Float3 iridescence = make_float3(r, g, b);

    // Blend with base albedo, stronger at grazing angles
    s.albedo = lerp(s.albedo, s.albedo * iridescence * 2.0f, fresnel);
    return s;
};

//==============================================================================
// Example 5: Glass Blend — diffuse <-> window-glass blending
// (docs/glass_blend_plan.md, docs/custom_material_callables.md)
//
// Requires the material to be a CUSTOM type (>= render::Material::CustomType) with
// bsdf_type_override = 3 (Dielectric): that classification routes the pixel
// into the G-Buffer PSR glass branch, where the resolver's s.glass_blend
// stochastically selects per pixel (static IGN dither, stable across frames):
//   glass_blend = 1  -> pure glass (trace through, legacy path)
//   glass_blend = 0  -> pure opaque (store surface, shade the diffuse side)
// Everything in between is a dithered mix; E[pixel] = b*glass + (1-b)*diffuse.
// The SAME SurfaceData params serve both sides: albedo/roughness drive the
// opaque roll, ior/attenuation drive the glass roll. The deterministic shadow
// path mixes automatically ((1-b) + b*(1-F)*absorption).
//
// RUNTIME PARAMS (ABI v2, docs/resolver_params_abi_plan.md): the split is
// driven from the engine's params buffer — UI sliders / --resolver-param
// move it with NO shader recompile. The base comes from paramFn at
// registration (captured by value). Scalars pack 4 per float4 in
// registration order, so the layout here is:
//   [0].x band_center  [0].y band_width
//   [0].z diffuse_r    [0].w diffuse_g   [1].x diffuse_b
//==============================================================================

// Sentinel for "no params available" (v1 host or registration overflow):
// the resolver keeps the hard-coded demo split.
static constexpr std::uint32_t kGlassBlendNoParamBase = ~0u;

static SurfaceResolveFn make_glass_blend_resolver(std::uint32_t base) {
    return [base](SurfaceData s, Var<MaterialData> mat,
                  Float2 screen_uv, Float3 wo, Float time,
                  const BindlessVar& tex, UInt w, UInt h) -> SurfaceData {
        if (base == kGlassBlendNoParamBase) {
            // v1/no-params fallback — original hard split at world x = -0.4
            // (sphere spans [-0.81, 0.01], center -0.4).
            s.glass_blend = ite(s.position.x > -0.4f, 0.0f, 1.0f);
            s.albedo      = make_float3(1.0f);
        } else {
            Float4 p_blend  = resolver_params(tex, base, 0u);
            Float4 p_albedo = resolver_params(tex, base, 1u);
            Float center = p_blend.x;
            Float width  = max(p_blend.y, 0.0f);
            // Width 0 → hard split (smoothstep over equal edges divides by
            // zero); the discarded lane is never selected. Glass on the LEFT
            // of the split: fraction 1 → trace through, 0 → opaque roll.
            Float hard = ite(s.position.x < center, 1.0f, 0.f);
            Float soft = 1.0f - smoothstep(center - width, center + width, s.position.x);
            s.glass_blend = ite(width <= 0.0f, hard, soft);
            // Scalars pack 4-per-float4: r/g/b = [0].z, [0].w, [1].x.
            s.albedo = make_float3(p_blend.z, p_blend.w, p_albedo.x);
        }
        // Diffuse-side params for the opaque roll. specular_trans must be 0 —
        // glass-authored materials carry 1, which routes the opaque roll's lobe
        // energy into transmission (black diffuse). (The engine's reclass helper
        // also zeroes it; set it here so the intent is explicit.)
        s.specular_trans = 0.0f;
        return s;
    };
}

//==============================================================================
// Example 6: Per-Instance Variation (track B2,
// docs/vertex-packing-instancing-plan.md §4) — hash-tinted albedo + optional
// authored per-instance params.
//
// Every surface resolve now carries s.instance_index (the TLAS instance row;
// the procedural index on procedural surfaces). This resolver derives a
// stable per-instance hue from it — unbounded instances, one material — and
// optionally reads the instance's authored 64 B params row:
//   Pipeline::setInstanceUserData(id, slot 0..3, float4)
// writes rows the resolver reads via instance_params(tex, s.instance_index, i).
// Here row 0 = (tint_strength, roughness_bias, emissive_boost, _).
//==============================================================================
static SurfaceResolveFn per_instance_variation_resolver =
    [](SurfaceData s, Var<MaterialData> mat,
       Float2 screen_uv, Float3 wo, Float time,
       const BindlessVar& tex, UInt w, UInt h) -> SurfaceData {
    // PCG hash of the instance row -> [0,1) hue (stable across frames).
    UInt h1 = s.instance_index * 747796405u + 2891336453u;
    UInt h2 = (h1 >> ((h1 >> 28u) + 4u)) ^ h1;
    UInt h3 = (h2 * 277803737u) ^ (h2 >> 22u);
    Float r01 = fract(static_cast<Float>(h3 * 2654435769u & 0xFFFFFFu) * (1.0f / 16777216.0f));
    Float hue = r01;
    // HSV -> RGB (h in [0,1), s = v = 1): 6-segment ramp.
    Float seg = hue * 6.0f;
    Float3 tint = saturate(make_float3(
        abs(seg - 3.0f) - 1.0f,
        2.0f - abs(seg - 2.0f),
        2.0f - abs(seg - 4.0f)));

    // Authored row (zeros until the app writes one — strengths then default
    // to a mild 25% tint blend, which also shows the hash variation bare).
    Float4 p0 = instance_params(tex, s.instance_index, 0u);
    Float strength = ite(p0.x > 0.0f, p0.x, 0.25f);
    s.albedo = lerp(s.albedo, s.albedo * tint, saturate(strength));
    s.roughness = saturate(s.roughness + p0.y);
    s.emission += tint * p0.z;
    return s;
};

//==============================================================================
// DLL Export Functions
//==============================================================================

extern "C" {

CUSTOM_MATERIAL_API void registerMaterialCallables(
    CallableRegisterFn registerFn, CallableClearFn clearFn) {

    registerFn("checkerboard",        checkerboard_resolver);
    registerFn("animated_roughness",  animated_roughness_resolver);
    registerFn("screenspace_outline", screenspace_outline_resolver);
    registerFn("iridescent_tint",     iridescent_tint_resolver);
    registerFn("glass_blend",         make_glass_blend_resolver(kGlassBlendNoParamBase));
    // Tag order is ABI: scenes hardcode dispatch types (MaterialTest's
    // glass_blend = 18). New callables append AFTER the existing ones.
    registerFn("per_instance_variation", per_instance_variation_resolver);

    printf("[CustomMaterialShader] Registered 6 custom callables\n");
}

// ABI v2 — callables + runtime tuning params (docs/resolver_params_abi_plan.md).
// paramFn returns the callable's float4 base in the engine's params buffer;
// capture it by value in the resolver and read via resolver_params(tex, base, i).
// Value edits host-side (UI sliders) never recompile shaders. paramFn may be
// null (host without params support) — skip registration then.
//
// glass_blend params (defaults = the hard-coded fallback: hard split at
// world x=-0.4, white diffuse side):
//   band_center/band_width — glass<->diffuse split position/softness
//     (width 0 = hard split; width > 0 = smoothstep over [c-w, c+w])
//   diffuse_r/g/b          — opaque-roll albedo
static const ResolverParamDesc kGlassBlendParams[] = {
    {"band_center", -0.8f,  0.5f, -.4f},
    {"band_width",   0.0f,  0.5f,  0.0f},
    {"diffuse_r",    0.0f,  1.0f,  1.0f},
    {"diffuse_g",    0.0f,  1.0f,  1.0f},
    {"diffuse_b",    0.0f,  1.0f,  1.0f},
};

CUSTOM_MATERIAL_API void registerMaterialCallables2(
    CallableRegisterFn registerFn, ParamRegisterFn paramFn, CallableClearFn clearFn) {

    registerFn("checkerboard",        checkerboard_resolver);
    registerFn("animated_roughness",  animated_roughness_resolver);
    registerFn("screenspace_outline", screenspace_outline_resolver);
    registerFn("iridescent_tint",     iridescent_tint_resolver);

    // Params FIRST: paramFn returns the float4 base the resolver captures.
    // ~0u (no paramFn / overflow) keeps the hard-coded fallback split.
    std::uint32_t blend_base = kGlassBlendNoParamBase;
    if (paramFn) {
        blend_base = paramFn("glass_blend", kGlassBlendParams, 5u);
    }
    registerFn("glass_blend", make_glass_blend_resolver(blend_base));
    // Tag order is ABI (see v1 registration): glass_blend stays 18, new
    // callables append after it — per_instance_variation = 19.
    registerFn("per_instance_variation", per_instance_variation_resolver);

    printf("[CustomMaterialShader] Registered 6 custom callables (v2, params%s, base %u)\n",
           paramFn ? "" : " unavailable", blend_base);
}

CUSTOM_MATERIAL_API void unregisterMaterialCallables(CallableClearFn clearFn) {
    clearFn();
    printf("[CustomMaterialShader] Unregistered custom callables\n");
}

// Layout-ABI version (track B2): the loader compares this against the exe's
// newtype::render::kCallableAbiVersion and force-rebuilds on mismatch —
// bump the engine constant on every SurfaceData/MaterialData/signature
// layout change and this export picks it up via the shared header.
CUSTOM_MATERIAL_API std::uint32_t ntCallableAbiVersion(void) {
    return newtype::render::kCallableAbiVersion;
}

} // extern "C"
