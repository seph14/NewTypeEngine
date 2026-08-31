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
// DLL Export Functions
//==============================================================================

extern "C" {

CUSTOM_MATERIAL_API void registerMaterialCallables(
    CallableRegisterFn registerFn, CallableClearFn clearFn) {

    registerFn("checkerboard",        checkerboard_resolver);
    registerFn("animated_roughness",  animated_roughness_resolver);
    registerFn("screenspace_outline", screenspace_outline_resolver);
    registerFn("iridescent_tint",     iridescent_tint_resolver);

    printf("[CustomMaterialShader] Registered 4 custom callables\n");
}

CUSTOM_MATERIAL_API void unregisterMaterialCallables(CallableClearFn clearFn) {
    clearFn();
    printf("[CustomMaterialShader] Unregistered custom callables\n");
}

} // extern "C"
