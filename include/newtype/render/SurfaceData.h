#pragma once

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include "newtype/render/BSDF.h"

namespace newtype::render {

using namespace luisa;
using namespace luisa::compute;

//==============================================================================
// SurfaceData — Resolved surface parameters after texture sampling
//==============================================================================

/**
 * @brief Resolved surface parameter bundle (DSL-only, never stored in a buffer)
 *
 * Constructed by resolve_surface() from MaterialData + texture sampling.
 * This is the single interface between "material resolution" and "BSDF evaluation".
 *
 * When no textures are assigned (all texIdx == -1), all fields match the flat
 * MaterialData constants — output is pixel-identical to the non-textured path.
 */
struct SurfaceData {
    // --- Resolved material parameters (after texture sampling) ---
    Float3 albedo;
    Float3 emission;
    Float  roughness       {0.5f};
    Float  metallic        {0.f};
    Float  ior             {1.5f};
    Float  albedo_alpha    {1.f};   // Resolved per-pixel opacity (albedo.w * texture.w)
    Float  alphacut        {0.f};   // Alpha cutout threshold
    Float  ao              {1.f};
    Float  sheen           {0.f};
    Float  sheen_tint      {0.f};
    Float  clearcoat       {0.f};
    Float  clearcoat_gloss {0.5f};
    Float  iridescence           {0.f};
    Float  iridescence_ior       {1.3f};
    Float  iridescence_thickness {0.f};
    Float  anisotropic           {0.f};
    Float  anisotropic_rot       {0.f};
    Float3 attenuation     {1.f, 1.f, 1.f};
    // For Conductor materials (when conductor_k != 0): complex IOR real part.
    // attenuation is overloaded — same storage, per-material-type meaning.
    Float3 conductor_k     {0.f, 0.f, 0.f};
    Float  attenuation_distance  {1.f};
    Float  specular_tint   {0.f};
    Float  specular_trans   {0.f};
    Float  flatness         {0.f};
    Float  diffuse_trans    {0.f};
    Float  fabric           {0.f};

    // --- Surface geometry (after normal map perturbation) ---
    Float3 ns;              // shading normal (possibly perturbed by normal map)
    Float3 geo_ns;          // geometric normal (never perturbed)
    Float3 tangent          {luisa::compute::make_float3(1.f, 0.f, 0.f)};
    Float  tangent_w        {1.f};     // bitangent handedness

    // --- Position (computed by resolve_surface, not stored in buffer) ---
    Float3 position         {0.f, 0.f, 0.f};  // Vertex-interpolated position from bindless buffer

    // --- UV (computed by resolve_surface / build_procedural_surface_base) ---
    Float2 uv               {0.f, 0.f};       // Interpolated UV from vertices (or packed .w for procedural)

    // --- Material classification ---
    UInt   material_type    {0u};   // Raw type from MaterialData (used for callable dispatch)
    UInt   bsdf_type        {0u};   // Cached from MaterialData.bsdf_type_override (set by resolve_surface). Do NOT override in callables for dielectric — G-Buffer PSR depends on the material-side value.

    // --- Phase 2D vertical layering: composed LobeList + coat/fuzz layer params ---
    // Default-constructed count=0 → single-layer path (Phase 1 behavior, bit-identical).
    // resolve_surface_layered populates these → composed-list path inside make_bsdf().
    LobeList composed_lobe_list{};
    UInt     composed_lobe_list_count{0u};
    UInt   coat_bsdf_type           {0u};   // 7=Clearcoat, 3=Dielectric (layer≥1)
    Float  coat_clearcoat_val       {0.f};
    Float  coat_clearcoat_gloss_val {0.5f};
    Float  coat_ior                 {1.5f};
    Float  coat_roughness           {0.f};
    Float3 coat_attenuation         {1.f, 1.f, 1.f};
    Float3 fuzz_albedo              {1.f, 1.f, 1.f};
    Float  fuzz_sheen_val           {0.f};
    Float  fuzz_sheen_tint_val      {0.f};
    Float  fuzz_sheen_roughness_val {0.5f};
    // Two-interface Fresnel terms, evaluated once in resolve_surface_layered and
    // forwarded to MaterialBSDF so evaluate/evaluate_split don't recompute.
    // coat_F12 = air→coat Fresnel (always populated when coat present).
    // coat_F23 = coat→base Fresnel (only when NT_ENABLE_TWO_INTERFACE_FRESNEL).
    Float  coat_F12                 {0.f};
    Float  coat_F23                 {0.f};

    // Coat/fuzz transmittance for the Burley BSSRDF probe add (shade shader).
    // = base_scale from resolve_surface_layered — (1-F12)(1-F23)(+fuzz
    // budget) — so coat-over-SSS darkens like every other base lobe. Defaults
    // to 1 for non-layered surfaces. The probe's own F_x0/F_x2 terms cover the
    // base-medium interface; this factor covers the coat interfaces only.
    Float  sss_probe_scale          {1.f};

    // --- Kulla-Conty MS-GGX invariants (NT_ENABLE_MS_GGX) ---
    // Computed once in resolve_surface (they depend only on material + wo, not
    // wi). f_ms itself is added in MaterialBSDF's spec blocks, which pay only
    // the E_i fit per evaluate call. Defaults = no compensation.
    Float  ms_e_o                   {1.f};
    Float  ms_e_avg                 {1.f};
    Float3 ms_f_avg                 {0.f, 0.f, 0.f};

    /** @brief Construct MaterialBSDF from resolved surface parameters */
    [[nodiscard]] MaterialBSDF make_bsdf() const noexcept {
        MaterialBSDF bsdf{
            albedo, roughness, metallic, ior,
            sheen, sheen_tint,
            clearcoat, clearcoat_gloss,
            iridescence, iridescence_ior, iridescence_thickness,
            anisotropic, anisotropic_rot,
            tangent, tangent_w,
            bsdf_type,
            flatness,
            fabric,
            specular_tint, specular_trans, attenuation,
            diffuse_trans, attenuation_distance,
            conductor_k
        };

        // Item 10: cache the anisotropy-rotated tangent once per construction
        // (before the composed/standard branch so both paths get it). ns is
        // the normal every evaluate/sample/pdf entry point receives for this
        // surface, so the hoisted value is bit-identical to the former
        // per-evaluate computation.
        bsdf.precompute_tangent_rotation(ns);

        // composed-list path. When resolve_surface_layered has
        // populated composed_lobe_list (count > 0), bypass build_lobe_list()
        // and use the composed list directly. Single-layer path (count == 0)
        // runs build_lobe_list() → Phase 1 slot-specialized dispatch, bit-identical.
        $if(composed_lobe_list_count > 0u) {
            bsdf.lobe_list = composed_lobe_list;
            bsdf.has_composed_lobe_list = true;
            // Forward coat/fuzz params
            bsdf.coat_bsdf_type           = coat_bsdf_type;
            bsdf.coat_clearcoat_val       = coat_clearcoat_val;
            bsdf.coat_clearcoat_gloss_val = coat_clearcoat_gloss_val;
            bsdf.coat_ior                 = coat_ior;
            bsdf.coat_roughness           = coat_roughness;
            bsdf.coat_attenuation         = coat_attenuation;
            bsdf.fuzz_albedo              = fuzz_albedo;
            bsdf.fuzz_sheen_val           = fuzz_sheen_val;
            bsdf.fuzz_sheen_tint_val      = fuzz_sheen_tint_val;
            bsdf.fuzz_sheen_roughness_val = fuzz_sheen_roughness_val;
            bsdf.coat_F12                 = coat_F12;
            bsdf.coat_F23                 = coat_F23;
        }
        $else {
            bsdf.build_lobe_list();
        };

        // Kulla-Conty invariants from resolve_surface (defaults = no f_ms).
        bsdf.ms_e_o   = ms_e_o;
        bsdf.ms_e_avg = ms_e_avg;
        bsdf.ms_f_avg = ms_f_avg;

        return bsdf;
    }

    /** Direct-lighting variant of make_bsdf() for NT_ENABLE_SSS_PROBE_DIRECT:
     *  when the Burley BSSRDF probe covers this pixel, zero the HK subsurface
     *  lobe so direct light doesn't double-count surface term + probe. The
     *  probe (additive in the shade shader) becomes the single direct SSS
     *  model; the HK lobe stays active for GI throughput.
     *  @param probe_covered per-pixel gate: Subsurface material, flatness > 0,
     *         and not a point primitive (the probe pass skips points, so they
     *         must keep the HK lobe for direct light). */
    [[nodiscard]] MaterialBSDF make_bsdf(Bool probe_covered) const noexcept {
        MaterialBSDF bsdf = make_bsdf();
#if NT_ENABLE_BSSRDF && NT_ENABLE_SSS_PROBE_DIRECT
        // Slot 3 = Subsurface in the single-layer layout; composed lists shift
        // base lobes by +1 (slot 4). Same layout build_lobe_list() and
        // resolve_surface_layered() write.
        $if(probe_covered) {
            $if(bsdf.has_composed_lobe_list) {
                bsdf.lobe_list.weights[4] = 0.f;
            } $else {
                bsdf.lobe_list.weights[3] = 0.f;
            };
        };
#else
        (void)probe_covered;
#endif
        return bsdf;
    }
};

} // namespace newtype::render
