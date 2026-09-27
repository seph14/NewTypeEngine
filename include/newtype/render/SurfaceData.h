#pragma once

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include "newtype/render/BSDF.h"
#include "newtype/render/CallableAbiVersion.h"

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
 * This is the single interface between "material resolution" and "BSDF
 * evaluation" — and the single OWNING materialization of the resolved
 * parameters. MaterialBSDF views into it (see SurfaceData::make_bsdf), so
 * the params exist exactly once per shade; building a MaterialBSDF is free.
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
    // Abbe number (0 = off) — read by the glass transport paths (PSR chain,
    // rough-glass gather); never by MaterialBSDF, whose transmission sampling
    // and evaluate()/pdf() stay monochromatic (d-line) for ReSTIR stability.
    Float  dispersion      {0.f};
    Float  flatness         {0.f};
    Float  diffuse_trans    {0.f};
    Float  fabric           {0.f};
    // Callable-driven glass blending (docs/glass_blend_plan.md): fraction of
    // the GLASS response for a surface classified dielectric (bsdf_type 3/11).
    // 1 = pure glass (default — bit-identical legacy path); custom callables
    // lower it toward 0 to blend the SAME surface params as an opaque BSDF.
    // Only read where the effective bsdf type is 3/11: the G-Buffer PSR roll
    // (PassDI), the deterministic shadow walk, and the slim shadow resolve.
    Float  glass_blend      {1.f};

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
    LobeListData composed_lobe_list{};
    UInt     composed_lobe_list_count{0u};
    UInt   coat_bsdf_type           {0u};   // 7=Clearcoat, 3=Dielectric (layer≥1)
    Float  coat_clearcoat_val       {0.f};
    Float  coat_clearcoat_gloss_val {0.5f};
    Float  coat_ior                 {1.5f};
    Float  coat_roughness           {0.f};
    Float3 coat_attenuation         {1.f, 1.f, 1.f};
    // Coat shading normal — the coat layer's own normal map (Clearcoat-type
    // layers carry normalTexIdx), perturbed from geo_ns over the same tangent
    // frame in resolve_surface_layered. Defaults to ns when the coat has no
    // normal texture, so untextured coats stay bit-identical to the base
    // normal. Only read by the slot-0 coat dispatch (never when coat absent).
    Float3 coat_ns;
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

    // --- Instance identity (track B2, docs/vertex-packing-instancing-plan.md) ---
    // TLAS instance row the surface was resolved from (0 = not populated /
    // identity resolvers never read it). Set before the polymorphic dispatch
    // by the resolve entries that know it; custom callables use it for
    // per-instance variation without material-pool pressure. For procedural
    // surfaces this carries the procedural instance index instead (NOT a
    // TLAS row — instance_params() rows exist for mesh instances only).
    // DLL-ABI: part of kCallableAbiVersion.
    UInt   instance_index           {0u};

    //----------------------------------------------------------------------
    // MaterialBSDF view factories.
    //
    // MaterialBSDF is a non-owning view (Expr members — see BSDF.h); these
    // factories assemble it in one aggregate initialization. Beyond the
    // lobe-weight DAG (build_standard_lobe_list) and the t_rot rotation,
    // construction emits ZERO IR statements — the former ~60 locals +
    // assignments per construction (24 positional copies, 14 coat/fuzz
    // forwards, composed-list branch, MS copies) are gone. The bsdf /
    // bsdf_direct / rough_bsdf triple-materialization in the shade kernel
    // now shares one set of values.
    //----------------------------------------------------------------------

    /// LobeList view: standard DAG merged with the composed list (layered
    /// surfaces) via per-slot ite — value-identical to the former
    /// $if(composed) branch-and-copy, including the optional HK-lobe zeroing
    /// used by the SSS probe direct-lighting path (standard slot 3 /
    /// composed slot 4, the same slots the old $if zeroed).
    [[nodiscard]] LobeList make_lobe_list(Expr<bool> zero_hk_lobe) const noexcept {
        LobeList std_list = build_standard_lobe_list(
            metallic, specular_trans, flatness, sheen, fabric, clearcoat, bsdf_type);
        Expr<bool> is_composed = composed_lobe_list_count > 0u;
        return LobeList{
            {ite(is_composed, composed_lobe_list.weights[0], std_list.weights[0]),
             ite(is_composed, composed_lobe_list.weights[1], std_list.weights[1]),
             ite(is_composed, composed_lobe_list.weights[2], std_list.weights[2]),
             // SSS slot: standard layout slot 3 — zeroable for the probe path.
             ite(is_composed, composed_lobe_list.weights[3], ite(zero_hk_lobe, 0.f, std_list.weights[3])),
             // SSS slot: composed layout slot 4 (base lobes shifted +1).
             ite(is_composed, ite(zero_hk_lobe, 0.f, composed_lobe_list.weights[4]), std_list.weights[4]),
             ite(is_composed, composed_lobe_list.weights[5], std_list.weights[5]),
             ite(is_composed, composed_lobe_list.weights[6], std_list.weights[6]),
             ite(is_composed, composed_lobe_list.weights[7], std_list.weights[7]),
             ite(is_composed, composed_lobe_list.weights[8], std_list.weights[8]),
             ite(is_composed, composed_lobe_list.weights[9], std_list.weights[9])},
            {ite(is_composed, composed_lobe_list.type_flags[0], std_list.type_flags[0]),
             ite(is_composed, composed_lobe_list.type_flags[1], std_list.type_flags[1]),
             ite(is_composed, composed_lobe_list.type_flags[2], std_list.type_flags[2]),
             ite(is_composed, composed_lobe_list.type_flags[3], std_list.type_flags[3]),
             ite(is_composed, composed_lobe_list.type_flags[4], std_list.type_flags[4]),
             ite(is_composed, composed_lobe_list.type_flags[5], std_list.type_flags[5]),
             ite(is_composed, composed_lobe_list.type_flags[6], std_list.type_flags[6]),
             ite(is_composed, composed_lobe_list.type_flags[7], std_list.type_flags[7]),
             ite(is_composed, composed_lobe_list.type_flags[8], std_list.type_flags[8]),
             ite(is_composed, composed_lobe_list.type_flags[9], std_list.type_flags[9])},
            ite(is_composed, composed_lobe_list.count, std_list.count)};
    }

    /// Assemble the MaterialBSDF view over this surface (plus an optional
    /// roughness override — see make_bsdf_roughened). t_rot is the same
    /// anisotropy rotation the former precompute_tangent_rotation(ns) cached.
    [[nodiscard]] MaterialBSDF assemble_bsdf(LobeList const& lobes, Expr<float> roughness_over) const noexcept {
        Expr<luisa::float3> bitangent = cross(ns, tangent) * tangent_w;
        Expr<luisa::float3> t_rot = normalize(
            tangent * cos(anisotropic_rot) + bitangent * sin(anisotropic_rot));
        return MaterialBSDF{
            albedo, roughness_over, metallic, ior,
            sheen, sheen_tint,
            clearcoat, clearcoat_gloss,
            iridescence, iridescence_ior, iridescence_thickness,
            anisotropic, anisotropic_rot,
            tangent, tangent_w,
            bsdf_type,
            flatness, fabric,
            specular_tint, specular_trans, attenuation,
            diffuse_trans, attenuation_distance,
            conductor_k,
            t_rot,
            lobes,
            composed_lobe_list_count > 0u,
            coat_bsdf_type, coat_clearcoat_val, coat_clearcoat_gloss_val,
            coat_ior, coat_roughness, coat_attenuation, coat_ns,
            fuzz_albedo, fuzz_sheen_val, fuzz_sheen_tint_val, fuzz_sheen_roughness_val,
            coat_F12, coat_F23,
            ms_e_o, ms_e_avg, ms_f_avg};
    }

    /** @brief Construct a non-owning MaterialBSDF view over this surface. */
    [[nodiscard]] MaterialBSDF make_bsdf() const noexcept {
        return assemble_bsdf(make_lobe_list(false), roughness);
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
#if NT_ENABLE_SSS_PROBE_DIRECT
        return assemble_bsdf(make_lobe_list(probe_covered), roughness);
#else
        (void)probe_covered;
        return make_bsdf();
#endif
    }

    /** RTXDI FinalShading MIS roughened reference: same view with roughness
     *  floored. Shares the exact lobe weights with the true BSDF — the
     *  standard weights (metallic / specular_trans / flatness / sheen /
     *  fabric / clearcoat / bsdf_type) have no roughness dependence, and the
     *  composed list never did — and the same MS-GGX invariants, matching the
     *  former copy-then-modify behavior (which never recomputed them)
     *  bit-for-bit. Replaces "rough_bsdf = bsdf; rough_bsdf.roughness = ...;
     *  build_lobe_list()". */
    [[nodiscard]] MaterialBSDF make_bsdf_roughened(Expr<float> roughness_floor) const noexcept {
        return assemble_bsdf(make_lobe_list(false), max(roughness, roughness_floor));
    }
};

} // namespace newtype::render
