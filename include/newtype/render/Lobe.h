#pragma once

// Lobe-list BSDF refactor (Disney principled alignment).
//
// Two representations:
//  - LobeListData: OWNING parallel arrays (Float/UInt locals). The
//    materialization point — SurfaceData::composed_lobe_list, written under
//    $if branches by resolve_surface_layered (assignment-under-branch needs
//    real locals so DXC can phi them).
//  - LobeList: NON-OWNING view (Expr arrays). Held by MaterialBSDF; every
//    element is an expression-DAG reference, so constructing / copying /
//    passing a LobeList emits ZERO IR statements. Built by
//    build_standard_lobe_list() (pure DAG from the Disney params) and merged
//    with the composed list via per-slot ite() in SurfaceData::make_bsdf().

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

namespace newtype::render {

using namespace luisa;
using namespace luisa::compute;

//==============================================================================
// LobeType — enumerates the 9 concrete lobe kinds emitted by build_lobe_list()
//==============================================================================

enum class LobeType : uint {
    Diffuse            = 0u,  // Burley retro-modulated Lambertian
    Subsurface         = 1u,  // Hanrahan-Krueger SSS shape (not full BSSRDF)
    SpecularDielectric = 2u,  // GGX dielectric reflection (Plastic spec lobe)
    SpecularMetal      = 3u,  // GGX conductor reflection (complex IOR or Schlick)
    Transmission       = 4u,  // GGX dielectric refraction (specular_trans)
    Sheen              = 5u,  // Charlie sheen — additive eval; sampled under NT_ENABLE_SHEEN_SAMPLING
    Clearcoat          = 6u,  // Berry/GGX coat — ADDITIVE (eval-only; composed slot 0 sampled under NT_ENABLE_COAT_SAMPLING)
    DeltaDielectric    = 7u,  // Delta reflection+refraction (Dielectric/ThinDielectric)
    Fabric             = 8u,  // Ashikhmin-Premoze diffuse variant
};

//==============================================================================
// Lobe.flags bit constants
//==============================================================================

inline constexpr uint kLobeIsAdditive     = 1u;  // eval-only; skip in sample pick + pdf
inline constexpr uint kLobeIsReflection   = 2u;  // routes to out_specular in evaluate_split
inline constexpr uint kLobeIsTransmission = 4u;  // routes to out_diffuse in evaluate_split

//==============================================================================
// Bit-packing helpers — store (type, flags) in a single UInt per slot
//   low  nibble (bits 0-3):  LobeType
//   high nibble (bits 4-7):  flags (kLobeIs*)
//==============================================================================

inline UInt lobe_type(UInt packed) noexcept {
    return packed & 0xFu;
}

inline UInt lobe_flags(UInt packed) noexcept {
    return (packed >> 4u) & 0xFu;
}

inline UInt pack_lobe(Expr<uint> type, Expr<uint> flags) noexcept {
    return (flags << 4u) | (type & 0xFu);
}

//==============================================================================
// Lobe — single lobe metadata (documentation only).
//
// The runtime representation is the parallel-array LobeList below; this struct
// exists to document the (type, weight, flags) tuple conceptually. LobeList
// can't use Var<Lobe>[N] because host-side C-style arrays of DSL Var types
// don't behave well across the DSL/frontend boundary, and LobeList itself
// isn't registered with LUISA_STRUCT (it's a host aggregate, not a buffer
// struct).
//==============================================================================

struct Lobe {
    uint  type;       // LobeType
    float weight;     // eval-time weight; sampling-pool weight unless additive
    uint  flags;      // combination of kLobe* bits
};

//==============================================================================
// LobeListData — OWNING fixed-capacity list (Float/UInt locals)
//==============================================================================

// Storage uses parallel arrays of scalar DSL vars rather than Var<Lobe>[N] so
// that compile-time-unrolled host `for` loops can index them with post-unroll
// constexpr indices. Written under $if branches (resolve_surface_layered) —
// assignment-under-branch requires owning locals (phi semantics).
// Phase 2D: cap 10 (coat slot 0-1 + base slots 2-8 + fuzz slot 9).
struct LobeListData {
    // Phase 2D: raised from 8 → 10 for vertical layering
    // (coat slot 0-1 + base slots 2-8 + fuzz slot 9).
    // DSL-var count: weights(10) + type_flags(10) + count(1) = 21.
    static constexpr uint kMaxLobes = 10u;
    Float weights[kMaxLobes];        // 10 vars
    UInt  type_flags[kMaxLobes];     // 10 vars — packed (type, flags)
    UInt  count;                     // 1 var = 21 total
};

//==============================================================================
// LobeList — NON-OWNING view (Expr arrays), zero IR cost to build/copy
//==============================================================================

// Same parallel-array layout and constexpr-index iteration pattern as
// LobeListData, but every element is an Expr reference into an existing
// expression DAG (computed lobe weights, composed-list locals, or literal
// constants). Aggregate-initialize with all kMaxLobes elements — Expr has no
// default constructor and no assignment, which is exactly the view
// discipline: build once in a factory, never mutate.
struct LobeList {
    static constexpr uint kMaxLobes = LobeListData::kMaxLobes;
    Expr<float> weights[kMaxLobes];
    Expr<uint>  type_flags[kMaxLobes];
    Expr<uint>  count;
};

} // namespace newtype::render
