#pragma once

// Lobe-list BSDF refactor (Disney principled alignment).
// See plan: C:\Users\barca\.claude\plans\piped-discovering-walrus.md
// Phase 2 (vertical layering): C:\Users\barca\.claude\plans\eager-orbiting-kitten.md
// Gated by NT_ENABLE_LOBE_LIST (default 0). When off, MaterialBSDF::lobe_list
// is not declared and the old dispatch path is bit-identical to pre-refactor.

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
// LobeList — fixed-capacity list built by build_lobe_list() at make_bsdf() time
//==============================================================================

// Storage uses parallel arrays of scalar DSL vars rather than Var<Lobe>[N] so
// that compile-time-unrolled host `for` loops can index them with post-unroll
// constexpr indices. The iteration pattern in evaluate/sample/pdf:
//   for (uint i = 0u; i < LobeList::kMaxLobes; ++i) {
//       $if(i < list.count) {
//           Float w = list.weights[i];        // i is constexpr post-unroll
//           UInt  tf = list.type_flags[i];
//           UInt  t = lobe_type(tf);
//           UInt  f = lobe_flags(tf);
//           ...
//       };
//   }
//
// Phase 2A bit-pack: types[] + flags[] collapsed into a single type_flags[]
// array. Cap stays at 8 for Phase 2A-2C (single-layer); raised to 10 at start
// of Phase 2D (profile-gated). DSL-var count: 17 (was 25 in Phase 1).
struct LobeList {
    // Phase 2D: raised from 8 → 10 for vertical layering
    // (coat slot 0-1 + base slots 2-8 + fuzz slot 9).
    // DSL-var count: weights(10) + type_flags(10) + count(1) = 21.
    static constexpr uint kMaxLobes = 10u;
    Float weights[kMaxLobes];        // 10 vars
    UInt  type_flags[kMaxLobes];     // 10 vars — packed (type, flags)
    UInt  count;                     // 1 var = 21 total
};

} // namespace newtype::render
