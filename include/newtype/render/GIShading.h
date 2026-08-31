#pragma once

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include "newtype/render/Shading.h"
#include "newtype/render/BSDF.h"

namespace newtype::render {

using namespace luisa;
using namespace luisa::compute;

//==============================================================================
// GI Target PDF — luminance(BRDF * radiance * cos_theta)
//==============================================================================

/**
 * @brief Evaluate target PDF for GI reservoir (scalar luminance)
 *
 * p_hat = luminance(BRDF(x1, wo, wi) * radiance * cos_theta)
 *
 * This is the "contribution function" used for RIS weights in GI.
 * Unlike DI, the "sample" is a surface point with cached radiance,
 * so we multiply BRDF * radiance instead of BRDF * emission.
 *
 * BSDF-flavored overload: takes a pre-built MaterialBSDF (typically from
 * surface.make_bsdf()) so the composed LobeList / coat / fuzz fields
 * propagate into target_pdf. Gate skips Dielectric (3) and ThinDielectric (11).
 */
[[nodiscard]] inline Float gi_evaluate_p_hat(
    const MaterialBSDF& bsdf,
    Float3 wo, Float3 wi, Float3 ns,
    Float3 radiance) noexcept
{
    Float cos_theta = max(0.0f, dot(ns, wi));
    Float3 result = def(make_float3(0.0f));
    $if(bsdf.bsdf_type != 3u & bsdf.bsdf_type != 11u) {
        Float3 brdf = bsdf.evaluate(wo, wi, ns);
        result = brdf * radiance * cos_theta;
    };
    return max(luminance(result), 0.0f);
}

//==============================================================================
// GI Geometric Jacobian — solid angle ratio (RTXDI-style)
//==============================================================================

/**
 * @brief Compute geometric Jacobian for GI reuse (solid angle ratio)
 *
 * J = (cos_new * dist_sq_old) / (cos_old * dist_sq_new)
 *
 * See Equation (11) in the ReSTIR GI paper.
 * Only depends on geometry — more stable than p_hat ratio Jacobian
 * which also includes BRDF variation.
 *
 * @param receiver_pos      Current pixel's world position
 * @param neighbor_pos      Neighbor's world position (or prev frame's position)
 * @param sample_pos        GI sample's cached position (x2)
 * @param sample_normal     GI sample's cached normal (x2's normal)
 */
[[nodiscard]] inline Float gi_geometric_jacobian(
    Float3 receiver_pos,
    Float3 neighbor_pos,
    Float3 sample_pos,
    Float3 sample_normal) noexcept
{
    // New: receiver → sample
    Float3 new_vec = receiver_pos - sample_pos;
    Float new_dist_sq = max(dot(new_vec, new_vec), 1e-10f);
    Float new_cos = max(dot(sample_normal, new_vec * rsqrt(new_dist_sq)), 0.0f);

    // Old: neighbor → sample
    Float3 old_vec = neighbor_pos - sample_pos;
    Float old_dist_sq = max(dot(old_vec, old_vec), 1e-10f);
    Float old_cos = max(dot(sample_normal, old_vec * rsqrt(old_dist_sq)), 0.0f);

    Float j = (new_cos * old_dist_sq) / max(old_cos * new_dist_sq, 1e-10f);

    // Reject invalid (NaN/inf or extreme ratio), clamp otherwise
    Bool valid = (j >= 0.1f) & (j <= 10.0f);
    return ite(valid, clamp(j, 1.0f / 3.0f, 3.0f), 0.0f);
}

} // namespace newtype::render
