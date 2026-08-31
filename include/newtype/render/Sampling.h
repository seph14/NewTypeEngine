#pragma once

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

namespace newtype::render {

using namespace luisa;
using namespace luisa::compute;

//==============================================================================
// Sampling Utilities (ported from LuisaRender)
//==============================================================================

/**
 * @brief Sample uniform disk concentrically (better mapping than polar)
 *
 * Maps unit square [0,1]^2 to unit disk with better preservation of area.
 * Uses Shirley's concentric mapping to avoid singularity at center.
 */
[[nodiscard]] Float2 sample_uniform_disk_concentric(Expr<luisa::float2> u) noexcept;

/**
 * @brief Sample cosine-weighted hemisphere
 *
 * Returns a direction in the upper hemisphere (+Z) with cosine-weighted
 * distribution. Perfect for Lambertian diffuse BSDF sampling.
 *
 * PDF = cos(theta) / pi
 */
[[nodiscard]] Float3 sample_cosine_hemisphere(Expr<luisa::float2> u) noexcept;

/**
 * @brief PDF of cosine-weighted hemisphere sampling
 *
 * Returns cos(theta) / pi for a given cos_theta value.
 */
[[nodiscard]] Float cosine_hemisphere_pdf(Expr<float> cos_theta) noexcept;

/**
 * @brief Sample uniform triangle (barycentric coordinates)
 *
 * Returns (u, v, w) barycentric coordinates where u+v+w=1.
 * Used for sampling points on triangle lights.
 */
[[nodiscard]] Float3 sample_uniform_triangle(Expr<luisa::float2> u) noexcept;

/**
 * @brief Sample uniform sphere
 *
 * Returns a direction uniformly distributed on the unit sphere.
 * PDF = 1 / (4 * pi)
 */
[[nodiscard]] Float3 sample_uniform_sphere(Expr<luisa::float2> u) noexcept;

/**
 * @brief Balance heuristic for Multiple Importance Sampling
 *
 * Standard MIS weight combining two sampling techniques.
 * w = nf * fPdf / (nf * fPdf + ng * gPdf)
 */
[[nodiscard]] Float balance_heuristic(
    Expr<uint> nf, Expr<float> fPdf,
    Expr<uint> ng, Expr<float> gPdf) noexcept;

/**
 * @brief Power heuristic for MIS (often better than balance)
 *
 * Uses squared PDFs for weight computation, reduces variance
 * when one sampling technique is much better than the other.
 *
 * w = (nf * fPdf)^2 / ((nf * fPdf)^2 + (ng * gPdf)^2)
 */
[[nodiscard]] Float power_heuristic(
    Expr<uint> nf, Expr<float> fPdf,
    Expr<uint> ng, Expr<float> gPdf) noexcept;

/**
 * @brief Balance heuristic with single sample per technique
 *
 * Convenience wrapper when nf = ng = 1.
 */
[[nodiscard]] Float balance_heuristic(Expr<float> fPdf, Expr<float> gPdf) noexcept;

/**
 * @brief Power heuristic with single sample per technique
 *
 * Convenience wrapper when nf = ng = 1.
 */
[[nodiscard]] Float power_heuristic(Expr<float> fPdf, Expr<float> gPdf) noexcept;

//==============================================================================
// Coordinate Frame Helpers (ported from LuisaRender)
//==============================================================================

/** @brief Create orthonormal basis from normal vector */
[[nodiscard]] inline Float3x3 make_orthonormal_basis(Float3 n) noexcept {
    static Callable impl = [](Float3 normal) noexcept {
        Float3 b1 = ite(abs(normal.y) < 0.999f,
            normalize(cross(normal, luisa::make_float3(0.0f, 1.0f, 0.0f))),
            luisa::make_float3(1.0f, 0.0f, 0.0f));
        Float3 b2 = cross(normal, b1);
        return make_float3x3(b1, b2, normal);
    };
    return impl(n);
}

/** @brief Create orthonormal basis from tangent + normal (for anisotropy)
 *
 * tangent_dir: world-space tangent direction (xyz of vertex tangent)
 * bitangent_sign: handedness from vertex tangent.w (-1 or +1)
 * normal: shading normal
 *
 * Returns basis (tangent, bitangent, normal) aligned to the surface tangent frame.
 */
[[nodiscard]] inline Float3x3 make_orthonormal_basis_tangent(
    Float3 tangent_dir, Float bitangent_sign, Float3 normal) noexcept {
    static Callable impl = [](Float3 tangent_dir, Float bitangent_sign, Float3 normal) noexcept {
        // Orthogonalize tangent against normal (Gram-Schmidt)
        Float3 t_raw = tangent_dir - normal * dot(normal, tangent_dir);
        Float len2 = dot(t_raw, t_raw);
        // If tangent is degenerate (parallel to normal), fall back to standard basis
        Float3 t_fallback = ite(abs(normal.y) < 0.999f,
            normalize(cross(normal, luisa::compute::make_float3(0.0f, 1.0f, 0.0f))),
            luisa::compute::make_float3(1.0f, 0.0f, 0.0f));
        Float3 t = ite(len2 > 1e-10f, normalize(t_raw), t_fallback);
        Float3 b = cross(normal, t) * bitangent_sign;
        return make_float3x3(t, b, normal);
    };
    return impl(tangent_dir, bitangent_sign, normal);
}

/** @brief Transform local direction (Z-up) to world space using normal */
[[nodiscard]] inline Float3 local_to_world(Float3 v, Float3 n) noexcept {
    Float3x3 tnb = make_orthonormal_basis(n);
    return tnb * v;
}

//==============================================================================
// Inline Math Helpers
//==============================================================================

[[nodiscard]] inline auto sqr(auto x) noexcept { return x * x; }
[[nodiscard]] inline auto one_minus_sqr(auto x) noexcept { return 1.0f - x * x; }
[[nodiscard]] inline auto abs_dot(Float3 u, Float3 v) noexcept { return abs(dsl::dot(u, v)); }
[[nodiscard]] inline auto cos_theta(Float3 w) { return w.z; }
[[nodiscard]] inline auto cos2_theta(Float3 w) { return w.z * w.z; }
[[nodiscard]] inline auto abs_cos_theta(Float3 w) { return abs(w.z); }
[[nodiscard]] inline auto sin2_theta(Float3 w) { return max(1.0f - cos2_theta(w), 0.0f); }
[[nodiscard]] inline auto sin_theta(Float3 w) { return sqrt(sin2_theta(w)); }
[[nodiscard]] inline auto tan_theta(Float3 w) { return sin_theta(w) / cos_theta(w); }
[[nodiscard]] inline auto tan2_theta(Float3 w) { return sin2_theta(w) / cos2_theta(w); }

[[nodiscard]] inline auto cos_phi(Float3 w) {
    auto sinTheta = sin_theta(w);
    return ite(sinTheta == 0.0f, 1.0f, clamp(w.x / sinTheta, -1.0f, 1.0f));
}

[[nodiscard]] inline auto sin_phi(Float3 w) {
    auto sinTheta = sin_theta(w);
    return ite(sinTheta == 0.0f, 0.0f, clamp(w.y / sinTheta, -1.0f, 1.0f));
}

[[nodiscard]] inline auto cos2_phi(Float3 w) { auto cp = cos_phi(w); return cp * cp; }
[[nodiscard]] inline auto sin2_phi(Float3 w) { auto sp = sin_phi(w); return sp * sp; }
[[nodiscard]] inline auto same_hemisphere(Float3 w, Float3 wp) noexcept { return w.z * wp.z > 0.0f; }

//==============================================================================
// Refraction (Snell's law)
//==============================================================================

/**
 * @brief Refract incident direction through surface with given IOR ratio
 *
 * @param incident  Direction pointing away from surface (toward the incoming side)
 * @param n         Surface normal (pointing toward the incoming side)
 * @param eta       IOR ratio: eta_i / eta_t (e.g., 1.0/1.5 for air→glass)
 * @return          Refracted direction. On total internal reflection, returns reflected direction.
 */
[[nodiscard]] inline Float3 refract_dir(Float3 incident, Float3 n, Float eta) noexcept {
    static Callable impl = [](Float3 incident, Float3 n, Float eta) noexcept {
        Float cos_i = dot(incident, n);
        Float sin2_t = eta * eta * (1.0f - cos_i * cos_i);
        Float cos_t = sqrt(max(1.0f - sin2_t, 0.0f));
        // Standard Snell's law: d = -incident (toward surface), n = outward normal
        //   t = eta * d + (eta * cos_i - cos_t) * n
        //     = -eta * incident + (eta * cos_i - cos_t) * n
        // Total internal reflection → reflect instead
        return ite(sin2_t < 1.0f,
            -eta * incident + (eta * cos_i - cos_t) * n,
            incident - 2.0f * cos_i * n);
    };
    return impl(incident, n, eta);
}

} // namespace newtype::render
