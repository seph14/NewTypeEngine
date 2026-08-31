#include "newtype/render/BSSRDF.h"

namespace newtype::render {

static constexpr float kLog2E = 1.44269504089f;
static constexpr float kEps   = 1e-6f;

Float3 bssrdf_scatter_distance(Expr<float3> albedo) noexcept {
    // Per-channel Burley fit: s = 1.9 - albedo + 3.5 * (albedo - 0.8)^2.
    // Scalar broadcast on `albedo - 0.8f` is unambiguous in DSL (Cinder vec3 / Luisa float3 clash
    // only fires for make_float3(0.8f) literal construction).
    Float3 a = albedo - 0.8f;
    return 1.9f - albedo + 3.5f * a * a;
}

Float bssrdf_profile_eval(Expr<float> r, Expr<float> s) noexcept {
    Float ss = max(s, kEps);
    Float rr = max(r, kEps);
    Float rd = rr / ss;
    return (exp(-rd) + exp(-rd * (1.0f / 3.0f)))
         / (8.0f * pi * ss * rr);
}

Float bssrdf_sample_radius(Expr<float> xi, Expr<float> s) noexcept {
    Float u = 1.0f - xi;                                  // CCDF: P(r > R) = u
    Float g = 1.0f + (4.0f * u) * (2.0f * u + sqrt(1.0f + (4.0f * u) * u));
    Float n = exp2(log2(g) * (-1.0f / 3.0f));             // g^(-1/3)
    Float p = (g * n) * n;                                // g^(+1/3) = (g * g^(-1/3)) * g^(-1/3)
    Float c = 1.0f + p + n;
    Float x = (3.0f / kLog2E) * log2(c / (4.0f * u));
    return x * s;                                         // Bug 1 fix: multiply, NOT divide
}

} // namespace newtype::render
