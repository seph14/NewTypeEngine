#include "newtype/render/Sampling.h"
#include <luisa/dsl/sugar.h>

namespace newtype::render {

using namespace luisa;
using namespace luisa::compute;

//==============================================================================
// Sampling Implementations
//==============================================================================

Float2 sample_uniform_disk_concentric(Expr<float2> u) noexcept {
    static Callable impl = [](Float2 u_in) noexcept {
        auto u = u_in * 2.0f - 1.0f;
        auto p = abs(u.x) > abs(u.y);
        auto r = ite(p, u.x, u.y);
        auto theta = ite(p, pi_over_four * (u.y / u.x), pi_over_two - pi_over_four * (u.x / u.y));
        return r * make_float2(cos(theta), sin(theta));
    };
    return impl(u);
}

Float3 sample_cosine_hemisphere(Expr<float2> u) noexcept {
    static Callable impl = [](Float2 u) noexcept {
        auto d = sample_uniform_disk_concentric(u);
        auto z = sqrt(max(1.0f - d.x * d.x - d.y * d.y, 0.0f));
        return make_float3(d.x, d.y, z);
    };
    return impl(u);
}

Float cosine_hemisphere_pdf(Expr<float> cos_theta) noexcept {
    return cos_theta * inv_pi;
}

Float3 sample_uniform_triangle(Expr<float2> u) noexcept {
    static Callable impl = [](Float2 u) noexcept {
        auto uv = ite(
            u.x < u.y,
            make_float2(0.5f * u.x, -0.5f * u.x + u.y),
            make_float2(-0.5f * u.y + u.x, 0.5f * u.y));
        return make_float3(uv, 1.0f - uv.x - uv.y);
    };
    return impl(u);
}

Float3 sample_uniform_sphere(Expr<float2> u) noexcept {
    static Callable impl = [](Float2 u) noexcept {
        auto z = 1.0f - 2.0f * u.x;
        auto r = sqrt(max(1.0f - z * z, 0.0f));
        auto phi = 2.0f * pi * u.y;
        return make_float3(r * cos(phi), r * sin(phi), z);
    };
    return impl(u);
}

Float balance_heuristic(
    Expr<uint> nf, Expr<float> fPdf,
    Expr<uint> ng, Expr<float> gPdf) noexcept {
    static Callable impl = [](UInt nf, Float fPdf, UInt ng, Float gPdf) noexcept {
        auto sum_f = nf * fPdf;
        auto sum = sum_f + ng * gPdf;
        return ite(sum == 0.0f, 0.0f, sum_f / sum);
    };
    return impl(nf, fPdf, ng, gPdf);
}

Float power_heuristic(
    Expr<uint> nf, Expr<float> fPdf,
    Expr<uint> ng, Expr<float> gPdf) noexcept {
    static Callable impl = [](UInt nf, Float fPdf, UInt ng, Float gPdf) noexcept {
        Float f = nf * fPdf, g = ng * gPdf;
        auto ff = f * f;
        auto gg = g * g;
        auto sum = ff + gg;
        return ite(abs(ff) > 1e30f, 1.f, ite(sum == 0.f, 0.f, ff / sum));
    };
    return impl(nf, fPdf, ng, gPdf);
}

Float balance_heuristic(Expr<float> fPdf, Expr<float> gPdf) noexcept {
    return balance_heuristic(1u, fPdf, 1u, gPdf);
}

Float power_heuristic(Expr<float> fPdf, Expr<float> gPdf) noexcept {
    return power_heuristic(1u, fPdf, 1u, gPdf);
}

} // namespace newtype::render
