#include "newtype/render/BSDF.h"
#include "newtype/render/Sampling.h"
#include "newtype/core/Config.h"
#include <luisa/dsl/sugar.h>

namespace newtype::render {

using namespace luisa;
using namespace luisa::compute;

//==============================================================================
// Fresnel Terms
//==============================================================================

Float fresnel_schlick(Expr<float> R0, Expr<float> cos_theta) noexcept {
    static Callable impl = [](Float R0, Float cos_theta) noexcept {
        auto pow5 = [](auto&& v) { return sqr(sqr(v)) * v; };
        return R0 + (1.0f - R0) * pow5(max(1.0f - cos_theta, 0.0f));
    };
    return impl(R0, cos_theta);
}

Float fresnel_dielectric(Expr<float> cos_theta_i, Expr<float> eta_i, Expr<float> eta_t) noexcept {
    static Callable impl = [](Float cos_theta_i, Float eta_i, Float eta_t) noexcept {
        auto cos_i = clamp(cos_theta_i, -1.0f, 1.0f);
        auto entering = cos_i > 0.0f;
        auto eta_i_actual = ite(entering, eta_i, eta_t);
        auto eta_t_actual = ite(entering, eta_t, eta_i);
        cos_i = abs(cos_i);

        auto sin_i = sqrt(max(0.0f, 1.0f - sqr(cos_i)));
        auto sin_t = eta_i_actual / eta_t_actual * sin_i;
        auto cos_t = sqrt(max(0.0f, 1.0f - sqr(sin_t)));

        auto R_parl = (eta_t_actual * cos_i - eta_i_actual * cos_t) /
                      (eta_t_actual * cos_i + eta_i_actual * cos_t);
        auto R_perp = (eta_i_actual * cos_i - eta_t_actual * cos_t) /
                      (eta_i_actual * cos_i + eta_t_actual * cos_t);

        return ite(sin_t < 1.0f, (R_parl * R_parl + R_perp * R_perp) * 0.5f, 1.0f);
    };
    return impl(cos_theta_i, eta_i, eta_t);
}

//==============================================================================
// GGX Microfacet Distribution
//==============================================================================

Float3 eval_thin_film_iridescence(
    Expr<float> cos_theta_d,
    Expr<float> thin_film_ior,
    Expr<float> thickness) noexcept {
    static Callable impl = [](Float cos_theta_d, Float thin_film_ior, Float thickness) noexcept {
        auto pow5 = [](auto&& v) { return sqr(sqr(v)) * v; };

        // Snell's law: transmitted angle inside film
        Float sin2_t = (1.0f - cos_theta_d * cos_theta_d) /
                       (thin_film_ior * thin_film_ior);
        Float cos_theta_t = sqrt(max(1.0f - sin2_t, 0.0f));

        // Schlick Fresnel at air->film (R12) and film->substrate (R23)
        Float R0 = sqr((thin_film_ior - 1.0f) / (thin_film_ior + 1.0f));
        Float R12 = R0 + (1.0f - R0) * pow5(max(1.0f - cos_theta_d, 0.0f));
        Float R23 = R0 + (1.0f - R0) * pow5(max(1.0f - cos_theta_t, 0.0f));

        // Optical path difference (nm): 2 * n_film * d * cos(theta_t)
        Float opd = 2.0f * thin_film_ior * thickness * cos_theta_t;

        // Phase for 3 wavelengths: R=630nm, G=530nm, B=460nm
        Float3 phase = 2.0f * 3.14159265359f * opd /
                       make_float3(630.0f, 530.0f, 460.0f);

        // Airy reflectance:
        // R = (R12 + R23 + 2*sqrt(R12*R23)*cos(phase))
        //   / (1 + R12*R23 + 2*sqrt(R12*R23)*cos(phase))
        Float sqrt_R12_R23 = sqrt(max(R12 * R23, 1e-10f));
        Float3 cos_phase = cos(phase);
        Float3 numer = make_float3(R12 + R23) + 2.0f * sqrt_R12_R23 * cos_phase;
        Float3 denom = make_float3(1.0f + R12 * R23) + 2.0f * sqrt_R12_R23 * cos_phase;

        return numer / max(denom, make_float3(1e-6f));
    };
    return impl(cos_theta_d, thin_film_ior, thickness);
}

Float3 fresnel_conductor(
    Expr<float> c_i, Expr<float3> eta_in, Expr<float3> k_in) noexcept {
    static Callable impl = [](Float c_i, Float3 eta, Float3 k) noexcept {
        Float  cos_i  = clamp(c_i, 0.0f, 1.0f);
        Float  sin2_i = max(1.0f - sqr(cos_i), 0.0f);

        // (eta + i*k)^2 = (eta^2 - k^2) + i*2*eta*k
        Float3 sq_re   = sqr(eta) - sqr(k);
        Float3 sq_im   = 2.0f * eta * k;
        Float3 sq_mag2 = sqr(sq_re) + sqr(sq_im);

        // sin2_t = sin2_i / (eta + i*k)^2  — complex division
        Float3 sin2_t_re = sin2_i * sq_re / sq_mag2;
        Float3 sin2_t_im = -sin2_i * sq_im / sq_mag2;

        // cos_t = sqrt(1 - sin2_t)  — complex sqrt via magnitude/angle
        Float3 arg_re  = 1.0f - sin2_t_re;
        Float3 arg_im  = -sin2_t_im;
        Float3 arg_mag = sqrt(sqr(arg_re) + sqr(arg_im));
        Float3 ct_re   = sqrt(max(0.5f * (arg_mag + arg_re), 0.0f));
        Float3 ct_im   = sign(arg_im) * sqrt(max(0.5f * (arg_mag - arg_re), 0.0f));

        // r_parl = (eta*cos_i - cos_t) / (eta*cos_i + cos_t)
        // (eta + i*k) * cos_i  -> Re = eta*cos_i, Im = k*cos_i  (complex!)
        Float3 ec_re = eta * cos_i;
        Float3 ec_im = k     * cos_i;
        Float3 np_re = ec_re - ct_re;
        Float3 np_im = ec_im - ct_im;
        Float3 dp_re = ec_re + ct_re;
        Float3 dp_im = ec_im + ct_im;
        Float3 dp_m2 = sqr(dp_re) + sqr(dp_im);
        Float3 rp_re = (np_re * dp_re + np_im * dp_im) / dp_m2;
        Float3 rp_im = (np_im * dp_re - np_re * dp_im) / dp_m2;

        // r_perp = (cos_i - eta*cos_t) / (cos_i + eta*cos_t)
        // (eta + i*k) * (ct_re + i*ct_im)  -> Re = eta*ct_re - k*ct_im, Im = eta*ct_im + k*ct_re
        Float3 kc_re = eta * ct_re - k * ct_im;
        Float3 kc_im = eta * ct_im + k * ct_re;
        Float3 nq_re = cos_i - kc_re;
        Float3 nq_im = -kc_im;
        Float3 dq_re = cos_i + kc_re;
        Float3 dq_im = kc_im;
        Float3 dq_m2 = sqr(dq_re) + sqr(dq_im);
        Float3 rq_re = (nq_re * dq_re + nq_im * dq_im) / dq_m2;
        Float3 rq_im = (nq_im * dq_re - nq_re * dq_im) / dq_m2;

        // F = (|r_par|^2 + |r_perp|^2) / 2
        return 0.5f * (sqr(rp_re) + sqr(rp_im) + sqr(rq_re) + sqr(rq_im));
    };
    return impl(c_i, eta_in, k_in);
}

Float ggx_distribution(Expr<float3> wh, Expr<float2> alpha) noexcept {
    static Callable impl = [](Float3 wh, Float2 alpha) noexcept {
        auto cos4_theta = sqr(sqr(wh.z));
        auto tan2_theta = (sqr(wh.x) + sqr(wh.y)) / sqr(wh.z);

        auto e = tan2_theta * (sqr(cos_phi(wh) / alpha.x) +
                              sqr(sin_phi(wh) / alpha.y));
        auto d = 1.0f / (pi * alpha.x * alpha.y * cos4_theta * sqr(1.0f + e));

        return ite(abs(tan2_theta) > 1e30f, 0.0f, d);
    };
    return impl(wh, alpha);
}

Float ggx_lambda(Expr<float3> w, Expr<float2> alpha) noexcept {
    static Callable impl = [](Float3 w, Float2 alpha) noexcept {
        auto tan_theta_val = abs(tan_theta(w));
        auto alpha2 = cos2_phi(w) * sqr(alpha.x) + sin2_phi(w) * sqr(alpha.y);
        auto alpha2_tan2 = alpha2 * sqr(tan_theta_val);
        return (-1.0f + sqrt(1.0f + alpha2_tan2)) * 0.5f;
    };
    return impl(w, alpha);
}

Float ggx_G1(Expr<float3> w, Expr<float2> alpha) noexcept {
    return 1.0f / (1.0f + ggx_lambda(w, alpha));
}

Float ggx_G(Expr<float3> wo, Expr<float3> wi, Expr<float2> alpha) noexcept {
    return 1.0f / (1.0f + ggx_lambda(wo, alpha) + ggx_lambda(wi, alpha));
}

Float3 sample_ggx_wh(Expr<float3> wo, Expr<float2> alpha, Expr<float2> u) noexcept {
    static Callable impl = [](Float3 wo, Float2 alpha, Float2 u) noexcept {
        // Transform wo to local "stretched" space
        auto wo_stretched = normalize(make_float3(
            alpha.x * wo.x, alpha.y * wo.y, wo.z));

        // Sample P22_{wi}(slope_x, slope_y)
        auto cos_theta = wo_stretched.z;
        auto slope = def(make_float2());

        $if(cos_theta > (1.0f - 1e-6f)) {
            // Special case: normal incidence
            auto r = sqrt(u.x / (1.0f - u.x));
            auto phi = 2.0f * pi * u.y;
            slope = r * make_float2(cos(phi), sin(phi));
        }
        $else {
            auto sin_theta = sqrt(max(0.0f, 1.0f - sqr(cos_theta)));
            auto tan_theta = sin_theta / cos_theta;
            auto a = 1.0f / tan_theta;
            auto G1 = 2.0f / (1.0f + sqrt(1.0f + 1.0f / sqr(a)));

            // Sample slope_x
            auto A = 2.0f * u.x / G1 - 1.0f;
            auto tmp = min(1.0f / (sqr(A) - 1.0f), 1e10f);
            auto B = tan_theta;
            auto D = sqrt(max(sqr(B * tmp) - (sqr(A) - sqr(B)) * tmp, 0.0f));
            auto slope_x_1 = B * tmp - D;
            auto slope_x_2 = B * tmp + D;
            auto slope_x = ite(
                (A < 0.0f) | (slope_x_2 * tan_theta > 1.0f),
                slope_x_1, slope_x_2);

            // Sample slope_y
            auto S = ite(u.y > 0.5f, 1.0f, -1.0f);
            auto U2 = ite(u.y > 0.5f, 2.0f * (u.y - 0.5f), 2.0f * (0.5f - u.y));
            auto z = (U2 * (U2 * (U2 * 0.27385f - 0.73369f) + 0.46341f)) /
                     (U2 * (U2 * (U2 * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
            auto slope_y = S * z * sqrt(1.0f + sqr(slope_x));
            slope = make_float2(slope_x, slope_y);
        };

        // Rotate and unstretch
        auto phi_s = atan2(wo_stretched.y, wo_stretched.x);
        slope = make_float2(
            cos(phi_s) * slope.x - sin(phi_s) * slope.y,
            sin(phi_s) * slope.x + cos(phi_s) * slope.y);
        slope = alpha * slope;

        // Compute normal
        return normalize(make_float3(-slope, 1.0f));
    };
    return impl(wo, alpha, u);
}

Float ggx_pdf(Expr<float3> wo, Expr<float3> wh, Expr<float2> alpha) noexcept {
    return ggx_distribution(wh, alpha) * ggx_G1(wo, alpha) *
           abs_dot(wo, wh) / abs_cos_theta(wo);
}

//==============================================================================
// Kulla-Conty Multiple-Scattering Energy Compensation (NT_ENABLE_MS_GGX)
//==============================================================================
#if NT_ENABLE_MS_GGX

namespace {

// Local-struct Float arrays follow the LobeList pattern (Lobe.h) — host-side
// C-style arrays of Var<T> misbehave, plain Float members with compile-time
// indices are the proven layout.
struct MSFitBasis {
    static constexpr uint kN = 7u;
    Float t[kN];
};

// Chebyshev tensor fit of G(mu, r) = 1 - E, deg 6x6 over
// u = 2*mu - 1 and v = 2*(r - 0.03)/0.97 - 1, r = sqrt(alpha).
// Generated by tools/ms_ggx_fit.py; max abs err 0.022, mean 0.004.
// (8x8 available in harness history at max err 0.017 — kept at 6x6 to bound
// live temporaries in the reuse-loop kernels; fps regressed ~15% at 8x8.)
constexpr float kMSChebCoeffs[7][7] = {
    { +0.1822246f, +0.2372415f, +0.0595624f, -0.0026065f, -0.0101111f, +0.0014082f, +0.0019110f },
    { +0.0874631f, +0.1648525f, +0.0658129f, -0.0170852f, -0.0050772f, -0.0037959f, +0.0024035f },
    { -0.0378095f, -0.0593319f, -0.0009097f, +0.0121639f, -0.0135601f, +0.0037665f, +0.0037637f },
    { +0.0163236f, +0.0216826f, -0.0063927f, +0.0008143f, +0.0100183f, -0.0093410f, +0.0000332f },
    { -0.0073146f, -0.0081628f, +0.0049467f, -0.0035710f, -0.0040303f, +0.0076933f, -0.0041857f },
    { +0.0025082f, +0.0023050f, -0.0025235f, +0.0021471f, +0.0012351f, -0.0039253f, +0.0031632f },
    { -0.0012487f, -0.0003251f, +0.0031632f, -0.0024450f, +0.0005353f, +0.0017418f, -0.0069061f },
};

} // namespace

Float ggx_directional_albedo_fit(Expr<float> mu_in, Expr<float> alpha_in) noexcept {
    constexpr float kRMin = 0.03f;
    constexpr float kRMax = 1.0f;

    Float r = sqrt(clamp(alpha_in, kRMin * kRMin, kRMax * kRMax));
    Float u = 2.0f * clamp(mu_in, 0.0f, 1.0f) - 1.0f;
    Float v = 2.0f * (r - kRMin) / (kRMax - kRMin) - 1.0f;

    MSFitBasis bu, bv;
    bu.t[0] = 1.0f;
    bu.t[1] = u;
    bv.t[0] = 1.0f;
    bv.t[1] = v;
    for (uint k = 2u; k < MSFitBasis::kN; ++k) {
        bu.t[k] = 2.0f * u * bu.t[k - 1u] - bu.t[k - 2u];
        bv.t[k] = 2.0f * v * bv.t[k - 1u] - bv.t[k - 2u];
    }

    Float g = def(0.0f);
    for (uint i = 0u; i < MSFitBasis::kN; ++i) {
        for (uint j = 0u; j < MSFitBasis::kN; ++j) {
            g += kMSChebCoeffs[i][j] * bu.t[i] * bv.t[j];
        }
    }
    return clamp(1.0f - g, 0.0f, 1.0f);
}

Float ggx_avg_albedo_fit(Expr<float> alpha_in) noexcept {
    // Deg-5 polynomial in alpha, tools/ms_ggx_fit.py [3]. Max abs err 8e-4.
    Float a = clamp(alpha_in, 0.0f, 1.0f);
    Float ev = 1.13749187f * a * a * a * a * a
             - 3.46793205f * a * a * a * a
             + 4.28363453f * a * a * a
             - 2.47468621f * a * a
             - 0.07002561f * a
             + 0.99994954f;
    return clamp(ev, 0.0f, 1.0f);
}

Float3 schlick_f_avg(Expr<float3> F0) noexcept {
    return F0 + (1.0f - F0) * (1.0f / 21.0f);
}

#endif // NT_ENABLE_MS_GGX

//==============================================================================
// Lambertian BSDF
//==============================================================================

Float3 LambertianBSDF::evaluate(
    Expr<float3> wo,
    Expr<float3> wi,
    Expr<float3> normal) const noexcept {

    // Local z-component of an orthonormal-basis transform is dot(v, normal),
    // so the hemisphere test needs no basis at all.
    return ite(dot(wo, normal) * dot(wi, normal) > 0.0f,
        albedo * inv_pi, make_float3(0.0f));
}

Float3 LambertianBSDF::sample(
    Expr<float3> wo,
    Expr<float3> normal,
    Expr<float2> u,
    Float& out_pdf) const noexcept {

    // Sample cosine-weighted hemisphere in local space
    Float3 wi_local = sample_cosine_hemisphere(u);

    // Make sure we sample in the same hemisphere as wo (relative to normal, not world z)
    wi_local.z *= sign(dot(wo, normal));

    // Transform to world space
    Float3x3 tnb = make_orthonormal_basis(normal);
    Float3 wi = tnb * wi_local;

    out_pdf = pdf(wo, wi, normal);
    return wi;
}

Float LambertianBSDF::pdf(
    Expr<float3> wo,
    Expr<float3> wi,
    Expr<float3> normal) const noexcept {

    return ite(dot(wo, normal) * dot(wi, normal) > 0.0f,
        abs(dot(wi, normal)) * inv_pi, 0.0f);
}

//==============================================================================
// Microfacet BSDF
//==============================================================================

Float3 MicrofacetBSDF::evaluate(
    Expr<float3> wo,
    Expr<float3> wi,
    Expr<float3> normal) const noexcept {

    Float3 result = def(make_float3(0.0f));

    // Transform to local space using tangent-aware basis
    Float3x3 tnb = make_orthonormal_basis_tangent(tangent_dir, bitangent_sign, normal);
    Float3 wo_local = transpose(tnb) * wo;
    Float3 wi_local = transpose(tnb) * wi;

    $if(same_hemisphere(wo_local, wi_local)) {
        Float3 wh = normalize(wo_local + wi_local);

        // Fresnel term — metallic workflow:
        //   Conductor (toggle ON + k!=0): FrComplex with (eta, k) per channel
        //   Conductor (legacy or toggle OFF): F0 = albedo, Schlick approximation
        //   Dielectric: F0 from IOR (scalar, monochromatic)
        Float3 F = def(make_float3(0.0f));
        $if(metallic > 0.5f) {
            // k != 0 is the per-material sentinel for complex-IOR mode. Legacy
            // conductors (k=0) fall through to Schlick; the FrComplex math would
            // divide by zero in sq_mag2 if entered with eta=k=0.
            Bool use_complex = any(conductor_k != 0.0f);
            $if(use_complex) {
                // Complex-IOR Fresnel — attenuation holds eta_re (overloaded storage)
                Float cos_wh = abs_dot(wi_local, wh);
                F = fresnel_conductor(cos_wh, conductor_eta, conductor_k);
            }
            $else {
                // Legacy Schlick — albedo is F0, F = albedo + (1-albedo)*(1-cos)^5
                auto pow5 = [](auto&& v) { return sqr(sqr(v)) * v; };
                Float3 one_minus_cos5 = make_float3(pow5(max(1.0f - abs_dot(wi_local, wh), 0.0f)));
                F = albedo + (1.0f - albedo) * one_minus_cos5;
            };
        }
        $else {
            // Dielectric: exact Fresnel (scalar, applied to all channels)
            F = make_float3(fresnel_dielectric(abs_dot(wi_local, wh), 1.0f, ior));
        };

        // Iridescence: replace base Fresnel with thin-film interference
        $if(iridescence > 0.0f & iridescence_thickness > 0.0f) {
            Float cos_theta_d = abs_dot(wi_local, wh);
            Float3 F_thin = eval_thin_film_iridescence(
                cos_theta_d, iridescence_ior, iridescence_thickness);
            F = F * (1.0f - iridescence) + F_thin * iridescence;
        };

        Float D = ggx_distribution(wh, alpha);
        Float G = ggx_G(wo_local, wi_local, alpha);

        // For metals: albedo is already baked into F (as F0), so no separate multiply.
        // For dielectrics: F is scalar white — no albedo tint on specular reflection.
        // Kulla-Conty f_ms is NOT added here — it lives in MaterialBSDF's spec
        // blocks using the resolve_surface-hoisted invariants (docs/ms_ggx_compensation.md).
        result = F * D * G / (4.0f * abs_cos_theta(wo_local) * abs_cos_theta(wi_local));
    };

    return result;
}

Float3 MicrofacetBSDF::sample(
    Expr<float3> wo,
    Expr<float3> normal,
    Expr<float2> u,
    Float& out_pdf) const noexcept {

    // Transform to local space using tangent-aware basis
    Float3x3 tnb = make_orthonormal_basis_tangent(tangent_dir, bitangent_sign, normal);
    Float3 wo_local = transpose(tnb) * wo;

    // Sample microfacet normal
    Float3 wh_local = sample_ggx_wh(wo_local, alpha, u);
    Float3 wi_local = reflect(-wo_local, wh_local);

    // Transform to world space
    Float3 wi = tnb * wi_local;

    out_pdf = pdf(wo, wi, normal);
    return wi;
}

Float MicrofacetBSDF::pdf(
    Expr<float3> wo,
    Expr<float3> wi,
    Expr<float3> normal) const noexcept {

    // Transform to local space using tangent-aware basis
    Float3x3 tnb = make_orthonormal_basis_tangent(tangent_dir, bitangent_sign, normal);
    Float3 wo_local = transpose(tnb) * wo;
    Float3 wi_local = transpose(tnb) * wi;

    Float p = def(0.0f);

    $if(same_hemisphere(wo_local, wi_local)) {
        Float3 wh = normalize(wo_local + wi_local);
        p = ggx_pdf(wo_local, wh, alpha) / (4.0f * dot(wo_local, wh));
    };

    return p;
}

//==============================================================================
// Sheen BSDF (Charlie distribution, KHR_materials_sheen)
//==============================================================================

// Smith lambda for the Charlie distribution — rational fit from Conty Estevez
// & Kulla, "Production Friendly Microfacet Sheen BRDF" (SIGGRAPH 2017), as
// codified in the KHR_materials_sheen reference implementation. Valid for
// |cos| < 0.5; mirrored beyond via the extrapolation exp(2*l(0.5) - l(1-x)).
static Float charlie_lambda(Expr<float> cos_theta, Expr<float> alpha) noexcept {
    static Callable impl = [](Float cos_theta, Float alpha) noexcept {
        Float a_cos = abs(cos_theta);
        Float t = (1.0f - alpha) * (1.0f - alpha);
        Float a = lerp(21.5473f, 25.3245f, t);
        Float b = lerp(3.82987f, 3.32435f, t);
        Float c = lerp(0.19823f, 0.16801f, t);
        Float d = lerp(-1.97760f, -1.27393f, t);
        Float e = lerp(-4.32054f, -4.85967f, t);
        auto l = [&](Float x) noexcept { return a / (1.0f + b * pow(x, c)) + d * x + e; };
        return ite(a_cos < 0.5f,
                   exp(l(a_cos)),
                   exp(2.0f * l(0.5f) - l(1.0f - a_cos)));
    };
    return impl(cos_theta, alpha);
}

Float3 SheenBSDF::evaluate(
    Expr<float3> wo,
    Expr<float3> wi,
    Expr<float3> normal) const noexcept {

    Float3 result = def(make_float3(0.0f));

    // wh / NoH and the NoV/NoL cosines are frame-invariant — evaluate
    // entirely in world space.
    $if(dot(wo, normal) * dot(wi, normal) > 0.0f) {
        Float3 wh = normalize(wo + wi);
        Float NoH = dot(normal, wh);
        Float NoV = abs(dot(wo, normal));
        Float NoL = abs(dot(wi, normal));

        Float alpha = max(roughness * roughness, 1e-3f);
        Float inv_alpha = 1.0f / alpha;

        // Charlie distribution: (2 + inv_alpha)/(2*pi) * sin(theta_h)^inv_alpha
        Float sin2_h = max(1.0f - NoH * NoH, 0.0f);
        Float D = (2.0f + inv_alpha) * pow(sin2_h, inv_alpha * 0.5f) / (2.0f * pi);

        // Smith shadowing with the rational-fit lambda
        Float G = 1.0f / (1.0f + charlie_lambda(NoV, alpha) + charlie_lambda(NoL, alpha));

        // mix(white, albedo, sheen_tint)
        Float3 sheen_color = make_float3(1.0f) * (1.0f - sheen_tint) + albedo * sheen_tint;
        result = sheen_strength * D * G / max(4.0f * NoV * NoL, 1e-6f) * sheen_color;
    };

    return result;
}

Float3 SheenBSDF::sample(
    Expr<float3> wo,
    Expr<float3> normal,
    Expr<float2> u,
    Float& out_pdf) const noexcept {

    // Cosine-weighted hemisphere — the Charlie lobe is broad, so cosine
    // sampling is a close (and cheapest) match. Mirrors LambertianBSDF::sample.
    Float3 wi_local = sample_cosine_hemisphere(u);
    wi_local.z *= sign(dot(wo, normal));

    Float3x3 tnb = make_orthonormal_basis(normal);
    Float3 wi = tnb * wi_local;

    out_pdf = pdf(wo, wi, normal);
    return wi;
}

Float SheenBSDF::pdf(
    Expr<float3> wo,
    Expr<float3> wi,
    Expr<float3> normal) const noexcept {

    return ite(dot(wo, normal) * dot(wi, normal) > 0.0f,
        abs(dot(wi, normal)) * inv_pi, 0.0f);
}

//==============================================================================
// Clearcoat BSDF (Disney Principled)
//==============================================================================

Float3 ClearcoatBSDF::evaluate(
    Expr<float3> wo,
    Expr<float3> wi,
    Expr<float3> normal) const noexcept {

    Float3 result = def(make_float3(0.0f));

    Float3x3 tnb = make_orthonormal_basis(normal);
    Float3 wo_local = transpose(tnb) * wo;
    Float3 wi_local = transpose(tnb) * wi;

    $if(same_hemisphere(wo_local, wi_local)) {
        Float3 wh = normalize(wo_local + wi_local);
        Float cos_theta_d = abs(dot(wi_local, wh));

        // Roughness from gloss: gloss=0 → roughness=0.1, gloss=1 → roughness=0.001
        Float roughness_cc = 0.1f - 0.099f * clearcoat_gloss;
        // kMinRoughness floor via the shared path (BSDF.h): the old 1e-4 alpha
        // floor let gloss→1 collapse the coat to a near-delta (fireflies on rare
        // in-lobe samples, dropout elsewhere), the same failure kMinRoughness
        // was introduced for on the main specular.
        Float2 alpha_cc = roughness_to_alpha(make_float2(roughness_cc));

        // Schlick Fresnel with fixed IOR=1.5 → R0 = 0.04
        constexpr float R0 = 0.04f;
        auto pow5 = [](auto&& v) { return sqr(sqr(v)) * v; };
        Float F_cc = R0 + (1.0f - R0) * pow5(1.0f - cos_theta_d);

        Float D_cc = ggx_distribution(wh, alpha_cc);
        Float G_cc = ggx_G(wo_local, wi_local, alpha_cc);

        result = make_float3(clearcoat_strength * F_cc * D_cc * G_cc /
                   (4.0f * abs_cos_theta(wo_local) * abs_cos_theta(wi_local)));
    };

    return result;
}

Float3 ClearcoatBSDF::sample(
    Expr<float3> wo,
    Expr<float3> normal,
    Expr<float2> u,
    Float& out_pdf) const noexcept {

    Float3x3 tnb = make_orthonormal_basis(normal);
    Float3 wo_local = transpose(tnb) * wo;

    Float roughness_cc = 0.1f - 0.099f * clearcoat_gloss;
    // Same kMinRoughness floor as evaluate() — keep the three call shapes identical.
    Float2 alpha_cc = roughness_to_alpha(make_float2(roughness_cc));

    Float3 wh_local = sample_ggx_wh(wo_local, alpha_cc, u);
    Float3 wi_local = reflect(-wo_local, wh_local);
    Float3 wi = tnb * wi_local;

    out_pdf = pdf(wo, wi, normal);
    return wi;
}

Float ClearcoatBSDF::pdf(
    Expr<float3> wo,
    Expr<float3> wi,
    Expr<float3> normal) const noexcept {

    Float3x3 tnb = make_orthonormal_basis(normal);
    Float3 wo_local = transpose(tnb) * wo;
    Float3 wi_local = transpose(tnb) * wi;

    Float p = def(0.0f);

    $if(same_hemisphere(wo_local, wi_local)) {
        Float roughness_cc = 0.1f - 0.099f * clearcoat_gloss;
        // Same kMinRoughness floor as evaluate() — keep the three call shapes identical.
        Float2 alpha_cc = roughness_to_alpha(make_float2(roughness_cc));
        Float3 wh = normalize(wo_local + wi_local);
        p = ggx_pdf(wo_local, wh, alpha_cc) / (4.0f * dot(wo_local, wh));
    };

    return p;
}

//==============================================================================
// Subsurface BSDF (Hanrahan-Krueger + Thin Transmission)
//==============================================================================

Float3 SubsurfaceBSDF::evaluate(
    Expr<float3> wo,
    Expr<float3> wi,
    Expr<float3> normal) const noexcept {

    Float3 result = def(make_float3(0.0f));

    Float cos_i = abs(dot(wi, normal));
    Float cos_o = abs(dot(wo, normal));
    // Mutual-grazing guard: cos_i + cos_o -> 0 makes f blow up to inf/NaN.
    Float cos_sum = max(cos_i + cos_o, 1e-4f);

    // Same branch probability as sample()/pdf(): transmission is picked with
    // p_trans, so the reflection lobe only carries the remaining (1 - p_trans)
    // energy. Without this, thin materials (paper) reflect ~full albedo AND
    // transmit diffuse_trans on top — total energy exceeds the albedo budget.
    Float F = fresnel_dielectric(cos_o, 1.0f, ior);
    Float p_trans = diffuse_trans * (1.0f - F);

    $if(dot(wo, normal) * dot(wi, normal) > 0.0f) {
        // Hanrahan-Krueger reflection: f = (1 - p_trans) * albedo / (pi * (cos_i + cos_o))
        result = (1.0f - p_trans) * albedo / (pi * cos_sum);
    }
    $else {
        // Diffuse transmission: (1-F) * transmission_color * diffuse_trans / (pi * (cos_i + cos_o))
        $if(diffuse_trans > 0.0f) {
            result = (1.0f - F) * transmission_color * diffuse_trans / (pi * cos_sum);
        };
    };

    return result;
}

Float3 SubsurfaceBSDF::sample(
    Expr<float3> wo,
    Expr<float3> normal,
    Expr<float2> u,
    Float& out_pdf) const noexcept {

    Float3x3 tnb = make_orthonormal_basis(normal);
    Float3 wo_local = transpose(tnb) * wo;
    Float cos_o = abs_cos_theta(wo_local);

    Float F = fresnel_dielectric(cos_o, 1.0f, ior);
    Float p_trans = diffuse_trans * (1.0f - F);

    Float3 wi_local = def(make_float3(0.0f));

    $if(u.x < p_trans) {
        // Transmission: cosine-weighted sample on opposite hemisphere
        Float2 u_trans = make_float2(u.x / max(p_trans, 1e-6f), u.y);
        Float3 wi_trans = sample_cosine_hemisphere(u_trans);
        wi_trans.z = -abs(wi_trans.z) * sign(cos_theta(wo_local));
        wi_local = wi_trans;
        out_pdf = p_trans * abs_cos_theta(wi_local) * inv_pi;
    }
    $else {
        // HK reflection: cosine-weighted hemisphere sampling. The sampled
        // integrand f * cos_i ~ cos_i / (cos_i + cos_o) is cosine-shaped, and
        // the estimator weight f * cos / pdf is then bounded by albedo.
        // The old uniform pdf fed a constant MIS pdf where the integrand
        // decays — grazing fireflies on the unclamped DI path.
        Float2 u_refl = make_float2((u.x - p_trans) / max(1.0f - p_trans, 1e-6f), u.y);
        Float3 wi_refl = sample_cosine_hemisphere(u_refl);
        wi_refl.z *= sign(cos_theta(wo_local));
        wi_local = wi_refl;
        out_pdf = (1.0f - p_trans) * abs_cos_theta(wi_local) * inv_pi;
    };

    return tnb * wi_local;
}

Float SubsurfaceBSDF::pdf(
    Expr<float3> wo,
    Expr<float3> wi,
    Expr<float3> normal) const noexcept {

    Float cos_o = abs(dot(wo, normal));
    Float F = fresnel_dielectric(cos_o, 1.0f, ior);
    Float p_trans = diffuse_trans * (1.0f - F);

    // Reflection branch: cosine-weighted pdf (matches sample()).
    Float cos_i = abs(dot(wi, normal));
    return ite(dot(wo, normal) * dot(wi, normal) > 0.0f,
        (1.0f - p_trans), p_trans) * cos_i * inv_pi;
}

//==============================================================================
// Fabric Diffuse BSDF (Ashikhmin-Premoze / Disney Fabric)
//==============================================================================

Float3 FabricDiffuseBSDF::evaluate(
    Expr<float3> wo,
    Expr<float3> wi,
    Expr<float3> normal) const noexcept {

    Float3 result = def(make_float3(0.0f));

    $if(dot(wo, normal) * dot(wi, normal) > 0.0f) {
        auto pow5 = [](auto&& v) { return sqr(sqr(v)) * v; };
        Float cos_o = abs(dot(wo, normal));
        Float cos_i = abs(dot(wi, normal));

        // Ashikhmin-Premoze: grazing-angle enhancement
        // f(wo, wi) = (21 / (20*pi)) * albedo * (1 - (1-cos_o)^5) * (1 - (1-cos_i)^5)
        Float f_wo = 1.0f - pow5(1.0f - cos_o);
        Float f_wi = 1.0f - pow5(1.0f - cos_i);
        result = albedo * (21.0f / (20.0f * pi)) * f_wo * f_wi;
    };

    return result;
}

Float3 FabricDiffuseBSDF::sample(
    Expr<float3> wo,
    Expr<float3> normal,
    Expr<float2> u,
    Float& out_pdf) const noexcept {

    // Cosine-weighted hemisphere sampling (same as Lambertian)
    Float r = sqrt(u.x);
    Float phi = 2.0f * pi * u.y;
    Float3 wi_local = make_float3(r * cos(phi), r * sin(phi), sqrt(1.0f - u.x));
    Float3x3 tnb = make_orthonormal_basis(normal);
    Float3 wi = tnb * wi_local;
    out_pdf = pdf(wo, wi, normal);
    return wi;
}

Float FabricDiffuseBSDF::pdf(
    Expr<float3> wo,
    Expr<float3> wi,
    Expr<float3> normal) const noexcept {

    return ite(dot(wo, normal) * dot(wi, normal) > 0.0f,
        abs(dot(wi, normal)) / pi, 0.0f);
}

//==============================================================================
// Thin Dielectric BSDF
//==============================================================================

Float3 ThinDielectricBSDF::evaluate(
    Expr<float3> wo,
    Expr<float3> wi,
    Expr<float3> normal) const noexcept {

    Float3 result = def(make_float3(0.0f));

    Float3x3 tnb = make_orthonormal_basis(normal);
    Float3 wo_local = transpose(tnb) * wo;
    Float3 wi_local = transpose(tnb) * wi;

    Float cos_theta_o = abs_cos_theta(wo_local);
    Float cos_theta_i = abs_cos_theta(wi_local);

    $if(cos_theta_o > 0.0f & cos_theta_i > 0.0f) {
        // Fresnel at normal incidence (thin wall: use average angle)
        Float3 wh = normalize(wo_local + wi_local);
        Float cos_theta_d = abs_dot(wi_local, wh);
        Float F = fresnel_dielectric(cos_theta_d, 1.0f, ior);

        $if(same_hemisphere(wo_local, wi_local)) {
            // Reflection lobe
            $if(roughness < 1e-4f) {
                // Delta reflection: only non-zero if wi == reflect(-wo, n)
                // In practice, for discrete evaluation this is ~0 unless wi is exactly the reflection
                // For integration with ReSTIR, we return 0 (delta lobe sampled, not evaluated)
                result = make_float3(0.0f);
            }
            $else {
                // Rough reflection: GGX microfacet
                Float2 alpha = roughness_to_alpha(make_float2(roughness));
                Float D = ggx_distribution(wh, alpha);
                Float G = ggx_G(wo_local, wi_local, alpha);
                result = make_float3(F) * D * G /
                         (4.0f * cos_theta_o * cos_theta_i);
            };
        }
        $else {
            // Transmission lobe: diffuse transmission
            // f_trans = (1-F) * albedo / pi
            result = (1.0f - F) * albedo * inv_pi;
        };
    };

    return result;
}

Float3 ThinDielectricBSDF::sample(
    Expr<float3> wo,
    Expr<float3> normal,
    Expr<float2> u,
    Float& out_pdf) const noexcept {

    Float3x3 tnb = make_orthonormal_basis(normal);
    Float3 wo_local = transpose(tnb) * wo;
    Float cos_theta_o = abs_cos_theta(wo_local);

    // Fresnel at normal incidence determines reflect vs transmit probability
    Float F = fresnel_dielectric(cos_theta_o, 1.0f, ior);

    Float3 wi_local = def(make_float3(0.0f));
    Float pdf_reflect = def(0.0f);
    Float pdf_transmit = def(0.0f);
    Bool chose_reflect = u.x < F;
    Float2 u_remapped = ite(chose_reflect,
        make_float2(u.x / max(F, 1e-6f), u.y),
        make_float2((u.x - F) / max(1.0f - F, 1e-6f), u.y));

    $if(chose_reflect) {
        $if(roughness < 1e-4f) {
            // Delta reflection
            wi_local = make_float3(-wo_local.x, -wo_local.y, wo_local.z);
            pdf_reflect = 1.0f;
        }
        $else {
            // Rough reflection: GGX sampling
            Float2 alpha = roughness_to_alpha(make_float2(roughness));
            Float3 wh_local = sample_ggx_wh(wo_local, alpha, u_remapped);
            wi_local = reflect(-wo_local, wh_local);
            Float3 wh2 = normalize(wo_local + wi_local);
            pdf_reflect = ggx_pdf(wo_local, wh2, alpha) / (4.0f * abs_dot(wo_local, wh2));
        };
    }
    $else {
        // Diffuse transmission: cosine-weighted hemisphere on the back side
        Float3 wi_trans = sample_cosine_hemisphere(u_remapped);
        // Flip to opposite hemisphere from wo
        wi_trans.z = -abs(wi_trans.z) * sign(cos_theta_o);
        wi_local = wi_trans;
        pdf_transmit = abs_cos_theta(wi_local) * inv_pi;
    };

    // Combined PDF
    out_pdf = F * pdf_reflect + (1.0f - F) * pdf_transmit;

    return tnb * wi_local;
}

Float ThinDielectricBSDF::pdf(
    Expr<float3> wo,
    Expr<float3> wi,
    Expr<float3> normal) const noexcept {

    Float3x3 tnb = make_orthonormal_basis(normal);
    Float3 wo_local = transpose(tnb) * wo;
    Float3 wi_local = transpose(tnb) * wi;

    Float cos_theta_o = abs_cos_theta(wo_local);
    Float F = fresnel_dielectric(cos_theta_o, 1.0f, ior);

    Float pdf_reflect = def(0.0f);
    Float pdf_transmit = def(0.0f);

    $if(same_hemisphere(wo_local, wi_local)) {
        $if(roughness < 1e-4f) {
            // Delta: zero PDF for non-exact directions (will never be queried in practice)
            pdf_reflect = 0.0f;
        }
        $else {
            Float2 alpha = roughness_to_alpha(make_float2(roughness));
            Float3 wh = normalize(wo_local + wi_local);
            pdf_reflect = ggx_pdf(wo_local, wh, alpha) / (4.0f * abs_dot(wo_local, wh));
        };
    }
    $else {
        pdf_transmit = abs_cos_theta(wi_local) * inv_pi;
    };

    return F * pdf_reflect + (1.0f - F) * pdf_transmit;
}

//==============================================================================
// Material BSDF (Combined)
//==============================================================================

// Population of MaterialBSDF::lobe_list from bsdf_type + Disney params.
// See plan piped-discovering-walrus.md §Per-material-type emission.
//
// Slot assignments use compile-time indices (0..kMaxLobes-1); RHS values are
// runtime DSL expressions computed from bsdf_type and the Disney param fields.
// Host-side loop with compile-time `i` is used for the initial all-null reset.
//
// Per-type emission:
//   ThinDielectric (11) / Dielectric (3): single DeltaDielectric lobe, count=1
//   Everything else (incl. bsdf_type=0 placeholder used by inline
//   constructions in Shading.h/GIShading.h): standard layered, count=7 —
//     slot 0: SpecularMetal    w = metallic
//     slot 1: Diffuse          w = (1-m)*(1-st)*(1-flat)*(1-fab)
//     slot 2: Fabric           w = (1-m)*(1-st)*(1-flat)*fab
//     slot 3: Subsurface       w = (1-m)*(1-st)*flat
//     slot 4: Transmission     w = (1-m)*st
//     slot 5: Sheen (Charlie, 0.25) w = 0.25*(1-m)*sheen — sampling pool
//             under NT_ENABLE_SHEEN_SAMPLING; budget (1-w5) off slots 1-3
//     slot 6: Clearcoat (add.) w = 0.25*clearcoat
// Sampling-pool weights (slots 0-4) sum to <= 1 by construction (the sheen
// budget removes w5 from the diffuse family, so diffuse-family + sheen stays
// energy-bounded even before the lobe's own G term).
//
// NOTE: Unlit (12) / Emissive (5) / Null (0) are NOT special-cased here —
// they fall through to standard layered. The plan called for empty lists but
// the current engine routes them through standard layered (the OLD $else
// branch in evaluate/pdf/sample). Matching that behavior in Phase 1; can be
// refined in Phase 2.
void build_lobe_list_for_layer(MaterialBSDF const& bsdf, LobeList& lobe_list) noexcept {
    using UL = LobeType;

    // --- Reset: all slots null (preserve DeltaDielectric slot-0 flags below) ---
    // We zero both the type and the flags here, then re-set per slot. The
    // DeltaDielectric path sets slot 0's flags to
    // (kLobeIsReflection | kLobeIsTransmission) which are still consumed by
    // evaluate_split's delta routing — do NOT blanket-clear flags anywhere else.
    for (uint i = 0u; i < LobeList::kMaxLobes; ++i) {
        lobe_list.type_flags[i] = pack_lobe(static_cast<uint>(UL::Diffuse), 0u);
        lobe_list.weights[i]    = 0.f;
    }

    // --- Disney canonical weights (standard-layered path) ---
    Float one_minus_metallic   = 1.f - bsdf.metallic;
    Float one_minus_spec_trans = 1.f - bsdf.specular_trans_val;
    Float one_minus_flat       = 1.f - bsdf.flatness_val;
    // Sheen weight: Disney 0.25 scale (same convention as the coat slot). The
    // same fraction is taken out of the diffuse-family budget below so the
    // additive Charlie lobe cannot push total reflectance past 1.
    Float sheen_w              = 0.25f * one_minus_metallic * bsdf.sheen_val;
    Float one_minus_sheen      = 1.f - sheen_w;
    Float diffuse_pool_budget  = one_minus_metallic * one_minus_spec_trans * one_minus_flat
                               * one_minus_sheen;

    // --- Slot 0 + count: depends on bsdf_type ---
    UInt  s0_type   = def(static_cast<uint>(UL::SpecularMetal));
    Float s0_weight = def(bsdf.metallic);
    UInt  s0_flags  = def(kLobeIsReflection);
    UInt  count_val = def(7u);

    // Delta-only fast path (Phase 2F): for Dielectric/ThinDielectric, only
    // slot 0 is read by downstream paths (gated by count==1). Skip writes to
    // slots 1-6 — they're guaranteed-zero from the reset above. Saves BW.
    Bool is_delta = (bsdf.bsdf_type == 11u) | (bsdf.bsdf_type == 3u);
    $if(is_delta) {
        s0_type   = static_cast<uint>(UL::DeltaDielectric);
        s0_weight = 1.f;
        s0_flags  = kLobeIsReflection | kLobeIsTransmission;
        count_val = 1u;
    };

    lobe_list.type_flags[0] = pack_lobe(s0_type, s0_flags);
    lobe_list.weights[0]    = s0_weight;

    // --- Slots 1-6: standard-layered lobes (skipped on delta-only path) ---
    // SSS flag bit narrowed in Phase 2E: transmission bit was dead (routing is
    // hard-coded per slot in evaluate_split). kLobeIsTransmission stays
    // defined for the DeltaDielectric delta path; not set on SSS anymore.
    $if(!is_delta) {
        lobe_list.type_flags[1] = pack_lobe(static_cast<uint>(UL::Diffuse), kLobeIsReflection);
        lobe_list.weights[1]    = diffuse_pool_budget * (1.f - bsdf.fabric_val);

        lobe_list.type_flags[2] = pack_lobe(static_cast<uint>(UL::Fabric), kLobeIsReflection);
        lobe_list.weights[2]    = diffuse_pool_budget * bsdf.fabric_val;

        lobe_list.type_flags[3] = pack_lobe(static_cast<uint>(UL::Subsurface), kLobeIsReflection);
        lobe_list.weights[3]    = one_minus_metallic * one_minus_spec_trans * bsdf.flatness_val
                                * one_minus_sheen;

        // Phase 2B: Transmission lobe properly wired in evaluate/sample/pdf.
        // Discarded is_tx out-param — existing TIR pdf math at BSDF.cpp:1727
        // already handles total internal reflection via F*pdf_r + (1-F)*pdf_t
        // mixture (pdf_t evaluates to 0 for same-hemisphere wi).
        lobe_list.type_flags[4] = pack_lobe(static_cast<uint>(UL::Transmission), kLobeIsTransmission);
        lobe_list.weights[4]    = one_minus_metallic * bsdf.specular_trans_val;

        lobe_list.type_flags[5] = pack_lobe(static_cast<uint>(UL::Sheen),
                                            kLobeIsReflection);
        lobe_list.weights[5]    = sheen_w;

        lobe_list.type_flags[6] = pack_lobe(static_cast<uint>(UL::Clearcoat),
                                            kLobeIsAdditive | kLobeIsReflection);
        lobe_list.weights[6]    = 0.25f * bsdf.clearcoat_val;
    };

    lobe_list.count = count_val;
}

void MaterialBSDF::precompute_tangent_rotation(
    Expr<luisa::float3> normal) noexcept {
    // Item 10 hoist: identical expression to the former per-evaluate
    // computation, so results are bit-identical for any construction site
    // that passes the same normal its evaluate entry points receive.
    Float3 bitangent = cross(normal, tangent_dir) * bitangent_sign;
    Float cos_r = cos(anisotropic_rot_val);
    Float sin_r = sin(anisotropic_rot_val);
    t_rot_val = normalize(tangent_dir * cos_r + bitangent * sin_r);
}

void MaterialBSDF::build_lobe_list() noexcept {
    // thin wrapper around the free function so Phase 2D's
    // resolve_surface_layered can build a LobeList for an arbitrary
    // per-layer MaterialBSDF without going through the member function.
    build_lobe_list_for_layer(*this, lobe_list);
    // build_lobe_list_for_layer writes the *standard-layered* (count==7 or 1)
    // layout into lobe_list. If this MaterialBSDF previously held a composed
    // list (has_composed_lobe_list==true, e.g. via copy-then-rebuild in the
    // GI FinalShading MIS path at PipelineInit.cpp:1427), the composed flag
    // must be cleared — otherwise evaluate/evaluate_split/sample/pdf would
    // route through the composed (9-slot) path against a standard-layered
    // lobe_list, reading type_flags/weights at the wrong slot indices.
    has_composed_lobe_list = false;
}

Float3 MaterialBSDF::evaluate(
    Expr<float3> wo,
    Expr<float3> wi,
    Expr<float3> normal) const noexcept {

    Float3 f = def(make_float3(0.f));

    $if(has_composed_lobe_list) {
        // ==============================================================
        // Phase 2D layered path (count==9).
        // Slot 0     : coat Clearcoat (additive)        [coat params]
        // Slots 1-7  : base lobes (base_bsdf.lobe_list shifted by 1)
        // Slot 8     : fuzz Sheen (additive)            [fuzz params]
        // Base slot 1 type varies (SpecularMetal or DeltaDielectric) — dispatch on lobe_type.
        // ==============================================================
        using UL = LobeType;

        // base_scale = (1-F12)·(1-F23): coat Fresnel transmission, attenuates
        // base lobes. coat_F12 / coat_F23 are forwarded from resolve_surface_layered
        // (single fresnel_dielectric per layered pixel, instead of a recompute
        // here; F23 stays 0 for conductor bases — see resolve_surface_layered).
        //
        // NT_ENABLE_TWO_INTERFACE_FRESNEL drops the legacy weights[0] > 0 gate so
        // Dielectric coats (whose slot-0 weight is coat_weight, not coat_weight *
        // clearcoat_val) still attenuate the base. Safe because coat_F12 / coat_F23
        // default to 0 when no coat → base_scale = 1.
        #if NT_ENABLE_TWO_INTERFACE_FRESNEL
        Float base_scale_cs = (1.f - coat_F12) * (1.f - coat_F23)
                            * (1.f - lobe_list.weights[8]);
        #else
        Float base_scale_cs = def(1.f - lobe_list.weights[8]);
        $if(lobe_list.weights[0] > 0.f) {
            Float cos_theta_o_cs = max(dot(wo, normal), 0.f);
            Float F_coat_cs = fresnel_dielectric(cos_theta_o_cs, 1.0f, coat_ior);
            base_scale_cs = (1.f - F_coat_cs) * (1.f - lobe_list.weights[8]);
        };
        #endif

        // Slot 0: coat. Clearcoat (coat_bsdf_type == 7) uses ClearcoatBSDF with
        // internal Schlick at IOR 1.5 (legacy behavior). Dielectric (coat_bsdf_type
        // == 3) uses MicrofacetBSDF with the coat's own ior + roughness for proper
        // F12-weighted reflection — gated by NT_ENABLE_TWO_INTERFACE_FRESNEL.
        // strength=1.0 for Clearcoat since weights[0] already carries the product.
        $if(lobe_list.weights[0] > 0.f) {
            #if NT_ENABLE_TWO_INTERFACE_FRESNEL
            $if(coat_bsdf_type == 3u) {
                Float2 coat_alpha = roughness_to_alpha(make_float2(coat_roughness));
                MicrofacetBSDF coat_microfacet{
                    make_float3(1.f), coat_alpha, 0.f, coat_ior,
                    0.f, 1.3f, 0.f,
                    tangent_dir, bitangent_sign,
                    attenuation_val, conductor_k_val};
                f += lobe_list.weights[0] * coat_microfacet.evaluate(wo, wi, normal);
            } $else {
                ClearcoatBSDF cc{1.0f, coat_clearcoat_gloss_val};
                f += lobe_list.weights[0] * cc.evaluate(wo, wi, normal);
            };
            #else
            ClearcoatBSDF cc{1.0f, coat_clearcoat_gloss_val};
            f += lobe_list.weights[0] * cc.evaluate(wo, wi, normal);
            #endif
        };

        // Slot 1: SpecularMetal OR DeltaDielectric (dispatch on lobe_type).
        // For SpecularMetal, always evaluate with weight = base_scale (pre-refactor
        // semantics: dielectric GGX is always present for non-metals, even when
        // metallic=0 and weights[1]==0). lobe_list.weights[1] stays = metallic*base_scale
        // for sampling; the eval weight is decoupled. DeltaDielectric base uses
        // weights[1] (= base_scale, > 0) since the is_delta branch sets base slot 0
        // weight to 1.
        {
            UInt t1 = lobe_type(lobe_list.type_flags[1]);
            $if(t1 == static_cast<uint>(UL::SpecularMetal)) {
                Float aspect = sqrt(max(1.f - anisotropic_val * 0.9f, 0.1f));
                Float2 alpha = roughness_to_alpha(make_float2(
                    max(roughness / aspect, 1e-4f),
                    max(roughness * aspect, 1e-4f)));
                // t_rot hoisted to construction time (t_rot_val, item 10).
                MicrofacetBSDF spec{albedo, alpha, metallic, ior,
                                    iridescence_val, iridescence_ior_val, iridescence_thickness_val,
                                    t_rot_val, bitangent_sign,
                                    attenuation_val, conductor_k_val};
                Float3 spec_eval = spec.evaluate(wo, wi, normal);
                Float spec_lum = max(dot(albedo, make_float3(0.2126f, 0.7152f, 0.0722f)), 1e-6f);
                Float3 albedo_chroma = albedo / spec_lum;
                Float3 spec_tint_color = (1.f - specular_tint_val) * make_float3(1.f) + specular_tint_val * albedo_chroma;
                f += base_scale_cs * spec_eval * spec_tint_color;
#if NT_ENABLE_MS_GGX
                // Kulla-Conty f_ms — invariants hoisted to resolve_surface;
                // only the E_i fit is per-call. See docs/ms_ggx_compensation.md.
                $if(dot(ms_f_avg, ms_f_avg) > 0.f) {
                    Float E_i_ms = ggx_directional_albedo_fit(
                        abs(dot(wi, normal)), sqrt(alpha.x * alpha.y));
                    f += base_scale_cs * ms_f_avg * ((1.f - ms_e_o) * (1.f - E_i_ms)
                                                     / (pi * max(1.f - ms_e_avg, 1e-3f)));
                };
#endif
            }
            $elif(t1 == static_cast<uint>(UL::DeltaDielectric)) {
                $if(lobe_list.weights[1] > 0.f) {
                    Float3 slot1_eval = def(make_float3(0.f));
                    $if(bsdf_type == 11u) {
                        ThinDielectricBSDF thin{albedo, roughness, ior};
                        slot1_eval = thin.evaluate(wo, wi, normal);
                    }
                    $else {
                        DielectricBSDF di{attenuation_val, roughness, ior};
                        Float3 di_eval = di.evaluate(wo, wi, normal);
                        Bool is_tx = dot(wo, normal) * dot(wi, normal) < 0.0f;
                        slot1_eval = ite(is_tx, di_eval * specular_trans_val, di_eval);
                    };
                    f += lobe_list.weights[1] * slot1_eval;
                };
            };
        };

        // Slot 2: Diffuse
        $if(lobe_list.weights[2] > 0.f) {
            LambertianBSDF diffuse{albedo};
            f += lobe_list.weights[2] * diffuse.evaluate(wo, wi, normal);
        };

        // Slot 3: Fabric
        $if(lobe_list.weights[3] > 0.f) {
            FabricDiffuseBSDF fabric{albedo};
            f += lobe_list.weights[3] * fabric.evaluate(wo, wi, normal);
        };

        // Slot 4: Subsurface
        $if(lobe_list.weights[4] > 0.f) {
            Float3 sss_tx_color = albedo * exp(-attenuation_val / max(attenuation_distance_val, 1e-4f));
            SubsurfaceBSDF sss{albedo * attenuation_val, sss_tx_color,
                                ite(flatness_val > 0.f, diffuse_trans_val, 0.f), ior};
            f += lobe_list.weights[4] * sss.evaluate(wo, wi, normal);
        };

        // Slot 5: Transmission
        $if(lobe_list.weights[5] > 0.f) {
            MicrofacetTransmissionBSDF tx{
                attenuation_val,
                roughness_to_alpha(make_float2(roughness)),
                1.f / ior, ior};
            f += lobe_list.weights[5] * tx.evaluate(wo, wi, normal);
        };

        // Slot 6: Sheen (additive — base's own)
        $if(lobe_list.weights[6] > 0.f) {
            SheenBSDF sheen{1.0f, sheen_tint_val, albedo, roughness};
            f += lobe_list.weights[6] * sheen.evaluate(wo, wi, normal);
        };

        // Slot 7: Clearcoat (additive — base's own). strength=1.0 for the same
        // reason as slot 0; weights[7] = 0.25 * base.clearcoat_val (set in
        // build_lobe_list_for_layer).
        $if(lobe_list.weights[7] > 0.f) {
            ClearcoatBSDF cc{1.0f, clearcoat_gloss_val};
            f += lobe_list.weights[7] * cc.evaluate(wo, wi, normal);
        };

        // Slot 8: Fuzz Sheen (additive — uses fuzz params)
        $if(lobe_list.weights[8] > 0.f) {
            SheenBSDF sheen{1.0f, fuzz_sheen_tint_val, fuzz_albedo, fuzz_sheen_roughness_val};
            f += lobe_list.weights[8] * sheen.evaluate(wo, wi, normal);
        };
    }
    $else {
        // ==============================================================
        // Phase 1 slot-specialized path (single-layer).
        // ==============================================================
        // --- Delta dielectric path (count==1, slot 0 only) ---
        $if(lobe_list.count == 1u) {
            Float w0 = lobe_list.weights[0];
            $if(w0 > 0.f) {
                Float3 slot0_eval = def(make_float3(0.f));
                $if(bsdf_type == 11u) {
                    ThinDielectricBSDF thin{albedo, roughness, ior};
                    slot0_eval = thin.evaluate(wo, wi, normal);
                }
                $else {
                    DielectricBSDF di{attenuation_val, roughness, ior};
                    Float3 di_eval = di.evaluate(wo, wi, normal);
                    Bool is_tx = dot(wo, normal) * dot(wi, normal) < 0.0f;
                    slot0_eval = ite(is_tx, di_eval * specular_trans_val, di_eval);
                };
                f = w0 * slot0_eval;
            };
        }
        // --- Standard layered path (count==7) ---
        $else {
            // Slot 0: SpecularMetal — always evaluate with weight 1 (pre-refactor
            // semantics). MicrofacetBSDF internally dispatches on `metallic`:
            // metallic<0.5 → dielectric GGX (F0 from ior), metallic>=0.5 → conductor.
            // For non-metals (metallic==0, weights[0]==0) this restores the
            // dielectric specular reflection that the slot-weight-gated path was
            // skipping. lobe_list.weights[0] (= metallic) is still used by sample()/pdf()
            // for the sampling-pool threshold; eval weight is decoupled.
            {
                Float aspect = sqrt(max(1.f - anisotropic_val * 0.9f, 0.1f));
                Float2 alpha = roughness_to_alpha(make_float2(
                    max(roughness / aspect, 1e-4f),
                    max(roughness * aspect, 1e-4f)));
                // t_rot hoisted to construction time (t_rot_val, item 10).
                MicrofacetBSDF spec{albedo, alpha, metallic, ior,
                                    iridescence_val, iridescence_ior_val, iridescence_thickness_val,
                                    t_rot_val, bitangent_sign,
                                    attenuation_val, conductor_k_val};
                Float3 spec_eval = spec.evaluate(wo, wi, normal);
                Float spec_lum = max(dot(albedo, make_float3(0.2126f, 0.7152f, 0.0722f)), 1e-6f);
                Float3 albedo_chroma = albedo / spec_lum;
                Float3 spec_tint_color = (1.f - specular_tint_val) * make_float3(1.f) + specular_tint_val * albedo_chroma;
                f += spec_eval * spec_tint_color;
#if NT_ENABLE_MS_GGX
                // Kulla-Conty f_ms — invariants hoisted to resolve_surface;
                // only the E_i fit is per-call. See docs/ms_ggx_compensation.md.
                $if(dot(ms_f_avg, ms_f_avg) > 0.f) {
                    Float E_i_ms = ggx_directional_albedo_fit(
                        abs(dot(wi, normal)), sqrt(alpha.x * alpha.y));
                    f += ms_f_avg * ((1.f - ms_e_o) * (1.f - E_i_ms)
                                     / (pi * max(1.f - ms_e_avg, 1e-3f)));
                };
#endif
            };

            // Slot 1: Diffuse
            $if(lobe_list.weights[1] > 0.f) {
                LambertianBSDF diffuse{albedo};
                f += lobe_list.weights[1] * diffuse.evaluate(wo, wi, normal);
            };

            // Slot 2: Fabric
            $if(lobe_list.weights[2] > 0.f) {
                FabricDiffuseBSDF fabric{albedo};
                f += lobe_list.weights[2] * fabric.evaluate(wo, wi, normal);
            };

            // Slot 3: Subsurface
            $if(lobe_list.weights[3] > 0.f) {
                Float3 sss_tx_color = albedo * exp(-attenuation_val / max(attenuation_distance_val, 1e-4f));
                SubsurfaceBSDF sss{albedo * attenuation_val, sss_tx_color,
                                    ite(flatness_val > 0.f, diffuse_trans_val, 0.f), ior};
                f += lobe_list.weights[3] * sss.evaluate(wo, wi, normal);
            };

            // Slot 4: Transmission (Phase 2B — MicrofacetTransmissionBSDF)
            $if(lobe_list.weights[4] > 0.f) {
                MicrofacetTransmissionBSDF tx{
                    attenuation_val,
                    roughness_to_alpha(make_float2(roughness)),
                    1.f / ior, ior};
                f += lobe_list.weights[4] * tx.evaluate(wo, wi, normal);
            };

            // Slot 5: Sheen (additive)
            $if(lobe_list.weights[5] > 0.f) {
                SheenBSDF sheen{1.0f, sheen_tint_val, albedo, roughness};
                f += lobe_list.weights[5] * sheen.evaluate(wo, wi, normal);
            };

            // Slot 6: Clearcoat (additive). strength=1.0 — weights[6] already
            // carries 0.25 * clearcoat_val from build_lobe_list_for_layer.
            $if(lobe_list.weights[6] > 0.f) {
                ClearcoatBSDF cc{1.0f, clearcoat_gloss_val};
                f += lobe_list.weights[6] * cc.evaluate(wo, wi, normal);
            };
        };
    };
    return f;
}

// (dispatch_lobe_evaluate_split removed in Option A slot-specialized refactor.
//  Per-slot routing now inlined directly in MaterialBSDF::evaluate_split.)

void MaterialBSDF::evaluate_split(
    Expr<float3> wo,
    Expr<float3> wi,
    Expr<float3> normal,
    Float3& out_diffuse,
    Float3& out_specular) const noexcept {
    out_diffuse  = def(make_float3(0.f));
    out_specular = def(make_float3(0.f));

    $if(has_composed_lobe_list) {
        // ==============================================================
        // Phase 2D layered path. Routing:
        //   slot 0 (coat Clearcoat)           → out_specular
        //   slot 1 (SpecularMetal|DeltaDielectric) → specular / split
        //   slot 2 Diffuse                    → out_diffuse
        //   slot 3 Fabric                     → out_diffuse
        //   slot 4 Subsurface                 → out_diffuse
        //   slot 5 Transmission               → out_diffuse
        //   slot 6 Sheen (base additive)      → out_specular
        //   slot 7 Clearcoat (base additive)  → out_specular
        //   slot 8 Fuzz Sheen                 → out_specular
        // ==============================================================
        using UL = LobeType;

        // base_scale = (1-F12)·(1-F23): forwarded from resolve_surface_layered.
        // NT_ENABLE_TWO_INTERFACE_FRESNEL drops the coat_clearcoat_val > 0 gate so
        // Dielectric coats (which don't set clearcoat_val) still attenuate the base.
        // Safe because coat_F12 / coat_F23 default to 0 → base_scale = 1 when no coat.
        #if NT_ENABLE_TWO_INTERFACE_FRESNEL
        Float base_scale_es = (1.f - coat_F12) * (1.f - coat_F23)
                            * (1.f - lobe_list.weights[8]);
        #else
        Float base_scale_es = def(1.f - lobe_list.weights[8]);
        $if(coat_clearcoat_val > 0.f) {
            Float cos_theta_o_es = max(dot(wo, normal), 0.f);
            Float F_coat_es = fresnel_dielectric(cos_theta_o_es, 1.0f, coat_ior);
            base_scale_es = (1.f - F_coat_es) * (1.f - lobe_list.weights[8]);
        };
        #endif

        // Slot 0: coat → specular. Dielectric coat uses MicrofacetBSDF (gated by
        // NT_ENABLE_TWO_INTERFACE_FRESNEL); Clearcoat uses ClearcoatBSDF.
        $if(lobe_list.weights[0] > 0.f) {
            #if NT_ENABLE_TWO_INTERFACE_FRESNEL
            $if(coat_bsdf_type == 3u) {
                Float2 coat_alpha = roughness_to_alpha(make_float2(coat_roughness));
                MicrofacetBSDF coat_microfacet{
                    make_float3(1.f), coat_alpha, 0.f, coat_ior,
                    0.f, 1.3f, 0.f,
                    tangent_dir, bitangent_sign,
                    attenuation_val, conductor_k_val};
                out_specular += lobe_list.weights[0] * coat_microfacet.evaluate(wo, wi, normal);
            } $else {
                ClearcoatBSDF cc{1.0f, coat_clearcoat_gloss_val};
                out_specular += lobe_list.weights[0] * cc.evaluate(wo, wi, normal);
            };
            #else
            ClearcoatBSDF cc{1.0f, coat_clearcoat_gloss_val};
            out_specular += lobe_list.weights[0] * cc.evaluate(wo, wi, normal);
            #endif
        };

        // Slot 1: SpecularMetal → specular, DeltaDielectric → split.
        // SpecularMetal always evaluated with weight = base_scale (pre-refactor
        // semantics: dielectric GGX always present for non-metals).
        {
            UInt t1 = lobe_type(lobe_list.type_flags[1]);
            $if(t1 == static_cast<uint>(UL::SpecularMetal)) {
                Float aspect = sqrt(max(1.f - anisotropic_val * 0.9f, 0.1f));
                Float2 alpha = roughness_to_alpha(make_float2(
                    max(roughness / aspect, 1e-4f),
                    max(roughness * aspect, 1e-4f)));
                // t_rot hoisted to construction time (t_rot_val, item 10).
                MicrofacetBSDF spec{albedo, alpha, metallic, ior,
                                    iridescence_val, iridescence_ior_val, iridescence_thickness_val,
                                    t_rot_val, bitangent_sign,
                                    attenuation_val, conductor_k_val};
                Float3 spec_eval = spec.evaluate(wo, wi, normal);
                Float spec_lum = max(dot(albedo, make_float3(0.2126f, 0.7152f, 0.0722f)), 1e-6f);
                Float3 albedo_chroma = albedo / spec_lum;
                Float3 spec_tint_color = (1.f - specular_tint_val) * make_float3(1.f) + specular_tint_val * albedo_chroma;
                out_specular += base_scale_es * spec_eval * spec_tint_color;
#if NT_ENABLE_MS_GGX
                // Kulla-Conty f_ms — invariants hoisted to resolve_surface;
                // only the E_i fit is per-call. See docs/ms_ggx_compensation.md.
                $if(dot(ms_f_avg, ms_f_avg) > 0.f) {
                    Float E_i_ms = ggx_directional_albedo_fit(
                        abs(dot(wi, normal)), sqrt(alpha.x * alpha.y));
                    out_specular += base_scale_es * ms_f_avg * ((1.f - ms_e_o) * (1.f - E_i_ms)
                                                                / (pi * max(1.f - ms_e_avg, 1e-3f)));
                };
#endif
            }
            $elif(t1 == static_cast<uint>(UL::DeltaDielectric)) {
                $if(lobe_list.weights[1] > 0.f) {
                    Bool is_refl = dot(wo, normal) * dot(wi, normal) > 0.0f;
                    $if(bsdf_type == 11u) {
                        ThinDielectricBSDF thin{albedo, roughness, ior};
                        Float3 eval = thin.evaluate(wo, wi, normal);
                        out_specular += lobe_list.weights[1] * ite(is_refl, eval, make_float3(0.f));
                        out_diffuse  += lobe_list.weights[1] * ite(is_refl, make_float3(0.f), eval);
                    }
                    $else {
                        DielectricBSDF di{attenuation_val, roughness, ior};
                        Float3 di_eval = di.evaluate(wo, wi, normal);
                        out_specular += lobe_list.weights[1] * ite(is_refl, di_eval, make_float3(0.f));
                        out_diffuse  += lobe_list.weights[1] * ite(is_refl, make_float3(0.f), di_eval * specular_trans_val);
                    };
                };
            };
        };

        // Slot 2: Diffuse → diffuse
        $if(lobe_list.weights[2] > 0.f) {
            LambertianBSDF diffuse{albedo};
            out_diffuse += lobe_list.weights[2] * diffuse.evaluate(wo, wi, normal);
        };

        // Slot 3: Fabric → diffuse
        $if(lobe_list.weights[3] > 0.f) {
            FabricDiffuseBSDF fabric{albedo};
            out_diffuse += lobe_list.weights[3] * fabric.evaluate(wo, wi, normal);
        };

        // Slot 4: Subsurface → diffuse
        $if(lobe_list.weights[4] > 0.f) {
            Float3 sss_tx_color = albedo * exp(-attenuation_val / max(attenuation_distance_val, 1e-4f));
            SubsurfaceBSDF sss{albedo * attenuation_val, sss_tx_color,
                                ite(flatness_val > 0.f, diffuse_trans_val, 0.f), ior};
            out_diffuse += lobe_list.weights[4] * sss.evaluate(wo, wi, normal);
        };

        // Slot 5: Transmission → diffuse
        $if(lobe_list.weights[5] > 0.f) {
            MicrofacetTransmissionBSDF tx{
                attenuation_val,
                roughness_to_alpha(make_float2(roughness)),
                1.f / ior, ior};
            out_diffuse += lobe_list.weights[5] * tx.evaluate(wo, wi, normal);
        };

        // Slot 6: Sheen (base additive) → specular. The sheen lobe is
        // mix(white, albedo, tint) — not albedo-proportional, so it must not
        // be divided by the diffuse demod factor (green/magenta amplification
        // on saturated fabrics); the spec channel has no albedo demod.
        $if(lobe_list.weights[6] > 0.f) {
            SheenBSDF sheen{1.0f, sheen_tint_val, albedo, roughness};
            out_specular += lobe_list.weights[6] * sheen.evaluate(wo, wi, normal);
        };

        // Slot 7: Clearcoat (base additive) → specular. strength=1.0.
        $if(lobe_list.weights[7] > 0.f) {
            ClearcoatBSDF cc{1.0f, clearcoat_gloss_val};
            out_specular += lobe_list.weights[7] * cc.evaluate(wo, wi, normal);
        };

        // Slot 8: Fuzz Sheen → specular (see slot 6: white-dominant lobe,
        // not albedo-proportional)
        $if(lobe_list.weights[8] > 0.f) {
            SheenBSDF sheen{1.0f, fuzz_sheen_tint_val, fuzz_albedo, fuzz_sheen_roughness_val};
            out_specular += lobe_list.weights[8] * sheen.evaluate(wo, wi, normal);
        };
    }
    $else {
        // ==============================================================
        // Phase 1 slot-specialized path (single-layer).
        // ==============================================================
        // --- Delta dielectric path (count==1) ---
        $if(lobe_list.count == 1u) {
            Float w0 = lobe_list.weights[0];
            $if(w0 > 0.f) {
                Bool is_refl = dot(wo, normal) * dot(wi, normal) > 0.0f;
                $if(bsdf_type == 11u) {
                    ThinDielectricBSDF thin{albedo, roughness, ior};
                    Float3 eval = thin.evaluate(wo, wi, normal);
                    out_specular += w0 * ite(is_refl, eval, make_float3(0.f));
                    out_diffuse  += w0 * ite(is_refl, make_float3(0.f), eval);
                }
                $else {
                    DielectricBSDF di{attenuation_val, roughness, ior};
                    Float3 di_eval = di.evaluate(wo, wi, normal);
                    out_specular += w0 * ite(is_refl, di_eval, make_float3(0.f));
                    out_diffuse  += w0 * ite(is_refl, make_float3(0.f), di_eval * specular_trans_val);
                };
            };
        }
        // --- Standard layered path ---
        $else {
            // Slot 0: SpecularMetal → specular channel. Always evaluated with
            // weight 1 (pre-refactor semantics — dielectric GGX always present
            // for non-metals, even when metallic==0 and weights[0]==0).
            {
                Float aspect = sqrt(max(1.f - anisotropic_val * 0.9f, 0.1f));
                Float2 alpha = roughness_to_alpha(make_float2(
                    max(roughness / aspect, 1e-4f),
                    max(roughness * aspect, 1e-4f)));
                // t_rot hoisted to construction time (t_rot_val, item 10).
                MicrofacetBSDF spec{albedo, alpha, metallic, ior,
                                    iridescence_val, iridescence_ior_val, iridescence_thickness_val,
                                    t_rot_val, bitangent_sign,
                                    attenuation_val, conductor_k_val};
                Float3 spec_eval = spec.evaluate(wo, wi, normal);
                Float spec_lum = max(dot(albedo, make_float3(0.2126f, 0.7152f, 0.0722f)), 1e-6f);
                Float3 albedo_chroma = albedo / spec_lum;
                Float3 spec_tint_color = (1.f - specular_tint_val) * make_float3(1.f) + specular_tint_val * albedo_chroma;
                out_specular += spec_eval * spec_tint_color;
#if NT_ENABLE_MS_GGX
                // Kulla-Conty f_ms — invariants hoisted to resolve_surface;
                // only the E_i fit is per-call. See docs/ms_ggx_compensation.md.
                $if(dot(ms_f_avg, ms_f_avg) > 0.f) {
                    Float E_i_ms = ggx_directional_albedo_fit(
                        abs(dot(wi, normal)), sqrt(alpha.x * alpha.y));
                    out_specular += ms_f_avg * ((1.f - ms_e_o) * (1.f - E_i_ms)
                                                / (pi * max(1.f - ms_e_avg, 1e-3f)));
                };
#endif
            };

            // Slot 1: Diffuse → diffuse channel
            $if(lobe_list.weights[1] > 0.f) {
                LambertianBSDF diffuse{albedo};
                out_diffuse += lobe_list.weights[1] * diffuse.evaluate(wo, wi, normal);
            };

            // Slot 2: Fabric → diffuse channel
            $if(lobe_list.weights[2] > 0.f) {
                FabricDiffuseBSDF fabric{albedo};
                out_diffuse += lobe_list.weights[2] * fabric.evaluate(wo, wi, normal);
            };

            // Slot 3: Subsurface → diffuse channel
            $if(lobe_list.weights[3] > 0.f) {
                Float3 sss_tx_color = albedo * exp(-attenuation_val / max(attenuation_distance_val, 1e-4f));
                SubsurfaceBSDF sss{albedo * attenuation_val, sss_tx_color,
                                    ite(flatness_val > 0.f, diffuse_trans_val, 0.f), ior};
                out_diffuse += lobe_list.weights[3] * sss.evaluate(wo, wi, normal);
            };

        // Slot 4: Transmission → diffuse channel (Phase 2B)
        $if(lobe_list.weights[4] > 0.f) {
            MicrofacetTransmissionBSDF tx{
                attenuation_val,
                roughness_to_alpha(make_float2(roughness)),
                1.f / ior, ior};
            out_diffuse += lobe_list.weights[4] * tx.evaluate(wo, wi, normal);
        };

        // Slot 5: Sheen → specular channel (white-dominant lobe, not
        // albedo-proportional — see composed-path slot 6 comment)
        $if(lobe_list.weights[5] > 0.f) {
            SheenBSDF sheen{1.0f, sheen_tint_val, albedo, roughness};
            out_specular += lobe_list.weights[5] * sheen.evaluate(wo, wi, normal);
        };

        // Slot 6: Clearcoat → specular channel. strength=1.0.
        $if(lobe_list.weights[6] > 0.f) {
            ClearcoatBSDF cc{1.0f, clearcoat_gloss_val};
            out_specular += lobe_list.weights[6] * cc.evaluate(wo, wi, normal);
        };
        };
    };
}

// (dispatch_lobe_sample removed in Option A slot-specialized refactor.
//  Per-slot sampling now inlined directly in MaterialBSDF::sample.)

Float3 MaterialBSDF::sample(
    Expr<float3> wo,
    Expr<float3> normal,
    Expr<float2> u,
    Float& out_pdf) const noexcept {
    Float r = u.x;
    Float3 wi = def(make_float3(0.f));
    Float p  = def(0.f);

    $if(has_composed_lobe_list) {
        // ==============================================================
        // Phase 2D layered path. Sampling pool = slots 1-5 (base's 0-4),
        // plus slot 0 (coat) under NT_ENABLE_COAT_SAMPLING. Base additive
        // (6, 7) and fuzz (8) are NOT in the pool.
        // ==============================================================
        using UL = LobeType;

        Float threshold = def(0.f);
        UInt  chosen = def(1u);
        Float weight_chosen = def(0.f);
        Float threshold_chosen = def(0.f);
        Bool  found = def(false);
        Float last_weight = def(0.f);
        Float last_threshold = def(0.f);
        UInt  last_chosen = def(1u);
        Bool  any_valid = def(false);

#if NT_ENABLE_COAT_SAMPLING || NT_ENABLE_SHEEN_SAMPLING
        // Pool renormalization: composed pool weights sum to base_scale (< 1),
        // and the coat weight can push the sum above 1 — rescale r by the pool
        // sum so the picker always lands on a real lobe instead of falling
        // through to a zero-weight transmission dispatch. Both flags off →
        // r_cmp = r (bit-identical).
        Float pool_sum = def(0.f);
#if NT_ENABLE_COAT_SAMPLING
        // Fresnel-gated coat pool weight: the coat lobe's directional energy at
        // NoV is ~F12 of the slot weight, so gate the pick probability by the
        // same factor — the single BRDF candidate then follows the energy
        // (normal incidence → base lobes, grazing → coat). weights[0] == 0
        // (no coat) → gate is 0. Must mirror MaterialBSDF::pdf exactly.
        Float coat_pool_w = lobe_list.weights[0] * coat_F12;
        pool_sum += coat_pool_w;
#endif
        pool_sum += lobe_list.weights[1];
        pool_sum += lobe_list.weights[2];
        pool_sum += lobe_list.weights[3];
        pool_sum += lobe_list.weights[4];
        pool_sum += lobe_list.weights[5];
#if NT_ENABLE_SHEEN_SAMPLING
        // Base's own sheen (6) + fuzz sheen (8) join the pool
        pool_sum += lobe_list.weights[6];
        pool_sum += lobe_list.weights[8];
#endif
        Float r_cmp = r * max(pool_sum, 1e-6f);
#else
        Float r_cmp = r;
#endif

        // Picker over pool slots
#if NT_ENABLE_COAT_SAMPLING
        // Slot 0: coat (Clearcoat or Dielectric) — gated weight, not weights[0]
        {
            Float w = coat_pool_w;
            threshold += w;
            last_weight = w; last_threshold = threshold; last_chosen = 0u;
            $if(!found & (r_cmp < threshold)) {
                chosen = 0u; weight_chosen = w; threshold_chosen = threshold; found = true;
            };
        };
#endif
        {
            Float w = lobe_list.weights[1];
            threshold += w;
            last_weight = w; last_threshold = threshold; last_chosen = 1u;
            any_valid = true;
            $if(!found & (r_cmp < threshold)) {
                chosen = 1u; weight_chosen = w; threshold_chosen = threshold; found = true;
            };
        };
        {
            Float w = lobe_list.weights[2];
            threshold += w;
            last_weight = w; last_threshold = threshold; last_chosen = 2u;
            $if(!found & (r_cmp < threshold)) {
                chosen = 2u; weight_chosen = w; threshold_chosen = threshold; found = true;
            };
        };
        {
            Float w = lobe_list.weights[3];
            threshold += w;
            last_weight = w; last_threshold = threshold; last_chosen = 3u;
            $if(!found & (r_cmp < threshold)) {
                chosen = 3u; weight_chosen = w; threshold_chosen = threshold; found = true;
            };
        };
        {
            Float w = lobe_list.weights[4];
            threshold += w;
            last_weight = w; last_threshold = threshold; last_chosen = 4u;
            $if(!found & (r_cmp < threshold)) {
                chosen = 4u; weight_chosen = w; threshold_chosen = threshold; found = true;
            };
        };
        {
            Float w = lobe_list.weights[5];
            threshold += w;
            last_weight = w; last_threshold = threshold; last_chosen = 5u;
            $if(!found & (r_cmp < threshold)) {
                chosen = 5u; weight_chosen = w; threshold_chosen = threshold; found = true;
            };
        };
#if NT_ENABLE_SHEEN_SAMPLING
        // Slots 6 (base's own sheen) + 8 (fuzz sheen)
        {
            Float w = lobe_list.weights[6];
            threshold += w;
            last_weight = w; last_threshold = threshold; last_chosen = 6u;
            $if(!found & (r_cmp < threshold)) {
                chosen = 6u; weight_chosen = w; threshold_chosen = threshold; found = true;
            };
        };
        {
            Float w = lobe_list.weights[8];
            threshold += w;
            last_weight = w; last_threshold = threshold; last_chosen = 8u;
            $if(!found & (r_cmp < threshold)) {
                chosen = 8u; weight_chosen = w; threshold_chosen = threshold; found = true;
            };
        };
#endif
        $if(!found & any_valid) {
            chosen = last_chosen;
            weight_chosen = last_weight;
            threshold_chosen = last_threshold;
        };

        Float threshold_before = threshold_chosen - weight_chosen;
        Float r_offset = clamp(r_cmp - threshold_before, 0.f, weight_chosen);
        Float2 u_local = make_float2(r_offset / max(weight_chosen, 1e-6f), u.y);

        // Sample dispatch
        Float lobe_pdf_sink = def(0.f);
#if NT_ENABLE_COAT_SAMPLING
        // Slot 0: coat — same dual dispatch as evaluate()'s slot 0
        $if(chosen == 0u) {
            #if NT_ENABLE_TWO_INTERFACE_FRESNEL
            $if(coat_bsdf_type == 3u) {
                Float2 coat_alpha = roughness_to_alpha(make_float2(coat_roughness));
                MicrofacetBSDF coat_microfacet{
                    make_float3(1.f), coat_alpha, 0.f, coat_ior,
                    0.f, 1.3f, 0.f,
                    tangent_dir, bitangent_sign,
                    attenuation_val, conductor_k_val};
                wi = coat_microfacet.sample(wo, normal, u_local, lobe_pdf_sink);
            } $else {
                ClearcoatBSDF cc{1.0f, coat_clearcoat_gloss_val};
                wi = cc.sample(wo, normal, u_local, lobe_pdf_sink);
            };
            #else
            ClearcoatBSDF cc{1.0f, coat_clearcoat_gloss_val};
            wi = cc.sample(wo, normal, u_local, lobe_pdf_sink);
            #endif
        };
#endif
        $if(chosen == 1u) {
            UInt t1 = lobe_type(lobe_list.type_flags[1]);
            $if(t1 == static_cast<uint>(UL::SpecularMetal)) {
                Float aspect = sqrt(max(1.f - anisotropic_val * 0.9f, 0.1f));
                Float2 alpha = roughness_to_alpha(make_float2(
                    max(roughness / aspect, 1e-4f),
                    max(roughness * aspect, 1e-4f)));
                // t_rot hoisted to construction time (t_rot_val, item 10).
                MicrofacetBSDF spec{albedo, alpha, metallic, ior,
                                    iridescence_val, iridescence_ior_val, iridescence_thickness_val,
                                    t_rot_val, bitangent_sign,
                                    attenuation_val, conductor_k_val};
                wi = spec.sample(wo, normal, u_local, lobe_pdf_sink);
            }
            $elif(t1 == static_cast<uint>(UL::DeltaDielectric)) {
                $if(bsdf_type == 11u) {
                    ThinDielectricBSDF thin{albedo, roughness, ior};
                    wi = thin.sample(wo, normal, u_local, lobe_pdf_sink);
                }
                $else {
                    DielectricBSDF di{attenuation_val, roughness, ior};
                    Bool discard_is_tx;
                    wi = di.sample(wo, normal, u_local, lobe_pdf_sink, discard_is_tx);
                };
            };
        }
        $elif(chosen == 2u) {
            LambertianBSDF diffuse{albedo};
            wi = diffuse.sample(wo, normal, u_local, lobe_pdf_sink);
        }
        $elif(chosen == 3u) {
            FabricDiffuseBSDF fabric{albedo};
            wi = fabric.sample(wo, normal, u_local, lobe_pdf_sink);
        }
        $elif(chosen == 4u) {
            Float3 sss_tx_color = albedo * exp(-attenuation_val / max(attenuation_distance_val, 1e-4f));
            SubsurfaceBSDF sss{albedo * attenuation_val, sss_tx_color,
                                ite(flatness_val > 0.f, diffuse_trans_val, 0.f), ior};
            wi = sss.sample(wo, normal, u_local, lobe_pdf_sink);
        }
        $elif(chosen == 5u) {
            MicrofacetTransmissionBSDF tx{
                attenuation_val,
                roughness_to_alpha(make_float2(roughness)),
                1.f / ior, ior};
            Bool discard_is_tx;
            wi = tx.sample(wo, normal, u_local, lobe_pdf_sink, discard_is_tx);
        }
#if NT_ENABLE_SHEEN_SAMPLING
        $elif(chosen == 6u) {
            SheenBSDF sheen{1.0f, sheen_tint_val, albedo, roughness};
            wi = sheen.sample(wo, normal, u_local, lobe_pdf_sink);
        }
        $elif(chosen == 8u) {
            SheenBSDF sheen{1.0f, fuzz_sheen_tint_val, fuzz_albedo, fuzz_sheen_roughness_val};
            wi = sheen.sample(wo, normal, u_local, lobe_pdf_sink);
        }
#endif
        ;

        // Mixture PDF over pool slots
#if NT_ENABLE_COAT_SAMPLING
        // Slot 0: coat — keeps f (evaluate slot 0) and p consistent
        $if(lobe_list.weights[0] > 0.f) {
            #if NT_ENABLE_TWO_INTERFACE_FRESNEL
            $if(coat_bsdf_type == 3u) {
                Float2 coat_alpha = roughness_to_alpha(make_float2(coat_roughness));
                MicrofacetBSDF coat_microfacet{
                    make_float3(1.f), coat_alpha, 0.f, coat_ior,
                    0.f, 1.3f, 0.f,
                    tangent_dir, bitangent_sign,
                    attenuation_val, conductor_k_val};
                p += coat_pool_w * coat_microfacet.pdf(wo, wi, normal);
            } $else {
                ClearcoatBSDF cc{1.0f, coat_clearcoat_gloss_val};
                p += coat_pool_w * cc.pdf(wo, wi, normal);
            };
            #else
            ClearcoatBSDF cc{1.0f, coat_clearcoat_gloss_val};
            p += coat_pool_w * cc.pdf(wo, wi, normal);
            #endif
        };
#endif
        $if(lobe_list.weights[1] > 0.f) {
            UInt t1 = lobe_type(lobe_list.type_flags[1]);
            $if(t1 == static_cast<uint>(UL::SpecularMetal)) {
                Float aspect = sqrt(max(1.f - anisotropic_val * 0.9f, 0.1f));
                Float2 alpha = roughness_to_alpha(make_float2(
                    max(roughness / aspect, 1e-4f),
                    max(roughness * aspect, 1e-4f)));
                // t_rot hoisted to construction time (t_rot_val, item 10).
                MicrofacetBSDF spec{albedo, alpha, metallic, ior,
                                    iridescence_val, iridescence_ior_val, iridescence_thickness_val,
                                    t_rot_val, bitangent_sign,
                                    attenuation_val, conductor_k_val};
                p += lobe_list.weights[1] * spec.pdf(wo, wi, normal);
            }
            $elif(t1 == static_cast<uint>(UL::DeltaDielectric)) {
                $if(bsdf_type == 11u) {
                    ThinDielectricBSDF thin{albedo, roughness, ior};
                    p += lobe_list.weights[1] * thin.pdf(wo, wi, normal);
                }
                $else {
                    DielectricBSDF di{attenuation_val, roughness, ior};
                    Float di_pdf = di.pdf(wo, wi, normal);
                    Bool is_tx = dot(wo, normal) * dot(wi, normal) < 0.0f;
                    p += lobe_list.weights[1] * ite(is_tx, di_pdf * specular_trans_val, di_pdf);
                };
            };
        };
        $if(lobe_list.weights[2] > 0.f) {
            LambertianBSDF diffuse{albedo};
            p += lobe_list.weights[2] * diffuse.pdf(wo, wi, normal);
        };
        $if(lobe_list.weights[3] > 0.f) {
            FabricDiffuseBSDF fabric{albedo};
            p += lobe_list.weights[3] * fabric.pdf(wo, wi, normal);
        };
        $if(lobe_list.weights[4] > 0.f) {
            Float3 sss_tx_color = albedo * exp(-attenuation_val / max(attenuation_distance_val, 1e-4f));
            SubsurfaceBSDF sss{albedo * attenuation_val, sss_tx_color,
                                ite(flatness_val > 0.f, diffuse_trans_val, 0.f), ior};
            p += lobe_list.weights[4] * sss.pdf(wo, wi, normal);
        };
        $if(lobe_list.weights[5] > 0.f) {
            MicrofacetTransmissionBSDF tx{
                attenuation_val,
                roughness_to_alpha(make_float2(roughness)),
                1.f / ior, ior};
            p += lobe_list.weights[5] * tx.pdf(wo, wi, normal);
        };
#if NT_ENABLE_SHEEN_SAMPLING
        // Slots 6 & 8: sheen (base's own + fuzz) — keeps f (evaluate slots
        // 6/8) and p consistent
        $if(lobe_list.weights[6] > 0.f) {
            SheenBSDF sheen{1.0f, sheen_tint_val, albedo, roughness};
            p += lobe_list.weights[6] * sheen.pdf(wo, wi, normal);
        };
        $if(lobe_list.weights[8] > 0.f) {
            SheenBSDF sheen{1.0f, fuzz_sheen_tint_val, fuzz_albedo, fuzz_sheen_roughness_val};
            p += lobe_list.weights[8] * sheen.pdf(wo, wi, normal);
        };
#endif
#if NT_ENABLE_COAT_SAMPLING || NT_ENABLE_SHEEN_SAMPLING
        // The picker picks slot i with probability w_i/pool_sum (r is rescaled
        // by pool_sum), so the true sampling density is mixture/pool_sum —
        // divide here or every f/pdf estimate is biased by 1/pool_sum.
        p = p / max(pool_sum, 1e-6f);
#endif
    }
    $else {
        // ==============================================================
        // Phase 1 slot-specialized path (single-layer).
        // ==============================================================
#if NT_ENABLE_SHEEN_SAMPLING
        // Pool renorm (see composed path): slots 0-4 sum to <= 1 and the
        // sheen weight pushes the sum past 1 — rescale r so the picker always
        // lands on a real lobe. Delta path (count==1): pool sums to 1 → no-op.
        Float pool_sum_sl = def(0.f);
        pool_sum_sl += lobe_list.weights[0];
        pool_sum_sl += lobe_list.weights[1];
        pool_sum_sl += lobe_list.weights[2];
        pool_sum_sl += lobe_list.weights[3];
        pool_sum_sl += lobe_list.weights[4];
        pool_sum_sl += lobe_list.weights[5];
        r = r * max(pool_sum_sl, 1e-6f);
#endif
        Float threshold = def(0.f);
        UInt  chosen = def(0u);
        Float weight_chosen = def(0.f);
        Float threshold_chosen = def(0.f);
        Bool  found = def(false);
        Float last_weight = def(0.f);
        Float last_threshold = def(0.f);
        UInt  last_chosen = def(0u);
        Bool  any_valid = def(false);

        // --- Picker: slots 0-4 ---
        {
            Float w = lobe_list.weights[0];
            threshold += w;
            last_weight = w; last_threshold = threshold; last_chosen = 0u;
            any_valid = true;
            $if(!found & (r < threshold)) {
                chosen = 0u; weight_chosen = w; threshold_chosen = threshold; found = true;
            };
        };
        $if(1u < lobe_list.count) {
            Float w = lobe_list.weights[1];
            threshold += w;
            last_weight = w; last_threshold = threshold; last_chosen = 1u;
            $if(!found & (r < threshold)) {
                chosen = 1u; weight_chosen = w; threshold_chosen = threshold; found = true;
            };
        };
        $if(2u < lobe_list.count) {
            Float w = lobe_list.weights[2];
            threshold += w;
            last_weight = w; last_threshold = threshold; last_chosen = 2u;
            $if(!found & (r < threshold)) {
                chosen = 2u; weight_chosen = w; threshold_chosen = threshold; found = true;
            };
        };
        $if(3u < lobe_list.count) {
            Float w = lobe_list.weights[3];
            threshold += w;
            last_weight = w; last_threshold = threshold; last_chosen = 3u;
            $if(!found & (r < threshold)) {
                chosen = 3u; weight_chosen = w; threshold_chosen = threshold; found = true;
            };
        };
        $if(4u < lobe_list.count) {
            Float w = lobe_list.weights[4];
            threshold += w;
            last_weight = w; last_threshold = threshold; last_chosen = 4u;
            $if(!found & (r < threshold)) {
                chosen = 4u; weight_chosen = w; threshold_chosen = threshold; found = true;
            };
        };
#if NT_ENABLE_SHEEN_SAMPLING
        $if(5u < lobe_list.count) {
            Float w = lobe_list.weights[5];
            threshold += w;
            last_weight = w; last_threshold = threshold; last_chosen = 5u;
            $if(!found & (r < threshold)) {
                chosen = 5u; weight_chosen = w; threshold_chosen = threshold; found = true;
            };
        };
#endif

        $if(!found & any_valid) {
            chosen = last_chosen;
            weight_chosen = last_weight;
            threshold_chosen = last_threshold;
        };

        Float threshold_before = threshold_chosen - weight_chosen;
        Float r_offset = clamp(r - threshold_before, 0.f, weight_chosen);
        Float2 u_local = make_float2(r_offset / max(weight_chosen, 1e-6f), u.y);

        Float lobe_pdf_sink = def(0.f);

        $if(chosen == 0u) {
            $if(lobe_list.count == 1u) {
                $if(bsdf_type == 11u) {
                    ThinDielectricBSDF thin{albedo, roughness, ior};
                    wi = thin.sample(wo, normal, u_local, lobe_pdf_sink);
                }
                $else {
                    DielectricBSDF di{attenuation_val, roughness, ior};
                    Bool discard_is_tx;
                    wi = di.sample(wo, normal, u_local, lobe_pdf_sink, discard_is_tx);
                };
            }
            $else {
                Float aspect = sqrt(max(1.f - anisotropic_val * 0.9f, 0.1f));
                Float2 alpha = roughness_to_alpha(make_float2(
                    max(roughness / aspect, 1e-4f),
                    max(roughness * aspect, 1e-4f)));
                // t_rot hoisted to construction time (t_rot_val, item 10).
                MicrofacetBSDF spec{albedo, alpha, metallic, ior,
                                    iridescence_val, iridescence_ior_val, iridescence_thickness_val,
                                    t_rot_val, bitangent_sign,
                                    attenuation_val, conductor_k_val};
                wi = spec.sample(wo, normal, u_local, lobe_pdf_sink);
            };
        }
        $elif(chosen == 1u) {
            LambertianBSDF diffuse{albedo};
            wi = diffuse.sample(wo, normal, u_local, lobe_pdf_sink);
        }
        $elif(chosen == 2u) {
            FabricDiffuseBSDF fabric{albedo};
            wi = fabric.sample(wo, normal, u_local, lobe_pdf_sink);
        }
        $elif(chosen == 3u) {
            Float3 sss_tx_color = albedo * exp(-attenuation_val / max(attenuation_distance_val, 1e-4f));
            SubsurfaceBSDF sss{albedo * attenuation_val, sss_tx_color,
                                ite(flatness_val > 0.f, diffuse_trans_val, 0.f), ior};
            wi = sss.sample(wo, normal, u_local, lobe_pdf_sink);
        }
        $elif(chosen == 4u) {
            MicrofacetTransmissionBSDF tx{
                attenuation_val,
                roughness_to_alpha(make_float2(roughness)),
                1.f / ior, ior};
            Bool discard_is_tx;
            wi = tx.sample(wo, normal, u_local, lobe_pdf_sink, discard_is_tx);
        }
#if NT_ENABLE_SHEEN_SAMPLING
        $elif(chosen == 5u) {
            SheenBSDF sheen{1.0f, sheen_tint_val, albedo, roughness};
            wi = sheen.sample(wo, normal, u_local, lobe_pdf_sink);
        }
#endif
        ;

        // Mixture PDF
        $if(lobe_list.weights[0] > 0.f) {
            $if(lobe_list.count == 1u) {
                $if(bsdf_type == 11u) {
                    ThinDielectricBSDF thin{albedo, roughness, ior};
                    p += lobe_list.weights[0] * thin.pdf(wo, wi, normal);
                }
                $else {
                    DielectricBSDF di{attenuation_val, roughness, ior};
                    Float di_pdf = di.pdf(wo, wi, normal);
                    Bool is_tx = dot(wo, normal) * dot(wi, normal) < 0.0f;
                    p += lobe_list.weights[0] * ite(is_tx, di_pdf * specular_trans_val, di_pdf);
                };
            }
            $else {
                Float aspect = sqrt(max(1.f - anisotropic_val * 0.9f, 0.1f));
                Float2 alpha = roughness_to_alpha(make_float2(
                    max(roughness / aspect, 1e-4f),
                    max(roughness * aspect, 1e-4f)));
                // t_rot hoisted to construction time (t_rot_val, item 10).
                MicrofacetBSDF spec{albedo, alpha, metallic, ior,
                                    iridescence_val, iridescence_ior_val, iridescence_thickness_val,
                                    t_rot_val, bitangent_sign,
                                    attenuation_val, conductor_k_val};
                p += lobe_list.weights[0] * spec.pdf(wo, wi, normal);
            };
        };
        $if(1u < lobe_list.count & lobe_list.weights[1] > 0.f) {
            LambertianBSDF diffuse{albedo};
            p += lobe_list.weights[1] * diffuse.pdf(wo, wi, normal);
        };
        $if(2u < lobe_list.count & lobe_list.weights[2] > 0.f) {
            FabricDiffuseBSDF fabric{albedo};
            p += lobe_list.weights[2] * fabric.pdf(wo, wi, normal);
        };
        $if(3u < lobe_list.count & lobe_list.weights[3] > 0.f) {
            Float3 sss_tx_color = albedo * exp(-attenuation_val / max(attenuation_distance_val, 1e-4f));
            SubsurfaceBSDF sss{albedo * attenuation_val, sss_tx_color,
                                ite(flatness_val > 0.f, diffuse_trans_val, 0.f), ior};
            p += lobe_list.weights[3] * sss.pdf(wo, wi, normal);
        };
        $if(4u < lobe_list.count & lobe_list.weights[4] > 0.f) {
            MicrofacetTransmissionBSDF tx{
                attenuation_val,
                roughness_to_alpha(make_float2(roughness)),
                1.f / ior, ior};
            p += lobe_list.weights[4] * tx.pdf(wo, wi, normal);
        };
#if NT_ENABLE_SHEEN_SAMPLING
        $if(5u < lobe_list.count & lobe_list.weights[5] > 0.f) {
            SheenBSDF sheen{1.0f, sheen_tint_val, albedo, roughness};
            p += lobe_list.weights[5] * sheen.pdf(wo, wi, normal);
        };
        // True sampling density is mixture/pool_sum (r was rescaled above).
        p = p / max(pool_sum_sl, 1e-6f);
#endif
    };

    out_pdf = p;
    return wi;
}

// (dispatch_lobe_pdf removed in Option A slot-specialized refactor.
//  Per-slot pdf now inlined directly in MaterialBSDF::pdf.)

Float MaterialBSDF::pdf(
    Expr<float3> wo,
    Expr<float3> wi,
    Expr<float3> normal) const noexcept {
    Float p = def(0.f);

    $if(has_composed_lobe_list) {
        // ==============================================================
        // Phase 2D layered path. Sum pdf over sampling-pool slots 1-5
        // (plus slot 0 under NT_ENABLE_COAT_SAMPLING — must mirror the
        // mixture pdf in sample() so f/p stays consistent).
        // ==============================================================
        using UL = LobeType;

#if NT_ENABLE_COAT_SAMPLING || NT_ENABLE_SHEEN_SAMPLING
        // Same pool sum as sample()'s picker — the mixture below must be
        // divided by it to report the true sampling density.
        Float pool_sum = def(0.f);
#if NT_ENABLE_COAT_SAMPLING
        // Fresnel-gated coat pool weight — MUST be the same expression as
        // sample()'s (weights[0] * coat_F12) or f/pdf goes biased.
        Float coat_pool_w = lobe_list.weights[0] * coat_F12;
        pool_sum += coat_pool_w;
#endif
        pool_sum += lobe_list.weights[1];
        pool_sum += lobe_list.weights[2];
        pool_sum += lobe_list.weights[3];
        pool_sum += lobe_list.weights[4];
        pool_sum += lobe_list.weights[5];
#if NT_ENABLE_SHEEN_SAMPLING
        pool_sum += lobe_list.weights[6];
        pool_sum += lobe_list.weights[8];
#endif
#endif

        // Slot 0: coat
#if NT_ENABLE_COAT_SAMPLING
        $if(lobe_list.weights[0] > 0.f) {
            #if NT_ENABLE_TWO_INTERFACE_FRESNEL
            $if(coat_bsdf_type == 3u) {
                Float2 coat_alpha = roughness_to_alpha(make_float2(coat_roughness));
                MicrofacetBSDF coat_microfacet{
                    make_float3(1.f), coat_alpha, 0.f, coat_ior,
                    0.f, 1.3f, 0.f,
                    tangent_dir, bitangent_sign,
                    attenuation_val, conductor_k_val};
                p += coat_pool_w * coat_microfacet.pdf(wo, wi, normal);
            } $else {
                ClearcoatBSDF cc{1.0f, coat_clearcoat_gloss_val};
                p += coat_pool_w * cc.pdf(wo, wi, normal);
            };
            #else
            ClearcoatBSDF cc{1.0f, coat_clearcoat_gloss_val};
            p += coat_pool_w * cc.pdf(wo, wi, normal);
            #endif
        };
#endif

        // Slot 1: SpecularMetal OR DeltaDielectric
        $if(lobe_list.weights[1] > 0.f) {
            UInt t1 = lobe_type(lobe_list.type_flags[1]);
            $if(t1 == static_cast<uint>(UL::SpecularMetal)) {
                Float aspect = sqrt(max(1.f - anisotropic_val * 0.9f, 0.1f));
                Float2 alpha = roughness_to_alpha(make_float2(
                    max(roughness / aspect, 1e-4f),
                    max(roughness * aspect, 1e-4f)));
                // t_rot hoisted to construction time (t_rot_val, item 10).
                MicrofacetBSDF spec{albedo, alpha, metallic, ior,
                                    iridescence_val, iridescence_ior_val, iridescence_thickness_val,
                                    t_rot_val, bitangent_sign,
                                    attenuation_val, conductor_k_val};
                p += lobe_list.weights[1] * spec.pdf(wo, wi, normal);
            }
            $elif(t1 == static_cast<uint>(UL::DeltaDielectric)) {
                $if(bsdf_type == 11u) {
                    ThinDielectricBSDF thin{albedo, roughness, ior};
                    p += lobe_list.weights[1] * thin.pdf(wo, wi, normal);
                }
                $else {
                    DielectricBSDF di{attenuation_val, roughness, ior};
                    Float di_pdf = di.pdf(wo, wi, normal);
                    Bool is_tx = dot(wo, normal) * dot(wi, normal) < 0.0f;
                    p += lobe_list.weights[1] * ite(is_tx, di_pdf * specular_trans_val, di_pdf);
                };
            };
        };

        // Slot 2: Diffuse
        $if(lobe_list.weights[2] > 0.f) {
            LambertianBSDF diffuse{albedo};
            p += lobe_list.weights[2] * diffuse.pdf(wo, wi, normal);
        };

        // Slot 3: Fabric
        $if(lobe_list.weights[3] > 0.f) {
            FabricDiffuseBSDF fabric{albedo};
            p += lobe_list.weights[3] * fabric.pdf(wo, wi, normal);
        };

        // Slot 4: Subsurface
        $if(lobe_list.weights[4] > 0.f) {
            Float3 sss_tx_color = albedo * exp(-attenuation_val / max(attenuation_distance_val, 1e-4f));
            SubsurfaceBSDF sss{albedo * attenuation_val, sss_tx_color,
                                ite(flatness_val > 0.f, diffuse_trans_val, 0.f), ior};
            p += lobe_list.weights[4] * sss.pdf(wo, wi, normal);
        };

        // Slot 5: Transmission
        $if(lobe_list.weights[5] > 0.f) {
            MicrofacetTransmissionBSDF tx{
                attenuation_val,
                roughness_to_alpha(make_float2(roughness)),
                1.f / ior, ior};
            p += lobe_list.weights[5] * tx.pdf(wo, wi, normal);
        };
#if NT_ENABLE_SHEEN_SAMPLING
        // Slots 6 & 8: sheen (base's own + fuzz) — mirror sample()'s mixture
        $if(lobe_list.weights[6] > 0.f) {
            SheenBSDF sheen{1.0f, sheen_tint_val, albedo, roughness};
            p += lobe_list.weights[6] * sheen.pdf(wo, wi, normal);
        };
        $if(lobe_list.weights[8] > 0.f) {
            SheenBSDF sheen{1.0f, fuzz_sheen_tint_val, fuzz_albedo, fuzz_sheen_roughness_val};
            p += lobe_list.weights[8] * sheen.pdf(wo, wi, normal);
        };
#endif
#if NT_ENABLE_COAT_SAMPLING || NT_ENABLE_SHEEN_SAMPLING
        // Mirror sample(): density is mixture/pool_sum
        p = p / max(pool_sum, 1e-6f);
#endif
    }
    $else {
        // ==============================================================
        // Phase 1 slot-specialized path (single-layer).
        // ==============================================================
#if NT_ENABLE_SHEEN_SAMPLING
        // Mirror sample()'s single-layer pool sum (weights 0-5).
        Float pool_sum_sl = def(0.f);
        pool_sum_sl += lobe_list.weights[0];
        pool_sum_sl += lobe_list.weights[1];
        pool_sum_sl += lobe_list.weights[2];
        pool_sum_sl += lobe_list.weights[3];
        pool_sum_sl += lobe_list.weights[4];
        pool_sum_sl += lobe_list.weights[5];
#endif
        // --- Delta dielectric path (count==1, slot 0 only) ---
        $if(lobe_list.count == 1u) {
            Float w0 = lobe_list.weights[0];
            $if(w0 > 0.f) {
                $if(bsdf_type == 11u) {
                    ThinDielectricBSDF thin{albedo, roughness, ior};
                    p = w0 * thin.pdf(wo, wi, normal);
                }
                $else {
                    DielectricBSDF di{attenuation_val, roughness, ior};
                    Float di_pdf = di.pdf(wo, wi, normal);
                    Bool is_tx = dot(wo, normal) * dot(wi, normal) < 0.0f;
                    p = w0 * ite(is_tx, di_pdf * specular_trans_val, di_pdf);
                };
            };
        }
        // --- Standard layered path ---
        $else {
            $if(lobe_list.weights[0] > 0.f) {
                Float aspect = sqrt(max(1.f - anisotropic_val * 0.9f, 0.1f));
                Float2 alpha = roughness_to_alpha(make_float2(
                    max(roughness / aspect, 1e-4f),
                    max(roughness * aspect, 1e-4f)));
                // t_rot hoisted to construction time (t_rot_val, item 10).
                MicrofacetBSDF spec{albedo, alpha, metallic, ior,
                                    iridescence_val, iridescence_ior_val, iridescence_thickness_val,
                                    t_rot_val, bitangent_sign,
                                    attenuation_val, conductor_k_val};
                p += lobe_list.weights[0] * spec.pdf(wo, wi, normal);
            };

            $if(lobe_list.weights[1] > 0.f) {
                LambertianBSDF diffuse{albedo};
                p += lobe_list.weights[1] * diffuse.pdf(wo, wi, normal);
            };
            $if(lobe_list.weights[2] > 0.f) {
                FabricDiffuseBSDF fabric{albedo};
                p += lobe_list.weights[2] * fabric.pdf(wo, wi, normal);
            };
            $if(lobe_list.weights[3] > 0.f) {
                Float3 sss_tx_color = albedo * exp(-attenuation_val / max(attenuation_distance_val, 1e-4f));
                SubsurfaceBSDF sss{albedo * attenuation_val, sss_tx_color,
                                    ite(flatness_val > 0.f, diffuse_trans_val, 0.f), ior};
                p += lobe_list.weights[3] * sss.pdf(wo, wi, normal);
            };
            $if(lobe_list.weights[4] > 0.f) {
                MicrofacetTransmissionBSDF tx{
                    attenuation_val,
                    roughness_to_alpha(make_float2(roughness)),
                    1.f / ior, ior};
                p += lobe_list.weights[4] * tx.pdf(wo, wi, normal);
            };
#if NT_ENABLE_SHEEN_SAMPLING
            $if(lobe_list.count > 5u & lobe_list.weights[5] > 0.f) {
                SheenBSDF sheen{1.0f, sheen_tint_val, albedo, roughness};
                p += lobe_list.weights[5] * sheen.pdf(wo, wi, normal);
            };
#endif
        };
#if NT_ENABLE_SHEEN_SAMPLING
        // Mirror sample(): density is mixture/pool_sum (1 on the delta path).
        p = p / max(pool_sum_sl, 1e-6f);
#endif
    };
    return p;
}

//==============================================================================
// Microfacet Transmission BSDF (Walter 2007)
//==============================================================================

Float3 MicrofacetTransmissionBSDF::evaluate(
    Expr<float3> wo,
    Expr<float3> wi,
    Expr<float3> normal) const noexcept {

    Float3 result = def(make_float3(0.0f));

    Float3x3 tnb = make_orthonormal_basis(normal);
    Float3 wo_local = transpose(tnb) * wo;
    Float3 wi_local = transpose(tnb) * wi;

    // Transmission: wo and wi must be on opposite sides of the surface
    $if(!same_hemisphere(wo_local, wi_local)) {
        // Compute refracted half-vector: wh = normalize(wo + eta * wi)
        Float3 wh = normalize(wo_local + eta * wi_local);

        // Ensure wh is on the same side as wo
        wh = ite(dot(wo_local, wh) < 0.0f, -wh, wh);

        Float F = fresnel_dielectric(abs_dot(wo_local, wh), 1.0f, ior);
        Float D = ggx_distribution(wh, alpha);
        Float G = ggx_G(wo_local, wi_local, alpha);

        Float dot_wo_wh = dot(wo_local, wh);
        Float dot_wi_wh = dot(wi_local, wh);

        // Walter 2007 BTDF: (1-F) * D * G * |wo.wh| * |wi.wh| / (|wo.n| * |wi.n| * (wo.wh + eta*wi.wh)^2)
        Float denom = dot_wo_wh + eta * dot_wi_wh;
        Float btdf = (1.0f - F) * D * G * abs(dot_wo_wh) * abs(dot_wi_wh) /
                     (abs_cos_theta(wo_local) * abs_cos_theta(wi_local) * denom * denom);

        result = albedo * btdf;
    };

    return result;
}

Float3 MicrofacetTransmissionBSDF::sample(
    Expr<float3> wo,
    Expr<float3> normal,
    Expr<float2> u,
    Float& out_pdf,
    Bool& out_is_transmission) const noexcept {

    Float3x3 tnb = make_orthonormal_basis(normal);
    Float3 wo_local = transpose(tnb) * wo;

    // Sample microfacet normal
    Float3 wh_local = sample_ggx_wh(wo_local, alpha, u);
    Float F = fresnel_dielectric(abs_dot(wo_local, wh_local), 1.0f, ior);

    Float3 wi_local = def(make_float3(0.0f));
    Float pdf_reflect = def(0.0f);
    Float pdf_transmit = def(0.0f);
    Bool is_transmission = def(false);

    $if(u.x < F) {
        // Reflection lobe
        wi_local = reflect(-wo_local, wh_local);
        is_transmission = false;
    }
    $else {
        // Transmission lobe — refract through microfacet
        Float cos_i = dot(wo_local, wh_local);
        Float sin2_t = eta * eta * (1.0f - cos_i * cos_i);
        $if(sin2_t < 1.0f) {
            wi_local = refract_dir(-wo_local, wh_local, eta);
            // Make sure wi is on the opposite side from wo
            wi_local = ite(same_hemisphere(wo_local, wi_local), -wi_local, wi_local);
        }
        $else {
            // Total internal reflection
            wi_local = reflect(-wo_local, wh_local);
            is_transmission = false;
        };
        is_transmission = sin2_t < 1.0f;
    };

    // Compute PDFs for both lobes
    Float d_wh = ggx_distribution(wh_local, alpha);
    Float g1_wo = ggx_G1(wo_local, alpha);
    Float pdf_wh = d_wh * g1_wo * abs_dot(wo_local, wh_local) / abs_cos_theta(wo_local);

    Float pdf_r = pdf_wh / (4.0f * abs_dot(wo_local, wh_local));

    // Transmission PDF: pdf_wh * |wi.wh| * eta^2 / (wo.wh + eta*wi.wh)^2
    Float dot_wi_wh = abs_dot(wi_local, wh_local);
    Float dot_wo_wh = abs_dot(wo_local, wh_local);
    Float denom_t = dot_wo_wh + eta * dot_wi_wh;
    Float pdf_t = pdf_wh * dot_wi_wh * eta * eta / max(denom_t * denom_t, 1e-10f);

    out_pdf = F * pdf_r + (1.0f - F) * pdf_t;
    out_is_transmission = is_transmission;

    Float3 wi = tnb * wi_local;
    return wi;
}

Float MicrofacetTransmissionBSDF::pdf(
    Expr<float3> wo,
    Expr<float3> wi,
    Expr<float3> normal) const noexcept {

    Float3x3 tnb = make_orthonormal_basis(normal);
    Float3 wo_local = transpose(tnb) * wo;
    Float3 wi_local = transpose(tnb) * wi;

    Float p = def(0.0f);

    Float3 wh = normalize(wo_local + eta * wi_local);
    wh = ite(dot(wo_local, wh) < 0.0f, -wh, wh);

    Float F = fresnel_dielectric(abs_dot(wo_local, wh), 1.0f, ior);

    // Reflection PDF
    Float pdf_r = def(0.0f);
    $if(same_hemisphere(wo_local, wi_local)) {
        Float3 wh_r = normalize(wo_local + wi_local);
        pdf_r = ggx_pdf(wo_local, wh_r, alpha) / (4.0f * abs_dot(wo_local, wh_r));
    };

    // Transmission PDF
    Float pdf_t = def(0.0f);
    $if(!same_hemisphere(wo_local, wi_local)) {
        Float dot_wo_wh = abs_dot(wo_local, wh);
        Float dot_wi_wh = abs_dot(wi_local, wh);
        Float denom = dot_wo_wh + eta * dot_wi_wh;
        pdf_t = ggx_pdf(wo_local, wh, alpha) * dot_wi_wh * eta * eta / max(denom * denom, 1e-10f);
    };

    p = F * pdf_r + (1.0f - F) * pdf_t;
    return p;
}

//==============================================================================
// Dielectric BSDF (Combined Reflection + Transmission)
//==============================================================================

Float3 DielectricBSDF::evaluate(
    Expr<float3> wo,
    Expr<float3> wi,
    Expr<float3> normal) const noexcept {

    MicrofacetBSDF reflection{albedo, roughness_to_alpha(make_float2(roughness)), 0.0f, ior};
    MicrofacetTransmissionBSDF transmission{albedo, roughness_to_alpha(make_float2(roughness)),
                                             1.0f / ior, ior};

    Float3 F_eval = reflection.evaluate(wo, wi, normal);
    Float3 T_eval = transmission.evaluate(wo, wi, normal);

    return F_eval + T_eval;
}

Float3 DielectricBSDF::sample(
    Expr<float3> wo,
    Expr<float3> normal,
    Expr<float2> u,
    Float& out_pdf,
    Bool& out_is_transmission) const noexcept {

    MicrofacetTransmissionBSDF bsdf{albedo, roughness_to_alpha(make_float2(roughness)),
                                    1.0f / ior, ior};
    return bsdf.sample(wo, normal, u, out_pdf, out_is_transmission);
}

Float DielectricBSDF::pdf(
    Expr<float3> wo,
    Expr<float3> wi,
    Expr<float3> normal) const noexcept {

    MicrofacetTransmissionBSDF bsdf{albedo, roughness_to_alpha(make_float2(roughness)),
                                    1.0f / ior, ior};
    return bsdf.pdf(wo, wi, normal);
}

} // namespace newtype::render
