#pragma once

// Burley 2015 normalized-diffusion BSSRDF (header-only DSL free functions).
//
// Reference math (verified against D:/Projects/RTX_study/ReSTIR-SSS/.../sss_diffusion_profile_burley.glsl):
//   scatter_distance   s = 1.9 - albedo + 3.5 * (albedo - 0.8)^2          // per RGB channel
//   diffusion profile  R(r, s) = (e^(-r/s) + e^(-r/(3s))) / (8*pi*s*r)
//   radial CDF         F(r) = 1 - 1/4*e^(-r/s) - 3/4*e^(-r/(3s))           // integral of 2*pi*r*R(r) dr = 1
//   analytical inverse (zero-radiance blog):
//       u  = 1 - xi                                                      // CCDF
//       g  = 1 + 4u*(2u + sqrt(1 + 4u^2))
//       n  = exp2(log2(g) * -1/3)                                        // g^(-1/3)
//       p  = (g*n)*n                                                     // g^(+1/3)
//       c  = 1 + p + n
//       x  = (3/LOG2e) * log2(c/(4u))
//       r  = x * s                                                       // multiply (NOT divide — Bug 1)
//
// Polar-to-area Jacobian: sampling (r, theta) with pdf f(r)=2*pi*r*R(r) and theta~U[0,2*pi)
// gives an *area* pdf of R(r) per unit surface area (the r*dtheta Jacobian cancels the 2*pi*r).
// This is what lets R(r_p) cancel between the BSSRDF numerator S and the sampling-pdf denominator.
//
// These are out-of-line free functions (NOT Callables) — single dispatch site per
// feedback-luisa-callable-compile-time.md.

#include <luisa/dsl/sugar.h>

namespace newtype::render {

using namespace luisa;
using namespace luisa::compute;

// Per-RGB-channel Burley scatter distance s = 1.9 - albedo + 3.5*(albedo-0.8)^2.
// Use scalar broadcast `albedo - 0.8f` (NOT make_float3(0.8f)) — Cinder vec3 / Luisa float3
// ambiguity resolves cleanly when broadcasting a scalar.
[[nodiscard]] Float3 bssrdf_scatter_distance(Expr<float3> albedo) noexcept;

// 2D diffusion profile R(r, s). Normalized: integral of 2*pi*r*R(r) dr over [0,inf) = 1.
[[nodiscard]] Float bssrdf_profile_eval(Expr<float> r, Expr<float> s) noexcept;

// Analytical inverse of F(r) = xi. Returns r = x * s (NOT x/s — Bug 1 fix).
[[nodiscard]] Float bssrdf_sample_radius(Expr<float> xi, Expr<float> s) noexcept;

} // namespace newtype::render
