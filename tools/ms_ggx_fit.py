#!/usr/bin/env python3
"""Kulla-Conty fit-validation harness (plan imperative-coalescing-penguin.md, Phase 1).

Computes reference single-scattering directional albedo E(mu_o, alpha) for GGX + Smith
(height-correlated) with F == 1, by VNDF importance sampling — a direct port of the
engine's sample_ggx_wh / ggx_G math (src/newtype/render/BSDF.cpp:138-223).

Then:
  1. Validates the UE4/Karis EnvBRDF fit (F0=1 -> E = A + B) against reference.
  2. Fits a polynomial for E_avg(alpha) = 2*int_0^1 E(mu,alpha) mu dmu.
  3. Furnace check: F_avg*E_ss + E_ms  should approximate  F_avg.

Usage: python tools/ms_ggx_fit.py
"""

import numpy as np

RNG = np.random.default_rng(42)
PI = np.pi

# Engine conventions (BSDF.h:135-149): roughness clamped to kMinRoughness=0.03, alpha = r^2.
K_MIN_ROUGHNESS = 0.03


# ----------------------------------------------------------------------------
# Reference BSDF math (ports of BSDF.cpp ggx_distribution / ggx_lambda / ggx_G)
# All inputs/outputs in local frame, normal = +z. w vectors are (..., 3).
# ----------------------------------------------------------------------------

def ggx_distribution(wh, alpha):
    cos4 = wh[..., 2] ** 4
    tan2 = (wh[..., 0] ** 2 + wh[..., 1] ** 2) / np.maximum(wh[..., 2] ** 2, 1e-20)
    # isotropic alpha (scalar)
    e = tan2 / alpha**2
    d = 1.0 / (PI * alpha * alpha * cos4 * (1.0 + e) ** 2)
    return np.where(tan2 > 1e30, 0.0, d)


def ggx_lambda(w, alpha):
    tan_theta = np.sqrt(w[..., 0] ** 2 + w[..., 1] ** 2) / np.maximum(np.abs(w[..., 2]), 1e-20)
    alpha2_tan2 = alpha**2 * tan_theta**2
    return (-1.0 + np.sqrt(1.0 + alpha2_tan2)) * 0.5


def ggx_g1(w, alpha):
    return 1.0 / (1.0 + ggx_lambda(w, alpha))


def ggx_g(wo, wi, alpha):
    return 1.0 / (1.0 + ggx_lambda(wo, alpha) + ggx_lambda(wi, alpha))


def sample_ggx_wh(wo, alpha, u1, u2):
    """Port of BSDF.cpp sample_ggx_wh (isotropic alpha). wo/u are broadcastable arrays."""
    sx = alpha * wo[..., 0]
    sy = alpha * wo[..., 1]
    sz = np.broadcast_to(wo[..., 2], sx.shape)
    wo_stretched = np.stack([sx, sy, sz], axis=-1)
    n = np.linalg.norm(wo_stretched, axis=-1, keepdims=True)
    wo_stretched = wo_stretched / n

    cos_theta = wo_stretched[..., 2]
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
    tan_theta = sin_theta / cos_theta

    # General case (BSDF.cpp:187-209). Vectorized branchless selection.
    a = 1.0 / tan_theta
    g1 = 2.0 / (1.0 + np.sqrt(1.0 + 1.0 / a**2))

    A = 2.0 * u1 / g1 - 1.0
    with np.errstate(divide="ignore"):
        tmp = np.minimum(1.0 / (A**2 - 1.0), 1e10)  # engine port: negative for |A|<1
    B = tan_theta
    D = np.sqrt(np.maximum((B * tmp) ** 2 - (A**2 - B**2) * tmp, 0.0))
    slope_x_1 = B * tmp - D
    slope_x_2 = B * tmp + D
    slope_x = np.where((A < 0.0) | (slope_x_2 * tan_theta > 1.0), slope_x_1, slope_x_2)

    S = np.where(u2 > 0.5, 1.0, -1.0)
    U2 = np.where(u2 > 0.5, 2.0 * (u2 - 0.5), 2.0 * (0.5 - u2))
    z = (U2 * (U2 * (U2 * 0.27385 - 0.73369) + 0.46341)) / (
        U2 * (U2 * (U2 * 0.093073 + 0.309420) - 1.0) + 0.597999)
    slope_y = S * z * np.sqrt(1.0 + slope_x**2)

    # Rotate by phi of stretched wo (BSDF.cpp:213-217)
    phi_s = np.arctan2(wo_stretched[..., 1], wo_stretched[..., 0])
    sx = np.cos(phi_s) * slope_x - np.sin(phi_s) * slope_y
    sy = np.sin(phi_s) * slope_x + np.cos(phi_s) * slope_y
    sx, sy = alpha * sx, alpha * sy

    wh = np.stack([-sx, -sy, np.ones_like(sx)], axis=-1)
    return wh / np.linalg.norm(wh, axis=-1, keepdims=True)


def e_reference(mu_o, alpha, n_samples=1 << 16):
    """E(mu_o, alpha) = int f_ss(wi) mu_i dwi, F==1, VNDF importance sampled."""
    wo = np.stack([np.sqrt(np.maximum(0.0, 1.0 - mu_o**2)), np.zeros_like(mu_o), mu_o], axis=-1)

    u1 = RNG.random((*np.shape(mu_o), n_samples))
    u2 = RNG.random((*np.shape(mu_o), n_samples))
    wo_b = wo[..., None, :]
    alpha_b = np.broadcast_to(np.asarray(alpha)[..., None], np.shape(u1))

    wh = sample_ggx_wh(wo_b, alpha_b, u1, u2)
    wi = 2.0 * np.sum(wo_b * wh, axis=-1, keepdims=True) * wh - wo_b  # reflect(wo, wh)
    mu_i = np.clip(wi[..., 2], 0.0, None)

    dot_wo_wh = np.sum(wo_b * wh, axis=-1)
    d = ggx_distribution(wh, alpha_b)
    g = ggx_g(wo_b, wi, alpha_b)
    f_ss = d * g / (4.0 * np.maximum(mu_o[..., None], 1e-6) * np.maximum(mu_i, 1e-6))

    pdf_wh = d * ggx_g1(wo_b, alpha_b) * dot_wo_wh / np.maximum(mu_o[..., None], 1e-6)
    pdf_wi = pdf_wh / (4.0 * np.maximum(np.abs(dot_wo_wh), 1e-9))

    contrib = f_ss * mu_i / np.maximum(pdf_wi, 1e-12)
    valid = (mu_i > 1e-6) & (dot_wo_wh > 1e-6)
    contrib = np.where(valid, contrib, 0.0)
    stderr = contrib.std(axis=-1) / np.sqrt(contrib.shape[-1])
    return contrib.mean(axis=-1), stderr


# ----------------------------------------------------------------------------
# Candidate analytic fits
# ----------------------------------------------------------------------------

def e_envbrdf(mu, roughness):
    """UE4 Karis split-integral fit, F0=1 -> E = A + B. mu/roughness same-shape arrays."""
    c0 = np.array([-1.0, -0.0275, -0.572, 0.022])
    c1 = np.array([1.0, 0.0425, 1.04, -0.04])
    r = roughness[..., None] * c0 + c1              # (..., 4)
    a004 = np.minimum(r[..., 0] ** 2, np.exp2(-9.28 * mu)) * mu + r[..., 1]
    a = -1.04 * a004 + r[..., 2]
    b = 1.04 * a004 + r[..., 3]
    return a + b


def e_avg_integrand(mu_grid, e_of_mu):
    return 2.0 * mu_grid * e_of_mu


def cheb_basis(x, deg):
    """Chebyshev T_0..T_deg on x in [-1, 1]."""
    ts = [np.ones_like(x), x]
    for _ in range(2, deg + 1):
        ts.append(2.0 * x * ts[-1] - ts[-2])
    return ts[: deg + 1]


def fit_cheb_2d(mu, x, g, deg_mu=7, deg_x=7, x_lo=None, x_hi=None):
    """Tensor Chebyshev fit of G(mu, x); mu,x mapped to [-1,1]. Returns coeffs + fit."""
    x_lo = x.min() if x_lo is None else x_lo
    x_hi = x.max() if x_hi is None else x_hi
    u = 2.0 * mu - 1.0
    v = 2.0 * (x - x_lo) / (x_hi - x_lo) - 1.0
    bu, bv = cheb_basis(u, deg_mu), cheb_basis(v, deg_x)
    design = [a * b for a in bu for b in bv]
    A_mat = np.stack([t.ravel() for t in design], axis=-1)
    coeffs, *_ = np.linalg.lstsq(A_mat, g.ravel(), rcond=None)
    fit = (A_mat @ coeffs).reshape(g.shape)
    return coeffs.reshape(deg_mu + 1, deg_x + 1), fit, (x_lo, x_hi)


def eval_e_2d(coeffs, mu, alpha):
    out = np.zeros_like(mu)
    for i in range(coeffs.shape[0]):
        for j in range(coeffs.shape[1]):
            out = out + coeffs[i, j] * mu**i * alpha**j
    return out


def fit_report():
    # Grids matching engine domain: roughness r in [0.03, 1], alpha = r^2 (log spaced),
    # mu_o in (0,1]. Shapes: (n_alpha, n_mu).
    r_grid = np.geomspace(K_MIN_ROUGHNESS, 1.0, 24)
    alpha_grid = r_grid**2
    mu_grid = np.concatenate([np.geomspace(0.02, 0.99, 23), [1.0]])  # 24

    aa, mm = np.meshgrid(alpha_grid, mu_grid, indexing="ij")
    rr, _ = np.meshgrid(r_grid, mu_grid, indexing="ij")

    print("Computing reference E (24x24 grid, 262144 samples each)...")
    e_ref, stderr = e_reference(mm, aa, n_samples=1 << 18)
    print(f"    max MC stderr = {stderr.max():.5f}  mean = {stderr.mean():.5f}")

    # --- 1. EnvBRDF fit error (gate check) ---
    e_fit = e_envbrdf(mm, rr)
    abs_err = np.abs(e_fit - e_ref)
    rel_err = abs_err / np.maximum(e_ref, 1e-3)
    print(f"\n[1] EnvBRDF (F0=1) vs reference E:")
    print(f"    max abs err = {abs_err.max():.4f}   mean abs err = {abs_err.mean():.4f}")
    print(f"    max rel err = {rel_err.max()*100:.1f}%  mean rel err = {rel_err.mean()*100:.2f}%")

    # --- 2. Own 2D fit for E — fit complement G = 1 - E on a Chebyshev tensor basis ---
    # Power-basis fits blow up (coeffs ~1e7, f32-unsafe). Chebyshev on [-1,1]^2 is stable.
    print("\n[2] Chebyshev tensor fit of G(mu, r) = 1 - E:")
    best = None
    for deg_mu, deg_x in ((6, 6), (7, 7), (8, 8)):
        coeffs, g_p, (x_lo, x_hi) = fit_cheb_2d(mm, rr, 1.0 - e_ref, deg_mu, deg_x)
        err = np.abs(g_p - (1.0 - e_ref))
        print(f"    deg {deg_mu}x{deg_x}: G max abs err = {err.max():.5f}  mean = {err.mean():.5f}"
              f"  max|coeff| = {np.abs(coeffs).max():.1f}")
        if best is None or err.max() < best[0]:
            best = (err.max(), coeffs, (deg_mu, deg_x), (x_lo, x_hi))
    _, coeffs, (deg_mu, deg_x), (x_lo, x_hi) = best
    print(f"    selected deg {deg_mu}x{deg_x}, r in [{x_lo:.4f}, {x_hi:.4f}]. "
          f"Coeffs c[i][j] * T_i(u(mu)) * T_j(v(r)), u/v mapped to [-1,1]:")
    for i in range(coeffs.shape[0]):
        print(f"      i={i}: " + ", ".join(f"{c:+.7f}" for c in coeffs[i]))

    # --- 3. E_avg(alpha) reference + polynomial fits ---
    e_avg_ref = np.trapezoid(e_avg_integrand(mu_grid, e_ref), mu_grid, axis=-1)
    print("\n[3] E_avg poly fits (vs alpha):")
    for deg in (4, 5):
        c = np.polyfit(alpha_grid, e_avg_ref, deg)
        err = np.abs(np.polyval(c, alpha_grid) - e_avg_ref)
        print(f"    deg {deg}: max abs err = {err.max():.5f}  mean = {err.mean():.5f}")
        print(f"      coeffs (highest power first): {', '.join(f'{v:.8f}' for v in c)}")

    # --- 4. Furnace check across f_ms scalar variants ---
    # f_ms = c' * (1-E_o)(1-E_i) / pi integrates to E_ms(mu_o) = c'*(1-E_o)*(1-E_avg).
    #   (a) F^2/(1-F*Eavg)   — commonly-cited colored KC variant (NON-conserving, rejected)
    #   (d) F/(1-Eavg)       — KC-consistent: reduces to published white formula at F=1,
    #                          exact total energy at all F_avg  [SELECTED]
    print("\n[4] Furnace check max |E_total - F_avg| by f_ms scalar variant:")
    cands = {
        "a) F^2/(1-F*Eavg)": lambda f, eav: f * f / np.maximum(1.0 - f * eav, 1e-4),
        "d) F/(1-Eavg)":     lambda f, eav: f / np.maximum(1.0 - eav, 1e-4),
    }
    for f_avg in (1.0, 0.9, 0.7, 0.5, 0.3, 0.04):
        line = f"    F_avg={f_avg:4.2f}: "
        for name, fn in cands.items():
            c_scalar = np.asarray(fn(f_avg, e_avg_ref)).reshape(-1)[:, None]
            e_ms = c_scalar * (1.0 - e_ref) * (1.0 - e_avg_ref[:, None])
            err = np.abs(f_avg * e_ref + e_ms - f_avg).max()
            line += f"{name}={err:.4f}  "
        print(line)

    # --- 5. E_ref table corner values for sanity ---
    print("\n[5] Reference E corners (rows: roughness, cols: mu = 0.05/0.5/1.0):")
    idx = [np.argmin(np.abs(mu_grid - 0.05)), np.argmin(np.abs(mu_grid - 0.5)), -1]
    for i, r in enumerate([0.03, 0.3, 1.0]):
        j = np.argmin(np.abs(r_grid - r))
        print(f"    r={r:4.2f}: " + "  ".join(f"{e_ref[j, k]:.4f}" for k in idx))


if __name__ == "__main__":
    fit_report()
