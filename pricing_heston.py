"""
pricing_heston.py — Heston stochastic volatility pricing (floating barrier).

Extracted from the already-verified Heston dashboard (see thesis chat log
for all correctness checks: DI Call parity, sigma_v/dv stability check,
running-max fix, etc.). No behavior change from that verified version.

Public API:
    mc_heston(...) -> dict            (do_put, do_call, di_call + SE, surv)
    mc_naive_bs(...) -> dict          (BS-dynamics MC, for comparison)
    heston_paths_sample(...) -> (t_arr, S_paths, V_paths)
    heston_npi_live(...) -> dict      (NPI 3D coarse grid + stability flag)
"""

import numpy as np
from numba import njit
from numpy.polynomial.hermite import hermgauss

MIN_RESOLUTION_RATIO = 3.0


def mc_heston(S0, K, alpha, r, kappa, theta, xi, rho, V0, T,
              N=50, M=15_000, seed=42):
    rng = np.random.default_rng(seed)
    b = -np.log(alpha); dt = T / N; disc = np.exp(-r * T)
    sqrt_dt = np.sqrt(dt); sqrt_1mr2 = np.sqrt(max(1.0 - rho**2, 0.0))

    Z1 = rng.standard_normal((M, N))
    Z2 = rng.standard_normal((M, N))
    W_V = Z1
    W_S = rho * Z1 + sqrt_1mr2 * Z2

    x = np.full(M, np.log(S0))
    var = np.full(M, V0)
    x_all = np.zeros((M, N + 1)); x_all[:, 0] = x

    for i in range(N):
        var_p = np.maximum(var, 0.0)
        sv = np.sqrt(var_p)
        x = x + (r - 0.5 * var_p) * dt + sv * sqrt_dt * W_S[:, i]
        var = var + kappa * (theta - var_p) * dt + xi * sv * sqrt_dt * W_V[:, i]
        var = np.maximum(var, 1e-10)
        x_all[:, i + 1] = x

    rm = np.maximum.accumulate(x_all, axis=1)
    dd = rm - x_all
    sv_m = np.all(dd[:, 1:] < b, axis=1)
    ki_m = ~sv_m
    ST = np.exp(x)

    pp = np.maximum(K - ST, 0) * sv_m
    cp = np.maximum(ST - K, 0) * sv_m
    di_cp = np.maximum(ST - K, 0) * ki_m
    do_put = float(disc * np.mean(pp))
    se_put = float(disc * np.std(pp, ddof=1) / np.sqrt(M))
    do_call = float(disc * np.mean(cp))
    se_call = float(disc * np.std(cp, ddof=1) / np.sqrt(M))
    di_call = float(disc * np.mean(di_cp))
    se_di_call = float(disc * np.std(di_cp, ddof=1) / np.sqrt(M))
    surv = float(sv_m.mean())
    return {"do_put": do_put, "se_put": se_put,
            "do_call": do_call, "se_call": se_call,
            "di_call": di_call, "se_di_call": se_di_call, "surv": surv}


def mc_naive_bs(S0, K, alpha, r, sigma, T, N=50, M=15_000, seed=42):
    rng = np.random.default_rng(seed)
    b = -np.log(alpha); dt = T / N; mu = r - 0.5 * sigma**2
    x0 = np.log(S0); disc = np.exp(-r * T)
    Z = rng.standard_normal((M, N))
    xp = x0 + np.cumsum(mu * dt + sigma * np.sqrt(dt) * Z, axis=1)
    xf = np.hstack([np.full((M, 1), x0), xp])
    rm = np.maximum.accumulate(xf, axis=1); dd = rm - xf
    sv = np.all(dd[:, 1:] < b, axis=1)
    pp = np.maximum(K - np.exp(xp[:, -1]), 0) * sv
    do_p = float(disc * np.mean(pp))
    se_p = float(disc * np.std(pp, ddof=1) / np.sqrt(M))
    return {"do_put": do_p, "se_put": se_p, "surv": float(sv.mean())}


def heston_paths_sample(S0, kappa, theta, xi, rho, V0, r, T, N=50,
                         n_paths=20, seed=0):
    rng = np.random.default_rng(seed)
    dt = T / N; sqrt_dt = np.sqrt(dt); sqrt_1mr2 = np.sqrt(max(1-rho**2, 0))
    Z1 = rng.standard_normal((n_paths, N))
    Z2 = rng.standard_normal((n_paths, N))
    W_V = Z1; W_S = rho * Z1 + sqrt_1mr2 * Z2

    t_arr = np.linspace(0, T, N + 1)
    S_all = np.zeros((n_paths, N + 1)); S_all[:, 0] = S0
    var_all = np.zeros((n_paths, N + 1)); var_all[:, 0] = V0
    x = np.full(n_paths, np.log(S0))
    var = np.full(n_paths, V0)
    for i in range(N):
        var_p = np.maximum(var, 0.0); sv = np.sqrt(var_p)
        x = x + (r - 0.5*var_p)*dt + sv*sqrt_dt*W_S[:, i]
        var = var + kappa*(theta - var_p)*dt + xi*sv*sqrt_dt*W_V[:, i]
        var = np.maximum(var, 1e-10)
        S_all[:, i+1] = np.exp(x)
        var_all[:, i+1] = var
    return t_arr, S_all, var_all


def _gauss_hermite_normal(nq):
    nodes, weights = hermgauss(nq)
    return np.sqrt(2.0) * nodes, weights / np.sqrt(np.pi)


@njit(cache=True, fastmath=True)
def _heston_step_mass(mass, xmin, dx, vmax, dv, b, md, gamma,
                       r, kappa, theta, xi, rho, dt, z, w):
    nx, nv, nd = mass.shape
    out = np.zeros_like(mass)
    ki = np.zeros((nx, nv), dtype=mass.dtype)
    sqrt_dt = np.sqrt(dt)
    sqrt1mr2 = np.sqrt(1.0 - rho * rho)
    for ix in range(nx):
        x = xmin + ix * dx
        for iv in range(nv):
            variance = iv * dv
            sqrt_v = np.sqrt(max(variance, 0.0))
            x_drift = (r - 0.5 * variance) * dt
            v_drift = kappa * (theta - variance) * dt
            for jd in range(nd):
                m = mass[ix, iv, jd]
                if m < 1.0e-24:
                    continue
                draw = b * (jd / md) ** gamma
                running_max = x + draw
                for qv in range(z.size):
                    zv = z[qv]
                    v_cont = variance + v_drift + xi * sqrt_v * sqrt_dt * zv
                    if v_cont < 0.0:
                        v_cont = 0.0
                    if v_cont >= vmax:
                        continue
                    fv = v_cont / dv
                    iv_lo = int(np.floor(fv))
                    wv_hi = fv - iv_lo
                    if iv_lo < 0 or iv_lo >= nv - 1:
                        continue
                    for qx in range(z.size):
                        zx = rho * zv + sqrt1mr2 * z[qx]
                        x_cont = x + x_drift + sqrt_v * sqrt_dt * zx
                        fx = (x_cont - xmin) / dx
                        ix_lo = int(np.floor(fx))
                        wx_hi = fx - ix_lo
                        if ix_lo < 0 or ix_lo >= nx - 1:
                            continue
                        bm = m * w[qv] * w[qx]
                        for ax in range(2):
                            ix_new = ix_lo + ax
                            wx = (1.0 - wx_hi) if ax == 0 else wx_hi
                            x_new = xmin + ix_new * dx
                            draw_new = running_max - x_new
                            if draw_new < 0.0:
                                draw_new = 0.0
                            if draw_new >= b:
                                for av in range(2):
                                    iv_new = iv_lo + av
                                    wv = (1.0 - wv_hi) if av == 0 else wv_hi
                                    ki[ix_new, iv_new] += bm * wx * wv
                                continue
                            fd = (draw_new / b) ** (1.0 / gamma) * md
                            jd_lo = int(np.floor(fd))
                            wd_hi = fd - jd_lo
                            if jd_lo >= nd:
                                continue
                            for av in range(2):
                                iv_new = iv_lo + av
                                wv = (1.0 - wv_hi) if av == 0 else wv_hi
                                tr = bm * wx * wv
                                out[ix_new, iv_new, jd_lo] += tr * (1.0 - wd_hi)
                                if jd_lo + 1 < nd:
                                    out[ix_new, iv_new, jd_lo + 1] += tr * wd_hi
    return out, ki


@njit(cache=True, fastmath=True)
def _heston_step_ki(mass_ki, xmin, dx, vmax, dv, r, kappa, theta, xi, rho,
                     dt, z, w):
    nx, nv = mass_ki.shape
    out = np.zeros_like(mass_ki)
    sqrt_dt = np.sqrt(dt)
    sqrt1mr2 = np.sqrt(1.0 - rho * rho)
    for ix in range(nx):
        x = xmin + ix * dx
        for iv in range(nv):
            variance = iv * dv
            sqrt_v = np.sqrt(max(variance, 0.0))
            x_drift = (r - 0.5 * variance) * dt
            v_drift = kappa * (theta - variance) * dt
            m = mass_ki[ix, iv]
            if m < 1.0e-24:
                continue
            for qv in range(z.size):
                zv = z[qv]
                v_cont = variance + v_drift + xi * sqrt_v * sqrt_dt * zv
                if v_cont < 0.0:
                    v_cont = 0.0
                if v_cont >= vmax:
                    continue
                fv = v_cont / dv
                iv_lo = int(np.floor(fv))
                wv_hi = fv - iv_lo
                if iv_lo < 0 or iv_lo >= nv - 1:
                    continue
                for qx in range(z.size):
                    zx = rho * zv + sqrt1mr2 * z[qx]
                    x_cont = x + x_drift + sqrt_v * sqrt_dt * zx
                    fx = (x_cont - xmin) / dx
                    ix_lo = int(np.floor(fx))
                    wx_hi = fx - ix_lo
                    if ix_lo < 0 or ix_lo >= nx - 1:
                        continue
                    bm = m * w[qv] * w[qx]
                    for ax in range(2):
                        ix_new = ix_lo + ax
                        wx = (1.0 - wx_hi) if ax == 0 else wx_hi
                        for av in range(2):
                            iv_new = iv_lo + av
                            wv = (1.0 - wv_hi) if av == 0 else wv_hi
                            out[ix_new, iv_new] += bm * wx * wv
    return out


def heston_npi_live(S0, K, alpha, r, kappa, theta, xi, rho, V0, T,
                     N=50, nx=121, nv=151, md=60, nq=3, gamma=2.0,
                     vmax=0.40):
    x0 = np.log(S0)
    b = -np.log(alpha)
    dt = T / N
    center = x0 + (r - 0.5 * theta) * T
    half_width = 6.0 * np.sqrt(theta * T) + b
    xmin = center - half_width
    dx = (center + half_width - xmin) / (nx - 1)
    dv = vmax / (nv - 1)

    sigma_v_step = xi * np.sqrt(max(V0, 0.0) * dt)
    resolution_ratio = sigma_v_step / dv if dv > 0 else np.inf
    stable = resolution_ratio >= MIN_RESOLUTION_RATIO

    mass = np.zeros((nx, nv, md), dtype=np.float64)
    fx0 = (x0 - xmin) / dx; ix0 = int(np.floor(fx0)); wx1 = fx0 - ix0
    fv0 = V0 / dv; iv0 = int(np.floor(fv0)); wv1 = fv0 - iv0
    mass[ix0, iv0, 0] = (1.0 - wx1) * (1.0 - wv1)
    mass[ix0 + 1, iv0, 0] = wx1 * (1.0 - wv1)
    mass[ix0, iv0 + 1, 0] = (1.0 - wx1) * wv1
    mass[ix0 + 1, iv0 + 1, 0] = wx1 * wv1

    z, w = _gauss_hermite_normal(nq)
    mass_ki = np.zeros((nx, nv), dtype=np.float64)
    for _ in range(N):
        mass, new_ki = _heston_step_mass(
            mass, xmin, dx, vmax, dv, b, md, gamma,
            r, kappa, theta, xi, rho, dt, z, w)
        mass_ki = _heston_step_ki(
            mass_ki, xmin, dx, vmax, dv,
            r, kappa, theta, xi, rho, dt, z, w) + new_ki

    x_grid = xmin + dx * np.arange(nx)
    s_arr = np.exp(x_grid)
    disc = np.exp(-r * T)
    mass_x = mass.sum(axis=(1, 2))
    do_put = float(disc * np.dot(np.maximum(K - s_arr, 0.0), mass_x))
    do_call = float(disc * np.dot(np.maximum(s_arr - K, 0.0), mass_x))
    mass_ki_x = mass_ki.sum(axis=1)
    di_call = float(disc * np.dot(np.maximum(s_arr - K, 0.0), mass_ki_x))
    return {
        "do_put": do_put, "do_call": do_call, "di_call": di_call,
        "surv": float(mass.sum()), "ki_total": float(mass_ki.sum()),
        "resolution_ratio": float(resolution_ratio), "stable": bool(stable),
    }


def warmup_numba():
    """Force-compile the numba-jitted kernels once, at import time."""
    wz, ww = _gauss_hermite_normal(2)
    wm = np.zeros((5, 160, 4), dtype=np.float64); wm[2, 10, 0] = 1.0
    wki = np.zeros((5, 160), dtype=np.float64)
    _heston_step_mass(wm, -0.5, 0.2, 0.4, 0.0025, 0.2231, 4, 2.0,
                       0.05, 2.0, 0.04, 0.3, -0.7, 0.02, wz, ww)
    _heston_step_ki(wki, -0.5, 0.2, 0.4, 0.0025,
                     0.05, 2.0, 0.04, 0.3, -0.7, 0.02, wz, ww)
