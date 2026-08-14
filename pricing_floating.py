"""
pricing_floating.py — Floating barrier pricing (B_t = alpha * running max).

Consolidates functions previously duplicated (with minor variations) across
Dashboards 4.4, 5.5, 6.6, 7 (Comparison), and reused by 9.9 (Adaptive NPI).

IMPORTANT: npi_uniform() here replaces the slow pure-Python double loop
that Dashboards 6.6 and 7 used (`for _ in range(N_steps): for mp in
range(Nx1): np.add.at(...)`), which does not scale well on shared/free
hosting CPUs. The replacement uses the same vectorized transition-kernel
technique already used in the Heston and Adaptive NPI dashboards
(precompute kernel once, argsort + np.add.at on flat arrays). It was
verified numerically equivalent to the original loop-based version before
being adopted — see thesis chat log for the cross-check.

Public API:
    bs_vanilla_simple(S0, K, r, sigma, T, opt="call") -> float
        Thin wrapper around pricing_bs.bs_vanilla with q=0, kept for
        call-site compatibility with the original per-dashboard signature.
    mc_naive(S0, K, alpha, r, sigma, T, N, M, seed) -> dict
    mc_bb(S0, K, alpha, r, sigma, T, N, M, seed) -> dict
    npi_uniform(S0, K, alpha, r, sigma, T, N_steps, Mx, Mv, x_width) -> dict
"""

import numpy as np
from scipy.stats import norm

from pricing_bs import bs_vanilla


def bs_vanilla_simple(S0, K, r, sigma, T, opt="call"):
    return bs_vanilla(S0, K, r, 0.0, sigma, T, opt)


# ── Monte Carlo Naive ──────────────────────────────────────────────────────

def mc_naive(S0, K, alpha, r, sigma, T, N=50, M=30_000, seed=42):
    rng = np.random.default_rng(seed)
    b = -np.log(alpha)
    dt = T / N
    mu = r - 0.5 * sigma ** 2
    x0 = np.log(S0)
    disc = np.exp(-r * T)

    Z = rng.standard_normal((M, N))
    xp = x0 + np.cumsum(mu * dt + sigma * np.sqrt(dt) * Z, axis=1)
    xf = np.hstack([np.full((M, 1), x0), xp])
    rm = np.maximum.accumulate(xf, axis=1)
    dd = rm - xf
    sv = np.all(dd[:, 1:] < b, axis=1)
    ST = np.exp(xp[:, -1])

    pp = np.maximum(K - ST, 0) * sv
    cp = np.maximum(ST - K, 0) * sv
    c_bs = bs_vanilla_simple(S0, K, r, sigma, T, "call")

    do_p = float(disc * np.mean(pp))
    se_p = float(disc * np.std(pp, ddof=1) / np.sqrt(M))
    do_c = float(disc * np.mean(cp))
    se_c = float(disc * np.std(cp, ddof=1) / np.sqrt(M))

    return {"do_put": do_p, "se_put": se_p,
            "do_call": do_c, "se_call": se_c,
            "di_call": c_bs - do_c, "surv": float(sv.mean()), "c_bs": c_bs}


# ── Monte Carlo with local Brownian Bridge correction ─────────────────────

def mc_bb(S0, K, alpha, r, sigma, T, N=50, M=30_000, seed=42):
    rng = np.random.default_rng(seed)
    b = -np.log(alpha)
    dt = T / N
    mu = r - 0.5 * sigma ** 2
    x0 = np.log(S0)
    disc = np.exp(-r * T)

    Z = rng.standard_normal((M, N))
    xp = x0 + np.cumsum(mu * dt + sigma * np.sqrt(dt) * Z, axis=1)
    xf = np.hstack([np.full((M, 1), x0), xp])
    rm = np.maximum.accumulate(xf, axis=1)
    dd = rm - xf

    v_a = dd[:, :-1]
    v_c = dd[:, 1:]
    both = (v_a < b) & (v_c < b)
    pc = np.where(both, np.exp(-2 * (b - v_a) * (b - v_c) / (sigma ** 2 * dt)), 1.0)
    pc = np.clip(pc, 0, 1)
    node_ok = v_c < b
    sw = np.prod(np.where(node_ok, 1 - pc, 0.0), axis=1)
    ST = np.exp(xp[:, -1])

    pp = np.maximum(K - ST, 0) * sw
    cp = np.maximum(ST - K, 0) * sw
    c_bs = bs_vanilla_simple(S0, K, r, sigma, T, "call")

    se_p = float(disc * np.std(pp, ddof=1) / np.sqrt(M))
    se_c = float(disc * np.std(cp, ddof=1) / np.sqrt(M))

    return {"do_put": float(disc * np.mean(pp)), "se_put": se_p,
            "do_call": float(disc * np.mean(cp)), "se_call": se_c,
            "di_call": c_bs - float(disc * np.mean(cp)),
            "surv": float(np.mean(sw)), "c_bs": c_bs}


# ── NPI (Numerical Path Integral), vectorized ──────────────────────────────

def npi_uniform(S0, K, alpha, r, sigma, T, N_steps=50, Mx=100, Mv=70,
                 x_width=4.5):
    """Vectorized NPI on a uniform (x, v) grid for the BS floating barrier.

    Numerically equivalent to the original per-dashboard `npi()` (pure
    Python double loop) but replaces the inner loops with a precomputed,
    flattened transition kernel (same technique as the Heston / Adaptive
    NPI dashboards), making it dramatically faster for the alpha/N sweeps
    used interactively.
    """
    x0 = np.log(S0)
    mu = r - 0.5 * sigma ** 2
    b = -np.log(alpha)
    dt = T / N_steps

    xc = x0 + mu * T
    half = x_width * sigma * np.sqrt(T)
    x_grid = np.linspace(xc - half, xc + half, Mx + 1)
    dv = b / Mv
    v_grid = np.arange(Mv) * dv

    Nx1 = Mx + 1
    dx = (x_grid[-1] - x_grid[0]) / Mx

    # Initial mass at (x0, v=0)
    G = np.zeros((Nx1, Mv))
    ix0 = (x0 - x_grid[0]) / dx
    ix0_f = int(np.clip(np.floor(ix0), 0, Mx - 1))
    wx1 = ix0 - ix0_f
    wx0 = 1.0 - wx1
    G[ix0_f, 0] = wx0
    G[ix0_f + 1, 0] = wx1

    # Precompute transition kernel (x-diffusion x v-drawdown reset), once.
    sv_dt = sigma * np.sqrt(dt)
    mu_dt = mu * dt
    diff_xm = x_grid[:, None] - x_grid[None, :]           # (Nx1, Nx1)
    K_prob = norm.pdf(diff_xm, mu_dt, sv_dt) * dx
    diff_3d = diff_xm[:, :, None]
    v_3d = v_grid[None, None, :]
    v_star = np.maximum(v_3d - diff_3d, 0.0)               # (Nx1, Nx1, Mv)
    alive = (v_star < b)

    j_frac = v_star / dv
    jlo = np.clip(np.floor(j_frac).astype(int), 0, Mv - 2)
    w_hi = (j_frac - jlo).astype(np.float32)
    w_lo = (1.0 - w_hi).astype(np.float32)

    Kexp = (K_prob[:, :, None] * alive).reshape(-1).astype(np.float32)
    jlo_3d = jlo.reshape(Nx1, Nx1, Mv)

    m_3d = np.arange(Nx1)[None, :, None] * np.ones((Nx1, 1, Mv), dtype=int)
    j_3d = np.arange(Mv)[None, None, :] * np.ones((Nx1, Nx1, 1), dtype=int)
    src = (m_3d * Mv + j_3d).reshape(-1)
    mp_3d = np.arange(Nx1)[:, None, None] * np.ones((1, Nx1, Mv), dtype=int)
    tlo = (mp_3d * Mv + jlo_3d).reshape(-1)
    thi = (mp_3d * Mv + (jlo_3d + 1)).reshape(-1)

    wlo_v = Kexp * w_lo.reshape(-1)
    whi_v = Kexp * w_hi.reshape(-1)
    mask = (wlo_v > 0) | (whi_v > 0)
    tgt_all = np.concatenate([tlo[mask], thi[mask]])
    src_all = np.concatenate([src[mask], src[mask]])
    wts_all = np.concatenate([wlo_v[mask], whi_v[mask]]).astype(np.float64)
    order = np.argsort(tgt_all)
    tgt_all = tgt_all[order]
    src_all = src_all[order]
    wts_all = wts_all[order]

    M_arr = G.reshape(-1)
    for _ in range(N_steps):
        vals = wts_all * M_arr[src_all]
        M_new = np.zeros(Nx1 * Mv)
        np.add.at(M_new, tgt_all, vals)
        M_arr = M_new

    G_out = M_arr.reshape(Nx1, Mv)
    G_x = G_out.sum(axis=1)
    S_arr = np.exp(x_grid)
    disc = np.exp(-r * T)

    do_put = float(disc * np.dot(np.maximum(K - S_arr, 0.0), G_x))
    do_call = float(disc * np.dot(np.maximum(S_arr - K, 0.0), G_x))
    surv = float(G_x.sum())
    c_bs = bs_vanilla_simple(S0, K, r, sigma, T, "call")

    return {"do_put": do_put, "do_call": do_call,
            "di_call": c_bs - do_call, "c_bs": c_bs, "surv": surv,
            "G_N": G_out, "x_grid": x_grid, "v_grid": v_grid, "b": b}


# Backward-compatible alias — the original per-dashboard functions were
# all named `npi(...)` with this exact signature.
npi = npi_uniform
