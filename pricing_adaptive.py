"""
pricing_adaptive.py — Adaptive vs uniform v-grid NPI (floating barrier, BS).

Extracted from the already-verified Adaptive NPI dashboard (see thesis
chat log: array-length mismatch fix in the v-grid-spacing panel, x_width
documented explicitly). No behavior change from that verified version.

npi_uniform_scriptgrid() differs from pricing_floating.npi_uniform() only
in its v-grid construction (b*(Mv-1)/Mv endpoint, matching Script 9.9's
own uniform baseline exactly) — kept separate rather than merged, since
the two were validated against different reference numbers in the thesis
and unifying them is out of scope for this consolidation pass.
"""

import numpy as np
from scipy.stats import norm

from pricing_bs import bs_vanilla


def bs_vanilla_simple(S0, K, r, sigma, T, opt="call"):
    return bs_vanilla(S0, K, r, 0.0, sigma, T, opt)


def mc_naive(S0, K, alpha, r, sigma, T, N=50, M=20_000, seed=42):
    rng = np.random.default_rng(seed)
    b = -np.log(alpha); dt = T/N; mu = r - 0.5*sigma**2
    x0 = np.log(S0); disc = np.exp(-r*T)
    Z = rng.standard_normal((M, N))
    xp = x0 + np.cumsum(mu*dt + sigma*np.sqrt(dt)*Z, axis=1)
    xf = np.hstack([np.full((M, 1), x0), xp])
    rm = np.maximum.accumulate(xf, axis=1); dd = rm - xf
    sv = np.all(dd[:, 1:] < b, axis=1); ST = np.exp(xp[:, -1])
    pp = np.maximum(K - ST, 0) * sv
    cp = np.maximum(ST - K, 0) * sv
    do_p = float(disc * np.mean(pp)); se_p = float(disc * np.std(pp, ddof=1) / np.sqrt(M))
    do_c = float(disc * np.mean(cp)); se_c = float(disc * np.std(cp, ddof=1) / np.sqrt(M))
    c_bs = bs_vanilla_simple(S0, K, r, sigma, T, "call")
    return {"do_put": do_p, "se_put": se_p, "surv": float(sv.mean()),
            "do_call": do_c, "se_call": se_c, "di_call": c_bs - do_c, "c_bs": c_bs}


def _npi_run(S0, K, alpha, r, sigma, T, N_steps, x_grid, v_grid):
    x0 = np.log(S0); mu = r - 0.5*sigma**2; b = -np.log(alpha)
    dt = T / N_steps; Mx = len(x_grid) - 1; Mv = len(v_grid)
    dx = (x_grid[-1] - x_grid[0]) / Mx

    G = np.zeros((Mx + 1, Mv))
    ix0 = (x0 - x_grid[0]) / dx
    ix0_f = int(np.clip(np.floor(ix0), 0, Mx - 1))
    wx1 = ix0 - ix0_f; wx0 = 1.0 - wx1
    G[ix0_f, 0] = wx0; G[ix0_f + 1, 0] = wx1

    sv_dt = sigma * np.sqrt(dt); mu_dt = mu * dt
    diff_xm = x_grid[:, None] - x_grid[None, :]
    K_prob = norm.pdf(diff_xm, mu_dt, sv_dt) * dx
    diff_3d = diff_xm[:, :, None]
    v_3d = v_grid[None, None, :]
    v_star = np.maximum(v_3d - diff_3d, 0.0)
    alive = (v_star < b)

    vs_flat = v_star.reshape(-1)
    jlo = np.searchsorted(v_grid, vs_flat, side="right") - 1
    jlo = np.clip(jlo, 0, Mv - 2)
    vlo = v_grid[jlo]; vhi = v_grid[jlo + 1]
    span = np.where(vhi - vlo > 1e-15, vhi - vlo, 1.0)
    w_hi = ((vs_flat - vlo) / span).astype(np.float32)
    w_lo = (1.0 - w_hi).astype(np.float32)

    Kexp = (K_prob[:, :, None] * alive).reshape(-1).astype(np.float32)
    Nx1 = Mx + 1
    jlo_3d = jlo.reshape(Nx1, Nx1, Mv)

    m_3d = np.arange(Nx1)[None, :, None] * np.ones((Nx1, 1, Mv), dtype=int)
    j_3d = np.arange(Mv)[None, None, :] * np.ones((Nx1, Nx1, 1), dtype=int)
    src = (m_3d * Mv + j_3d).reshape(-1)
    mp_3d = np.arange(Nx1)[:, None, None] * np.ones((1, Nx1, Mv), dtype=int)
    tlo = (mp_3d * Mv + jlo_3d).reshape(-1)
    thi = (mp_3d * Mv + (jlo_3d + 1)).reshape(-1)
    wlo_v = (Kexp * w_lo)
    whi_v = (Kexp * w_hi)
    mask = (wlo_v > 0) | (whi_v > 0)
    tgt_all = np.concatenate([tlo[mask], thi[mask]])
    src_all = np.concatenate([src[mask], src[mask]])
    wts_all = np.concatenate([wlo_v[mask], whi_v[mask]]).astype(np.float64)
    order = np.argsort(tgt_all)
    tgt_all = tgt_all[order]; src_all = src_all[order]; wts_all = wts_all[order]

    M_arr = G.reshape(-1)
    for _ in range(N_steps):
        vals = wts_all * M_arr[src_all]
        M_new = np.zeros(Nx1 * Mv)
        np.add.at(M_new, tgt_all, vals)
        M_arr = M_new

    G_out = M_arr.reshape(Nx1, Mv)
    G_x = G_out.sum(axis=1)
    S_arr = np.exp(x_grid); disc = np.exp(-r * T)
    do_put = float(disc * np.dot(np.maximum(K - S_arr, 0), G_x))
    do_call = float(disc * np.dot(np.maximum(S_arr - K, 0), G_x))
    surv = float(G_x.sum())
    c_bs = bs_vanilla_simple(S0, K, r, sigma, T, "call")
    return {"do_put": do_put, "do_call": do_call, "di_call": c_bs - do_call,
            "c_bs": c_bs, "surv": surv, "G_N": G_out,
            "x_grid": x_grid, "v_grid": v_grid}


def npi_uniform_scriptgrid(S0, K, alpha, r, sigma, T, N_steps=50, Mx=60,
                            Mv=50, x_width=4.0):
    x0 = np.log(S0); mu = r - 0.5*sigma**2; b = -np.log(alpha)
    xc = x0 + mu*T; half = x_width*sigma*np.sqrt(T)
    x_grid = np.linspace(xc - half, xc + half, Mx + 1)
    v_grid = np.linspace(0, b * (Mv - 1)/Mv, Mv)
    return _npi_run(S0, K, alpha, r, sigma, T, N_steps, x_grid, v_grid)


def npi_adaptive(S0, K, alpha, r, sigma, T, N_steps=50, Mx=60, Mv=50,
                  x_width=4.0, v_gamma=2.0):
    x0 = np.log(S0); mu = r - 0.5*sigma**2; b = -np.log(alpha)
    xc = x0 + mu*T; half = x_width*sigma*np.sqrt(T)
    x_grid = np.linspace(xc - half, xc + half, Mx + 1)
    j_arr = np.arange(Mv)
    v_grid = b * (j_arr / Mv) ** v_gamma
    return _npi_run(S0, K, alpha, r, sigma, T, N_steps, x_grid, v_grid)
