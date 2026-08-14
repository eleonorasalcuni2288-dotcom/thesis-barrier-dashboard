"""
pricing_bs.py — Black-Scholes analytical formulas (fixed barrier).

Consolidates the Reiner-Rubinstein (1991) barrier formulas that were
previously duplicated with minor variations across Dashboards 1, 2, 3
(_rr_components / _rr) and Dashboards 4.4, 5.5, 6.6, 7, 9.9, 10
(bs_vanilla). All variants were verified numerically equivalent before
consolidation — see thesis chat log for the cross-check.

Public API:
    bs_vanilla(S, K, r, q, sigma, T, opt="call") -> float
    bs_down_barrier(S, K, H, r, q, sigma, T) -> dict | None
        Returns None if H >= S (barrier already breached).
        Otherwise returns:
            do_put, di_put, do_call, di_call, vanilla_put, vanilla_call
"""

import numpy as np
from scipy.stats import norm

N = norm.cdf


def bs_vanilla(S, K, r, q, sigma, T, opt="call"):
    """Standard Black-Scholes vanilla price (continuous dividend yield q)."""
    sv = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / sv
    d2 = d1 - sv
    if opt == "call":
        return float(S * np.exp(-q * T) * N(d1) - K * np.exp(-r * T) * N(d2))
    return float(K * np.exp(-r * T) * N(-d2) - S * np.exp(-q * T) * N(-d1))


def _rr_components(S, K, H, r, q, sigma, T):
    """Reiner-Rubinstein (1991) intermediate terms x1,x2,y1,y2,Sd,Kd,f1,f2,sv."""
    b = r - q
    sv = sigma * np.sqrt(T)
    mu = (b - 0.5 * sigma ** 2) / sigma ** 2
    x1 = np.log(S / K) / sv + (1 + mu) * sv
    x2 = np.log(S / H) / sv + (1 + mu) * sv
    y1 = np.log(H ** 2 / (S * K)) / sv + (1 + mu) * sv
    y2 = np.log(H / S) / sv + (1 + mu) * sv
    Sd = S * np.exp(-q * T)
    Kd = K * np.exp(-r * T)
    f1 = (H / S) ** (2 * (mu + 1))
    f2 = (H / S) ** (2 * mu)
    return x1, x2, y1, y2, Sd, Kd, f1, f2, sv


def bs_down_barrier(S, K, H, r, q, sigma, T):
    """Down-and-Out/In Put and Call, analytical (Reiner-Rubinstein).

    Returns None if H >= S (barrier already breached at t=0 — undefined).
    Otherwise a dict with do_put, di_put, do_call, di_call,
    vanilla_put, vanilla_call.
    """
    if H >= S:
        return None

    vp = bs_vanilla(S, K, r, q, sigma, T, "put")
    vc = bs_vanilla(S, K, r, q, sigma, T, "call")

    if K < H:
        # Degenerate case: strike below barrier.
        return {"do_put": 0.0, "di_put": vp, "do_call": 0.0, "di_call": vc,
                "vanilla_put": vp, "vanilla_call": vc}

    x1, x2, y1, y2, Sd, Kd, f1, f2, sv = _rr_components(S, K, H, r, q, sigma, T)

    Ap = -Sd * N(-x1) + Kd * N(-x1 + sv)
    Bp = -Sd * N(-x2) + Kd * N(-x2 + sv)
    Cp = -Sd * f1 * N(-y1) + Kd * f2 * N(-y1 + sv)
    Dp = -Sd * f1 * N(-y2) + Kd * f2 * N(-y2 + sv)
    do_put = Ap - Bp - Cp + Dp
    di_put = vp - do_put

    Cc = Sd * f1 * N(y1) - Kd * f2 * N(y1 - sv)
    di_call = Cc
    do_call = vc - di_call

    return {
        "do_put": float(do_put), "di_put": float(di_put),
        "do_call": float(do_call), "di_call": float(di_call),
        "vanilla_put": float(vp), "vanilla_call": float(vc),
    }


def bs_do_put(S, K, H, r, q, sigma, T):
    """Convenience wrapper: Down-and-Out Put only. Returns 0.0 if K < H."""
    res = bs_down_barrier(S, K, H, r, q, sigma, T)
    return 0.0 if res is None else res["do_put"]


def bs_di_call(S, K, H, r, q, sigma, T):
    """Convenience wrapper: Down-and-In Call only. Returns 0.0 if K < H."""
    res = bs_down_barrier(S, K, H, r, q, sigma, T)
    return 0.0 if res is None else res["di_call"]
