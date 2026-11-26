"""Noise sweeps and comparator protocols.

Every quantity is a single-shot trace distance D between the two hypothesis
output states (Helstrom success probability (1 + D)/2 at equal priors).

Comparators evaluated at every parameter point
----------------------------------------------
witness            the fixed word ABABABAAB
best_all_L         max over all 2^(L+1)-1 words of length <= L   (L = 9, 12)
best_grouped_L     max over grouped words (<= 1 switch) of length <= L
best_inter_L       max over non-grouped words of length <= L
grouped_long       max over A^k B^m and B^m A^k, k <= K_MAX, m <= M_MAX
                   (finite proxy for "unlimited" grouped words)
grouped_time       same family, restricted to total duration <= duration of
                   the witness (4 windows + 5 pulses)
best_all_time      max over all words of length <= 12 with that same duration cap
best_inter_time    same, non-grouped words only
ramsey_oracle      one free-area y pulse + one free-duration sensing window,
                   optimised with full knowledge of the noise
ramsey_nominal     the Ramsey design that is optimal for known gamma and
                   *nominal* calibration, evaluated on the true model
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from . import channels as ch
from .enumeration import _dist, summarise
from .protocols import (Model, WITNESS, gaussian_ensemble, output_states,
                        ramsey_distance, ramsey_nominal_design, ramsey_optimised)

K_MAX, M_MAX = 60, 24
GAP_TOL = 1e-9
EPS0 = np.pi / 3


def grouped_family(model: Model, weights=None, k_max=K_MAX, m_max=M_MAX):
    """Best of A^k B^m / B^m A^k (k<=k_max, m<=m_max), plus the best subject
    to duration m + k*rho_c <= 4 + 5*rho_c.  Returns dict of arrays."""
    gates = model.gates()
    lead = model.shape if weights is None else model.shape[1:]
    rho_c = np.broadcast_to(np.asarray(model.rho_c, float), model.shape)
    if weights is not None:
        rho_c = rho_c[0]
    budget = 4 + 5 * rho_c
    best = np.zeros(lead)
    best_t = np.zeros(lead)
    arg = np.zeros(lead + (3,), dtype=int)       # (order, k, m); order 0 = A first
    rho0 = np.broadcast_to(ch.RHO0, model.shape + (2, 2)).copy()
    for order, (first, second, n1, n2) in enumerate(
            (("A", "B", k_max, m_max), ("B", "A", m_max, k_max))):
        p1, m1 = rho0, rho0
        for i in range(n1 + 1):
            if i:
                p1, m1 = ch.apply(gates[(first, 1)], p1), ch.apply(gates[(first, -1)], m1)
            p2, m2 = p1, m1
            for j in range(n2 + 1):
                if j:
                    p2, m2 = ch.apply(gates[(second, 1)], p2), ch.apply(gates[(second, -1)], m2)
                k, m = (i, j) if order == 0 else (j, i)
                d = _dist(p2, m2, weights)
                better = d > best + 1e-13
                best = np.where(better, d, best)
                arg[better] = (order, k, m)
                ok = (m + k * rho_c) <= budget + 1e-12
                best_t = np.where(ok & (d > best_t), d, best_t)
    return {"grouped_long": best, "grouped_time": best_t, "grouped_long_arg": arg}


def comparators(model: Model, weights=None, max_len: int = 12, ramsey: bool = True):
    lead = model.shape if weights is None else model.shape[1:]
    summ = summarise(model, max_len, weights=weights)
    out = {}
    plus, minus = output_states(WITNESS, model)
    out["witness"] = _dist(plus, minus, weights)
    ball, bgrp, bint = summ.budget("all"), summ.budget("grouped"), summ.budget("interleaved")
    arg_all, arg_grp = summ.budget_arg("all"), summ.budget_arg("grouped")
    words = np.array(summ.words)
    for L in sorted({min(9, max_len), max_len}):
        out[f"best_all_{L}"] = ball[L]
        out[f"best_grouped_{L}"] = bgrp[L]
        out[f"best_inter_{L}"] = bint[L]
        out[f"gap_{L}"] = ball[L] - bgrp[L]
        out[f"argbest_all_{L}"] = words[arg_all[L]]
        out[f"argbest_grouped_{L}"] = words[arg_grp[L]]
    out["best_all_time"], out["best_inter_time"] = summ.time_all, summ.time_inter
    out.update(grouped_family(model, weights))
    if ramsey:
        d, th, s = ramsey_optimised(model, weights)
        out["ramsey_oracle"], out["ramsey_oracle_theta"], out["ramsey_oracle_s"] = d, th, s
        gam = np.broadcast_to(np.asarray(model.gamma, float), model.shape)
        gam = gam if weights is None else gam[0]
        th_n, s_n = ramsey_nominal_design(gam)
        if weights is None:
            out["ramsey_nominal"] = ramsey_distance(th_n, s_n, model)
        else:
            out["ramsey_nominal"] = ramsey_distance(th_n[None], s_n[None], model, weights)
    return {k: np.broadcast_to(v, lead + np.shape(v)[len(lead):]) if isinstance(v, np.ndarray)
            else v for k, v in out.items()}


# ----------------------------------------------------------------- sweeps ---

AXES = {
    # name: (values, human label)
    "gamma": (np.linspace(0.0, 0.5, 201), "dephasing exponent per sensing window"),
    "delta": (np.linspace(-0.15, 0.15, 301), "fractional systematic pulse-area error"),
    "eta": (np.linspace(-0.4, 0.4, 321), "fractional systematic field-magnitude error"),
    "rho_c": (np.linspace(0.0, 1.0, 201), "control duration / sensing window"),
    "sigma_delta": (np.linspace(0.0, 0.15, 31), "std of quasi-static Gaussian pulse-area error"),
    "sigma_eta": (np.linspace(0.0, 0.4, 41), "std of quasi-static Gaussian field-magnitude error"),
}


def _model_for(axis: str, values, **fixed):
    base = dict(eps=EPS0, gamma=0.0, delta=0.0, eta=0.0, rho_c=0.0)
    base.update(fixed)
    values = np.asarray(values, float)
    if axis in ("sigma_delta", "sigma_eta"):
        return gaussian_ensemble(Model(**base), **{axis: values})
    base[axis] = values + np.asarray(base[axis], float)
    return Model(**base), None


def sweep_1d(axis: str, values=None, max_len: int = 12, **fixed):
    values = AXES[axis][0] if values is None else np.asarray(values, float)
    model, weights = _model_for(axis, values, **fixed)
    res = comparators(model, weights, max_len)
    res["x"] = values
    return res


def sweep_2d(axis_x: str, xs, axis_y: str, ys, max_len: int = 12, **fixed):
    """Enumerate on a grid of two *model* parameters (eps, gamma, delta, eta, rho_c)."""
    base = dict(eps=EPS0, gamma=0.0, delta=0.0, eta=0.0, rho_c=0.0)
    base.update(fixed)
    xg, yg = np.meshgrid(np.asarray(xs, float), np.asarray(ys, float), indexing="ij")
    base[axis_x], base[axis_y] = xg, yg
    res = comparators(Model(**base), None, max_len, ramsey=False)
    res["x"], res["y"] = np.asarray(xs, float), np.asarray(ys, float)
    return res


def _scalar_gap(axis: str, value: float, **fixed) -> float:
    """witness minus best grouped word of length <= 9 at one point."""
    model, weights = _model_for(axis, np.array([value]), **fixed)
    summ = summarise(model, 9, weights=weights)
    plus, minus = output_states(WITNESS, model)
    return float((_dist(plus, minus, weights) - summ.budget("grouped")[9])[0])


def witness_threshold(axis: str, lo: float, hi: float, **fixed) -> float:
    """Root of (witness - best grouped <= 9) between lo (advantage) and hi (none)."""
    return brentq(lambda v: _scalar_gap(axis, v, **fixed), lo, hi, xtol=1e-12, rtol=1e-12)


def last_positive(x, gap, tol=GAP_TOL):
    """Connected interval around x=0 (the nominal point) on which gap > tol.
    Returns (left, right) grid values, or None if the nominal point has no gap."""
    x, gap = np.asarray(x), np.asarray(gap)
    i0 = int(np.argmin(np.abs(x)))
    if gap[i0] <= tol:
        return None
    lo = hi = i0
    while lo > 0 and gap[lo - 1] > tol:
        lo -= 1
    while hi < len(x) - 1 and gap[hi + 1] > tol:
        hi += 1
    return float(x[lo]), float(x[hi])


def multi_shot_grouped(n_shots: int, d_single: float) -> float:
    """Helstrom distance for n independent copies of a pair of *pure* states
    with single-copy trace distance d (overlap^2 = 1 - d^2)."""
    return float(np.sqrt(1 - (1 - d_single ** 2) ** n_shots))
