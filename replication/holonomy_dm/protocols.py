"""Words over {A, B}, grouped comparators and Ramsey comparators."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from . import channels as ch

WITNESS = "ABABABAAB"


@dataclass(frozen=True)
class Model:
    """Parameter point(s); every field broadcasts against the others."""
    eps: object = np.pi / 3
    gamma: object = 0.0
    delta: object = 0.0
    eta: object = 0.0
    rho_c: object = 0.0

    def letter(self, name: str, h: int) -> np.ndarray:
        if name == "A":
            return ch.control_channel(h, self.eps, self.delta, self.rho_c, self.gamma, self.eta)
        if name == "B":
            shape = np.broadcast(self.eps, self.gamma, self.delta, self.eta, self.rho_c).shape
            return np.broadcast_to(
                ch.sensing_channel(h, self.gamma, self.eta), shape + (2, 2, 2))
        raise ValueError(name)

    def gates(self) -> dict:
        return {(n, h): self.letter(n, h) for n in "AB" for h in (+1, -1)}

    @property
    def shape(self):
        return np.broadcast(self.eps, self.gamma, self.delta, self.eta, self.rho_c).shape


def switches(word: str) -> int:
    return sum(a != b for a, b in zip(word, word[1:]))


def is_grouped(word: str) -> bool:
    """At most one switch between A and B (the repository's definition)."""
    return switches(word) <= 1


def all_words(max_len: int, min_len: int = 0):
    for n in range(min_len, max_len + 1):
        for letters in product("AB", repeat=n):
            yield "".join(letters)


def grouped_words(max_len: int):
    seen = set()
    for n in range(0, max_len + 1):
        for k in range(0, n + 1):
            for w in ("A" * k + "B" * (n - k), "B" * (n - k) + "A" * k):
                if w not in seen:
                    seen.add(w)
                    yield w


def output_states(word: str, model: Model, gates: dict | None = None):
    """rho_+ and rho_- after applying ``word`` chronologically to |0><0|."""
    gates = gates or model.gates()
    out = []
    for h in (+1, -1):
        rho = np.broadcast_to(ch.RHO0, model.shape + (2, 2)).copy()
        for name in word:
            rho = ch.apply(gates[(name, h)], rho)
        out.append(rho)
    return out[0], out[1]


def distance(word: str, model: Model, gates: dict | None = None, reference: bool = False):
    plus, minus = output_states(word, model, gates)
    return ch.trace_distance(plus, minus) if reference else ch.trace_distance_fast(plus, minus)


def total_time(word: str, rho_c) -> np.ndarray:
    """Duration in sensing windows: each B costs 1, each A costs rho_c."""
    return word.count("B") + word.count("A") * np.asarray(rho_c, float)


def gaussian_ensemble(model: Model, sigma_delta: float | np.ndarray = 0.0,
                      sigma_eta: float | np.ndarray = 0.0, nodes: int = 201):
    """Quasi-static Gaussian calibration noise as an explicit ensemble.

    Returns (ensemble_model, weights).  Axis 0 of ensemble_model indexes
    Gauss-Hermite nodes; delta -> delta + sigma_delta*x, eta -> eta + sigma_eta*x'
    with independent standard-normal x, x'.  The error is the same for every
    letter within a shot and redrawn between shots, so the hypotheses are the
    weighted averages of the output *states* (a convex mixture of CPTP maps,
    hence CPTP).  sigma_* may be arrays broadcasting with the model.
    """
    def rule(sigma):
        if not np.any(np.asarray(sigma)):
            return np.zeros(1), np.ones(1)
        x, w = np.polynomial.hermite_e.hermegauss(nodes)
        if not np.all(np.isfinite(w)):
            raise FloatingPointError(f"Gauss-Hermite weights overflow at {nodes} nodes")
        return x, w / w.sum()

    (xd, wd), (xe, we) = rule(sigma_delta), rule(sigma_eta)
    lead = np.broadcast(model.eps, model.gamma, model.delta, model.eta, model.rho_c,
                        sigma_delta, sigma_eta).shape
    ex = (slice(None), None) + (None,) * len(lead)
    ey = (None, slice(None)) + (None,) * len(lead)
    d = np.asarray(model.delta, float) + np.asarray(sigma_delta, float) * xd[ex]
    e = np.asarray(model.eta, float) + np.asarray(sigma_eta, float) * xe[ey]
    shape = (len(xd), len(xe)) + lead
    flat = lambda v: np.broadcast_to(np.asarray(v, float), shape).reshape((-1,) + lead)
    ens = Model(flat(model.eps), flat(model.gamma), flat(d), flat(e), flat(model.rho_c))
    return ens, np.outer(wd, we).reshape(-1)


def averaged_distance(word: str, ens: Model, weights: np.ndarray) -> np.ndarray:
    plus, minus = output_states(word, ens)
    return ch.trace_distance_fast(np.tensordot(weights, plus, axes=(0, 0)),
                                  np.tensordot(weights, minus, axes=(0, 0)))


# ---------------------------------------------------------------- Ramsey ----

def ramsey_distance(theta, s, model: Model, weights=None) -> np.ndarray:
    """Prepare with one y pulse of free area ``theta`` (same pulse model and Rabi
    frequency as letter A), sense for ``s`` windows (continuous), optimal
    terminal measurement.  theta, s broadcast against the model.  With
    ``weights``, axis 0 is an ensemble and states are averaged first."""
    out = []
    theta = np.asarray(theta, float)
    scale = theta / np.asarray(model.eps, float)
    for h in (+1, -1):
        prep = ch.control_channel(h, model.eps, model.delta, model.rho_c, model.gamma,
                                  model.eta, area_scale=scale)
        sense = ch.sensing_channel(h, model.gamma, model.eta, duration=s)
        rho = ch.apply(sense, ch.apply(prep, ch.RHO0))
        if weights is not None:
            rho = np.tensordot(weights, rho, axes=(0, 0))
        out.append(rho)
    return ch.trace_distance_fast(out[0], out[1])


def ramsey_nominal_design(gamma):
    """Optimal (theta, s) for the nominal model: ideal pulses, known gamma,
    no calibration errors.  Maximises sin(theta) exp(-gamma s) sin(s pi/2)."""
    gamma = np.asarray(gamma, float)
    s = (2 / np.pi) * np.arctan2(np.pi / 2, gamma)
    return np.full_like(s, np.pi / 2), s


def ramsey_optimised(model: Model, weights=None, n_theta: int = 49, n_s: int = 61,
                     s_max: float = 3.0, refinements: int = 4):
    """Oracle Ramsey: maximise D over the pulse area theta in (0, 2 pi] and the
    sensing duration s in (0, s_max] with full knowledge of every noise
    parameter (and of the ensemble, if any).  Coarse grid followed by
    successive local grid refinements; a numerical lower bound on the true
    optimum.  Ties are broken towards shorter protocols.  Returns (D, theta, s).
    """
    lead = model.shape if weights is None else model.shape[1:]
    n_lead = len(lead)

    def evaluate(tt, sg):
        # tt, sg: (n, m) + lead  ->  D with the same shape (row by row to bound memory)
        out = np.empty(tt.shape)
        for i in range(tt.shape[0]):
            if weights is None:
                out[i] = ramsey_distance(tt[i], sg[i], model)
            else:
                ens = lambda v: np.broadcast_to(np.asarray(v, float), model.shape)[
                    (slice(None), None)]
                m = Model(*[ens(getattr(model, k))
                            for k in ("eps", "gamma", "delta", "eta", "rho_c")])
                out[i] = ramsey_distance(tt[i][None], sg[i][None], m, weights)
        return out

    centre_t = np.full(lead, np.pi)
    centre_s = np.full(lead, s_max / 2)
    half_t, half_s = np.pi, s_max / 2
    best = np.full(lead, -1.0)
    best_t, best_s = centre_t.copy(), centre_s.copy()
    for _ in range(refinements + 1):
        ut = np.linspace(-1, 1, n_theta)[(slice(None), None) + (None,) * n_lead]
        us = np.linspace(-1, 1, n_s)[(None, slice(None)) + (None,) * n_lead]
        tt = np.clip(centre_t + half_t * ut, 1e-9, 2 * np.pi) + 0 * us
        sg = np.clip(centre_s + half_s * us, 1e-9, s_max) + 0 * ut
        val = evaluate(tt, sg) - 1e-12 * (sg + tt)
        flat = val.reshape((-1,) + lead)
        idx = np.argmax(flat, axis=0)
        take = lambda arr: np.take_along_axis(arr.reshape((-1,) + lead), idx[None], axis=0)[0]
        v = take(val)
        better = v > best
        best = np.where(better, v, best)
        best_t = np.where(better, take(tt), best_t)
        best_s = np.where(better, take(sg), best_s)
        centre_t, centre_s = best_t, best_s
        half_t, half_s = half_t * 4 / (n_theta - 1), half_s * 4 / (n_s - 1)
    return ramsey_distance(best_t, best_s, model, weights), best_t, best_s
