"""Exhaustive enumeration of every word over {A, B} up to a given depth.

Depth-first traversal of the binary word tree; each node holds the pair of
explicit density matrices (rho_+, rho_-) for every parameter point, so each
of the 2^(L+1)-1 words costs one channel application per hypothesis.
No symmetry reduction or pruning is applied (leading B / trailing A words
are enumerated like any other).
"""
from __future__ import annotations

import numpy as np

from . import channels as ch
from .protocols import Model, is_grouped


def _dist(plus, minus, weights):
    if weights is not None:
        plus = np.tensordot(weights, plus, axes=(0, 0))
        minus = np.tensordot(weights, minus, axes=(0, 0))
    return ch.trace_distance_fast(plus, minus)


def enumerate_words(model: Model, max_len: int, visit, weights=None) -> int:
    """Call visit(word, D) for every word of length 0..max_len. Returns count.

    If ``weights`` is given, axis 0 of the model is an ensemble (quasi-static
    noise); the *states* are averaged with these weights before the distance
    is taken, i.e. the hypotheses are the ensemble-averaged density matrices.
    """
    gates = model.gates()
    shape = model.shape
    rho0 = np.broadcast_to(ch.RHO0, shape + (2, 2)).copy()
    count = 0

    def rec(word, plus, minus):
        nonlocal count
        visit(word, _dist(plus, minus, weights))
        count += 1
        if len(word) == max_len:
            return
        for name in "AB":
            rec(word + name, ch.apply(gates[(name, +1)], plus),
                ch.apply(gates[(name, -1)], minus))

    rec("", rho0, rho0)
    return count


class Summary:
    """Running maxima by length, by grouped/interleaved class and by (nA, nB)."""

    def __init__(self, shape, max_len: int, keep_all: bool = False, rho_c=None):
        self.max_len = max_len
        # optional equal-duration class: words no longer than the witness
        # (4 sensing windows + 5 pulses), any length <= max_len
        self.rho_c = None if rho_c is None else np.broadcast_to(np.asarray(rho_c, float), shape)
        self.time_all = np.zeros(shape)
        self.time_inter = np.zeros(shape)
        z = lambda: np.full((max_len + 1,) + tuple(shape), -1.0)
        self.best_all, self.best_grouped, self.best_inter = z(), z(), z()
        self.arg_all = np.zeros((max_len + 1,) + tuple(shape), dtype=np.int64)
        self.arg_grouped = np.zeros_like(self.arg_all)
        self.count_all = {}      # (nA, nB) -> best over all words with these counts
        self.count_grouped = {}  # (nA, nB) -> best over grouped words with these counts
        self.words = []
        self.keep_all = keep_all
        self.table = []

    def __call__(self, word, d):
        idx = len(self.words)
        self.words.append(word)
        if self.keep_all:
            self.table.append(d.copy())
        n = len(word)
        grouped = is_grouped(word)
        better = d > self.best_all[n]
        self.best_all[n] = np.where(better, d, self.best_all[n])
        self.arg_all[n] = np.where(better, idx, self.arg_all[n])
        if grouped:
            better = d > self.best_grouped[n]
            self.best_grouped[n] = np.where(better, d, self.best_grouped[n])
            self.arg_grouped[n] = np.where(better, idx, self.arg_grouped[n])
        else:
            self.best_inter[n] = np.maximum(self.best_inter[n], d)
        key = (word.count("A"), word.count("B"))
        if self.rho_c is not None:
            ok = key[1] + key[0] * self.rho_c <= 4 + 5 * self.rho_c + 1e-12
            self.time_all = np.where(ok & (d > self.time_all), d, self.time_all)
            if not grouped:
                self.time_inter = np.where(ok & (d > self.time_inter), d, self.time_inter)
        self.count_all[key] = np.maximum(self.count_all.get(key, -1.0), d)
        if grouped:
            self.count_grouped[key] = np.maximum(self.count_grouped.get(key, -1.0), d)

    # cumulative ("budget <= L") views -------------------------------------
    def budget(self, which: str) -> np.ndarray:
        arr = {"all": self.best_all, "grouped": self.best_grouped,
               "interleaved": self.best_inter}[which]
        return np.maximum.accumulate(np.maximum(arr, 0.0), axis=0)

    def budget_arg(self, which: str) -> np.ndarray:
        """Index (into self.words) of the first-found shortest maximiser within budget L."""
        val = {"all": self.best_all, "grouped": self.best_grouped}[which]
        arg = {"all": self.arg_all, "grouped": self.arg_grouped}[which]
        out = np.zeros_like(arg)
        run = np.full(val.shape[1:], -1.0)
        cur = np.zeros(val.shape[1:], dtype=np.int64)
        for n in range(val.shape[0]):
            better = val[n] > run + 1e-13
            run = np.where(better, val[n], run)
            cur = np.where(better, arg[n], cur)
            out[n] = cur
        return out


def summarise(model: Model, max_len: int, keep_all: bool = False, weights=None) -> Summary:
    shape = model.shape if weights is None else model.shape[1:]
    rho_c = np.broadcast_to(np.asarray(model.rho_c, float), model.shape)
    s = Summary(shape, max_len, keep_all, rho_c if weights is None else rho_c[0])
    n = enumerate_words(model, max_len, s, weights)
    assert n == 2 ** (max_len + 1) - 1
    return s
