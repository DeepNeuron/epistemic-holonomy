import numpy as np

from holonomy_dm import channels as ch
from holonomy_dm.enumeration import summarise
from holonomy_dm.protocols import Model, WITNESS, all_words, distance, is_grouped

EPS = np.linspace(0, np.pi, 721)


def test_witness_value():
    assert abs(distance(WITNESS, Model(), reference=True) - 9 * np.sqrt(3) / 16) < 1e-14


def test_grouped_closed_form():
    eps = np.linspace(0.05, 3.0, 60)
    for k in range(0, 7):
        for m in range(0, 6):
            d = distance("A" * k + "B" * m, Model(eps=eps))
            assert np.allclose(d, np.abs(np.sin(k * eps) * np.sin(m * np.pi / 2)), atol=1e-13)
            assert np.allclose(distance("B" * m + "A" * k, Model(eps=eps)), 0, atol=1e-13)


def test_enumeration_is_exhaustive_and_unpruned():
    s = summarise(Model(eps=EPS[:3]), 6)
    assert sorted(s.words) == sorted(all_words(6))
    assert sum(is_grouped(w) for w in s.words) == 1 + 2 + sum(2 * n for n in range(2, 7))


def test_no_gap_through_depth_8_and_gap_at_9_numerical():
    s = summarise(Model(eps=EPS), 9)
    gap = s.budget("all") - s.budget("grouped")
    assert np.max(np.abs(gap[:9])) < 1e-12          # numerical evidence only
    i = int(np.argmin(np.abs(EPS - np.pi / 3)))
    assert abs(gap[9, i] - (9 * np.sqrt(3) / 16 - np.sqrt(3) / 2)) < 1e-13
    assert abs(gap[9].max() - gap[9, i]) < 1e-3


def test_agrees_with_repository_bloch_polynomial(certificate):
    """Cross-implementation check: density matrices vs the certificate's recurrence."""
    import sympy as sp
    eps = np.array([0.31, 1.0471975511965976, 1.9, 2.77])
    worst = 0.0
    for word in all_words(9, 1):
        y = sp.lambdify(certificate.t, certificate.scaled_y(word), "numpy")
        ref = np.abs(np.sin(eps) * np.broadcast_to(y(np.cos(eps)), eps.shape))
        worst = max(worst, float(np.max(np.abs(distance(word, Model(eps=eps)) - ref))))
    assert worst < 1e-12


def _axis(v):
    v = np.asarray(v, float) / np.linalg.norm(v)
    return v[0] * ch.SX + v[1] * ch.SY + v[2] * ch.SZ


def _d2(word, ua, ub):
    out = []
    for h in (1, -1):
        rho = ch.RHO0.copy()
        for name in word:
            u = ua if name == "A" else (ub if h == 1 else ub.conj().T)
            rho = u @ rho @ u.conj().T
        out.append(rho)
    return float(ch.trace_distance(*out)) ** 2


def test_tilted_axis_cases_from_the_certificate():
    cases = {"BAB": (ch.rot(_axis([0, 1, 0]), np.pi), ch.rot(_axis([0, 1, 1]), np.pi / 3), 27 / 32, 3 / 8),
             "ABAB": (ch.rot(_axis([0, 1, 1]), np.pi / 2), ch.rot(_axis([0, 0, 1]), np.pi / 3), 249 / 256, 3 / 4)}
    for word, (ua, ub, value, ceiling) in cases.items():
        assert abs(_d2(word, ua, ub) - value) < 1e-13
        grouped = max(_d2(g, ua, ub) for k in range(25) for m in range(25)
                      for g in ("A" * k + "B" * m, "B" * m + "A" * k))
        assert abs(grouped - ceiling) < 1e-12       # also holds for long grouped words
