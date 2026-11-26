import numpy as np
import pytest

from holonomy_dm import sweeps
from holonomy_dm.protocols import (Model, WITNESS, averaged_distance, distance,
                                   gaussian_ensemble, ramsey_distance, ramsey_optimised)

GAMMA_STAR = 0.5 * np.log(8 / 7)


def test_witness_dephasing_closed_form():
    g = np.linspace(0, 2, 50)
    lam = np.exp(-g)
    assert np.allclose(distance(WITNESS, Model(gamma=g)), np.sqrt(3) / 16 * lam * (1 + 8 * lam ** 2), atol=1e-13)
    assert np.allclose(distance("AB", Model(gamma=g)), np.sqrt(3) / 2 * lam, atol=1e-13)


def test_dephasing_threshold():
    assert abs(sweeps.witness_threshold("gamma", 0.01, 0.2) - GAMMA_STAR) < 1e-10


def test_ramsey_closed_forms():
    g = np.array([0.0, 0.1, 0.7])
    d, theta, s = ramsey_optimised(Model(gamma=g))
    s_star = (2 / np.pi) * np.arctan2(np.pi / 2, g)
    assert np.allclose(d, np.exp(-g * s_star) * np.sin(s_star * np.pi / 2), atol=1e-9)
    assert np.allclose(theta, np.pi / 2, atol=1e-4) and np.allclose(s, s_star, atol=1e-4)
    assert np.isclose(ramsey_distance(np.pi / 2, 1.0, Model(delta=0.1)), np.cos(0.05 * np.pi))


def test_gaussian_ensemble_matches_analytic_average():
    sig = np.array([0.05, 0.2, 0.4])
    ens, w = gaussian_ensemble(Model(), sigma_eta=sig)
    assert np.allclose(averaged_distance("AB", ens, w),
                       np.sqrt(3) / 2 * np.exp(-(np.pi * sig / 2) ** 2 / 2), atol=1e-12)
    ens, w = gaussian_ensemble(Model(), sigma_delta=sig)
    assert np.allclose(averaged_distance("AB", ens, w),
                       np.sqrt(3) / 2 * np.exp(-(np.pi * sig / 3) ** 2 / 2), atol=1e-12)
    # long grouped word: aliasing would show up here first
    ens, w = gaussian_ensemble(Model(), sigma_eta=sig)
    exact21 = np.sqrt(3) / 2 * np.abs(np.sin(21 * np.pi / 2)) * np.exp(-(21 * np.pi * sig / 2) ** 2 / 2)
    assert np.allclose(averaged_distance("A" + "B" * 21, ens, w), exact21, atol=1e-10)


def test_regression_against_committed_results(results):
    """Recompute a spread of committed numbers; fails if any implementation drifts."""
    axes = results["sweeps_summary"]["axes"]
    assert abs(axes["gamma"]["witness_beats_grouped9_window"][1]["value"] - GAMMA_STAR) < 1e-9
    lo, hi = (e["value"] for e in axes["delta"]["witness_beats_grouped9_window"])
    assert abs(sweeps.witness_threshold("delta", -0.05, -0.02) - lo) < 1e-9
    assert abs(sweeps.witness_threshold("delta", 0.02, 0.05) - hi) < 1e-9
    pts = {"gamma": [0.0, 0.05, 0.3], "delta": [-0.1, 0.02], "eta": [-0.2, 0.1], "rho_c": [0.1, 0.5]}
    import csv
    from pathlib import Path
    res_dir = Path(__file__).resolve().parents[1] / "results"
    for axis, values in pts.items():
        with open(res_dir / f"sweep_{axis}.csv") as fh:
            rows = list(csv.DictReader(fh))
        xs = np.array([float(r[axis]) for r in rows])
        new = sweeps.sweep_1d(axis, np.array(values), max_len=9)
        for j, v in enumerate(values):
            row = rows[int(np.argmin(np.abs(xs - v)))]
            assert abs(float(row[axis]) - v) < 1e-12
            for key in ("witness", "best_all_9", "best_grouped_9", "grouped_long", "ramsey_oracle"):
                assert float(row[key]) == pytest.approx(float(new[key][j]), abs=2e-8), (axis, v, key)


def test_committed_exact_and_ideal_results(results):
    assert results["exact"]["witness"]["D_squared"] == "243/256"
    assert results["exact"]["unlimited_grouped_ceiling"]["ceiling_D_squared"] == "3/4"
    by = results["ideal_scan"]["by_budget"]
    for L in range(1, 9):
        assert by[str(L)]["fraction_of_grid_with_gap"] == 0
    assert by["9"]["best_word_at_pi_over_3"] == WITNESS
    for L in range(9, 13):
        assert by[str(L)]["max_gap"] == pytest.approx(9 * np.sqrt(3) / 16 - np.sqrt(3) / 2, abs=1e-9)
    search = results["exact_universal_search_depth12"]["by_length"]
    assert [search[str(L)]["n_witnesses"] for L in range(13)] == [0] * 9 + [3, 14, 48, 144]
