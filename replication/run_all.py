#!/usr/bin/env python
"""Regenerate every numerical result in replication/results.

    python replication/run_all.py                 # full run (about 15-25 min, 2 cores)
    python replication/run_all.py --quick         # reduced grids, depth 10 (CI smoke run)
    python replication/run_all.py --exact-depth 12  # exact universal search to depth 12 (slow)

Numerical output is numerical evidence only.  The only exact statements are
those produced by holonomy_dm.exact (written to results/exact.json).
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from holonomy_dm import exact, sweeps  # noqa: E402
from holonomy_dm.enumeration import summarise  # noqa: E402
from holonomy_dm.protocols import Model, WITNESS, is_grouped, switches  # noqa: E402

TOL = sweeps.GAP_TOL


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return float(f"{float(obj):.12g}")
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.str_,)):
        return str(obj)
    return obj


def dump(name, obj, out):
    path = out / name
    path.write_text(json.dumps(jsonable(obj), indent=1) + "\n")
    print(f"wrote {path}", flush=True)


def write_csv(name, columns: dict, out):
    path = out / name
    keys = list(columns)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(keys)
        for row in zip(*[np.asarray(columns[k]).reshape(-1) for k in keys]):
            w.writerow([f"{v:.10g}" if isinstance(v, (float, np.floating)) else v for v in row])
    print(f"wrote {path}", flush=True)


def versions():
    import matplotlib
    import scipy
    import sympy
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "sympy": sympy.__version__,
            "matplotlib": matplotlib.__version__, "platform": platform.platform(),
            "machine": platform.machine()}


# ---------------------------------------------------------------- stages ----

def stage_certificate(out):
    """Rerun the repository's own certificate unmodified and record what it prints."""
    script = HERE.parent / "epistemic_holonomy_certificate.py"
    runs = []
    for _ in range(3):
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
        runs.append(time.perf_counter() - t0)
    (out / "certificate_stdout.txt").write_text(proc.stdout)
    dump("certificate_run.json", {
        "command": "python epistemic_holonomy_certificate.py",
        "returncode": proc.returncode, "stderr": proc.stderr,
        "stdout_lines": proc.stdout.splitlines(),
        "wall_seconds_each_run": runs, "wall_seconds_median": float(np.median(runs)),
        "versions": versions(),
    }, out)


def stage_exact(out, depth):
    import sympy as sp
    t0 = time.perf_counter()
    w = exact.witness_exact()
    lam, poly = exact.witness_dephased_polynomial()
    crossing = sp.solve(sp.Eq(poly, sp.Rational(3, 4) * lam ** 2), lam)
    lam_star = [r for r in crossing if r.is_real and 0 < float(r) < 1]
    ceil = exact.grouped_ceiling_exact()
    t1 = time.perf_counter()
    search = exact.universal_search(depth, verbose=True)
    t2 = time.perf_counter()
    dump("exact.json", {
        "witness": {
            "word": WITNESS, "epsilon": "pi/3",
            "rho_plus": [[str(e) for e in w["rho_plus"].row(i)] for i in range(2)],
            "rho_minus": [[str(e) for e in w["rho_minus"].row(i)] for i in range(2)],
            "D_squared": str(w["D_squared"]), "D": str(w["D"]),
            "eigenvalues_of_rho_plus_minus_rho_minus": [str(e) for e in w["eigenvalues_of_difference"]],
            "difference_equals_this_times_sigma_y": str(w["difference_over_sigma_y"]),
            "purity_rho_plus": str(w["purity_plus"]),
        },
        "witness_under_dephasing": {
            "lambda": "exp(-gamma), coherence factor per sensing window",
            "D_squared_polynomial": str(poly),
            "grouped_AB_D_squared": "3*lambda**2/4",
            "crossing_lambda": [str(r) for r in lam_star],
            "crossing_gamma": [str(sp.simplify(-sp.log(r))) for r in lam_star],
            "crossing_gamma_float": [float(-sp.log(r)) for r in lam_star],
        },
        "unlimited_grouped_ceiling": {k: str(v) for k, v in ceil.items()},
        "universal_search": {"max_depth": depth, "by_length": search,
                             "seconds": t2 - t1},
        "seconds_witness_and_ceiling": t1 - t0,
    }, out)


def _intervals(x, mask):
    idx = np.flatnonzero(mask)
    if not idx.size:
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[cuts + 1]]
    ends = np.r_[idx[cuts], idx[-1]]
    return [[float(x[a]), float(x[b])] for a, b in zip(starts, ends)]


def stage_ideal(out, n_eps, depth):
    eps = np.linspace(0.0, np.pi, n_eps)
    i0 = int(np.argmin(np.abs(eps - np.pi / 3)))
    assert abs(eps[i0] - np.pi / 3) < 1e-12, "grid must contain pi/3"
    t0 = time.perf_counter()
    summ = summarise(Model(eps=eps), depth, keep_all=True)
    secs = time.perf_counter() - t0
    ball, bgrp = summ.budget("all"), summ.budget("grouped")
    arg = summ.budget_arg("all")
    table = np.array(summ.table)
    by_len = {}
    for L in range(1, depth + 1):
        gap = ball[L] - bgrp[L]
        pos = gap > TOL
        by_len[L] = {
            "max_gap": gap.max(), "eps_over_pi_at_max_gap": eps[gap.argmax()] / np.pi,
            "fraction_of_grid_with_gap": pos.mean(),
            "eps_over_pi_intervals_with_gap": [[a / np.pi, b / np.pi] for a, b in _intervals(eps, pos)],
            "best_word_at_pi_over_3": summ.words[arg[L, i0]],
            "best_all_at_pi_over_3": ball[L, i0], "best_grouped_at_pi_over_3": bgrp[L, i0],
            "max_abs_gap_if_no_advantage": float(np.abs(gap).max()) if not pos.any() else None,
        }
    # equal counts of control (A) and sensing (B) operations
    counts = []
    for (na, nb) in sorted(summ.count_all):
        if na + nb == 0:
            continue
        allv = summ.count_all[(na, nb)]
        g_exact = summ.count_grouped[(na, nb)]
        g_most = np.max([summ.count_grouped[(a, b)] for a in range(na + 1) for b in range(nb + 1)], axis=0)
        counts.append({
            "n_control": na, "n_sensing": nb,
            "pi3_best_any_order": allv[i0], "pi3_grouped_same_counts": g_exact[i0],
            "pi3_grouped_at_most_counts": g_most[i0],
            "pi3_gap_vs_same_counts": allv[i0] - g_exact[i0],
            "pi3_gap_vs_at_most_counts": allv[i0] - g_most[i0],
            "sup_eps_gap_vs_same_counts": float((allv - g_exact).max()),
            "sup_eps_gap_vs_at_most_counts": float((allv - g_most).max()),
            "eps_over_pi_at_sup_gap_vs_at_most": float(eps[(allv - g_most).argmax()] / np.pi),
        })
    dump("ideal_scan.json", {
        "description": "noise-free model, all words to the stated depth, eps grid on [0, pi]",
        "depth": depth, "n_words": len(summ.words), "n_eps": n_eps, "gap_tolerance": TOL,
        "enumeration_seconds": secs, "by_budget": by_len, "equal_counts": counts,
    }, out)
    write_csv("ideal_scan_by_budget.csv", {
        "eps_over_pi": np.tile(eps / np.pi, depth),
        "budget": np.repeat(np.arange(1, depth + 1), n_eps),
        "best_all": ball[1:].reshape(-1), "best_grouped": bgrp[1:].reshape(-1),
        "gap": (ball[1:] - bgrp[1:]).reshape(-1)}, out)
    write_csv("all_words_at_pi_over_3.csv", {
        "word": np.array(summ.words), "length": np.array([len(w) for w in summ.words]),
        "n_control": np.array([w.count("A") for w in summ.words]),
        "n_sensing": np.array([w.count("B") for w in summ.words]),
        "switches": np.array([switches(w) for w in summ.words]),
        "grouped": np.array([int(is_grouped(w)) for w in summ.words]),
        "D": table[:, i0]}, out)
    write_csv("equal_counts.csv", {k: np.array([c[k] for c in counts]) for k in counts[0]}, out)


SCALAR_KEYS = ["witness", "best_all_9", "best_inter_9", "best_grouped_9", "gap_9",
               "best_all_12", "best_inter_12", "best_grouped_12", "gap_12",
               "best_all_time", "best_inter_time", "grouped_long", "grouped_time",
               "ramsey_oracle", "ramsey_oracle_theta", "ramsey_oracle_s", "ramsey_nominal"]


def _refine(axis, x, gapw):
    """Connected interval around the nominal point where the witness beats every
    grouped word of length <= 9, with both ends refined by root finding."""
    win = sweeps.last_positive(x, gapw, 0.0)
    if win is None:
        return None
    ends = []
    for edge, step in ((win[0], -1), (win[1], +1)):
        i = int(np.argmin(np.abs(x - edge)))
        j = i + step
        if j < 0 or j >= len(x):
            ends.append({"value": float(edge), "status": "sweep range exhausted (advantage persists to the end of the grid)"})
            continue
        ends.append({"value": sweeps.witness_threshold(axis, float(x[i]), float(x[j])),
                     "status": "root of witness - best grouped (<=9), bracketed on the grid"})
    return ends


def stage_sweeps(out, depth, thin):
    res_all = {}
    for axis, (values, label) in sweeps.AXES.items():
        values = values[::thin] if thin > 1 else values
        t0 = time.perf_counter()
        res = sweeps.sweep_1d(axis, values, max_len=depth)
        secs = time.perf_counter() - t0
        x = res["x"]
        keys = [k for k in SCALAR_KEYS if k in res] + [k for k in res if k.startswith(("best_all_", "best_grouped_", "best_inter_", "gap_")) and k not in SCALAR_KEYS]
        cols = {axis: x}
        cols.update({k: np.asarray(res[k], float) for k in keys})
        for k in res:
            if k.startswith("argbest"):
                cols[k] = np.asarray(res[k])
        arg = np.asarray(res["grouped_long_arg"])
        cols["grouped_long_word"] = np.array([
            ("A" if o == 0 else "B") + f"^{k if o == 0 else m} " + ("B" if o == 0 else "A") + f"^{m if o == 0 else k}"
            for o, k, m in arg.reshape(-1, 3)])
        write_csv(f"sweep_{axis}.csv", cols, out)
        gapw = res["witness"] - res[f"best_grouped_{min(9, depth)}"]
        info = {"label": label, "n_points": len(x), "seconds": secs,
                "nominal": {k: float(np.asarray(res[k])[int(np.argmin(np.abs(x)))]) for k in keys},
                "witness_beats_grouped9_window": _refine(axis, x, gapw)}
        for L in sorted({min(9, depth), depth}):
            info[f"any_word_le{L}_beats_grouped_le{L}_window_grid"] = sweeps.last_positive(x, res[f"gap_{L}"])
            info[f"fraction_of_sweep_with_gap_{L}"] = float((np.asarray(res[f"gap_{L}"]) > TOL).mean())
        info["fraction_of_sweep_where_best_word_beats_ramsey_oracle"] = float(
            (np.asarray(res[f"best_all_{depth}"]) > np.asarray(res["ramsey_oracle"]) + 1e-6).mean())
        info["first_x_where_best_word_beats_ramsey_oracle"] = next(
            (float(v) for v, a, b in zip(x, res[f"best_all_{depth}"], res["ramsey_oracle"]) if a > b + 1e-6), None)
        res_all[axis] = info
    dump("sweeps_summary.json", {"depth": depth, "nominal_eps": "pi/3",
                                 "K_MAX": sweeps.K_MAX, "M_MAX": sweeps.M_MAX,
                                 "axes": res_all}, out)


def stage_maps(out, depth, n):
    maps = {
        "eps_gamma": ("eps", np.linspace(0, np.pi, (9 * n) // 4 + 1), "gamma", np.linspace(0, 0.3, n // 2 + 1), {}),
        "delta_eta": ("delta", np.linspace(-0.1, 0.1, n + 1), "eta", np.linspace(-0.3, 0.3, n + 1), {}),
        "rhoc_gamma": ("rho_c", np.linspace(0, 0.5, n // 2 + 1), "gamma", np.linspace(0, 0.2, n // 2 + 1), {}),
    }
    meta = {}
    for name, (ax, xs, ay, ys, fixed) in maps.items():
        t0 = time.perf_counter()
        res = sweeps.sweep_2d(ax, xs, ay, ys, max_len=depth, **fixed)
        xg, yg = np.meshgrid(xs, ys, indexing="ij")
        keys = [k for k in SCALAR_KEYS if k in res] + [
            k for k in res if k.startswith(("best_all_", "best_grouped_", "best_inter_", "gap_"))
            and k not in SCALAR_KEYS]
        cols = {ax: xg, ay: yg}
        cols.update({k: np.asarray(res[k], float) for k in keys})
        write_csv(f"map_{name}.csv", cols, out)
        Lq = min(9, depth)
        meta[name] = {
            "x": ax, "y": ay, "shape": list(xg.shape), "seconds": time.perf_counter() - t0,
            f"fraction_gap_{Lq}": float((res[f"gap_{Lq}"] > TOL).mean()),
            f"fraction_gap_{depth}": float((res[f"gap_{depth}"] > TOL).mean()),
            "fraction_witness_beats_grouped9": float((res["witness"] > res[f"best_grouped_{Lq}"] + TOL).mean()),
            "fraction_witness_beats_grouped_long": float((res["witness"] > res["grouped_long"] + TOL).mean()),
        }
    dump("maps_summary.json", meta, out)


def stage_misc(out):
    from holonomy_dm.protocols import gaussian_ensemble
    d_ab = float(np.sqrt(3) / 2)
    conv = {}
    for nodes in (51, 101, 201):
        ens, w = gaussian_ensemble(Model(), sigma_eta=np.array([0.2, 0.4]), nodes=nodes)
        conv[f"sigma_eta_nodes_{nodes}"] = sweeps.grouped_family(ens, w)["grouped_long"]
        ens, w = gaussian_ensemble(Model(), sigma_delta=np.array([0.05, 0.15]), nodes=nodes)
        conv[f"sigma_delta_nodes_{nodes}"] = sweeps.grouped_family(ens, w)["grouped_long"]
    dump("misc.json", {
        "repeated_grouped_shots": {
            "note": ("N independent copies of the pure output states of AB at eps=pi/3 with a "
                     "collective optimal measurement; needs resets, which the restricted model forbids."),
            "D_single_AB": d_ab,
            "D_N_copies": {n: sweeps.multi_shot_grouped(n, d_ab) for n in (1, 2, 3, 4)},
            "traversals": {n: 2 * n for n in (1, 2, 3, 4)},
            "witness_D": 9 * np.sqrt(3) / 16,
        },
        "gauss_hermite_convergence_grouped_long": conv,
        "versions": versions(),
    }, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--exact-depth", type=int, default=None)
    ap.add_argument("--out", type=Path, default=HERE / "results")
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["certificate", "exact", "ideal", "sweeps", "maps", "misc"])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    depth = 10 if a.quick else 12
    todo = a.only or ["certificate", "exact", "ideal", "sweeps", "maps", "misc"]
    t0 = time.perf_counter()
    if "certificate" in todo:
        stage_certificate(a.out)
    if "exact" in todo:
        stage_exact(a.out, a.exact_depth or 9)
    if "ideal" in todo:
        stage_ideal(a.out, 361 if a.quick else 3601, depth)
    if "sweeps" in todo:
        stage_sweeps(a.out, depth, 10 if a.quick else 1)
    if "maps" in todo:
        stage_maps(a.out, depth, 20 if a.quick else 80)
    if "misc" in todo:
        stage_misc(a.out)
    print(f"total {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
