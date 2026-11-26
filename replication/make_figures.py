#!/usr/bin/env python
"""Build the figures in replication/figures from the CSV files in replication/results.

    python replication/make_figures.py [--results DIR] [--out DIR]

Figures are written as vector PDF and 300-dpi PNG.  Colour identity is always
doubled by line style and a legend, so the figures survive greyscale printing.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent

INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
# fixed series identity (colour, line style) used in every panel
SERIES = {
    "witness": ("#257de0", "-", "witness ABABABAAB"),
    "best_all": ("#f47230", "--", "best word, any order"),
    "best_grouped": ("#00b894", "-.", "best grouped word"),
    "grouped_long": ("#f2a600", ":", "best grouped, up to 60 pulses"),
    "ramsey_oracle": ("#ee7fac", (0, (5, 1, 1, 1, 1, 1)), "optimised Ramsey"),
}
DEPTH = 12  # overwritten in main() from results/sweeps_summary.json
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#ffffff", "#cce3fd", "#84b9f3", "#338ced", "#1b64be", "#0e3d7e"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "savefig.dpi": 300, "figure.dpi": 150,
})


def read_csv(path):
    with open(path) as fh:
        rows = list(csv.reader(fh))
    cols = {}
    for i, name in enumerate(rows[0]):
        vals = [r[i] for r in rows[1:]]
        try:
            cols[name] = np.array(vals, float)
        except ValueError:
            cols[name] = np.array(vals)
    return cols


def line(ax, x, y, key, label=True, **kw):
    c, ls, lab = SERIES[key]
    ax.plot(x, y, color=c, ls=ls, label=lab if label else None, **kw)


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", out / name)


def fig_ideal(res, out):
    d = read_csv(res / "ideal_scan_by_budget.csv")
    budgets = [b for b in (8, 9, DEPTH) if b in set(d["budget"].astype(int))]
    fig, axes = plt.subplots(len(budgets), 1, figsize=(6.3, 1.75 * len(budgets)), sharex=True)
    for ax, b in zip(np.atleast_1d(axes), budgets):
        m = d["budget"] == b
        x, a, g = d["eps_over_pi"][m], d["best_all"][m], d["best_grouped"][m]
        ax.fill_between(x, g, a, where=(a - g) > 1e-9, color=SERIES["best_all"][0], alpha=0.18, lw=0)
        line(ax, x, g, "best_grouped")
        line(ax, x, a, "best_all", lw=1.1)
        ax.set_ylim(0.8, 1.01)
        ax.set_ylabel("trace distance D")
        frac = np.mean((a - g) > 1e-9)
        ax.set_title(f"Budget {b} operations — strict gain on {100 * frac:.0f}% of the pulse-angle grid"
                     if frac else f"Budget {b} operations — no gain anywhere on the grid", loc="left")
    axes[-1].set_xlabel(r"pulse angle $\varepsilon/\pi$")
    axes[0].legend(loc="lower center", ncol=2)
    fig.tight_layout()
    save(fig, out, "fig1_ideal_budget_scan")


def fig_dephasing(res, out, summary):
    d = read_csv(res / "sweep_gamma.csv")
    x = d["gamma"]
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    line(ax, x, d["ramsey_oracle"], "ramsey_oracle")
    line(ax, x, d[f"best_grouped_{DEPTH}"], "best_grouped")
    line(ax, x, d[f"best_all_{DEPTH}"], "best_all")
    line(ax, x, d["witness"], "witness")
    g = summary["axes"]["gamma"]["witness_beats_grouped9_window"][1]["value"]
    ax.axvline(g, color=INK2, lw=0.8)
    ax.annotate(f"witness = grouped\n" + r"$\gamma^\ast=\frac{1}{2}\ln\frac{8}{7}\approx$" + f"{g:.4f}",
                (g, 0.42), xytext=(8, 0), textcoords="offset points", color=INK, fontsize=7, va="center")
    ax.set_xlabel(r"dephasing per sensing window $\gamma=\tau/T_2$")
    ax.set_ylabel("trace distance D")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0.2, 1.02)
    ax.set_title(rf"Pure dephasing, $\varepsilon=\pi/3$, words up to {DEPTH} operations", loc="left")
    ax.legend(loc="upper right")
    save(fig, out, "fig2_dephasing")


def fig_calibration(res, out):
    panels = [("delta", r"systematic pulse-area error $\delta$", "Known pulse-area offset"),
              ("eta", r"systematic field-magnitude error $\eta$", "Known field-magnitude offset"),
              ("sigma_delta", r"pulse-area spread $\sigma_\delta$", "Shot-to-shot pulse-area spread"),
              ("sigma_eta", r"field-magnitude spread $\sigma_\eta$", "Shot-to-shot field-magnitude spread")]
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.0), sharey=True)
    for ax, (name, xlabel, title) in zip(axes.flat, panels):
        d = read_csv(res / f"sweep_{name}.csv")
        x = d[name]
        line(ax, x, d["ramsey_oracle"], "ramsey_oracle")
        line(ax, x, d["grouped_long"], "grouped_long", lw=1.2)
        line(ax, x, d[f"best_grouped_{DEPTH}"], "best_grouped")
        line(ax, x, d[f"best_all_{DEPTH}"], "best_all")
        line(ax, x, d["witness"], "witness")
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left")
        ax.set_ylim(0.4, 1.02)
        ax.set_xlim(x.min(), x.max())
    for ax in axes[:, 0]:
        ax.set_ylabel("trace distance D")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, out, "fig3_calibration")


def fig_control(res, out):
    d = read_csv(res / "sweep_rho_c.csv")
    x = d["rho_c"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
    ax = axes[0]
    line(ax, x, d["ramsey_oracle"], "ramsey_oracle")
    line(ax, x, d[f"best_grouped_{DEPTH}"], "best_grouped")
    line(ax, x, d[f"best_all_{DEPTH}"], "best_all")
    line(ax, x, d["witness"], "witness")
    ax.set_title(f"Equal budget: up to {DEPTH} operations", loc="left")
    ax = axes[1]
    line(ax, x, d["grouped_time"], "best_grouped")
    line(ax, x, d["best_all_time"], "best_all")
    line(ax, x, d["witness"], "witness")
    ax.set_title("Equal duration: no longer than the witness", loc="left")
    for ax in axes:
        ax.set_xlabel(r"control-pulse duration / sensing window, $\rho_c$")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.03)
    axes[0].set_ylabel("trace distance D")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    save(fig, out, "fig4_control_duration")


def fig_maps(res, out):
    specs = [("eps_gamma", "eps", "gamma", r"pulse angle $\varepsilon/\pi$", r"dephasing $\gamma$", np.pi),
             ("delta_eta", "delta", "eta", r"pulse-area error $\delta$", r"field error $\eta$", 1.0),
             ("rhoc_gamma", "rho_c", "gamma", r"control duration $\rho_c$", r"dephasing $\gamma$", 1.0)]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), layout="constrained")
    for ax, (name, kx, ky, lx, ly, sx) in zip(axes, specs):
        d = read_csv(res / f"map_{name}.csv")
        xs, ys = np.unique(d[kx]), np.unique(d[ky])
        shape = (len(xs), len(ys))
        gap = np.clip(d[f"gap_{DEPTH}"].reshape(shape), 0, None)
        im = ax.pcolormesh(xs / sx, ys, gap.T, cmap=SEQ, vmin=0, vmax=0.2, shading="nearest", rasterized=True)
        wit = (d["witness"] - d["best_grouped_9"]).reshape(shape)
        if (wit > 0).any() and (wit < 0).any():
            ax.contour(xs / sx, ys, wit.T, levels=[0], colors=[INK], linewidths=0.9)
        ax.set_xlabel(lx)
        ax.set_ylabel(ly)
        ax.grid(False)
    axes[0].set_title("Angle and dephasing", loc="left")
    axes[1].set_title("Calibration offsets", loc="left")
    axes[2].set_title("Pulse duration and dephasing", loc="left")
    axes[0].set_ylim(0, 0.12)
    cb = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.06, pad=0.04, aspect=60, extend="max")
    cb.set_label(f"gain in D of the best word over the best grouped word (both up to {DEPTH} operations; white = none)\n"
                 "black line: boundary of the region where the witness beats every grouped word of up to 9 operations")
    cb.outline.set_linewidth(0.4)
    save(fig, out, "fig5_advantage_maps")


def fig_counts(res, out):
    d = read_csv(res / "equal_counts.csv")
    na, nb = d["n_control"].astype(int), d["n_sensing"].astype(int)
    n = int(max(na.max(), nb.max()))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2), sharey=True)
    for ax, key, title in ((axes[0], "pi3_gap_vs_at_most_counts", r"At $\varepsilon=\pi/3$"),
                           (axes[1], "sup_eps_gap_vs_at_most_counts", r"Largest over all $\varepsilon$")):
        grid = np.full((n + 1, n + 1), np.nan)
        grid[na, nb] = np.clip(d[key], 0, None)
        im = ax.pcolormesh(np.arange(n + 2) - 0.5, np.arange(n + 2) - 0.5, grid, cmap=SEQ,
                           vmin=0, vmax=0.12, edgecolors="white", linewidth=1.0)
        for a, b, v in zip(na, nb, d[key]):
            if v > 0.005:
                ax.text(b, a, f"{v:.2f}".lstrip("0"), ha="center", va="center", fontsize=5.2,
                        color=INK)
        ax.set_aspect("equal")
        ax.set_xlim(-0.5, n + 0.5)
        ax.set_ylim(-0.5, n + 0.5)
        ax.set_xticks(range(0, n + 1, 2))
        ax.set_yticks(range(0, n + 1, 2))
        ax.set_xlabel("sensing windows used")
        ax.set_title(title, loc="left")
        ax.grid(False)
    axes[0].set_ylabel("control pulses used")
    cb = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
    cb.set_label("gain over grouped words with\nno more pulses and no more windows")
    cb.outline.set_linewidth(0.4)
    save(fig, out, "fig6_equal_counts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--out", type=Path, default=HERE / "figures")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((a.results / "sweeps_summary.json").read_text())
    global DEPTH
    DEPTH = int(summary["depth"])
    fig_ideal(a.results, a.out)
    fig_dephasing(a.results, a.out, summary)
    fig_calibration(a.results, a.out)
    fig_control(a.results, a.out)
    fig_maps(a.results, a.out)
    fig_counts(a.results, a.out)


if __name__ == "__main__":
    main()
