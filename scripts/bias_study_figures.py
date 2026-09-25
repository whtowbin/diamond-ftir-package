"""Anonymised summary figures from scripts/bias_study.py results (no spectra are plotted).

    uv run python scripts/bias_study_figures.py local_data/bias_study_v2/results.jsonl

Writes docs/figures/study_*.png. Datasets appear only as D1, D2, ... with their quality
group, in the order of the local dataset list.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"
METHODS = {
    "multistage": ("multistage (old default)", "#b2182b"),
    "single_opt": ("single ASLS, searched", "#2166ac"),
    "joint_1e8": ("joint, lam 1e8", "#66bd63"),
    "joint_1e9": ("joint, lam 1e9", "#1b7837"),
}
OUTCOMES = {
    "usable": "#1b7837",
    "nitrogen_saturated": "#fdae61",
    "diamond_saturated": "#d73027",
    "load_error": "#999999",
    "fit_error": "#555555",
}


def load(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    order = list(dict.fromkeys(r["dataset"] for r in rows))
    alias = {d: f"D{i + 1}" for i, d in enumerate(order)}
    quality = {r["dataset"]: r["quality"] for r in rows}
    return rows, order, alias, quality


def xlabels(order, alias, quality):
    return [f"{alias[d]}\n{quality[d]}" for d in order]


def fig_recovery(rows, order, alias, quality):
    use = [r for r in rows if r["outcome"] == "usable"]
    fig, ax = plt.subplots(figsize=(16, 5.5))
    width = 0.8 / len(METHODS)
    for k, (m, (label, col)) in enumerate(METHODS.items()):
        data = []
        for d in order:
            v = [
                r["methods"].get(m, {}).get("thickness_recovery")
                for r in use
                if r["dataset"] == d
            ]
            data.append([a for a in v if a is not None and np.isfinite(a)] or [np.nan])
        pos = np.arange(len(order)) + (k - (len(METHODS) - 1) / 2) * width
        bp = ax.boxplot(
            data,
            positions=pos,
            widths=width * 0.9,
            patch_artist=True,
            showfliers=False,
            whis=(10, 90),
            medianprops={"color": "k"},
        )
        for b in bp["boxes"]:
            b.set_facecolor(col)
            b.set_alpha(0.8)
        ax.plot([], [], color=col, lw=8, label=label)
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_xticks(range(len(order)), xlabels(order, alias, quality), fontsize=8)
    ax.set_ylim(0.3, 1.3)
    ax.set_ylabel("recovered Δt / injected Δt")
    ax.set_title(
        "Thickness recovery by dataset and baseline (box: quartiles; whiskers: 10th-90th percentile; 1.00 = unbiased)"
    )
    ax.legend(ncol=4, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "study_thickness_recovery.png", dpi=110)
    plt.close(fig)


def fig_windows(rows, order, alias, quality):
    use = [r for r in rows if r["outcome"] == "usable"]
    windows = list(use[0]["windows"])
    grid = np.full((len(order), len(windows)), np.nan)
    for i, d in enumerate(order):
        rs = [r for r in use if r["dataset"] == d and r["windows"]["950-1350"]["N"] > 1]
        for j, w in enumerate(windows):
            if rs:
                grid[i, j] = np.median(
                    [
                        (r["windows"][w]["N"] / r["windows"]["950-1350"]["N"] - 1) * 100
                        for r in rs
                    ]
                )
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(
        np.clip(grid, -30, 30), cmap="RdBu_r", vmin=-30, vmax=30, aspect="auto"
    )
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isfinite(grid[i, j]):
                ax.text(
                    j, i, f"{grid[i, j]:+.0f}", ha="center", va="center", fontsize=8
                )
    ax.set_xticks(range(len(windows)), windows, rotation=30)
    ax.set_yticks(range(len(order)), [f"{alias[d]} ({quality[d]})" for d in order])
    ax.set_title("Median change in total N (%) versus the 950-1350 cm⁻¹ window")
    fig.colorbar(im, ax=ax, label="% change (colour clipped at ±30)")
    fig.tight_layout()
    fig.savefig(OUT / "study_window_shift.png", dpi=110)
    plt.close(fig)


def fig_r2(rows, order, alias, quality):
    fig, ax = plt.subplots(figsize=(14, 4.5))
    data = [
        [r["diamond_r2"] for r in rows if r["dataset"] == d and "diamond_r2" in r]
        or [np.nan]
        for d in order
    ]
    ax.boxplot(data, whis=(5, 95), showfliers=False)
    for i, v in enumerate(data):
        ax.scatter(
            np.full(len(v), i + 1)
            + np.random.default_rng(i).uniform(-0.2, 0.2, len(v)),
            v,
            s=4,
            alpha=0.3,
            color="0.3",
        )
    ax.axhline(0.8, color="#b2182b", ls="--", lw=1, label="flag threshold 0.8")
    ax.set_xticks(range(1, len(order) + 1), xlabels(order, alias, quality), fontsize=8)
    ax.set_ylim(-0.5, 1.05)
    ax.set_ylabel("diamond-likeness R²")
    ax.set_title(
        "Match to the type IIa diamond shape (mixed = instrument-labelled non-diamond, many actually diamond)"
    )
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "study_diamond_r2.png", dpi=110)
    plt.close(fig)


def fig_outcomes(rows, order, alias, quality):
    counts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        counts[r["dataset"]][r["outcome"]] += 1
    fig, ax = plt.subplots(figsize=(14, 4))
    bottom = np.zeros(len(order))
    for o, col in OUTCOMES.items():
        frac = np.array([counts[d][o] / max(1, sum(counts[d].values())) for d in order])
        ax.bar(
            range(len(order)), frac, bottom=bottom, color=col, label=o.replace("_", " ")
        )
        bottom += frac
    ax.set_xticks(range(len(order)), xlabels(order, alias, quality), fontsize=8)
    ax.set_ylabel("fraction of sampled spectra")
    ax.set_title("Outcome per sampled spectrum")
    ax.legend(ncol=5, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "study_outcomes.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = load(args.results)
    for f in (fig_outcomes, fig_recovery, fig_windows, fig_r2):
        f(*data)
        print("done", f.__name__)
