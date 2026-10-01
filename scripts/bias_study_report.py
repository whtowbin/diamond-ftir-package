"""Summarise scripts/bias_study.py results.

    uv run python scripts/bias_study_report.py local_data/bias_study/results.jsonl [--anonymise]

--anonymise replaces dataset labels with D1, D2, ... (quality level kept) so the tables
can be pasted into committed docs.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METHODS = (
    "multistage",
    "multistage_no_rubber",
    "single_fixed",
    "single_opt",
    "joint_1e8",
    "joint_1e9",
)


def _med(v, digits=2):
    v = np.asarray([a for a in v if a is not None and np.isfinite(a)], dtype=float)
    if v.size == 0:
        return "-"
    return f"{np.median(v):.{digits}f}"


def _pct(v, q):
    v = np.asarray([a for a in v if a is not None and np.isfinite(a)], dtype=float)
    return f"{np.percentile(v, q):.2f}" if v.size else "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--anonymise", action="store_true")
    args = ap.parse_args()
    rows = [
        json.loads(line)
        for line in args.results.read_text().splitlines()
        if line.strip()
    ]

    labels = sorted(
        {r["dataset"] for r in rows},
        key=lambda d: [r["dataset"] for r in rows].index(d),
    )
    alias = {d: (f"D{i + 1}" if args.anonymise else d) for i, d in enumerate(labels)}
    by = defaultdict(list)
    for r in rows:
        by[r["dataset"]].append(r)

    print("## Outcomes and quality by dataset\n")
    print(
        "| dataset | quality | n | usable | N saturated | diamond saturated | load/fit error | spacing | noise (cm⁻¹, norm.) | thickness (cm) | platelet (cm⁻¹) | diamond R² |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in labels:
        rs = by[d]
        c = defaultdict(int)
        for r in rs:
            c[r["outcome"]] += 1
        g = [r for r in rs if "t_ref" in r]
        print(
            f"| {alias[d]} | {rs[0]['quality']} | {len(rs)} | {c['usable']} | {c['nitrogen_saturated']} | "
            f"{c['diamond_saturated']} | {c['load_error'] + c['fit_error']} | "
            f"{_med([r.get('spacing') for r in rs])} | {_med([r.get('noise_norm') for r in g], 3)} | "
            f"{_med([r.get('t_ref') for r in g], 3)} | {_med([r.get('platelet') for r in g], 1)} | "
            f"{_med([r.get('diamond_r2') for r in g], 2)} |"
        )

    print(
        "\n## Thickness recovery by baseline (1.00 = unbiased); median, and p10 in brackets\n"
    )
    print("| dataset | " + " | ".join(METHODS) + " |")
    print("|---|" + "---|" * len(METHODS))
    use = [r for r in rows if r["outcome"] == "usable"]
    for d in [*labels, "ALL"]:
        rs = use if d == "ALL" else [r for r in use if r["dataset"] == d]
        if not rs:
            continue
        cells = []
        for m in METHODS:
            v = [r["methods"].get(m, {}).get("thickness_recovery") for r in rs]
            cells.append(f"{_med(v)} [{_pct(v, 10)}]")
        print(f"| {alias.get(d, d)} ({len(rs)}) | " + " | ".join(cells) + " |")

    print(
        "\n## Nitrogen window: N relative to 950-1350, and leakage (ppm per 1 cm⁻¹ feature)\n"
    )
    windows = list(use[0]["windows"]) if use else []
    print("| dataset | " + " | ".join(windows) + " |")
    print("|---|" + "---|" * len(windows))
    for d in [*labels, "ALL"]:
        rs = use if d == "ALL" else [r for r in use if r["dataset"] == d]
        if not rs:
            continue
        cells = []
        for w in windows:
            rel = [
                (r["windows"][w]["N"] / r["windows"]["950-1350"]["N"] - 1) * 100
                for r in rs
                if r["windows"]["950-1350"]["N"] > 1
            ]
            cells.append(f"{_med(rel, 1)}%")
        print(f"| {alias.get(d, d)} | " + " | ".join(cells) + " |")
    if use:
        print("\n| leakage (all usable) | " + " | ".join(windows) + " |")
        print("|---|" + "---|" * len(windows))
        for key in ("leak_platelet", "leak_low"):
            print(
                f"| {key} | "
                + " | ".join(
                    _med([r["windows"][w][key] for r in use], 1) for w in windows
                )
                + " |"
            )

    print("\n## Diamond-likeness (R² to the type IIa shape) by dataset\n")
    print("| dataset | p5 | p25 | median | fraction R² < 0.8 |")
    print("|---|---|---|---|---|")
    for d in labels:
        v = [r["diamond_r2"] for r in by[d] if "diamond_r2" in r]
        if v:
            print(
                f"| {alias[d]} | {_pct(v, 5)} | {_pct(v, 25)} | {_med(v)} | {np.mean(np.array(v) < 0.8):.2f} |"
            )


if __name__ == "__main__":
    main()
