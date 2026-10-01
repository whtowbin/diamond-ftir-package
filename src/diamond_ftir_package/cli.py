"""Command line interface: ``diamond-ftir run <file-or-folder> [-o out.csv] [--params settings.json]``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .params import AnalysisParams, MapParams
from .pipeline import analyze_paths, collect_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="diamond-ftir", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Analyse a spectrum file or a folder of spectra.")
    run.add_argument("path", type=Path, help="CSV/SPA/SPC file, or a folder of them.")
    run.add_argument(
        "-o", "--output", type=Path, help="Write results to this CSV (default: print)."
    )
    run.add_argument(
        "--params", type=Path, help="Settings JSON (see 'diamond-ftir defaults')."
    )
    run.add_argument(
        "--amber", action="store_true", help="Also measure amber-centre bands."
    )
    run.add_argument(
        "--duration-ma",
        type=float,
        help="Mantle residence time (Ma): adds nitrogen and platelet model temperatures.",
    )
    run.add_argument(
        "--recipe",
        action="append",
        default=[],
        help="Extra analysis recipe (built-in name or .toml/.json file); repeatable.",
    )

    mp = sub.add_parser("map", help="Analyse an OMNIC .map file pixel by pixel.")
    mp.add_argument("path", type=Path, help="The .map file.")
    mp.add_argument(
        "-o", "--output", type=Path, default=Path("Results"), help="Output folder."
    )
    mp.add_argument(
        "--params", type=Path, help="Settings JSON (may include a 'map' section)."
    )
    mp.add_argument(
        "--jobs", type=int, help="Worker processes (0 = all cores but one, 1 = serial)."
    )
    mp.add_argument(
        "--baseline-search",
        choices=["fixed", "optimize", "grid"],
        help="Override baseline_search.",
    )
    mp.add_argument(
        "--baseline-method",
        choices=["joint", "multistage", "single"],
        help="Override baseline_method.",
    )
    mp.add_argument(
        "--strategy",
        choices=["per_pixel", "calibrate_fixed", "calibrate_local"],
        help="How the baseline search is shared across pixels.",
    )
    thick = mp.add_mutually_exclusive_group()
    thick.add_argument(
        "--thickness-um", type=float, help="Use this known thickness (µm)."
    )
    thick.add_argument(
        "--survey-thickness",
        action="store_true",
        help="Fit a smooth thickness field from a coarse survey (polished plates); edge and "
        "outlier pixels keep their own fitted thickness.",
    )
    mp.add_argument(
        "--no-examples", action="store_true", help="Skip example fit plots."
    )
    mp.add_argument(
        "--duration-ma",
        type=float,
        help="Mantle residence time (Ma): adds nitrogen and platelet model temperatures.",
    )
    mp.add_argument(
        "--recipe",
        action="append",
        default=[],
        help="Extra analysis recipe (built-in name or .toml/.json file); repeatable.",
    )

    sub.add_parser("defaults", help="Print the default settings as JSON.")
    return parser


def _recipe_ref(ref: str) -> str:
    """Built-in names stay as they are; files become absolute paths."""
    from .core import builtin_recipes, resolve_recipe

    if ref not in builtin_recipes():
        ref = str(Path(ref).resolve())
    problems = resolve_recipe(ref).validate()
    if problems:
        raise SystemExit(f"recipe {ref}: " + "; ".join(problems))
    return ref


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "defaults":
        print(
            json.dumps(
                AnalysisParams().to_dict() | {"map": MapParams().to_dict()}, indent=2
            )
        )
        return 0
    if args.command == "map":
        return run_map(args)

    params = AnalysisParams()
    if args.params:
        params = AnalysisParams.from_dict(json.loads(args.params.read_text()))
    if args.amber:
        params.run_amber = True
    if args.duration_ma:
        params.thermometry.duration_ma = args.duration_ma
    if args.recipe:
        params.recipes = tuple(params.recipes) + tuple(
            _recipe_ref(r) for r in args.recipe
        )

    files = collect_files(args.path)
    if not files:
        print(f"No supported spectra found in {args.path}", file=sys.stderr)
        return 1

    rows = analyze_paths(
        files,
        params,
        progress=lambda i, n, name: print(f"[{i}/{n}] {name}", file=sys.stderr),
    )
    table = pd.DataFrame(rows)
    if args.output:
        table.to_csv(args.output, index=False)
        print(f"Wrote {len(table)} rows to {args.output}", file=sys.stderr)
    else:
        print(table.to_string(index=False))
    return 0 if (table["Status"] == "OK").all() else 2


if __name__ == "__main__":
    raise SystemExit(main())


def run_map(args: argparse.Namespace) -> int:
    from .maps import load_map, process_map, save_example_fits, save_map_outputs

    settings = json.loads(args.params.read_text()) if args.params else {}
    params = AnalysisParams.from_dict(settings)
    map_params = MapParams.from_dict(settings.get("map", {}))
    if args.jobs is not None:
        map_params.n_jobs = args.jobs
    if args.baseline_search:
        params.diamond.baseline_search = args.baseline_search
    if args.duration_ma:
        params.thermometry.duration_ma = args.duration_ma
    if args.recipe:
        params.recipes = tuple(params.recipes) + tuple(
            _recipe_ref(r) for r in args.recipe
        )
    if args.baseline_method:
        params.diamond.baseline_method = args.baseline_method
    if args.strategy:
        map_params.baseline_strategy = args.strategy
    if args.thickness_um:
        map_params.thickness_mode, map_params.known_thickness_um = (
            "known",
            args.thickness_um,
        )
    elif args.survey_thickness:
        map_params.thickness_mode = "survey"

    stem = args.path.stem
    data = load_map(args.path)
    step = max(1, data.sizes["y"] * data.sizes["x"] // 20)

    def progress(done: int, total: int) -> None:
        if done % step < 64 or done == total:
            print(f"  {done}/{total} pixels", file=sys.stderr)

    result = process_map(data, params, map_params, progress=progress)
    written = save_map_outputs(result, args.output, stem)
    if not args.no_examples:
        written += save_example_fits(
            data, args.output, stem, params, map_params, result
        )
    a = result.attrs
    print(
        f"{a['pixels_ok']} of {a['pixels_on_sample']} on-sample pixels analysed "
        f"({a['pixels_error']} errors). {len(written)} files in {args.output}",
        file=sys.stderr,
    )
    return 0 if a["pixels_error"] == 0 else 2
