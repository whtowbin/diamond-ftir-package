"""Record current pipeline outputs so refactors can be checked for unchanged behaviour.

    uv run python scripts/make_snapshot.py
        synthetic spectra only -> tests/data/snapshot.json (committed)
    uv run python scripts/make_snapshot.py --local /path/to/spectra
        your own spectra -> local_data/snapshot_local.json (git-ignored, never committed)

Snapshots record what the code produced, NOT independently verified values. Local
snapshot keys are anonymised (sample_001, ...) with a private key file next to them.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from diamond_ftir_package.params import AnalysisParams
from diamond_ftir_package.pipeline import collect_files, load_spectrum, run_analysis
from tests.synthetic import make_spectrum

SYNTHETIC_CASES = {
    "synthetic_IaAB": {
        "a_ppm": 200,
        "b_ppm": 300,
        "h3107_height": 0.5,
        "platelet_height": 1.0,
        "noise": 0.0005,
    },
    "synthetic_Ib": {"c_ppm": 100, "a_ppm": 10, "noise": 0.0005},
    "synthetic_IaA_thick": {"a_ppm": 400, "noise": 0.0005, "thickness_cm": 0.3},
}
LOCAL_SNAPSHOT = ROOT / "local_data" / "snapshot_local.json"
LOCAL_KEY = ROOT / "local_data" / "snapshot_local_key.json"


def _floats(row: dict) -> dict:
    return {k: float(v) for k, v in row.items() if not isinstance(v, str)}


def synthetic_snapshot() -> dict:
    params = AnalysisParams(run_amber=True)
    return {
        name: _floats(run_analysis(make_spectrum(**kw)[0], params))
        for name, kw in SYNTHETIC_CASES.items()
    }


def local_snapshot(folder: Path) -> tuple[dict, dict]:
    """Results keyed sample_001..., and the private key mapping keys to file names."""
    params = AnalysisParams(run_amber=True)
    results, key = {}, {}
    for n, path in enumerate(collect_files(folder), start=1):
        name = f"sample_{n:03d}"
        key[name] = path.name
        try:
            results[name] = _floats(run_analysis(load_spectrum(path), params))
        except Exception as e:  # noqa: BLE001 - record failures too; they are behaviour
            results[name] = {"error": str(e)}
    return results, key


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=Path, help="Folder of your own spectra.")
    args = parser.parse_args()
    if args.local:
        results, key = local_snapshot(args.local)
        LOCAL_SNAPSHOT.parent.mkdir(exist_ok=True)
        LOCAL_SNAPSHOT.write_text(
            json.dumps({"folder": str(args.local), **results}, indent=1)
        )
        LOCAL_KEY.write_text(json.dumps(key, indent=1))
        print(f"Wrote {LOCAL_SNAPSHOT} ({len(results)} spectra; not committed)")
    else:
        target = ROOT / "tests" / "data" / "snapshot.json"
        target.write_text(json.dumps(synthetic_snapshot(), indent=1))
        print(f"Wrote {target}")
