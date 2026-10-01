"""Refactors must not change results: compare against the recorded snapshots.

Snapshots record what the code produced, not independently verified truth. Regenerate
them with scripts/make_snapshot.py only when a change in results is intended and understood.
"""

import json
import math
import os
from pathlib import Path

import pytest

from scripts.make_snapshot import LOCAL_SNAPSHOT, local_snapshot, synthetic_snapshot

SNAPSHOT = Path(__file__).parent / "data" / "snapshot.json"


def _compare(expected: dict, actual: dict) -> None:
    for sample, values in expected.items():
        for key, value in values.items():
            got = actual[sample][key]
            if isinstance(value, str) or isinstance(got, str):
                assert got == value, f"{sample} {key}"
            elif math.isnan(value):
                assert math.isnan(got), f"{sample} {key}"
            else:
                assert got == pytest.approx(value, rel=1e-9, abs=1e-12), (
                    f"{sample} {key}"
                )


def test_synthetic_results_match_snapshot():
    _compare(json.loads(SNAPSHOT.read_text()), synthetic_snapshot())


@pytest.mark.skipif(
    not (os.environ.get("DIAMOND_FTIR_LOCAL_DATA") and LOCAL_SNAPSHOT.exists()),
    reason="set DIAMOND_FTIR_LOCAL_DATA and run scripts/make_snapshot.py --local first",
)
def test_local_results_match_local_snapshot():
    """Opt-in regression on your own (uncommitted) spectra."""
    expected = json.loads(LOCAL_SNAPSHOT.read_text())
    expected.pop("folder")
    actual, _ = local_snapshot(Path(os.environ["DIAMOND_FTIR_LOCAL_DATA"]))
    _compare(expected, actual)
