import json

import numpy as np
import pandas as pd

from diamond_ftir_package.cli import main

from .synthetic import make_spectrum


def _write_csv(path, **kw):
    s, _ = make_spectrum(**kw)
    pd.DataFrame({"wn": s.X, "abs": s.Y}).to_csv(path, index=False)


def test_run_folder_writes_csv_and_reports_bad_files(tmp_path):
    _write_csv(tmp_path / "good-1.csv", a_ppm=200, b_ppm=300, noise=0.0005)
    (tmp_path / "bad-2.csv").write_text("wn,abs\n1,2\n2,3\n3,4\n")
    out = tmp_path / "out.csv"
    code = main(["run", str(tmp_path), "-o", str(out)])
    table = pd.read_csv(out).set_index("Filename")
    assert table.loc["good-1.csv", "Status"] == "OK"
    assert table.loc["bad-2.csv", "Status"].startswith("Error")
    assert code == 2  # non-zero when any file failed


def test_defaults_command_prints_valid_json(capsys):
    assert main(["defaults"]) == 0
    assert "nitrogen" in json.loads(capsys.readouterr().out)


def test_csv_with_and_without_header_load_identically(tmp_path):
    from diamond_ftir_package.LoadCSV import read_two_columns

    s, _ = make_spectrum(a_ppm=100)
    body = "\n".join(f"{x},{y}" for x, y in zip(s.X, s.Y, strict=True))
    (tmp_path / "with.csv").write_text("Wavenumber (cm-1) ,Absorbance\n" + body)
    (tmp_path / "without.csv").write_text(body)
    xa, ya = read_two_columns(tmp_path / "with.csv")
    xb, yb = read_two_columns(tmp_path / "without.csv")
    assert len(xa) == len(xb) == len(s.X)
    np.testing.assert_array_equal(ya, yb)
