import json

from diamond_ftir_package.params import AnalysisParams


def test_defaults_match_historical_hardcoded_values():
    p = AnalysisParams()
    assert (p.nitrogen.wn_low, p.nitrogen.wn_high) == (950, 1350)
    assert p.nitrogen.max_C_or_B == 0.01
    assert (p.nitrogen.a_ppm_per_cm, p.nitrogen.b_ppm_per_cm) == (16.5, 79.4)
    assert p.diamond.saturation_cutoff == 2.5 and p.diamond.stdev_cut_off == 0.5
    assert p.hydrogen.peak_3107 == (3103, 3110)
    assert p.platelet.search_window == (1355, 1380)
    assert len(p.amber.bands) == 9


def test_json_roundtrip_preserves_settings():
    p = AnalysisParams()
    p.nitrogen.wn_low = 1000
    p.run_amber = True
    restored = AnalysisParams.from_dict(json.loads(json.dumps(p.to_dict())))
    assert restored == p


def test_partial_dict_keeps_other_defaults():
    p = AnalysisParams.from_dict({"nitrogen": {"max_C_or_B": 0.05}})
    assert p.nitrogen.max_C_or_B == 0.05
    assert p.nitrogen.wn_low == 950


def test_every_setting_has_a_description_for_the_gui_tooltip():
    from dataclasses import fields

    p = AnalysisParams()
    for section in ("diamond", "nitrogen", "hydrogen", "platelet", "amber"):
        for f in fields(getattr(p, section)):
            assert f.metadata.get("description"), (
                f"{section}.{f.name} has no description"
            )
