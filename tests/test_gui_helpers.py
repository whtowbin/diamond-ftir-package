"""GUI logic that does not need a display: text <-> parameter conversion."""

import pytest

tk = pytest.importorskip("tkinter")

from diamond_ftir_package.gui import _format_value, _parse_value


def test_scalar_types_follow_the_default():
    assert _parse_value(950, "1000") == 1000 and isinstance(
        _parse_value(950, "1000"), int
    )
    assert _parse_value(0.01, "0.05") == 0.05
    assert _parse_value("Whittaker", " ALS ") == "ALS"


def test_int_field_accepts_decimal_input():
    assert _parse_value(950, "950.5") == 950.5


def test_range_roundtrip():
    assert _format_value((3103, 3110)) == "3103, 3110"
    assert _parse_value((3103, 3110), "3100; 3115") == (3100, 3115)


def test_bad_number_raises_value_error():
    with pytest.raises(ValueError):
        _parse_value(0.5, "abc")


def test_bool_settings_parse_before_int():
    assert _parse_value(True, "False") is False
    assert _parse_value(False, "yes") is True
    with pytest.raises(ValueError):
        _parse_value(True, "maybe")
