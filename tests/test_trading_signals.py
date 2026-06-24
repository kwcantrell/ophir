import pytest

from ophir.trading.forecast import OphirForecast
from ophir.trading.signals import (
    CORE_WEIGHTS,
    TACTICAL_WEIGHTS,
    blend_signals,
    cross_sectional_normalize,
    normalize,
    ophir_signals,
)
from ophir.trading.types import SignalWeights


def test_normalize_clamps() -> None:
    assert normalize(5.0, 0.0, 10.0) == pytest.approx(0.0)
    assert normalize(10.0, 0.0, 10.0) == pytest.approx(1.0)
    assert normalize(0.0, 0.0, 10.0) == pytest.approx(-1.0)
    assert normalize(-5.0, 0.0, 10.0) == pytest.approx(-1.0)
    assert normalize(15.0, 0.0, 10.0) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="lo"):
        normalize(1.0, 1.0, 1.0)


def test_blend_all_present() -> None:
    w = SignalWeights(ophir=0.5, momentum=0.3, sentiment=0.2)
    assert blend_signals(1.0, 1.0, 1.0, w) == pytest.approx(1.0)
    assert blend_signals(0.0, 0.0, 0.0, w) == pytest.approx(0.0)


def test_blend_ophir_absent_renormalizes() -> None:
    w = SignalWeights(ophir=0.6, momentum=0.25, sentiment=0.15)
    # momentum=1, sentiment=-1, weights renormalize over 0.25/0.15
    expected = (0.25 * 1.0 + 0.15 * -1.0) / (0.25 + 0.15)
    assert blend_signals(None, 1.0, -1.0, w) == pytest.approx(expected)


def test_blend_zero_weights_raises() -> None:
    w = SignalWeights(ophir=0.0, momentum=0.0, sentiment=0.0)
    with pytest.raises(ValueError, match="zero"):
        blend_signals(None, 1.0, 1.0, w)


def test_preset_weights_exist() -> None:
    assert CORE_WEIGHTS.ophir == pytest.approx(0.6)
    assert TACTICAL_WEIGHTS.momentum == pytest.approx(0.5)


def _fc(symbol: str, r_close: float) -> OphirForecast:
    return OphirForecast(symbol=symbol, r_close=r_close, upside=0.0, downside=0.0)


def test_ophir_signals_empty_returns_empty() -> None:
    assert ophir_signals({}) == {}


def test_ophir_signals_single_symbol_is_neutral() -> None:
    # One candidate has no cross-sectional dispersion -> no signal.
    assert ophir_signals({"AAPL": _fc("AAPL", 0.05)}) == {"AAPL": 0.0}


def test_ophir_signals_all_identical_is_neutral() -> None:
    out = ophir_signals({"A": _fc("A", 0.01), "B": _fc("B", 0.01)})
    assert out == {"A": 0.0, "B": 0.0}


def test_ophir_signals_cross_sectional_sign() -> None:
    out = ophir_signals({"HI": _fc("HI", 0.05), "MID": _fc("MID", 0.0), "LO": _fc("LO", -0.05)})
    assert out["HI"] > 0.0
    assert out["LO"] < 0.0
    assert out["MID"] == pytest.approx(0.0)
    assert out["HI"] == pytest.approx(-out["LO"])  # symmetric around the mean


def test_ophir_signals_clamps_to_unit_interval() -> None:
    # An outlier saturates at +1; every score stays within [-1, 1].
    out = ophir_signals({f"S{i}": _fc(f"S{i}", v) for i, v in enumerate([0.0, 0.0, 0.0, 1.0])})
    assert all(-1.0 <= s <= 1.0 for s in out.values())
    assert out["S3"] == pytest.approx(1.0)


def test_cross_sectional_normalize_empty() -> None:
    assert cross_sectional_normalize({}) == {}


def test_cross_sectional_normalize_single_is_neutral() -> None:
    assert cross_sectional_normalize({"A": 0.05}) == {"A": 0.0}


def test_cross_sectional_normalize_all_identical_is_neutral() -> None:
    assert cross_sectional_normalize({"A": 0.01, "B": 0.01}) == {"A": 0.0, "B": 0.0}


def test_cross_sectional_normalize_sign_and_clamp() -> None:
    out = cross_sectional_normalize({"HI": 0.05, "MID": 0.0, "LO": -0.05})
    assert out["HI"] > 0.0
    assert out["LO"] < 0.0
    assert out["MID"] == pytest.approx(0.0)
    assert out["HI"] == pytest.approx(-out["LO"])
    assert all(-1.0 <= v <= 1.0 for v in out.values())
