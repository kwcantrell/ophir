import pytest

from ophir.trading.signals import (
    CORE_WEIGHTS,
    TACTICAL_WEIGHTS,
    blend_signals,
    normalize,
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
