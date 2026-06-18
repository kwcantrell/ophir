import math

import pytest

from ophir.trading.metrics import (
    calibration_error,
    daily_returns,
    hit_rate,
    max_drawdown,
    sharpe,
    total_return,
)


def test_total_return() -> None:
    assert total_return([100.0, 110.0]) == pytest.approx(0.10)
    assert total_return([100.0]) == 0.0
    assert total_return([]) == 0.0


def test_daily_returns() -> None:
    assert daily_returns([100.0, 110.0, 99.0]) == pytest.approx([0.10, -0.10])


def test_sharpe_zero_variance_is_zero() -> None:
    assert sharpe([0.01, 0.01, 0.01]) == 0.0


def test_sharpe_positive() -> None:
    s = sharpe([0.01, -0.005, 0.02, 0.0], periods_per_year=252)
    assert s > 0.0
    assert math.isfinite(s)


def test_max_drawdown() -> None:
    assert max_drawdown([100.0, 120.0, 90.0, 110.0]) == pytest.approx(-0.25)
    assert max_drawdown([]) == 0.0


def test_hit_rate() -> None:
    assert hit_rate([True, False, True, True]) == pytest.approx(0.75)
    assert hit_rate([]) == 0.0


def test_calibration_error() -> None:
    assert calibration_error([0.1, 0.2], [0.0, 0.4]) == pytest.approx(0.15)
    with pytest.raises(ValueError, match="length"):
        calibration_error([0.1], [0.1, 0.2])
