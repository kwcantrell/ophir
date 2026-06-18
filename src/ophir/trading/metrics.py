"""Pure performance metrics over equity curves and decision outcomes."""

import math
from collections.abc import Sequence


def total_return(equity_curve: Sequence[float]) -> float:
    """Fractional return from first to last equity point."""
    if len(equity_curve) < 2 or equity_curve[0] == 0:
        return 0.0
    return equity_curve[-1] / equity_curve[0] - 1.0


def daily_returns(equity_curve: Sequence[float]) -> list[float]:
    """Period-over-period simple returns."""
    out: list[float] = []
    for prev, cur in zip(equity_curve[:-1], equity_curve[1:], strict=False):  # noqa: RUF007
        out.append(0.0 if prev == 0 else cur / prev - 1.0)
    return out


def sharpe(returns: Sequence[float], periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio (zero risk-free rate); 0.0 if undefined."""
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    return mean / std * math.sqrt(periods_per_year)


def max_drawdown(equity_curve: Sequence[float]) -> float:
    """Most-negative peak-to-trough drawdown as a non-positive fraction."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    worst = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - 1.0)
    return worst


def hit_rate(outcomes: Sequence[bool]) -> float:
    """Fraction of ``True`` outcomes; 0.0 if empty."""
    if not outcomes:
        return 0.0
    return sum(1 for o in outcomes if o) / len(outcomes)


def calibration_error(predicted: Sequence[float], realized: Sequence[float]) -> float:
    """Mean absolute error between predicted and realized values."""
    if len(predicted) != len(realized):
        raise ValueError("predicted and realized must have equal length")
    if not predicted:
        return 0.0
    return sum(abs(p - r) for p, r in zip(predicted, realized, strict=True)) / len(predicted)
