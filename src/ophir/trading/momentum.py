"""Deterministic momentum signal producer for the trading core.

Mirrors the forecast seam (:mod:`ophir.trading.forecast`): a CPU/offline data
loader plus pure metrics. Momentum is a mechanical function of recent price
bars, so it lives as a reproducible primitive rather than an LLM judgment.
"""

from collections.abc import Mapping, Sequence
from math import log
from statistics import fmean, stdev

from ophir.trading.signals import cross_sectional_normalize


def momentum_score(closes: Sequence[float], lookback: int = 63, skip: int = 5) -> float | None:
    """Information ratio of daily log returns over a skip-adjusted window.

    Over the window of ``lookback`` daily returns ending ``skip`` bars before the
    latest close, returns ``mean / std`` of the daily log returns (sample std,
    ``ddof=1``). The ``skip`` excludes the most-recent ``skip`` bars, whose
    returns are reversal-prone; including them would load the signal on reversal.

    Parameters
    ----------
    closes : sequence of float
        Positive daily closing prices, oldest first.
    lookback : int, optional
        Number of daily returns in the window. Defaults to ``63``.
    skip : int, optional
        Number of most-recent bars to exclude. Defaults to ``5``.

    Returns
    -------
    float or None
        The information ratio, or ``None`` when there is too little history
        (``len(closes) < lookback + skip + 1``) or the window's return variance
        is zero.
    """
    if lookback < 2:
        return None
    if len(closes) < lookback + skip + 1:
        return None
    end = len(closes) - skip
    start = end - lookback - 1
    window = closes[start:end]
    rets = [log(window[i] / window[i - 1]) for i in range(1, len(window))]
    spread = stdev(rets)
    if spread == 0.0:
        return None
    return fmean(rets) / spread


def momentum_signals(
    closes_by_symbol: Mapping[str, Sequence[float]],
    lookback: int = 63,
    skip: int = 5,
) -> dict[str, float]:
    """Cross-sectionally score per-symbol momentum into ``[-1, 1]``.

    Computes :func:`momentum_score` per symbol, drops symbols whose score is
    ``None`` (insufficient history or zero variance), and normalizes the
    survivors with :func:`ophir.trading.signals.cross_sectional_normalize`.

    Parameters
    ----------
    closes_by_symbol : mapping of str to sequence of float
        Per-symbol daily closing prices, oldest first.
    lookback : int, optional
        Window length in daily returns. Defaults to ``63``.
    skip : int, optional
        Most-recent bars to exclude. Defaults to ``5``.

    Returns
    -------
    dict[str, float]
        Per-symbol momentum score in ``[-1, 1]``; symbols without a defined
        score are omitted. Empty input or all-``None`` yields ``{}``.
    """
    raw: dict[str, float] = {}
    for symbol, closes in closes_by_symbol.items():
        score = momentum_score(closes, lookback=lookback, skip=skip)
        if score is not None:
            raw[symbol] = score
    return cross_sectional_normalize(raw)
