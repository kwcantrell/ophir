"""Normalize and blend the per-candidate signal components into one score.

Each component is expected in ``[-1, 1]``. The ophir forecast may be ``None``
(no CUDA/checkpoint, or the name is uncovered); the blend then degrades to the
remaining signals by renormalizing their weights.
"""

from collections.abc import Mapping
from statistics import fmean, pstdev

from ophir.trading.forecast import OphirForecast
from ophir.trading.types import SignalWeights

CORE_WEIGHTS = SignalWeights(ophir=0.6, momentum=0.25, sentiment=0.15)
TACTICAL_WEIGHTS = SignalWeights(ophir=0.2, momentum=0.5, sentiment=0.3)


def normalize(value: float, lo: float, hi: float) -> float:
    """Linearly map ``value`` over ``[lo, hi]`` into ``[-1, 1]``, clamped."""
    if lo >= hi:
        raise ValueError("lo must be < hi")
    frac = (value - lo) / (hi - lo)
    scaled = 2.0 * frac - 1.0
    return max(-1.0, min(1.0, scaled))


def blend_signals(
    ophir: float | None, momentum: float, sentiment: float, weights: SignalWeights
) -> float:
    """Weighted blend of the available signal components, result in ``[-1, 1]``."""
    pairs: list[tuple[float, float]] = [
        (weights.momentum, momentum),
        (weights.sentiment, sentiment),
    ]
    if ophir is not None:
        pairs.append((weights.ophir, ophir))
    total_weight = sum(w for w, _ in pairs)
    if total_weight == 0:
        raise ValueError("active signal weights sum to zero")
    return sum(w * v for w, v in pairs) / total_weight


def ophir_signals(forecasts: Mapping[str, OphirForecast]) -> dict[str, float]:
    """Cross-sectionally score per-symbol forecasts into ``[-1, 1]``.

    Ranks the day's candidates on ``r_close`` by demeaning, dividing by the
    cross-sectional (population) standard deviation, and clamping to
    ``[-1, 1]``. The model's measured skill is cross-sectional (rank-IC), so the
    score is relative to the other candidates rather than an absolute return.

    Parameters
    ----------
    forecasts : mapping of str to OphirForecast
        Per-symbol forecasts for the day's candidate set.

    Returns
    -------
    dict[str, float]
        Per-symbol score in ``[-1, 1]``. Empty input yields ``{}``. When the
        cross-sectional dispersion is zero (a single candidate, or an
        all-identical day), every score is ``0.0`` — no dispersion, no signal.
    """
    if not forecasts:
        return {}
    closes = [f.r_close for f in forecasts.values()]
    mean = fmean(closes)
    std = pstdev(closes)
    if std == 0.0:
        return dict.fromkeys(forecasts, 0.0)
    return {
        symbol: max(-1.0, min(1.0, (f.r_close - mean) / std)) for symbol, f in forecasts.items()
    }
