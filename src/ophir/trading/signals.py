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


def cross_sectional_normalize(values: Mapping[str, float]) -> dict[str, float]:
    """Cross-sectionally map per-key values into ``[-1, 1]``.

    Demeans by the cross-sectional mean, divides by the population standard
    deviation, and clamps to ``[-1, 1]``.

    Parameters
    ----------
    values : mapping of str to float
        One raw score per key (e.g. per symbol) for the day's candidate set.

    Returns
    -------
    dict[str, float]
        Per-key score in ``[-1, 1]``. Empty input yields ``{}``. When the
        cross-sectional dispersion is zero (a single key, or all-identical
        values), every score is ``0.0`` — no dispersion, no signal.
    """
    if not values:
        return {}
    xs = list(values.values())
    mean = fmean(xs)
    std = pstdev(xs)
    if std == 0.0:
        return dict.fromkeys(values, 0.0)
    return {key: max(-1.0, min(1.0, (value - mean) / std)) for key, value in values.items()}


def ophir_signals(forecasts: Mapping[str, OphirForecast]) -> dict[str, float]:
    """Cross-sectionally score per-symbol forecasts into ``[-1, 1]``.

    Ranks the day's candidates on ``r_close`` via
    :func:`cross_sectional_normalize`. The model's measured skill is
    cross-sectional (rank-IC), so the score is relative to the other candidates
    rather than an absolute return.

    Parameters
    ----------
    forecasts : mapping of str to OphirForecast
        Per-symbol forecasts for the day's candidate set.

    Returns
    -------
    dict[str, float]
        Per-symbol score in ``[-1, 1]``; ``{}`` for empty input, all-``0.0`` when
        the cross-sectional dispersion is zero.
    """
    return cross_sectional_normalize({symbol: f.r_close for symbol, f in forecasts.items()})
