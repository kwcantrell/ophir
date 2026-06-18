"""Adapter seam for ophir model forecasts.

This module defines the contract the trading loop uses to obtain per-symbol
forecasts. Actual CUDA inference (loading a checkpoint and running the model) is
a future enhancement implemented behind :func:`load_forecasts`; until then the
function reports availability only and never raises, so the loop degrades to the
non-ophir signals when no checkpoint is present.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OphirForecast:
    """One symbol's three forward targets from the ophir model."""

    symbol: str
    r_close: float
    upside: float
    downside: float


def _has_checkpoint(model_dir: str | Path) -> bool:
    path = Path(model_dir)
    return path.is_dir() and any(path.glob("*.ckpt"))


def load_forecasts(
    symbols: Sequence[str], model_dir: str | Path | None
) -> dict[str, OphirForecast]:
    """Return per-symbol ophir forecasts, or ``{}`` if the model is unavailable."""
    if model_dir is None or not _has_checkpoint(model_dir):
        return {}
    # Inference not yet wired; report availability without fabricating forecasts.
    return {}
