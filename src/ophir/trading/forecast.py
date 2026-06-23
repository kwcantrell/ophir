"""Adapter seam for ophir model forecasts.

This module defines the contract the trading loop uses to obtain per-symbol
forecasts. CUDA inference (loading a checkpoint and running the model) is wired
behind :func:`load_forecasts` and guarded by availability checks; the function
degrades to ``{}`` when forecasts are unavailable and logs the reason, so the
trading loop falls back to non-ophir signals without raising.
"""

import logging
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
    """Return per-symbol offset-1 ophir forecasts, or ``{}`` if unavailable.

    Builds the most-recent inference window per symbol, loads the IC-best base
    checkpoint, and runs the model's day-1 (``response_size=1``) forward,
    returning the raw log-space ``r_close`` / ``upside`` / ``downside`` channels.

    Returns ``{}`` — and logs a ``WARNING`` — when forecasts are *unavailable*:
    no ``model_dir`` or checkpoint is present, CUDA is unavailable (the
    flex-attention forward is CUDA-only), or the data tree is absent.
    Availability failures are caught and logged rather than swallowed silently;
    inference-time errors (shape mismatches, RuntimeErrors from the forward) are
    *not* caught and will propagate.

    Parameters
    ----------
    symbols : sequence of str
        Ticker symbols to forecast.
    model_dir : str or pathlib.Path or None
        Directory holding the base checkpoint. ``None`` (or a directory with no
        ``*.ckpt``) yields ``{}``.

    Returns
    -------
    dict[str, OphirForecast]
        One forecast per symbol that produced a prediction; empty when forecasts
        are unavailable.
    """
    if model_dir is None or not _has_checkpoint(model_dir):
        return {}

    import torch

    if not torch.cuda.is_available():
        return {}

    from ophir import register
    from ophir.ticker import build_latest_inputs

    _log = logging.getLogger(__name__)
    try:
        inputs = build_latest_inputs(symbols)
    except (FileNotFoundError, OSError) as exc:
        _log.warning("load_forecasts: %s", exc)
        return {}
    if not inputs:
        return {}

    try:
        model = register.load_base_model_ckpt(strict=False, time_version=False)
    except (IndexError, FileNotFoundError, OSError) as exc:
        _log.warning("load_forecasts: %s", exc)
        return {}

    model = model.cuda().eval()
    out: dict[str, OphirForecast] = {}
    with torch.no_grad():
        for symbol, payload in inputs.items():
            batch = {key: value.unsqueeze(0) for key, value in payload.items()}
            prediction = model(batch)
            out[symbol] = OphirForecast(
                symbol=symbol,
                r_close=float(prediction.predicted_r_close[0, 0]),
                upside=float(prediction.predicted_upside[0, 0]),
                downside=float(prediction.predicted_downside[0, 0]),
            )
    return out
