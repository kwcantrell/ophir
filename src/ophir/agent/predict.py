"""Run the trained ophir model to forecast a ticker's next 90 days.

Loads the base checkpoint, feeds the latest 365-day window (via
:func:`ophir.agent.feed.latest_window_tensors`), and returns a structured
:class:`Forecast` with the predicted relative-close / upside / downside paths
plus a horizon cumulative return used for ranking. Mirrors the inference recipe
in ``ophir.ui``; requires a CUDA GPU and a trained checkpoint.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from ophir.agent import audit
from ophir.agent.feed import forecast_window_tensors, load_history, parquet_path
from ophir.agent.ingest import ingest

# flex_attention compiles its kernel via Triton, which has no working backend on
# native Windows; force the eager (uncompiled) attention path there so inference
# runs. Linux/WSL keep the fast compiled path. Must precede any torch import.
if sys.platform == "win32":
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

if TYPE_CHECKING:
    from ophir.training_models import LightningOHLCPredictor


@dataclass(frozen=True, slots=True)
class Forecast:
    """A model forecast for one ticker over the prediction horizon.

    Attributes
    ----------
    symbol : str
        Ticker symbol.
    asof : str
        ISO date of the last input bar the forecast is conditioned on.
    horizon : int
        Number of forecast days (channels in each path).
    r_close, upside, downside : list[float]
        Per-day predicted relative-close return / upside / downside log ratios.
    cum_return : float
        Cumulative predicted return over the horizon (``exp(sum(r_close)) - 1``).
    score : float
        Ranking score (defaults to ``cum_return``).
    """

    symbol: str
    asof: str
    horizon: int
    r_close: list[float]
    upside: list[float]
    downside: list[float]
    cum_return: float
    score: float


@lru_cache(maxsize=1)
def load_predictor() -> LightningOHLCPredictor:
    """Load the base checkpoint onto the GPU in eval mode (cached per process).

    Uses the validation-best checkpoint (``time_version=False``), which yields
    sane forecast magnitudes; the time-interval checkpoint is degenerate for
    forecasting.
    """
    from ophir.register import load_base_model_ckpt

    model = load_base_model_ckpt(time_version=False)
    return model.cuda().eval()


def predict_ticker(
    symbol: str,
    *,
    model: LightningOHLCPredictor | None = None,
    ensure_data: bool = True,
    stocks_dir: str | None = None,
    seq_len: int = 365,
    response_size: int = 90,
    as_of: Any = None,
) -> Forecast:
    """Forecast one ticker's next ``response_size`` days.

    Parameters
    ----------
    symbol : str
        Ticker symbol (case-insensitive).
    model : LightningOHLCPredictor, optional
        A loaded model; defaults to the cached :func:`load_predictor`.
    ensure_data : bool, optional
        If ``True`` and the ticker has no ingested parquet, ingest it first.
    stocks_dir : str, optional
        Override for the parquet root.
    seq_len, response_size : int, optional
        Input window length and forecast horizon (defaults ``365`` / ``90``).
    as_of : timestamp-like, optional
        Condition the forecast on bars through this session, dropping today's
        forming bar (see :func:`ophir.agent.feed.load_history`). Defaults to the
        last closed NYSE session.

    Returns
    -------
    Forecast
        The structured forecast for ``symbol``.
    """
    import torch

    symbol = symbol.upper().strip()
    if ensure_data and not parquet_path(symbol, override=stocks_dir).exists():
        ingest(symbol, stocks_dir=stocks_dir)
    if model is None:
        model = load_predictor()

    md = forecast_window_tensors(
        symbol, seq_len=seq_len, response_size=response_size, as_of=as_of, stocks_dir=stocks_dir
    )
    with torch.no_grad():
        out = model(md)

    r_close = out.predicted_r_close.detach().cpu().reshape(-1).tolist()
    upside = out.predicted_upside.detach().cpu().reshape(-1).tolist()
    downside = out.predicted_downside.detach().cpu().reshape(-1).tolist()
    cum_return = math.expm1(float(sum(r_close)))
    asof = str(load_history(symbol, as_of=as_of, stocks_dir=stocks_dir).index.max().date())

    forecast = Forecast(
        symbol=symbol,
        asof=asof,
        horizon=len(r_close),
        r_close=r_close,
        upside=upside,
        downside=downside,
        cum_return=cum_return,
        score=cum_return,
    )
    audit.log_event(
        "forecast", symbol=symbol, asof=asof, horizon=forecast.horizon, cum_return=cum_return
    )
    return forecast


def predict_many(
    symbols: list[str],
    *,
    model: LightningOHLCPredictor | None = None,
    ensure_data: bool = True,
    stocks_dir: str | None = None,
    response_size: int = 90,
    as_of: Any = None,
) -> list[Forecast]:
    """Forecast several tickers; failures (including a stale feed) are logged and skipped."""
    if model is None:
        model = load_predictor()
    forecasts: list[Forecast] = []
    for symbol in symbols:
        try:
            forecasts.append(
                predict_ticker(
                    symbol,
                    model=model,
                    ensure_data=ensure_data,
                    stocks_dir=stocks_dir,
                    response_size=response_size,
                    as_of=as_of,
                )
            )
        except (ValueError, FileNotFoundError, OSError) as exc:
            audit.log_event("forecast_failed", symbol=symbol, error=str(exc))
            print(f"[predict] {symbol}: FAILED -- {exc}")
    return forecasts


def rank(forecasts: list[Forecast], top_k: int = 5) -> list[Forecast]:
    """Return the ``top_k`` forecasts by descending score."""
    return sorted(forecasts, key=lambda f: f.score, reverse=True)[:top_k]
