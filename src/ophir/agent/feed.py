"""Read ingested per-ticker data back as model-ready frames and tensors.

These helpers load the parquet written by :mod:`ophir.agent.ingest` and
bridge it into the ophir model's input contract by reusing
:func:`ophir.ticker.extract_features` and
:func:`ophir.ticker.extract_model_data`. Heavy imports (``ophir.ticker`` pulls
in ``torch``; ``ophir.register`` resolves the on-disk layout) are deferred so
importing this module stays cheap and GPU-free.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

_OHLCV_COLS = ["high", "low", "close", "volume"]


def resolve_stocks_root(override: str | None = None) -> Path:
    """Return the directory holding ``symbol=<SYMBOL>/data.parquet`` partitions.

    Parameters
    ----------
    override : str, optional
        Explicit directory. When ``None``, defaults to ophir's
        ``<DATA_DIR>/days/stocks`` (resolved lazily).

    Returns
    -------
    pathlib.Path
        The per-symbol parquet root.
    """
    if override is not None:
        return Path(override)
    from ophir.register import get_default_data_days_dir

    return Path(get_default_data_days_dir()) / "stocks"


def parquet_path(symbol: str, *, override: str | None = None) -> Path:
    """Return the parquet path for ``symbol`` under the stocks root."""
    return resolve_stocks_root(override) / f"symbol={symbol.upper()}" / "data.parquet"


def load_daily_ohlcv(symbol: str, *, stocks_dir: str | None = None) -> pd.DataFrame:
    """Load an ingested ticker as a datetime-indexed daily OHLCV frame.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    stocks_dir : str, optional
        Override for the parquet root (see :func:`resolve_stocks_root`).

    Returns
    -------
    pandas.DataFrame
        ``high`` / ``low`` / ``close`` / ``volume`` indexed by ``utc_time`` and
        sorted ascending -- the shape :func:`ophir.ticker.extract_features`
        expects.
    """
    path = parquet_path(symbol, override=stocks_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No ingested data for {symbol!r} at {path}. Run `ophir ingest {symbol}` first."
        )
    df = pd.read_parquet(path)
    df["utc_time"] = pd.to_datetime(df["utc_time"])
    df = df.set_index("utc_time").sort_index()
    return df[_OHLCV_COLS]


def latest_window_tensors(
    symbol: str,
    seq_len: int = 365,
    response_size: int = 90,
    *,
    stocks_dir: str | None = None,
) -> dict[str, Any]:
    """Build the model-input tensors for ``symbol``'s most recent window.

    Reuses :func:`ophir.ticker.extract_features` and
    :func:`ophir.ticker.extract_model_data`, so the result matches
    :class:`ophir.model_data.OHLCMulitClassPredictorInput`.

    Parameters
    ----------
    symbol : str
        Ticker symbol (must have been ingested).
    seq_len : int, optional
        Window length in calendar days. Defaults to ``365`` (the model's
        production window).
    response_size : int, optional
        Number of trailing days the model predicts. Defaults to ``90``.
    stocks_dir : str, optional
        Override for the parquet root.

    Returns
    -------
    dict
        ``feature_input`` ``(S, 13)``, ``targets`` ``(S, 3)``,
        ``trade_occured`` ``(S,)`` and ``response_size``.
    """
    from ophir.ticker import extract_features, extract_model_data

    df = load_daily_ohlcv(symbol, stocks_dir=stocks_dir)
    features = extract_features(df)
    window = features.iloc[-seq_len:]
    return extract_model_data(window, response_size)
