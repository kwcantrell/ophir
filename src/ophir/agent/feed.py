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

from ophir.agent.market_calendar import last_closed_session

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


def load_history(
    symbol: str,
    *,
    as_of: Any = None,
    stocks_dir: str | None = None,
) -> pd.DataFrame:
    """Load a ticker's OHLCV trimmed to the last completed session.

    Keeps only bars on or before ``as_of`` -- the last closed NYSE session by
    default -- so a mid-session run drops today's still-forming bar and conditions
    on the prior completed day (T-1 close). Refuses a stale feed (latest bar older
    than the expected session) so the live forecast never runs on out-of-date data;
    callers (e.g. :func:`ophir.agent.predict.predict_many`) treat that as a per-ticker
    skip.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    as_of : timestamp-like, optional
        Cutoff session. Defaults to :func:`~ophir.agent.market_calendar.last_closed_session`.
    stocks_dir : str, optional
        Override for the parquet root.

    Returns
    -------
    pandas.DataFrame
        OHLCV through ``as_of`` (inclusive), datetime-indexed and sorted ascending.

    Raises
    ------
    ValueError
        If no bar exists on/before the cutoff, or the latest bar is stale.
    """
    df = load_daily_ohlcv(symbol, stocks_dir=stocks_dir)
    cutoff = last_closed_session() if as_of is None else pd.Timestamp(as_of).normalize()
    df = df.loc[df.index <= cutoff]
    if df.empty:
        raise ValueError(f"No bars on/before {cutoff.date()} for {symbol!r}")
    latest = df.index.max()
    if latest.normalize() < cutoff:
        raise ValueError(
            f"{symbol!r} feed is stale: latest bar {latest.date()} < last session {cutoff.date()}"
        )
    return df


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


def forecast_window_tensors(
    symbol: str,
    seq_len: int = 365,
    response_size: int = 90,
    *,
    as_of: Any = None,
    stocks_dir: str | None = None,
) -> dict[str, Any]:
    """Build a *forward-looking* model input for ``symbol``.

    Takes the trailing ``seq_len - response_size`` real feature rows and appends
    ``response_size`` zeroed placeholder rows, so the model's predicted trailing
    block is a genuine forecast of the next ``response_size`` days rather than a
    reconstruction of known days.

    The placeholder rows are marked ``trade_occured=True`` even though no trade
    has occurred yet: that flag drives the attention padding mask, and training
    always presents the response block as real (non-padding) days that attend to
    history. Marking them ``False`` here (as an earlier version did) padding-masks
    the response tokens out of attention entirely, so they cannot see the ticker's
    history and the model emits an identical constant for every ticker. The
    "no real data" aspect is already conveyed by zeroing the feature values.

    Parameters
    ----------
    symbol : str
        Ticker symbol (must have been ingested).
    seq_len : int, optional
        Total window length. Defaults to ``365``.
    response_size : int, optional
        Forecast horizon / number of zeroed future rows. Defaults to ``90``.
    as_of : timestamp-like, optional
        Condition the history on bars through this session (see :func:`load_history`).
        Defaults to the last closed NYSE session, dropping today's forming bar.
    stocks_dir : str, optional
        Override for the parquet root.

    Returns
    -------
    dict
        ``feature_input`` ``(S, 13)``, ``targets`` ``(S, 3)``,
        ``trade_occured`` ``(S,)`` and ``response_size``.
    """
    from ophir.ticker import extract_features, extract_model_data

    features = extract_features(load_history(symbol, as_of=as_of, stocks_dir=stocks_dir))
    history_size = seq_len - response_size
    history = features.iloc[-history_size:].reset_index(drop=True)
    future = pd.DataFrame(0.0, index=range(response_size), columns=history.columns)
    future["trade_occured"] = True  # participate in attention (see history); see docstring
    future = future.astype(history.dtypes)
    window = pd.concat([history, future], ignore_index=True)
    return extract_model_data(window, response_size)
