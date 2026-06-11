"""Fetch daily OHLC from Yahoo Finance and persist it model-ready.

Pulls history for a ticker, normalizes it to the schema
:func:`ophir.ticker.extract_features` consumes, runs non-fatal quality checks,
and writes it to the Hive layout ``symbol=<SYMBOL>/data.parquet`` so the
existing ``StockHanlder`` / ``StockStreamer`` pipeline reads it unchanged.

Yahoo's ``auto_adjust=True`` output is already split/dividend-adjusted, so the
separate split back-adjustment in :mod:`ophir.ticker` is intentionally skipped
on this path (applying it again would double-adjust).
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import pandas as pd  # type: ignore[import-untyped]

from ophir.agent.feed import parquet_path

if TYPE_CHECKING:
    from pathlib import Path

_OHLCV_COLS = ["high", "low", "close", "volume"]
_MIN_MODEL_DAYS = 455  # 365-day window + ~90 calendar days of rolling-feature warmup
_STALE_DAYS = 5


def _fetch_yahoo(symbol: str, days: int) -> pd.DataFrame:
    """Fetch ``days`` of split/dividend-adjusted daily bars from Yahoo Finance."""
    import yfinance as yf

    end = dt.date.today()
    start = end - dt.timedelta(days=days)
    raw = yf.Ticker(symbol).history(
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=True,
    )
    if raw.empty:
        raise ValueError(
            f"No Yahoo Finance data for {symbol!r} between {start} and {end} "
            "(invalid or delisted symbol?)."
        )
    return raw


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Coerce a yfinance frame to a tz-naive, deduplicated daily OHLCV frame."""
    df = raw.rename(columns=str.lower)[_OHLCV_COLS].copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "utc_time"
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.dropna(subset=["high", "low", "close"])


def _quality_warnings(df: pd.DataFrame, days: int) -> list[str]:
    """Return human-readable staleness / sparsity warnings (non-fatal)."""
    warnings: list[str] = []
    last = df.index.max()
    age = (pd.Timestamp.today().normalize() - last.normalize()).days
    if age > _STALE_DAYS:
        warnings.append(f"last bar {last.date()} is {age} days old (stale feed?)")
    if days < _MIN_MODEL_DAYS or len(df) < 365:
        warnings.append(
            f"only {len(df)} rows for {days} requested days -- the model's "
            f"365-day window may not fully populate (want ~{_MIN_MODEL_DAYS} days)"
        )
    return warnings


def _persist(df: pd.DataFrame, symbol: str, override: str | None) -> Path:
    """Write ``df`` to ``symbol=<SYMBOL>/data.parquet`` in the Hive layout."""
    out = df.reset_index()[["utc_time", *_OHLCV_COLS]]
    dest = parquet_path(symbol, override=override)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    return dest


def ingest(symbol: str, days: int = 730, *, stocks_dir: str | None = None) -> Path:
    """Ingest one ticker's daily OHLC into a model-ready parquet.

    Parameters
    ----------
    symbol : str
        Ticker symbol (case-insensitive).
    days : int, optional
        Calendar days of history to fetch. Defaults to ``730`` (~2 years) --
        enough for the model's 365-day window plus rolling-feature warmup.
    stocks_dir : str, optional
        Override for the parquet root. Defaults to ophir's
        ``<DATA_DIR>/days/stocks``.

    Returns
    -------
    pathlib.Path
        The written parquet path.
    """
    symbol = symbol.upper().strip()
    df = _normalize(_fetch_yahoo(symbol, days))
    if df.empty:
        raise ValueError(f"No usable rows for {symbol!r} after normalization.")
    for warning in _quality_warnings(df, days):
        print(f"[ingest] {symbol}: WARNING {warning}")
    dest = _persist(df, symbol, stocks_dir)
    print(
        f"[ingest] {symbol}: {len(df)} rows "
        f"{df.index.min().date()}..{df.index.max().date()} "
        f"(last close {df['close'].iloc[-1]:.2f}) -> {dest}"
    )
    return dest


def ingest_many(
    symbols: list[str], days: int = 730, *, stocks_dir: str | None = None
) -> dict[str, Path]:
    """Ingest several tickers; returns ``{symbol: parquet_path}``.

    A failed symbol is reported and skipped so one bad ticker does not abort the
    whole batch.
    """
    paths: dict[str, Path] = {}
    for symbol in symbols:
        try:
            paths[symbol.upper().strip()] = ingest(symbol, days, stocks_dir=stocks_dir)
        except (ValueError, OSError) as exc:
            print(f"[ingest] {symbol}: FAILED -- {exc}")
    return paths
