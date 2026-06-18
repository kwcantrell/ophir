"""High-quality dataset curation.

Scans the per-symbol parquet tree, scores each symbol on four quality
dimensions — liquidity, history length & continuity, price sanity, and
staleness/flatlines — and persists a curated *allowlist* plus a per-symbol
stats JSON. Training consumes the allowlist through
:meth:`ophir.ticker.StockHanlder.keep_stocks` (see ``ophir train
--use-quality-allowlist``).

Row-level cleaning (:func:`ophir.ticker.clean_daily_ohlcv`) is applied here so
the per-symbol metrics reflect exactly the data training sees when run with
``--clean-rows``.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Annotated, Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import typer

from ophir.ticker import clean_daily_ohlcv, get_stock_parquets


@dataclass(frozen=True)
class QualityThresholds:
    """Thresholds defining a high-quality symbol.

    Attributes
    ----------
    min_median_dollar_volume : float
        Liquidity floor on the median of ``close * volume``.
    min_trading_days : int
        Minimum number of (cleaned) trading days.
    max_missing_day_fraction : float
        Maximum fraction of *business* days with no trade over the symbol's
        span (a continuity gate; weekends are excluded from the denominator).
    min_median_close : float
        Penny-stock floor on the median close.
    max_return_spikes : int
        Maximum number of pre-clean ``|r_close| > max_abs_r_close`` days.
    max_abs_r_close : float
        Single-day log-return magnitude treated as a split error / glitch.
    max_flat_run : int
        Maximum allowed run of identical consecutive closes (staleness).
    max_zero_volume_fraction : float
        Maximum fraction of zero/negative-volume days (staleness).
    """

    min_median_dollar_volume: float = 1_000_000.0
    min_trading_days: int = 252
    max_missing_day_fraction: float = 0.10
    min_median_close: float = 5.0
    max_return_spikes: int = 0
    max_abs_r_close: float = 0.75
    max_flat_run: int = 10
    max_zero_volume_fraction: float = 0.05


@dataclass(frozen=True)
class SymbolQuality:
    """Per-symbol quality metrics and the pass/fail verdict.

    All metrics are computed on the daily-aggregated frame; the liquidity,
    history, price-sanity, and flat-run metrics use the *cleaned* frame, while
    ``n_return_spikes`` and ``zero_volume_fraction`` are measured on the
    pre-clean frame (cleaning removes the very rows they count).
    """

    symbol: str
    median_dollar_volume: float
    n_trading_days: int
    calendar_span_days: int
    missing_day_fraction: float
    median_close: float
    n_return_spikes: int
    max_flat_run: int
    zero_volume_fraction: float
    passed: bool
    fail_reasons: tuple[str, ...]


def _daily_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Daily-aggregate raw ticks, mirroring ``StockHanlder.stock_df``.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw frame with a ``utc_time`` column and OHLCV columns.

    Returns
    -------
    pandas.DataFrame
        A date-indexed daily OHLCV frame.
    """
    df = df.copy()
    df["date"] = df["utc_time"].dt.normalize()
    daily = (
        df.groupby("date")
        .agg({"high": "max", "low": "min", "close": "last", "volume": "sum"})
        .sort_index()
    )
    return daily.dropna(subset=["high", "low", "close"])


def _max_flat_run(closes: pd.Series) -> int:
    """Return the longest run of identical consecutive closes."""
    if len(closes) == 0:
        return 0
    groups = (closes != closes.shift()).cumsum()
    return int(closes.groupby(groups).size().max())


def _missing_day_fraction(index: pd.DatetimeIndex, n_trading_days: int) -> float:
    """Fraction of business days in the span with no trade.

    Uses a business-day denominator (``numpy.busday_count``) so weekends do not
    register as a structural ~30% gap; market holidays remain counted as
    business days, which keeps the metric a slight over-estimate rather than an
    artifact.
    """
    if n_trading_days < 2:
        return 0.0
    start = index.min().date()
    end = index.max().date()
    business_days = int(np.busday_count(start, end)) + 1
    if business_days <= 0:
        return 0.0
    return float(max(0.0, 1.0 - n_trading_days / business_days))


def compute_symbol_quality(
    raw_df: pd.DataFrame,
    *,
    thresholds: QualityThresholds,
) -> SymbolQuality | None:
    """Compute quality metrics for one symbol's raw tick frame.

    Parameters
    ----------
    raw_df : pandas.DataFrame
        Raw parquet frame for one symbol (``utc_time`` + OHLCV columns).
    thresholds : QualityThresholds
        The gates each metric is compared against.

    Returns
    -------
    SymbolQuality or None
        The metrics and verdict, or ``None`` if the symbol has no usable rows
        after aggregation.
    """
    daily = _daily_aggregate(raw_df)
    if daily.empty:
        return None

    r_close = np.log(daily["close"] / daily["close"].shift(1))
    n_return_spikes = int((r_close.abs() > thresholds.max_abs_r_close).sum())
    zero_volume_fraction = float((daily["volume"] <= 0).mean())

    cleaned = clean_daily_ohlcv(daily, max_abs_r_close=thresholds.max_abs_r_close)
    if cleaned.empty:
        return None

    median_dollar_volume = float((cleaned["close"] * cleaned["volume"]).median())
    n_trading_days = len(cleaned)
    calendar_span_days = int((cleaned.index.max() - cleaned.index.min()).days)
    missing_day_fraction = _missing_day_fraction(cleaned.index, n_trading_days)
    median_close = float(cleaned["close"].median())
    max_flat_run = _max_flat_run(cleaned["close"])

    fail_reasons: list[str] = []
    if median_dollar_volume < thresholds.min_median_dollar_volume:
        fail_reasons.append("liquidity")
    if n_trading_days < thresholds.min_trading_days:
        fail_reasons.append("history_length")
    if missing_day_fraction > thresholds.max_missing_day_fraction:
        fail_reasons.append("continuity")
    if median_close < thresholds.min_median_close:
        fail_reasons.append("penny_stock")
    if n_return_spikes > thresholds.max_return_spikes:
        fail_reasons.append("return_spikes")
    if max_flat_run > thresholds.max_flat_run:
        fail_reasons.append("flatline")
    if zero_volume_fraction > thresholds.max_zero_volume_fraction:
        fail_reasons.append("zero_volume")

    return SymbolQuality(
        symbol="",
        median_dollar_volume=median_dollar_volume,
        n_trading_days=n_trading_days,
        calendar_span_days=calendar_span_days,
        missing_day_fraction=missing_day_fraction,
        median_close=median_close,
        n_return_spikes=n_return_spikes,
        max_flat_run=max_flat_run,
        zero_volume_fraction=zero_volume_fraction,
        passed=not fail_reasons,
        fail_reasons=tuple(fail_reasons),
    )


def curate_symbols(
    base_path: str,
    *,
    thresholds: QualityThresholds,
    symbols: list[str] | None = None,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Scan the parquet tree and score every symbol.

    Parameters
    ----------
    base_path : str
        Directory of per-symbol parquet partitions (``ticker=<SYM>/*.parquet``).
    thresholds : QualityThresholds
        The quality gates.
    symbols : list[str], optional
        If given, only these symbols are scanned (intersected with the tree).

    Returns
    -------
    tuple[list[str], dict[str, dict]]
        The sorted list of passing symbols and a ``{symbol: metrics}`` stats
        mapping. Symbols that fail to load (or have no usable rows) are recorded
        with a ``"load_error"`` / ``"empty"`` entry and excluded from the
        allowlist rather than aborting the scan.
    """
    stock_dict = get_stock_parquets(base_path)
    if symbols is not None:
        wanted = set(symbols)
        stock_dict = {s: p for s, p in stock_dict.items() if s in wanted}

    passing: list[str] = []
    stats: dict[str, dict[str, Any]] = {}
    for symbol in sorted(stock_dict):
        try:
            raw_df = pd.read_parquet(stock_dict[symbol])
            quality = compute_symbol_quality(raw_df, thresholds=thresholds)
        except Exception as exc:  # record and continue the scan
            stats[symbol] = {"passed": False, "fail_reasons": ["load_error"], "error": str(exc)}
            continue
        if quality is None:
            stats[symbol] = {"passed": False, "fail_reasons": ["empty"]}
            continue
        record = asdict(quality)
        record["symbol"] = symbol
        stats[symbol] = record
        if quality.passed:
            passing.append(symbol)

    return sorted(passing), stats


app = typer.Typer()


@app.command()
def curate(
    data_dir: Annotated[str | None, typer.Option(help="Override the data directory")] = None,
    min_dollar_volume: Annotated[
        float, typer.Option(help="Liquidity floor on median close*volume")
    ] = 1_000_000.0,
    min_trading_days: Annotated[int, typer.Option(help="Minimum cleaned trading days")] = 252,
    max_missing_day_fraction: Annotated[
        float, typer.Option(help="Max fraction of business days with no trade")
    ] = 0.10,
    min_median_close: Annotated[
        float, typer.Option(help="Penny-stock floor on the median close")
    ] = 5.0,
    max_return_spikes: Annotated[int, typer.Option(help="Max pre-clean |r_close| spike days")] = 0,
    max_abs_r_close: Annotated[
        float, typer.Option(help="Single-day log return treated as a glitch")
    ] = 0.75,
    max_flat_run: Annotated[int, typer.Option(help="Max run of identical consecutive closes")] = 10,
    max_zero_volume_fraction: Annotated[
        float, typer.Option(help="Max fraction of zero-volume days")
    ] = 0.05,
    use_sp500: Annotated[
        bool, typer.Option(help="Restrict to S&P 500 symbols (network fetch)")
    ] = False,
) -> None:
    """Scan the parquet tree and write the quality allowlist + stats JSON.

    Backs the ``ophir curate`` command. Writes
    ``<DATA_DIR>/quality-symbols.txt`` (the allowlist consumed by
    ``ophir train --use-quality-allowlist``) and ``<DATA_DIR>/quality-stats.json``
    (every symbol's metrics and fail reasons).
    """
    from ophir import register

    base_path = os.path.join(data_dir or register.get_default_data_days_dir(), "stocks")
    thresholds = QualityThresholds(
        min_median_dollar_volume=min_dollar_volume,
        min_trading_days=min_trading_days,
        max_missing_day_fraction=max_missing_day_fraction,
        min_median_close=min_median_close,
        max_return_spikes=max_return_spikes,
        max_abs_r_close=max_abs_r_close,
        max_flat_run=max_flat_run,
        max_zero_volume_fraction=max_zero_volume_fraction,
    )

    symbols: list[str] | None = None
    if use_sp500:
        from ophir.ticker import get_sp_500_symbols

        symbols = get_sp_500_symbols()

    passing, stats = curate_symbols(base_path, thresholds=thresholds, symbols=symbols)

    register.set_quality_symbols(passing)
    with open(register.quality_stats_path(), "w") as f:
        json.dump(stats, f, sort_keys=True, indent=2)

    typer.echo(f"Curated {len(passing)}/{len(stats)} symbols passed quality gates.")
