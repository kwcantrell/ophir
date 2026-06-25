"""S&P 500 symbol fetching and stock-split history (network-backed)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd  # type: ignore[import-untyped]

if TYPE_CHECKING:
    import numpy as np


def get_sp_500_symbols() -> list[str]:
    """Fetch the current S&P 500 constituent symbols from Wikipedia.

    Returns
    -------
    list[str]
        Ticker symbols scraped from the Wikipedia constituents table.

    Notes
    -----
    Performs a live HTTP request and is invoked at import time by
    ``ophir.ui``.
    """
    # Wikipedia URL for S&P 500 companies
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/58.0.3029.110 Safari/537.3"
        )
    }

    try:
        dfs = pd.read_html(url, storage_options={"User-Agent": headers["User-Agent"]})
    except Exception as e:
        print(f"An error occurred: {e}")

    return [str(s) for s in dfs[0]["Symbol"]]


def get_splits(tickers: list[str], cache_path: str | None = None) -> dict[str, StockSplit | None]:
    """Fetch (and cache) stock-split history for ``tickers`` from Yahoo Finance.

    Previously fetched results are read from a pickle cache; only missing
    tickers trigger network calls. Tickers with no splits are recorded with a
    sentinel so they are not re-queried.

    Parameters
    ----------
    tickers : list[str]
        Symbols to fetch split history for.
    cache_path : str, optional
        Pickle cache path. Defaults to ``<DATA_DIR>/yf_splits_cache.pkl``.

    Returns
    -------
    dict[str, StockSplit]
        Mapping of symbol to :class:`StockSplit` for tickers that have splits.
    """
    import pickle

    import yfinance as yf
    from tqdm import tqdm

    if cache_path is None:
        from ophir.register import DATA_DIR

        cache_path = os.path.join(DATA_DIR, "yf_splits_cache.pkl")

    cached: dict[str, StockSplit | None]
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        missing = [t for t in tickers if t not in cached]
        if not missing:
            # Pinned: the cache-hit early return preserves None sentinels;
            # the late return filters them. See get_splits tests.
            return {t: cached[t] for t in tickers if t in cached}
    else:
        cached = {}
        missing = list(tickers)

    for ticker in tqdm(missing, desc="Fetching splits"):
        try:
            series = yf.Ticker(ticker).splits
        except Exception as e:
            print(f"[get_splits] {ticker} failed: {e}")
            continue
        if series is None or len(series) == 0:
            cached[ticker] = None  # sentinel: queried, no splits
            continue
        naive_index = (
            series.index.tz_localize(None) if series.index.tz is not None else series.index
        )
        cached[ticker] = StockSplit(
            id=ticker,
            dates=list(naive_index.to_numpy()),
            ratios=list(series.to_numpy().astype(float)),
        )

    with open(cache_path, "wb") as f:
        pickle.dump(cached, f)

    return {t: cached[t] for t in tickers if cached.get(t) is not None}


@dataclass
class StockSplit:
    """Split history for a single stock.

    Attributes
    ----------
    id : str
        The ticker symbol.
    dates : list[numpy.datetime64]
        Effective split dates.
    ratios : list[float]
        Split ratios aligned with ``dates``.
    """

    id: str
    dates: list[np.datetime64]
    ratios: list[float]

    def apply_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """Back-adjust prices (and inversely volume) for the recorded splits.

        Parameters
        ----------
        df : pandas.DataFrame
            Datetime-indexed OHLCV frame.

        Returns
        -------
        pandas.DataFrame
            ``df`` with ``close`` divided, and ``volume`` multiplied, by the
            cumulative pre-split adjustment factor.

        Notes
        -----
        Back-adjustment uses the full known split history, so a past row is
        scaled using splits dated after it. This is standard for back-adjusted
        series and the model trains on (largely split-invariant) log returns,
        so the effect is negligible; it is intentionally *not* point-in-time.
        """
        df = df.sort_index()

        # Create cumulative adjustment factor
        adj_factor = pd.Series(1.0, index=df.index)

        for date, ratio in zip(self.dates, self.ratios, strict=False):
            split_date = pd.to_datetime(date)

            # Apply to all dates BEFORE split date
            adj_factor.loc[adj_factor.index < split_date] /= ratio

        # Apply price adjustments
        price_cols = ["close"]
        df[price_cols] = df[price_cols].mul(adj_factor, axis=0)

        # Volume goes the opposite way
        if "volume" in df.columns:
            df["volume"] = df["volume"] / adj_factor

        return df
