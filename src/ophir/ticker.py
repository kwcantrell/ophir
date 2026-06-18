"""Stock data ingestion, split adjustment, feature extraction, and datasets.

This module turns per-stock parquet files into the fixed-length feature
windows the model consumes. The pipeline is: discover parquet files
(:func:`get_stock_parquets`), load and filter them (:class:`StockHanlder`),
optionally back-adjust for splits (:class:`StockSplit`), compute the
13-feature representation (:func:`extract_features`), slice into windows
(:class:`StockStreamer`), and expose them as ``torch`` datasets
(:class:`StockStreamerDataset`, :class:`StockHandlerDataset`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


def get_stock_parquets(base_path: str) -> dict[str, str]:
    """Map each stock symbol to its parquet file under ``base_path``.

    Expects Hive-style partition directories named ``<key>=<symbol>``, each
    containing a single ``.parquet`` file.

    Parameters
    ----------
    base_path : str
        Directory containing the per-symbol partition directories.

    Returns
    -------
    dict[str, str]
        Mapping of symbol to absolute parquet path.
    """
    stock_dirs = os.listdir(base_path)

    def parquet(path: str) -> str | None:
        for p in os.listdir(os.path.join(base_path, path)):
            if p.endswith(".parquet"):
                return os.path.join(path, p)
        return None

    # Pinned latent behavior: partitions with no .parquet pass `None` to
    # `os.path.join`, which raises TypeError. See test
    # `test_get_stock_parquets_malformed_partition_raises_typeerror`.
    stocks = {
        path.split("=")[-1]: os.path.join(base_path, parquet(path))  # type: ignore[arg-type]
        for path in stock_dirs
        if "=" in path
    }
    return stocks


def get_starts(df: pd.DataFrame, seq_len: int, offset: int) -> np.ndarray[Any, Any]:
    """Compute window start indices spanning ``df``.

    Parameters
    ----------
    df : pandas.DataFrame
        The (preprocessed) frame to slice.
    seq_len : int
        Window length.
    offset : int
        Stride between consecutive window starts.

    Returns
    -------
    numpy.ndarray
        Integer start positions ``0, offset, 2·offset, …`` up to
        ``len(df) - seq_len``.
    """
    num_start = max(0, len(df) - seq_len)
    starts = np.arange(0, num_start, offset)
    return starts


def get_start_dates(df: pd.DataFrame, seq_len: int, offset: int) -> np.ndarray[Any, Any]:
    """Compute window start *dates* on a daily calendar over ``df``.

    Parameters
    ----------
    df : pandas.DataFrame
        A datetime-indexed frame.
    seq_len : int
        Window length in calendar days.
    offset : int
        Stride between consecutive window starts.

    Returns
    -------
    numpy.ndarray
        The calendar dates at which each window begins.
    """
    dates = df.index.to_series()
    calendar = pd.date_range(dates.min(), dates.max(), freq="D")
    starts = np.arange(0, len(calendar) - seq_len, offset)
    return np.asarray(calendar[starts].to_numpy())


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
        from .register import DATA_DIR

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


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 13-feature model representation from an OHLCV frame.

    Features: log time delta, log close return ``r_close``, and rolling
    normalized returns / normalized volume / volatility over 10-, 20-, and
    60-day windows, plus the ``upside`` and ``downside`` log ratios. The frame
    is reindexed onto a daily calendar; padded days are zero-filled and flagged
    by ``trade_occured``. All features are point-in-time (trailing rolling
    windows only), so this introduces no lookahead.

    Parameters
    ----------
    df : pandas.DataFrame
        Datetime-indexed frame with ``high`` / ``low`` / ``close`` /
        ``volume`` columns.

    Returns
    -------
    pandas.DataFrame
        The calendar-padded feature columns (empty if ``df`` is empty).
    """
    feature_cols: list[str] = []

    def add_feature(feature_col: str, feature_val: Any, df: pd.DataFrame) -> pd.DataFrame:
        df = df.assign(**{feature_col: feature_val})
        feature_cols.append(feature_col)
        return df

    # time delta
    delta_days = df.index.to_series().diff().dt.days
    df = add_feature("time_delta", np.log(delta_days), df)

    # returns
    prev_close = df["close"].shift(1)
    df = add_feature("r_close", np.log(df["close"] / prev_close), df)

    def add_rolling_features(window_size: int, df: pd.DataFrame) -> pd.DataFrame:
        # normalized returns
        eps = 1e-8
        r_close = df["r_close"]
        r_close_rolling_std = r_close.rolling(window_size).std()
        df = add_feature(f"{window_size}_norm_returns", r_close / (r_close_rolling_std + eps), df)

        # volatility normalization
        log_volume: pd.DataFrame = np.log(df["volume"] + eps)
        mu = log_volume.rolling(window_size).mean()
        sigma = log_volume.rolling(window_size).std()
        norm_volume = (log_volume - mu) / (sigma + eps).clip(-5, 5)
        df = add_feature(f"{window_size}_norm_volume", norm_volume, df)
        df = add_feature(f"{window_size}_volatility", r_close_rolling_std, df)
        return df

    df = add_rolling_features(10, df)
    df = add_rolling_features(20, df)
    df = add_rolling_features(60, df)
    df = add_feature("upside", np.log(df["high"] / df["close"]), df)
    df = add_feature("downside", np.log(df["close"] / df["low"]), df)

    if df.empty:
        return df

    calendar = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df_pad = df.reindex(index=calendar)

    df_pad = add_feature("trade_occured", df_pad["close"].notna(), df_pad)

    pad_value = 0.0
    for col in feature_cols:
        df_pad[col] = df_pad[col].fillna(pad_value)

    return df_pad[feature_cols]


@dataclass(kw_only=True)
class StockStreamer:
    """Slice one stock's OHLC history into fixed-length feature windows.

    On construction the frame is split-adjusted (if ``stock_split`` is given),
    feature-extracted, and indexed by window start positions.

    Attributes
    ----------
    ohlc_df : pandas.DataFrame
        Raw OHLCV history for one stock.
    seq_len : int
        Window length.
    offset : int
        Stride between windows; ``-1`` means non-overlapping (``= seq_len``).
    stock_split : StockSplit, optional
        Split history to back-adjust prices with.
    shuffle : bool, optional
        If ``True``, iterate windows in random order.
    """

    ohlc_df: pd.DataFrame
    seq_len: int
    offset: int
    stock_split: StockSplit | None = None
    shuffle: bool = False

    def __post_init__(self) -> None:
        if len(self.ohlc_df) < 1:
            self.starts: np.ndarray[Any, Any] | list[Any] = []
            self.iterator = iter(self.create_iterator())
            return

        # apply stock splits
        if self.stock_split is not None:
            self.ohlc_df = self.stock_split.apply_splits(self.ohlc_df)
        self.preprocessed_ohlc_df = extract_features(self.ohlc_df)
        self.iterator = iter(self.create_iterator())
        if self.offset == -1:
            self.offset = self.seq_len

        self.starts = get_starts(self.preprocessed_ohlc_df, self.seq_len, self.offset)

    @property
    def size(self) -> int:
        """Number of windows available for this stock."""
        return len(self.starts)

    def get_starting_close(self, df: pd.DataFrame) -> float:
        """Return the last raw close at or before the window's start date.

        Parameters
        ----------
        df : pandas.DataFrame
            A window slice (used only for its start date).

        Returns
        -------
        float
            The anchor close used to reconstruct absolute prices.
        """
        date = df.index.min()
        date_mask = self.ohlc_df.index <= date
        return float(self.ohlc_df[date_mask].iloc[-1]["close"])

    def get_ohlcs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reconstruct absolute target/predicted OHLC candles from returns.

        Parameters
        ----------
        df : pandas.DataFrame
            A :meth:`OHLCMulitClassPredictorInput.to_pandas` frame holding the
            target/predicted ``r_close`` / ``upside`` / ``downside`` series.

        Returns
        -------
        pandas.DataFrame
            ``df`` with absolute ``target_*`` and ``predicted_*`` open / high /
            low / close columns (padding days removed).
        """
        start_close = self.get_starting_close(df)

        # remove pad days
        trade_mask = df["trade_occured"].to_numpy().reshape(-1)
        df = df.loc[trade_mask]

        df = df.assign(
            **{
                "target_close": start_close * np.exp(df["target_r_close"].cumsum()),
                "predicted_close": start_close * np.exp(df["predicted_r_close"].cumsum()),
            }
        )

        df["target_open"] = df["target_close"].shift(1)
        df["predicted_open"] = df["predicted_close"].shift(1)

        df["target_high"] = df["target_close"] * df["target_upside"]
        df["predicted_high"] = df["predicted_close"] * df["predicted_upside"]

        df["target_low"] = df["target_close"] * df["target_downside"]
        df["predicted_low"] = df["predicted_close"] * df["predicted_downside"]

        return df

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, i: int) -> pd.DataFrame:
        """Return the window of length ``seq_len`` starting at row ``i``."""
        return self.preprocessed_ohlc_df.iloc[i : i + self.seq_len]

    def create_iterator(self) -> Iterator[pd.DataFrame]:
        """Yield each window once, in random order when ``shuffle`` is set.

        Yields
        ------
        pandas.DataFrame
            A ``seq_len``-row slice of the preprocessed frame.
        """
        starts = np.arange(len(self.starts))
        if self.shuffle:
            np.random.shuffle(starts)
        for start in starts:
            start = self.starts[start]
            yield self.preprocessed_ohlc_df.iloc[start : start + self.seq_len]

    def next(self) -> pd.DataFrame:
        """Return the next window, advancing the internal iterator."""
        return next(self.iterator)


@dataclass(kw_only=True)
class StockHanlder:
    """Discover, load, filter, and stream a collection of stocks.

    Wraps a directory of per-symbol parquet files, applying optional
    year/volume/history filters and producing either raw daily frames or
    :class:`StockStreamer` objects.

    Attributes
    ----------
    seq_len : int
        Window length passed to produced streamers.
    base_path : str
        Directory of per-symbol parquet partitions.
    return_stock_id : bool
        Reserved flag for callers that also want the symbol id.
    return_streamer : bool
        If ``True``, indexing returns a :class:`StockStreamer`; otherwise a
        raw daily :class:`pandas.DataFrame`.
    stock_splits : dict[str, StockSplit], optional
        Split history keyed by symbol.
    shuffle : bool, optional
        Propagated to produced streamers.
    offset : int, optional
        Window stride; ``-1`` means non-overlapping.
    min_year, max_year : int, optional
        Inclusive lower / exclusive upper year filters.
    min_volume : float, optional
        Drop stocks whose mean volume is below this threshold.
    min_ticker_history : int, optional
        Drop stocks spanning fewer than this many days.
    """

    seq_len: int
    base_path: str
    return_stock_id: bool
    return_streamer: bool
    stock_splits: dict[str, StockSplit | None] | None = None
    shuffle: bool = False
    offset: int = -1
    min_year: int | None = None
    max_year: int | None = None
    min_volume: float | None = None
    min_ticker_history: int | None = None
    stocks: list[str] = field(init=False)
    stock_dict: dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        self.stock_dict = get_stock_parquets(self.base_path)
        self.stocks = list(self.stock_dict.keys())

        if self.offset == -1:
            self.offset = self.seq_len

    def __len__(self) -> int:
        return len(self.stocks)

    def __getitem__(self, index: int | str) -> StockStreamer | pd.DataFrame:
        if isinstance(index, str):
            index = self.stocks.index(index)
        stock = self.stocks[index]
        return self.stock(stock)

    def stock(self, stock: str) -> StockStreamer | pd.DataFrame:
        """Return one stock as a streamer or a raw frame.

        Parameters
        ----------
        stock : str
            The ticker symbol.

        Returns
        -------
        StockStreamer or pandas.DataFrame
            Depending on ``return_streamer``.
        """
        if self.return_streamer:
            return self.stock_streamer(stock)
        else:
            return self.stock_df(stock)

    def keep_stocks(self, stock_list: Iterable[str]) -> None:
        """Restrict the handler to the intersection with ``stock_list``.

        Parameters
        ----------
        stock_list : Iterable[str]
            Symbols to keep; others are dropped.
        """
        stock_list = list(stock_list)
        kept_stocks = {
            stock: self.stock_dict[stock] for stock in stock_list if stock in self.stock_dict
        }
        not_found = [stock for stock in stock_list if stock not in self.stock_dict]
        print(
            f"stocks kept: {len(kept_stocks)}/{len(self.stock_dict)}, "
            f"stocks not found: {len(not_found)}"
        )
        self.stock_dict = kept_stocks
        self.stocks = list(self.stock_dict.keys())

    def stock_df(self, stock: str) -> pd.DataFrame:
        """Load and daily-aggregate one stock, applying configured filters.

        Parameters
        ----------
        stock : str
            The ticker symbol.

        Returns
        -------
        pandas.DataFrame
            A date-indexed daily OHLCV frame, or an empty frame if the stock
            fails the volume / history filters.
        """
        path = self.stock_dict[stock]
        df = pd.read_parquet(path)

        if self.min_volume is not None and df["volume"].mean() < self.min_volume:
            return pd.DataFrame()

        if self.min_ticker_history is not None:
            days_spanned = (df["utc_time"].max() - df["utc_time"].min()).days
            if days_spanned < self.min_ticker_history:
                return pd.DataFrame()
        df["date"] = df["utc_time"].dt.normalize()
        df = (
            df.groupby("date")
            .agg({"high": "max", "low": "min", "close": "last", "volume": "sum"})
            .sort_index()
        )
        df = df.dropna(subset=["high", "low", "close"])

        if len(df) < 1:
            return df

        if self.min_year is not None:
            df = df.loc[df.index.to_series().dt.year >= self.min_year]

        if self.max_year is not None:
            df = df.loc[df.index.to_series().dt.year < self.max_year]

        return df

    def stock_streamer(self, stock: str) -> StockStreamer:
        """Build a :class:`StockStreamer` for one stock.

        Parameters
        ----------
        stock : str
            The ticker symbol.

        Returns
        -------
        StockStreamer
            A streamer over the (split-adjusted) loaded frame.
        """
        stock_split = None
        if self.stock_splits is not None and stock in self.stock_splits:
            stock_split = self.stock_splits[stock]

        return StockStreamer(
            ohlc_df=self.stock_df(stock),
            seq_len=self.seq_len,
            offset=self.offset,
            shuffle=self.shuffle,
            stock_split=stock_split,
        )


def extract_model_data(
    df: pd.DataFrame,
    response_size: int | np.ndarray[Any, Any],
    return_date: bool = False,
) -> dict[str, Any]:
    """Package a feature window into the model's input tensors.

    Parameters
    ----------
    df : pandas.DataFrame
        A preprocessed feature window (output of :func:`extract_features`).
    response_size : int or numpy.ndarray
        Number of trailing days the model must predict. Either a Python
        ``int`` or a length-1 ``numpy`` array (as produced by the dataset
        wrappers). Both produce a tensor that the ``LightningOHLCPredictor``
        normalizes via ``squeeze``.
    return_date : bool, optional
        If ``True``, also include the ``time`` index. Defaults to ``False``.

    Returns
    -------
    dict
        Keys ``feature_input``, ``targets``, ``trade_occured``,
        ``response_size`` (and ``time`` when ``return_date`` is set), suitable
        for :class:`~ophir.model_data.OHLCMulitClassPredictorInput`.
    """
    features = [c for c, d in zip(df.columns, df.dtypes, strict=False) if d != np.bool]
    feature_input = df[features].to_numpy()
    targets = df[["r_close", "upside", "downside"]].to_numpy()
    trade_occured = df["trade_occured"].to_numpy()
    model_data: dict[str, Any] = {
        "feature_input": torch.from_numpy(feature_input).float(),
        "targets": torch.from_numpy(targets).float(),
        "trade_occured": torch.from_numpy(trade_occured),
        "response_size": torch.from_numpy(np.array([response_size])),
    }
    if return_date:
        model_data["time"] = df.index.to_numpy()
    return model_data


class StockStreamerDataset(Dataset[dict[str, Any]]):
    """Map-style dataset over a fixed list of :class:`StockStreamer` objects.

    Indices address windows across all streamers via cumulative lengths; an
    exhausted streamer's iterator is transparently restarted.
    """

    def __init__(
        self,
        stock_streamers: list[StockStreamer],
        response_size: int,
        return_date: bool = False,
    ) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        stock_streamers : list[StockStreamer]
            The streamers to draw windows from.
        response_size : int
            Number of trailing days to predict.
        return_date : bool, optional
            Forwarded to :func:`extract_model_data`. Defaults to ``False``.
        """
        self.stock_streamers = stock_streamers
        self.iterators = [iter(streamer.create_iterator()) for streamer in self.stock_streamers]
        self.lengths = np.array(np.cumsum([stremer.size for stremer in self.stock_streamers]))

        self.response_size = np.array([response_size])
        self.return_date = return_date

    def __len__(self) -> int:
        return int(self.lengths[-1])

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = np.argwhere(self.lengths > index)[0].squeeze()
        try:
            df = next(self.iterators[index])
        except StopIteration:
            self.iterators[index] = iter(self.stock_streamers[index].create_iterator())
            df = next(self.iterators[index])
        return extract_model_data(df, self.response_size, self.return_date)


class StockHandlerDataset(IterableDataset[dict[str, Any]]):
    """Iterable dataset that streams windows from a :class:`StockHanlder`.

    Shards stocks across DataLoader workers and keeps a small rotating cache
    of active streamers, sampling a window from a random cached streamer at a
    time.
    """

    def __init__(
        self, stock_hanlder: StockHanlder, response_size: int, cache_size: int = 1
    ) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        stock_hanlder : StockHanlder
            The handler whose stocks will be streamed.
        response_size : int
            Number of trailing days to predict.
        cache_size : int, optional
            Number of streamers kept active concurrently. Defaults to ``1``.
        """
        self.stock_hanlder = stock_hanlder
        self.response_size = np.array([response_size])
        self.cache_size = cache_size

        print(
            f"Creating StockHandlerDataset with offset: {self.stock_hanlder.offset} "
            f"and cache: {self.cache_size}"
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Yield model-input dicts, sharded per worker.

        Yields
        ------
        dict
            An :func:`extract_model_data` payload for a sampled window.
        """
        worker_info = get_worker_info()
        if worker_info is None:
            # Single-process data loading
            start = 0
            step = 1
        else:
            # Multi-worker data loading
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            start = worker_id
            step = num_workers

        processed_stocks = 0
        cache: list[StockStreamer] = []
        cur_stock = 0
        shard_stock_indices = np.arange(start, len(self.stock_hanlder), step)
        while processed_stocks < len(shard_stock_indices):
            if len(cache) < self.cache_size and cur_stock < len(shard_stock_indices):
                stock_ind = shard_stock_indices[cur_stock]
                streamer = self.stock_hanlder[stock_ind]
                cache.append(streamer)
                cur_stock += 1

            cache_index = np.random.randint(len(cache))

            try:
                df = cache[cache_index].next()
                yield extract_model_data(df, self.response_size)
            except StopIteration:
                processed_stocks += 1
                cache.pop(cache_index)
