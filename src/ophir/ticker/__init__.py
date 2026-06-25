"""Stock data ingestion, split adjustment, feature extraction, and datasets.

This module turns per-stock parquet files into the fixed-length feature
windows the model consumes. The pipeline is: discover parquet files
(:func:`get_stock_parquets`), load and filter them (:class:`StockHandler`),
optionally back-adjust for splits (:class:`StockSplit`), compute the
12-feature representation (:func:`extract_features`), slice into windows
(:class:`StockStreamer`), and expose them as ``torch`` datasets
(:class:`StockStreamerDataset`, :class:`StockHandlerDataset`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from ophir.ticker.features import clean_daily_ohlcv, extract_features
from ophir.ticker.paths import get_start_dates, get_starts, get_stock_parquets
from ophir.ticker.splits import StockSplit, get_sp_500_symbols, get_splits
from ophir.ticker.streamer import StockStreamer

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

__all__ = [
    "StockHandler",
    "StockHandlerDataset",
    "StockSplit",
    "StockStreamer",
    "StockStreamerDataset",
    "build_latest_inputs",
    "clean_daily_ohlcv",
    "extract_features",
    "extract_model_data",
    "get_sp_500_symbols",
    "get_splits",
    "get_start_dates",
    "get_starts",
    "get_stock_parquets",
]


@dataclass(kw_only=True)
class StockHandler:
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
    clean_rows : bool, optional
        If ``True``, run :func:`clean_daily_ohlcv` on each stock's daily frame
        (dropping zero-volume and return-spike rows) before slicing windows.
        Defaults to ``False``.
    max_abs_r_close : float, optional
        Return-spike threshold forwarded to :func:`clean_daily_ohlcv` when
        ``clean_rows`` is set. Defaults to ``0.75``.
    cache_frames : bool, optional
        If ``True``, memoize each symbol's loaded daily frame so repeated
        streaming epochs reuse it instead of re-reading the parquet/SQLite
        source. The cached frame is only ever read downstream (split-adjustment
        and feature extraction return new frames), so this is output-identical.
        Defaults to ``False``.
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
    clean_rows: bool = False
    max_abs_r_close: float = 0.75
    source: Literal["parquet", "sqlite"] = "parquet"
    cache_frames: bool = False
    stocks: list[str] = field(init=False)
    stock_dict: dict[str, str] = field(init=False)
    _frame_cache: dict[str, pd.DataFrame] = field(init=False)

    def __post_init__(self) -> None:
        if self.source == "sqlite":
            from ophir.sqlite_store import get_stock_tables

            self.stock_dict = get_stock_tables(self.base_path)
        else:
            self.stock_dict = get_stock_parquets(self.base_path)
        self.stocks = list(self.stock_dict.keys())
        self._frame_cache = {}

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

        When ``cache_frames`` is set, the result (including filter-rejected
        empty frames) is memoized per symbol so subsequent epochs skip the disk
        read and aggregation.

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
        if self.cache_frames and stock in self._frame_cache:
            return self._frame_cache[stock]
        df = self._load_stock_df(stock)
        if self.cache_frames:
            self._frame_cache[stock] = df
        return df

    def _load_stock_df(self, stock: str) -> pd.DataFrame:
        """Read and daily-aggregate one stock's source frame (uncached)."""
        if self.source == "sqlite":
            from ophir.sqlite_store import read_stock_table

            df = read_stock_table(self.base_path, self.stock_dict[stock])
        else:
            df = pd.read_parquet(self.stock_dict[stock])

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

        if self.clean_rows:
            df = clean_daily_ohlcv(df, max_abs_r_close=self.max_abs_r_close)
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
            symbol=stock,
        )


def extract_model_data(
    df: pd.DataFrame,
    response_size: int | np.ndarray[Any, Any],
    return_date: bool = False,
    stock_id: int | None = None,
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
    stock_id : int, optional
        If given, also include opt-in eval identity: ``stock_id`` (a 0-dim
        ``long`` tensor) and ``date_ordinal`` (an int64 calendar-day ordinal of
        shape ``(seq_len,)`` from ``df.index``). Defaults to ``None`` (the keys
        are omitted, so the training path is unaffected).

    Returns
    -------
    dict
        Keys ``feature_input``, ``targets``, ``trade_occured``,
        ``response_size`` (and ``time`` when ``return_date`` is set, and
        ``stock_id`` / ``date_ordinal`` when ``stock_id`` is given), suitable
        for :class:`~ophir.model_data.OHLCMultiClassPredictorInput`.
    """
    features = [c for c, d in zip(df.columns, df.dtypes, strict=False) if d != np.dtype(bool)]
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
    if stock_id is not None:
        model_data["stock_id"] = torch.tensor(stock_id, dtype=torch.long)
        ordinals = df.index.to_numpy().astype("datetime64[D]").astype(np.int64)
        model_data["date_ordinal"] = torch.from_numpy(ordinals)
    return model_data


def build_latest_inputs(
    symbols: Sequence[str],
    seq_len: int = 365,
    base_path: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the most-recent inference window per symbol at ``response_size=1``.

    For each requested symbol, loads its history from the parquet tree, takes the
    single most-recent ``seq_len``-row feature window, and packages it with
    :func:`extract_model_data` for an offset-1 (day-1) forecast. Symbols absent
    from the tree, or with too little history to form one window, are silently
    skipped — the live forecast seam degrades rather than raising.

    Parameters
    ----------
    symbols : sequence of str
        Ticker symbols to build inference windows for.
    seq_len : int, optional
        Window length the model consumes. Defaults to ``365``.
    base_path : str, optional
        Root of the Hive-partitioned parquet tree. When ``None``, defaults to
        ``register.get_default_data_days_dir()/stocks``.

    Returns
    -------
    dict[str, dict[str, Any]]
        ``{symbol: extract_model_data payload}`` for each symbol that produced a
        window. Empty when no requested symbol is available.
    """
    if base_path is None:
        from ophir import register

        base_path = os.path.join(register.get_default_data_days_dir(), "stocks")

    handler = StockHandler(
        seq_len=seq_len,
        base_path=base_path,
        return_stock_id=False,
        return_streamer=True,
    )
    out: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        try:
            streamer = handler[symbol]
        except ValueError:
            continue  # symbol not in the tree
        if not isinstance(streamer, StockStreamer) or streamer.size == 0:
            continue  # too little history to form a window
        window = streamer[int(streamer.starts[-1])]  # most-recent window
        out[symbol] = extract_model_data(window, 1)
    return out


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
    """Iterable dataset that streams windows from a :class:`StockHandler`.

    Shards stocks across DataLoader workers and keeps a small rotating cache
    of active streamers, sampling a window from a random cached streamer at a
    time.
    """

    def __init__(
        self,
        stock_hanlder: StockHandler,
        response_size: int,
        cache_size: int = 8,
        return_identity: bool = False,
    ) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        stock_hanlder : StockHandler
            The handler whose stocks will be streamed.
        response_size : int
            Number of trailing days to predict.
        cache_size : int, optional
            Number of streamers kept active concurrently. Defaults to ``8``
            (matching the training default) so windows mix across stocks; a
            cache of ``1`` drains one stock fully before the next, yielding
            strongly autocorrelated batches.
        return_identity : bool, optional
            If ``True``, each yielded payload also carries the opt-in eval
            identity (``stock_id`` / ``date_ordinal``); see
            :func:`extract_model_data`. Defaults to ``False`` so the training
            path is unaffected.
        """
        self.stock_hanlder = stock_hanlder
        self.response_size = np.array([response_size])
        self.cache_size = cache_size
        self.return_identity = return_identity

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
        cache: list[tuple[int, StockStreamer]] = []
        cur_stock = 0
        shard_stock_indices = np.arange(start, len(self.stock_hanlder), step)
        while processed_stocks < len(shard_stock_indices):
            if len(cache) < self.cache_size and cur_stock < len(shard_stock_indices):
                stock_ind = int(shard_stock_indices[cur_stock])
                streamer = self.stock_hanlder[stock_ind]
                cache.append((stock_ind, streamer))
                cur_stock += 1

            cache_index = np.random.randint(len(cache))
            stock_ind, streamer = cache[cache_index]

            try:
                df = streamer.next()
                sid = stock_ind if self.return_identity else None
                yield extract_model_data(df, self.response_size, stock_id=sid)
            except StopIteration:
                processed_stocks += 1
                cache.pop(cache_index)
