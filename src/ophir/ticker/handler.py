"""StockHandler: discover, load, filter, and stream a collection of stocks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import pandas as pd  # type: ignore[import-untyped]

from ophir.ticker.features import clean_daily_ohlcv
from ophir.ticker.paths import get_stock_parquets
from ophir.ticker.streamer import StockStreamer

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ophir.ticker.splits import StockSplit


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
