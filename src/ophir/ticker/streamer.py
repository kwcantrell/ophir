"""StockStreamer: slice one stock's history into fixed-length windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ophir.ticker.features import extract_features
from ophir.ticker.paths import get_starts

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pandas as pd  # type: ignore[import-untyped]

    from ophir.ticker.splits import StockSplit


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
    symbol : str, optional
        The ticker symbol this streamer was built for; carried for the opt-in
        eval identity path. Defaults to ``None``.
    """

    ohlc_df: pd.DataFrame
    seq_len: int
    offset: int
    stock_split: StockSplit | None = None
    shuffle: bool = False
    symbol: str | None = None

    def __post_init__(self) -> None:
        if len(self.ohlc_df) < 1:
            self.starts: np.ndarray[Any, Any] | list[Any] = []
            self.iterator = iter(self.create_iterator())
            return

        # apply stock splits
        if self.stock_split is not None:
            self.ohlc_df = self.stock_split.apply_splits(self.ohlc_df)
        self.preprocessed_ohlc_df = extract_features(self.ohlc_df)
        if self.offset == -1:
            self.offset = self.seq_len

        first_valid = int(self.preprocessed_ohlc_df["feature_valid"].to_numpy().argmax())
        self.starts = get_starts(self.preprocessed_ohlc_df, self.seq_len, self.offset, first_valid)
        self.iterator = iter(self.create_iterator())

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
            A :meth:`OHLCMultiClassPredictorInput.to_pandas` frame holding the
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
