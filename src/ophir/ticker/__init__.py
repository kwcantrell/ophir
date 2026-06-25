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
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from ophir.ticker.features import clean_daily_ohlcv, extract_features
from ophir.ticker.handler import StockHandler
from ophir.ticker.paths import get_start_dates, get_starts, get_stock_parquets
from ophir.ticker.splits import StockSplit, get_sp_500_symbols, get_splits
from ophir.ticker.streamer import StockStreamer

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import pandas as pd  # type: ignore[import-untyped]

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
