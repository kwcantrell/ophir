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

from typing import TYPE_CHECKING, Any

import numpy as np
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from ophir.ticker.features import clean_daily_ohlcv, extract_features
from ophir.ticker.handler import StockHandler
from ophir.ticker.inputs import build_latest_inputs, extract_model_data
from ophir.ticker.paths import get_start_dates, get_starts, get_stock_parquets
from ophir.ticker.splits import StockSplit, get_sp_500_symbols, get_splits
from ophir.ticker.streamer import StockStreamer

if TYPE_CHECKING:
    from collections.abc import Iterator

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
