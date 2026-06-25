"""Stock data ingestion, split adjustment, feature extraction, and datasets.

This package turns per-stock parquet files into the fixed-length feature
windows the model consumes: parquet discovery + window math
(:mod:`ophir.ticker.paths`), symbol/split data (:mod:`ophir.ticker.splits`),
cleaning + features (:mod:`ophir.ticker.features`), the streaming primitive
(:mod:`ophir.ticker.streamer`) and the handler over a collection of stocks
(:mod:`ophir.ticker.handler`), model-input builders
(:mod:`ophir.ticker.inputs`), and the ``torch`` datasets
(:mod:`ophir.ticker.datasets`). The public API is re-exported here, so
``from ophir.ticker import ...`` works exactly as it did when this was a single
module.
"""

from __future__ import annotations

from ophir.ticker.datasets import StockHandlerDataset, StockStreamerDataset
from ophir.ticker.features import clean_daily_ohlcv, extract_features
from ophir.ticker.handler import StockHandler
from ophir.ticker.inputs import build_latest_inputs, extract_model_data
from ophir.ticker.paths import get_start_dates, get_starts, get_stock_parquets
from ophir.ticker.splits import StockSplit, get_sp_500_symbols, get_splits
from ophir.ticker.streamer import StockStreamer

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
