"""Builders that package feature windows into model-input tensors."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ophir.ticker.handler import StockHandler
from ophir.ticker.streamer import StockStreamer

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd  # type: ignore[import-untyped]


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
