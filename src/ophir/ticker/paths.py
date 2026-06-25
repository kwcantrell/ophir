"""Parquet-file discovery and window-index math for the ticker pipeline."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


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


def get_starts(
    df: pd.DataFrame, seq_len: int, offset: int, first_valid_row: int = 0
) -> np.ndarray[Any, Any]:
    """Compute window start indices spanning ``df``.

    Parameters
    ----------
    df : pandas.DataFrame
        The (preprocessed) frame to slice.
    seq_len : int
        Window length.
    offset : int
        Stride between consecutive window starts.
    first_valid_row : int, optional
        Minimum start index; windows before this position contain warm-up
        rows and are excluded. Defaults to ``0`` (no restriction).

    Returns
    -------
    numpy.ndarray
        Integer start positions ``first_valid_row, first_valid_row+offset, …``
        up to ``len(df) - seq_len``.
    """
    num_start = max(0, len(df) - seq_len + 1)
    starts = np.arange(first_valid_row, num_start, offset)
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
    starts = np.arange(0, max(0, len(calendar) - seq_len + 1), offset)
    return np.asarray(calendar[starts].to_numpy())
