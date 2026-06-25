"""OHLCV cleaning and the 12-feature model representation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


def clean_daily_ohlcv(
    df: pd.DataFrame,
    *,
    max_abs_r_close: float = 0.75,
    drop_zero_volume: bool = True,
) -> pd.DataFrame:
    """Drop point-in-time-safe bad rows from a daily OHLCV frame.

    Operates on the date-indexed daily frame produced by aggregation, *before*
    split-adjustment and :func:`extract_features`. Deterministic and
    lookahead-free: every decision uses only the row itself and its immediate
    *retained* predecessor close, consistent with the forecast-masking contract
    in ``CLAUDE.md``.

    Rows are dropped (never repaired) in two passes:

    1. Zero/negative-volume days (when ``drop_zero_volume``) — non-trading
       artifacts.
    2. Return-spike days where
       ``abs(log(close / prev_close)) > max_abs_r_close`` — split errors or
       data glitches. ``prev_close`` is the prior *surviving* close (the return
       is recomputed after the volume drop), so the check chains correctly. The
       first row, whose return is undefined, is always kept.

    Repairing spikes (e.g. interpolation) would invent prices and is hard to
    keep lookahead-safe, so this function only drops.

    Parameters
    ----------
    df : pandas.DataFrame
        Date-indexed daily OHLCV frame (``high`` / ``low`` / ``close`` /
        ``volume``), as produced by :meth:`StockHandler.stock_df`.
    max_abs_r_close : float, optional
        Maximum allowed absolute single-day log return. Defaults to ``0.75``
        (just above a 2:1 split's ``0.69``).
    drop_zero_volume : bool, optional
        Drop days with non-positive volume. Defaults to ``True``.

    Returns
    -------
    pandas.DataFrame
        ``df`` with the offending rows removed (empty in, empty out).
    """
    if df.empty:
        return df

    if drop_zero_volume:
        df = df.loc[df["volume"] > 0]

    if df.empty:
        return df

    r_close = np.log(df["close"] / df["close"].shift(1))
    keep = r_close.isna() | (r_close.abs() <= max_abs_r_close)
    return df.loc[keep]


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 12-feature model representation from an OHLCV frame.

    Features: log close return ``r_close``, rolling normalized returns /
    normalized volume / volatility over 10-, 20-, and 60-day windows, plus the
    ``upside`` and ``downside`` log ratios. The frame is reindexed onto a daily
    calendar; padded days are zero-filled and flagged by ``trade_occured``. All
    features are point-in-time (trailing rolling windows only), so this
    introduces no lookahead.

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
    if df.index.has_duplicates:
        raise ValueError(
            "extract_features requires a unique DatetimeIndex; got duplicate "
            "dates. Aggregate to one row per day before calling."
        )

    feature_cols: list[str] = []

    def add_feature(feature_col: str, feature_val: Any, df: pd.DataFrame) -> pd.DataFrame:
        df = df.assign(**{feature_col: feature_val})
        feature_cols.append(feature_col)
        return df

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

    # A row is "valid" only once every rolling feature is defined: the largest
    # rolling window is 60, so the first 59 rows are warm-up. Capture this on the
    # trading-day frame, before calendar padding adds non-trading rows.
    largest_window = 60
    valid = pd.Series(True, index=df.index)
    valid.iloc[: largest_window - 1] = False

    calendar = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df_pad = df.reindex(index=calendar)

    df_pad = add_feature("trade_occured", df_pad["close"].notna(), df_pad)
    # Padding rows are never valid; warm-up rows were flagged above. Anything not
    # explicitly valid (padding gaps) defaults to False.
    df_pad["feature_valid"] = valid.reindex(calendar, fill_value=False)

    pad_value = 0.0
    for col in feature_cols:
        df_pad[col] = df_pad[col].fillna(pad_value)

    return df_pad[[*feature_cols, "feature_valid"]]
