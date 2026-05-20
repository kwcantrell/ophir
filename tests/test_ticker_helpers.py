"""Tests for the pure helpers: parquet discovery, window starts, splits."""

import numpy as np
import pandas as pd
import pytest

from ophir.ticker import StockSplit, get_start_dates, get_starts, get_stock_parquets

# --------------------------------------------------------------------------- #
# get_stock_parquets
# --------------------------------------------------------------------------- #


def test_get_stock_parquets_discovers_partitions(parquet_dir):
    base_path, _ = parquet_dir
    result = get_stock_parquets(base_path)

    assert set(result) == {"AAA", "BBB", "CCC"}  # decoy "_logs" excluded
    for symbol, path in result.items():
        assert path.endswith(".parquet")
        assert f"symbol={symbol}" in path
        # the discovered path is real and loadable
        pd.read_parquet(path)


def test_get_stock_parquets_ignores_dirs_without_equals(tmp_path):
    (tmp_path / "symbol=AAA").mkdir()
    (tmp_path / "symbol=AAA" / "x.parquet").write_bytes(b"")
    (tmp_path / "not_a_partition").mkdir()

    result = get_stock_parquets(str(tmp_path))

    assert list(result) == ["AAA"]


def test_get_stock_parquets_malformed_partition_raises_typeerror(tmp_path):
    # Partition dir with no .parquet inside -> inner parquet() returns None ->
    # os.path.join(base, None) raises. Pinned latent behavior (ambiguous fix).
    (tmp_path / "symbol=ZZZ").mkdir()

    with pytest.raises(TypeError):
        get_stock_parquets(str(tmp_path))


# --------------------------------------------------------------------------- #
# get_starts
# --------------------------------------------------------------------------- #


def _frame(n_rows):
    return pd.DataFrame(np.zeros((n_rows, 1)))


def test_get_starts_regular_stride():
    starts = get_starts(_frame(120), seq_len=30, offset=10)
    np.testing.assert_array_equal(starts, np.arange(0, 90, 10))


def test_get_starts_offset_larger_than_span_yields_single_start():
    starts = get_starts(_frame(50), seq_len=40, offset=100)
    np.testing.assert_array_equal(starts, np.array([0]))


def test_get_starts_seq_len_ge_len_is_empty():
    assert get_starts(_frame(30), seq_len=30, offset=5).size == 0
    assert get_starts(_frame(10), seq_len=40, offset=5).size == 0


# --------------------------------------------------------------------------- #
# get_start_dates
# --------------------------------------------------------------------------- #


def test_get_start_dates_matches_calendar_slice(ohlcv_df):
    seq_len, offset = 10, 5
    result = get_start_dates(ohlcv_df, seq_len, offset)

    calendar = pd.date_range(ohlcv_df.index.min(), ohlcv_df.index.max(), freq="D")
    expected = calendar[np.arange(0, len(calendar) - seq_len, offset)].to_numpy()

    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.dtype("datetime64[ns]")


def test_get_start_dates_seq_len_exceeds_calendar_is_empty(ohlcv_df):
    calendar = pd.date_range(ohlcv_df.index.min(), ohlcv_df.index.max(), freq="D")
    result = get_start_dates(ohlcv_df, seq_len=len(calendar) + 5, offset=3)

    assert result.size == 0
    assert result.dtype == np.dtype("datetime64[ns]")


# --------------------------------------------------------------------------- #
# StockSplit.apply_splits
# --------------------------------------------------------------------------- #


def _ohlcv_for_splits():
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    return pd.DataFrame(
        {
            "high": np.full(6, 110.0),
            "low": np.full(6, 90.0),
            "close": np.full(6, 100.0),
            "volume": np.full(6, 1000.0),
        },
        index=idx,
    )


def test_apply_splits_back_adjusts_close_and_volume():
    df = _ohlcv_for_splits()
    split = StockSplit(id="X", dates=[np.datetime64("2020-01-04")], ratios=[2.0])

    out = split.apply_splits(df)

    split_date = pd.Timestamp("2020-01-04")
    before = out.index < split_date
    after = ~before

    np.testing.assert_allclose(out.loc[before, "close"], 50.0)
    np.testing.assert_allclose(out.loc[before, "volume"], 2000.0)
    np.testing.assert_allclose(out.loc[after, "close"], 100.0)
    np.testing.assert_allclose(out.loc[after, "volume"], 1000.0)


def test_apply_splits_without_volume_column():
    df = _ohlcv_for_splits().drop(columns=["volume"])
    split = StockSplit(id="X", dates=[np.datetime64("2020-01-04")], ratios=[2.0])

    out = split.apply_splits(df)

    assert "volume" not in out.columns
    np.testing.assert_allclose(out.loc[out.index < pd.Timestamp("2020-01-04"), "close"], 50.0)


def test_apply_splits_sorts_index():
    df = _ohlcv_for_splits().iloc[::-1]  # reversed -> not monotonic
    split = StockSplit(id="X", dates=[np.datetime64("2020-01-04")], ratios=[2.0])

    out = split.apply_splits(df)

    assert out.index.is_monotonic_increasing


def test_apply_splits_multiple_splits_are_cumulative():
    df = _ohlcv_for_splits()
    split = StockSplit(
        id="X",
        dates=[np.datetime64("2020-01-03"), np.datetime64("2020-01-05")],
        ratios=[2.0, 3.0],
    )

    out = split.apply_splits(df)

    # before both splits -> divided by 2*3 = 6
    np.testing.assert_allclose(out.loc[pd.Timestamp("2020-01-01"), "close"], 100.0 / 6.0)
    np.testing.assert_allclose(out.loc[pd.Timestamp("2020-01-01"), "volume"], 1000.0 * 6.0)
    # between the two splits -> divided by 3 only
    np.testing.assert_allclose(out.loc[pd.Timestamp("2020-01-04"), "close"], 100.0 / 3.0)
    # on/after the last split -> unchanged
    np.testing.assert_allclose(out.loc[pd.Timestamp("2020-01-05"), "close"], 100.0)


def test_apply_splits_no_splits_is_identity():
    df = _ohlcv_for_splits()
    split = StockSplit(id="X", dates=[], ratios=[])

    out = split.apply_splits(df)

    np.testing.assert_allclose(out["close"], 100.0)
    np.testing.assert_allclose(out["volume"], 1000.0)


def test_apply_splits_mismatched_dates_ratios_truncates():
    # zip(..., strict=False) silently truncates to the shorter sequence.
    # Pinned current behavior (defensive strict=True would be a behavior change).
    df = _ohlcv_for_splits()
    split = StockSplit(
        id="X",
        dates=[np.datetime64("2020-01-03"), np.datetime64("2020-01-05")],
        ratios=[2.0],  # second date has no ratio -> dropped
    )

    out = split.apply_splits(df)

    # only the 2020-01-03 split applied (factor 2 before that date)
    np.testing.assert_allclose(out.loc[pd.Timestamp("2020-01-01"), "close"], 50.0)
    np.testing.assert_allclose(out.loc[pd.Timestamp("2020-01-04"), "close"], 100.0)
