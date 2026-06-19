"""Tests for ``StockHanlder`` discovery, filters, indexing, and keep_stocks."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ophir.sqlite_store import build_sqlite_store
from ophir.ticker import StockHanlder, StockStreamer


def _handler(base_path, **kwargs):
    defaults = {
        "seq_len": 20,
        "base_path": base_path,
        "return_stock_id": False,
        "return_streamer": False,
    }
    defaults.update(kwargs)
    return StockHanlder(**defaults)


# --------------------------------------------------------------------------- #
# frame cache
# --------------------------------------------------------------------------- #


def test_stock_df_memoizes_when_cache_frames_enabled(parquet_dir):
    # With caching on, repeated loads of the same symbol reuse one frame instead
    # of re-reading parquet and re-aggregating every epoch.
    base_path, _ = parquet_dir
    handler = _handler(base_path, cache_frames=True)
    stock = handler.stocks[0]
    assert handler.stock_df(stock) is handler.stock_df(stock)


def test_stock_df_rereads_when_cache_disabled(parquet_dir):
    base_path, _ = parquet_dir
    handler = _handler(base_path)  # cache_frames defaults to False
    stock = handler.stocks[0]
    assert handler.stock_df(stock) is not handler.stock_df(stock)


# --------------------------------------------------------------------------- #
# discovery / indexing
# --------------------------------------------------------------------------- #


def test_handler_discovers_stocks(parquet_dir):
    base_path, _ = parquet_dir
    handler = _handler(base_path)

    assert len(handler) == 3
    assert set(handler.stocks) == {"AAA", "BBB", "CCC"}
    assert handler.offset == handler.seq_len == 20  # offset == -1 remapped


def test_handler_int_and_symbol_indexing_agree(parquet_dir):
    base_path, _ = parquet_dir
    handler = _handler(base_path)

    by_symbol = handler["AAA"]
    by_int = handler[handler.stocks.index("AAA")]

    assert isinstance(by_symbol, pd.DataFrame)
    assert by_symbol.equals(by_int)


def test_handler_unknown_symbol_raises_valueerror(parquet_dir):
    base_path, _ = parquet_dir
    handler = _handler(base_path)

    with pytest.raises(ValueError, match="not in list"):
        _ = handler["NOPE"]


def test_handler_return_streamer_and_split_passthrough(parquet_dir, stock_split):
    base_path, _ = parquet_dir

    plain = _handler(base_path, return_streamer=True)
    assert isinstance(plain["AAA"], StockStreamer)
    assert plain["AAA"].stock_split is None

    with_split = _handler(base_path, return_streamer=True, stock_splits={"AAA": stock_split})
    assert with_split["AAA"].stock_split is stock_split


def test_handler_shuffle_propagates_to_streamers(parquet_dir):
    base_path, _ = parquet_dir
    handler = _handler(base_path, return_streamer=True, shuffle=True)

    streamer = handler["AAA"]
    assert streamer.shuffle is True


# --------------------------------------------------------------------------- #
# stock_df aggregation + filters
# --------------------------------------------------------------------------- #


def test_stock_df_daily_aggregation(parquet_dir):
    base_path, _ = parquet_dir
    df = _handler(base_path).stock_df("AAA")

    assert list(df.columns) == ["high", "low", "close", "volume"]
    assert df.index.is_monotonic_increasing
    assert not df[["high", "low", "close"]].isna().to_numpy().any()
    assert len(df) == 80  # 80 business days, two intraday ticks each


def test_min_volume_filter_drops_low_volume_stock(parquet_dir):
    base_path, _ = parquet_dir
    handler = _handler(base_path, min_volume=1000)

    assert handler.stock_df("CCC").empty  # volume == 1.0
    assert not handler.stock_df("AAA").empty


def test_min_ticker_history_filter(parquet_dir):
    base_path, _ = parquet_dir
    handler = _handler(base_path, min_ticker_history=30)

    assert handler.stock_df("BBB").empty  # ~5-day span
    assert not handler.stock_df("AAA").empty


def test_year_filters(parquet_dir):
    base_path, _ = parquet_dir

    assert _handler(base_path, min_year=2021).stock_df("AAA").empty
    assert _handler(base_path, max_year=2020).stock_df("AAA").empty
    full = _handler(base_path, min_year=2020, max_year=2021).stock_df("AAA")
    assert len(full) == 80


def test_combined_filters_intersection(parquet_dir):
    # All three filters active at once: AAA passes everything; BBB fails on
    # history; CCC fails on volume. Each filter's bypass path stays correct
    # when its peers are also configured.
    base_path, _ = parquet_dir
    handler = _handler(
        base_path,
        min_volume=1000,
        min_ticker_history=30,
        min_year=2020,
        max_year=2021,
    )

    assert not handler.stock_df("AAA").empty
    assert handler.stock_df("BBB").empty  # ~5-day span < min_ticker_history
    assert handler.stock_df("CCC").empty  # volume == 1.0 < min_volume


def test_stock_df_all_nan_returns_empty(tmp_path):
    part = tmp_path / "symbol=NAN"
    part.mkdir()
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    pd.DataFrame(
        {
            "utc_time": idx,
            "high": np.full(10, np.nan),
            "low": np.full(10, np.nan),
            "close": np.full(10, np.nan),
            "volume": np.full(10, 500.0),
        }
    ).to_parquet(part / "data.parquet")

    df = _handler(str(tmp_path)).stock_df("NAN")
    assert df.empty


# --------------------------------------------------------------------------- #
# clean_rows row-level cleaning
# --------------------------------------------------------------------------- #


def _glitch_parquet(tmp_path):
    part = tmp_path / "symbol=GLT"
    part.mkdir()
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    close = np.full(10, 100.0)
    close[5] = 1000.0  # spike up at idx5, snap-back down at idx6
    volume = np.full(10, 500.0)
    volume[3] = 0.0  # zero-volume day
    pd.DataFrame(
        {
            "utc_time": idx,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": volume,
        }
    ).to_parquet(part / "data.parquet")
    return str(tmp_path)


def test_clean_rows_disabled_by_default(tmp_path):
    base = _glitch_parquet(tmp_path)
    assert len(_handler(base).stock_df("GLT")) == 10


def test_clean_rows_drops_zero_volume_and_spikes(tmp_path):
    base = _glitch_parquet(tmp_path)
    df = _handler(base, clean_rows=True).stock_df("GLT")

    # zero-volume day (1) + spike-up + snap-back (2) removed.
    assert len(df) == 7
    assert (df["volume"] > 0).all()


# --------------------------------------------------------------------------- #
# keep_stocks (the fixed bug)
# --------------------------------------------------------------------------- #


def test_keep_stocks_reports_not_found(parquet_dir, capsys):
    base_path, _ = parquet_dir
    handler = _handler(base_path)

    handler.keep_stocks(["AAA", "ZZZ", "QQQ"])

    assert handler.stocks == ["AAA"]
    assert set(handler.stock_dict) == {"AAA"}
    out = capsys.readouterr().out
    assert "stocks kept: 1/3" in out
    assert "stocks not found: 2" in out


def test_keep_stocks_empty_list(parquet_dir, capsys):
    base_path, _ = parquet_dir
    handler = _handler(base_path)

    handler.keep_stocks([])

    assert handler.stocks == []
    out = capsys.readouterr().out
    assert "stocks kept: 0/3" in out
    assert "stocks not found: 0" in out


def test_keep_stocks_handles_duplicates_and_generators(parquet_dir, capsys):
    base_path, _ = parquet_dir
    handler = _handler(base_path)

    handler.keep_stocks(s for s in ["AAA", "AAA", "ZZZ"])  # generator input

    assert handler.stocks == ["AAA"]
    out = capsys.readouterr().out
    assert "stocks kept: 1/3" in out
    assert "stocks not found: 1" in out


# --------------------------------------------------------------------------- #
# source toggle: sqlite vs parquet
# --------------------------------------------------------------------------- #


def test_stockhandler_sqlite_source_matches_parquet(parquet_dir, tmp_path):
    base_path, _paths = parquet_dir
    db_path = str(tmp_path / "stocks.db")
    build_sqlite_store(base_path, db_path)

    pq = _handler(base_path)
    sq = _handler(db_path, source="sqlite")

    assert set(sq.stocks) == set(pq.stocks)
    for sym in pq.stocks:
        assert_frame_equal(sq.stock_df(sym), pq.stock_df(sym))


def test_stockhandler_defaults_to_parquet(parquet_dir):
    base_path, _ = parquet_dir
    assert _handler(base_path).source == "parquet"
