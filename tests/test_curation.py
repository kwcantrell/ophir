"""Tests for the dataset-curation pipeline (:mod:`ophir.curation`).

Covers the shared row cleaner (:func:`ophir.ticker.clean_daily_ohlcv`), the
per-symbol quality metrics and verdict, the tree scan, the
``quality-symbols``/stats persistence helpers in :mod:`ophir.register`, and the
``curate`` command body. Everything is offline and never touches the package's
on-disk ``.ophir/`` layout (the persistence tests monkeypatch ``DATA_DIR``).
"""

import json

import numpy as np
import pandas as pd

from ophir import curation, register
from ophir.curation import QualityThresholds, compute_symbol_quality, curate_symbols
from ophir.ticker import clean_daily_ohlcv

# Thresholds that pass everything; individual tests tighten one knob at a time
# so each failure reason is isolated.
LENIENT = QualityThresholds(
    min_median_dollar_volume=0.0,
    min_trading_days=1,
    max_missing_day_fraction=1.0,
    min_median_close=0.0,
    max_return_spikes=10_000,
    max_abs_r_close=0.75,
    max_flat_run=10_000,
    max_zero_volume_fraction=1.0,
)


def _daily(n_days=120, base_price=100.0, volume=50_000.0, seed=7):
    """A clean date-indexed daily OHLCV frame on consecutive business days."""
    idx = pd.date_range("2020-01-01", periods=n_days, freq="B")
    rng = np.random.default_rng(seed)
    close = base_price * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=n_days)))
    return pd.DataFrame(
        {
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(n_days, float(volume)),
        },
        index=idx,
    )


def _raw(daily):
    """Convert a daily frame to the raw ``utc_time``-columned parquet shape."""
    return pd.DataFrame(
        {
            "utc_time": daily.index,
            "high": daily["high"].to_numpy(),
            "low": daily["low"].to_numpy(),
            "close": daily["close"].to_numpy(),
            "volume": daily["volume"].to_numpy(),
        }
    )


# --------------------------------------------------------------------------- #
# clean_daily_ohlcv
# --------------------------------------------------------------------------- #


def test_clean_drops_zero_and_negative_volume():
    daily = _daily(n_days=10)
    daily.iloc[3, daily.columns.get_loc("volume")] = 0.0
    daily.iloc[7, daily.columns.get_loc("volume")] = -5.0

    cleaned = clean_daily_ohlcv(daily)

    assert len(cleaned) == 8
    assert (cleaned["volume"] > 0).all()


def test_clean_drops_return_spike_and_keeps_neighbors():
    daily = _daily(n_days=10)
    daily.iloc[5, daily.columns.get_loc("close")] = daily["close"].iloc[4] * 5.0

    cleaned = clean_daily_ohlcv(daily, max_abs_r_close=0.75)

    # The spike day (up) and the snap-back day (down) both exceed the threshold.
    assert daily.index[5] not in cleaned.index
    assert daily.index[6] not in cleaned.index
    assert daily.index[4] in cleaned.index


def test_clean_spike_check_chains_after_volume_drop():
    # A zero-volume row sits between two normal closes; once it is dropped the
    # surviving neighbours define a normal return, so nothing else is dropped.
    daily = _daily(n_days=6)
    daily.iloc[2, daily.columns.get_loc("volume")] = 0.0

    cleaned = clean_daily_ohlcv(daily)

    assert len(cleaned) == 5
    assert daily.index[1] in cleaned.index
    assert daily.index[3] in cleaned.index


def test_clean_keeps_first_row_with_undefined_return():
    daily = _daily(n_days=4)
    # A huge *first* close has no predecessor, so its return is NaN -> kept.
    daily.iloc[0, daily.columns.get_loc("close")] = 1e6
    daily.iloc[0, daily.columns.get_loc("high")] = 1e6
    daily.iloc[0, daily.columns.get_loc("low")] = 1e6

    cleaned = clean_daily_ohlcv(daily, max_abs_r_close=0.75)

    assert daily.index[0] in cleaned.index


def test_clean_empty_in_empty_out():
    empty = pd.DataFrame(
        {"high": [], "low": [], "close": [], "volume": []},
        index=pd.DatetimeIndex([]),
    )
    assert clean_daily_ohlcv(empty).empty


def test_clean_is_deterministic():
    daily = _daily(n_days=20)
    daily.iloc[10, daily.columns.get_loc("volume")] = 0.0
    assert clean_daily_ohlcv(daily).equals(clean_daily_ohlcv(daily))


def test_clean_is_lookahead_safe():
    # A spike at row i must never drop row i-1 (no forward-looking decision).
    daily = _daily(n_days=8)
    daily.iloc[5, daily.columns.get_loc("close")] = daily["close"].iloc[4] * 5.0

    cleaned = clean_daily_ohlcv(daily, max_abs_r_close=0.75)

    assert daily.index[4] in cleaned.index


# --------------------------------------------------------------------------- #
# compute_symbol_quality
# --------------------------------------------------------------------------- #


def test_clean_business_day_frame_has_no_missing_days():
    q = compute_symbol_quality(_raw(_daily(n_days=120)), thresholds=LENIENT)
    assert q is not None
    assert q.missing_day_fraction == 0.0


def test_penny_stock_fails_price_sanity():
    thresholds = QualityThresholds(
        min_median_dollar_volume=0.0,
        min_trading_days=1,
        max_missing_day_fraction=1.0,
        min_median_close=5.0,
        max_return_spikes=10_000,
        max_flat_run=10_000,
        max_zero_volume_fraction=1.0,
    )
    q = compute_symbol_quality(_raw(_daily(base_price=2.0)), thresholds=thresholds)
    assert q is not None
    assert q.fail_reasons == ("penny_stock",)


def test_low_dollar_volume_fails_liquidity():
    thresholds = QualityThresholds(
        min_median_dollar_volume=1_000_000.0,
        min_trading_days=1,
        max_missing_day_fraction=1.0,
        min_median_close=0.0,
        max_return_spikes=10_000,
        max_flat_run=10_000,
        max_zero_volume_fraction=1.0,
    )
    q = compute_symbol_quality(_raw(_daily(base_price=100.0, volume=10.0)), thresholds=thresholds)
    assert q is not None
    assert q.fail_reasons == ("liquidity",)


def test_short_history_fails_history_length():
    thresholds = QualityThresholds(
        min_median_dollar_volume=0.0,
        min_trading_days=252,
        max_missing_day_fraction=1.0,
        min_median_close=0.0,
        max_return_spikes=10_000,
        max_flat_run=10_000,
        max_zero_volume_fraction=1.0,
    )
    q = compute_symbol_quality(_raw(_daily(n_days=30)), thresholds=thresholds)
    assert q is not None
    assert "history_length" in q.fail_reasons


def test_flatline_fails_staleness():
    thresholds = QualityThresholds(
        min_median_dollar_volume=0.0,
        min_trading_days=1,
        max_missing_day_fraction=1.0,
        min_median_close=0.0,
        max_return_spikes=10_000,
        max_flat_run=10,
        max_zero_volume_fraction=1.0,
    )
    daily = _daily(n_days=60)
    daily.iloc[5:25, daily.columns.get_loc("close")] = daily["close"].iloc[5]
    q = compute_symbol_quality(_raw(daily), thresholds=thresholds)
    assert q is not None
    assert q.max_flat_run > 10
    assert q.fail_reasons == ("flatline",)


def test_zero_volume_fraction_fails_staleness():
    thresholds = QualityThresholds(
        min_median_dollar_volume=0.0,
        min_trading_days=1,
        max_missing_day_fraction=1.0,
        min_median_close=0.0,
        max_return_spikes=10_000,
        max_flat_run=10_000,
        max_zero_volume_fraction=0.05,
    )
    daily = _daily(n_days=100)
    daily.iloc[:20, daily.columns.get_loc("volume")] = 0.0
    q = compute_symbol_quality(_raw(daily), thresholds=thresholds)
    assert q is not None
    assert q.zero_volume_fraction == 0.2
    assert "zero_volume" in q.fail_reasons


def test_return_spikes_counted_pre_clean_and_fail():
    thresholds = QualityThresholds(
        min_median_dollar_volume=0.0,
        min_trading_days=1,
        max_missing_day_fraction=1.0,
        min_median_close=0.0,
        max_return_spikes=0,
        max_abs_r_close=0.75,
        max_flat_run=10_000,
        max_zero_volume_fraction=1.0,
    )
    daily = _daily(n_days=40)
    daily.iloc[20, daily.columns.get_loc("close")] = daily["close"].iloc[19] * 5.0
    q = compute_symbol_quality(_raw(daily), thresholds=thresholds)
    assert q is not None
    assert q.n_return_spikes >= 1
    assert "return_spikes" in q.fail_reasons


def test_clean_symbol_passes_all_gates():
    q = compute_symbol_quality(_raw(_daily(n_days=400)), thresholds=QualityThresholds())
    assert q is not None
    assert q.passed
    assert q.fail_reasons == ()


def test_empty_symbol_returns_none():
    empty = pd.DataFrame(
        {"utc_time": pd.to_datetime([]), "high": [], "low": [], "close": [], "volume": []}
    )
    assert compute_symbol_quality(empty, thresholds=LENIENT) is None


# --------------------------------------------------------------------------- #
# curate_symbols
# --------------------------------------------------------------------------- #


def test_curate_symbols_passes_under_lenient_thresholds(parquet_dir):
    base_path, _ = parquet_dir
    passing, stats = curate_symbols(base_path, thresholds=LENIENT)

    assert passing == ["AAA", "BBB", "CCC"]
    assert set(stats) == {"AAA", "BBB", "CCC"}
    for record in stats.values():
        assert "median_dollar_volume" in record
        assert "fail_reasons" in record


def test_curate_symbols_history_gate_drops_short(parquet_dir):
    base_path, _ = parquet_dir
    thresholds = QualityThresholds(
        min_median_dollar_volume=0.0,
        min_trading_days=10,
        max_missing_day_fraction=1.0,
        min_median_close=0.0,
        max_return_spikes=10_000,
        max_flat_run=10_000,
        max_zero_volume_fraction=1.0,
    )
    passing, stats = curate_symbols(base_path, thresholds=thresholds)

    assert "BBB" not in passing  # ~5-day span
    assert "history_length" in stats["BBB"]["fail_reasons"]


def test_curate_symbols_intersects_symbol_list(parquet_dir):
    base_path, _ = parquet_dir
    passing, stats = curate_symbols(base_path, thresholds=LENIENT, symbols=["AAA"])

    assert passing == ["AAA"]
    assert set(stats) == {"AAA"}


def test_curate_symbols_records_load_error_not_fatal(tmp_path):
    good = tmp_path / "symbol=AAA"
    good.mkdir()
    _raw(_daily(n_days=300)).to_parquet(good / "data.parquet")

    bad = tmp_path / "symbol=BAD"
    bad.mkdir()
    (bad / "data.parquet").write_bytes(b"not a parquet file")

    passing, stats = curate_symbols(str(tmp_path), thresholds=QualityThresholds())

    assert "AAA" in passing
    assert stats["BAD"]["fail_reasons"] == ["load_error"]
    assert "error" in stats["BAD"]


# --------------------------------------------------------------------------- #
# register persistence helpers
# --------------------------------------------------------------------------- #


def test_quality_symbols_roundtrip_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(register.layout, "DATA_DIR", str(tmp_path))

    assert register.fetch_quality_symbols_list() == []

    register.set_quality_symbols(["CCC", "AAA", "AAA", "BBB"])
    assert register.fetch_quality_symbols_list() == ["AAA", "BBB", "CCC"]

    register.set_quality_symbols(["ZZZ"])  # overwrite, not union
    assert register.fetch_quality_symbols_list() == ["ZZZ"]

    register.clear_quality_symbols()
    assert register.fetch_quality_symbols_list() == []


# --------------------------------------------------------------------------- #
# curate command body
# --------------------------------------------------------------------------- #


def test_curate_command_writes_allowlist_and_stats(tmp_path, monkeypatch):
    # The command resolves base_path as <data_dir>/stocks, so build that layout.
    data_dir = tmp_path / "data"
    stocks = data_dir / "stocks"
    for sym in ("AAA", "BBB"):
        part = stocks / f"symbol={sym}"
        part.mkdir(parents=True)
        _raw(_daily(n_days=300)).to_parquet(part / "data.parquet")

    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(register.layout, "DATA_DIR", str(out))

    curation.curate(data_dir=str(data_dir), min_dollar_volume=0.0)

    assert register.fetch_quality_symbols_list() == ["AAA", "BBB"]
    with open(register.quality_stats_path()) as f:
        stats = json.load(f)
    assert set(stats) == {"AAA", "BBB"}
