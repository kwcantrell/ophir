"""Tests for the SQLite per-ticker stock store."""

import json
import sqlite3

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from ophir.sqlite_store import (
    build_sqlite_store,
    get_stock_tables,
    read_stock_table,
    sanitize_table_name,
)


def test_sanitize_basic():
    used: set[str] = set()
    assert sanitize_table_name("A", used) == "t_A"
    assert "t_A" in used


def test_sanitize_replaces_non_alphanumerics():
    used: set[str] = set()
    assert sanitize_table_name("A.WD", used) == "t_A_WD"
    assert sanitize_table_name("AAC.U", used) == "t_AAC_U"


def test_sanitize_resolves_collisions():
    used: set[str] = set()
    first = sanitize_table_name("A.WD", used)
    second = sanitize_table_name("A_WD", used)  # sanitizes to the same base
    assert first == "t_A_WD"
    assert second == "t_A_WD_2"
    assert {"t_A_WD", "t_A_WD_2"} <= used


def test_build_sqlite_store_creates_manifest_and_tables(parquet_dir, tmp_path):
    base_path, paths = parquet_dir
    db_path = str(tmp_path / "stocks.db")

    written = build_sqlite_store(base_path, db_path)

    assert written == len(paths)  # AAA, BBB, CCC

    conn = sqlite3.connect(db_path)
    try:
        manifest = dict(conn.execute("SELECT ticker, table_name FROM _tickers").fetchall())
        assert set(manifest) == set(paths)  # every symbol present

        # one table per ticker, row count matches the source parquet
        for sym, parquet_path in paths.items():
            table = manifest[sym]
            (n,) = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()
            assert n == len(pd.read_parquet(parquet_path))

        # dtypes JSON is stored, with utc_time recorded as a datetime dtype
        (dtypes_json,) = conn.execute(
            "SELECT dtypes FROM _tickers WHERE ticker = ?", ("AAA",)
        ).fetchone()
        dtypes = json.loads(dtypes_json)
        assert dtypes["utc_time"].startswith("datetime")
        # utc_time is physically stored as integer ns
        (kind,) = conn.execute(
            f'SELECT typeof(utc_time) FROM "{manifest["AAA"]}" LIMIT 1'
        ).fetchone()
        assert kind == "integer"
    finally:
        conn.close()


def test_build_sqlite_store_is_idempotent(parquet_dir, tmp_path):
    base_path, paths = parquet_dir
    db_path = str(tmp_path / "stocks.db")

    assert build_sqlite_store(base_path, db_path) == len(paths)
    # second run skips everything already present
    assert build_sqlite_store(base_path, db_path) == 0
    # overwrite rewrites every table
    assert build_sqlite_store(base_path, db_path, overwrite=True) == len(paths)


def test_get_stock_tables_maps_every_symbol(parquet_dir, tmp_path):
    base_path, paths = parquet_dir
    db_path = str(tmp_path / "stocks.db")
    build_sqlite_store(base_path, db_path)

    tables = get_stock_tables(db_path)
    assert set(tables) == set(paths)
    assert tables["AAA"] == "t_AAA"


def test_read_stock_table_round_trips_parquet(parquet_dir, tmp_path):
    base_path, paths = parquet_dir
    db_path = str(tmp_path / "stocks.db")
    build_sqlite_store(base_path, db_path)
    tables = get_stock_tables(db_path)

    for sym, parquet_path in paths.items():
        expected = pd.read_parquet(parquet_path)
        if "ticker" in expected.columns:
            expected = expected.drop(columns=["ticker"])
        actual = read_stock_table(db_path, tables[sym])
        assert_frame_equal(actual, expected)


def test_read_stock_table_round_trips_real_schema(tmp_path):
    """Round-trip the 8-column production schema through build/read."""
    rng = np.random.default_rng(42)
    n = 6

    volume = np.array(rng.integers(100_000, 10_000_000, size=n), dtype="int32")
    open_ = np.array(rng.uniform(100.0, 500.0, size=n), dtype="float64")
    close = np.array(rng.uniform(100.0, 500.0, size=n), dtype="float64")
    high = np.array(rng.uniform(100.0, 500.0, size=n), dtype="float64")
    low = np.array(rng.uniform(100.0, 500.0, size=n), dtype="float64")
    window_start = np.array(
        pd.date_range("2024-01-01", periods=n, freq="D").astype("int64"),
        dtype="int64",
    )
    transactions = np.array(rng.integers(1_000, 50_000, size=n), dtype="int32")
    utc_time = pd.date_range("2024-01-01", periods=n, freq="D")

    source = pd.DataFrame(
        {
            "volume": volume,
            "open": open_,
            "close": close,
            "high": high,
            "low": low,
            "window_start": window_start,
            "transactions": transactions,
            "utc_time": utc_time,
            "ticker": ["ZZZ"] * n,
        }
    )

    base = tmp_path / "parquets"
    base.mkdir()
    part = base / "ticker=ZZZ"
    part.mkdir()
    parquet_path = part / "data.parquet"
    source.to_parquet(parquet_path, index=False)

    # Read back to get the exact dtypes pyarrow will have stored/restored,
    # then drop ticker — this is the ground truth for assert_frame_equal.
    expected = pd.read_parquet(parquet_path).drop(columns=["ticker"])

    db_path = str(tmp_path / "stocks.db")
    build_sqlite_store(str(base), db_path)
    tables = get_stock_tables(db_path)
    actual = read_stock_table(db_path, tables["ZZZ"])

    assert_frame_equal(actual, expected)
