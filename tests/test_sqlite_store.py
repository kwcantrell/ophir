"""Tests for the SQLite per-ticker stock store."""

import json
import sqlite3

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
