"""Single-file SQLite store for per-ticker stock data.

Mirrors the parquet ingest path in :mod:`ophir.ticker`: one table per ticker
in a single database, with a ``_tickers`` manifest mapping each true symbol to
its sanitized table name and the column dtypes needed to restore frames
byte-identically to ``pandas.read_parquet``.
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing

import pandas as pd  # type: ignore[import-untyped]

from ophir.ticker import get_stock_parquets

_MANIFEST_DDL = (
    "CREATE TABLE IF NOT EXISTS _tickers ("
    "ticker TEXT PRIMARY KEY, "
    "table_name TEXT NOT NULL UNIQUE, "
    "dtypes TEXT NOT NULL)"
)


def sanitize_table_name(ticker: str, used: set[str]) -> str:
    """Return a unique, SQL-safe table name for ``ticker``.

    The name is ``t_`` followed by ``ticker`` with every non-alphanumeric
    character replaced by ``_``. If that name is already in ``used``, a
    numeric suffix (``_2``, ``_3``, …) is appended until it is unique. The
    chosen name is added to ``used``.

    Parameters
    ----------
    ticker : str
        The true ticker symbol (may contain ``.`` and other punctuation).
    used : set[str]
        Names already assigned; mutated in place with the returned name.

    Returns
    -------
    str
        A unique table name safe to use as a SQLite identifier.
    """
    base = "t_" + re.sub(r"[^0-9A-Za-z]", "_", ticker)
    name = base
    suffix = 2
    while name in used:
        name = f"{base}_{suffix}"
        suffix += 1
    used.add(name)
    return name


def _prepare_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Drop the redundant ``ticker`` column and encode datetimes as int64 ns.

    Returns the storage-ready frame and the original column dtype map (in
    column order) that :func:`read_stock_table` replays.
    """
    if "ticker" in df.columns:
        df = df.drop(columns=["ticker"])
    dtypes = {col: str(df[col].dtype) for col in df.columns}
    for col, dtype in dtypes.items():
        if dtype.startswith("datetime"):
            df[col] = df[col].astype("int64")
    return df, dtypes


def build_sqlite_store(parquet_base: str, db_path: str, *, overwrite: bool = False) -> int:
    """Convert a Hive-partitioned parquet tree into a SQLite store.

    Each ticker becomes its own table; the ``_tickers`` manifest records the
    ``ticker -> table_name`` mapping and the column dtypes needed to restore
    frames identically to ``pandas.read_parquet``. Tickers already present are
    skipped unless ``overwrite`` is set, so an interrupted run can resume.

    Parameters
    ----------
    parquet_base : str
        Directory of ``<key>=<symbol>`` partition dirs (as read by
        :func:`ophir.ticker.get_stock_parquets`).
    db_path : str
        Destination SQLite file; created if absent.
    overwrite : bool, optional
        If ``True``, rewrite tables for tickers already in the manifest.

    Returns
    -------
    int
        The number of ticker tables written during this call.
    """
    stock_dict = get_stock_parquets(parquet_base)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(_MANIFEST_DDL)

        existing = {row[0] for row in conn.execute("SELECT ticker FROM _tickers")}
        used = {row[0] for row in conn.execute("SELECT table_name FROM _tickers")}

        written = 0
        for ticker, parquet_path in stock_dict.items():
            if ticker in existing and not overwrite:
                continue

            frame, dtypes = _prepare_frame(pd.read_parquet(parquet_path))

            if ticker in existing:
                old = conn.execute(
                    "SELECT table_name FROM _tickers WHERE ticker = ?",
                    (ticker,),
                ).fetchone()[0]
                conn.execute(f'DROP TABLE IF EXISTS "{old}"')
                used.discard(old)
                table = sanitize_table_name(ticker, used)
                conn.execute(
                    "UPDATE _tickers SET table_name = ?, dtypes = ? WHERE ticker = ?",
                    (table, json.dumps(dtypes), ticker),
                )
            else:
                table = sanitize_table_name(ticker, used)
                conn.execute(
                    "INSERT INTO _tickers (ticker, table_name, dtypes) VALUES (?, ?, ?)",
                    (ticker, table, json.dumps(dtypes)),
                )

            frame.to_sql(table, conn, if_exists="replace", index=False)
            written += 1

        conn.commit()
    return written


def get_stock_tables(db_path: str) -> dict[str, str]:
    """Map each ticker symbol to its table name in the SQLite store.

    The SQLite analog of :func:`ophir.ticker.get_stock_parquets`.

    Parameters
    ----------
    db_path : str
        Path to the SQLite store.

    Returns
    -------
    dict[str, str]
        Mapping of true ticker symbol to its (sanitized) table name.
    """
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute("SELECT ticker, table_name FROM _tickers").fetchall()
    return dict(rows)


def read_stock_table(db_path: str, table_name: str) -> pd.DataFrame:
    """Read one ticker table, restoring the original parquet dtypes.

    Datetime columns (stored as int64 epoch-ns) are returned as
    ``datetime64[ns]``; every other column is cast back to the pandas dtype
    recorded at write time, in the original column order.

    Parameters
    ----------
    db_path : str
        Path to the SQLite store.
    table_name : str
        The sanitized table name (from :func:`get_stock_tables`).

    Returns
    -------
    pandas.DataFrame
        A frame identical to ``pandas.read_parquet`` of the source partition
        (minus the redundant ``ticker`` column).
    """
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT dtypes FROM _tickers WHERE table_name = ?", (table_name,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no manifest entry for table {table_name!r}")
        dtypes: dict[str, str] = json.loads(row[0])
        df = pd.read_sql(f'SELECT * FROM "{table_name}"', conn)

    for col, dtype in dtypes.items():
        if dtype.startswith("datetime"):
            df[col] = pd.to_datetime(df[col], unit="ns")
        else:
            df[col] = df[col].astype(dtype)
    return df[list(dtypes)]
