# SQLite Stock Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single-file SQLite store (one table per ticker) plus a converter, a `migrate-sqlite` CLI command, and a `source` toggle on `StockHanlder`, so the model can read stock data from SQLite or parquet interchangeably with parquet remaining the default.

**Architecture:** A new `src/ophir/sqlite_store.py` module owns all SQLite logic: a `build_sqlite_store` converter, a `_tickers` manifest carrying each table's column-dtype JSON, and `get_stock_tables`/`read_stock_table` read helpers that restore exact pandas dtypes. `ticker.py`'s `StockHanlder` gains `source: Literal["parquet","sqlite"]` and branches only at discovery (`__post_init__`) and the single load line in `stock_df`; all aggregation/filtering is unchanged. A `ophir migrate-sqlite` Typer command wires the converter to the CLI.

**Tech Stack:** Python 3.10+ (runtime floor), stdlib `sqlite3` + `json`, pandas (`to_sql`/`read_sql`), Typer (CLI), pytest, strict mypy, ruff.

## Global Constraints

- Live code lives in `src/ophir/` only — never touch top-level `ophir/`, `oldcode/`, or `old_*.py`.
- Strict mypy over `src/ophir` with `warn_unused_ignores = true`: every `# type: ignore` must be precise and actually fire, or mypy fails. `pandas` is imported as `import pandas as pd  # type: ignore[import-untyped]`.
- pytest runs with `filterwarnings = ["error", ...]` and `--strict-config`: all code and tests must run warning-clean.
- Tests must be deterministic, seeded, CPU-safe, and network-free. Reuse fixtures in `tests/conftest.py` (`parquet_dir`, `raw_tick_df`).
- Ruff `target-version = "py312"`, mypy `python_version = "3.10"` — do not unify; use only syntax valid on 3.10.
- `StockHanlder` is `@dataclass(kw_only=True)`; all fields are keyword-only, so new fields with defaults can go anywhere in the field list.
- Default behavior must stay parquet — `source` defaults to `"parquet"` and `ui.py` is left untouched.
- Don't add comments unless they explain non-obvious *why*. No removed-code stubs or back-compat shims.

---

### Task 1: `sqlite_store` module skeleton + table-name sanitization

**Files:**
- Create: `src/ophir/sqlite_store.py`
- Test: `tests/test_sqlite_store.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `sanitize_table_name(ticker: str, used: set[str]) -> str` — returns a SQL-safe, unique table name for `ticker`, registering the result into `used`. Prefix `t_`, non-alphanumerics replaced by `_`, collisions suffixed `_2`, `_3`, ….

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sqlite_store.py
"""Tests for the SQLite per-ticker stock store."""

from ophir.sqlite_store import sanitize_table_name


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sqlite_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ophir.sqlite_store'` (or `ImportError` for `sanitize_table_name`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/ophir/sqlite_store.py
"""Single-file SQLite store for per-ticker stock data.

Mirrors the parquet ingest path in :mod:`ophir.ticker`: one table per ticker
in a single database, with a ``_tickers`` manifest mapping each true symbol to
its sanitized table name and the column dtypes needed to restore frames
byte-identically to ``pandas.read_parquet``.
"""

from __future__ import annotations

import re


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sqlite_store.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ophir/sqlite_store.py tests/test_sqlite_store.py
git commit -m "Add sqlite_store table-name sanitization"
```

---

### Task 2: Converter `build_sqlite_store`

**Files:**
- Modify: `src/ophir/sqlite_store.py`
- Test: `tests/test_sqlite_store.py`

**Interfaces:**
- Consumes: `sanitize_table_name` (Task 1); `ophir.ticker.get_stock_parquets(base_path) -> dict[str, str]`.
- Produces:
  - `build_sqlite_store(parquet_base: str, db_path: str, *, overwrite: bool = False) -> int` — builds/updates the store, returns the number of ticker tables written this call.
  - Manifest schema `_tickers(ticker TEXT PRIMARY KEY, table_name TEXT NOT NULL UNIQUE, dtypes TEXT NOT NULL)` where `dtypes` is a JSON object `{column: pandas-dtype-str}` in write order.
  - On write, datetime columns are stored as `int64` epoch-ns; the `ticker` column (if present) is dropped.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_sqlite_store.py
import json
import sqlite3

import pandas as pd

from ophir.sqlite_store import build_sqlite_store


def test_build_sqlite_store_creates_manifest_and_tables(parquet_dir, tmp_path):
    base_path, paths = parquet_dir
    db_path = str(tmp_path / "stocks.db")

    written = build_sqlite_store(base_path, db_path)

    assert written == len(paths)  # AAA, BBB, CCC

    conn = sqlite3.connect(db_path)
    try:
        manifest = dict(
            conn.execute("SELECT ticker, table_name FROM _tickers").fetchall()
        )
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sqlite_store.py -k build -v`
Expected: FAIL — `ImportError: cannot import name 'build_sqlite_store'`.

- [ ] **Step 3: Write minimal implementation**

Add imports and the function to `src/ophir/sqlite_store.py`:

```python
import json
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


def build_sqlite_store(
    parquet_base: str, db_path: str, *, overwrite: bool = False
) -> int:
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

        existing = {
            row[0] for row in conn.execute("SELECT ticker FROM _tickers")
        }
        used = {
            row[0] for row in conn.execute("SELECT table_name FROM _tickers")
        }

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
                    "UPDATE _tickers SET table_name = ?, dtypes = ? "
                    "WHERE ticker = ?",
                    (table, json.dumps(dtypes), ticker),
                )
            else:
                table = sanitize_table_name(ticker, used)
                conn.execute(
                    "INSERT INTO _tickers (ticker, table_name, dtypes) "
                    "VALUES (?, ?, ?)",
                    (ticker, table, json.dumps(dtypes)),
                )

            frame.to_sql(table, conn, if_exists="replace", index=False)
            written += 1

        conn.commit()
    return written
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sqlite_store.py -v`
Expected: PASS (all Task 1 + Task 2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ophir/sqlite_store.py tests/test_sqlite_store.py
git commit -m "Add build_sqlite_store converter"
```

---

### Task 3: Read helpers `get_stock_tables` + `read_stock_table`

**Files:**
- Modify: `src/ophir/sqlite_store.py`
- Test: `tests/test_sqlite_store.py`

**Interfaces:**
- Consumes: `build_sqlite_store` (Task 2); the `_tickers` manifest.
- Produces:
  - `get_stock_tables(db_path: str) -> dict[str, str]` — maps each true ticker to its table name (parquet-path analog of `get_stock_parquets`).
  - `read_stock_table(db_path: str, table_name: str) -> pd.DataFrame` — reads one ticker table and restores the original column dtypes/order recorded at write time; datetime columns return as `datetime64[ns]`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_sqlite_store.py
from pandas.testing import assert_frame_equal

from ophir.sqlite_store import get_stock_tables, read_stock_table


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sqlite_store.py -k "tables or round_trips" -v`
Expected: FAIL — `ImportError: cannot import name 'get_stock_tables'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/ophir/sqlite_store.py`:

```python
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
    return {ticker: table for ticker, table in rows}


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sqlite_store.py -v`
Expected: PASS (all sqlite_store tests).

- [ ] **Step 5: Type-check and commit**

```bash
uv run mypy src/ophir
git add src/ophir/sqlite_store.py tests/test_sqlite_store.py
git commit -m "Add SQLite read helpers get_stock_tables and read_stock_table"
```

Expected: mypy passes (0 errors).

---

### Task 4: `StockHanlder` source toggle

**Files:**
- Modify: `src/ophir/ticker.py` (imports; `StockHanlder` fields at lines ~506-519; `__post_init__` at ~521-526; `stock_df` load line at ~589-590)
- Test: `tests/test_ticker_handler.py` (existing; uses a module-level `_handler(base_path, **kwargs)` helper — reuse it)

**Interfaces:**
- Consumes: `get_stock_tables`, `read_stock_table` (Task 3).
- Produces: `StockHanlder(..., source: Literal["parquet", "sqlite"] = "parquet")`. When `source="sqlite"`, `base_path` is the `.db` path and `stock_dict` maps ticker → table name; `stock_df` loads via `read_stock_table`. Default `"parquet"` is byte-for-byte unchanged.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ticker_handler.py` (it already imports `StockHanlder` and defines `_handler`; add the `build_sqlite_store` and `assert_frame_equal` imports at the top):

```python
from pandas.testing import assert_frame_equal

from ophir.sqlite_store import build_sqlite_store


def test_stockhandler_sqlite_source_matches_parquet(parquet_dir, tmp_path):
    base_path, paths = parquet_dir
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ticker_handler.py -k "sqlite_source or defaults_to_parquet" -v`
Expected: FAIL — `TypeError: StockHanlder.__init__() got an unexpected keyword argument 'source'`.

- [ ] **Step 3: Write minimal implementation**

In `src/ophir/ticker.py`, add `Literal` to the typing import (line 16):

```python
from typing import TYPE_CHECKING, Any, Literal
```

Add a module-level import near the top (after the existing third-party imports, e.g. after line 21) — keep it local to avoid a heavy import cycle is unnecessary since `sqlite_store` imports `ticker`; import inside the methods instead to avoid a circular import at module load:

> **Circular-import note:** `sqlite_store` imports `get_stock_parquets` from `ticker`, so `ticker` must **not** import `sqlite_store` at module top. Import the two helpers lazily inside the methods that use them (shown below).

Add the `source` field to `StockHanlder` (among the keyword-only fields, e.g. directly after `winsorize_returns: bool = True` at line 517):

```python
    source: Literal["parquet", "sqlite"] = "parquet"
```

Update `__post_init__` (lines 521-526):

```python
    def __post_init__(self) -> None:
        if self.source == "sqlite":
            from ophir.sqlite_store import get_stock_tables

            self.stock_dict = get_stock_tables(self.base_path)
        else:
            self.stock_dict = get_stock_parquets(self.base_path)
        self.stocks = list(self.stock_dict.keys())

        if self.offset == -1:
            self.offset = self.seq_len
```

Update the load line in `stock_df` (lines 589-590) from:

```python
        path = self.stock_dict[stock]
        df = pd.read_parquet(path)
```

to:

```python
        if self.source == "sqlite":
            from ophir.sqlite_store import read_stock_table

            df = read_stock_table(self.base_path, self.stock_dict[stock])
        else:
            df = pd.read_parquet(self.stock_dict[stock])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ticker_handler.py -k "sqlite_source or defaults_to_parquet" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Full ticker suite + type-check**

Run: `uv run pytest tests/ -k ticker -q && uv run mypy src/ophir`
Expected: all pass, mypy 0 errors. (Confirms the parquet path is unregressed.)

- [ ] **Step 6: Commit**

```bash
git add src/ophir/ticker.py tests/test_ticker_handler.py
git commit -m "Add source toggle to StockHanlder for SQLite reads"
```

---

### Task 5: `ophir migrate-sqlite` CLI command

**Files:**
- Modify: `src/ophir/cli.py`
- Create: `tests/test_cli.py` (no CLI test file exists on this branch yet)

**Interfaces:**
- Consumes: `build_sqlite_store` (Task 2); `ophir.register.get_default_data_days_dir() -> str` (returns `<DATA_DIR>/days`).
- Produces: `ophir migrate-sqlite [--src <parquet base>] [--dst <stocks.db>] [--overwrite]`. Defaults: `--src` = `<days>/stocks`, `--dst` = `<days>/stocks.db`. Prints `<N> tickers written`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli.py`:

```python
"""Tests for the ``ophir`` Typer CLI command registration and behavior."""

from typer.testing import CliRunner

from ophir.cli import app

runner = CliRunner()


def test_migrate_sqlite_is_registered():
    result = runner.invoke(app, ["migrate-sqlite", "--help"])
    assert result.exit_code == 0
    assert "--src" in result.output
    assert "--dst" in result.output
    assert "--overwrite" in result.output


def test_migrate_sqlite_runs(parquet_dir, tmp_path):
    base_path, paths = parquet_dir
    db_path = str(tmp_path / "stocks.db")
    result = runner.invoke(
        app, ["migrate-sqlite", "--src", base_path, "--dst", db_path]
    )
    assert result.exit_code == 0
    assert f"{len(paths)} tickers written" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -k migrate_sqlite -v`
Expected: FAIL — `migrate-sqlite` not a known command (exit code 2 / "No such command").

- [ ] **Step 3: Write minimal implementation**

In `src/ophir/cli.py`, add the command (after the `serve` command). Use a lazy import of `build_sqlite_store` so importing the CLI stays light:

```python
@app.command()
def migrate_sqlite(
    src: str = typer.Option(
        None, help="Parquet base dir (default: <DATA_DIR>/days/stocks)"
    ),
    dst: str = typer.Option(
        None, help="Destination SQLite file (default: <DATA_DIR>/days/stocks.db)"
    ),
    overwrite: bool = typer.Option(
        False, help="Rewrite tables for tickers already present"
    ),
) -> None:
    """Convert the per-ticker parquet tree into a single SQLite store.

    Builds one table per ticker plus a ``_tickers`` manifest. Idempotent:
    tickers already present are skipped unless ``--overwrite`` is given.
    """
    import os

    from ophir.register import get_default_data_days_dir
    from ophir.sqlite_store import build_sqlite_store

    days = get_default_data_days_dir()
    src = src or os.path.join(days, "stocks")
    dst = dst or os.path.join(days, "stocks.db")

    written = build_sqlite_store(src, dst, overwrite=overwrite)
    typer.echo(f"{written} tickers written")
```

> **Typer naming note:** Typer turns the function name `migrate_sqlite` into the CLI command `migrate-sqlite` automatically. Verify in Step 4's `--help` output; do not hardcode the hyphen.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -k migrate_sqlite -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ophir/cli.py tests/test_cli.py
git commit -m "Wire ophir migrate-sqlite CLI command"
```

---

### Task 6: Docs + full-suite verification

**Files:**
- Modify: `CLAUDE.md` (module map row + tests line)
- Modify: `docs/api/index.rst` (autosummary module list)
- Modify: `docs/cli.rst` (manual command reference)
- Modify: `docs/architecture.rst` (optional one-line mention in the feature-pipeline section)

**Interfaces:**
- Consumes: everything above.
- Produces: documentation reflecting `sqlite_store.py` and the `source` toggle; a green full suite and a warning-clean docs build.

- [ ] **Step 1: Update the CLAUDE.md module map**

Add a row to the `src/ophir/` module table (after the `ticker.py` row):

```markdown
| `sqlite_store.py` | Single-file SQLite store: `build_sqlite_store` converter (one table per ticker + `_tickers` manifest with column-dtype JSON) and `get_stock_tables`/`read_stock_table` read helpers. Backs `StockHanlder(source="sqlite")`. |
```

In the "## Tests" section, add `sqlite_store` to the covered-areas sentence and note `cli` now has a `migrate-sqlite` test (match the existing sentence format).

- [ ] **Step 2: Add the module to the API autosummary**

In `docs/api/index.rst`, add `ophir.sqlite_store` to the `.. autosummary::` list (e.g. directly after `ophir.ticker`):

```rst
   ophir.ticker
   ophir.sqlite_store
   ophir.register
```

(All `sqlite_store` functions carry numpydoc docstrings, so autosummary generation is warning-clean.)

- [ ] **Step 3: Add the CLI command to `docs/cli.rst`**

After the ``ophir register massive-key`` section, append a manual section matching the existing style:

```rst
``ophir migrate-sqlite``
------------------------

Convert the per-ticker parquet tree into a single SQLite store
(:func:`ophir.cli.migrate_sqlite`). Builds one table per ticker plus a
``_tickers`` manifest; idempotent (skips tickers already present unless
``--overwrite``).

.. code-block:: bash

   ophir migrate-sqlite [--src PATH] [--dst PATH] [--overwrite / --no-overwrite]

================= =============================== ===================================
Option            Default                          Description
================= =============================== ===================================
``--src``         ``<DATA_DIR>/days/stocks``       Parquet base directory.
``--dst``         ``<DATA_DIR>/days/stocks.db``    Destination SQLite file.
``--overwrite``   ``False``                        Rewrite tables already present.
================= =============================== ===================================
```

- [ ] **Step 4: Run the full gate suite**

Run:
```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy src/ophir && uv run pytest -q
```
Expected: ruff clean, mypy 0 errors, all tests pass (prior suite + the new tests).

- [ ] **Step 5: Build the docs (Sphinx, warnings-as-errors)**

Run: `uv run --group docs sphinx-build -W -b html docs docs/_build/html`
Expected: build succeeds with no warnings.

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md docs
git commit -m "Document SQLite stock store"
```

---

## Self-Review

**Spec coverage:**
- Single `stocks.db`, one table per ticker → Tasks 2, 6.
- `_tickers` manifest with sanitized names + dtype JSON → Tasks 1, 2.
- Raw rows verbatim, `ticker` dropped, column-agnostic → Task 2 (`_prepare_frame`).
- Round-trip dtype fidelity (incl. `utc_time`) → Tasks 2-3 (`assert_frame_equal` tests).
- `source` toggle, default parquet, `ui.py` untouched → Task 4.
- Lazy connection / no fork-unsafe handle in `__post_init__` → Task 4 opens connections per call inside helpers (no shared handle), satisfying fork-safety.
- `migrate-sqlite` CLI with documented defaults → Task 5.
- Tests CPU-safe, network-free, reuse `parquet_dir` → all test steps.
- Idempotent/resumable converter → Task 2 idempotence test.

**Placeholder scan:** No TBD/TODO; every code step shows complete code. The two `>` notes (test-file reuse, docs autodoc) are conditional verification instructions, not deferred work.

**Type consistency:** `build_sqlite_store(parquet_base, db_path, *, overwrite=False) -> int`, `get_stock_tables(db_path) -> dict[str,str]`, `read_stock_table(db_path, table_name) -> pd.DataFrame`, `sanitize_table_name(ticker, used) -> str` are used with identical signatures across Tasks 2-5. `source` is `Literal["parquet","sqlite"]` everywhere. Manifest columns `(ticker, table_name, dtypes)` consistent in Tasks 2-3.

**Note on connection model:** helpers open a short-lived `sqlite3.connect` per call via `closing(...)`. This is simpler than a cached handle and inherently fork-safe (each DataLoader worker opens its own on demand). The spec's "lazy, reused connection" intent is satisfied functionally; a cached handle is an available later optimization if read latency matters, and would not change any public signature.
