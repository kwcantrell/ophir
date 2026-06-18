# SQLite stock store — design

**Date:** 2026-06-18
**Status:** Approved (brainstorming)

## Goal

Migrate the per-ticker stock data at `src/ophir/.ophir/data/days/stocks/`
(currently ~34,701 Hive-partitioned parquet directories, 2.1 GB) into a single
SQLite database with **one table per ticker**, and add a SQLite-backed read path
to `StockHanlder` **behind a toggle** so the existing parquet path keeps working
unchanged. Default behavior stays parquet; SQLite is opt-in.

## Scope

- **In scope:** a converter, a single-file SQLite store, a `source` toggle on
  `StockHanlder`, a `migrate-sqlite` CLI command, and tests.
- **Out of scope (YAGNI):** sharding across multiple DB files, incremental/delta
  updates, per-table indexes (we read whole tables, never query within one), and
  any change to `ui.py`'s default source.

## Decisions (from brainstorming)

| Question | Decision |
| --- | --- |
| End goal | Build the converter **and** a SQLite read path, behind a toggle; parquet still works. |
| DB layout | **Single `stocks.db`** file, one table per ticker. |
| Table content | **Raw parquet rows verbatim** — aggregation/filters stay in `stock_df`. |
| Toggle API | **Explicit `source: Literal["parquet","sqlite"] = "parquet"`** on `StockHanlder`. |

**Recorded caveat:** 34k single-purpose tables is mildly anti-idiomatic for
SQLite (the "natural" schema is one `prices` table with a `ticker` column +
index). One-table-per-ticker was an explicit requirement, so that is what this
design builds. The known cost — SQLite parses the *entire* `sqlite_master`
schema on the first query of each connection — is a one-time, per-connection hit
amortized by long-lived DataLoader-worker processes. If it ever shows up in
profiling, sharding is a clean later optimization hidden behind the same
`ticker → table` interface.

## Architecture

A new module `src/ophir/sqlite_store.py` owns all SQLite-specific logic so
`ticker.py` only gains a thin branch. Three concerns:

1. **Converter** — `build_sqlite_store`: reads the parquet tree via the existing
   `get_stock_parquets`, writes one table per ticker into `stocks.db`, plus a
   `_tickers` manifest.
2. **Read helpers** — `get_stock_tables`, `read_stock_table`: SQLite analogs of
   `get_stock_parquets` + `pd.read_parquet`, returning frames with **identical
   dtypes** to the parquet path.
3. **Toggle** — `StockHanlder` gains `source`; `__post_init__` and `stock_df`
   branch on it. The aggregation/filter logic in `stock_df` is untouched.

### Why a manifest table

Ticker symbols contain characters awkward for SQL identifiers (`A.WD`, `AAC.U`,
`AAC.WS`). Instead of quoting identifiers everywhere, the converter sanitizes
each ticker to a safe table name, resolves collisions deterministically, and
records the true `ticker → table_name` mapping in `_tickers`. The read path never
guesses a table name — it looks it up. This also gives `get_stock_tables` an O(1)
source of truth rather than scraping `sqlite_master`.

## Schema

**Manifest** — also carries each table's column dtype map so the read path can
restore the exact pandas dtypes (see "Round-trip fidelity"):

```sql
CREATE TABLE _tickers (
    ticker     TEXT PRIMARY KEY,     -- true symbol, e.g. "A.WD"
    table_name TEXT NOT NULL UNIQUE, -- sanitized, e.g. "t_A_WD"
    dtypes     TEXT NOT NULL         -- JSON {column: pandas-dtype-str}, write order
);
```

**Per-ticker tables are created dynamically** by `pandas.DataFrame.to_sql`, so
the converter is **column-agnostic** — it stores whatever columns the parquet
holds. This matters because the real data has 8 columns
(`volume, open, close, high, low, window_start, transactions, utc_time`) while
the test fixture has only 5 (`utc_time, high, low, close, volume`), and column
dtypes differ between them (real `volume` is `int32`; the fixture's is
`float64`). Hard-coding a fixed `CREATE TABLE` or fixed per-column dtypes would
break one or the other. The only column-specific rule is dropping `ticker` (the
parquet dictionary column) when present — it is redundant (the table *is* the
ticker) and unused by `stock_df`.

Real data lands as, e.g.:

```sql
CREATE TABLE "t_A" (
    volume        INTEGER,
    open          REAL,
    close         REAL,
    high          REAL,
    low           REAL,
    window_start  INTEGER,   -- epoch nanoseconds, as-is
    transactions  INTEGER,
    utc_time      INTEGER     -- epoch nanoseconds (lossless; see below)
);
```

**Sanitization:** `table_name = "t_" + re.sub(r"[^0-9A-Za-z]", "_", ticker)`; on
collision append `_2`, `_3`, …. The `t_` prefix guarantees a valid identifier
even for all-numeric or reserved-word symbols.

## Round-trip fidelity

SQLite has only INTEGER/REAL/TEXT affinity, so a naive `to_sql` round-trip
loses pandas dtype detail (`int32` → `int64`) and mangles datetimes. The store
therefore records each column's original pandas dtype string in the `_tickers`
`dtypes` JSON at write time, in column order, and the read path replays it.

- **On write:** capture `dtypes = {c: str(frame[c].dtype) for c in frame.columns}`
  (after dropping `ticker`). Any datetime column (dtype starting with
  `"datetime"`) is converted to `int64` epoch-ns via `.astype("int64")` before
  `to_sql`, so it stores losslessly as INTEGER.
- **On read:** `read_stock_table(db_path, table_name)` looks up the table's
  `dtypes` JSON from `_tickers`, reads the table, and for each column: a
  `"datetime"`-prefixed dtype is restored with `pd.to_datetime(col, unit="ns")`;
  every other column is restored with `.astype(stored_dtype)`. Columns are
  returned in the recorded write order.

The fidelity-critical column is **`utc_time`** — `stock_df` relies on it being
`datetime64[ns]` (it does `.max() - .min()` and `.dt.normalize()`), and the
datetime branch above guarantees that. The whole contract is pinned by a test
asserting `StockHanlder(source="sqlite").stock_df(t)` is
`assert_frame_equal`-identical to `source="parquet"` on the same fixture, plus a
raw `read_stock_table` vs `pd.read_parquet` equality check.

## Converter

`build_sqlite_store(parquet_base, db_path, *, overwrite=False) -> int`:

- Opens `db_path`, creates `_tickers` if absent.
- Iterates `get_stock_parquets(parquet_base)` (reuses existing discovery so
  behavior matches the parquet path exactly), wrapped in `tqdm`.
- For each ticker: read parquet → drop `ticker` column if present → capture the
  column dtype map → convert datetime columns to int64 ns →
  `to_sql(table_name, ...)`; insert `(ticker, table_name, dtypes_json)` into
  `_tickers`.
- **Idempotent/resumable:** a ticker already in `_tickers` is skipped unless
  `overwrite=True`, so an interrupted 34k-table run can resume. Returns count
  written this call.
- `PRAGMA journal_mode=WAL` and `synchronous=NORMAL` for bulk-write speed;
  per-ticker inserts wrapped in a transaction.

## CLI

In `cli.py`, alongside existing commands:

```
ophir migrate-sqlite [--src <parquet base>] [--dst <stocks.db>] [--overwrite]
```

Defaults: `--src` = `DATA_DIR/days/stocks`, `--dst` = `DATA_DIR/days/stocks.db`
(sibling of the parquet tree). Prints `N tickers written`.

## Read path in `ticker.py`

- `StockHanlder` gains `source: Literal["parquet", "sqlite"] = "parquet"`.
- `__post_init__` branches:
  - `source="parquet"` → `self.stock_dict = get_stock_parquets(base_path)` (unchanged).
  - `source="sqlite"` → `self.stock_dict = get_stock_tables(base_path)` where
    `base_path` is the `.db` path; values are table names, not file paths.
- `stock_df` branches only at the load line:
  - parquet → `pd.read_parquet(path)` (unchanged).
  - sqlite → `read_stock_table(self.base_path, self.stock_dict[stock])`.
  - everything after (volume filter, history filter, groupby-date agg, year
    filters) is shared and untouched.
- `ui.py` keeps `source="parquet"` (its default) — no behavior change unless
  later flipped.
- **Connection lifecycle:** the SQLite connection is opened lazily on first use
  and reused, **not** in `__post_init__` — SQLite connections are not fork-safe,
  so each DataLoader worker process must open its own.

## Testing

All tests CPU-safe, network-free, deterministic, reusing the existing
`parquet_dir` fixture from `conftest.py`.

**`tests/test_sqlite_store.py`** (new):

- `build_sqlite_store` on a small fixture → `stocks.db` exists, `_tickers` maps
  every ticker, one table per ticker with correct row counts.
- **Round-trip equivalence (key test):** for each ticker, `read_stock_table(...)`
  is `assert_frame_equal`-identical (values + dtypes) to `pd.read_parquet(...)`
  after dropping the `ticker` column.
- Sanitization: a fixture ticker with a dot maps to a valid table name; a forced
  collision gets a `_2` suffix.
- Idempotence: second run without `overwrite` returns 0 new; `overwrite=True`
  rewrites.
- Edge: empty/missing parquet partition handled as the parquet path handles it.

**`tests/test_ticker.py`** (extend):

- **Source-equivalence:** `StockHanlder(base_path=db, source="sqlite").stock_df(t)`
  equals `StockHanlder(base_path=parquet_dir, source="parquet").stock_df(t)` for
  every ticker — proving the toggle is behavior-preserving through the
  *aggregated* frame, not just raw load.

**`tests/test_cli.py`** (extend): `ophir migrate-sqlite --help` smoke test +
registration check.

**Whole-suite gates:** `uv run pytest`, `uv run mypy src/ophir` (strict; sqlite3
is stdlib-typed), `uv run ruff check . && uv run ruff format --check .`.
