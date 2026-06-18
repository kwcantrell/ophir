---
name: gather-ohlc
description: >-
  Use when you need to fetch, refresh, read, or debug daily OHLC price bars for any
  ticker (e.g. AAPL, MSFT) or the SPY benchmark in ophir. Reuse `ingest()` /
  `ingest_many()` to pull split/dividend-adjusted daily bars from Yahoo Finance and
  write the model-ready parquet, and `load_daily_ohlcv()` / `load_history()` to read
  them back. Reach for it whenever OHLCV data is stale, missing, or needs extending
  to a new symbol or source.
---

# Gather daily OHLC

Fetch split/dividend-adjusted daily price bars from Yahoo Finance and persist them in
ophir's Hive parquet layout so the existing feature/model pipeline reads them unchanged.

See [docs/data-inputs.md](docs/data-inputs.md) for the full data-source catalog.

## Source
- **Provider:** Yahoo Finance via `yfinance` —
  `yf.Ticker(sym).history(start, end, interval="1d", auto_adjust=True)`.
  `auto_adjust=True` output is already split/dividend-adjusted, so ophir's separate
  split back-adjustment is intentionally skipped on this path.
- Reuse these (do not re-fetch by hand):
  - [agent/ingest.py:85](src/ophir/agent/ingest.py:85) — `ingest(symbol, days=730, *, stocks_dir=None)`: fetch + normalize + persist one ticker, returns the written `Path`.
  - [agent/ingest.py:119](src/ophir/agent/ingest.py:119) — `ingest_many(symbols, days=730, *, stocks_dir=None)`: batch ingest, returns `{symbol: Path}`; a bad ticker is skipped, not fatal.
  - [agent/ingest.py:30](src/ophir/agent/ingest.py:30) — `_fetch_yahoo`; [agent/ingest.py:50](src/ophir/agent/ingest.py:50) — `_normalize`; [agent/ingest.py:61](src/ophir/agent/ingest.py:61) — `_quality_warnings`; [agent/ingest.py:76](src/ophir/agent/ingest.py:76) — `_persist` (internal helpers).
  - [agent/feed.py:49](src/ophir/agent/feed.py:49) — `load_daily_ohlcv(symbol, *, stocks_dir=None)`: read the parquet as a datetime-indexed frame.
  - [agent/feed.py:77](src/ophir/agent/feed.py:77) — `load_history(symbol, *, as_of=None, stocks_dir=None)`: same, trimmed to the last closed NYSE session (drops today's forming bar); raises on a stale feed.
  - [agent/feed.py:44](src/ophir/agent/feed.py:44) — `parquet_path(symbol, *, override=None)`: resolve the on-disk path.
  - [agent/feed.py:163](src/ophir/agent/feed.py:163) — `forecast_window_tensors(...)`: build a forward-looking model input from the loaded history.

## Fields
- Written/read columns: **`high`, `low`, `close`, `volume`** (`_OHLCV_COLS`,
  [agent/ingest.py:25](src/ophir/agent/ingest.py:25)).
- Index: `utc_time` — tz-naive (any tz is stripped in `_normalize`), sorted ascending,
  de-duplicated (keeps last). `load_daily_ohlcv` returns it datetime-indexed.
- On disk: `<DATA_DIR>/days/stocks/symbol=<SYM>/data.parquet` (default root from
  `ophir.register.get_default_data_days_dir()`; override with `stocks_dir` / `override`).
- The **SPY benchmark uses the exact same mechanism** — `ingest("SPY")` then
  `load_daily_ohlcv("SPY")["close"]`.

## Fail-safe & caching
- Empty / invalid / delisted symbol → no Yahoo rows → **`ValueError`** in `_fetch_yahoo`;
  inside `ingest_many` that symbol is reported and **skipped**, not fatal to the batch.
- `load_daily_ohlcv` raises **`FileNotFoundError`** if the parquet does not exist (run
  `ophir ingest <SYM>` first).
- `load_history` raises **`ValueError`** when the feed is stale (latest bar older than the
  last closed session) or no bar exists on/before the cutoff — callers treat that as a
  per-ticker skip so the live forecast never runs on out-of-date data.
- Non-fatal **quality warnings** (printed, `_quality_warnings`): stale (`last bar > 5d` old)
  or sparse (`< ~455` requested days / `< 365` rows) data — the 365-day model window may
  not fully populate.
- **Caching:** ingest overwrites `data.parquet` each run (no incremental merge); re-run to
  refresh. No rate-limit handling beyond yfinance's own.

## How to use / extend
- Refresh one ticker: `ophir ingest AAPL --days 730` (CLI →
  [cli.py:57](src/ophir/cli.py:57)); or in code `from ophir.agent.ingest import ingest;
  ingest("AAPL")`. The CLI exposes a single symbol; for a batch call `ingest_many([...])`
  in code (`ophir backtest` auto-ingests its universe + `SPY` the same way).
- Read it back: `load_daily_ohlcv("AAPL")` for the raw frame, or
  `load_history("AAPL")` for the look-ahead-safe (T-1 close) frame.
- **Add a sibling source** the same way: write a `_fetch_<provider>` returning a raw frame,
  feed it through `_normalize` (→ tz-naive `high/low/close/volume`) and `_persist` (→ the
  same `symbol=<SYM>/data.parquet` layout). Once the parquet matches that schema,
  `load_daily_ohlcv` / `load_history` and the whole model pipeline consume it unchanged.
