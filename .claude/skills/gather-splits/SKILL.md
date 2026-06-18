---
name: gather-splits
description: >-
  Use when you need raw stock-split history (e.g. a 4:1 AAPL split date/ratio) to
  back-adjust *unadjusted* OHLC prices, or when debugging/extending split handling in
  the ticker pipeline. Reuse `get_splits` + `StockSplit.apply_splits` from
  `src/ophir/ticker.py` — never re-scrape yfinance `.splits` by hand. NOTE: this is
  essentially unused in the live/training path (OHLC is fetched with auto_adjust=True,
  already split/dividend adjusted); reach for it only for raw-data adjustment.
---

# Gather stock splits

Fetch and cache Yahoo Finance stock-split history, then back-adjust raw OHLCV prices for
splits. For the full data-source catalog, see [docs/data-inputs.md](docs/data-inputs.md).

## Source
- **Yahoo Finance** via `yfinance`: `yf.Ticker(sym).splits` (a pandas Series of split
  ratios indexed by effective split date). No API key or endpoint URL.
- Reuse, do not rewrite:
  - [ticker.py:139](src/ophir/ticker.py:139) — `get_splits(tickers, cache_path=None) -> dict[str, StockSplit | None]`: pickle-cached batch fetch.
  - [ticker.py:205](src/ophir/ticker.py:205) — `@dataclass StockSplit(id, dates, ratios)`: one stock's split history.
  - [ticker.py:223](src/ophir/ticker.py:223) — `StockSplit.apply_splits(df) -> pd.DataFrame`: back-adjust an OHLCV frame.

## Fields
- `get_splits` returns `dict[str, StockSplit]` — only tickers that actually have splits.
  Tickers queried with no splits are dropped from the return (kept as a `None` sentinel in
  the cache only).
- `StockSplit` attributes:
  - `id: str` — the ticker symbol.
  - `dates: list[np.datetime64]` — effective split dates (tz-stripped to naive).
  - `ratios: list[float]` — split ratios aligned with `dates` (e.g. `4.0` for a 4:1 split).
- `apply_splits(df)` expects a datetime-indexed frame with a `close` column (and optional
  `volume`). It builds a cumulative pre-split adjustment factor, divides `close` by it, and
  multiplies `volume` by it (volume goes the opposite way). Returns the sorted, adjusted frame.

## Fail-safe & caching
- **Cache:** pickle at `<DATA_DIR>/yf_splits_cache.pkl` (override via `cache_path=`).
  `DATA_DIR` is resolved lazily from `ophir.register`. Only **missing** tickers hit the
  network; a `None` value means "queried, already known to have no splits" so it is not
  re-fetched. Cache-hit fast path: if every requested ticker is already cached, it returns
  immediately without any network call.
- **Failure handling:** each ticker is fetched in its own `try/except` — a per-ticker
  failure is printed (`[get_splits] {ticker} failed: ...`) and skipped, so one bad symbol
  never aborts the batch and the surviving results are still pickled to disk.
- **Staleness:** the cache never auto-expires. To force a refresh of a split that changed,
  delete `yf_splits_cache.pkl` (or that ticker's entry) and re-run.

## How to use / extend
- Fetch + cache, then adjust one raw frame:
  ```python
  from ophir.ticker import get_splits
  splits = get_splits(["AAPL", "TSLA", "NVDA"])      # network only for uncached symbols
  raw_df = ...                                         # datetime-indexed OHLCV, NOT auto-adjusted
  adj = splits["AAPL"].apply_splits(raw_df) if "AAPL" in splits else raw_df
  ```
- Or pass the whole map into the loader: `StockHanlder(..., stock_splits=get_splits(symbols))`
  ([ticker.py:618](src/ophir/ticker.py:618) wires each `StockSplit` into its `StockStreamer`).
- **Caveat first:** before adjusting, confirm your OHLC is actually raw. The live/training
  pipeline pulls OHLC with `auto_adjust=True` (already split/dividend adjusted), so applying
  `apply_splits` on top double-counts. Only adjust frames you know are unadjusted.
- **Add a sibling corporate-action source** (e.g. dividends) the same way: add a
  `get_dividends(tickers, cache_path=None)` that reads `yf.Ticker(sym).dividends`, reuse the
  same pickle-cache + `None`-sentinel + per-ticker `try/except` pattern from `get_splits`,
  and pair it with a small dataclass exposing an `apply_*(df)` adjustment method mirroring
  `StockSplit`.
