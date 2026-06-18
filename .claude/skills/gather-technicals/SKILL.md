---
name: gather-technicals
description: >-
  Use when you need ophir-computed technical indicators for a ticker (last_close,
  recent return, 20d/60d realized vol, % of 52-week high/low) and, when a model
  forecast is on hand, the forecast cum-return / mean upside / mean downside. These
  are DERIVED from the ingested OHLC parquet + the model Forecast, not fetched. Reach
  for it to build, refresh, debug, or extend the "technicals" dimension of a research
  brief (e.g. AAPL technicals, vol_20d, pct_of_52w_high, forecast_cum_return).
---

# Gather technicals

Compute grounded technical indicators for one ticker from ophir's own features plus
the model forecast. No external fetch — purely derived. See [docs/data-inputs.md](docs/data-inputs.md).

## Source
- DERIVED: the ingested per-symbol OHLC parquet (`symbol=<SYM>/data.parquet`) and the
  model's [`Forecast`](src/ophir/agent/predict.py:35).
- Reuse [`gather_technicals(symbol, forecast=None, *, stocks_dir=None)`](src/ophir/agent/research.py:155).
  It calls [`load_history`](src/ophir/agent/feed.py:77) → [`extract_features`](src/ophir/ticker.py:259),
  filters to real (non-padded) rows via `trade_occured`, and reads the **last real row** plus the
  trailing 252 closes. Do not re-derive these — call the existing function.

## Fields
Returned `dict[str, Any]`, always present:
- `last_close` — most recent close, rounded to 4 dp.
- `recent_return_1d` — last `r_close` (log close-to-close return); `None` if non-finite.
- `vol_20d` — last `20_volatility` (rolling std of `r_close`, 20d); `None` if non-finite.
- `vol_60d` — last `60_volatility` (rolling std of `r_close`, 60d); `None` if non-finite.
- `pct_of_52w_high` — `last_close / max(last 252 closes)`, 4 dp.
- `pct_above_52w_low` — `last_close / min(last 252 closes)`, 4 dp.

Added **only when a `forecast` is supplied** (from `forecast.cum_return` / `.upside` / `.downside`):
- `forecast_cum_return` — `forecast.cum_return`, 5 dp.
- `forecast_mean_upside` — mean of `forecast.upside`, 5 dp.
- `forecast_mean_downside` — mean of `forecast.downside`, 5 dp.

## Fail-safe & caching
- Depends on **gather-ohlc**: the parquet must already exist and be fresh. There is no fetch here.
- Missing parquet → `load_daily_ohlcv` raises `FileNotFoundError`; `load_history` raises `ValueError`
  on a stale/empty window (latest bar older than the last closed NYSE session). These propagate up
  and the caller skips the ticker — [`research_many`](src/ophir/agent/research.py:332) catches
  `(ValueError, FileNotFoundError, OSError)`, logs a `research_failed` audit event, and moves on.
- `recent_return_1d` / `vol_20d` / `vol_60d` are passed through `_num` and become `None` (never NaN/inf).
- No caching/rate limit of its own — staleness is enforced upstream by `load_history`, recency by the
  parquet (refresh via gather-ohlc, not here).

## How to use / extend
- Direct: `from ophir.agent.research import gather_technicals; gather_technicals("AAPL")`
  (no forecast), or pass a [`Forecast`](src/ophir/agent/predict.py:35) from
  `ophir.agent.predict.predict_ticker` to add the three forecast fields.
- In context this dimension is assembled inside
  [`research_ticker`](src/ophir/agent/research.py:255) alongside `gather_fundamentals` /
  `gather_news`; it appends `ophir:features` (and `ophir:forecast`) to the brief's `sources`.
- To add a sibling derived indicator: compute it inside `gather_technicals` from `feats` /
  `last` / `df` (e.g. a new rolling feature already produced by `extract_features`), round it,
  and put it in the `tech` dict — keep every value finite via `_num`, and gate forecast-only
  fields behind `if forecast is not None`.
