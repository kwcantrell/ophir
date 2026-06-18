# Ophir data inputs

This catalogs every external and derived data input the ophir trading agent consumes on a daily run, with the exact function to reuse, the fields it returns, how it is persisted/cached, and how it fails safe. Each input names the `gather-*` skill that documents it in depth (or `(derived/none)` when there is no fetch step). Use this page as the index; jump to a skill for the call recipe and extension notes.

## Price & market data

- **OHLC daily bars (yfinance)**
  - Provider: Yahoo Finance via `yfinance` — `yf.Ticker(symbol).history(start, end, interval="1d", auto_adjust=True)`.
  - Fetch: [agent/ingest.py:85](src/ophir/agent/ingest.py:85) (`ingest`), low-level fetch [agent/ingest.py:30](src/ophir/agent/ingest.py:30) (`_fetch_yahoo`). Read back with [agent/feed.py:49](src/ophir/agent/feed.py:49) (`load_daily_ohlcv`) and the session-trimmed [agent/feed.py:77](src/ophir/agent/feed.py:77) (`load_history`).
  - Fields: `high`, `low`, `close`, `volume`, indexed by `utc_time` (tz-naive, deduped, sorted ascending). `auto_adjust=True` so bars are already split/dividend-adjusted.
  - Persistence: written to the Hive layout `<DATA_DIR>/days/stocks/symbol=<SYMBOL>/data.parquet` (path via [agent/feed.py:44](src/ophir/agent/feed.py:44), `parquet_path`); re-ingest overwrites.
  - Fail-safe: empty/delisted symbol raises `ValueError`; staleness (last bar > 5 days old) and sparsity emit non-fatal warnings ([agent/ingest.py:61](src/ophir/agent/ingest.py:61)). `load_history` refuses a stale feed (`ValueError`) so callers skip the ticker.
  - Skill: gather-ohlc

- **SPY benchmark**
  - Provider: same yfinance OHLC mechanism — SPY is just another ticker ingested and read through `ingest` / `load_daily_ohlcv`.
  - Fetch: [agent/ingest.py:85](src/ophir/agent/ingest.py:85), read via [agent/feed.py:49](src/ophir/agent/feed.py:49).
  - Fields: identical to the OHLC bars above (`high`/`low`/`close`/`volume`).
  - Persistence: `symbol=SPY/data.parquet` under the same stocks root.
  - Fail-safe: same as OHLC bars.
  - Skill: gather-ohlc

- **Stock-split history (yfinance `.splits`)**
  - Provider: Yahoo Finance — `yf.Ticker(ticker).splits`.
  - Fetch: [ticker.py:139](src/ophir/ticker.py:139) (`get_splits`); apply via [ticker.py:223](src/ophir/ticker.py:223) (`StockSplit.apply_splits`).
  - Fields: per ticker a `StockSplit` of `id`, `dates` (effective split dates), `ratios`; tickers with no splits map to a `None` sentinel.
  - Persistence: pickle cache at `<DATA_DIR>/yf_splits_cache.pkl`; only missing tickers hit the network, sentinels prevent re-querying.
  - Fail-safe: a failed ticker is printed and skipped. NOTE: largely dormant in the live path — OHLC is ingested with `auto_adjust=True`, so splits are only needed for raw/unadjusted price adjustment.
  - Skill: gather-splits

## Research data

- **Fundamentals (yfinance `.info`)**
  - Provider: Yahoo Finance — `yf.Ticker(symbol).info`.
  - Fetch: [agent/research.py:107](src/ophir/agent/research.py:107) (`gather_fundamentals`).
  - Fields: curated `_FUNDAMENTAL_KEYS` subset — `sector`, `industry`, `marketCap`, `trailingPE`, `forwardPE`, `profitMargins`, `beta`, `dividendYield`, `fiftyTwoWeekHigh`, `fiftyTwoWeekLow`, `currentPrice`.
  - Persistence: none of its own; lands in `ResearchBrief.fundamentals` (audit-logged via the research event).
  - Fail-safe: any network/parse error returns `{"error": "fundamentals unavailable (...)"}` (grounded-but-empty), never raising.
  - Skill: gather-fundamentals

- **News (yfinance `.news`)**
  - Provider: Yahoo Finance — `yf.Ticker(symbol).news`.
  - Fetch: [agent/research.py:144](src/ophir/agent/research.py:144) (`gather_news`), normalized per-item by [agent/research.py:118](src/ophir/agent/research.py:118) (`_normalize_news_item`).
  - Fields: per headline `title`, `publisher`, `link`, `published` (normalized across yfinance's old flat and new nested payload shapes); limited to `research_news_limit` (default 8).
  - Persistence: none of its own; lands in `ResearchBrief.news`, with `link`s recorded in `sources`.
  - Fail-safe: any error returns `[]` (no news); the brief still synthesizes.
  - Skill: gather-news

- **Technicals (derived from OHLC + forecast)**
  - Provider: derived — no external fetch; computed from the ingested OHLC parquet and the model `Forecast`.
  - Fetch: [agent/research.py:155](src/ophir/agent/research.py:155) (`gather_technicals`), reusing [ticker.py:259](src/ophir/ticker.py:259) (`extract_features`).
  - Fields: `last_close`, `recent_return_1d`, `vol_20d`, `vol_60d`, `pct_of_52w_high`, `pct_above_52w_low`; when a `Forecast` is supplied also `forecast_cum_return`, `forecast_mean_upside`, `forecast_mean_downside`.
  - Persistence: none of its own; lands in `ResearchBrief.technicals`.
  - Fail-safe: requires ingested OHLC — `load_history` raises on a missing/stale feed, which `research_many` logs and skips.
  - Skill: gather-technicals

## Universe & calendar

- **S&P 500 constituents (Wikipedia)**
  - Provider: Wikipedia — `pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")`.
  - Fetch: [ticker.py:108](src/ophir/ticker.py:108) (`get_sp_500_symbols`).
  - Fields: a `list[str]` of ticker symbols from the constituents table's `Symbol` column.
  - Persistence: none — live HTTP request each call (invoked at import time by `ophir.ui`).
  - Fail-safe: weak — a fetch error is printed and `dfs` is left unbound, so the call then raises; treat the universe as point-in-time (survivorship caveat for backtests).
  - Skill: gather-sp500-universe

- **Curated watchlist (`ophir-bot/watchlist.txt`)**
  - Provider: operational file in the `ophir-bot` repo — one ticker per line, `#` comments and blanks ignored (~28 names today).
  - Fetch: read by `ophir-bot/rebalance.ps1`, not by `src/ophir` code; the daily run scores every name here, then the manager + risk gate pick `top_k`.
  - Fields: bare ticker symbols (e.g. `AAPL`, `NVDA`).
  - Persistence: the file itself; hand-edited.
  - Fail-safe: operational concern of the runner script, not the library.
  - Skill: (operational; no skill)

- **NYSE trading calendar (pandas-market-calendars)**
  - Provider: `pandas_market_calendars` — `mcal.get_calendar("NYSE")`.
  - Fetch: [agent/market_calendar.py:48](src/ophir/agent/market_calendar.py:48) (`last_closed_session`) and [agent/market_calendar.py:80](src/ophir/agent/market_calendar.py:80) (`is_trading_day`).
  - Fields: `last_closed_session` returns a tz-naive midnight `Timestamp` of the most recently closed session; `is_trading_day` returns a `bool`.
  - Persistence: the calendar object is `lru_cache`d per process ([agent/market_calendar.py:29](src/ophir/agent/market_calendar.py:29)); anchored on UTC/US-Eastern, never the host clock.
  - Fail-safe: no closed session in the 12-day lookback raises `ValueError`; the daily run uses `is_trading_day` to skip weekends/holidays rather than forecast on stale data.
  - Skill: gather-market-calendar

## Account & execution

- **Alpaca paper account + positions (alpaca-py)**
  - Provider: Alpaca **paper** trading via `alpaca-py` — `alpaca.trading.client.TradingClient(..., paper=True)`.
  - Fetch: [agent/execute.py:119](src/ophir/agent/execute.py:119) (`AlpacaPaperBroker`); `.get_account()` [agent/execute.py:140](src/ophir/agent/execute.py:140), `.get_positions()` [agent/execute.py:152](src/ophir/agent/execute.py:152). `PaperBroker` ([agent/execute.py:77](src/ophir/agent/execute.py:77)) is the in-memory simulated alternative for dry runs/tests.
  - Fields: `Account(equity, cash, drawdown, daily_loss)` (all `Decimal`/float); positions as `{symbol: market_value}` (`Decimal`).
  - Persistence: none locally — the broker is the source of truth, reconciled every cycle ([agent/execute.py:191](src/ophir/agent/execute.py:191), `reconcile`).
  - Fail-safe: missing `AGENT_ALPACA_KEY_ID` / `AGENT_ALPACA_SECRET_KEY` raises at construction; a single rejected order is logged and skipped, never aborting the rebalance; `dry_run=True` is the default (plans only).
  - Skill: gather-account

## Model artifacts (derived / loaded; no gather-skill)

- **13 engineered features**
  - Derived from the ingested OHLC frame — no fetch.
  - Compute: [ticker.py:259](src/ophir/ticker.py:259) (`extract_features`).
  - Fields: `time_delta`, `r_close`, the 10/20/60-day `*_norm_returns` / `*_norm_volume` / `*_volatility`, `upside`, `downside`, plus the `trade_occured` calendar-padding flag.
  - Persistence: none — recomputed per window; padded days zero-filled.
  - Fail-safe: empty input returns an empty frame.
  - Skill: (derived/none)

- **Per-ticker forecast (`Forecast`)**
  - Derived by running the trained model over the latest 365-day window — no fetch.
  - Compute: [agent/predict.py:78](src/ophir/agent/predict.py:78) (`predict_ticker` → `Forecast`), windowed by [agent/feed.py:163](src/ophir/agent/feed.py:163) (`forecast_window_tensors`).
  - Fields: `symbol`, `asof`, `horizon`, per-day `r_close`/`upside`/`downside` lists, `cum_return`, `score`.
  - Persistence: none of its own; logged to the audit trail as a `forecast` event.
  - Fail-safe: a stale/missing feed raises and `predict_many` logs + skips the ticker; requires a CUDA GPU.
  - Skill: (derived/none)

- **Trained checkpoint**
  - Loaded from disk — no fetch.
  - Load: [agent/predict.py:64](src/ophir/agent/predict.py:64) (`load_predictor`, process-cached) → [register.py:330](src/ophir/register.py:330) (`load_base_model_ckpt`, `time_version=False` for the validation-best forecast checkpoint); selection via the `basebest-*.ckpt` resolver [register.py:189](src/ophir/register.py:189) (`_latest_ckpt`).
  - Fields: a `LightningOHLCPredictor` on the GPU in eval mode.
  - Persistence: `.ckpt` files under `<MODEL_DIR>` (`src/ophir/.ophir/model/`).
  - Fail-safe: no matching checkpoint raises `FileNotFoundError`.
  - Skill: (derived/none)

## LLM

- **Local Ollama `gpt-oss:20b`**
  - Provider: a local Ollama server via `langchain_ollama.ChatOllama`; a *service dependency*, not fetched data.
  - Use: decide / research / debate / manage / report stages — e.g. the research synthesis in [agent/research.py:300](src/ophir/agent/research.py:300) and the end-of-day report in [agent/execute.py:317](src/ophir/agent/execute.py:317).
  - Config: `ollama_model` (`gpt-oss:20b`), `ollama_base_url`, `ollama_num_ctx` (16384) in [agent/config.py:36](src/ophir/agent/config.py:36).
  - Persistence: none — stateless calls; `temperature=0`, JSON-constrained where a structured reply is required.
  - Fail-safe: every LLM stage degrades to a neutral/templated result on error (e.g. research falls back to a neutral analysis, the report to a plain templated summary) while keeping the grounded data intact.
  - Skill: (service dependency; no gather-skill)

## Dormant / dead code

- **MASSIVE REST client**
  - Provider: `massive.RESTClient`, authenticated from `src/ophir/.ophir/.massive_key` (written by `ophir register massive-key`).
  - Defined at [register.py:167](src/ophir/register.py:167) (`get_massive_client`).
  - Status: **defined but never called** anywhere in `src/` — a grep for `get_massive_client(` finds only the definition (callers appear solely in docs/CLAUDE.md prose). No live or training path consumes MASSIVE data; OHLC comes entirely from yfinance. Safe to remove (function, the `massive-key` CLI command, and the doc references) when cleaning up.
  - Skill: (dead code; no skill)

## Skills

- [gather-ohlc](../.claude/skills/gather-ohlc/SKILL.md) — daily OHLC bars (and the SPY benchmark) from yfinance.
- [gather-splits](../.claude/skills/gather-splits/SKILL.md) — stock-split history from yfinance `.splits`.
- [gather-fundamentals](../.claude/skills/gather-fundamentals/SKILL.md) — company fundamentals from yfinance `.info`.
- [gather-news](../.claude/skills/gather-news/SKILL.md) — recent headlines from yfinance `.news`.
- [gather-technicals](../.claude/skills/gather-technicals/SKILL.md) — derived indicators from OHLC + forecast.
- [gather-sp500-universe](../.claude/skills/gather-sp500-universe/SKILL.md) — S&P 500 constituents from Wikipedia.
- [gather-market-calendar](../.claude/skills/gather-market-calendar/SKILL.md) — NYSE sessions from pandas-market-calendars.
- [gather-account](../.claude/skills/gather-account/SKILL.md) — Alpaca paper account + positions via alpaca-py.
- [data-inputs](../.claude/skills/data-inputs/SKILL.md) — the main index skill mapping every input above to its gather skill.
