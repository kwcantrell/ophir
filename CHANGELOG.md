# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.10.1] - 2026-06-17

### Fixed

- Hardened bull/bear thesis JSON parsing. A `Thesis` field validator coerces a
  `key_points` / `key_risks` value that the model returns as a single bulleted string
  into a clean list and drops a leaked `key_risks':[...]` fragment, and
  `_extract_json_object` is now string-aware so a `}` inside a value (e.g. in a summary)
  no longer truncates the object early.

## [0.10.0] - 2026-06-17

### Added

- `ophir.agent.market_calendar` (`last_closed_session`, `is_trading_day`) backed by the
  NYSE calendar (new `pandas-market-calendars` dependency). The live run forecasts off the
  last *closed* session and the `ophir trade` / `ophir manage` cycle no-ops on non-trading
  days (weekends, holidays) instead of trading on stale data.
- Per-stock dossiers: `ophir.agent.report.write_reports` writes one `<SYMBOL>.md` per
  considered ticker — model forecast, quant/Ollama decisions, research stance, the bull and
  bear theses, and the manager's accept/reject decision with risk-gate notes — plus an
  `INDEX.md` roll-up, under `AGENT_REPORT_DIR` (new `report_dir` setting, default
  `<DATA_DIR>/reports`). Both `ophir trade` and `ophir manage` emit them.

### Fixed

- Forecast differentiation: the forecast (and backtest) response block is now marked
  `trade_occured=True` so its tokens attend to the ticker's history; the padding mask had
  blanked them out, collapsing every ticker to an identical constant.
- The manager no longer fails safe to all-cash on a large book: every Ollama call sets
  `num_ctx=ollama_num_ctx` (16384) so the aggregated dossier prompt no longer fills the
  context window before any JSON is emitted.
- Execution: full-position exits liquidate by quantity (`close_position`) rather than a
  notional that rounds to more shares than held; order notionals are quantized to cents
  (Alpaca rejects more than two decimal places); and `place_orders` logs and continues past
  a single rejected order instead of aborting the whole rebalance.

### Changed

- The live forecast conditions on the last completed NYSE session — today's still-forming
  bar is dropped — via the new `feed.load_history`, which also refuses a stale feed;
  `predict` and `research` both route through it.

## [0.9.1] - 2026-06-17

### Fixed

- Base training no longer dies on a NaN-poison batch: `training_step` /
  `validation_step` skip any batch whose loss is non-finite (a single bad bar — e.g.
  a `low` of `0.00`, making `downside = log(close/low) = +inf` — used to propagate to
  NaN weights and corrupt the entire run). `_masked_mean` also guards the empty-mask
  case defensively.

### Added

- `ophir.training_callbacks.GradNormMonitor` logs the pre-clip gradient norm each
  step, wired in through a new optional `extra_callbacks` parameter on
  `register.fetch_base_trainer`.

### Changed

- `fetch_base_trainer` now also saves a stable `*-last.ckpt` (the most-trained
  weights); near-random out-of-sample validation loss makes best-val checkpoint
  selection unreliable for this model.
- Training `DataLoader`s run in-process on Windows (the streaming dataset holds an
  un-picklable generator) and keep their parallel workers on Linux.

## [0.9.0] - 2026-06-15

### Added

- Backtest & validation for the quant signal: `ophir.agent.backtest` and the
  `ophir backtest` CLI command. `compute_metrics` reports total / annualized
  return, Sharpe, Sortino, max drawdown, Calmar, and hit rate. `walk_forward` is
  an event-driven, **no-look-ahead** backtest — it forecasts as-of each rebalance
  date, builds a BUY book through the same `apply_risk_gate` used live, marks it to
  the next bar minus per-turnover basis-point costs, and benchmarks against SPY
  buy-and-hold. `purged_kfold` (purge + embargo for overlapping forecast-horizon
  labels) and `signal_cv` report the per-fold Spearman information coefficient
  (constant/degenerate predictions are skipped safely). The LLM layers are not
  backtested; they are validated by paper track record.

## [0.8.0] - 2026-06-15

### Added

- Paper-execution layer `ophir.agent.execute` and the `ophir trade` CLI command.
  A `Broker` protocol with an in-process simulated `PaperBroker` (deterministic,
  no network) and an `AlpacaPaperBroker` adapter (real Alpaca **paper** account,
  dynamically imported, constructed only with credentials). `reconcile` deltas the
  target portfolio against the broker's positions into orders (target =
  `weight * equity`, **sells before buys**, `Decimal` money, a no-trade band, and a
  deterministic `client_order_id` for idempotency); `place_orders` **defaults to a
  dry run** (logs the plan, submits nothing) and only submits with an explicit
  opt-in; `daily_report` is a reporting-only LLM summary that fails safe to a
  template. `ophir trade <SYMBOLS> [--top-k] [--broker paper|alpaca]
  [--execute/--dry-run]` chains predict → decide → research → debate → manage →
  reconcile → place_orders, feeding the broker account's drawdown / daily loss into
  the risk-gate kill-switch.

### Changed

- Add the `alpaca-py` dependency (paper-trading broker adapter) and refresh
  `uv.lock`.

## [0.7.0] - 2026-06-15

### Added

- Manager + deterministic risk gate `ophir.agent.manage` and the `ophir manage`
  CLI command — the final portfolio decision. A manager LLM ingests each
  candidate's full ensemble (forecast + quant/Ollama decisions + research brief +
  bull/bear debate) and returns ranked picks each with a conviction and rationale
  (never raw weights); deterministic code sizes the convictions by
  inverse-volatility scaled toward `annual_vol_target`; and a risk gate enforces
  the per-name cap and gross-exposure limit, drops unknown/stale/non-finite picks,
  and halts to all-cash on a drawdown / daily-loss kill-switch. The manager fails
  safe to an all-cash portfolio when Ollama is unreachable. `ophir manage` chains
  predict → decide → research → debate → manage and prints the gated target
  portfolio. Reuses the existing `AgentSettings` risk knobs.

## [0.6.0] - 2026-06-15

### Added

- Bull/bear debate layer `ophir.agent.debate` and the `ophir debate` CLI command:
  for each top-ranked ticker's research brief, the local `gpt-oss:20b` model
  argues an independent bullish and a bearish thesis (summary, key points, key
  risks, stance strength) using only the brief's grounded data, validated and
  fail-safe to a neutral thesis when Ollama is unreachable. Weighing the two
  sides is a later phase.

## [0.5.1] - 2026-06-15

### Fixed

- The decision and research LLM tracks now call Ollama with JSON-constrained
  decoding (`format="json"`), so `ophir decide` and `ophir research` reliably
  receive valid JSON instead of occasionally emitting unparseable output that
  forced the fail-safe (a HOLD decision or a neutral brief).

## [0.5.0] - 2026-06-15

### Added

- Research layer `ophir.agent.research` and the `ophir research` CLI command:
  for each top-ranked ticker, gather grounded fundamentals (Yahoo Finance
  `.info`), recent news (Yahoo Finance `.news`), and technicals (ophir features
  plus the model forecast), then have the local `gpt-oss:20b` model summarize
  *only that data* into a cited `ResearchBrief` (validated; fail-safe to a
  neutral brief with the grounded data intact when Ollama is unreachable). Adds
  a `research_news_limit` setting to `AgentSettings`.

### Fixed

- The CLI now reconfigures stdout / stderr to UTF-8 at startup, so
  `ophir research` and `ophir decide` no longer crash with `UnicodeEncodeError`
  when printing LLM-generated Unicode (curly quotes, non-breaking hyphens) on a
  Windows cp1252 console.

## [0.4.0] - 2026-06-14

### Added

- Trading-agent decision layer `ophir.agent.decide` and the `ophir decide` CLI
  command: turn each model `Forecast` into a buy/sell/hold `Decision` via two
  tracks — a deterministic quant rule (`buy_threshold` / `sell_threshold` with an
  optional downside penalty) and a local `gpt-oss:20b` Ollama verdict (grounded
  in the forecast numbers, validated against a JSON schema, fail-safe to `HOLD`)
  — compared side by side with an agreement flag. Every decision is written to
  the audit trail.
- `AgentSettings` gains the decision thresholds plus `ollama_model` and
  `ollama_base_url` (env `AGENT_OLLAMA_BASE_URL`); `ophir decide` prints a
  preflight warning when the Ollama track is selected but the server is
  unreachable. Adds a "Setting up Ollama" installation runbook and API / CLI /
  README documentation.

## [0.3.2] - 2026-06-14

### Fixed

- `_latest_base_ckpt` / `_latest_finetuned_ckpt` crashed with `IndexError` when
  two or more matching checkpoints lacked a `-v<N>` suffix (the best-epoch
  checkpoints, named `…basebest-epoch=NN-val_loss=X`). They now share a robust
  `_latest_ckpt` helper that orders `-v<N>` versions, falls back to the most
  recently modified file otherwise, and raises `FileNotFoundError` when nothing
  matches — restoring `ophir predict`, `ophir rank`, and `ophir decide`.

## [0.3.1] - 2026-06-10

### Fixed

- `OHLCMulitClassPredictor.forward` now zeros the response-region rows of
  `feature_input` before the feature MLP. `r_close` / `upside` / `downside` are
  both input features and targets, so the self-attending response tokens could
  copy the answer instead of forecasting; at inference (future rows zeroed) the
  model collapsed to an identical-per-ticker constant. The change is
  shape-preserving (existing checkpoints still load); the model must be retrained
  to benefit.

## [0.3.0] - 2026-06-10

### Added

- Trading-agent prediction layer under `ophir.agent`: `config` (pydantic-settings
  with paper / dry-run / allow-live defaults and a live-mode guard), `audit`
  (structlog append-only JSON audit trail), and `predict` (a `Forecast` dataclass
  with `predict_ticker` / `predict_many` / `rank`). `ophir.agent.feed` gains
  `forecast_window_tensors`, which builds a forward-looking window (real history
  plus zeroed future rows) for genuine forecasts.
- CLI commands `ophir predict <SYMBOL>`, `ophir rank <SYMBOLS> [--top-k]`, and
  `ophir train` (full-US-market trainer, `<2024` train / `>=2024` validation
  split, fine-tune or from-scratch via `--finetune-from` / `--max-steps`).
  `register.fetch_base_trainer` gains a `max_steps` argument.

### Changed

- Add `pydantic-settings` and `structlog` dependencies. Source `torch` from the
  PyTorch CUDA 13.0 index and pin `torch<2.11` (flex-attention compilation
  regresses on 2.11+); refresh `uv.lock` accordingly.

## [0.2.1] - 2026-06-04

### Added

- `trading-best-practices` Claude Code skill
  (`.claude/skills/trading-best-practices/SKILL.md`) capturing trading-system
  best practices (paper-first defaults, a pre-trade risk gate + drawdown
  kill-switch, look-ahead/survivorship-bias-free backtests, and LLM-in-the-loop
  safety) to guide future trading-agent work. Repo tooling; no runtime impact on
  the `ophir` package.

## [0.2.0] - 2026-06-04

### Added

- `ophir ingest <SYMBOL> [--days N]` command and the `ophir.agent` ingestion
  modules: fetch a ticker's daily OHLC from Yahoo Finance and persist it
  model-ready in the existing parquet layout, reusing
  `ophir.ticker.extract_features` / `extract_model_data`.
  `ophir.agent.feed.latest_window_tensors` bridges the most recent window
  to the model's `(S, 13)` / `(S, 3)` input tensors. Yahoo data is fetched
  split/dividend-adjusted (`auto_adjust=True`); ophir's split back-adjustment
  is skipped on this path to avoid double-adjustment. No GPU required.

## [0.1.7] - 2026-05-20

### Fixed

- The chat panel's system prompt was constructed as a 4-tuple of strings
  (`SystemMessage(content=(str, str, str, str,))`) due to a trailing comma
  per line, sending a tuple to the LLM instead of a single string.
  Concatenated into one string so the chat panel now receives the intended
  system prompt.

### Changed

- Type-annotate `ophir.ui` and drop it from the mypy `ignore_errors`
  overrides — the override block is now empty and removed. Drop the
  `gradio.*` `ignore_missing_imports` override (gradio 6.4 ships
  `py.typed`); add `plotly.*` to the consolidated `ignore_missing_imports`
  block (plotly ships no stubs). Add `warn_unused_ignores = true` to
  `[tool.mypy]`. Make `return_ckpt_path: Literal[True]` keyword-only in
  both `load_base_model_ckpt` / `load_fintuned_ckpt` overloads so callers
  can pass it with `strict` defaulted.

## [0.1.6] - 2026-05-20

### Changed

- Type-annotate `ophir.models` and `ophir.ticker`; drop them from the ruff
  `ANN` per-file-ignores and the mypy `ignore_errors` overrides. Widen
  `extract_model_data(response_size)` from `int` to `int | numpy.ndarray`
  to match existing call sites in `StockStreamerDataset` and
  `StockHandlerDataset` (non-breaking — both forms already worked at
  runtime). Widen `get_splits` return to `dict[str, StockSplit | None]`
  to reflect the sentinel passthrough already exercised by tests.
  Consolidate the third-party `ignore_missing_imports` mypy override into
  one block covering `gradio`, `massive`, `tqdm`, and `yfinance`.

## [0.1.5] - 2026-05-20

### Changed

- Type-annotate `ophir.model_data`, `ophir.register`, and
  `ophir.training_models` and drop them from the coupled ruff `ANN`
  per-file-ignores and the mypy `ignore_errors` overrides. No user-facing
  behavior change. `models.py`, `ticker.py`, and `ui.py` remain suppressed.

## [0.1.4] - 2026-05-19

### Added

- Initial `pytest` configuration (`[tool.pytest.ini_options]`) and a
  deterministic, network-free test suite for `ophir.ticker` (73 tests across
  helpers, network, features, handler, streamer, and datasets). `pd.read_html`
  and `yfinance.Ticker` are mocked; fixtures never touch the package's
  `.ophir/` layout. Adds `pytest>=8.0` and `pytest-mock>=3.14` to the dev
  group.

### Fixed

- `StockHanlder.narrow_stocks` previously printed `stocks not found: 0`
  regardless of input (`len(stock_list) - len(stock_list)`). Now reports the
  actual count of unmatched symbols.

## [0.1.3] - 2026-05-19

### Added

- `CLAUDE.md` agent-bootstrap guide at the repo root: `src/ophir/` module
  map, dev workflow, type-checking tech-debt layout, and runtime
  requirements. Lets new agent sessions ramp up without re-exploring. Repo
  tooling; no runtime impact on the `ophir` package.

## [0.1.2] - 2026-05-15

### Added

- Static type checking via `mypy` (strict; configured in `pyproject.toml`),
  enforced by a local `pre-commit` hook that runs `mypy` against the project
  venv. The 5 legacy torch/lightning modules and `ophir.ui` are relaxed via
  `ignore_errors` overrides that mirror the existing ruff `ANN`
  per-file-ignores as documented tech debt. Repo tooling; no runtime impact on
  the `ophir` package.

## [0.1.1] - 2026-05-14

### Added

- `commit` Claude Code skill (`.claude/skills/commit/SKILL.md`) that automates
  release-hygiene commits: groups the working tree into ordered commit units,
  infers SemVer bumps, updates the changelog and version files, and commits
  only after explicit confirmation. Repo tooling; no runtime impact on the
  `ophir` package.
- Project README, changelog, and a Sphinx + autodoc documentation site,
  published to GitHub Pages via a `Docs` GitHub Actions workflow.

## [0.1.0] - 2026-05-14

### Added

- Initial project scaffold, `.gitignore`, and `pyproject.toml`.
- Core transformer models for OHLC prediction.
- Multi-stock price predictor support and training/residual notebooks.
- Pip-installable package with CLI commands to download datasets.
- `ophir` console script with `serve` and `register` subcommands.
- Model validation utilities.

### Changed

- Converted the architecture from a GPT-style causal decoder LLM to a
  BERT-style full-encoder masked LLM.
- Migrated to a `src/` layout and removed the abandoned top-level `ophir/`.
- Made the entire codebase ruff-clean.
- Track the shared `.claude/settings.json`; ignore local settings.

### Fixed

- Corrected the base exponent in `RotationNumberEmbeddings` so the maximum
  value yields a rotation of π.
- Model validation and minor fixes.

[Unreleased]: https://github.com/kwcantrell/ophir/compare/v0.10.1...HEAD
[0.10.1]: https://github.com/kwcantrell/ophir/compare/v0.10.0...v0.10.1
[0.10.0]: https://github.com/kwcantrell/ophir/compare/v0.9.1...v0.10.0
[0.9.1]: https://github.com/kwcantrell/ophir/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/kwcantrell/ophir/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/kwcantrell/ophir/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/kwcantrell/ophir/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/kwcantrell/ophir/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/kwcantrell/ophir/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/kwcantrell/ophir/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/kwcantrell/ophir/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/kwcantrell/ophir/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/kwcantrell/ophir/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/kwcantrell/ophir/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/kwcantrell/ophir/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/kwcantrell/ophir/compare/v0.1.7...v0.2.0
[0.1.7]: https://github.com/kwcantrell/ophir/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/kwcantrell/ophir/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/kwcantrell/ophir/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/kwcantrell/ophir/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/kwcantrell/ophir/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/kwcantrell/ophir/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/kwcantrell/ophir/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/kwcantrell/ophir/releases/tag/v0.1.0
