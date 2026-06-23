# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- ReZero depth diagnostic: opt-in `rezero_init`, `--decouple-rezero-schedule`,
  and `--log-rezero-gates` training knobs (all default to current behavior),
  a `rezero_gate_stats` helper, and `dashboard.summarize_rezero_runs` to compare
  experiment arms. See `docs/rezero-diagnostic-runbook.md`.
- Add `ophir sweep`: an Optuna hyperparameter sweep harness that searches
  optimizer, loss-weight, and architecture-tier hyperparameters by mean
  cross-sectional rank-IC on `r_close`, with proxy-budget search (ASHA pruning,
  resumable SQLite study) and a full-budget confirm phase. Exposes the
  previously-buried `rezero_lr`, `betas`, and loss-weight knobs on `ophir train`
  and adds an opt-in `val_rank_ic` validation metric.
- `alpaca-trader` Claude Code skill (`.claude/skills/alpaca-trader/`) and
  `ophir.trading` deterministic core. The skill provides morning/evening
  workflow automation (proposal → safety gate → paper order → outcome scoring)
  driven by Alpaca MCP. The trading core covers: typed domain models
  (`types.py`), config loading (`config.py`), a position-sizing and
  order-validity safety gate (`safety.py`), an append-only JSON-Lines ledger
  (`ledger.py`), portfolio performance metrics (`metrics.py`), multi-factor
  signal blending with graceful ophir-absent fallback (`signals.py`),
  per-entity section-upsert memory (`memory.py`), exposure aggregation
  (`exposure.py`), end-of-day outcome scoring (`outcomes.py`), an ophir
  forecast adapter (`forecast.py`), and a `trade` CLI subcommand group
  (`cli.py`). The memories knowledge-base tree (`memories/tickers/`,
  `memories/sectors/`, `memories/ledger/`) is seeded at repo root. Unit tests
  cover all trading-core modules (full suite: 231 passing).
- Cross-sectional rank-IC in the validation report (`ophir.evaluate`):
  `accumulate_targets` now also collects per-`r_close`-prediction `(stock_id,
  date)` identity (when the loader carries it), exposed as `r_close_ids` /
  `r_close_dates` on `AccumulatedEval`. New pure helper
  `dedupe_by_ticker_date(pred, target, ids, dates)` keeps the first prediction
  per `(ticker, date)` (stable order) and returns the per-row date strings for
  `rank_ic`. `evaluate_model` reports `rank_ic_mean` / `rank_ic_ir` for
  `r_close`, and the `evaluate` command builds its loader with
  `return_identity=True`. Covered by new tests in `tests/test_evaluate.py`.
- Opt-in eval identity plumbing: `OHLCMulitClassPredictorInput` gains optional
  `stock_id` / `date_ordinal` tensor fields (default `None`, mirroring `time`);
  `extract_model_data(..., stock_id=...)` emits a 0-dim `long` `stock_id` and an
  int64 `(seq_len,)` `date_ordinal` from the window index; `StockStreamer` gains
  a `symbol` field; `StockHandlerDataset(..., return_identity=False)` and
  `build_dataloader(..., return_identity=False)` thread the flag through. All
  opt-in and off by default, so the training collate and path are byte-for-byte
  unchanged. Covered by new tests in `tests/test_model_data.py`,
  `tests/test_ticker_features.py`, and `tests/test_ticker_datasets.py`.
- `prefix_last_observed` + `AccumulatedEval` (`ophir.evaluate`): the validation
  report now scores `upside`/`downside` against a persistence baseline. The new
  pure helper `prefix_last_observed(values, trade_occured, response_size)`
  returns each row's value at its last traded prefix position (falling back to
  prefix position 0); `accumulate_targets` carries that value flat across the
  horizon as a baseline and now returns an `AccumulatedEval` (masked
  `channels` + per-channel `baselines`) instead of a bare dict. `evaluate_model`
  reports `skill_vs_persistence` for the two magnitude channels. Covered by new
  unit tests in `tests/test_evaluate.py`.
- `skill_score_vs_baseline` (`ophir.evaluate`): pure, CPU-safe RMSE skill
  score against an arbitrary baseline tensor —
  `1 - rmse(pred, target) / rmse(baseline, target)`. Returns `nan` for empty
  input or a zero-RMSE baseline. Lets the non-negative `upside`/`downside`
  channels be scored against a persistence/EWMA forecast rather than having no
  reference point. Covered by two new unit tests in `tests/test_evaluate.py`.
- `rank_ic` / `_spearman` (`ophir.evaluate`): pure, CPU-safe daily
  cross-sectional rank-IC metric. `rank_ic(pred, target, dates)` groups
  predictions and targets by day label, computes the Spearman rank correlation
  within each cross-section, and returns `{"ic_mean", "ic_std", "ic_ir",
  "n_days"}`. Covered by two new unit tests in `tests/test_evaluate.py`.

- `pool_prefix_embedding` (module-level, `ophir.models`): mean-pools the
  prefix (observed-history) positions `x[:, :-response_size]` into one vector
  per example for the UI PCA projection, replacing the previous pool over the
  masked forecast block. Signal-bearing prefix positions now drive the per-stock
  embedding; zero impact on forecast loss.
- `apply_output_activations` (module-level, `ophir.models`): passes `r_close`
  through unchanged and applies `softplus` to the `upside` and `downside`
  channels so the two log-magnitude heads are guaranteed non-negative. Negative
  values would invert the reconstructed candle (`high < close` or `low > close`)
  via the `.exp()` call in `model_data.py`. Wired into `OHLCMulitClassPredictor.forward`
  immediately after `out_ff`. Existing checkpoints remain loadable but will need
  a retrain to benefit from the constrained output distribution.
- `extract_features` now emits a `feature_valid` boolean column: `False` for
  the first 59 warm-up rows (where the 60-day rolling features are undefined)
  and for calendar-padding rows, `True` otherwise. `StockStreamer` uses this
  flag to skip warm-up rows when computing window start positions so no
  zero-filled warm-up features enter any training window.
- `robust_scale` helper (`ophir.training_models`): computes a Gaussian-equivalent
  scale from the median absolute deviation (`1.4826 * MAD`), floored at `1e-4`.
  Each channel's smooth-L1 `beta` in `compute_loss` is now derived from the
  MAD of its masked target so Huber's transition sits at the actual noise scale
  rather than a hardcoded constant.
- `--sampler {tpe,random}` and `--no-prune` options on `ophir sweep`: choose
  between TPE (default) and random search, and optionally disable ASHA pruning.
- `ophir importances <study>`: reports fANOVA and MDI hyperparameter
  importances for a completed sweep study, with a reliability warning for
  biased (TPE/ASHA) designs. Requires `scikit-learn` (added as a dependency
  for the Optuna importance evaluators).
- `ophir.ceiling`: offline helpers for the forecasting-ceiling investigation —
  run IC-trajectory summary (peak / best-checkpoint / final `val_rank_ic`),
  multi-seed aggregation + minimum-detectable-effect, and naive cross-sectional
  baselines reusing the production rank-IC math.
- E3 forecast-horizon diagnostic: `ophir.ceiling.signal_decay_curve` /
  `pooled_baseline_ceiling` (reversal IC vs forecast lead + matched-horizon
  ceiling), `ophir.evaluate.rank_ic_by_offset` (per-horizon IC decomposition), and
  a gated `ophir train --log-offset-ic` flag that logs `val_rank_ic_h{N}`.
- Forecast-ceiling confirmation harness in `ophir.ceiling`:
  `per_offset_shuffle_null` (per-offset within-day permutation null),
  `run_offset_ic` (multi-snapshot `val_rank_ic_h*` aggregation), and
  `confirm_offset_skill` + `scripts/confirm_offset_skill.py` (multi-seed
  per-offset verdict table). Promote `_trading_day_offsets` to public
  `trading_day_offsets`.
- `val_rank_ic_near`: logged each validation pass alongside the pooled
  `val_rank_ic`; drives best-checkpoint selection when identity tensors are
  present, giving a short-horizon (leads 1–5) operating-point metric that is
  not diluted by the full 90-day horizon mix.
- `near_band_reversal_ceiling` (`ophir.ceiling`): clean near-band naive-reversal
  ceiling — mean per-lead reversal IC over leads 1–``k`` (default 5) via
  :func:`signal_decay_curve`. Rigorous comparand for a near-band model
  operating point; avoids the mixed-offset pooled-lag=1 artifact (~0.119) that
  does not isolate a 1-trading-day reversal.
- `ophir.trading.forecast.load_forecasts` now returns per-symbol offset-1
  forecasts (raw log-space `r_close`/`upside`/`downside`) from the IC-best
  checkpoint when CUDA and data are available; still degrades to `{}` otherwise.
- `ophir.ticker.build_latest_inputs` builds the most-recent `response_size=1`
  inference window per symbol.

### Changed

- Best-checkpoint filename now embeds the monitored metric
  (`val_rank_ic_near` when selecting on near-IC, else `val_loss`) instead of
  always labelling it `val_loss`.
- Normalized the three multi-target loss weights (`r_close`, `upside`,
  `downside`) by their sum so the weights control task balance only and no
  longer co-vary with total loss scale. Added a tunable `close_weight`
  (default `1.0`) to complement the existing `upside_weight` and
  `downside_weight`. Note: existing default configs see a halved loss magnitude
  (effective LR rescale); prior checkpoints and learning-rate settings are not
  directly comparable.
- `pool_prefix_embedding` now masks padding positions (non-traded days) out of
  the prefix mean-pool, so the UI PCA embedding is driven by genuine price
  history rather than zero-filled padding rows.
- `OHLCMulitClassPredictor.forward` now clamps `response_size` to
  `[1, seq_len - 1]` before use, guarding against out-of-range values that
  would produce empty prefix slices or index errors.

### Fixed

- Removed the broken `use_cache` property/setter from `LightningOHLCPredictor`;
  calling it raised `AttributeError` because the underlying predictor has no
  `ohlc_percentage_change` or `volume_percentage_change` attributes.
- `pca_projection()` was applying a redundant `.mean(1)` on stock embeddings
  already pooled by the model, collapsing the embedding dimension to a scalar
  and making the UI PCA projection degenerate. Removed the double-mean.

## [0.7.0] - 2026-06-18

### Removed

- Drop the `time_delta` input feature; the model feature set is now **12**
  columns (was 13) and `feature_mlp` is `nn.Linear(12, ...)`. Existing base
  checkpoints are architecturally incompatible and **must be retrained**.
  `time_delta` was near-binary (effectively `{0, log 3}`), overloaded `0`
  across consecutive / first / padding rows, largely redundant with the
  positional + ALiBi encoding on the padded daily calendar, and carried a
  latent `log(0) -> -inf` hazard.

### Fixed

- `extract_features` now rejects a duplicate-date index up front with a clear
  `ValueError` instead of failing deep in the calendar reindex with an opaque
  pandas error.

## [0.6.5] - 2026-06-18

### Fixed

- `ophir migrate-sqlite` crashed with `table "t_CPV" already exists` on real
  data because `sanitize_table_name` deduped table names case-sensitively while
  SQLite identifiers are case-insensitive; tickers differing only in case (e.g.
  `CPV` / `CpV`) produced distinct Python strings that collided as the same
  SQLite table. Deduplication now compares names case-insensitively. Verified
  end-to-end over all 34,700 tickers.

## [0.6.4] - 2026-06-18

### Added

- `cache_frames` option on `StockHanlder`: memoizes each symbol's loaded daily
  frame so streaming epochs skip re-reading and re-aggregating the
  parquet/SQLite source on every pass. Output-identical (split-adjustment and
  feature extraction return new frames, so the cached frame is only ever read).
  `build_split_handlers` enables it for training.

### Changed

- Raise `StockHandlerDataset`'s default `cache_size` from `1` to `8` (the
  training default), so direct instantiation mixes windows across stocks
  instead of draining one stock fully into strongly autocorrelated batches.

## [0.6.3] - 2026-06-18

### Fixed

- `get_starts` and `get_start_dates` dropped the final full window of every
  stock (a half-open `arange` stopped at `len(df) - seq_len`, exactly the last
  valid start). The freshest window — the one most wanted at inference — was
  silently lost each epoch. Stops are now inclusive (`len(df) - seq_len + 1`).
- numpy's global RNG was duplicated across DataLoader workers: the pipeline
  shuffles windows and samples the streamer cache via numpy, but `DataLoader`
  reseeds only `torch`/`random` per worker, and `ophir train` defaults
  `seed=None` (so Lightning's worker seeding never engaged). Forked workers drew
  correlated samples. `build_dataloader` now installs a `_seed_worker`
  `worker_init_fn` that seeds numpy and `random` from `torch.initial_seed`.

### Changed

- Version-proof the feature dtype filter in `extract_model_data` (`np.bool` →
  `np.dtype(bool)`) and build `StockStreamer`'s window iterator only after its
  `starts`/`offset` are computed, removing a latent ordering dependency.

## [0.6.2] - 2026-06-18

### Changed

- Switch mixed-precision training and inference from `16-mixed` (fp16) to
  `bf16-mixed` across `fetch_base_trainer`, `fetch_finetune_trainer`, and
  `predict_trainer`. bf16's wider dynamic range suits the ALiBi bias, ReZero
  scaling, and `exp` candle reconstruction better than fp16, and removes the
  fp16 gradient scaler. `run_training` now also sets
  `torch.set_float32_matmul_precision("high")` to enable TF32 for the residual
  fp32 matmuls.

## [0.6.1] - 2026-06-18

### Changed

- Refresh `uv.lock` to the revision 3 lockfile format (adds `upload-time`
  metadata); no dependency changes.

## [0.6.0] - 2026-06-18

### Added

- `loss_decay` hyperparameter on `LightningOHLCPredictor` and the matching
  `--loss-decay` flag on `ophir train` (default `0.6`; `1.0` disables the
  decay). It is saved to and restored from checkpoints, so `ophir finetune`
  inherits the value automatically.

### Changed

- The forecast loss now weights each predicted day geometrically across the
  response block, punishing nearer-term errors more than further-out ones.
  Reductions use a normalized weighted masked mean, keeping the loss scale
  comparable to the previous uniform loss. By default training is no longer
  uniform over the horizon; pass `--loss-decay 1.0` to recover the old
  behavior.

## [0.5.0] - 2026-06-18

### Added

- `ophir curate` (`ophir.curation`): scan the per-stock parquet tree and write a
  high-quality **symbol allowlist** (`<DATA_DIR>/quality-symbols.txt`) plus a
  per-symbol metrics file (`<DATA_DIR>/quality-stats.json`). Each symbol is
  scored on four dimensions — liquidity (median dollar-volume), history length &
  continuity (cleaned trading days and a business-day-denominated gap fraction),
  price sanity (penny-stock floor and split-error return spikes), and
  staleness/flatlines (identical-close runs and zero-volume days) — with
  per-criterion thresholds exposed as CLI options. `ophir train` /
  `ophir finetune` gained `--use-quality-allowlist` to restrict training to the
  allowlist via `StockHanlder.keep_stocks` (intersecting with `--use-sp500`).
- `clean_daily_ohlcv` (`ophir.ticker`): a deterministic, lookahead-free
  row-level cleaner that drops zero-volume and return-spike days from a daily
  OHLCV frame. It runs during curation (so metrics match training-time data) and
  at load time via the new `StockHanlder.clean_rows` field, exposed on
  `ophir train` / `ophir finetune` as `--clean-rows` / `--max-abs-r-close`.

## [0.4.1] - 2026-06-18

### Fixed

- `ophir serve` no longer crashes at startup when building the embedding
  scatter plot. `build_embedding_figure` ran every S&P 500 ticker through the
  model before checking its history length, so a ticker with fewer days than
  the 90-day response horizon (e.g. FISV at 51) produced a negative prefix
  slice in the response-block masking and raised a tensor-size mismatch. The
  insufficient-history guard now runs before inference, so short-history
  tickers are skipped and the dashboard launches.

## [0.4.0] - 2026-06-18

### Added

- `ophir evaluate` (`ophir.evaluate`): score a checkpoint on the held-out
  validation set and print a per-target accuracy report — MAE, RMSE, bias, plus
  directional accuracy and a zero-baseline skill score for `r_close`. It rebuilds
  the same by-date validation split as `train`, restricts predictions/targets to
  the response block and trading days (the same mask as the training loss), and
  by default reports both base checkpoints (best-`val_loss` and time-interval)
  side by side; `--finetuned` evaluates the finetuned checkpoint instead. The
  metric core is pure/CPU-safe and covered by `tests/test_evaluate.py`; a new
  "Evaluation" tab in `ophir dashboard` exposes the same report on demand.

### Changed

- The `ophir dashboard` loss plot now defaults to the per-pass aggregate
  (`*_epoch`) series with an epoch/step granularity toggle. The per-step
  validation series re-evaluates the same fixed, `limit_val_batches`-long batch
  order every pass, so plotting it produced a misleading sawtooth; the epoch
  series is the one that reflects learning.

## [0.3.0] - 2026-06-18

### Added

- In-repo training entrypoints `ophir train` (base pre-training) and
  `ophir finetune` (`ophir.train`), wiring the streaming datasets to
  `LightningOHLCPredictor` and the trainer factories — previously the only way
  to train was an unversioned off-repo driver. The train/validation split is
  **by date** with an embargo gap (`build_split_handlers`): disjoint year ranges
  separated by at least `ceil(seq_len / 365)` skipped years so no window
  straddles the boundary. Parameter guards reject invalid model dimensions up
  front (including `emb_dim // num_heads < 16`, which PyTorch flex-attention
  cannot compile on CUDA). The step budget is sized to the data by default —
  `ophir train --epochs N` derives `max_steps = N * ceil(num_windows / batch_size)`
  (so the cosine schedule anneals over the actual dataset) with `--max-steps`
  available as an explicit override. `num_windows` is *estimated* from a
  `--window-sample` of stocks rather than scanning the whole dataset, so sizing
  is fast on large corpora. Validation is **step-based** (`--val-every-steps`,
  bounded by `--val-batches`) instead of once per epoch, so `val_loss` is
  reported frequently even when an epoch spans the entire dataset; the
  best-`val_loss` checkpoint now saves whenever validation runs.
- `ophir dashboard` (`ophir.dashboard`): a standalone, import-safe live training
  dashboard with a per-target loss panel (read from the new `CSVLogger`
  `metrics.csv`, auto-refreshing on a timer) and an on-demand response-block
  leakage check against the latest checkpoint.
- `ophir.leakage`: reusable leakage scorers — `response_block_leakage_score`
  (CPU-safe, exercises only the masking helper) and `end_to_end_leakage_scores`
  (per-target, full CUDA forward). Covered by `tests/test_leakage_score.py`.
- A `CSVLogger` alongside the existing `TensorBoardLogger` in both trainer
  factories, so training metrics are written to an easily parsed `metrics.csv`.

### Changed

- `LightningOHLCPredictor` now exposes the optimizer/scheduler hyper-parameters
  (`lr`, `rezero_lr`, `weight_decay`, `betas`, `warmup_ratio`, `max_steps`) as
  constructor arguments (saved with the checkpoint). `configure_optimizers` no
  longer hardcodes a 100k-step cosine horizon — it is derived from
  `trainer.estimated_stepping_batches`, falling back to `max_steps` for the
  unsized streaming dataset. `fetch_base_trainer` gained a `max_steps` parameter
  so the trainer and schedule share one horizon. New tests in
  `tests/test_optimizer.py`.

## [0.2.0] - 2026-06-18

### Added

- `scripts/leakage_viz.py`: a Gradio app that renders input-day attribution
  heatmaps comparing the model with the response-block masking fix **off**
  (leaky) vs **on** (fixed). Leakage shows as a bright diagonal in the response
  region (forecasting a day from that same day's inputs); the fixed model
  leaves that region dark.
- Leakage regression tests: `tests/test_models_leakage.py` (CPU, pins the
  masking helper) and `tests/test_models_leakage_realdata.py` (real-data
  end-to-end through the GPU forward, auto-skipped without CUDA/data/checkpoint).

### Fixed

- **Data leakage:** the model was fed the values it was asked to forecast.
  `feature_input` carried the response-block days' features (`r_close` /
  `upside` / `downside` and the rolling features derived from them), which are
  exactly the prediction targets, so the task was solvable by identity and all
  losses/UI predictions were reading the answer. `OHLCMulitClassPredictor` now
  replaces the response block with a learned `mask_token` before the
  transformer (`_apply_response_mask`), forcing a genuine forecast from the
  prefix. Existing checkpoints are invalidated and must be retrained.

### Removed

- The unused `winsorize_returns` flag on `extract_features` / `StockHanlder`,
  which clipped `r_close` to full-series (future-inclusive) quantiles — a
  lookahead foot-gun that was never wired into the streaming path.

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

[Unreleased]: https://github.com/kwcantrell/ophir/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/kwcantrell/ophir/compare/v0.6.5...v0.7.0
[0.6.5]: https://github.com/kwcantrell/ophir/compare/v0.6.4...v0.6.5
[0.6.4]: https://github.com/kwcantrell/ophir/compare/v0.6.3...v0.6.4
[0.6.3]: https://github.com/kwcantrell/ophir/compare/v0.6.2...v0.6.3
[0.6.2]: https://github.com/kwcantrell/ophir/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/kwcantrell/ophir/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/kwcantrell/ophir/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/kwcantrell/ophir/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/kwcantrell/ophir/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/kwcantrell/ophir/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/kwcantrell/ophir/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/kwcantrell/ophir/compare/v0.1.7...v0.2.0
[0.1.7]: https://github.com/kwcantrell/ophir/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/kwcantrell/ophir/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/kwcantrell/ophir/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/kwcantrell/ophir/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/kwcantrell/ophir/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/kwcantrell/ophir/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/kwcantrell/ophir/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/kwcantrell/ophir/releases/tag/v0.1.0
