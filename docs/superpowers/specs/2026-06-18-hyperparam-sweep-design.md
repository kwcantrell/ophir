# Hyperparameter sweep harness — design

**Date:** 2026-06-18
**Status:** Approved (design); implementation pending

## Goal

Add reusable, in-repo infrastructure that runs many training trials and
automatically finds good hyperparameter configurations for the Ophir model. The
harness must score trials on a metric aligned with the end goal (stock-ranking
skill), survive crashes on hours-long single-GPU runs, and leave the existing
training path byte-for-byte unchanged when not in use.

## Background / current state

- All hyperparameters live as plain Python defaults on `train.train()` in
  `src/ophir/train.py`, exposed as CLI flags via Typer (`ophir train`). There is
  **no** sweep, config, or experiment-tracking infrastructure today.
- Optimizer/schedule/loss hparams are constructor defaults on
  `LightningOHLCPredictor.__init__` (`training_models.py:46-58`):
  `lr=2e-4`, `rezero_lr=3e-4`, `weight_decay=0.01`, `betas=(0.9, 0.95)`,
  `warmup_ratio=0.03`, `max_steps=100000`, `loss_decay=0.6`.
- Loss combination weights are a hardcoded literal:
  `close_loss + 0.5*upside_loss + 0.5*downside_loss` (`training_models.py:293`).
- Architecture params (`emb_dim`, `num_layers`, `num_heads`) are required fields
  on the frozen `OHLCMulitClassParameters` dataclass (`models.py:47-73`), with
  constraints `emb_dim % 4 == 0` and `emb_dim % num_heads == 0`.
- `train()` forwards only `lr/weight_decay/warmup_ratio/max_steps/loss_decay` to
  the model — `rezero_lr`, `betas`, and the loss weights are **not** currently
  reachable from the training entry point.
- The eval report already computes cross-sectional rank-IC for `r_close`
  (commits c62e2bd / 618697d); a `rank_ic` pure helper exists in `evaluate.py`.
- Runtime needs a CUDA GPU; CPU runs do not work. The actual `fit` therefore
  cannot run in CI.

## Approach

Optuna-based harness (chosen over a hand-rolled random/grid loop and over Ray
Tune / W&B Sweeps). Optuna gives TPE Bayesian search, built-in ASHA pruning, and
SQLite-backed study persistence/resumability with a single lightweight
dependency that ships type hints. Ray/W&B are overkill for a single GPU; a
hand-rolled loop would reimplement pruning + persistence + resumability that
Optuna already provides.

## Objective

Each trial is scored on **mean cross-sectional rank-IC on `r_close`** over the
validation set, maximized. `val_loss` and the upside/downside/close skill scores
are logged as **secondary** metrics for inspection but are not optimized —
`val_loss` is a poor proxy for ranking skill (the robust smooth-L1 loss can
reward low-variance, near-mean predictions that have near-zero IC).

## Components

### 1. `val_rank_ic` validation metric (plumbing)

The Lightning wrapper currently logs only `val_loss`. Add an always-on
`val_rank_ic` metric computed from validation predictions via the existing
`rank_ic` helper, logged each validation pass. It serves two purposes: a useful
standalone secondary metric, and the signal Optuna's
`PyTorchLightningPruningCallback` monitors to prune lagging trials early. Keep it
cheap (a correlation over predictions already gathered during validation).

### 2. Exposed hyperparameters (plumbing)

Make three currently-buried params reachable from `train()`, **all defaulting to
their current values** so a normal `ophir train` is byte-for-byte unchanged:

- `rezero_lr`, `betas` — already constructor args on `LightningOHLCPredictor`;
  add to the `train()` signature and forward them.
- `upside_weight`, `downside_weight` — new constructor args on
  `LightningOHLCPredictor` (defaults `0.5`, `0.5`), used in `compute_loss` in
  place of the hardcoded literals at `training_models.py:293`; add to `train()`
  and forward.

### 3. `src/ophir/sweep.py` — the harness

- **Study:** Optuna study with TPE sampler, SQLite storage under `.ophir/sweep/`,
  `load_if_exists=True` so re-running the same study name resumes.
- **Pruner:** ASHA (`SuccessiveHalvingPruner`) reporting `val_rank_ic` at each
  validation interval.
- **`objective(trial)`:** samples the search space, maps it to `train()` kwargs,
  runs a **proxy** training (reduced budget) with the Optuna pruning callback,
  and returns the best `val_rank_ic`. Per-trial seed = `base_seed + trial.number`.
- **Confirm phase:** pull the top-K trials (default 5) from the study, retrain
  each at **full budget**, run `evaluate.py` for the authoritative rank-IC +
  skill scores, and emit a ranked results table.

### 4. CLI — `ophir sweep`

`ophir sweep [--trials N] [--study NAME] [--storage PATH] [--confirm-top K]
[--proxy-steps …]`, wired into the Typer app in `cli.py`. Re-running the same
`--study` resumes the existing SQLite study.

## Search space (v1)

- **Optimizer/schedule (core):** `lr` (log-uniform), `rezero_lr` (log-uniform),
  `weight_decay` (log-uniform), `warmup_ratio`, `loss_decay`, `betas[1]`.
- **Loss weighting:** `upside_weight`, `downside_weight`.
- **Architecture:** a small **categorical** over a few `(emb_dim, num_layers,
  num_heads)` size tiers (e.g. small / base / large) rather than free-form —
  keeps memory bounded and trials comparable, and each tier satisfies the
  `emb_dim % 4 == 0` / `emb_dim % num_heads == 0` constraints by construction.

### Fixed for v1 (not swept)

`seq_len`, `offset`, `response_size`, `batch_size`, rolling windows (10/20/60).
Changing these alters the data pipeline and the eval comparison basis (most
plumbing risk, least clear payoff). Holding them fixed also keeps the validation
slice identical across trials, so `val_rank_ic` is comparable trial-to-trial.
Sweeping windowing is a deliberate follow-up, out of scope here.

## Strategy: proxy → confirm

- **Phase 1 (search):** proxy training per trial — reduced budget via existing
  `train()` knobs (fewer steps; optionally a fixed smaller ticker universe).
  Trials report `val_rank_ic` at each validation interval; ASHA prunes laggards.
- **Phase 2 (confirm):** top-K configs retrained at full budget, scored on the
  real `evaluate.py` rank-IC + skill scores, producing the final ranked table.

This buys cheap signal during search and pays full training cost only on the
survivors, rather than trusting a short proxy to pick the winner outright.

## Compute posture

Designed for the moderate single-GPU regime but scalable: `--trials` is a knob
(works at ~20 or ~200+). Pruning + the proxy phase keep per-trial cost low;
SQLite persistence makes the study resumable if a run is killed mid-sweep.

## Testing

Per repo convention (deterministic, seeded, CPU-safe, network-free). The actual
`fit` needs CUDA and is mocked; the pure pieces are tested directly:

- search-space sampling: given a fake/`FixedTrial`, returns a valid config
  honoring the head-divisibility constraints;
- config → `train()`-kwargs mapping;
- `val_rank_ic` computation on synthetic predictions;
- top-K selection from an in-memory Optuna study with injected trial values.

## Dependency & typing

- Add `optuna` to project dependencies.
- Strict mypy + `warn_unused_ignores` is in force. Optuna ships type hints; if
  mypy still flags it, add `optuna` to the `ignore_missing_imports` override list
  alongside `massive`/`plotly`/`tqdm`/`yfinance`.

## Non-goals (v1)

- Sweeping data/windowing params.
- Distributed / multi-GPU parallel trials.
- A web dashboard (TensorBoard/CSV logging already exists; Optuna's study DB can
  be inspected separately).

## Backward-compatibility contract

A default `ophir train` must remain byte-for-byte identical: every newly exposed
param defaults to its current value, the new `val_rank_ic` metric is additive,
and the sweep lives in a separate module + CLI command. This mirrors the repo's
established opt-in convention for pipeline additions.
