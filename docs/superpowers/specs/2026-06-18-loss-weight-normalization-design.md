# Design: loss-weight normalization, sweep importance diagnostics, and minor robustness fixes

**Date:** 2026-06-18
**Status:** Approved (design); pending implementation plan

## Background

An adversarial review investigated the claim that `pool_prefix_embedding`
(`src/ophir/models.py`) was causing `downside_weight` to dominate the Optuna
sweep (fANOVA importance > 0.85 on the `val_rank_ic` objective). Two independent
reviewers (bug-hunter and devil's advocate) converged on the same conclusion:

- **The hypothesis is mechanically impossible.** `pool_prefix_embedding`'s only
  output is `stock_embeddings`, consumed exclusively by inference/UI paths
  (`ui.py:200`, `model_data.py` PCA helper under `torch.no_grad()`). It is never
  read by `compute_loss` or the `val_rank_ic` objective, so it has no gradient
  path into training and cannot mediate `downside_weight`'s effect.
- **The real driver is a loss-weight design issue.** In the combined loss
  (`training_models.py:330`):
  `close_loss + upside_weight*upside_loss + downside_weight*downside_loss`,
  `close_loss` carries a fixed implicit weight of `1.0` while `upside_weight`
  and `downside_weight` are free in `[0.25, 1.0]`. This is both an *asymmetry*
  (close has no knob) and a *scale coupling* (total loss magnitude co-varies
  with the weights, so a weight knob doubles as an effective-LR knob). On a
  shared trunk feeding a 3-wide head, `downside` is the fattest-tailed channel
  (largest `robust_scale` beta), so its weight is the highest-leverage free
  scalar — plausibly the real reason it dominates.
- **The exact > 0.85 magnitude is partly an artifact.** fANOVA is computed only
  over *completed* trials, but the sweep uses `TPESampler` +
  `SuccessiveHalvingPruner` (`sweep.py:120-121`). The surviving set is
  selection-biased and non-i.i.d., which biases fANOVA. The ranking is
  believable; the precise number is not.

This design addresses the two substantive follow-ups plus two minor robustness
issues surfaced during the review.

## Goals

1. **Confirm the cause cleanly** — make the sweep able to run a control study
   (RandomSampler, no pruner) and report trustworthy importances, so the
   `downside_weight` effect can be verified rather than assumed.
2. **Fix the loss-weight asymmetry and scale coupling** — normalize the three
   weights so the knobs control task balance only, not total loss scale.
3. **Fix two minor robustness issues** in `pool_prefix_embedding` and the
   model's `response_size` handling.

## Non-goals

- No change to the forecast-masking contract (`_apply_response_mask`) — the
  leakage pin in `tests/test_models_leakage.py` must still pass.
- No change to the model architecture, the `robust_scale` beta, or the
  per-channel smooth-L1 losses themselves.
- No retraining or re-running of sweeps as part of this work (CUDA-only; left to
  the user). We build the harness; the user runs it.

## Part A — Normalize the three loss weights

### Module change (`training_models.py`)

Add `close_weight: float = 1.0` to `LightningOHLCPredictor.__init__`, stored as
`self.close_weight` and included in `save_hyperparameters()`. Change the combined
return of `compute_loss` (currently `training_models.py:330`) to a
weight-normalized mean:

```python
total_w = self.close_weight + self.upside_weight + self.downside_weight
return (
    self.close_weight * close_loss
    + self.upside_weight * upside_loss
    + self.downside_weight * downside_loss
) / max(total_w, 1e-8)
```

The per-channel losses, their logging, masking, and time-decay weighting are
unchanged. Only the final combination changes.

**Property:** total-loss scale is now invariant to a uniform rescaling of the
weights — `(k*a, k*b, k*c)` yields the same loss as `(a, b, c)`. The weights
therefore express *only* the relative task balance, removing the effective-LR
coupling that inflated `downside_weight`'s leverage.

### Threading

`close_weight` is added (default `1.0`) to both `run_training` entry points
(`train.py:348` and `train.py:482`) and passed to `LightningOHLCPredictor`,
mirroring the existing `upside_weight` / `downside_weight` plumbing.

### Sweep search space (`sweep.py:sample_config`)

`close_weight` is a real, tunable model knob at the API level. In the **sweep**,
however, it is **anchored to `1.0`** to avoid the redundant scale dimension that
normalization introduces (under normalization `(0.5,0.5,0.5) == (1,1,1)`, so
sampling all three freely wastes search budget). Concretely, `sample_config`
returns `"close_weight": 1.0` (fixed, not a `suggest_float`) and continues to
sample `upside_weight` and `downside_weight` in `[0.25, 1.0]`. This yields the
same expressiveness as three free weights post-normalization, with no wasted
search dimension.

### Backward-compatibility note

With defaults `(close=1.0, upside=0.5, downside=0.5)` the divisor is `2.0`, so
the default loss magnitude *halves* relative to the current code. This
effectively rescales the LR for existing configs. This is the intended
decoupling, but it means prior checkpoints and tuned LRs are not directly
comparable across this change. Documented in the CHANGELOG.

## Part B — Sweep importance diagnostics

### `run_sweep` parametrization (`sweep.py`)

Add two parameters: `sampler: str = "tpe"` and `prune: bool = True`.

- `sampler="tpe"` → `optuna.samplers.TPESampler(seed=base_seed)` (current
  behavior); `sampler="random"` → `optuna.samplers.RandomSampler(seed=base_seed)`.
- `prune=True` → `optuna.pruners.SuccessiveHalvingPruner()` (current behavior);
  `prune=False` → `optuna.pruners.NopPruner()`.

Defaults preserve today's behavior exactly. A small private builder
(e.g. `_build_sampler(name, seed)` / `_build_pruner(prune)`) keeps the
construction testable.

### CLI `sweep` options (`cli.py`)

Add `--sampler [tpe|random]` (default `tpe`) and `--no-prune` (flag, default
pruning on), threaded into `run_sweep`.

### Importances helper (`sweep.py`)

`compute_importances(study) -> dict` returns:

```python
{"fanova": {param: importance, ...},
 "mdi":    {param: importance, ...},
 "n_completed": int}
```

using Optuna's `FanovaImportanceEvaluator` and `MeanDecreaseImpurityImportanceEvaluator`
over completed trials. Returns empty importance dicts (and `n_completed`) when
too few trials completed to evaluate, rather than raising.

A pure formatting helper `format_importances(result, *, sampler, pruned) -> str`
renders both rankings and emits a **warning line** when `sampler == "tpe"` or
`pruned is True` (fANOVA over a TPE/ASHA-filtered set is biased) or when
`n_completed` is small. This helper takes plain dicts/values so it is CPU-safe
and testable without an Optuna run.

### CLI `importances` command (`cli.py`)

`ophir importances <study> [--storage ...]` loads the study (same storage
defaulting as `sweep`), calls `compute_importances`, infers whether the study
used a non-random sampler / pruning where determinable, and prints
`format_importances(...)`.

### Resulting workflow

```
ophir sweep --sampler random --no-prune --trials 80 ...   # clean control study
ophir importances <study>                                  # trustworthy read
```

## Part C — Minor robustness fixes

### C1 — Padding-masked prefix pool (`models.py`)

`pool_prefix_embedding` currently means over all prefix positions with no
padding mask (`x[:, :-response_size].mean(dim=1)`), folding no-trade/padded rows
(which are non-zero after `feature_mlp`/`pe`/MLPs) into the per-stock embedding.
This only affects the UI/PCA embedding (not training), but is still wrong for
short-history tickers.

New signature: `pool_prefix_embedding(x, response_size, trade_occured)`.
Compute a masked mean over the prefix using `trade_occured[:, :prefix_len]`:

```python
prefix = x[:, :-response_size]
m = trade_occured[:, : x.shape[1] - response_size].unsqueeze(-1).to(prefix.dtype)
denom = m.sum(dim=1)
masked = (prefix * m).sum(dim=1) / denom.clamp_min(1.0)
# rows with no valid prefix positions fall back to the unmasked mean
no_valid = (denom.squeeze(-1) == 0)
masked[no_valid] = prefix[no_valid].mean(dim=1)
return masked
```

Update the caller (`models.py:483`) to pass `input.trade_occured`, and update
`tests/test_models_output.py::test_pool_prefix_embedding_ignores_response_block`
for the new signature.

### C2 — `response_size` guard (`models.py`)

Validate `0 < response_size < seq_len` at the **very top** of
`OHLCMulitClassPredictor.forward` (reading `seq_len` from
`input.feature_input.shape[1]`), raising a clear `ValueError` otherwise. Placing
the check before the feature projection and any flex-attention call means it
fires on CPU, so it is unit-testable without a GPU. This protects both the
response slice (`x[:, -response_size:]`) and the prefix pool from the
empty-slice / NaN edge when `response_size == 0` or `>= seq_len`.

## Testing (CPU-safe, seeded, network-free — per repo convention)

- **Loss normalization** (`tests/test_training_models.py`): using the existing
  fake-forward-output pattern, assert (a) **scale invariance** — multiplying all
  three weights by `k` leaves the combined loss unchanged; (b) correct relative
  weighting for a known set of per-channel losses; (c) defaults `(1.0,0.5,0.5)`
  produce the documented normalized value.
- **Masked pool** (`tests/test_models_output.py`): padded prefix positions are
  excluded from the mean; an all-padded row falls back to the unmasked mean
  without NaN; the existing "ignores response block" property still holds.
- **`response_size` guard** (`tests/test_models_*` or `test_training_models`):
  `ValueError` on `response_size >= seq_len` and `response_size == 0` (tested via
  a CPU-constructable path or a direct call to the guarded helper).
- **Importances** (`tests/test_sweep.py`): `format_importances` rendered from a
  fake result dict — asserts the warning line appears for `sampler="tpe"` /
  `pruned=True` and is absent for a random/un-pruned study; `_build_sampler` /
  `_build_pruner` return the correct Optuna types.
- **`sample_config`** (`tests/test_sweep.py`): updated to assert
  `config["close_weight"] == 1.0` (anchored) and that `upside_weight` /
  `downside_weight` remain sampled.

## Files touched

- `src/ophir/training_models.py` — `close_weight` field + normalized combine.
- `src/ophir/train.py` — thread `close_weight` through both entry points.
- `src/ophir/sweep.py` — anchored `close_weight` in `sample_config`;
  `sampler`/`prune` params on `run_sweep`; `compute_importances` +
  `format_importances` + builders.
- `src/ophir/models.py` — masked `pool_prefix_embedding`; `response_size` guard.
- `src/ophir/cli.py` — `--sampler` / `--no-prune` on `sweep`; new `importances`
  command.
- `tests/` — `test_training_models.py`, `test_models_output.py`,
  `test_sweep.py` (and a guard test).
- `CHANGELOG.md` — loss-scale change note + new diagnostics.

## Risks / call-outs

- **Loss-scale shift** changes effective LR for existing default configs (see
  Part A back-compat note). Intentional; documented.
- The masked-pool signature change is a breaking change to a public helper, but
  its only caller and only test are updated in the same change.
- Importance evaluators depend on Optuna internals; `compute_importances` must
  degrade gracefully (empty dicts) when too few trials completed.
