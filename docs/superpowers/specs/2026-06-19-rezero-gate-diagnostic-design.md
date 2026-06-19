# Design: ReZero gate-opening diagnostic — is the transformer depth helping?

**Date:** 2026-06-19
**Status:** Approved (design); pending implementation plan

## Background

A clean Optuna control study (`downside-control-v2`, RandomSampler + NopPruner,
80 trials, normalized loss) found `rezero_lr` overwhelmingly dominant on the
`val_rank_ic` objective (fANOVA 0.60 / MDI 0.39), far above every other knob.

Inspecting trained checkpoints explained why: the ReZero gate scalars
(`TransformerBlock._rezero`, one per block, initialized to `0.0`) are **stuck
near zero** — magnitudes ~0.013–0.021 with mixed signs across base (6-block) and
large (8-block) checkpoints. Each block contributes only ~2% of its sublayer
signal, so the model operates close to the identity/skip path (near-linear), and
`rezero_lr` dominates because it is the bottleneck controlling how much of the
network is online at all. Best proxy `val_rank_ic` was only ~0.044, consistent
with an under-used-depth, near-linear model.

This is expected ReZero *mechanism* (gates gate performance), not a correctness
bug — the implementation is sound (correct AdamW group, `weight_decay=0` on the
gates, no numerical issue). But gates frozen near 0 across budgets is a real
signal that the depth may not be earning its keep. Two non-standard choices are
plausible contributors: the rezero group rides the cosine warmup→decay schedule
(narrow window to grow), and one shared scalar gates both attention and MLP.

## Goal

Determine whether the transformer depth actually contributes to predictive
performance, by (1) instrumenting the gates so we can see them open, (2) adding
opt-in architecture knobs to *make* them open, and (3) running a fair-budget
comparison matrix against a near-linear baseline.

## Non-goals

- No change to default training behavior. Every new knob defaults to the current
  behavior (byte-for-byte unchanged training when unset).
- No change to the forecast-masking contract (`tests/test_models_leakage.py`).
- We build and unit-test the reusable code (instrumentation + knobs + analysis
  helper). The GPU experiment runs are the user's to launch; we supply exact
  commands and an analysis helper.

## Part 1 — Gate instrumentation

Add a pure, CPU-testable helper in `src/ophir/models.py`:

```python
def rezero_gate_stats(model: nn.Module) -> dict[str, float | list[float]]:
    """Per-layer and aggregate magnitudes of the ReZero gate scalars."""
```

It iterates `model.named_parameters()` for names containing `"rezero"`, returning
`{"mean_abs": float, "max_abs": float, "per_layer": [float, ...]}` (empty/zeroed
safely when there are none).

`LightningOHLCPredictor` gains an opt-in flag `log_rezero_gates: bool = False`.
When set, `on_validation_epoch_end` logs `rezero_mean_abs` and `rezero_max_abs`
(scalars, via `self.log`) so each arm records whether its gates opened. Logging
extra scalars does not change the training computation or RNG; the flag keeps the
default CSV columns unchanged.

## Part 2 — Opt-in architecture knobs (default = current behavior)

### 2a. `rezero_init: float = 0.0`

- Added to `OHLCMulitClassParameters` (as a defaulted field, after the existing
  fields; `__post_init__` asserts unchanged).
- `TransformerBlock.__init__` uses `nn.Parameter(torch.tensor(hparams.rezero_init,
  dtype=torch.float))` instead of the hardcoded `0.0`.
- `LightningOHLCPredictor` stores `rezero_init` and passes it into
  `OHLCMulitClassParameters(...)`. `reset_rezero` fills with `rezero_init` (not a
  hardcoded `0`) so re-zero semantics stay consistent; default `0.0` keeps current
  behavior.

### 2b. `decouple_rezero_schedule: bool = False`

- `LightningOHLCPredictor.configure_optimizers`: when `False` (default), keep the
  current single-lambda `get_cosine_schedule_with_warmup` over all three param
  groups — unchanged. When `True`, build a `torch.optim.lr_scheduler.LambdaLR`
  with a per-group lambda list `[cosine, cosine, flat]`, where the rezero group
  (index 2) uses a **flat** schedule (linear warmup then constant `1.0`, no cosine
  decay) and the decayed/no-decay groups keep the cosine lambda.
- The two lambdas are module-level pure functions (replicating the transformers
  cosine formula for the decayed groups) so they are CPU-testable without
  training.

Both knobs thread through `run_training` and `train` exactly like `close_weight`
did (defaults preserve behavior), plus the `--log-rezero-gates` flag.

## Part 3 — The experiment matrix

All arms run at a **fixed fair budget of 10,000 steps** (the proxy's 2,000 was
too short for gates to open), same `--seed 0`, same date split, base tier
(`emb_dim=128`, `num_heads=8`), with `--val-identity` and `--log-rezero-gates`:

| Arm | Config (deltas from base defaults) | Question |
| --- | --- | --- |
| A. Shallow | `--num-layers 1` | Near-linear floor |
| B. Deep/default | `--num-layers 6` | Confirms gates stay closed ⇒ deep ≈ shallow |
| C. Deep/high-LR | `--num-layers 6 --rezero-lr 3e-3` | Do gates open via LR alone? |
| D. Deep/init | `--num-layers 6 --rezero-init 0.1` | Gates start clearly open (well above the observed ~0.02 plateau) |
| E. Deep/un-decayed | `--num-layers 6 --decouple-rezero-schedule` | Gates keep growing (no cosine decay) |

Single-variable arms keep the diagnosis clean.

### Decision criterion

- **Primary:** does any deep arm (C/D/E) beat the shallow floor (A) on final
  `val_rank_ic` by a meaningful margin?
  - Yes ⇒ depth helps once the gates open → pursue the gate-opening fix.
  - No ⇒ depth is not earning its keep on this data → stop investing in depth.
- **Secondary (from instrumentation):** gate magnitudes per arm separate "gates
  opened but didn't help" (depth genuinely inert) from "gates never opened" (an
  optimization problem we can fix). Arm B is expected to confirm closed gates ≈
  shallow.

## Part 4 — Orchestration & analysis

- The five runs are GPU work the user launches; the plan supplies the exact
  `ophir train` command per arm (each creates a new CSVLogger `version_N`; record
  the arm→version mapping).
- Add a pure, CPU-testable helper in `src/ophir/dashboard.py` (the module that
  already reads the `CSVLogger`):

```python
def summarize_rezero_runs(versions: dict[str, str]) -> "pandas.DataFrame":
    """Tabulate final val_rank_ic and final rezero gate magnitudes per arm.

    ``versions`` maps an arm label to its CSVLogger version directory; reads each
    ``metrics.csv`` and returns one row per arm.
    """
```

(pandas import stays lazy/local, consistent with the module.)

## Testing (CPU-safe, seeded, network-free — per repo convention)

- `rezero_gate_stats` on a constructed `OHLCMulitClassPredictor` (CPU): correct
  per-layer values, mean/max aggregation, and empty-safe behavior.
- `rezero_init` actually sets `_rezero` to the configured value across all blocks
  (CPU model construction); `reset_rezero` restores that value.
- Decoupled scheduler: build the optimizer + `LambdaLR` on a tiny CPU module,
  step it across warmup and decay phases, assert the rezero group's LR stays flat
  (no decay) while the other groups follow cosine — and that with the flag off the
  schedule is byte-for-byte the current cosine for all groups.
- `summarize_rezero_runs` against tiny synthetic `metrics.csv` fixtures (no
  training): correct final-row extraction per arm.
- `run_training` forwards `rezero_init` / `decouple_rezero_schedule` /
  `log_rezero_gates` to the model (the `close_weight`-style forwarding test).

## Files touched

- `src/ophir/models.py` — `rezero_gate_stats`; `rezero_init` on
  `OHLCMulitClassParameters` and `TransformerBlock`.
- `src/ophir/training_models.py` — `rezero_init` / `decouple_rezero_schedule` /
  `log_rezero_gates` fields; gate logging in `on_validation_epoch_end`; per-group
  scheduler in `configure_optimizers`; `reset_rezero` fills with `rezero_init`.
- `src/ophir/train.py` — thread the three new knobs through `run_training` and
  `train` (CLI flags `--rezero-init`, `--decouple-rezero-schedule`,
  `--log-rezero-gates`).
- `src/ophir/dashboard.py` — `summarize_rezero_runs`.
- `tests/` — `test_models_output.py` / `test_models_*`, `test_training_models.py`,
  `test_train.py`, `test_dashboard.py`.
- `CHANGELOG.md` — new diagnostic knobs.

## Risks / call-outs

- The diagnostic's validity hinges on a fair budget; 10k steps is a judgment call
  (between the 2k proxy and 20k full). If even arm E's gates stay near 0 at 10k,
  re-run at a longer budget before concluding depth is inert.
- `decouple_rezero_schedule` replaces the transformers helper with a hand-rolled
  per-group `LambdaLR`; the off-path must remain identical, pinned by a test.
- `rezero_init=0.1` is chosen to start clearly above the observed ~0.02 plateau;
  it is an experiment value, not a new default.
