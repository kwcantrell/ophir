# Forecast-ceiling operating-point fix — design

**Date:** 2026-06-23
**Branch:** `forecast-ceiling-operating-point-fix`
**Status:** approved design; implementation plan to follow.

## Background

The forecasting ceiling is a **task-framing** problem, not an optimizer or
architecture one. Confirmed multi-seed (2026-06-23, 3 seeds): the model carries
genuine near-horizon cross-sectional skill at trading-day **offsets 1–5, pooled
IC ≈ +0.066** (conservative best-`val_loss` checkpoints; true peak likely
higher), clearing its 0.036 null on all three seeds. The pooled 90-day
`val_rank_ic` dilutes that to ~0.012–0.02 — the predicted 3–5×. Skill
concentrates in offsets 1–5 and dilutes below significance by offset 10.

The fix is an **operating-point / metric change**: collapse to a short horizon.
No new architecture. Full evidence: `docs/forecast-ceiling-results.md`
(confirmation section) and the `forecast-ceiling-investigation` memory.

## Scope

This spec covers **A + B**. Component C (wiring `trading/forecast.py` to offset-1
inference) is **deferred to its own spec** once B settles which model the seam
should read.

- **A.** Productionize a near-band validation metric and flip the
  best-checkpoint monitor to it.
- **B.** Decide the operating point (read-near vs short-horizon retrain) by a
  multi-seed experiment, benchmarked against a freshly recomputed clean
  near-band reversal ceiling.

## Component A — near-band metric + checkpoint win

The validation buffers (`pred`/`target`/`ids`/`dates`/`offsets`) are already
accumulated in `validation_step` whenever the val loader carries identity
(`training_models.py:456-474`), independent of `--log-offset-ic` (that flag only
gates the per-offset `h*` *logging*). So the near metric is free to compute.

### A1. `val_rank_ic_near` helper

Add to `training_models.py`, mirroring `val_rank_ic` (line 65):

```
val_rank_ic_near(pred, target, ids, dates, offsets, k=5) -> float
```

Selects rows with `1 <= offset <= k`, then reuses
`evaluate.dedupe_by_ticker_date` + `evaluate.rank_ic` so the live metric and the
offline report share identical math. Returns `nan` for empty input (no rows in
band / no identity collected).

### A2. Log it ungated

In `on_validation_epoch_end` (`training_models.py:478`), whenever `preds` exist
(identity present), compute and log `val_rank_ic_near` **unconditionally** — not
behind `--log-offset-ic`. The per-offset `h*` logging stays gated exactly as it
is today. The `offsets` buffer is already populated, so no new accumulation is
needed.

### A3. Near band size `K`

Expose as a constructor kwarg `near_offset_k: int = 5` on
`LightningOHLCPredictor` (default = the validated band). No new CLI flag: the
logged metric name (`val_rank_ic_near`) is independent of `K`, so nothing
downstream needs to know it. A flag can be added later if tuning `K` ever
becomes interesting (YAGNI for now).

### A4. Checkpoint monitor

`register.py:111` currently configures the best-checkpoint `ModelCheckpoint` with
`monitor="val_loss"` (default `mode="min"`). `val_loss` is anti-aligned with IC
(IC peaks mid-run then droops as the cosine LR anneals; `val_loss`-best ≈ drooped
IC, ~½ peak), so switching the monitor recovers roughly 2× for free.

IC only exists when the val loader carries identity, so the switch is **gated on
that**, decided statically at trainer-construction time:

- The Trainer factory in `register.py` gains a boolean parameter (e.g.
  `monitor_near_ic: bool`).
- When `True`: best-checkpoint callback uses
  `monitor="val_rank_ic_near", mode="max"`.
- When `False`: keeps `monitor="val_loss"` (default `mode="min"`).
- `train.py` derives the flag from `--val-identity` (identity is a precondition
  for IC) and passes it through.

No custom callback is needed — the decision is static per run. There is no
`EarlyStopping` callback today, so nothing else changes. The time-interval
checkpoint (`register.py:102`) is untouched.

## Component B — settle the operating point by experiment

### B1. Runs

Train a short-horizon model and compare its near-band skill to the existing
90-day model's near slice.

- `response_size ≈ 10` **calendar** days. The response block is calendar-dense
  (`ticker.py:398`, `pd.date_range(freq="D")`); ~10 calendar days covers
  trading-day leads 1–5 (≈ 7–10 calendar days). The skill is at trading-day
  offsets, so the calendar `response_size` must be chosen to cover them.
- Seeds `{0, 1, 2}` — the same seed-stability bar the confirmation cleared.
- Same proxy config as the confirmation:
  `--emb-dim 128 --num-heads 8 --num-layers 6 --max-steps 10000 --val-identity
  --log-offset-ic --val-batches 200`.
- Eval: reuse the `.superpowers/sdd/gpu/` harvest + pooled-near eval patterns to
  compute pooled near-IC (offsets 1–5) per seed.

`response_size` is a free parameter end-to-end (validated only
`1 <= response_size <= seq_len-1`; masks and reconstruction are dynamic). The
short-horizon training/eval scripts stay **gitignored scratch** (reuse the
confirmation harness); only the ceiling helper below is productionized.

### B2. Clean near-band reversal ceiling (productionized + tested)

Add `near_band_reversal_ceiling` to `ophir.ceiling`: the mean clean per-lead
1-trading-day reversal IC over leads 1–5 (E3 Step A's rigorous per-lead method).

This **replaces** the inflated ~0.119 figure printed during the confirmation,
which used `lagged_target_signal(lag=1)` on *mixed-offset* pooled rows — so the
lag-1 was not a clean 1-trading-day reversal and the number is unreliable.
Because this ceiling is the comparand we cite, it is productionized with a CPU
test rather than left in scratch.

### B3. Decision rule

Adopt the short-horizon **retrain** only if its pooled near-IC (offsets 1–5)
beats the 90-day **near-slice** (≈ 0.066) by a **seed-stable margin** — a clearly
higher mean across all three seeds. Otherwise default to **read-near**: operate
on offset-1 of the existing 90-day model (which already emits per-response-day
forecasts).

Independently, report the chosen operating point against B2's clean ceiling:

- model **>** clean near-band reversal ceiling ⇒ the operating-point fix is the
  whole story.
- model **≤** ceiling ⇒ architectural headroom remains (the response-block
  masking denies the model the 1-day feature, `models.py:434-460`); record that
  for a future architecture spec.

### B4. Outputs

Record the runs, the chosen operating point, and the ceiling comparison in
`docs/forecast-ceiling-results.md`; update the `forecast-ceiling-investigation`
memory.

## Testing

CPU-only, offline, deterministic (`filterwarnings=error`, mypy strict,
NumPy-style docstrings):

- `val_rank_ic_near`: perfect ranking → positive; empty input → `nan`;
  offset-filter correctness (rows outside `1..K` excluded); `K` boundary
  (offset == K included, K+1 excluded).
- `near_band_reversal_ceiling`: a synthetic reversal signal yields the expected
  sign/magnitude; clean per-lead behaviour.
- `register` Trainer factory: asserts the best-checkpoint `monitor`/`mode` flips
  correctly on the boolean (`val_rank_ic_near`/`max` when set,
  `val_loss`/`min` when not).

The B training and eval runs are **manual GPU**, not part of the pytest suite.

## Constraints

mypy strict / Python 3.10 floor; ruff 3.12; NumPy-style docstrings; reuse the
production rank-IC math (`cross_sectional_ic` → `dedupe_by_ticker_date` +
`rank_ic`) so offline and live metrics agree. pytest stays offline + CPU-only.
The safety gate (`trading/safety.py`) is non-overridable; the system is
paper-only (`account_mode`). C is deferred, so this spec touches no trading code.

## Pointers

- Results log: `docs/forecast-ceiling-results.md` (E0/E1/E3 + 2026-06-23
  confirmation).
- Implementation handoff: `docs/forecast-ceiling-fix-implementation-context.md`.
- Memory: `forecast-ceiling-investigation`, `sweep-importance-findings`,
  `dev-workflow-preference`.
