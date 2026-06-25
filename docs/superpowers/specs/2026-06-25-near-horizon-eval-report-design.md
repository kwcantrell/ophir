# Near-horizon operating point in the eval report — design

**Date:** 2026-06-25
**Branch:** `near-horizon-eval-report`
**Status:** approved design; implementation plan to follow.

## Background

Surfaced by a graphify pass over the repo (community 19 "Ceiling Signal Decay",
god node `OHLCMultiClassPredictorInput`). The forecasting ceiling is a
**task-framing** problem already diagnosed: the model carries genuine
near-horizon cross-sectional skill at trading-day offsets 1–5
(pooled IC ≈ +0.066, multi-seed confirmed 2026-06-23), but the pooled 90-day
`val_rank_ic` dilutes that 3–5× to ~0.012–0.02. The operating point is the
**near band** (offsets `1..k`, `k=5`).

Two of the three consumers already operate at the near horizon:

- **Training** (`training_models.py`) logs `val_rank_ic` (pooled),
  `val_rank_ic_near` (the near-band operating point, ungated since the
  06-23 operating-point-fix), and per-offset buckets (`_OFFSET_BUCKETS =
  (1, 2, 5, 10, 20, 40, 90)`, gated on `log_offset_ic`). Its val buffers carry
  `offsets`.
- **Trading** (`trading/forecast.py:load_forecasts`) consumes
  `predicted_r_close[0, 0]` — offset-1, the near horizon.

**The standalone evaluation report is the only consumer still on the pooled
number.** `evaluate.py:evaluate_model` (what `ophir evaluate` prints) computes
and reports **only** the pooled `rank_ic_mean` / `rank_ic_ir`. Its accumulator
`AccumulatedEval` carries `r_close_ids` and `r_close_dates` but **not** offsets,
so it physically cannot break IC down by horizon. The function that would —
`rank_ic_by_offset` (`evaluate.py:277`) — already exists, is already tested
(`test_evaluate.py:304`), and is already called from `training_models.py:560`;
it is simply never reached from the report path.

The result: the headline number a human reads when evaluating a checkpoint
understates the model 3–5× and disagrees with both the trader and the training
logs. The 06-23 `val_rank_ic_near` docstring already promises "the live metric
and the offline report agree" — but that offline wiring was never built. This
spec builds it.

## Scope

Measurement-only completion. No model retrain, no architecture change, no
training-logging change, no `forecast.py` change. We surface signal the model
already produces, in the one report that still hides it.

In scope:

- **A.** Accumulate per-row forecast offsets in the eval path.
- **B.** Compute and report the near-band operating point + the per-offset curve
  in `evaluate_model`, alongside the unchanged pooled metric.

Out of scope (YAGNI):

- No retrain — the existing checkpoint already produces these numbers.
- No `trading/forecast.py` change — it already reads offset-1. Letting the trader
  consume a near-*band* aggregate instead of pure offset-1 is a deliberate future
  item.
- No training-logging change — it is already correct.

## Component A — accumulate offsets in the eval path

`accumulate_targets` already collects `r_close` identity (ids + dates) parallel
to the masked predictions whenever the val loader carries identity
(`return_identity=True`). The offset of each row is the 1-based response-position
lead — exactly the quantity `training_models` puts in its `offsets` buffer.

### A1. `AccumulatedEval.r_close_offsets`

Add an optional field `r_close_offsets: torch.Tensor | None = None`, populated
row-for-row with `r_close_ids` / `r_close_dates`, gated on the same
`return_identity` opt-in. When identity is off, it stays `None` and the report
falls back to pooled-only (current behavior, unchanged).

### A2. Offset construction must match training exactly

The eval offset for each kept row must equal training's `cat_offsets` for the
same row, or the offline and live near-IC will not reconcile. Mirror the
training construction (the 1-based response-position index over the response
window `-rs:`, masked identically to pred/target). This is the **single
correctness risk** in the change and is pinned by the reconciliation test (C2).

## Component B — report the operating point

### B1. `evaluate_model` computes near + per-offset

When identity **and** offsets are present, in addition to the existing pooled
`rank_ic`:

- **Near band:** select rows with `1 <= offset <= k` (`k = 5`, the established
  near band), then reuse `dedupe_by_ticker_date` + `rank_ic` — the identical math
  `val_rank_ic_near` uses. Add as `rank_ic_near` to the `r_close` results.
- **Per-offset curve:** call the existing `rank_ic_by_offset` with
  `_OFFSET_BUCKETS`. Add the `h{offset}` map to the results.

The pooled `rank_ic_mean` / `rank_ic_ir` are computed exactly as today and left
**byte-for-byte unchanged**.

### B2. Report formatting

The formatted `ophir evaluate` output makes `rank_ic_near` the **headline**
operating-point number, keeps pooled `rank_ic_mean` beneath it as context, and
prints the per-offset `h*` curve so the decay is visible. When offsets are absent
(identity off), the report is unchanged from today.

### B3. The near-`k` source

`k` defaults to `5` (the established near band, matching the trained
`near_offset_k`). Surface it as a single module-level default in `evaluate.py`
so the report has one source of truth; the reconciliation test ties it to
`val_rank_ic_near`'s default so the two cannot silently diverge.

## Data flow

```
val loader (return_identity=True)
  → accumulate_targets  [captures pred, target, ids, dates, OFFSETS]
  → evaluate_model
       → rank_ic                      (pooled — kept, unchanged)
       → rank_ic over offsets 1..k    (rank_ic_near — headline)
       → rank_ic_by_offset(_OFFSET_BUCKETS)  (per-offset curve)
  → report: near (headline) + pooled (context) + curve
```

## Testing

CPU-only, offline, deterministic; no warnings (`filterwarnings = error`).

- **C1. Accumulator carries offsets.** `accumulate_targets` with identity yields
  `r_close_offsets` aligned 1:1 with `r_close_ids` / `r_close_dates`; absent when
  identity is off.
- **C2. Reconciliation (the key test).** `evaluate_model`'s `rank_ic_near` equals
  a direct `val_rank_ic_near(pred, target, ids, dates, offsets, k)` on the same
  accumulated tensors — guaranteeing offline and live math agree.
- **C3. Per-offset wiring.** `evaluate_model` returns an `h{offset}` map equal to
  a direct `rank_ic_by_offset` call on the same data.
- **C4. Pooled regression guard.** The pooled `rank_ic_mean` / `rank_ic_ir` are
  unchanged versus the pre-change report on identical input — protects historical
  run comparisons.
- **C5. Empty-band safety.** A response window with no rows in `1..k` yields
  `nan` for `rank_ic_near` without emitting a warning.

## Acceptance criterion

Running `ophir evaluate` on the existing checkpoint shows a near-horizon
rank-IC materially above the pooled `rank_ic_mean`, reconciling with the
`val_rank_ic_near` value the training logs already report for that checkpoint.
