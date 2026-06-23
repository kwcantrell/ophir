# Forecast-ceiling fix — context for a fresh session

Handoff for picking up the **fix** after the diagnostic investigation (E0/E1/E3)
concluded. Read this, then start a brainstorm → spec → plan cycle for the fix.
Everything referenced lives in the repo unless noted.

## TL;DR — what was found and what the fix is

`val_rank_ic` was floored at ~0.014 and would not move under any optimizer tuning.
The investigation proved the ceiling is the **task framing (90-day horizon + the
pooled metric), not the optimizer or the architecture**:

- The model **already has ~0.06–0.10 near-horizon cross-sectional IC** (per
  trading-day-lead offset 1–5), which is **3–5× the pooled `val_rank_ic`
  (~0.012–0.02)**. The 90-day response horizon plus the per-`(ticker,date)` dedup
  (which mixes incomparable horizon offsets into one daily cross-section) dilutes
  the model's real skill away.
- The model **beats the matched-horizon naive ceiling at every near offset** — its
  skill is genuine cross-sectional structure, not just 1-day reversal.

**The fix is an operating-point / metric change, NOT a new architecture.** The
usable skill lives at offsets 1–10; expose it instead of diluting it.

Full evidence + tables: `docs/forecast-ceiling-results.md` (sections E0, E1, E3).
One-line memory: `forecast-ceiling-investigation` (in the project memory index).

## The fix design space (for the brainstorm)

Ordered cheapest-first, matching how the investigation was run.

1. **Confirm the Step-B magnitudes first (cheap, do before committing to a fix).**
   Step B was single-seed, 6-min proxy (`version_260`), and per-offset IC is noisy
   (std ~0.08–0.17/snapshot). Before banking "~0.06–0.10," re-run multi-seed with
   more val batches and add a **per-offset shuffle null** (E1's null was on the
   pooled metric only — see `ophir.ceiling.shuffle_within_day` for the pattern; it
   needs to be applied within each offset bucket). The *shape* (near≫far,
   beats-ceiling) is robust; the absolute numbers are what need confirming.

2. **Near-free operating-point change — re-read near-offset, no retrain.** The
   90-day-trained model *already* emits a per-response-day forecast; offset 1 (the
   first response day) is the day-1 forecast the trading seam needs. Candidate fix:
   (a) report/checkpoint on a **near-offset IC** (e.g. offset-1, or a 1–10 mean)
   instead of the pooled metric, and (b) at inference, read the offset-1 prediction
   for `trading/forecast.py`. This could expose ~0.05–0.10 with zero retraining.
   Open question the brainstorm must answer: is a 90-day-trained model's offset-1
   slice as good as a model trained for a short horizon? (Test it.)

3. **Short-horizon training (if (2) underperforms).** Train at a small
   `response_size` (the flag is free end-to-end; nothing requires 90 — see below).
   Benchmark against the reversal-at-matched-horizon ceiling and the per-offset
   null. The `loss_decay` weighting already up-weights near offsets; a short
   horizon may or may not beat the long model's offset-1 slice.

4. **Fold in the E0 free win** wherever checkpointing/early-stop is touched:
   monitor `val_rank_ic` (or the near-offset IC), not `val_loss` — they are
   anti-aligned (IC peaks mid-run then droops as the cosine LR anneals; the saved
   `val_loss`-best checkpoint catches ~half the peak IC).

**Deferred decision now resolved:** path-preserve-vs-collapse → **collapse** to a
short horizon. The trading seam needs only the near-horizon day-1 forecast; the far
block carries no skill (offset >~30 IC ≈ 0 or negative). A multi-day path can still
be emitted via a near-offset-weighted readout if the UI ever needs it.

## Tooling already built (on branch `forecast-ceiling-gate`)

All offline/CPU, fully typed, tested:
- `ophir.ceiling.run_ic_summary(metrics_csv)` → peak / best-`val_loss`-ckpt / final
  `val_rank_ic` + `peak_step` from a run's `metrics.csv`.
- `ophir.ceiling.aggregate_ic`, `mde_for_group_difference` → multi-seed mean/min and
  the minimum-detectable-effect (MDE ≈ 0.0069 at 3 seeds on this setup).
- `ophir.ceiling.signal_decay_curve(target, ids, dates, leads, kind=...)`,
  `pooled_baseline_ceiling(decay, response_size)` → reversal/momentum IC by
  trading-day lead + the matched-horizon ceiling.
- `ophir.ceiling.lagged_target_signal`, `cross_sectional_ic`, `dedupe_rows`,
  `shuffle_within_day` → naive baselines + within-day null, reusing production
  rank-IC math.
- `ophir.evaluate.rank_ic_by_offset(pred, target, ids, dates, offsets, buckets)` →
  per-offset (trading-day-lead) cross-sectional IC; `_OFFSET_BUCKETS` in
  `training_models.py` is the offset list.
- `ophir train --log-offset-ic` → logs `val_rank_ic_h{1,2,5,10,20,40,90}` during
  validation (gated; default off, default path unchanged). Offset = trading-day
  rank within the response block (`training_models._trading_day_offsets`).

## Key code facts (verified during the investigation)

- **`response_size` is a free parameter end-to-end**, validated only
  `1 <= response_size <= seq_len-1` (`models.py:482-483`, `train.py:607-608`).
  Nothing bakes in 90; masks/reconstruction are dynamic (`dashboard.py` runs the
  pipeline at `response_size=1`). CLI defaults of 90 live independently in
  `train.py` (`train`/`finetune`/`run_training`), `evaluate.py`, `cli.py` (sweep),
  `dashboard.py`, and the `ui.py:44` constant.
- **The response block is masked to a single learned token + position**
  (`models.py:_apply_response_mask`, `models.py:434-460`). The model predicts each
  response day from the prefix + position only.
- **`val_rank_ic` pools ALL horizon offsets** into each calendar-date cross-section,
  deduped to one row per `(ticker, date)` (`evaluate.py:205-274`, accumulation at
  `training_models.py:439-476`). This pooling is the dilution mechanism.
- **The trading seam is horizon-agnostic and not yet wired:**
  `trading/forecast.py` exposes only a scalar `OphirForecast(r_close, upside,
  downside)` per symbol (effectively day-1); `load_forecasts` returns `{}` until
  inference is wired. Wiring it to read the offset-1 model prediction is part of the
  fix's downstream.
- The input window is calendar-dense (`ticker.py:398` `pd.date_range(freq="D")`),
  weekends/holidays are padding rows (`trade_occured`), so a 90-*calendar*-day
  response block is only ~64 trading days — offset-90 buckets are empty.

## How to reproduce the diagnostics

- **Harvest the val cross-section (CPU, no model, ~minutes)** + decay curve: the
  Step-A snippet is in `docs/superpowers/plans/2026-06-21-forecast-horizon-diagnostic.md`
  (Task 5). It builds the val loader via `build_split_handlers` /
  `build_dataloader(..., return_identity=True)` and reads `target_r_close` /
  `stock_id` / `date_ordinal` off `OHLCMulitClassPredictorInput` on CPU.
- **Per-offset model IC:** `uv run ophir train --emb-dim 128 --num-heads 8
  --num-layers 6 --max-steps 10000 --seed 0 --val-identity --log-offset-ic`
  (~6 min on the RTX 3090), then average the `val_rank_ic_h*` columns across the
  validation snapshots in `metrics.csv` (single snapshots are too noisy). Run
  `version_260` is the existing one.

## Constraints / environment (carry into the fix)

- mypy strict / Python 3.10 floor; ruff 3.12; NumPy-style docstrings; reuse
  production IC math. pytest stays offline + CPU-only (`filterwarnings = error`);
  training/eval need the RTX 3090 via `uv run`.
- Dev workflow: brainstorm → spec → plan → subagent-driven execution (see the
  `dev-workflow-preference` memory). Specs/plans under `docs/superpowers/`.
- Branch hygiene: `main` has unpushed commits and an intentionally-uncommitted
  `.claude/settings.json` plus a modified `docs/rezero-init-sweep-runbook.md` —
  leave all three alone; stage only your own files.

## Pointers

- Results log: `docs/forecast-ceiling-results.md`
- Investigation spec: `docs/superpowers/specs/2026-06-20-forecast-ceiling-investigation-design.md`
- E3 diagnostic spec: `docs/superpowers/specs/2026-06-21-forecast-horizon-diagnostic-design.md`
- E3 plan: `docs/superpowers/plans/2026-06-21-forecast-horizon-diagnostic.md`
- Memory: `forecast-ceiling-investigation`, `sweep-importance-findings`,
  `dev-workflow-preference`.
