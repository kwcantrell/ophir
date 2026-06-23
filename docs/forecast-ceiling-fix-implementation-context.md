# Forecast-ceiling fix (operating point) — implementation handoff

Handoff for the **fix itself**, now that the confirmation harness has settled the
diagnosis. Read this, then start a brainstorm → spec → plan cycle for the
operating-point fix. Everything referenced lives in the repo unless noted.

## Next session: start here

1. `cd /home/kalen/ophir` and start the session (so `CLAUDE.md` loads).
2. **Pick the branch first.** The confirmation harness is merged to `main`
   (`main` is ahead of `origin/main`, unpushed). Create a fresh feature branch off
   `main` for the fix; do not work on `main` directly. Leave the
   intentionally-uncommitted user files alone (`.claude/settings.json`,
   `.gitignore`, `docs/forecast-ceiling-fix-context.md`,
   `docs/rezero-init-sweep-runbook.md`, `.graphifyignore`) — stage only your files.
3. Have the **RTX 3090 ready via `uv run`** for the train/eval runs (a local
   `llama-server` sometimes holds ~18 GB — free it before GPU work). Tests stay
   offline + CPU-only.
4. Paste this opening prompt:

   > Read `docs/forecast-ceiling-fix-implementation-context.md` — it's a handoff for
   > the forecast-ceiling **operating-point fix** after the multi-seed confirmation
   > concluded. Near-horizon skill is confirmed (pooled offsets 1–5 IC ≈ +0.066,
   > seed-stable, clears its null) and the dilution is the 90-day pooled metric.
   > Let's start the brainstorm → spec → plan cycle for the fix. Brainstorm first.

## TL;DR — what's confirmed and what the fix is

The forecasting ceiling is the **task framing (90-day horizon + pooled
`val_rank_ic`), not the optimizer or architecture.** Confirmed multi-seed
(2026-06-23, 3 seeds): the model carries genuine near-horizon cross-sectional skill
at **offsets 1–5, pooled IC ≈ +0.066** (conservative best-`val_loss` checkpoints,
~½ peak per E0; true peak likely ~0.10+), **clearing its 0.036 null on all 3
seeds**, vs the diluted pooled `val_rank_ic` ~0.012–0.02 (the predicted 3–5×). Skill
concentrates in offsets 1–5; by 1–10 it dilutes below significance.

**The fix is an operating-point / metric change — collapse to a short horizon. No
new architecture.** Full evidence: `docs/forecast-ceiling-results.md`
(confirmation section, 2026-06-23) and the `forecast-ceiling-investigation` memory.

## The fix design space (for the brainstorm)

Three components; (B) is the spine, (A) and (C) fold in.

**(A) Productionize the near-band measure + E0 checkpoint win.**
- Add a gated, logged **`val_rank_ic_near`** = pooled offsets 1–5 cross-sectional IC
  (the *right* instrument — the per-offset split is underpowered; see "what NOT to
  do" below). Reuse the existing val buffers in
  `training_models.on_validation_epoch_end` (pred/target/ids/dates/offsets are
  already accumulated when `--log-offset-ic` is on); select `offset in 1..K`, dedupe
  by `(ticker,date)`, score with production `rank_ic`.
- **E0 free win:** `ModelCheckpoint` / early-stop currently monitor `val_loss`,
  which is *anti-aligned* with IC (IC peaks mid-run then droops as the cosine LR
  anneals; `val_loss`-best ≈ drooped IC, ~½ peak). Monitor `val_rank_ic_near` (or
  `val_rank_ic`) instead — recovers ~2× for free. See `register.py:102-115`
  (the two `ModelCheckpoint` callbacks).

**(B) The operating point — short-horizon vs. read-near. Decide by experiment.**
- Two candidates, both cheap to test:
  1. **Read-near, no retrain:** the 90-day-trained model already emits per-response-day
     forecasts; operate on offset 1 (and report `val_rank_ic_near`). This is what the
     confirmation eval did (≈0.066 conservative).
  2. **Short-horizon retrain:** train at a small `response_size` so the model spends
     all its capacity on the near band. **Open question the brainstorm must answer:
     does a short-horizon model's near IC beat the 90-day model's near slice?** Test
     it — it may or may not.
- **`response_size` is a free parameter** end-to-end, validated only
  `1 <= response_size <= seq_len-1` (`models.py:482-483`, `train.py:607-608`); masks
  and reconstruction are dynamic (`dashboard.py` runs at `response_size=1`). CLI
  defaults of 90 live independently in `train.py`, `evaluate.py`, `cli.py`,
  `dashboard.py`, and `ui.py:44`.
- **Watch the units.** The response block is **calendar-dense**
  (`ticker.py:398` `pd.date_range(freq="D")`); a 90-*calendar*-day block ≈ ~64
  *trading* days. The skill is at trading-day **offsets** 1–5
  (`training_models.trading_day_offsets`, now public). A short `response_size` in
  calendar days must be chosen to cover trading-day leads 1–5 (≈ 7–10 calendar days).

**(C) Wire the trading seam.** `trading/forecast.py` exposes a scalar
`OphirForecast(r_close, upside, downside)` per symbol (effectively day-1) and
`load_forecasts` returns `{}` until inference is wired. Wire it to read the model's
**offset-1** prediction. The seam is horizon-agnostic, so collapse is safe; the
Gradio `ui.py` is the only consumer of a multi-day path (emit a near-offset-weighted
readout if it ever needs one).

## Benchmarks the fix must report (and one correction)

- Benchmark the operating point's IC against the **matched near-band reversal
  ceiling**, not the pooled-1–90 ceiling (which is ~0). E3 Step A found the clean
  per-lead reversal IC ≈ **+0.053 at lead 1**, collapsing by lead 2.
- **Correction to carry forward:** the confirmation eval printed a pooled-1–5
  "reversal ceiling" of ~0.119, but that used `lagged_target_signal(lag=1)` on
  *mixed-offset* pooled rows, so the lag-1 isn't a clean 1-trading-day reversal — the
  number is inflated/unreliable. **Recompute a clean pooled near-band reversal
  ceiling** (e.g. mean of E3's per-lead reversal over leads 1–5, ≈ low-0.0x) before
  claiming the model does/doesn't beat naive reversal. E3 Step A's per-lead curve is
  the rigorous comparand. If the model (~0.066) beats the clean near-band ceiling,
  the operating-point fix is the whole story; if not, architectural headroom remains
  (the response-block masking denies the model the 1-day feature — `models.py:434-460`).

## What NOT to do (lessons from the confirmation)

- **Do not use the per-offset `val_rank_ic_h*` / `confirm_offset_skill` flag as the
  skill measure.** At a fixed offset the daily cross-sections are tiny (median 3
  names/day → equal-day-weighted `rank_ic` null p95 ≈ 0.10); the split is underpowered
  and `clears_null` also has a single-draw-vs-denoised scale mismatch (both documented
  in `confirm_offset_skill`'s docstring). Pool near offsets into one daily
  cross-section instead — pooling 1–5 stays dense (median 12 names/day, null p95
  0.036) because windows are sparse so `(ticker,date)` dedup drops ~0 rows.
- Don't re-derive the dilution story; it's confirmed. The job now is the fix.

## Tooling already built (on `main`)

- `ophir.ceiling`: `per_offset_shuffle_null` (per-offset within-day permutation null,
  `NullBand`), `run_offset_ic` (multi-snapshot `val_rank_ic_h*` aggregation,
  `OffsetRunIC`), `confirm_offset_skill` + `format_verdict_table` (`OffsetVerdict`),
  plus the prior `run_ic_summary` / `aggregate_ic` / `signal_decay_curve` /
  `cross_sectional_ic` / `shuffle_within_day`. `scripts/confirm_offset_skill.py`.
- `ophir.evaluate.rank_ic_by_offset`; `ophir train --log-offset-ic` logs
  `val_rank_ic_h{1,2,5,10,20,40,90}`; `training_models.trading_day_offsets` (public).
- Scratch from the confirmation run (gitignored, may be stale):
  `.superpowers/sdd/gpu/` holds `harvest.py` (CPU val cross-section harvest with
  offsets), `eval_pooled_near.py` (offline pooled-near checkpoint eval), and
  `train.log`. Reuse the harvest/eval patterns — but the checkpoints there are the
  3 confirmation seeds (best-`val_loss`), saved to the shared `register.MODEL_DIR`
  with `-vN` suffixes.

## How to reproduce / extend the confirmation

- 3-seed proxy with offset-IC logging (~6 min each on the 3090):
  `uv run ophir train --emb-dim 128 --num-heads 8 --num-layers 6 --max-steps 10000
  --seed {0,1,2} --val-identity --log-offset-ic --val-batches 200`
  (versions 261/262/263 are the existing confirmation runs).
- Harvest + pooled-near eval: `.superpowers/sdd/gpu/harvest.py` then
  `.superpowers/sdd/gpu/eval_pooled_near.py` (CPU harvest is model-free; the eval
  loads checkpoints on GPU).

## Constraints / environment

- mypy strict / Python 3.10 floor; ruff 3.12; NumPy-style docstrings; reuse the
  production rank-IC math (`cross_sectional_ic` → `dedupe_by_ticker_date` +
  `rank_ic`) so offline and live metrics agree. pytest stays offline + CPU-only
  (`filterwarnings = error`). The safety gate (`trading/safety.py`) is
  non-overridable; the system is paper-only (`account_mode`).
- Dev workflow: brainstorm → spec → plan → subagent-driven execution
  (`dev-workflow-preference` memory). Specs/plans under `docs/superpowers/`.

## Pointers

- Results log: `docs/forecast-ceiling-results.md` (E0/E1/E3 + the 2026-06-23
  confirmation section).
- Confirmation spec/plan:
  `docs/superpowers/specs/2026-06-23-forecast-ceiling-confirm-harness-design.md`,
  `docs/superpowers/plans/2026-06-23-forecast-ceiling-confirm-harness.md`.
- The prior (confirmation) handoff: `docs/forecast-ceiling-fix-context.md`.
- Memory: `forecast-ceiling-investigation`, `sweep-importance-findings`,
  `dev-workflow-preference`.
