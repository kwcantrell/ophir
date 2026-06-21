# Forecast-horizon diagnostic (E3) — design spec

**Date:** 2026-06-21
**Status:** Design approved; ready for implementation planning.
**Goal:** Decide *where* the forecasting-skill ceiling fix lives by measuring how
much short-horizon cross-sectional signal exists and whether the model captures
it — without yet committing to a fix.

## Background

The measurement gate (E0/E1, see `docs/forecast-ceiling-results.md` and
`docs/superpowers/specs/2026-06-20-forecast-ceiling-investigation-design.md`)
localized ophir's cross-sectional rank-IC ceiling to the **task structure**, not
the optimizer:

- **E0:** on a corrected peak-IC ruler, the rezero/optimizer family is exhausted;
  true baseline skill is peak `val_rank_ic` ≈ 0.027 (the 0.0139 headline was the
  annealed final-step reading). MDE ≈ 0.0069 at 3 seeds.
- **E1:** a costless 1-day cross-sectional **reversal** rule scores IC ≈ 0.053; the
  metric is unbiased (within-day-shuffle null ≈ 0). The signal exists at roughly
  2× the model's pooled skill.
- **Mechanism:** the model forecasts a **90-day response block masked to
  position-only tokens** (`models.py:_apply_response_mask`), and `val_rank_ic`
  pools all 1–90-day horizon offsets into each calendar-date cross-section
  (`evaluate.py` accumulation). The naive reversal exploits the easiest 1-day lead;
  the architecture denies the model that feature across most of its predicted block.

## Key grounding facts (verified in code)

- `response_size` is a **free parameter** end-to-end, validated only
  `1 <= response_size <= seq_len-1` (`models.py:482-483`, `train.py:607-608`).
  Nothing bakes in 90; reconstruction and masks are dynamic
  (`dashboard.py` already runs the pipeline at `response_size=1`).
- The **trading seam is horizon-agnostic**: `trading/forecast.py` exposes only a
  scalar `OphirForecast(r_close, upside, downside)` per symbol (effectively the
  first forecast day); inference is not yet wired. A short horizon does **not**
  break the product contract. Only the Gradio UI (`ui.py`) consumes the full path.
- `val_rank_ic` pools **all** horizon offsets, keyed by each prediction's calendar
  date and deduped to one row per `(ticker, date)` (`evaluate.py:323-349`,
  `dedupe_by_ticker_date`, `rank_ic`). Shrinking the horizon changes *which
  forecast leads* populate the metric, not just the sample count.

## Scope decision

E3 is a **diagnostic only**. It does not implement a fix. The path-preservation
question (keep a multi-day forecast path vs. collapse to a short horizon) is
**deferred to E3's verdict** — the user explicitly chose "decide after the
diagnostic."

## What E3 must decide

Distinguish two worlds, which imply very different fixes:

1. **Diluted-but-captured:** the model already captures short-horizon skill and the
   90-day pooling merely averages it down → fix is a near-free operating-point /
   metric change (operate short, checkpoint on IC).
2. **Not-captured:** the model fails to reach the signal even at the easy 1-day
   lead → a real architectural fix is needed (per-day legitimate conditioning), and
   path-preservation becomes a live design question.

Plus: quantify how steeply the exploitable cross-sectional signal decays with
forecast lead, which sets the right operating horizon.

## Design

### Step A — signal-decay curve + matched-horizon ceiling (free; no GPU, no retrain)

Corrects the apples-to-oranges in E1: "reversal 0.053 vs model 0.027" compared a
1-day baseline against the model's 90-day-pooled metric. Step A computes the
reversal IC **at each forecast lead** and the **matched-horizon-mix ceiling** — the
fair comparand for the model's pooled 0.027.

- **Harvest once, cache:** rerun the E1 CPU harvest (build val handlers, iterate the
  `return_identity=True` loader, extract `target_r_close` / `stock_id` /
  `date_ordinal` from the response block on CPU — no model, no CUDA) and save
  `(target, ids, dates)` to a temp file so Step A and Step B's baseline reuse it.
- **`ceiling.signal_decay_curve(target, ids, dates, leads, *, kind="reversal")
  -> dict[int, float]`:** for each lead `L`, reuse the existing
  `lagged_target_signal(lag=L)` + `cross_sectional_ic` (negated signal for
  reversal) to get cross-sectional IC at lead `L`. Default
  `leads = (1, 2, 3, 5, 10, 20, 40, 90)`.
- **`ceiling.pooled_baseline_ceiling(decay, response_size) -> float`:** the
  matched-horizon ceiling, defined as the mean of the decay curve over leads
  `1..response_size`. Documented as an *approximation* of the metric's true
  offset-pooling (the real metric dedups offsets per `(ticker, date)`; the uniform
  mean is the defensible proxy, not an exact replica).

**Deliverable:** the decay curve, the 1-day ceiling (~0.05), and the 90-day pooled
ceiling, each set beside the model's pooled peak IC (0.027).

**Falsifiable forks:**
- pooled ceiling ≪ 0.027 → the model already beats matched-horizon naive; the
  "loses to a one-liner" reading was the horizon confound; fix = operate short.
- pooled ceiling ≈ 0.05 and roughly flat → signal is broadly available across leads
  and the model is genuinely leaving skill on the table.

### Step B — per-offset model IC instrumentation (one cheap 10k run)

Answers what Step A cannot: does the *model's* skill concentrate at near horizons
like the ceiling does? Instruments the **validation loop** and reads per-offset IC
at the val step where pooled `val_rank_ic` peaks — avoiding E0's drooped-checkpoint
confound (no reliance on the `val_loss`-best saved checkpoint).

- **`evaluate.rank_ic_by_offset(pred, target, ids, dates, offsets, buckets)
  -> dict[str, float]`:** `offsets` is a per-row integer tensor giving each
  prediction's response-position offset (1..`response_size`); `buckets` is an
  iterable of integer offsets to report (e.g. `(1, 2, 5, 10, 20, 40, 90)`). For
  each requested offset `h`, select the rows with `offsets == h`, then compute that
  bucket's cross-sectional IC by reusing `dedupe_by_ticker_date` + `rank_ic`.
  Returns one key per bucket, named `h{offset}` (e.g. `"h1"`, `"h5"`), mapping to
  the bucket's `ic_mean` (NaN for an empty/insufficient bucket). Pure;
  offline-testable.
- **Validation accumulation change (`training_models.py`):** alongside the existing
  `pred/target/ids/dates` buffers, accumulate each row's **response-position
  offset** — the column index within the `[:, -rs:]` slice, plus 1. At epoch end,
  when a new `log_offset_ic` flag is set, log `val_rank_ic_h{1,2,5,10,20,40,90}`
  via `rank_ic_by_offset`. **Gated exactly like `log_rezero_gates`** so the default
  training path — and the offline/CPU test suite — are unchanged.
- **CLI:** a `--log-offset-ic` flag on `train` mirroring `--log-rezero-gates`,
  threaded through `run_training` to the Lightning module.
- **One operational run:** a single 10k training with `--log-offset-ic
  --val-identity` (diagnostic defaults: `--emb-dim 128 --num-heads 8
  --num-layers 6 --max-steps 10000 --seed 0`), then read the per-offset IC columns
  from `metrics.csv` at the peak pooled-`val_rank_ic` step (via
  `ceiling.run_ic_summary` to find the peak step) and overlay on the Step-A
  ceiling-per-lead.

**Decision (from A + B):**
- Model IC at h=1 approaches the ~0.05 ceiling and decays with offset like the
  ceiling → world 1 (diluted-but-captured); fix = operate short / IC-checkpoint;
  recommend **collapse to a short horizon**.
- Model IC stays ~flat near 0.027 even at h=1 while the ceiling is ~0.05 → world 2
  (not-captured); fix is architectural (per-day legitimate conditioning), and
  **path-preservation becomes live**.
- This is where the deferred path-preserve-vs-collapse call is made.

## Components & interfaces

| Unit | File | Kind | Interface |
| ---- | ---- | ---- | --------- |
| `signal_decay_curve` | `ceiling.py` | pure (TDD) | `(target, ids, dates, leads, *, kind="reversal") -> dict[int, float]` |
| `pooled_baseline_ceiling` | `ceiling.py` | pure (TDD) | `(decay: dict[int, float], response_size: int) -> float` |
| `rank_ic_by_offset` | `evaluate.py` | pure (TDD) | `(pred, target, ids, dates, offsets, buckets) -> dict[str, float]` |
| offset accumulation + `log_offset_ic` | `training_models.py` | gated instrumentation | logs `val_rank_ic_h{N}`; default path unchanged |
| `--log-offset-ic` | `train.py` CLI | flag | mirrors `--log-rezero-gates` |
| Step A harvest+curve script | operational | snippet | reuses E1 harvest; writes results |
| Step B run + analysis | operational | snippet + 1 GPU run | reads peak-step per-offset IC |
| E3 results section | `docs/forecast-ceiling-results.md` | doc | curve, ceiling, per-offset IC, decision |

## Testing

- Pure helpers (`signal_decay_curve`, `pooled_baseline_ceiling`,
  `rank_ic_by_offset`) are TDD'd offline with synthetic arrays/tensors; reuse the
  production `rank_ic` / `dedupe_by_ticker_date`.
- The offset-bucketing logic is unit-tested via `rank_ic_by_offset`; the Lightning
  validation wiring is exercised by the operational Step-B run (not the CPU suite).
- The `log_offset_ic` gate must leave the default validation path (and therefore
  the existing offline tests) byte-for-byte unchanged when the flag is off.

## Hard constraints

- mypy strict / Python 3.10 floor; ruff 3.12; NumPy-style docstrings; reuse
  production IC math (never reimplement Spearman / day-grouping).
- pytest stays offline + CPU-only (`filterwarnings = error`); synthetic fixtures;
  no network / CUDA / `.ophir/` access in tests.
- Operational runs use the RTX 3090 via `uv run`.
- Leave `main`'s unpushed commits, the uncommitted `.claude/settings.json`, and the
  modified `docs/rezero-init-sweep-runbook.md` untouched; stage only files this work
  creates/edits.

## Out of scope (explicit)

- **The fix itself** — operate-short retrain *or* per-day legitimate conditioning —
  is the next spec, selected by this diagnostic's verdict.
- The IC-monitored checkpoint free-win from E0 (separate follow-up).
- The retrain `response_size` sweep (Approach 2; only if Step B is ambiguous).
- Wiring `trading/forecast.py` inference.

## Success criteria

E3 succeeds when it produces, above the MDE where applicable: (1) the signal-decay
curve and matched-horizon ceiling; (2) the model's per-offset IC at peak; and (3) a
written, evidence-backed verdict assigning the ceiling to world 1 (operating-point
fix) or world 2 (architectural fix), with the path-preserve-vs-collapse decision
made. A higher IC is not required — a defensible decision is.
