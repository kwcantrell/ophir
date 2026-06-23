# Forecast-ceiling confirmation harness — design spec

**Date:** 2026-06-23
**Status:** Design approved; ready for implementation planning.
**Goal:** Confirm — multi-seed, with a per-offset significance band — the
near-offset cross-sectional skill the E3 diagnostic measured single-seed, before
committing to the operating-point fix.

## Background

E3 (see `docs/forecast-ceiling-results.md` and
`docs/superpowers/specs/2026-06-21-forecast-horizon-diagnostic-design.md`)
localized ophir's `val_rank_ic` ceiling to the **task framing** — the 90-day
response horizon plus the per-`(ticker, date)` pooled metric — not the optimizer
or architecture. Its Step B found the model already carries **~0.06–0.10
near-horizon cross-sectional IC** (per trading-day-lead offsets 1–5), 3–5× the
pooled `val_rank_ic` (~0.012–0.02), and beats the matched-horizon naive ceiling at
every near offset.

Two caveats make Step B's *magnitudes* provisional, even though its *shape*
(near≫far, beats-ceiling) is robust:

- **Single-seed, single proxy run** (`version_260`, the 6-min config).
- **Per-offset IC is noisy** (per-snapshot std ~0.08–0.17), and E1's shuffle null
  was computed on the **pooled** metric only — there is no significance band on the
  individual near-offset buckets, whose thin daily cross-sections inflate chance IC.

Before the fix banks "~0.06–0.10," this harness re-confirms the magnitudes
multi-seed and gives each near offset its own null band.

## Scope

Confirmation only. This is **design-space item 1** from
`docs/forecast-ceiling-fix-context.md`. It does **not** change the operating
point, checkpointing, or the trading seam (items 2–4). It produces the evidence
that authorizes those.

In scope:
- A **per-offset permutation null** in `ophir.ceiling`.
- **Multi-seed / multi-snapshot aggregation** of the logged `val_rank_ic_h*`.
- A **report assembler** + thin script that prints the per-offset verdict table.
- The **GPU runs** that feed it (multi-seed `ophir train --log-offset-ic` + the
  Step-A cross-section harvest), executed as part of this work.

Out of scope: any model/metric/checkpoint/inference change; new architecture.

## Key grounding facts (verified in code)

- `ophir.evaluate.rank_ic_by_offset(pred, target, ids, dates, offsets, buckets)`
  already computes per-offset pooled-day rank-IC, reusing `dedupe_by_ticker_date`
  + `rank_ic`. `_OFFSET_BUCKETS = (1, 2, 5, 10, 20, 40, 90)`
  (`training_models.py:21`).
- `ophir train --log-offset-ic` logs `val_rank_ic_h{1,2,5,10,20,40,90}` per
  validation epoch (`training_models.py:501-513`); gated, default off, default path
  unchanged.
- `training_models._trading_day_offsets(trade_occured)` is **pure CPU tensor
  logic** (`cumsum` over the trade mask) — reusable outside training for the
  harvest. Offset = 1-based trading-day lead within the response block, the same
  unit as `signal_decay_curve`'s leads.
- `ophir.ceiling` already provides the reusable primitives: `shuffle_within_day`
  (within-day permutation, advances a `torch.Generator` in place),
  `cross_sectional_ic` (production rank-IC via `dedupe_by_ticker_date` + `rank_ic`,
  drops non-finite rows), `aggregate_ic` (mean/min/max/std/n over seeds),
  `mde_for_group_difference`, and `run_ic_summary` (the per-run pattern to mirror).
- pytest stays offline + CPU-only with `filterwarnings = error`; mypy strict /
  Python 3.10 floor; NumPy-style docstrings.

## Design

### Component 1 — per-offset permutation null (`ophir.ceiling`)

```
@dataclass(frozen=True)
class NullBand:
    mean: float    # mean null IC (expected ~0)
    std: float     # sample std of the null IC distribution
    p05: float     # 5th percentile
    p95: float     # 95th percentile
    n_perms: int
    n_rows: int    # rows in this offset bucket

def per_offset_shuffle_null(
    target, ids, dates, offsets, buckets, *, n_perms, generator
) -> dict[str, NullBand]: ...
```

For each bucket `h`: select rows with `offsets == h`; `n_perms` times call
`shuffle_within_day(target_h, dates_h, generator)` and score
`cross_sectional_ic(target_h, shuffled_h, ids_h, dates_h)["ic_mean"]`; reduce the
collected per-perm IC means to a `NullBand`. Keys are `"h{h}"`, matching
`rank_ic_by_offset`.

**Why target-vs-shuffled-self (model-free).** The null IC distribution under H0
is driven by the per-day cross-section group sizes, not by the identity of the
signal; correlating the target against a within-day shuffle of itself yields the
same band as shuffling against the model's predictions, so the model is never
needed on CPU. This makes the null band fully reproducible from the Step-A harvest
alone. The expected mean is ~0; the spread is what we compare against. Documented
explicitly in the docstring.

Edge cases: an empty bucket (e.g. offset 90, which is empty on a calendar-dense
~64-trading-day block) → a `NullBand` of `nan`s, never an exception. `nan`-safe via
the existing `cross_sectional_ic` finite-filter.

### Component 2 — multi-seed / multi-snapshot aggregation (`ophir.ceiling`)

```
@dataclass(frozen=True)
class OffsetRunIC:
    snapshot_mean: float   # mean val_rank_ic_h{h} over non-burn-in snapshots (headline)
    peak: float            # max over snapshots (diagnostic: ceiling + E0 droop)
    n_snapshots: int

def run_offset_ic(metrics_csv, buckets, *, burn_in_steps=0) -> dict[str, OffsetRunIC]: ...
```

Mirrors `run_ic_summary`: parse `val_rank_ic_h{h}` columns (tolerating the
CSVLogger `_epoch` suffix via the existing `_pick_column`), drop NaN rows and rows
with `step < burn_in_steps` (warmup, whose untrained zeros would bias the mean
down), reduce to `snapshot_mean` + `peak` per offset.

Cross-seed: collect each run's per-offset `snapshot_mean` into a list and feed the
existing `aggregate_ic` → per-offset `ICAggregate` (mean/min/max/std/n across
seeds). Seed-noise scale and MDE come from the existing
`mde_for_group_difference`.

### Component 3 — report assembler + script

```
@dataclass(frozen=True)
class OffsetVerdict:
    offset: int
    seed_mean: float       # cross-seed mean of snapshot_mean
    seed_std: float
    n_seeds: int
    peak: float            # cross-seed mean of per-run peak
    null: NullBand
    clears_null: bool      # seed_mean > null.p95

def confirm_offset_skill(
    metrics_csvs, harvest, buckets, *, n_perms, burn_in_steps=0, seed
) -> list[OffsetVerdict]: ...
```

`harvest` is the CPU-harvested `(target, ids, dates, offsets)` tuple/struct;
`offsets` come from the promoted `trading_day_offsets` (see below). The assembler
runs Component 1 once on the harvest, Component 2 across the metrics CSVs, joins
them per offset, and reports `clears_null = seed_mean > null.p95` plus the
cross-seed MDE (reported once for the table, from the per-offset replicate std).

`scripts/confirm_offset_skill.py` — thin wrapper matching `scripts/leakage_viz.py`:
loads a saved harvest (`.pt`/`.npz`) + a list of metrics.csv paths, calls
`confirm_offset_skill`, prints the table. No new heavy logic in the script.

### Supporting change — promote `_trading_day_offsets`

Rename `training_models._trading_day_offsets` → public `trading_day_offsets` (keep
a private alias if any internal caller churn is undesirable), since the harvest now
reuses it outside training. Pure function, no behavior change.

## Data flow

```
GPU (this work):
  ophir train --log-offset-ic  (seeds 0..S-1)  ->  version_*/metrics.csv  (val_rank_ic_h*)
  Step-A harvest (CPU, model-free)             ->  harvest{target,ids,dates,offsets}.pt

CPU harness:
  metrics.csv[]  -> run_offset_ic -> aggregate_ic ----\
                                                        confirm_offset_skill -> table
  harvest        -> per_offset_shuffle_null ----------/
```

## GPU execution plan (run as part of this work)

1. Multi-seed proxy runs: `uv run ophir train --emb-dim 128 --num-heads 8
   --num-layers 6 --max-steps 10000 --seed {0,1,2} --val-identity
   --log-offset-ic` (~6 min each on the RTX 3090). Reuse `version_260` as seed 0 if
   its config matches. **GPU contention:** a local `llama-server` currently holds
   ~18.5 GB; free it or shrink the batch before the runs.
2. Step-A harvest (CPU, no model, minutes): build the val loader via
   `build_split_handlers` / `build_dataloader(..., return_identity=True)`, read
   `target_r_close` / `stock_id` / `date_ordinal` off
   `OHLCMulitClassPredictorInput`, derive offsets via `trading_day_offsets` on
   `trade_occured`; save to a `.pt`. (Snippet: E3 plan Task 5.)
3. Run `scripts/confirm_offset_skill.py` over the seed CSVs + harvest; record the
   table in `docs/forecast-ceiling-results.md` and update the
   `forecast-ceiling-investigation` memory if magnitudes shift.

## Testing (CPU, offline, `filterwarnings = error`)

Synthetic fixtures with a known planted per-offset signal and a seeded generator:

- **Null brackets zero:** `per_offset_shuffle_null` mean ≈ 0 within its own std;
  band widens as a bucket's daily cross-sections thin.
- **Significance discrimination:** a planted-signal offset has IC > `null.p95`
  (`clears_null` True); a pure-noise offset does not.
- **Denoising:** `run_offset_ic.snapshot_mean` over a noisy synthetic trajectory is
  closer to the planted truth than any single snapshot; `peak ≥ snapshot_mean`;
  `burn_in_steps` excludes early rows.
- **Aggregation:** cross-seed `aggregate_ic` mean/std/n match hand-computed values;
  `confirm_offset_skill` joins null + aggregate per offset correctly.
- **Degeneracy:** empty/short buckets and all-NaN columns yield `nan` bands /
  `OffsetRunIC`, never exceptions; CSVLogger `_epoch` suffix tolerated.

## Success criteria

The harness confirms (or corrects) Step B's claim: across ≥3 seeds, near offsets
(1–5) show seed-mean IC that **clears its own p95 null band** and lands in a
defensible range; far offsets (≥40) sit inside their null. If confirmed, the fix
(item 2: operate on the near-offset IC, read offset-1 for the trading seam) is
authorized. If the magnitudes collapse multi-seed, the fix premise is revisited
before any retrain.

## Pointers

- Handoff: `docs/forecast-ceiling-fix-context.md`
- Results log: `docs/forecast-ceiling-results.md` (E0/E1/E3)
- E3 spec/plan: `docs/superpowers/specs/2026-06-21-forecast-horizon-diagnostic-design.md`,
  `docs/superpowers/plans/2026-06-21-forecast-horizon-diagnostic.md`
- Memory: `forecast-ceiling-investigation`, `dev-workflow-preference`
