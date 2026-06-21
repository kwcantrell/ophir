# Forecasting-ceiling investigation — design spec

**Date:** 2026-06-20
**Status:** Design approved; ready for implementation planning.
**Goal:** Find where ophir's *forecasting-skill ceiling* lives and lift it. The
cross-sectional `val_rank_ic` is floored at ~0.014 (final-step, 10k-step proxy)
and will not move under any optimizer tuning. This investigation is **not**
another hyperparameter sweep — it pressure-tests the structural suspects upstream
of the optimizer and produces a prioritized, falsifiable experiment ladder.

## Background — what is already known

The optimizer is exhausted as a lever (see project memory
`sweep-importance-findings` and `docs/rezero-init-sweep-runbook.md`):

- `rezero_lr` is the dominant knob but already at its sweet spot (raising it hurts).
- `rezero_init` is seed noise (confirmed across 3 seeds; the apparent 56% gain did
  not survive multi-seed confirmation).
- `downside_weight`'s importance was a loss-weight scale-coupling artifact, fixed
  by sum-normalizing the loss weights.
- `lr` / `warmup_ratio` / `beta2` are minor.

Conclusion going in: the ceiling is **upstream** of the optimizer.

## Scope decisions (set with the user)

- **Everything is negotiable, including the metric.** The 90-day horizon, the raw
  per-ticker OHLC targets, and cross-sectional `val_rank_ic` as the north star are
  all fair game. The real goal is forecast usefulness for the trading core;
  `val_rank_ic` is a convenient scalar, not a fixed objective.
- **Compute is not the constraint.** Multi-seed full-budget matrices are fine. The
  limiting resources are attention and the risk of chasing seed noise (at
  |IC|~0.01–0.02, only multi-seed aggregates are trustworthy). "Cheapest-
  disambiguating-first" therefore means *ordering by how much each result forks the
  next and not building on an untrustworthy ruler* — not saving GPU hours.
- **Strategy: measurement-first, then fan out** (the approved approach). Two
  zero-GPU disambiguations gate everything; structural experiments fan out in
  parallel afterward.

## Findings from code exploration (the reframing)

These five findings reshaped the candidate list the investigation came in with.
File references are point-in-time anchors; verify before acting on them.

**Finding A — the reported IC is schedule-corrupted, and free to re-measure.**
The cosine LR `T_max` is tied to total steps (`training_models.py:538,547-548`).
A 10k-step proxy fully anneals to LR≈0 by step 10k; a full run is barely past
warmup at the same step. The proxy is **not** a truncated full run — it is a
complete, fully-annealed short run. In the 14 on-disk 10k runs, **peak
`val_rank_ic` runs 2–3× the final-step IC** (e.g. v247: peak 0.0300 vs final
0.0139). The "0.0139 baseline" in the memory note is the final-step number. The
best checkpoint monitors `val_loss`, not `val_rank_ic`, and there is no
early-stop. So the entire rezero comparison was run on annealed final-step IC.

**Finding B — train/eval mismatch (cross-sectional metric, per-ticker inputs).**
`val_rank_ic` ranks tickers *within a calendar day* (`evaluate.py:205-250`,
grouped by day), but all 12 inputs are purely per-ticker, self-normalized
price/volume transforms (`ticker.py:328-410`) — no market-relative, sector, or
rank-vs-universe features — and targets are per-ticker raw (`ticker.py:801`). The
model has no cross-sectional information to rank with.

**Finding C — the 90-day position-only response block.** `response_size` defaults
to 90 (`cli.py:99`, `train.py:323`). The entire response block is masked to a
single learned token + position embeddings (`models.py:434-460`), so the model
predicts a 90-day-ahead daily-return path with *zero per-day input features* in
the response region — only the prefix and position.

**Finding D — feature-content caps.** `open` is dropped entirely (only
reconstructed downstream as prior close, `ticker.py:511`); trailing-std
self-normalization removes absolute return scale (`ticker.py:370-371`);
weekend/holiday rows are zero-padded into every sequence (`ticker.py:398-408`); an
optional 0.75 log-return spike-drop deletes high-signal event days
(`ticker.py:323-324`); high/low are not split-adjusted while close is, distorting
`upside`/`downside` around splits (`ticker.py:261-266,385-386`).

**Finding E — leakage controls are clean, not over-strict.** Hard response mask,
no backward over-masking; the prefix retains all legitimately-available history
(`models.py:434-460`). `leakage.py` is a *verifier* (score should read ~0), not a
control. This is the least likely place the ceiling lives.

These map the user's original three directions as follows: Direction 1 (proxy
lying) → folded into E0 + E2; Direction 2 (feature signal) → split into the
structural suspects E3/E4/E5; Direction 3 (leakage eating signal) → **demoted**
(Finding E), parked unless later results are inexplicable.

## Foundation — measurement discipline (threads through every experiment)

Two non-negotiable rules, derived from Finding A and the memory-note lesson:

1. **Report best-IC-checkpoint (or peak) `val_rank_ic`, never annealed final-step.**
2. **Every comparison is multi-seed (≥3), reported as mean + min, never a
   single-seed delta.**

E0's first deliverable is the **seed-noise standard deviation → minimum
detectable effect (MDE)**. No later experiment may claim a win that does not clear
the MDE.

## The experiment ladder

### E0 — Re-measure existing artifacts (zero GPU)

- Extract peak IC, best-`val_loss`-checkpoint IC, and final IC for all 14 on-disk
  runs (`version_246`–`259`).
- Compute seed-to-seed IC std at fixed config → **the MDE** that governs the rest
  of the investigation.
- Recompute the `rezero_init` baseline-vs-0.1 comparison **on peak IC** across
  seeds 0/1/2.
- **Falsifiable fork:** if the rezero comparison still shows nothing on peak IC,
  that knob is genuinely exhausted. If it flips, the prior conclusion was a
  measurement artifact (still within proxy budget). Either way, exit with a
  trustworthy ruler.

### E1 — Naive baselines + null control (CPU / minutes)

- Cross-sectional rank-IC of trivial, *untrained* predictors on the val set:
  predict-zero, last-day return (momentum), −last-day (reversal), trailing-mean,
  trailing-vol. Reuses the eval harness only.
- Null control: shuffle targets within day → confirm IC≈0 (the metric is not
  structurally inflated).
- **Falsifiable fork:** if a naive signal already matches the model's peak IC, the
  model barely beats naive → ceiling is target/features, not training. If naive≈0
  and model>0 → real learned skill, headroom exists.

> **GATE.** E0 + E1 decide whether "the model can be better" is the right frame or
> whether to pivot toward the target (E6) immediately.

### E2 — Full-budget reality check (≥3 seeds, full epoch-driven runs)

- Run the default config at full budget, IC logged every 500 steps; compare
  sustained/peak IC to the 10k-proxy peak.
- Secondary: re-run one contrasting pair (e.g. high vs low `rezero_lr`) at full
  budget to check whether the **proxy's ranking of configs survives**.
- **Falsifiable fork:**
  - Climbs materially above proxy peak → the proxy was lying; all proxy-based
    conclusions (incl. rezero) must be re-run at budget; ceiling was
    budget/schedule.
  - Plateaus near proxy peak → proxy ranking trustworthy; ceiling is structural →
    fan out below.

### Structural fan-out (parallel, at the budget regime E2 dictates)

E3 and E6 are the prior leverage bets — the 90-day position-only response and the
per-ticker-inputs-vs-cross-sectional-metric mismatch — expected to dominate the
feature tweaks in E5.

- **E3 — Horizon / response structure (Finding C).** Train
  `response_size ∈ {1 or 5, 20, 90}`, otherwise matched, multi-seed. If
  short-horizon IC ≫ 90-day beyond MDE → the 90-day position-only response is a
  primary ceiling; restructure (shorter horizon, or per-day response
  conditioning). If flat → horizon isn't it.
- **E4 — Cross-sectional information (Finding B).** Add a minimal cross-sectional
  feature set (universe-demeaned return, universe-rank of trailing return/vol,
  market return) — a bounded pipeline change to date-align tickers, which the
  per-ticker streaming architecture does not currently do. Single-variable add,
  multi-seed. If IC lifts beyond MDE → the train/eval mismatch was a primary
  ceiling. Flat → per-ticker info is the limit for this metric.
- **E5 — Feature-content battery (Finding D), each single-variable, multi-seed,
  gated by MDE:** add `open`/overnight-gap; remove calendar zero-padding (pack
  trading days, encode date deltas); winsorize-instead-of-drop the 0.75 spike
  days; fix the high/low split-adjustment inconsistency. Keep what clears MDE.
- **E6 — Target / metric alignment (Finding B/C, strategic).** Audit what
  `signals.py` / `forecast.py` actually consume from the forecast; measure the
  correlation between `val_rank_ic` and a trading-core outcome — if they diverge,
  the wrong scalar is being optimized. Test more-predictable targets (forward
  realized vol, longer-horizon cumulative return, sign/quantile classification). A
  trading-useful target with far higher skill is the highest-leverage pivot.

### Explicitly out of scope

- **Leakage as a signal-eater (original Direction 3).** Finding E shows the
  controls are clean and not over-strict. Parked unless later results are
  inexplicable.
- **Further optimizer/rezero tuning.** Exhausted; see memory note.

## Dependencies and ordering

```
E0 (free) ─┐
E1 (free) ─┴─► GATE ─► E2 (big fork) ─► { E3, E4, E5, E6 }  (parallel)
                          │
                          └─ if proxy was lying: re-run prior conclusions at budget
```

E0 and E1 are independent and run first (free, and either can change the premise).
E2 is the single fork that sets the budget regime for the structural fan-out. E3–E6
are independent of each other and run in parallel once E2 reports.

## Deliverables

- The **MDE number** (E0) and a corrected read of the prior rezero conclusion.
- A **naive-baseline IC table** (E1) calibrating whether 0.02–0.04 is good or bad.
- A **full-budget vs proxy IC comparison** (E2) that resolves the proxy-fidelity
  question and re-validates (or invalidates) proxy-based rankings.
- For each structural experiment that clears the MDE: a confirmed, multi-seed
  signal-improvement lever, with the falsified alternatives recorded.
- A short written verdict on where the ceiling lives and the recommended pivot.

## Success criteria

The investigation succeeds when it produces a **defensible, multi-seed answer to
"where is the ceiling"** — not necessarily a higher IC. Concretely: each of
Findings A–D is converted into a confirmed-or-falsified claim measured above the
MDE, and the highest-leverage lever (or the conclusion that the target itself must
change) is identified with evidence.
