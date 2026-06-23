# Forecasting-ceiling investigation — results log

Living record of the measurement-gate experiments (E0/E1/E2). Plan:
`docs/superpowers/plans/2026-06-20-forecast-ceiling-measurement-gate.md`.
Spec: `docs/superpowers/specs/2026-06-20-forecast-ceiling-investigation-design.md`.
Helpers: `ophir.ceiling`.

---

## E0 — re-measure the 14 on-disk 10k runs (2026-06-20, zero GPU)

Ruler decision: **peak `val_rank_ic` is the working ruler**, not final-step. The
existing best checkpoint monitors `val_loss`, which is misaligned with IC (see
below).

### Per-run IC trajectory (peak / best-val_loss-ckpt / final)

| version | config (init, seed) | peak IC @ step | best-ckpt IC | final IC | peak/final |
| ------- | ------------------- | -------------- | ------------ | -------- | ---------- |
| 246 | shallow L1 | 0.0080 @7999 | 0.0038 | 0.0049 | 1.62× |
| 247 | i0.0 s0 | **0.0300** @8999 | 0.0139 | 0.0139 | 2.16× |
| 248 | (deep arm) | 0.0238 @5499 | 0.0092 | 0.0092 | 2.59× |
| 249 | (deep arm) | 0.0238 @5499 | 0.0092 | 0.0092 | 2.59× |
| 250 | high rezero_lr | 0.0351 @5499 | 0.0113 | 0.0113 | 3.11× |
| 251 | (early-stopped) | 0.0020 @499 | 0.0020 | 0.0020 | 1.00× |
| 252 | i0.1 s0 | 0.0299 @5499 | 0.0216 | 0.0216 | 1.38× |
| 253 | i0.05 s0 | 0.0264 @8999 | 0.0141 | 0.0141 | 1.87× |
| 254 | i0.2 s0 | 0.0263 @5499 | 0.0167 | 0.0167 | 1.57× |
| 255 | i0.4 s0 | 0.0271 @5499 | 0.0210 | 0.0210 | 1.29× |
| 256 | i0.0 s1 | 0.0290 @5999 | 0.0178 | 0.0109 | 2.66× |
| 257 | i0.1 s1 | 0.0247 @8499 | 0.0082 | 0.0068 | 3.66× |
| 258 | i0.0 s2 | 0.0223 @7499 | 0.0083 | 0.0171 | 1.31× |
| 259 | i0.1 s2 | 0.0173 @3499 | 0.0052 | 0.0106 | 1.63× |

### MDE (noise floor)

From the `rezero_init=0.0` baseline at seeds 0/1/2 (v247/256/258), **peak** ICs
`[0.0300, 0.0290, 0.0223]`, mean **0.0271**, sample std 0.0042.
**MDE = 0.0069** (2σ on a 3-seed group-mean difference). No effect smaller than
this is believable at this seed count.

### Rezero conclusion re-read on the better ruler

| init | peak IC mean | min | max |
| ---- | ------------ | --- | --- |
| 0.0  | 0.0271 | 0.0223 | 0.0300 |
| 0.1  | 0.0240 | 0.0173 | 0.0299 |

delta(0.1−0.0) = **−0.0031**, well inside the MDE of 0.0069 → **WITHIN NOISE**.
(Final-step read used by the prior conclusion: 0.0139 vs 0.0130 — same direction,
also within noise.)

### Verdict

1. **The rezero "don't tune it" conclusion HOLDS on peak IC.** `init=0.1` does not
   beat baseline on the better ruler either (it is slightly lower, within noise).
   No prior conclusion is overturned by the measurement fix. The premise "the
   optimizer/rezero family is exhausted" stands.

2. **But the absolute skill was understated ~2×.** Peak IC for the baseline is
   ~0.027, not the headline 0.0139. The final-step / `val_loss`-checkpoint reading
   halves the apparent skill. This is the specific sense in which the proxy "lies":
   not in the *ranking* of configs, but in the *level* — anyone reading final-step
   or the saved checkpoint sees ~half the model's peak cross-sectional skill.

3. **New actionable finding — checkpoint criterion is anti-aligned with IC.** IC
   peaks mid-run (steps ~5500–9000, i.e. 55–90% through) then **droops as the
   cosine LR anneals**, while `val_loss` keeps falling to the final step. So
   minimizing `val_loss` (the current `ModelCheckpoint` monitor) actively selects
   *away* from peak IC — `best_ckpt_ic` equals `final_ic` for most runs. Switching
   the checkpoint/early-stop monitor to `val_rank_ic` would capture ~2× higher
   cross-sectional skill **for free**, no retraining. This is a concrete lever to
   carry into the structural phase (and a likely cheap win independent of E2).

4. This sharpens E2: the proxy fully anneals by 10k, so the IC droop is a
   schedule artifact. At full budget the cosine is stretched — does peak IC sustain
   higher/longer, or plateau near ~0.027? That is the next fork.

---

## E1 — naive-baseline calibration (2026-06-20, CPU)

Harvested the full 2024+ validation cross-section on CPU (no model, no CUDA):
1160 batches → 2,254,083 response rows → deduped to one row per (ticker, date)
across **301 distinct days**. Scored trivial, untrained signals with the same
production rank-IC math the model is evaluated on (`ophir.evaluate.rank_ic`).

| signal | cross-sectional IC | n_days |
| ------ | ------------------ | ------ |
| momentum (prev-day return) | **−0.0515** | 281 |
| reversal (−prev-day return) | **+0.0533** | 281 |
| null (target shuffled within day, 20 draws) | mean +0.0041, max\|·\| 0.0297 | — |
| *(reference) model baseline PEAK IC* | *0.0271* | — |

### Verdict

1. **The metric is ~unbiased.** The within-day-shuffle null averages +0.0041,
   inside the MDE of 0.0069. The cross-sectional rank-IC is not structurally
   inflated; nonzero IC means real signal.

2. **The signal exists and is large — a one-line reversal rule scores ~0.053,
   nearly 2× the model's peak 0.027.** Daily equity returns carry strong
   cross-sectional *short-term reversal* (a well-documented anomaly): yesterday's
   biggest losers tend to be today's relative winners. So the ceiling is **not**
   "there's no signal" — there is roughly double the model's current skill sitting
   in a costless baseline.

3. **The model is not capturing it — and the architecture explains why (Finding
   C).** The comparison is horizon-confounded, and the confound *is* the result:
   the naive reversal always predicts **1 day ahead** from the immediately prior
   real return — the single easiest, most predictive feature. The model predicts a
   **90-day response block whose every day is masked to a position-only token**
   (`models.py` `_apply_response_mask`), and `val_rank_ic` pools all 1–90-day
   horizon offsets. For response day 1 the prefix still holds yesterday's return;
   for response day 90 the nearest real data is 90 days stale. The model's
   architecture **structurally denies itself the short-term-reversal feature** for
   almost its entire predicted block, then is scored on an average dominated by the
   hard far-horizon days. ~0.027 is what leaks through.

4. **Implication — this points hard at E3 (horizon/response structure) as the
   dominant ceiling, ahead of feature work (E5) or more budget (E2).** Concretely:
   - There is headroom (≈0.05 of cross-sectional signal demonstrably exists), but
     it is locked behind the task framing, not the optimizer — consistent with the
     "optimizer exhausted" premise.
   - Future model runs must be benchmarked against the **reversal baseline at
     matched horizon**, not against 0.027. At 1-day horizon the bar is ~0.05.
   - E2 (full-budget) is still worth running, but its bar is now ~0.05, not 0.027,
     and it predicts the *same* hard 90-day block — so budget alone is unlikely to
     close the gap. E2 now mainly tests whether the schedule droop (E0 finding 3)
     is recoverable at budget; the structural fix is E3.

### Caveat

momentum/reversal here are strictly 1-day-ahead (lag-1 prior trading day), an
*upper* reference for the easiest horizon — not a like-for-like competitor to the
model's pooled 1–90-day metric. E3 should compare model-vs-reversal at each
horizon to quantify how much of the ~0.05 the model recovers as the horizon shrinks.

---

## E3 — horizon diagnostic

Spec: `docs/superpowers/specs/2026-06-21-forecast-horizon-diagnostic-design.md`.
Tooling: `ophir.ceiling.signal_decay_curve` / `pooled_baseline_ceiling`,
`ophir.evaluate.rank_ic_by_offset`, gated `ophir train --log-offset-ic`.

### Step A — signal-decay curve + matched-horizon ceiling (2026-06-22, CPU)

Same harvested val cross-section as E1 (2,254,083 rows, 301 days). Reversal IC by
forecast lead (trading days):

| lead | 1 | 2 | 3 | 5 | 10 | 20 | 40 | 90 |
| ---- | - | - | - | - | -- | -- | -- | -- |
| reversal IC | **+0.0533** | +0.0028 | +0.0194 | −0.0159 | −0.0315 | −0.0216 | −0.0182 | +0.0031 |

- **1-day ceiling: +0.0533.**
- **90-day pooled (matched-horizon) ceiling: −0.0011 ≈ 0.**
- Reference: model baseline pooled peak IC = 0.0271; MDE = 0.0069.

### Step A verdict — E1 reframed (not overturned)

1. **The signal is almost entirely at lead 1.** Reversal IC is +0.053 at lead 1,
   collapses to ~0 by lead 2, then **flips sign to momentum** (negative reversal IC)
   across leads 5–40, returning to ~0 by lead 90 — the classic short-reversal /
   medium-momentum autocorrelation structure. Pooled across 1–90 it averages to ~0.

2. **The model's +0.027 is ABOVE the fair matched-horizon ceiling, not below it.**
   The apples-to-oranges in E1 is now quantified: the matched-horizon-mix naive
   ceiling at the model's 90-day-pooled operating point is ≈ 0 (−0.0011), while the
   model gets +0.0271 — beating matched-horizon naive by ~0.027 (≈ 4× the MDE). So
   "the model loses to a one-liner" was the horizon confound; at its real operating
   point the model *adds* cross-sectional skill beyond any single-lag reversal/
   momentum signal.

3. **The opportunity is the short horizon, not the pooled one.** ~0.05 of skill is
   locked at lead 1; the 90-day pooling dilutes it (and the sign-flip cancels it).
   A model operating at lead 1 could target ~0.05 vs the pooled 0.027 — *if* it can
   capture the lead-1 reversal.

4. **This sharpens Step B's question.** Does the existing 90-day model's IC at
   offset 1 already approach ~0.05 (→ it captures the reversal and the pooled 0.027
   is dilution → **world 1**, operate-short / collapse-horizon fix), or does it stay
   ~flat near 0.027 across offsets (→ its +0.027 is uniform cross-sectional
   structure, *not* reversal; it does not grab the lead-1 signal even at offset 1 →
   **world 2**, reversal-aware architectural fix)? Step B's per-offset model IC,
   overlaid on this curve, decides it.

### Step B — per-offset model IC (2026-06-22, one 10k run, `version_260`)

Trained one 10k model with `--log-offset-ic` (6 min on the 3090). Single-snapshot
per-offset IC is very noisy (each day's cross-section split 7 ways leaves few
tickers per offset), so the table below averages each offset's IC across the 20
logged validation snapshots (and, as a maturity check, across the 10 highest
pooled-IC snapshots). Offset is the **trading-day lead** within the response block,
matching the Step-A ceiling's lead unit.

| offset h | model IC (mean of 20) | model IC (top-10) | reversal ceiling |
| -------- | --------------------- | ----------------- | ---------------- |
| 1  | +0.0958 | +0.0607 | +0.0533 |
| 2  | +0.1010 | +0.1448 | +0.0028 |
| 5  | +0.0665 | +0.1045 | −0.0159 |
| 10 | +0.0398 | +0.0359 | −0.0315 |
| 20 | +0.0191 | +0.0152 | −0.0216 |
| 40 | −0.0266 | −0.0386 | −0.0182 |
| 90 | (empty) | (empty) | +0.0031 |

Pooled `val_rank_ic`: peak 0.0300, mean-over-snapshots 0.0115.
(Offset 90 is empty — a 90-*calendar*-day response block holds only ~64 trading
days; this confirms the trading-day-offset instrumentation.)

### E3 verdict — WORLD 1 (diluted-but-captured)

1. **The model's near-horizon skill is 3–5× its pooled metric.** Per-offset IC is
   ~+0.06 to +0.10 at offsets 1–5 and decays monotonically to ~0 by offset ~30 and
   negative beyond — while the pooled `val_rank_ic` is only ~0.012–0.02. The 90-day
   pooling (plus the per-`(ticker,date)` dedup that mixes incomparable offsets into
   one daily cross-section) dilutes the model's real skill ~3–5×. The skill *is*
   captured; the operating point and metric hide it.

2. **The model beats the matched-horizon naive ceiling at every near offset**, not
   just at lead 1 (offset 1 ≈ the reversal ceiling; offsets 2–20 are solidly
   positive where naive reversal/momentum is negative or zero). So the +0.027 the
   model earns is genuine cross-sectional structure beyond any single-lag signal —
   refuting world 2.

3. **Decision — operating-point/metric fix; collapse to a short horizon.** No
   reversal-aware architecture is required. The usable skill lives at offsets 1–10;
   predicting/scoring there (a short `response_size`, or a near-offset-weighted
   readout) should expose ~0.05–0.10 of cross-sectional IC versus the ~0.02 pooled
   today. The deferred **path-preserve-vs-collapse** question resolves to **collapse**:
   the trading seam needs only the near-horizon forecast anyway, and the skill
   isn't in the far block. (A multi-day path can still be emitted via a
   near-offset-weighted readout if the UI ever needs it — but it carries no skill
   past offset ~30.)

### Caveats / confirmation before banking

- Single seed, 6-min proxy, 50 val batches. The per-offset **shape** (near≫far,
  decaying, beats-ceiling) is robust across both the 20-snapshot and top-10 reads,
  but absolute magnitudes are noisy (per-offset std ~0.08–0.17 per snapshot).
- The striking "pooled 0.012 vs per-offset 0.10" gap and the absolute per-offset
  ICs should be confirmed with a multi-seed run, more val batches, and a *per-offset*
  shuffle null (E1's null was on the pooled metric) before locking the magnitudes.
  The qualitative verdict (world 1; collapse-horizon) does not depend on the exact
  numbers.

### Hand-off

Next is the **fix spec**: a short-horizon (or near-offset-weighted) operating
point — train/eval at small `response_size`, benchmark against the
reversal-at-matched-horizon ceiling and a per-offset null, targeting usable IC
~0.05–0.10 vs the current pooled ~0.02. The E0 free win (checkpoint/early-stop on
`val_rank_ic`) folds in there too.
