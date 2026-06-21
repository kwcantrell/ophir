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
