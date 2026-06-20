# ReZero init sweep — runbook

Find the best `rezero_init` and confirm the depth gain is real, not seed noise.

Background: the gate-opening diagnostic (`docs/rezero-diagnostic-runbook.md`)
found that `rezero_init=0.1` beat the zero-init default by ~56% at 10k steps
(val_rank_ic 0.0216 vs 0.0139) — the zero-init gates are *gate-starved* (they
crawl to only ~0.018 in 10k steps). Starting them open helps; cranking
`rezero_lr` to force them open does **not** (it destabilizes). So this sweep
varies `rezero_init` only, with the normal `rezero_lr`. Uses the existing
`--rezero-init` flag — no code changes. Requires CUDA.

Fixed setup for every run (matches the diagnostic for comparability):
`--emb-dim 128 --num-heads 8 --num-layers 6 --max-steps 10000 --val-identity --log-rezero-gates`.
After launching each run, verify the flag took (the earlier Arm D attempt
silently kept `rezero_init: 0.0`):

```bash
ls -dt src/ophir/.ophir/model/csv-logger/version_* | head -1 \
  | xargs -I{} grep -E "rezero_init|num_layers" {}/hparams.yaml
```

## Phase 1 — init grid at seed 0

Two points are already run — **reuse them, don't re-run**:
- `rezero_init=0.0` → `version_247`
- `rezero_init=0.1` → `version_252`

Run the gaps (record each new `version_N`):

```bash
ophir train --emb-dim 128 --num-heads 8 --num-layers 6 --rezero-init 0.05 --max-steps 10000 --seed 0 --val-identity --log-rezero-gates
ophir train --emb-dim 128 --num-heads 8 --num-layers 6 --rezero-init 0.2  --max-steps 10000 --seed 0 --val-identity --log-rezero-gates
ophir train --emb-dim 128 --num-heads 8 --num-layers 6 --rezero-init 0.4  --max-steps 10000 --seed 0 --val-identity --log-rezero-gates
```

Summarize (fill in the new version dirs):

```python
from ophir.dashboard import summarize_rezero_runs
base = "src/ophir/.ophir/model/csv-logger"
print(summarize_rezero_runs({
    "i0.00_s0": f"{base}/version_247",   # reused
    "i0.05_s0": f"{base}/version_AAA",
    "i0.10_s0": f"{base}/version_252",   # reused
    "i0.20_s0": f"{base}/version_BBB",
    "i0.40_s0": f"{base}/version_CCC",
}))
```

Pick the init with the highest seed-0 `val_rank_ic` — call it `I*`. **Make sure
the peak is interior**, not at a grid edge: if `0.4` is still the best, the
optimum isn't bracketed — add `0.8`; if `val_rank_ic` already turns over by
`0.2`, the peak is bracketed and you can stop.

## Phase 2 — multi-seed confirmation

Single-seed deltas are noisy (absolute IC is ~0.01–0.02). Re-run the baseline
(`0.0`) and `I*` across two more seeds; seed 0 is already done in Phase 1.

```bash
for SEED in 1 2; do
  ophir train --emb-dim 128 --num-heads 8 --num-layers 6 --rezero-init 0.0 --max-steps 10000 --seed $SEED --val-identity --log-rezero-gates
  ophir train --emb-dim 128 --num-heads 8 --num-layers 6 --rezero-init <I*> --max-steps 10000 --seed $SEED --val-identity --log-rezero-gates
done
```

Aggregate per init across seeds (label runs `i<init>_s<seed>`):

```python
import pandas as pd
from ophir.dashboard import summarize_rezero_runs
base = "src/ophir/.ophir/model/csv-logger"
runs = {
    "i0.0_s0": f"{base}/version_247",
    "i0.0_s1": f"{base}/version_DDD",
    "i0.0_s2": f"{base}/version_EEE",
    f"iISTAR_s0": f"{base}/version_252",  # if I* == 0.1; else its Phase-1 version
    f"iISTAR_s1": f"{base}/version_FFF",
    f"iISTAR_s2": f"{base}/version_GGG",
}
df = summarize_rezero_runs(runs)
df[["init", "seed"]] = df["arm"].str.extract(r"i([A-Za-z0-9.]+)_s(\d+)")
agg = (
    df.groupby("init")["val_rank_ic"]
      .agg(["mean", "min", "max", "count"])
      .sort_values("mean", ascending=False)
)
print(agg)
```

## Decision criterion

- **Phase 1:** choose `I*` = the init with the highest seed-0 `val_rank_ic`,
  with the peak bracketed (interior to the grid).
- **Phase 2:** `I*` is confirmed if its mean `val_rank_ic` across seeds 0–2 beats
  the `0.0` baseline with little/no range overlap. If confirmed, promoting the
  default (`rezero_init` 0.0 → `I*`) is worth a follow-up — but that changes
  default training behavior, so spec/plan it separately rather than editing the
  default here.

## Cost & caveats

- ~7 new 10k-step runs (Phase 1: 3; Phase 2: 4), reusing 2 already done.
- This is a single fixed 10k-step proxy budget. The *ranking* of inits should
  hold, but the absolute optimum may shift at full budget — re-check `I*` at the
  full training budget before locking it in as a default.
- Keep `rezero_lr` at its default throughout; the diagnostic showed raising it to
  open the gates backfires. This sweep isolates initialization.
