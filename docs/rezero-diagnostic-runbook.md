# ReZero gate-opening diagnostic — runbook

Determines whether the transformer depth is helping, by comparing a near-linear
baseline against deep models with the ReZero gates forced open. Requires CUDA.
All arms use the base tier (`emb_dim=128`, `num_heads=8`), `--seed 0`, a fixed
10,000-step budget, and log both `val_rank_ic` and the ReZero gate magnitudes.

Run each arm (each creates a new `csv-logger/version_N` under the model dir —
record which version is which):

```bash
# A. Shallow baseline (near-linear floor)
ophir train --emb-dim 128 --num-heads 8 --num-layers 1 \
  --max-steps 10000 --seed 0 --val-identity --log-rezero-gates

# B. Deep, default (gates expected to stay closed)
ophir train --emb-dim 128 --num-heads 8 --num-layers 6 \
  --max-steps 10000 --seed 0 --val-identity --log-rezero-gates

# C. Deep, high rezero_lr (open gates via LR)
ophir train --emb-dim 128 --num-heads 8 --num-layers 6 --rezero-lr 3e-3 \
  --max-steps 10000 --seed 0 --val-identity --log-rezero-gates

# D. Deep, non-zero init (gates start open)
ophir train --emb-dim 128 --num-heads 8 --num-layers 6 --rezero-init 0.1 \
  --max-steps 10000 --seed 0 --val-identity --log-rezero-gates

# E. Deep, un-decayed rezero schedule (gates keep growing)
ophir train --emb-dim 128 --num-heads 8 --num-layers 6 --decouple-rezero-schedule \
  --max-steps 10000 --seed 0 --val-identity --log-rezero-gates
```

Then compare (point each label at its `version_N` directory under
`<model_dir>/csv-logger/`):

```python
from ophir.dashboard import summarize_rezero_runs

print(summarize_rezero_runs({
    "A_shallow":   "<model_dir>/csv-logger/version_0",
    "B_deep":      "<model_dir>/csv-logger/version_1",
    "C_high_lr":   "<model_dir>/csv-logger/version_2",
    "D_init":      "<model_dir>/csv-logger/version_3",
    "E_undecayed": "<model_dir>/csv-logger/version_4",
}))
```

**Verdict:** if any deep arm (C/D/E) beats the shallow floor (A) on
`val_rank_ic` by a meaningful margin, depth helps once the gates open. If not,
the secondary `rezero_mean_abs` column separates "gates opened but depth didn't
help" (depth genuinely inert) from "gates never opened" (re-run at a longer
budget before concluding).
