# Checkpoint promotion & cleanup runbook

`load_base_model_ckpt(time_version=False)` (used by `load_forecasts`,
`evaluate`, `train`, `dashboard`) loads a single **canonical** checkpoint:

    src/ophir/.ophir/model/ophir-ohlc-base-best.ckpt   (register.BASE_BEST_CKPT)

There is no auto-selection — promotion is an explicit copy.

## Promote a checkpoint to canonical

Training writes best-epoch candidates to `…/.ophir/model/candidates/`. To make
one the live base model, copy it onto the canonical path:

    cp src/ophir/.ophir/model/candidates/<chosen>.ckpt \
       src/ophir/.ophir/model/ophir-ohlc-base-best.ckpt

Prefer a `val_rank_ic_near`-monitored candidate (filename contains
`val_rank_ic_near`) over a `val_loss` one — `val_loss` is anti-aligned with
cross-sectional IC (~0.5x peak), per the op-point investigation.

## Clean up the stale candidate zoo (operational, destructive)

The model dir historically accumulated ~125 `val_loss`-monitored checkpoints
(`ophir-ohlc-basebest-epoch=*-val_loss=*.ckpt`, ~15 GB). They no longer affect
resolution (selection is by the exact canonical filename), but waste disk. After
confirming the canonical file loads, archive or delete them:

    # inspect first
    ls -lh src/ophir/.ophir/model/ophir-ohlc-basebest-epoch=*-val_loss=*.ckpt | head

    # then, once satisfied, remove them
    rm src/ophir/.ophir/model/ophir-ohlc-basebest-epoch=*-val_loss=*.ckpt

This is not automated and is never run by the test suite.
