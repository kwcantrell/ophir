# Deterministic canonical-checkpoint resolution — design

Fix the latent bug that makes `load_base_model_ckpt(time_version=False)` raise
`IndexError` (so `load_forecasts` returns `{}` unconditionally), by resolving
"the base model" to an **explicit canonical checkpoint file** instead of a
fragile filename-prefix glob. Establish a clean candidate-save convention.

This is **Piece A** of the "90-day IC-best checkpoint" work. **Piece B** — the
GPU training run that produces a `val_rank_ic_near`-monitored checkpoint — is a
separate, out-of-scope operational task that this design *unblocks* (a fresh
checkpoint becomes loadable by promoting it onto the canonical path).

## Background — the bug (verified 2026-06-23)

`load_base_model_ckpt(time_version=False)` builds a prefix from
`BASE_NAME + EPOCH_MODIFIER` and strips at the first `{`:
`"ophir-ohlc-base" + "best-{epoch:02d}-{val_loss:.5f}"` → `"ophir-ohlc-basebest-"`.
That prefix matches ~125 stale auto-saved checkpoints
(`ophir-ohlc-basebest-epoch=00-val_loss=….ckpt`), so `_latest_base_ckpt` enters
its multi-match branch, filters for `"{filename}-v"` (i.e. `…basebest--v`, which
matches none), and indexes `[-1]` of an empty list → **`IndexError`**.
`load_forecasts` catches it and returns `{}`. The live ophir path is therefore
dead today — masked because the CUDA forward is untested.

Additional findings:

- **No IC-best checkpoint exists.** Every matching file — including the
  hand-placed `ophir-ohlc-base-best.ckpt` (epoch 1, `val_loss`-monitored, no
  `near_offset_k`/`monitor_near_ic` hparams) — predates the op-point work and is
  `val_loss`-monitored. The op-point investigation found `val_loss` is
  anti-aligned with IC (~0.5× peak).
- **`time_version=True` works.** The rolling `-time-check-v<N>` path (125 files,
  all `-v<N>`) resolves fine; only `time_version=False` is broken.
- **A 15 GB zoo.** ~125 stale candidates × ~126 MB clutter `MODEL_DIR` and are
  what triggers the multi-match `IndexError`.
- `BASE_MODEL_CKPT` (`ophir-ohlc-base.ckpt`) is a defined-but-unused constant and
  the file does not exist; the real artifact is `ophir-ohlc-base-best.ckpt`.

## Design decision

Resolve `time_version=False` to an **explicit canonical file**, not a glob. A
fixed known path cannot hit the `IndexError`, cannot be confused by the zoo, and
cannot mis-sort `val_loss` ahead of `val_rank_ic_near`. Promotion of a new
checkpoint becomes a visible, auditable act (copy it onto the canonical path) —
the opposite of the implicit parse-and-sort selection that caused the bug.

## Components

### 1. Extract a pure path resolver (offline-testable)

New `register._resolve_base_ckpt_path(file_name: str | None = None, time_version: bool = True) -> str`
— returns the resolved checkpoint *path* via filesystem listing only, **loading
no model**. `load_base_model_ckpt` calls it and then loads. This makes the
resolution logic unit-testable against a `tmp` `MODEL_DIR` without a 126 MB
Lightning load or CUDA.

Behavior:

```
name = file_name if file_name is not None else BASE_NAME
if time_version:
    return os.path.join(MODEL_DIR, _latest_base_ckpt(name + TIME_MODIFIER))
path = os.path.join(MODEL_DIR, f"{name}-best.ckpt")
if not os.path.exists(path):
    raise FileNotFoundError(f"canonical base checkpoint not found: {path}")
return path
```

- `time_version=False`, default `file_name` → `MODEL_DIR/ophir-ohlc-base-best.ckpt`
  directly. No glob, no version-parse, no `IndexError`.
- A custom `file_name` generalizes to `MODEL_DIR/{file_name}-best.ckpt`.
- Absent canonical → `FileNotFoundError`, which `load_forecasts` already catches
  (its guard is `(IndexError, FileNotFoundError, OSError)`) → `{}`. Degradation
  preserved.
- **The resolver reads the live module global `MODEL_DIR`** (via
  `os.path.join(MODEL_DIR, …)`), not the frozen `BASE_BEST_CKPT` constant, so
  tests can `monkeypatch` `register.MODEL_DIR`.

### 2. Canonical-path constant

Add `BASE_BEST_CKPT = os.path.join(MODEL_DIR, f"{BASE_NAME}-best.ckpt")` next to
the existing constants — a convenience reference for callers and the promotion
runbook. Leave the unused `BASE_MODEL_CKPT` untouched (out of scope).

### 3. `load_base_model_ckpt` uses the resolver

Replace the in-body prefix construction (`file_name += TIME_MODIFIER / +=
EPOCH_MODIFIER`, `.split("{")[0]`, `_latest_base_ckpt`) with a single
`_resolve_base_ckpt_path(file_name, time_version)` call, then
`load_from_checkpoint(path, strict=strict)`. Signature and overloads
(`strict`, `return_ckpt_path`, `file_name`, `time_version`) are unchanged. All
`time_version=False` callers (`load_forecasts`, `evaluate.py` "best-val",
`train.py`, `dashboard.py`) transparently get the canonical file.

### 4. Harden `_latest_base_ckpt` (defensive)

The shared helper still serves the `time_version=True` path. Make it robust to
the two latent footguns rather than `IndexError`:

```
base_paths = sorted(p for p in os.listdir(MODEL_DIR) if filename in p)
if not base_paths:
    raise FileNotFoundError(f"no checkpoint matching {filename!r} in {MODEL_DIR}")
versioned = sorted(
    (int(v.removeprefix(f"{filename}-v").removesuffix(".ckpt")), v)
    for v in base_paths if f"{filename}-v" in v
)
return versioned[-1][1] if versioned else base_paths[-1]
```

- No matches → clear `FileNotFoundError` (not `IndexError`).
- Matches but none versioned → return the sorted last match (not `IndexError`).
- Behavior for the real `-time-check-v<N>` set is unchanged (highest `-v<N>`).

### 5. Candidate-save hygiene (code, going forward)

Point the best-checkpoint `ModelCheckpoint` at a `candidates/` subdir so new
training runs stop cluttering `MODEL_DIR` and canonical promotion stays
unambiguous. In `_best_checkpoint_callback`, change
`dirpath=MODEL_DIR` → `dirpath=os.path.join(MODEL_DIR, "candidates")`.
The time-check rolling `ModelCheckpoint` in `fetch_base_trainer` stays at
`MODEL_DIR` (the `time_version=True` glob resolves it in the root). After Piece B,
the IC-best candidate lands in `candidates/`; promotion = copy it onto
`BASE_BEST_CKPT`.

## Hygiene — operational (NOT automated, NOT in tests)

Selection no longer depends on the ~15 GB of stale `…val_loss=….ckpt`
candidates, so they are harmless to correctness — but they waste disk. The spec
**documents** an archive/delete step for the user to run (or for the agent to run
with explicit confirmation); destructive deletion of existing artifacts is
**not** baked into code or tests. Suggested: move
`ophir-ohlc-basebest-epoch=*` out of `MODEL_DIR` (e.g. to a backup dir) or
delete after confirming the canonical file loads.

## Error handling & degradation

- `time_version=False` + absent canonical → `FileNotFoundError` →
  `load_forecasts` → `{}` (unchanged degradation contract).
- `time_version=True` + no time-check files → `FileNotFoundError` (was a
  confusing `IndexError`).
- Nothing here touches CUDA, the trading safety gate, or `account_mode`.

## Testing

All offline + CPU (`filterwarnings = error`); never load a real model, never
touch CUDA or the package `.ophir/`. Tests `monkeypatch` `register.MODEL_DIR` to
`tmp_path` and create empty marker files.

- `_resolve_base_ckpt_path`:
  - `time_version=False`, canonical present → returns `…/ophir-ohlc-base-best.ckpt`.
  - canonical absent → raises `FileNotFoundError`.
  - custom `file_name="foo"` → `…/foo-best.ckpt`.
  - **the zoo is ignored:** with the canonical file AND many
    `ophir-ohlc-basebest-epoch=00-val_loss=….ckpt` markers present,
    `time_version=False` returns the canonical file and does **not** raise
    `IndexError` (this is the regression test for the reported bug).
  - `time_version=True` with `…-time-check-v1.ckpt`/`-v2.ckpt` → returns `-v2`.
- `_latest_base_ckpt`:
  - matches present but none versioned → returns the sole/sorted match (no raise).
  - no matches → `FileNotFoundError`.
- `_best_checkpoint_callback`: the returned callback's `dirpath` ends with
  `candidates` (construct the real `ModelCheckpoint`; constructing it does not
  write to disk). Update any existing test that asserts `dirpath == MODEL_DIR`.

## Out of scope

- **Piece B** — the GPU `ophir train --val-identity` run that produces the IC-best
  90-day checkpoint. This design unblocks it; promotion is a copy onto
  `BASE_BEST_CKPT`.
- Deleting the existing stale checkpoints (user runs the documented cleanup).
- Repurposing/removing the unused `BASE_MODEL_CKPT` constant.

## Constraints

- mypy `strict = True`, Python 3.10 floor; ruff 3.12; NumPy-style docstrings.
- pytest `filterwarnings = error`; offline + CPU-only; never touch
  network/CUDA/`.ophir/`.
- Update `CHANGELOG.md` `[Unreleased]`.
