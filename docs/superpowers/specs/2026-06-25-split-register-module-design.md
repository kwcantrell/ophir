# Split `register.py` Into a Focused `ophir.register` Package — Design

**Date:** 2026-06-25
**Status:** Approved, ready for implementation plan

## Problem

`src/ophir/register.py` (641 lines) is the runner-up "one file doing too much"
flagged by the graphify knowledge-graph passes — the same shape as the
`ticker.py` monolith that was just split. Its code is the source behind ~6 graph
communities (Register & Symbol Lists, Checkpoint Loaders, Trainer Factories,
Canonical Checkpoint Resolution, Feature-Dim Mismatch) and it stacks six
separable responsibilities:

| Lines (approx) | Responsibility |
|----------------|----------------|
| 34–42 | `.ophir/` filesystem-layout constants + path helpers |
| 301–410 | ignore / quality symbol-list management |
| 54–220, 608 | PyTorch-Lightning `Trainer` factories |
| 243–300, 411–607 | checkpoint resolution + loaders + error hints |
| 221, 622–end | MASSIVE API client + its `massive_key` Typer command |

CLAUDE.md already describes the module as "`.ophir/` filesystem layout,
checkpoint loaders, `Trainer` factories" — three concerns named in one line.

## Goal

Move `register.py`'s contents into a focused `ophir.register` *package* of
single-responsibility modules, with **zero behavioral change** and **zero edits
to external import sites**, gated by the existing test + type-check suite.

## Blast radius (verified)

18 files import `register`. **16 use `from ophir import register`** (whole-module
import, then `register.X`); only 2 use `from ophir.register import X`. Both forms
resolve unchanged once `register` is a package whose `__init__` re-exports the
public surface, so no external import statement changes.

`register.app` (a `typer.Typer()` carrying the `massive_key` command) is part of
the public surface — `cli.py` calls `app.add_typer(register.app, name="register")`.
The `__init__` must therefore re-export `app` in addition to the constants and
functions.

## Verified dependency layering

An intra-file symbol-usage scan (ignoring docstrings) yields a clean **star** —
every group depends only on `layout`, which depends on nothing:

```
layout       (no internal deps)
symbols      -> layout
trainers     -> layout
checkpoints  -> layout
client       -> layout
```

Acyclic. Module creation order: `layout` first, then `symbols` / `trainers` /
`checkpoints` / `client` in any order, then `__init__`.
(`checkpoints` and `trainers` also import from `ophir.training_models` and
`lightning` — those are *external* deps, unaffected by the split.)

## Design

### Target structure

Replace the single file `src/ophir/register.py` with a package
`src/ophir/register/` containing:

| New module | Contents | Internal deps |
|-----------|----------|---------------|
| `layout.py` | `OPHIR_DIR`, `DATA_DIR`, `MODEL_DIR`, `BASE_NAME`, `FINETUNE_NAME`, `BASE_MODEL_CKPT`, `BASE_BEST_CKPT`, `TIME_MODIFIER`, `EPOCH_MODIFIER`, `get_default_data_days_dir`, `quality_stats_path` | — |
| `symbols.py` | `clear_ignore_symbols`, `set_ignore_symbols`, `fetch_ignore_symbols_list`, `set_quality_symbols`, `fetch_quality_symbols_list`, `clear_quality_symbols` | layout |
| `trainers.py` | `_best_checkpoint_callback`, `fetch_base_trainer`, `fetch_finetune_trainer`, `predict_trainer` | layout |
| `checkpoints.py` | `_latest_base_ckpt`, `_latest_finetuned_ckpt`, `_resolve_base_ckpt_path`, `_feature_dim_mismatch`, `_raise_load_error_with_hint`, `load_base_model_ckpt`, `load_finetuned_ckpt` | layout |
| `client.py` | `get_massive_client`, the Typer `app`, and the `massive_key` command | layout |
| `__init__.py` | explicit re-exports of every public name **and `app`** + `__all__` | all |

Private helpers (leading-underscore) move with the public symbol that uses them.

### Compatibility shim

`src/ophir/register/__init__.py` re-exports every name the old module exposed —
the layout **constants**, every public function, and the Typer `app` — so
`from ophir import register` + `register.<anything>` and the two
`from ophir.register import X` sites resolve unchanged. `__init__` defines
`__all__` listing the public surface. No external import statement is edited.

### Decisions

1. **Package, not flat sibling modules** — `register/__init__.py` as the compat
   shim, matching the ticker split.
2. **`app` + `massive_key` live in `client.py`** alongside `get_massive_client`.
   The command only stores the client's API key, so MASSIVE access is one
   cohesive unit; a separate `commands.py` would over-split for a single
   command. `register.app` resolves to `client.app` via the `__init__` re-export.
3. **Re-export constants, not just callables.** `register.BASE_BEST_CKPT`,
   `register.DATA_DIR`, `register.MODEL_DIR`, etc. are imported widely (and by
   the recent canonical-checkpoint-resolution work); the `__init__` must expose
   them and the parity check must verify them.
4. **`FINETUNE_NAME = "ophire-ohlc-finetuned"` is unchanged** — the same
   persisted-string debt documented in the misspelled-abstraction work; renaming
   it would break resolution of existing checkpoints.

### Out of scope

- No signature, behavior, or logic changes to any moved symbol.
- No public-symbol renames (including the `ophire` typo above).
- No external import-statement changes (the shim makes them unnecessary).
- `ceiling.py`, `training_models.py`, `train.py`, `evaluate.py` are not touched —
  they are large but cohesive (single concern / single class), not grab-bags.

## Revision (2026-06-25): shared monkeypatched constants

Execution surfaced a coupling the original design missed: 13 tests redirect
`register`'s directories with a single `monkeypatch.setattr(register, "MODEL_DIR", tmp)`
or `"DATA_DIR"`. This worked when every function lived in one module reading one
module-global. Splitting functions into submodules that each bind their own
`from layout import DATA_DIR` copy breaks the single patch point, and some
`MODEL_DIR` tests call functions that land in *both* `trainers` and `checkpoints`,
so there is no clean one-patch-per-submodule mapping.

**Decision (approved): `layout` is the single source of truth for the constants,
accessed live.** Submodules import the `layout` *module* (`from ophir.register import
layout`) and reference `layout.DATA_DIR` / `layout.MODEL_DIR` / `layout.BASE_NAME`
/ etc. **at call time** (never a value-copy `from layout import DATA_DIR`).
`layout.py`'s own helpers read their module-global constants directly. The 13
tests are retargeted from `setattr(register, "<CONST>", ...)` to
`setattr(register.layout, "<CONST>", ...)` — one canonical patch point that every
consumer observes. These test edits are permitted (the analogue of the ticker
split's mock-target retargets) and are flagged explicitly.

## Testing strategy

Behavior-preserving move, so the **regression gate is the existing suite**, not
new tests:

- `uv run pytest` (offline + CPU-only, `filterwarnings = error`) must stay green
  — it exercises every moved symbol, and `tests/test_register.py` /
  `tests/test_patch_targets.py` cover the register surface directly.
- `uv run mypy src/ophir` (strict) must stay clean — catches any broken
  intra-package reference.
- **Public-surface parity check, extended beyond callables:** the set of public
  names exported by `ophir.register` — every constant, every function, and
  `app` — must be identical before and after. Capture the baseline as
  `[n for n in dir(register) if not n.startswith("_")]` plus an explicit assert
  that `register.app` exists and is a `typer.Typer`.
- The `tests/test_patch_targets.py` guard automatically re-validates any test
  that patches `ophir.register.*` internals once the symbols move.

## Success criteria

- `src/ophir/register.py` is gone; `src/ophir/register/` exists with the six
  files above, each holding one responsibility.
- `from ophir import register` + every `register.<name>` access, and both
  `from ophir.register import X` sites, resolve exactly as before
  (public-surface parity check passes, including constants and `app`).
- `pytest`, `mypy src/ophir`, and `ruff check`/`format --check` are all green.
- No external file's import statements changed (mock-target retargets, if any,
  are the only permitted test-side edits — flagged explicitly, as in the ticker
  split).
- `CHANGELOG.md` `[Unreleased]` notes the internal reorganization.
