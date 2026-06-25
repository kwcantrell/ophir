# Fix Misspelled Core Abstractions — Design

**Date:** 2026-06-25
**Status:** Approved, ready for implementation plan

## Problem

A graphify knowledge-graph pass over the repo surfaced that several of the
codebase's most-connected "god nodes" — the core abstractions referenced from
almost every module — carry spelling errors in their identifiers. Because these
types are central, the typos have propagated across the whole tree, including
tests and scripts. They are the kind of debt that becomes invisible precisely
because it is everywhere.

Confirmed scope (whole-word `grep` over `src/`, `tests/`, `scripts/`):

| Misspelled identifier | Correct | Occurrences | Files |
|-----------------------|---------|-------------|-------|
| `OHLCMulitClassPredictor` / `OHLCMulitClassPredictorInput` / `OHLCMulitClassParameters` (`Mulit`) | `OHLCMultiClass*` | 90 | 13 |
| `StockHanlder` (`Hanlder`) | `StockHandler` | 45 | 13 |
| `load_fintuned_ckpt` (`fintuned`) | `load_finetuned_ckpt` | 4 | 2 |

The `fintuned` case is also an internal inconsistency: the same module already
spells the sibling helpers correctly (`_latest_finetuned_ckpt`, the `finetuned`
CLI flag, `FINETUNE_NAME`).

## Goal

Rename the five misspelled Python identifiers to their correct spellings, with
zero behavioral change, gated by the existing test and type-check suite.

## Scope

### In scope — rename these Python identifiers

- `OHLCMulitClassPredictor` → `OHLCMultiClassPredictor` (`src/ophir/models.py:417`)
- `OHLCMulitClassPredictorInput` → `OHLCMultiClassPredictorInput` (`src/ophir/model_data.py:12`)
- `OHLCMulitClassParameters` → `OHLCMultiClassParameters` (`src/ophir/models.py:54`)
- `StockHanlder` → `StockHandler` (`src/ophir/ticker.py:550`)
- `load_fintuned_ckpt` → `load_finetuned_ckpt` (`src/ophir/register.py:563` + overloads)

All call sites across `src/ophir`, `tests/`, and `scripts/` are updated in the
same change.

### Out of scope — persisted on-disk strings

The following are **not** renamed, because they name files that already exist on
disk; changing them would break resolution of previously-saved checkpoints:

- `FINETUNE_NAME = "ophire-ohlc-finetuned"` (`register.py:38`) — note the
  separate `ophire` → `ophir` wart. **Deliberately left as known debt**: fixing
  it would require a checkpoint-migration step (renaming saved files and any
  references in run metadata), which is not justified for a cosmetic string.
- `BASE_NAME` and any `ModelCheckpoint(filename=...)` literals.

### Decisions

1. **`load_fintuned_ckpt`: hard rename, no deprecated alias.** It is a
   package-internal API with no known external consumers; an alias would only
   preserve the misspelling.
2. **No checkpoint migration.** The `ophire-ohlc-finetuned` string and all other
   persisted filenames are untouched (see above).

## Why this is safe for existing checkpoints

Checkpoints are saved/loaded through PyTorch-Lightning
(`ModelCheckpoint` + `save_hyperparameters()` + `load_from_checkpoint`):

- `state_dict` keys derive from **module attribute paths**, not class names. The
  attribute holding the predictor is `self.ohlc_predictor` — that name is not
  changing — so every key is unaffected.
- `LightningOHLCPredictor.__init__` takes only **primitive scalars**
  (`emb_dim`, `num_layers`, `lr`, …). The `OHLCMulitClassParameters` object is
  constructed as a *local* inside `__init__`; it is never an `__init__`
  argument, so `save_hyperparameters()` stores only primitives. No renamed class
  is pickled into any checkpoint.
- `register.py` reads checkpoints with `torch.load(..., weights_only=False)`
  only to diagnose feature-dim drift; it unpickles primitives + tensors, not
  custom classes.

Therefore the rename cannot break loading of any existing checkpoint, and no
backward-compatibility shim is required.

## Implementation approach

Mechanical, behavior-preserving sweep — no logic changes.

1. **Baseline (must be green before any edit):**
   - `uv run pytest`
   - `uv run mypy src/ophir`
2. **Rename one identifier at a time**, whole-word matched, across `src/`,
   `tests/`, `scripts/`, and docstrings/comments that reference the symbol.
3. **After each rename, re-run the gate:**
   - `uv run mypy src/ophir` (strict mode turns any missed reference into a
     type error)
   - `uv run pytest` (`filterwarnings = error`; offline + CPU-only)
4. **Final pass:** `uv run ruff check . && uv run ruff format --check .`
5. Update the `[Unreleased]` section of `CHANGELOG.md`.

## Testing strategy

This is a rename with no behavioral change, so the **regression gate is the
existing suite**, not new tests:

- The full `pytest` run (which must stay offline and CPU-only) verifies imports,
  construction, and behavior are unchanged.
- `mypy --strict` verifies there are no dangling references to the old names
  anywhere in `src/ophir`.
- A final `grep` for `Mulit`, `Hanlder`, and `fintuned` across `src/`, `tests/`,
  and `scripts/` must return zero hits (excluding the intentionally-retained
  persisted strings).

## Success criteria

- All five identifiers are renamed everywhere they are used.
- `grep -rn 'Mulit\|Hanlder\|fintuned' src tests scripts` returns no Python
  identifier hits (the `ophire-ohlc-finetuned` string literal is the only
  documented exception).
- `pytest`, `mypy src/ophir`, and `ruff check`/`format --check` are all green.
- `CHANGELOG.md` `[Unreleased]` notes the rename.
