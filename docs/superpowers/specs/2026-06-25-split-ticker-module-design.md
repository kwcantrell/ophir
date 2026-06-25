# Split `ticker.py` Into a Focused `ophir.ticker` Package — Design

**Date:** 2026-06-25
**Status:** Approved, ready for implementation plan

## Problem

A graphify knowledge-graph pass over the repo flagged `src/ophir/ticker.py` as
the clearest "one file doing too much" in the codebase:

- **999 lines** — ~1.5× the next-largest module.
- It is the source file behind **six** distinct graph communities (Stock
  Handler, Stock Streamer, Streaming Datasets, Splits & Symbol Fetch, Feature
  Extraction, Ticker Window Helpers).
- It owns the `StockHandler` god node (34 edges, the highest-betweenness
  data-layer bridge).

The file stacks seven separable responsibilities:

| Lines (approx) | Responsibility | Symbols |
|----------------|----------------|---------|
| 27–112 | Parquet discovery + window-index math | `get_stock_parquets`, `get_starts`, `get_start_dates` |
| 113–270 | Symbol & split fetching (network) | `get_sp_500_symbols`, `get_splits`, `StockSplit` |
| 271–412 | Cleaning + feature extraction | `clean_daily_ohlcv`, `extract_features` |
| 413–548 | Streaming primitive | `StockStreamer` |
| 549–765 | The handler (god node) | `StockHandler` |
| 766–870 | Model-data builders | `extract_model_data`, `build_latest_inputs` |
| 871–999 | Torch datasets | `StockStreamerDataset`, `StockHandlerDataset` |

The test suite is *already* organized along these exact seams
(`test_ticker_helpers.py`, `test_ticker_network.py`, `test_ticker_features.py`,
`test_ticker_streamer.py`, `test_ticker_handler.py`, `test_ticker_datasets.py`,
`test_ticker_forecast_inputs.py`), which confirms the split is natural rather
than imposed.

## Goal

Move `ticker.py`'s contents into a focused `ophir.ticker` *package* of small,
single-responsibility modules, with **zero behavioral change** and **zero
changes to the 22 external import sites**, gated by the existing test and
type-check suite.

## Verified dependency layering

A reference scan of intra-file symbol usage (ignoring docstring mentions)
yields a clean acyclic dependency graph:

```
paths     (no internal deps)
splits    (no internal deps)
features  (no internal deps)        # the lone features->handler edge is a
                                    # docstring cross-ref, not real coupling
streamer  -> paths, splits, features
handler   -> streamer, paths, splits, features
inputs    -> handler, streamer, features
datasets  -> handler, inputs, streamer
```

Topological order for module creation: `paths`/`splits`/`features` →
`streamer` → `handler` → `inputs` → `datasets` → `__init__`.

## Design

### Target structure

Replace the single file `src/ophir/ticker.py` with a package
`src/ophir/ticker/` containing:

| New module | Contents | Internal deps |
|-----------|----------|---------------|
| `paths.py` | `get_stock_parquets`, `get_starts`, `get_start_dates` | — |
| `splits.py` | `StockSplit`, `get_sp_500_symbols`, `get_splits` | — |
| `features.py` | `clean_daily_ohlcv`, `extract_features` | — |
| `streamer.py` | `StockStreamer` | paths, splits, features |
| `handler.py` | `StockHandler` | streamer, paths, splits, features |
| `inputs.py` | `extract_model_data`, `build_latest_inputs` | handler, streamer, features |
| `datasets.py` | `StockStreamerDataset`, `StockHandlerDataset` | handler, inputs, streamer |
| `__init__.py` | explicit re-exports of the full public API | all of the above |

Each module stays under ~220 LOC and carries exactly one responsibility.

### Compatibility shim

`src/ophir/ticker/__init__.py` re-exports every public name the old module
exposed, so `from ophir.ticker import StockHandler` (and all other existing
imports) resolve unchanged. The `__init__` also defines `__all__` listing the
public surface. **No external import site is edited** — only the physical
location of the code changes.

### Decisions

1. **Package, not flat sibling modules.** `ophir/ticker/__init__.py` as the
   compat shim is cleaner than leaving a `ticker.py` stub beside new sibling
   files.
2. **`StockStreamer` and `StockHandler` live in separate modules.** The handler
   is the god node and earns its own focused file; the streamer is its own
   primitive.
3. **Builders module is named `inputs.py`, not `model_data.py`.** A top-level
   `src/ophir/model_data.py` already exists; reusing that name inside the
   package would be confusing.
4. **Zero call-site churn.** All external code keeps importing from
   `ophir.ticker`. Only the new intra-package imports are added. Private helpers
   (leading-underscore functions) move with the public symbol that uses them.

### Out of scope

- No signature, behavior, or logic changes to any moved symbol.
- No renaming of public symbols.
- No updates to external import statements (the shim makes them unnecessary).
- `register.py` (the runner-up grab-bag) is **not** touched — it is a separate
  future candidate.

## Testing strategy

This is a behavior-preserving move, so the **regression gate is the existing
suite**, not new tests:

- `uv run pytest` (offline + CPU-only, `filterwarnings = error`) must stay green
  — it exercises every moved symbol through the already-seam-aligned test files.
- `uv run mypy src/ophir` (strict) must stay clean — it catches any broken
  intra-package reference as a type error.
- **Public-API parity check:** the sorted set of public names exported by
  `ophir.ticker` (`[n for n in dir(ophir.ticker) if not n.startswith("_")]`,
  or the module's `__all__`) must be identical before and after the split.

## Success criteria

- `src/ophir/ticker.py` is gone; `src/ophir/ticker/` exists with the eight files
  above, each holding one responsibility.
- `import ophir.ticker as t` plus every existing `from ophir.ticker import ...`
  resolves exactly as before (public-API parity check passes).
- `pytest`, `mypy src/ophir`, and `ruff check`/`format --check` are all green.
- No external file's import statements changed.
- `CHANGELOG.md` `[Unreleased]` notes the internal reorganization.
