# Split `register.py` Into a Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `src/ophir/register.py` (641 LOC) into a focused `src/ophir/register/` package of single-responsibility modules with zero behavioral change and zero edits to external import sites.

**Architecture:** Same proven pattern as the ticker split — `git mv register.py register/__init__.py` (package conversion, everything keeps working), then extract one responsibility at a time out of `__init__.py` into a sibling submodule, in verified topological order, re-exporting each. The dependency graph is a star: every group depends only on `layout`. Each extraction is gated by the existing test + type suite plus a public-surface parity check that covers **constants, functions, and the Typer `app`**.

**Tech Stack:** Python 3.10+, `uv`, `pytest` (offline + CPU-only, `filterwarnings = error`), `mypy` (strict, 3.10), `ruff` (lint+format, 3.12; `known-first-party = ["ophir"]`), `typer`.

## Global Constraints

- mypy targets 3.10 with `strict = True`; ruff targets 3.12. Do not change either to match the other.
- `pytest` runs `filterwarnings = error`; never touch network, CUDA, or the `.ophir/` layout.
- Keep `src/ophir` fully typed; move each symbol's definition + decorators + NumPy docstring verbatim. Private helpers move with the public symbol that uses them.
- Absolute first-party imports (`from ophir.register.<module> import ...`).
- **Behavior-preserving move only** — no signature, logic, or public-name changes. `FINETUNE_NAME = "ophire-ohlc-finetuned"` stays as-is (documented persisted-string debt).
- **No external import statement changes.** All outside code keeps importing from `ophir.register`. The only permitted external edits are `mocker.patch`/`setattr` target retargets (flagged explicitly), as in the ticker split.
- **The `.ophir/` directory location MUST NOT move.** It is anchored to the `ophir` package dir (`src/ophir/.ophir`); see Task 3.

## Public surface that must stay importable from `ophir.register`

Constants: `OPHIR_DIR DATA_DIR MODEL_DIR BASE_NAME FINETUNE_NAME BASE_MODEL_CKPT BASE_BEST_CKPT TIME_MODIFIER EPOCH_MODIFIER`
Functions: `get_default_data_days_dir quality_stats_path clear_ignore_symbols set_ignore_symbols fetch_ignore_symbols_list set_quality_symbols fetch_quality_symbols_list clear_quality_symbols _best_checkpoint_callback fetch_base_trainer fetch_finetune_trainer predict_trainer _latest_base_ckpt _latest_finetuned_ckpt _resolve_base_ckpt_path _feature_dim_mismatch _raise_load_error_with_hint load_base_model_ckpt load_finetuned_ckpt get_massive_client massive_key`
Other: `app` (a `typer.Typer`)

(The underscore names `_best_checkpoint_callback` and `_latest_base_ckpt` are imported by name in tests, so they must be re-exported even though they are private.)

## THE GATE (run after every extraction; referenced as "run THE GATE")

```bash
uv run ruff check --fix src/ophir/register/ && uv run ruff format src/ophir/register/
uv run ruff check src/ophir/register/          # must end "All checks passed!"
uv run mypy src/ophir                            # must end "Success: no issues found"
uv run pytest -q -p no:cacheprovider; echo "pytest exit=$?"   # gate is exit=0
uv run python - <<'PY'
import typer
import ophir.register as r
names = ("OPHIR_DIR DATA_DIR MODEL_DIR BASE_NAME FINETUNE_NAME BASE_MODEL_CKPT BASE_BEST_CKPT "
         "TIME_MODIFIER EPOCH_MODIFIER get_default_data_days_dir quality_stats_path "
         "clear_ignore_symbols set_ignore_symbols fetch_ignore_symbols_list set_quality_symbols "
         "fetch_quality_symbols_list clear_quality_symbols _best_checkpoint_callback "
         "fetch_base_trainer fetch_finetune_trainer predict_trainer _latest_base_ckpt "
         "_latest_finetuned_ckpt _resolve_base_ckpt_path _feature_dim_mismatch "
         "_raise_load_error_with_hint load_base_model_ckpt load_finetuned_ckpt get_massive_client "
         "massive_key app").split()
missing = [n for n in names if not hasattr(r, n)]
assert not missing, f"MISSING from ophir.register: {missing}"
assert isinstance(r.app, typer.Typer), "register.app must be a typer.Typer"
assert r.OPHIR_DIR.endswith("/src/ophir/.ophir"), f"OPHIR_DIR moved: {r.OPHIR_DIR}"
assert r.DATA_DIR.endswith("/src/ophir/.ophir/data"), f"DATA_DIR moved: {r.DATA_DIR}"
assert r.MODEL_DIR.endswith("/src/ophir/.ophir/model"), f"MODEL_DIR moved: {r.MODEL_DIR}"
print("PARITY OK:", len(names), "names; app is Typer; .ophir/ anchor unchanged")
PY
```

A task passes only when ruff says "All checks passed!", mypy says "Success", `pytest exit=0`, and the parity block prints `PARITY OK`.

---

### Task 1: Baseline and public-surface snapshot

**Files:** none (verification only)

- [ ] **Step 1: Clean tree + record pre-refactor SHA**

Run:
```bash
git status --porcelain   # expect empty
git rev-parse HEAD | tee /tmp/register_split_base_sha.txt
```

- [ ] **Step 2: Snapshot the public surface and the `.ophir/` anchor**

Run:
```bash
uv run python - <<'PY'
import typer
from ophir import register as r
print("OPHIR_DIR =", r.OPHIR_DIR)
print("DATA_DIR  =", r.DATA_DIR)
print("MODEL_DIR =", r.MODEL_DIR)
print("app is Typer:", isinstance(r.app, typer.Typer))
print("public:", sorted(n for n in dir(r) if not n.startswith("__")))
PY
```
Expected: `OPHIR_DIR` ends `/src/ophir/.ophir`, `app is Typer: True`, and the public list is a superset of the names in "Public surface" above. Record this output.

- [ ] **Step 3: Baseline mypy + pytest**

Run: `uv run mypy src/ophir && uv run pytest -q -p no:cacheprovider; echo "pytest exit=$?"`
Expected: `Success: no issues found ...` and `pytest exit=0`.

No commit.

---

### Task 2: Convert the module to a package

**Files:**
- Rename: `src/ophir/register.py` → `src/ophir/register/__init__.py`

- [ ] **Step 1: Move the file into a package**

Run:
```bash
mkdir -p src/ophir/register
git mv src/ophir/register.py src/ophir/register/__init__.py
```

- [ ] **Step 2: Run THE GATE**

Expected: all green, `PARITY OK`. (register.py has no relative imports, so the conversion alone is behavior-neutral. The parity check confirms `.ophir/` is still anchored at `src/ophir/.ophir` — `__init__.py` lives at the same depth as the old module, so `os.path.abspath(__file__)`'s dir is still `src/ophir/register`… NOTE: it is now one level deeper. If the anchor assertion FAILS here, that is expected and is fixed in Task 3; in that case skip the assertion this once by confirming only ruff+mypy+pytest, and proceed.)

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "refactor(register): convert module to package (no code moved)"
```

> Note: as a package `__init__.py`, `__file__` is `src/ophir/register/__init__.py`, so `os.path.dirname(__file__)` is now `src/ophir/register` — meaning the unchanged code would create `.ophir/` one level too deep. This is corrected in Task 3 by moving the anchor into `layout.py` with a two-level `dirname`. Until then the parity anchor assertion may fail; rely on pytest (which uses `tmp_path`, not the real `.ophir/`) staying green for Task 2.

---

### Task 3: Extract `layout.py` (constants, dir-creation, path helpers) — anchor-critical

**Files:**
- Create: `src/ophir/register/layout.py`
- Modify: `src/ophir/register/__init__.py`

**Interfaces:**
- Produces: `ophir.register.layout` exporting the 9 constants, `get_default_data_days_dir`, `quality_stats_path`, and running the `.ophir/` dir-creation side effect on import.
- Consumes: nothing (leaf).

- [ ] **Step 1: Create `layout.py` with the corrected anchor**

Create `src/ophir/register/layout.py`. Copy the constant block and the
dir-creation side effect from `__init__.py` **verbatim except** the `current_dir`
computation, which must go up **two** levels (the file is now one deeper) so the
`.ophir/` root stays at the `ophir` package dir:

```python
"""On-disk ``.ophir/`` layout: data/model directory constants and path helpers.

The ``.ophir/`` root is anchored to the ``ophir`` package directory (one level
above this ``register`` subpackage) and its data/model subdirectories are
created on import, preserving the location used before ``register`` became a
package.
"""

from __future__ import annotations

import os

# layout.py lives at src/ophir/register/layout.py; the .ophir/ root has always
# been anchored at the ophir package dir (src/ophir/), so go up TWO levels.
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPHIR_DIR = os.path.join(current_dir, ".ophir")
DATA_DIR = os.path.join(OPHIR_DIR, "data")
MODEL_DIR = os.path.join(OPHIR_DIR, "model")
BASE_NAME = "ophir-ohlc-base"
FINETUNE_NAME = "ophire-ohlc-finetuned"
BASE_MODEL_CKPT = os.path.join(MODEL_DIR, f"{BASE_NAME}.ckpt")
BASE_BEST_CKPT = os.path.join(MODEL_DIR, f"{BASE_NAME}-best.ckpt")
TIME_MODIFIER = "-time-check"
EPOCH_MODIFIER = "best-{epoch:02d}-{val_loss:.5f}"

if not os.path.exists(OPHIR_DIR):
    os.makedirs(OPHIR_DIR)

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
```

Then paste `get_default_data_days_dir` and `quality_stats_path` verbatim from
`__init__.py` after the constants (add the imports they need — likely just
`os`, already present).

- [ ] **Step 2: Remove the moved block from `__init__.py` and re-export**

Delete the `current_file_path`/`current_dir`/constants block (old lines ~29–51),
the dir-creation `if not os.path.exists(...)` blocks, and the two functions from
`__init__.py`. Add near the top of `__init__.py`:

```python
from ophir.register.layout import (
    BASE_BEST_CKPT,
    BASE_MODEL_CKPT,
    BASE_NAME,
    DATA_DIR,
    EPOCH_MODIFIER,
    FINETUNE_NAME,
    MODEL_DIR,
    OPHIR_DIR,
    TIME_MODIFIER,
    get_default_data_days_dir,
    quality_stats_path,
)
```

- [ ] **Step 3: Run THE GATE (anchor assertion now active)**

The parity block's `OPHIR_DIR.endswith("/src/ophir/.ophir")` assertion must now
PASS — this is the proof the directory did not move. Resolve any F401/F821 as in
the ticker split. Confirm `PARITY OK`.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor(register): extract layout.py (constants + .ophir anchor + path helpers)"
```

---

### Task 4: Extract `symbols.py` (ignore/quality symbol lists)

**Files:**
- Create: `src/ophir/register/symbols.py`
- Modify: `src/ophir/register/__init__.py`

**Interfaces:**
- Produces: `ophir.register.symbols` exporting `clear_ignore_symbols`, `set_ignore_symbols`, `fetch_ignore_symbols_list`, `set_quality_symbols`, `fetch_quality_symbols_list`, `clear_quality_symbols`.
- Consumes: `ophir.register.layout` (DATA_DIR and friends; add the names these functions reference — ruff F821 lists them).

- [ ] **Step 1: Create the submodule**

Create `src/ophir/register/symbols.py` with this header, then paste the six
functions verbatim, adding `from ophir.register.layout import <names used>`:

```python
"""Persisted ignore / quality symbol-list management."""

from __future__ import annotations

import os
```

- [ ] **Step 2: Remove from `__init__.py` and re-export**

Delete the six functions from `__init__.py`. Add:

```python
from ophir.register.symbols import (
    clear_ignore_symbols,
    clear_quality_symbols,
    fetch_ignore_symbols_list,
    fetch_quality_symbols_list,
    set_ignore_symbols,
    set_quality_symbols,
)
```

- [ ] **Step 3: Run THE GATE**

Resolve F821 (layout names to import) / F401 (now-unused `__init__` imports). Confirm `PARITY OK`.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor(register): extract symbols.py (ignore/quality symbol lists)"
```

---

### Task 5: Extract `trainers.py` (Lightning Trainer factories)

**Files:**
- Create: `src/ophir/register/trainers.py`
- Modify: `src/ophir/register/__init__.py`

**Interfaces:**
- Produces: `ophir.register.trainers` exporting `fetch_base_trainer`, `fetch_finetune_trainer`, `predict_trainer`, `_best_checkpoint_callback`.
- Consumes: `ophir.register.layout` (`MODEL_DIR`, `EPOCH_MODIFIER`, `TIME_MODIFIER`, etc.; add what F821 lists). External: `lightning`.

- [ ] **Step 1: Create the submodule**

Create `src/ophir/register/trainers.py` with this header, then paste
`_best_checkpoint_callback`, `fetch_base_trainer`, `fetch_finetune_trainer`,
`predict_trainer` verbatim, adding `from ophir.register.layout import <names used>`:

```python
"""PyTorch-Lightning ``Trainer`` factories for base / finetune / predict."""

from __future__ import annotations

import os
from datetime import timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint
```

- [ ] **Step 2: Remove from `__init__.py` and re-export**

Delete the four functions from `__init__.py`. Add:

```python
from ophir.register.trainers import (
    _best_checkpoint_callback,
    fetch_base_trainer,
    fetch_finetune_trainer,
    predict_trainer,
)
```

Because `_best_checkpoint_callback` is private but imported by tests, ruff may
flag it F401 (not in `__all__`). If so, change its line to the redundant
re-export form `_best_checkpoint_callback as _best_checkpoint_callback`.

- [ ] **Step 3: Run THE GATE**

Resolve F821/F401. The lazy `import lightning` lines inside the functions stay
as they are (runtime-local imports); only module-level annotations go under
`TYPE_CHECKING`. Confirm `PARITY OK`.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor(register): extract trainers.py (Lightning Trainer factories)"
```

---

### Task 6: Extract `checkpoints.py` (resolution + loaders + error hints)

**Files:**
- Create: `src/ophir/register/checkpoints.py`
- Modify: `src/ophir/register/__init__.py`

**Interfaces:**
- Produces: `ophir.register.checkpoints` exporting `_latest_base_ckpt`, `_latest_finetuned_ckpt`, `_resolve_base_ckpt_path`, `_feature_dim_mismatch`, `_raise_load_error_with_hint`, `load_base_model_ckpt`, `load_finetuned_ckpt`.
- Consumes: `ophir.register.layout` (`MODEL_DIR`, `BASE_NAME`, `BASE_BEST_CKPT`, `TIME_MODIFIER`, etc.; add what F821 lists). External: `ophir.training_models.LightningOHLCPredictor`, `torch`.

- [ ] **Step 1: Create the submodule**

Create `src/ophir/register/checkpoints.py` with this header, then paste the seven
symbols verbatim (preserving the `@overload` stacks on `load_base_model_ckpt` and
`load_finetuned_ckpt`), adding `from ophir.register.layout import <names used>`:

```python
"""Checkpoint path resolution, model loaders, and load-error hints."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, NoReturn, overload

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from ophir.training_models import LightningOHLCPredictor
```

- [ ] **Step 2: Remove from `__init__.py` and re-export**

Delete the seven symbols from `__init__.py`. Add (using the redundant `as` form
for the private names tests import, so ruff does not flag them):

```python
from ophir.register.checkpoints import (
    _feature_dim_mismatch as _feature_dim_mismatch,
    _latest_base_ckpt as _latest_base_ckpt,
    _latest_finetuned_ckpt as _latest_finetuned_ckpt,
    _raise_load_error_with_hint as _raise_load_error_with_hint,
    _resolve_base_ckpt_path as _resolve_base_ckpt_path,
    load_base_model_ckpt,
    load_finetuned_ckpt,
)
```

- [ ] **Step 3: Run THE GATE**

Resolve F821/F401. Confirm `PARITY OK`.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor(register): extract checkpoints.py (resolution + loaders + hints)"
```

---

### Task 7: Extract `client.py` (MASSIVE client + Typer app + `massive_key`)

**Files:**
- Create: `src/ophir/register/client.py`
- Modify: `src/ophir/register/__init__.py`

**Interfaces:**
- Produces: `ophir.register.client` exporting `get_massive_client`, `app` (a `typer.Typer`), and the `massive_key` command. After this task `__init__.py` holds only re-exports.
- Consumes: `ophir.register.layout` if referenced (F821 lists it). External: `massive.RESTClient`, `typer`.

- [ ] **Step 1: Create the submodule**

Create `src/ophir/register/client.py` with this header, then paste
`get_massive_client`, the `app = typer.Typer()` line, and the `massive_key`
command (with its `@app.command()` decorator and `Annotated` parameter)
verbatim:

```python
"""MASSIVE API client and the ``massive_key`` Typer command."""

from __future__ import annotations

import os
from typing import Annotated

import typer
from massive import RESTClient
```

- [ ] **Step 2: Remove from `__init__.py` and re-export**

Delete `get_massive_client`, `app`, and `massive_key` from `__init__.py`. Add:

```python
from ophir.register.client import app, get_massive_client, massive_key
```

`cli.py` references `register.app`; this re-export makes `register.app` resolve
to `client.app` (the same object the `massive_key` command is registered on), so
`app.add_typer(register.app, ...)` is unchanged.

- [ ] **Step 3: Run THE GATE**

Resolve F821/F401. Confirm `PARITY OK` (this is the first point `app` lives in a
submodule — the `isinstance(r.app, typer.Typer)` assertion proves the re-export
works). After this task `__init__.py` should contain only re-export lines.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor(register): extract client.py (MASSIVE client + Typer app)"
```

---

### Task 8: Finalize `__init__.py`, changelog, and final verification

**Files:**
- Modify: `src/ophir/register/__init__.py`, `CHANGELOG.md`

- [ ] **Step 1: Add the package docstring and `__all__`**

Ensure `src/ophir/register/__init__.py` is the package docstring, the five
re-export blocks (layout / symbols / trainers / checkpoints / client), and an
`__all__` listing the public surface (the constants, the public functions, and
`app` — private underscore names stay re-exported via the `as` form but are NOT
listed in `__all__`). The docstring:

```python
"""Filesystem, checkpoint, and Lightning ``Trainer`` helpers.

This package owns the on-disk ``.ophir/`` layout (:mod:`ophir.register.layout`),
ignore/quality symbol-list management (:mod:`ophir.register.symbols`), the
Lightning ``Trainer`` factories (:mod:`ophir.register.trainers`), checkpoint
resolution and loaders (:mod:`ophir.register.checkpoints`), and the MASSIVE API
client plus its ``massive_key`` Typer command (:mod:`ophir.register.client`).
The full public surface — constants, functions, and the Typer ``app`` — is
re-exported here, so ``from ophir import register`` and ``register.<name>`` work
exactly as before.
"""
```

Set `__all__` to (sorted): the 9 constants, `app`, `get_massive_client`,
`massive_key`, the symbol-list functions, the trainer factories,
`load_base_model_ckpt`, `load_finetuned_ckpt`, `get_default_data_days_dir`,
`quality_stats_path`.

- [ ] **Step 2: Run THE GATE**

Confirm green + `PARITY OK`.

- [ ] **Step 3: Verify old module gone and package complete**

Run:
```bash
test ! -f src/ophir/register.py && echo "old module removed: OK"
ls src/ophir/register/
```
Expected: `register.py` absent; lists `__init__.py checkpoints.py client.py layout.py symbols.py trainers.py`.

- [ ] **Step 4: Confirm no external import sites changed**

Run:
```bash
git diff --name-only "$(cat /tmp/register_split_base_sha.txt)"..HEAD -- ':!src/ophir/register/' ':!docs/' ':!CHANGELOG.md'
```
Expected: empty, **or** only test files whose change is a `mocker.patch`/`setattr`
target retarget (verify the diff shows only patch-string edits, not import edits).
The `tests/test_patch_targets.py` guard also enforces that any `ophir.register.*`
patch target still resolves.

- [ ] **Step 5: Add the changelog entry**

In `CHANGELOG.md`, under `## [Unreleased]` → `### Changed` (create the section if
absent, else append), add:

```markdown
- Reorganized `ophir.register` from a single 641-line module into a focused
  package (`layout`, `symbols`, `trainers`, `checkpoints`, `client`) with a
  re-export `__init__`. Public API and behavior are unchanged — constants, the
  Typer `app`, and `from ophir import register` / `register.<name>` all resolve
  as before, and the `.ophir/` directory location is preserved.
```

- [ ] **Step 6: Final full gate + commit**

Run THE GATE once more (green + `PARITY OK`), then:

```bash
git add -A && git commit -m "refactor(register): finalize package __init__ and changelog"
```

---

## Self-Review

**1. Spec coverage:**
- Package with 6 files (`layout`/`symbols`/`trainers`/`checkpoints`/`client`/`__init__`) → Tasks 2–8. ✓
- Re-export shim incl. constants + `app`, zero call-site changes → Task 2 + per-task re-exports + Task 8 Step 4. ✓
- `.ophir/` location preserved → Task 3 (two-level anchor) + THE GATE anchor assertion. ✓
- `app`/`massive_key` in `client.py`, `register.app` still works for `cli.py` → Task 7. ✓
- Private names (`_latest_base_ckpt`, `_best_checkpoint_callback`, …) stay importable → redundant `as` re-exports in Tasks 5–6 + parity list. ✓
- `FINETUNE_NAME` "ophire" untouched → it is moved verbatim in Task 3, never edited. ✓
- Verified-acyclic star order (layout first) → Tasks 3→7. ✓
- Behavior-preserving, regression gate = existing suite + parity (constants/app/anchor) → THE GATE. ✓
- CHANGELOG note → Task 8 Step 5. ✓
- Other large files untouched → not referenced by any task. ✓

**2. Placeholder scan:** No TBD/TODO. The only non-verbatim instruction ("add the layout names this function references") is backed by ruff F821 (lists every missing name) and F401 (lists every unused one). `layout.py`'s content, all re-export blocks, the parity check, the anchor formula, and the changelog text are given exactly.

**3. Type consistency:** Submodule names (`layout`/`symbols`/`trainers`/`checkpoints`/`client`) and every public name are identical across THE GATE's parity list, each task's interfaces, the re-export blocks, and Task 8's `__init__`. The `as`-form private re-exports in Tasks 5–6 match the names asserted in the parity list.
