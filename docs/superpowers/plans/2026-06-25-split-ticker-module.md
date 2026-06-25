# Split `ticker.py` Into a Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `src/ophir/ticker.py` (999 LOC) into a focused `src/ophir/ticker/` package of single-responsibility modules with zero behavioral change and zero edits to external import sites.

**Architecture:** Convert the module to a package in one `git mv` (`ticker.py` → `ticker/__init__.py`), so every existing `from ophir.ticker import ...` keeps working immediately. Then extract one responsibility at a time out of `__init__.py` into a sibling submodule, in verified topological order, re-exporting each from `__init__.py`. Each extraction is independently gated by the existing test + type suite plus a public-API parity check.

**Tech Stack:** Python 3.10+, `uv` for commands, `pytest` (offline + CPU-only, `filterwarnings = error`), `mypy` (strict, targets 3.10), `ruff` (lint+format, targets 3.12; `known-first-party = ["ophir"]`).

## Global Constraints

- mypy targets Python 3.10 with `strict = True`; ruff targets 3.12. Do not change either to match the other.
- `pytest` runs `filterwarnings = error`; it must never touch the network, CUDA, or the package `.ophir/` layout.
- Keep `src/ophir` fully typed; preserve every NumPy-style docstring verbatim when moving a symbol.
- Imports use absolute first-party paths (`from ophir.ticker.<module> import ...`), matching `known-first-party = ["ophir"]`.
- **Behavior-preserving move only:** no signature, logic, or public-name changes. Moved code (definition + decorators + docstring) is copied verbatim. Private helpers move with the public symbol that uses them.
- **No external import site is edited.** All outside code keeps importing from `ophir.ticker`.
- `register.py` is explicitly out of scope.

## The 14 public symbols (must stay importable from `ophir.ticker` at every step)

```
get_stock_parquets  get_starts  get_start_dates
get_sp_500_symbols  get_splits  StockSplit
clean_daily_ohlcv   extract_features
StockStreamer  StockHandler
extract_model_data  build_latest_inputs
StockStreamerDataset  StockHandlerDataset
```

## THE GATE (run after every extraction; referenced as "run THE GATE" below)

```bash
uv run ruff check --fix src/ophir/ticker/ && uv run ruff format src/ophir/ticker/
uv run ruff check src/ophir/ticker/          # must end "All checks passed!" (no F401 unused / F821 undefined)
uv run mypy src/ophir                          # must end "Success: no issues found"
uv run pytest -q -p no:cacheprovider; echo "pytest exit=$?"   # gate is exit=0
uv run python -c "import ophir.ticker as t; names='get_stock_parquets get_starts get_start_dates get_sp_500_symbols get_splits StockSplit clean_daily_ohlcv extract_features StockStreamer StockHandler extract_model_data build_latest_inputs StockStreamerDataset StockHandlerDataset'.split(); missing=[n for n in names if not hasattr(t, n)]; print('PARITY', 'OK 14/14' if not missing else f'MISSING {missing}'); assert not missing"
```

A task passes only when ruff says "All checks passed!", mypy says "Success", `pytest exit=0`, and parity prints "OK 14/14".

---

### Task 1: Baseline and public-API snapshot

**Files:** none (verification only)

- [ ] **Step 1: Clean working tree and record the pre-refactor commit**

Run: `git status --porcelain`
Expected: empty. If not, stop and resolve.

Then record the starting commit for the final import-site check (Task 10 Step 4):

```bash
git rev-parse HEAD | tee /tmp/ticker_split_base_sha.txt
```

- [ ] **Step 2: Baseline mypy + pytest**

Run: `uv run mypy src/ophir && uv run pytest -q -p no:cacheprovider; echo "pytest exit=$?"`
Expected: `Success: no issues found ...` and `pytest exit=0`.

- [ ] **Step 3: Confirm the 14-name public API is exactly what `ophir.ticker` defines today**

Run:
```bash
uv run python -c "import ophir.ticker as t; defined=sorted(n for n in dir(t) if not n.startswith('_') and getattr(getattr(t,n),'__module__',None)=='ophir.ticker'); print(defined)"
```
Expected: a list equal (as a set) to the 14 public symbols above. If it differs, stop — the plan's symbol inventory is wrong and must be corrected before proceeding.

No commit (verification only).

---

### Task 2: Convert the module to a package

**Files:**
- Rename: `src/ophir/ticker.py` → `src/ophir/ticker/__init__.py`

**Interfaces:**
- Produces: `ophir.ticker` as a package whose `__init__.py` contains all original code, so all 14 symbols remain importable unchanged.

- [ ] **Step 1: Move the file into a package**

Run:
```bash
mkdir -p src/ophir/ticker
git mv src/ophir/ticker.py src/ophir/ticker/__init__.py
```

- [ ] **Step 2: Run THE GATE**

Expected: all four checks green, parity OK 14/14. (Nothing moved yet — this proves the module→package conversion alone is behavior-neutral.)

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "refactor(ticker): convert module to package (no code moved)"
```

---

### Task 3: Extract `paths.py` (parquet discovery + window math)

**Files:**
- Create: `src/ophir/ticker/paths.py`
- Modify: `src/ophir/ticker/__init__.py`

**Interfaces:**
- Produces: `ophir.ticker.paths` exporting `get_stock_parquets`, `get_starts`, `get_start_dates`.
- Consumes: nothing (leaf module — no intra-package deps).

- [ ] **Step 1: Create the submodule**

Create `src/ophir/ticker/paths.py` starting with this header, then paste the `get_stock_parquets`, `get_starts`, and `get_start_dates` definitions (decorators + full NumPy docstrings) verbatim from `__init__.py`:

```python
"""Parquet-file discovery and window-index math for the ticker pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
```

- [ ] **Step 2: Remove the moved defs from `__init__.py` and re-export**

Delete the three function definitions from `src/ophir/ticker/__init__.py`. Add this near the top of `__init__.py` (after its module docstring/imports):

```python
from ophir.ticker.paths import get_start_dates, get_starts, get_stock_parquets
```

- [ ] **Step 3: Run THE GATE**

`ruff --fix` resolves import ordering and reports any F401 (now-unused import left in `__init__.py` — remove it) or F821 (a name `paths.py` uses but did not import — add it). Iterate until "All checks passed!", then confirm mypy/pytest/parity.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(ticker): extract paths.py (parquet discovery + window math)"
```

---

### Task 4: Extract `splits.py` (symbol & split fetching)

**Files:**
- Create: `src/ophir/ticker/splits.py`
- Modify: `src/ophir/ticker/__init__.py`

**Interfaces:**
- Produces: `ophir.ticker.splits` exporting `StockSplit`, `get_sp_500_symbols`, `get_splits`.
- Consumes: nothing (leaf module).

- [ ] **Step 1: Create the submodule**

Create `src/ophir/ticker/splits.py` with this header, then paste the `StockSplit` dataclass and the `get_sp_500_symbols` and `get_splits` functions verbatim:

```python
"""S&P 500 symbol fetching and stock-split history (network-backed)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from collections.abc import Sequence
```

- [ ] **Step 2: Remove the moved defs from `__init__.py` and re-export**

Delete `StockSplit`, `get_sp_500_symbols`, `get_splits` from `__init__.py`. Add:

```python
from ophir.ticker.splits import StockSplit, get_sp_500_symbols, get_splits
```

- [ ] **Step 3: Run THE GATE**

Resolve any F401/F821 as in Task 3 (e.g. `StockSplit` uses `field`/`os`; drop unused imports left behind in `__init__.py`). Confirm green + parity.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(ticker): extract splits.py (symbol + split fetching)"
```

---

### Task 5: Extract `features.py` (cleaning + feature extraction)

**Files:**
- Create: `src/ophir/ticker/features.py`
- Modify: `src/ophir/ticker/__init__.py`

**Interfaces:**
- Produces: `ophir.ticker.features` exporting `clean_daily_ohlcv`, `extract_features`.
- Consumes: nothing (leaf module).

- [ ] **Step 1: Create the submodule**

Create `src/ophir/ticker/features.py` with this header, then paste `clean_daily_ohlcv` and `extract_features` verbatim:

```python
"""OHLCV cleaning and the 12-feature model representation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
```

- [ ] **Step 2: Remove the moved defs from `__init__.py` and re-export**

Delete `clean_daily_ohlcv`, `extract_features` from `__init__.py`. Add:

```python
from ophir.ticker.features import clean_daily_ohlcv, extract_features
```

- [ ] **Step 3: Run THE GATE**

Resolve any F401/F821. Confirm green + parity.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(ticker): extract features.py (cleaning + feature extraction)"
```

---

### Task 6: Extract `streamer.py` (`StockStreamer`)

**Files:**
- Create: `src/ophir/ticker/streamer.py`
- Modify: `src/ophir/ticker/__init__.py`

**Interfaces:**
- Produces: `ophir.ticker.streamer` exporting `StockStreamer`.
- Consumes: `ophir.ticker.paths`, `ophir.ticker.splits`, `ophir.ticker.features` (add only the names `StockStreamer` actually references; ruff F821 lists them).

- [ ] **Step 1: Create the submodule**

Create `src/ophir/ticker/streamer.py` with this header, then paste the `StockStreamer` dataclass verbatim. Add `from ophir.ticker.{paths,splits,features} import <names used>` for the sibling symbols it references:

```python
"""StockStreamer: slice one stock's history into fixed-length windows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch

if TYPE_CHECKING:
    from collections.abc import Iterator
```

- [ ] **Step 2: Remove the moved def from `__init__.py` and re-export**

Delete `StockStreamer` from `__init__.py`. Add:

```python
from ophir.ticker.streamer import StockStreamer
```

- [ ] **Step 3: Run THE GATE**

Run THE GATE; F821 names the exact sibling imports to add to `streamer.py`, F401 names unused leftovers. Iterate to green + parity.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(ticker): extract streamer.py (StockStreamer)"
```

---

### Task 7: Extract `handler.py` (`StockHandler`)

**Files:**
- Create: `src/ophir/ticker/handler.py`
- Modify: `src/ophir/ticker/__init__.py`

**Interfaces:**
- Produces: `ophir.ticker.handler` exporting `StockHandler`.
- Consumes: `ophir.ticker.streamer`, `ophir.ticker.paths`, `ophir.ticker.splits`, `ophir.ticker.features` (add the names `StockHandler` references; ruff F821 lists them).

- [ ] **Step 1: Create the submodule**

Create `src/ophir/ticker/handler.py` with this header, then paste the `StockHandler` dataclass verbatim, adding `from ophir.ticker.{streamer,paths,splits,features} import <names used>`:

```python
"""StockHandler: discover, load, filter, and stream a collection of stocks."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
```

- [ ] **Step 2: Remove the moved def from `__init__.py` and re-export**

Delete `StockHandler` from `__init__.py`. Add:

```python
from ophir.ticker.handler import StockHandler
```

- [ ] **Step 3: Run THE GATE**

Iterate F821/F401 to green + parity.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(ticker): extract handler.py (StockHandler)"
```

---

### Task 8: Extract `inputs.py` (model-data builders)

**Files:**
- Create: `src/ophir/ticker/inputs.py`
- Modify: `src/ophir/ticker/__init__.py`

**Interfaces:**
- Produces: `ophir.ticker.inputs` exporting `extract_model_data`, `build_latest_inputs`.
- Consumes: `ophir.ticker.handler`, `ophir.ticker.streamer`, `ophir.ticker.features` (add the names used; ruff F821 lists them).

- [ ] **Step 1: Create the submodule**

Create `src/ophir/ticker/inputs.py` with this header, then paste `extract_model_data` and `build_latest_inputs` verbatim, adding the sibling imports they reference:

```python
"""Builders that package feature windows into model-input tensors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch

if TYPE_CHECKING:
    from collections.abc import Sequence
```

- [ ] **Step 2: Remove the moved defs from `__init__.py` and re-export**

Delete `extract_model_data`, `build_latest_inputs` from `__init__.py`. Add:

```python
from ophir.ticker.inputs import build_latest_inputs, extract_model_data
```

- [ ] **Step 3: Run THE GATE**

Iterate F821/F401 to green + parity.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(ticker): extract inputs.py (model-data builders)"
```

---

### Task 9: Extract `datasets.py` (torch datasets)

**Files:**
- Create: `src/ophir/ticker/datasets.py`
- Modify: `src/ophir/ticker/__init__.py`

**Interfaces:**
- Produces: `ophir.ticker.datasets` exporting `StockStreamerDataset`, `StockHandlerDataset`.
- Consumes: `ophir.ticker.handler`, `ophir.ticker.inputs`, `ophir.ticker.streamer` (add the names used; ruff F821 lists them).

- [ ] **Step 1: Create the submodule**

Create `src/ophir/ticker/datasets.py` with this header, then paste `StockStreamerDataset` and `StockHandlerDataset` verbatim, adding the sibling imports they reference:

```python
"""Torch ``Dataset`` / ``IterableDataset`` wrappers over the streaming layer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

if TYPE_CHECKING:
    from collections.abc import Iterator
```

- [ ] **Step 2: Remove the moved defs from `__init__.py` and re-export**

Delete `StockStreamerDataset`, `StockHandlerDataset` from `__init__.py`. Add:

```python
from ophir.ticker.datasets import StockHandlerDataset, StockStreamerDataset
```

- [ ] **Step 3: Run THE GATE**

Iterate F821/F401 to green + parity. After this task, `__init__.py` should contain only the module docstring and the seven `from ophir.ticker.<module> import ...` lines.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(ticker): extract datasets.py (torch datasets)"
```

---

### Task 10: Finalize `__init__.py`, changelog, and final verification

**Files:**
- Modify: `src/ophir/ticker/__init__.py`, `CHANGELOG.md`

- [ ] **Step 1: Add the package docstring and `__all__`**

Ensure `src/ophir/ticker/__init__.py` is exactly the original module docstring, the seven re-export lines (in topological order), and an `__all__`. It should read like:

```python
"""Stock data ingestion, split adjustment, feature extraction, and datasets.

This package turns per-stock parquet files into the fixed-length feature
windows the model consumes: discovery + window math (:mod:`ophir.ticker.paths`),
symbol/split data (:mod:`ophir.ticker.splits`), cleaning + features
(:mod:`ophir.ticker.features`), the streaming primitive
(:mod:`ophir.ticker.streamer`) and handler (:mod:`ophir.ticker.handler`),
model-input builders (:mod:`ophir.ticker.inputs`), and torch datasets
(:mod:`ophir.ticker.datasets`).
"""

from __future__ import annotations

from ophir.ticker.paths import get_start_dates, get_starts, get_stock_parquets
from ophir.ticker.splits import StockSplit, get_sp_500_symbols, get_splits
from ophir.ticker.features import clean_daily_ohlcv, extract_features
from ophir.ticker.streamer import StockStreamer
from ophir.ticker.handler import StockHandler
from ophir.ticker.inputs import build_latest_inputs, extract_model_data
from ophir.ticker.datasets import StockHandlerDataset, StockStreamerDataset

__all__ = [
    "StockHandler",
    "StockHandlerDataset",
    "StockSplit",
    "StockStreamer",
    "StockStreamerDataset",
    "build_latest_inputs",
    "clean_daily_ohlcv",
    "extract_features",
    "extract_model_data",
    "get_sp_500_symbols",
    "get_splits",
    "get_start_dates",
    "get_starts",
    "get_stock_parquets",
]
```

- [ ] **Step 2: Run THE GATE**

Confirm green + parity. (`__all__` makes the re-export lines "used" so ruff will not flag F401 on them; if it does, the import list and `__all__` disagree — reconcile.)

- [ ] **Step 3: Verify the old file is gone and the package is complete**

Run:
```bash
test ! -f src/ophir/ticker.py && echo "old module removed: OK"
ls src/ophir/ticker/
```
Expected: `ticker.py` absent; directory lists `__init__.py datasets.py features.py handler.py inputs.py paths.py splits.py streamer.py`.

- [ ] **Step 4: Confirm no external import sites changed**

Run: `git diff --name-only "$(cat /tmp/ticker_split_base_sha.txt)"..HEAD -- ':!src/ophir/ticker/' ':!docs/' ':!CHANGELOG.md'`
Expected: empty — no file outside `src/ophir/ticker/` (and the docs/changelog) was touched, i.e. no test or other `src` file had its imports edited. If any appear, investigate — the shim should have made call-site edits unnecessary.

- [ ] **Step 5: Add the changelog entry**

In `CHANGELOG.md`, under `## [Unreleased]`, add a `### Changed` section (or append to it if present) as the first item:

```markdown
### Changed

- Reorganized `ophir.ticker` from a single 999-line module into a focused
  package (`paths`, `splits`, `features`, `streamer`, `handler`, `inputs`,
  `datasets`) with a re-export `__init__`. Public API and behavior are
  unchanged; `from ophir.ticker import ...` continues to work as before.
```

- [ ] **Step 6: Final full gate + commit**

Run THE GATE one last time (green + parity), then:

```bash
git add -A
git commit -m "refactor(ticker): finalize package __init__ and changelog"
```

---

## Self-Review

**1. Spec coverage:**
- Package with 8 files (`paths`/`splits`/`features`/`streamer`/`handler`/`inputs`/`datasets`/`__init__`) → Tasks 2–10. ✓
- Re-export shim, zero call-site changes → Task 2 (immediate) + per-task re-exports + Task 10 Step 4 verification. ✓
- `inputs.py` name (avoiding `model_data` clash) → Task 8. ✓
- `streamer`/`handler` separate → Tasks 6 & 7. ✓
- Verified topological extraction order → Tasks 3→9 follow paths/splits/features → streamer → handler → inputs → datasets. ✓
- Behavior-preserving, regression gate = existing suite → THE GATE on every task. ✓
- Public-API parity check → THE GATE's 14-name `hasattr` assertion. ✓
- CHANGELOG note → Task 10 Step 5. ✓
- `register.py` untouched → not referenced by any task. ✓

**2. Placeholder scan:** No TBD/TODO. The only non-verbatim instruction ("add the sibling imports the moved symbol references") is backed by a deterministic mechanism — ruff F821 lists every missing name and F401 every unused one — not a vague "handle imports." Module headers and re-export/`__all__` lines are given exactly.

**3. Type consistency:** Submodule names (`paths`, `splits`, `features`, `streamer`, `handler`, `inputs`, `datasets`) and the 14 public symbol names are identical across THE GATE, every task's interfaces, and Task 10's `__init__`/`__all__`. Re-export import lines in Tasks 3–9 match the consolidated list in Task 10 Step 1.
