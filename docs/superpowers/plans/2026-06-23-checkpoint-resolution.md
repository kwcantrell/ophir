# Canonical-Checkpoint Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the `IndexError` that makes `load_base_model_ckpt(time_version=False)` (and thus `load_forecasts`) fail, by resolving "the base model" to an explicit canonical checkpoint file, and route new training candidates into a `candidates/` subdir.

**Architecture:** Extract a pure, offline-testable path resolver (`_resolve_base_ckpt_path`) that returns a checkpoint path without loading a model: `time_version=False` → the explicit canonical `{MODEL_DIR}/{name}-best.ckpt`; `time_version=True` → the latest rolling `-time-check-v<N>` via a hardened `_latest_base_ckpt`. `load_base_model_ckpt` delegates to it. The best-checkpoint callback saves to `MODEL_DIR/candidates/`.

**Tech Stack:** Python 3.10+ (`from __future__ import annotations`), pytest + `monkeypatch`/`tmp_path`, PyTorch-Lightning `ModelCheckpoint` (constructed, never saved, in tests).

## Global Constraints

- mypy is `strict = True`, targets Python 3.10 — keep `src/ophir` fully typed.
- ruff targets 3.12; run `uv run ruff check .` and `uv run ruff format --check .`.
- pytest runs `filterwarnings = error`; tests must stay **offline + CPU-only** and never load a real model, touch CUDA, or write to the package `.ophir/` layout. Use `tmp_path` and `monkeypatch.setattr(register, "MODEL_DIR", str(tmp_path))`.
- NumPy-style docstrings throughout `src/ophir`, matching existing density.
- Update the `[Unreleased]` section of `CHANGELOG.md`.
- `register.py` already has `from __future__ import annotations` and `import os` at the top — `str | None` annotations are fine; do not re-import `os`.
- Run tests with `uv run pytest`; single file via `uv run pytest tests/test_register.py`.

---

### Task 1: Harden `_latest_base_ckpt`

**Files:**
- Modify: `src/ophir/register.py` (the `_latest_base_ckpt` function, around line 241)
- Test: `tests/test_register.py` (append)

**Interfaces:**
- Produces: `_latest_base_ckpt(filename: str) -> str` — returns the highest `-v<N>` match, or the sorted-last match when none are versioned; raises `FileNotFoundError` when nothing matches (previously raised `IndexError` in both degenerate cases).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_register.py` (the file already imports `pytest`, `register`, and `_best_checkpoint_callback`):

```python
def test_latest_base_ckpt_picks_highest_version(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(register, "MODEL_DIR", str(tmp_path))
    for v in (1, 2, 10):
        (tmp_path / f"m-time-check-v{v}.ckpt").write_text("")
    assert register._latest_base_ckpt("m-time-check") == "m-time-check-v10.ckpt"


def test_latest_base_ckpt_no_match_raises(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(register, "MODEL_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        register._latest_base_ckpt("nothing")


def test_latest_base_ckpt_unversioned_matches_no_indexerror(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: many matches, none with a `-v<N>` suffix -> sorted-last, no IndexError.
    monkeypatch.setattr(register, "MODEL_DIR", str(tmp_path))
    for name in ("p-a.ckpt", "p-b.ckpt", "p-c.ckpt"):
        (tmp_path / name).write_text("")
    assert register._latest_base_ckpt("p-") == "p-c.ckpt"
```

Add `from typing import Any` to the test file's imports if it is not already present (the file already imports `from typing import Any`).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_register.py -k latest_base_ckpt -v`
Expected: FAIL — `test_latest_base_ckpt_no_match_raises` raises `IndexError` (not `FileNotFoundError`), and `test_latest_base_ckpt_unversioned_matches_no_indexerror` raises `IndexError`.

- [ ] **Step 3: Harden the implementation**

Replace the body of `_latest_base_ckpt` in `src/ophir/register.py` with:

```python
def _latest_base_ckpt(filename: str) -> str:
    """Return the filename of the most recent base checkpoint.

    Parameters
    ----------
    filename : str
        Substring that base checkpoint files must contain.

    Returns
    -------
    str
        The highest ``-v<N>`` version among matches, or the sorted-last match
        when none carry a ``-v<N>`` suffix.

    Raises
    ------
    FileNotFoundError
        When no file in :data:`MODEL_DIR` contains ``filename``.
    """
    base_paths = sorted(path for path in os.listdir(MODEL_DIR) if filename in path)
    if not base_paths:
        raise FileNotFoundError(f"no checkpoint matching {filename!r} in {MODEL_DIR}")
    versioned = sorted(
        (int(version.removeprefix(f"{filename}-v").removesuffix(".ckpt")), version)
        for version in base_paths
        if f"{filename}-v" in version
    )
    return versioned[-1][1] if versioned else base_paths[-1]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_register.py -v`
Expected: PASS (the three new tests plus every pre-existing `test_register` test).

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy src/ophir && uv run ruff check src/ophir/register.py tests/test_register.py && uv run ruff format --check src/ophir/register.py tests/test_register.py`
Expected: no errors. (If format check fails, run `uv run ruff format <file>` and re-run.)

- [ ] **Step 6: Commit**

```bash
git add src/ophir/register.py tests/test_register.py
git commit -m "fix: _latest_base_ckpt degrades to FileNotFoundError instead of IndexError"
```

---

### Task 2: Canonical resolver + rewire `load_base_model_ckpt`

**Files:**
- Modify: `src/ophir/register.py` (add `BASE_BEST_CKPT` constant; add `_resolve_base_ckpt_path`; rewire `load_base_model_ckpt` body)
- Test: `tests/test_register.py` (append)

**Interfaces:**
- Consumes: `_latest_base_ckpt` (Task 1); module constants `BASE_NAME`, `TIME_MODIFIER`, `MODEL_DIR`.
- Produces:
  - `BASE_BEST_CKPT: str` — `os.path.join(MODEL_DIR, f"{BASE_NAME}-best.ckpt")`.
  - `_resolve_base_ckpt_path(file_name: str | None = None, time_version: bool = True) -> str` — resolves a checkpoint path (no model load); `time_version=False` → `{MODEL_DIR}/{name}-best.ckpt` (raises `FileNotFoundError` if absent); `time_version=True` → latest time-check via `_latest_base_ckpt`.
  - `load_base_model_ckpt(...)` — unchanged signature/overloads, now delegating to `_resolve_base_ckpt_path`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_register.py`:

```python
def test_base_best_ckpt_constant_name() -> None:
    assert register.BASE_BEST_CKPT.endswith("ophir-ohlc-base-best.ckpt")


def test_resolve_canonical_present(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(register, "MODEL_DIR", str(tmp_path))
    (tmp_path / "ophir-ohlc-base-best.ckpt").write_text("")
    assert register._resolve_base_ckpt_path(time_version=False) == str(
        tmp_path / "ophir-ohlc-base-best.ckpt"
    )


def test_resolve_canonical_absent_raises(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(register, "MODEL_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        register._resolve_base_ckpt_path(time_version=False)


def test_resolve_custom_file_name(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(register, "MODEL_DIR", str(tmp_path))
    (tmp_path / "foo-best.ckpt").write_text("")
    assert register._resolve_base_ckpt_path("foo", time_version=False) == str(
        tmp_path / "foo-best.ckpt"
    )


def test_resolve_ignores_val_loss_zoo(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression for the reported bug: the canonical file resolves even with the
    # ~125-file val_loss zoo present (previously an IndexError -> load_forecasts {}).
    monkeypatch.setattr(register, "MODEL_DIR", str(tmp_path))
    (tmp_path / "ophir-ohlc-base-best.ckpt").write_text("")
    for vl in ("0.011", "0.012", "0.013"):
        (tmp_path / f"ophir-ohlc-basebest-epoch=00-val_loss={vl}.ckpt").write_text("")
    assert register._resolve_base_ckpt_path(time_version=False) == str(
        tmp_path / "ophir-ohlc-base-best.ckpt"
    )


def test_resolve_time_version_picks_latest(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(register, "MODEL_DIR", str(tmp_path))
    (tmp_path / "ophir-ohlc-base-time-check-v1.ckpt").write_text("")
    (tmp_path / "ophir-ohlc-base-time-check-v2.ckpt").write_text("")
    assert register._resolve_base_ckpt_path(time_version=True) == str(
        tmp_path / "ophir-ohlc-base-time-check-v2.ckpt"
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_register.py -k "resolve or base_best" -v`
Expected: FAIL — `AttributeError: module 'ophir.register' has no attribute 'BASE_BEST_CKPT'` / `'_resolve_base_ckpt_path'`.

- [ ] **Step 3: Add the constant and resolver**

In `src/ophir/register.py`, add the constant immediately after the existing `BASE_MODEL_CKPT = ...` line:

```python
BASE_BEST_CKPT = os.path.join(MODEL_DIR, f"{BASE_NAME}-best.ckpt")
```

Add the resolver function just above `def load_base_model_ckpt` (after the overloads, before the implementation):

```python
def _resolve_base_ckpt_path(
    file_name: str | None = None, time_version: bool = True
) -> str:
    """Resolve a base checkpoint path without loading the model.

    ``time_version=True`` selects the latest rolling ``-time-check-v<N>``
    checkpoint; ``time_version=False`` selects the explicit canonical
    best-checkpoint ``{MODEL_DIR}/{name}-best.ckpt``.

    Parameters
    ----------
    file_name : str or None, optional
        Base name. ``None`` uses :data:`BASE_NAME`.
    time_version : bool, optional
        Select the rolling time-check checkpoint (``True``) or the canonical
        best checkpoint (``False``). Defaults to ``True``.

    Returns
    -------
    str
        Absolute path to the resolved checkpoint.

    Raises
    ------
    FileNotFoundError
        When no matching checkpoint exists.
    """
    name = file_name if file_name is not None else BASE_NAME
    if time_version:
        return os.path.join(MODEL_DIR, _latest_base_ckpt(name + TIME_MODIFIER))
    path = os.path.join(MODEL_DIR, f"{name}-best.ckpt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"canonical base checkpoint not found: {path}")
    return path
```

- [ ] **Step 4: Rewire `load_base_model_ckpt`**

In `src/ophir/register.py`, replace the implementation body of `load_base_model_ckpt` (the block that currently does `if file_name is None`, the `TIME_MODIFIER`/`EPOCH_MODIFIER` concatenation, `.split("{")[0]`, `_latest_base_ckpt`, and the `os.path.join(MODEL_DIR, latest_version)`) with:

```python
    from ophir.training_models import LightningOHLCPredictor

    last_ckpt = _resolve_base_ckpt_path(file_name, time_version)
    print(f"loading {last_ckpt}")

    if not return_ckpt_path:
        return LightningOHLCPredictor.load_from_checkpoint(last_ckpt, strict=strict)
    return LightningOHLCPredictor.load_from_checkpoint(last_ckpt, strict=strict), last_ckpt
```

Leave the function signature, overloads, and docstring above this body unchanged.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_register.py -v`
Expected: PASS (all new resolver/constant tests plus the pre-existing suite). No real model is loaded — the tests exercise `_resolve_base_ckpt_path` directly.

- [ ] **Step 6: Typecheck and lint**

Run: `uv run mypy src/ophir && uv run ruff check src/ophir/register.py tests/test_register.py && uv run ruff format --check src/ophir/register.py tests/test_register.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/ophir/register.py tests/test_register.py
git commit -m "fix: resolve time_version=False to explicit canonical checkpoint"
```

---

### Task 3: Route candidates to a subdir; docs + CHANGELOG

**Files:**
- Modify: `src/ophir/register.py` (the `_best_checkpoint_callback` `dirpath`)
- Create: `docs/checkpoint-promotion.md`
- Modify: `CHANGELOG.md`
- Test: `tests/test_register.py` (append)

**Interfaces:**
- Consumes: `_best_checkpoint_callback(file_name: str, monitor_near_ic: bool) -> ModelCheckpoint` (existing).
- Produces: the best-checkpoint callback now saves into `{MODEL_DIR}/candidates/` instead of `{MODEL_DIR}`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_register.py`:

```python
def test_best_checkpoint_saves_to_candidates_subdir(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(register, "MODEL_DIR", str(tmp_path))
    cb = _best_checkpoint_callback("model", monitor_near_ic=True)
    assert cb.dirpath is not None
    assert str(cb.dirpath).endswith("candidates")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_register.py::test_best_checkpoint_saves_to_candidates_subdir -v`
Expected: FAIL — `cb.dirpath` ends with the `MODEL_DIR` root, not `candidates`.

- [ ] **Step 3: Redirect the callback dirpath**

In `src/ophir/register.py`, inside `_best_checkpoint_callback`, change the `ModelCheckpoint` `dirpath` argument:

```python
        dirpath=os.path.join(MODEL_DIR, "candidates"),
```

(Leave the `time_checkpoint_callback` in `fetch_base_trainer` at `dirpath=MODEL_DIR` — the `time_version=True` resolver globs the root for `-time-check-v<N>` files.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_register.py -v`
Expected: PASS (the new test plus the pre-existing `test_best_checkpoint_*` tests, which assert `.monitor`/`.mode`/`.filename` and are unaffected by `dirpath`).

- [ ] **Step 5: Add the promotion runbook**

Create `docs/checkpoint-promotion.md`:

```markdown
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
```

- [ ] **Step 6: Update the CHANGELOG**

In `CHANGELOG.md`, under `## [Unreleased]`, add a `### Fixed` subsection (create it above the existing `### Added` subsection) with this single bullet — it covers the resolver fix, the helper hardening, and the candidates redirect together, so no separate `### Added` bullet is needed:

```markdown
### Fixed

- `load_base_model_ckpt(time_version=False)` (and therefore `load_forecasts`)
  raised `IndexError` and degraded to `{}` whenever multiple non-versioned
  checkpoints matched its derived prefix. It now resolves an explicit canonical
  checkpoint (`register.BASE_BEST_CKPT` = `ophir-ohlc-base-best.ckpt`) via a pure,
  offline-testable `_resolve_base_ckpt_path`, and `_latest_base_ckpt` degrades to
  `FileNotFoundError` instead of `IndexError`. Best-epoch training candidates now
  save to a `candidates/` subdir; promotion to canonical is an explicit copy (see
  `docs/checkpoint-promotion.md`).
```

- [ ] **Step 7: Full suite, typecheck, lint**

Run: `uv run pytest && uv run mypy src/ophir && uv run ruff check . && uv run ruff format --check .`
Expected: all pass. (If format check fails, run `uv run ruff format .` and re-run.)

- [ ] **Step 8: Commit**

```bash
git add src/ophir/register.py tests/test_register.py docs/checkpoint-promotion.md CHANGELOG.md
git commit -m "feat: route best-checkpoint candidates to candidates/ subdir + promotion runbook"
```

---

## Notes for the implementer

- **`monkeypatch.setattr(register, "MODEL_DIR", str(tmp_path))`** is the key to keeping these tests offline: the resolver and helpers read the module global `MODEL_DIR` at call time, so the patch redirects every filesystem listing into `tmp_path`. Never point a test at the real `src/ophir/.ophir/` tree.
- **No model is ever loaded in tests.** The bug and the fix live in *path resolution* (`_resolve_base_ckpt_path` / `_latest_base_ckpt`), which is why they were extracted — they return strings from a directory listing. `load_base_model_ckpt` itself (which calls `LightningOHLCPredictor.load_from_checkpoint`) is not unit-tested here, matching the existing CUDA/runtime boundary.
- **`cb.dirpath` assertion uses `endswith("candidates")`** because Lightning may normalize the stored path; constructing a `ModelCheckpoint` does not create the directory or write to disk.
- **Out of scope:** Piece B (the GPU `ophir train --val-identity` run producing the IC-best checkpoint), deleting the existing stale checkpoints (the runbook documents it for the user to run), and the unused `BASE_MODEL_CKPT` constant.
- **Why this matters:** once merged, `load_forecasts` returns real forecasts from the existing canonical checkpoint (no GPU); Piece B later swaps in a higher-quality IC-best artifact through the same path via an explicit copy.
