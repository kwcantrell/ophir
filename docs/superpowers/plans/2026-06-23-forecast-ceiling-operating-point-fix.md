# Forecast-ceiling operating-point fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Productionize a near-band validation metric and switch best-checkpoint selection to it, then settle the forecast operating point (read-near vs short-horizon retrain) by a multi-seed experiment benchmarked against a clean near-band reversal ceiling.

**Architecture:** Components A (metric + checkpoint) and B (experiment + ceiling helper). A adds `val_rank_ic_near` (pooled offsets 1–K cross-sectional IC, ungated) to the Lightning validation hook and flips the best-checkpoint `ModelCheckpoint` from `val_loss` (min) to `val_rank_ic_near` (max) whenever the validation loader carries identity. B adds a tested `near_band_reversal_ceiling` helper, then runs short-horizon training at 3 seeds and records the operating-point decision. Component C (wiring `trading/forecast.py`) is deferred to its own spec.

**Tech Stack:** Python 3.10+, PyTorch, PyTorch-Lightning, pytest, uv.

## Global Constraints

- mypy is `strict = True`, targets Python 3.10; run `uv run mypy src/ophir`.
- ruff targets Python 3.12; run `uv run ruff check . && uv run ruff format --check .`.
- pytest runs `filterwarnings = error` — any project-owned warning fails the suite.
- Tests must never touch the network, CUDA, or the package `.ophir/` layout; use `tmp_path` and the seeded fixtures in `tests/conftest.py`.
- NumPy-style docstrings throughout `src/ophir`, matching existing density.
- Imports: `known-first-party = ["ophir"]` (ruff/isort ordering).
- Update the `[Unreleased]` section of `CHANGELOG.md` for notable changes.
- Reuse production rank-IC math (`dedupe_by_ticker_date` + `rank_ic`) so offline and live metrics agree.
- Run commands with `uv run` (e.g. `uv run pytest`).

---

### Task 1: `val_rank_ic_near` metric helper

**Files:**
- Modify: `src/ophir/training_models.py` (add helper next to `val_rank_ic`, line 65)
- Test: `tests/test_training_models.py`

**Interfaces:**
- Consumes: `evaluate.dedupe_by_ticker_date`, `evaluate.rank_ic` (already imported lazily inside `val_rank_ic`).
- Produces: `val_rank_ic_near(pred, target, ids, dates, offsets, k=5) -> float` — pooled cross-sectional rank-IC over rows whose trading-day offset is in `1..k`; returns `nan` for empty input or no rows in band.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_training_models.py` (extend the existing import from `ophir.training_models` to include `val_rank_ic_near`):

```python
def test_val_rank_ic_near_filters_to_band() -> None:
    # Two days, three tickers each. Offset-1 rows rank the same as targets
    # (IC=1); out-of-band offset-9 rows are deliberately anti-ranked. With
    # k=5 only the offset-1 rows count, so the pooled near-IC is ~1.0.
    pred = torch.tensor([3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    target = torch.tensor([0.3, 0.2, 0.1, 0.1, 0.2, 0.3, 0.3, 0.2, 0.1])
    ids = torch.tensor([1, 2, 3, 1, 2, 3, 1, 2, 3])
    dates = torch.tensor([10, 10, 10, 11, 11, 11, 12, 12, 12])
    offsets = torch.tensor([1, 1, 1, 1, 1, 1, 9, 9, 9])
    assert val_rank_ic_near(pred, target, ids, dates, offsets, k=5) > 0.99


def test_val_rank_ic_near_k_boundary_is_inclusive() -> None:
    # Single day, three tickers, all at offset 5. With k=5 the row is in band
    # (perfect ranking -> ~1.0); with k=4 it is excluded -> nan.
    pred = torch.tensor([3.0, 2.0, 1.0])
    target = torch.tensor([0.3, 0.2, 0.1])
    ids = torch.tensor([1, 2, 3])
    dates = torch.tensor([10, 10, 10])
    offsets = torch.tensor([5, 5, 5])
    assert val_rank_ic_near(pred, target, ids, dates, offsets, k=5) > 0.99
    assert val_rank_ic_near(pred, target, ids, dates, offsets, k=4) != \
        val_rank_ic_near(pred, target, ids, dates, offsets, k=4)  # NaN


def test_val_rank_ic_near_empty_is_nan() -> None:
    empty = torch.tensor([])
    result = val_rank_ic_near(
        empty, empty, empty.long(), empty.long(), empty.long(), k=5
    )
    assert result != result  # NaN
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training_models.py -k val_rank_ic_near -v`
Expected: FAIL with `ImportError` / `cannot import name 'val_rank_ic_near'`.

- [ ] **Step 3: Write the helper**

In `src/ophir/training_models.py`, immediately after `val_rank_ic` (after line 83), add:

```python
def val_rank_ic_near(
    pred: torch.Tensor,
    target: torch.Tensor,
    ids: torch.Tensor,
    dates: torch.Tensor,
    offsets: torch.Tensor,
    k: int = 5,
) -> float:
    """Pooled cross-sectional rank-IC over near forecast offsets ``1..k``.

    Restricts the validation rows to trading-day offsets ``1 <= offset <= k``
    (the near band where cross-sectional skill concentrates), then dedupes to
    one prediction per ``(ticker, date)`` and averages the daily Spearman
    rank correlation, reusing the eval module's helpers so the live metric and
    the offline report agree. Pooling the near offsets keeps each day's
    cross-section dense (overlapping windows are sparse, so the dedup drops
    almost nothing). Returns ``nan`` when no rows fall in the band.

    Parameters
    ----------
    pred, target, ids, dates : torch.Tensor
        Equal-length 1-D tensors, as accumulated for :func:`val_rank_ic`.
    offsets : torch.Tensor
        Same-length integer tensor of 1-based trading-day forecast offsets.
    k : int, optional
        Inclusive upper bound of the near band. Defaults to ``5``.

    Returns
    -------
    float
        Mean daily rank-IC over the near band, or ``nan`` if empty.
    """
    from .evaluate import dedupe_by_ticker_date, rank_ic

    if pred.numel() == 0:
        return float("nan")
    sel = (offsets >= 1) & (offsets <= k)
    if not bool(sel.any()):
        return float("nan")
    dp, dt, dd = dedupe_by_ticker_date(pred[sel], target[sel], ids[sel], dates[sel])
    return rank_ic(dp, dt, dd)["ic_mean"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_training_models.py -k val_rank_ic_near -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ophir/training_models.py tests/test_training_models.py
git commit -m "feat: add val_rank_ic_near pooled near-band metric helper"
```

---

### Task 2: Log `val_rank_ic_near` ungated + `near_offset_k` param

**Files:**
- Modify: `src/ophir/training_models.py` — `LightningOHLCPredictor.__init__` (kwargs around line 113) and `on_validation_epoch_end` (line 478)
- Test: `tests/test_training_models.py`

**Interfaces:**
- Consumes: `val_rank_ic_near` (Task 1); the already-populated `self._val_ic_buffers` keys `pred`/`target`/`ids`/`dates`/`offsets` (filled in `validation_step`, lines 466-474).
- Produces: a `val_rank_ic_near` value logged every validation pass that collects identity, independent of `--log-offset-ic`; new attribute `self.near_offset_k: int`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_training_models.py`:

```python
def test_on_validation_epoch_end_logs_near_ic_ungated() -> None:
    # log_offset_ic stays False (default); near-IC must still be logged.
    model = _build_predictor()
    logged: dict[str, float] = {}
    model.log = lambda name, value, **kw: logged.__setitem__(name, float(value))  # type: ignore[method-assign]
    # One day, three tickers, perfect near-band (offset 1) ranking.
    model._val_ic_buffers["pred"] = [torch.tensor([3.0, 2.0, 1.0])]
    model._val_ic_buffers["target"] = [torch.tensor([0.3, 0.2, 0.1])]
    model._val_ic_buffers["ids"] = [torch.tensor([1, 2, 3])]
    model._val_ic_buffers["dates"] = [torch.tensor([10, 10, 10])]
    model._val_ic_buffers["offsets"] = [torch.tensor([1, 1, 1])]

    model.on_validation_epoch_end()

    assert "val_rank_ic_near" in logged
    assert logged["val_rank_ic_near"] > 0.99


def test_near_offset_k_defaults_to_five() -> None:
    assert _build_predictor().near_offset_k == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training_models.py -k "near_ic_ungated or near_offset_k_defaults" -v`
Expected: FAIL — `AttributeError: ... near_offset_k` and `val_rank_ic_near` missing from `logged`.

- [ ] **Step 3a: Add the constructor parameter**

In `src/ophir/training_models.py`, add `near_offset_k: int = 5,` to `__init__` (in the kwarg block ending at line 115, e.g. right after `log_offset_ic: bool = False,`). Add a docstring entry matching the existing style:

```
        near_offset_k : int, optional
            Inclusive upper bound (in trading-day offsets) of the near band
            scored by ``val_rank_ic_near``. Defaults to ``5`` (the validated
            band where cross-sectional skill concentrates).
```

In the `__init__` body, where the other simple hyper-parameters are stored (alongside `self.log_offset_ic = log_offset_ic`), add:

```python
        self.near_offset_k = near_offset_k
```

If `__init__` calls `self.save_hyperparameters(...)` with an explicit list, add `"near_offset_k"` to it; if it calls `self.save_hyperparameters()` with no args, no change is needed.

- [ ] **Step 3b: Log the metric ungated**

In `on_validation_epoch_end`, inside the existing `if preds:` block (after the `val_rank_ic` log at line 500, before the `if self.log_offset_ic and preds:` block), add:

```python
            near_ic = val_rank_ic_near(
                torch.cat(preds),
                torch.cat(self._val_ic_buffers["target"]),
                torch.cat(self._val_ic_buffers["ids"]),
                torch.cat(self._val_ic_buffers["dates"]),
                torch.cat(self._val_ic_buffers["offsets"]),
                self.near_offset_k,
            )
            self.log("val_rank_ic_near", near_ic, prog_bar=False, on_epoch=True, logger=True)
```

(`val_rank_ic_near` is module-level in the same file — no import needed.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_training_models.py -k "near_ic_ungated or near_offset_k_defaults" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the full module test + typecheck**

Run: `uv run pytest tests/test_training_models.py -q && uv run mypy src/ophir`
Expected: all pass; mypy clean.

- [ ] **Step 6: Commit**

```bash
git add src/ophir/training_models.py tests/test_training_models.py
git commit -m "feat: log val_rank_ic_near each validation pass, add near_offset_k"
```

---

### Task 3: Switch best-checkpoint monitor to near-IC when identity is present

**Files:**
- Modify: `src/ophir/register.py` — extract a CPU-testable callback factory and add a `monitor_near_ic` parameter to `fetch_base_trainer` (lines 53-117)
- Modify: `src/ophir/train.py` — pass `monitor_near_ic=val_identity` into `fetch_base_trainer` (call at line 440)
- Test: `tests/test_register.py` (create if absent)

**Interfaces:**
- Consumes: nothing new.
- Produces: `_best_checkpoint_callback(file_name: str, monitor_near_ic: bool) -> ModelCheckpoint` in `register.py`; `fetch_base_trainer(..., monitor_near_ic: bool = False)`.

Rationale: `fetch_base_trainer` builds a CUDA `Trainer`, which cannot be constructed in CPU-only tests. Extracting the callback factory lets us test the monitor/mode wiring without a Trainer.

- [ ] **Step 1: Write the failing test**

Create `tests/test_register.py`:

```python
"""Tests for register.py trainer/checkpoint wiring."""

from ophir.register import _best_checkpoint_callback


def test_best_checkpoint_monitors_near_ic_when_flagged() -> None:
    cb = _best_checkpoint_callback("model", monitor_near_ic=True)
    assert cb.monitor == "val_rank_ic_near"
    assert cb.mode == "max"


def test_best_checkpoint_defaults_to_val_loss() -> None:
    cb = _best_checkpoint_callback("model", monitor_near_ic=False)
    assert cb.monitor == "val_loss"
    assert cb.mode == "min"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_register.py -v`
Expected: FAIL with `ImportError: cannot import name '_best_checkpoint_callback'`.

- [ ] **Step 3: Extract the factory and add the parameter**

In `src/ophir/register.py`, the `ModelCheckpoint` import currently lives inside `fetch_base_trainer`. Add a module-level import near the top of the file (with the other lightning imports if present, otherwise a local import inside the new function is acceptable to keep CUDA out of import time — prefer a local import inside `_best_checkpoint_callback`).

Add this function above `fetch_base_trainer` (before line 53):

```python
def _best_checkpoint_callback(file_name: str, monitor_near_ic: bool) -> "ModelCheckpoint":
    """Build the best-checkpoint callback, monitoring near-IC or ``val_loss``.

    ``val_loss`` is anti-aligned with cross-sectional IC (IC peaks mid-run then
    droops as the cosine LR anneals), so when the validation loader carries
    identity we select on ``val_rank_ic_near`` (maximising) instead. Without
    identity that metric is never logged, so we fall back to ``val_loss``
    (minimising).

    Parameters
    ----------
    file_name : str
        Base name for the checkpoint files.
    monitor_near_ic : bool
        When ``True`` monitor ``val_rank_ic_near`` (``mode="max"``); otherwise
        monitor ``val_loss`` (``mode="min"``).

    Returns
    -------
    ModelCheckpoint
        The configured best-checkpoint callback.
    """
    from lightning.pytorch.callbacks import ModelCheckpoint

    monitor, mode = ("val_rank_ic_near", "max") if monitor_near_ic else ("val_loss", "min")
    return ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        dirpath=MODEL_DIR,
        filename=file_name + EPOCH_MODIFIER,
        save_top_k=1,
        save_on_train_epoch_end=False,
    )
```

Add `monitor_near_ic: bool = False,` to the `fetch_base_trainer` signature (after `extra_callbacks` on line 58) and document it in the NumPy docstring:

```
    monitor_near_ic : bool, optional
        When ``True`` the best-checkpoint callback selects on
        ``val_rank_ic_near`` (``mode="max"``) instead of ``val_loss``. Only
        valid when the validation loader logs that metric (identity present).
        Defaults to ``False``.
```

Replace the inline `epoch_checkpoint_callback = ModelCheckpoint(...)` block (lines 111-117) with:

```python
    epoch_checkpoint_callback = _best_checkpoint_callback(file_name, monitor_near_ic)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_register.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Wire `train.py` to pass the flag**

In `src/ophir/train.py`, the `register.fetch_base_trainer(...)` call (line 440) is inside `run_training`, which has the `val_identity` parameter (line 355/418). Add the argument:

```python
    trainer = register.fetch_base_trainer(
        max_steps=max_steps,
        val_check_interval=val_every_steps,
        limit_val_batches=val_batches,
        extra_callbacks=callbacks,
        monitor_near_ic=val_identity,
    )
```

- [ ] **Step 6: Typecheck and commit**

Run: `uv run mypy src/ophir && uv run ruff check src/ophir/register.py src/ophir/train.py`
Expected: clean.

```bash
git add src/ophir/register.py src/ophir/train.py tests/test_register.py
git commit -m "feat: select best checkpoint on val_rank_ic_near when identity present"
```

---

### Task 4: `near_band_reversal_ceiling` benchmark helper

**Files:**
- Modify: `src/ophir/ceiling.py` (add near `signal_decay_curve`, line 560)
- Test: `tests/test_ceiling.py` (extend; create if absent)

**Interfaces:**
- Consumes: `signal_decay_curve(target, ids, dates, leads, kind="reversal")` (line 560) — already returns `{lead: ic_mean}` of the clean per-lead 1-trading-day reversal IC.
- Produces: `near_band_reversal_ceiling(target, ids, dates, k=5) -> float` — mean clean reversal IC over leads `1..k`.

Rationale: the confirmation eval printed a pooled-1–5 "reversal ceiling" of ~0.119 via `lagged_target_signal(lag=1)` on *mixed-offset* pooled rows; that is not a clean 1-trading-day reversal and is unreliable. The clean comparand is the mean of `signal_decay_curve`'s per-lead reversal over leads 1–5.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ceiling.py` (match the existing import style for `ophir.ceiling`):

```python
def test_near_band_reversal_ceiling_matches_mean_of_decay_curve() -> None:
    import torch

    from ophir.ceiling import near_band_reversal_ceiling, signal_decay_curve

    # Build a small reversal-flavoured panel: returns alternate sign per step
    # per ticker, so a negated lag-1 signal correlates with the next return.
    torch.manual_seed(0)
    n_tickers, n_days = 6, 30
    ids = torch.arange(n_tickers).repeat(n_days)
    dates = torch.arange(n_days).repeat_interleave(n_tickers)
    base = torch.randn(n_tickers)
    rows = []
    for d in range(n_days):
        rows.append(((-1.0) ** d) * base + 0.01 * torch.randn(n_tickers))
    target = torch.cat(rows)

    curve = signal_decay_curve(target, ids, dates, leads=[1, 2, 3, 4, 5], kind="reversal")
    expected = sum(curve.values()) / len(curve)
    got = near_band_reversal_ceiling(target, ids, dates, k=5)
    assert abs(got - expected) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ceiling.py -k near_band_reversal_ceiling -v`
Expected: FAIL with `ImportError: cannot import name 'near_band_reversal_ceiling'`.

- [ ] **Step 3: Write the helper**

In `src/ophir/ceiling.py`, after `signal_decay_curve` (after line 599), add:

```python
def near_band_reversal_ceiling(
    target: torch.Tensor,
    ids: torch.Tensor,
    dates: torch.Tensor,
    k: int = 5,
) -> float:
    """Clean near-band naive-reversal ceiling: mean per-lead reversal IC over ``1..k``.

    Averages :func:`signal_decay_curve`'s clean per-lead reversal IC across
    forecast leads ``1..k`` (in trading-day observations). Each lead uses a
    proper single-lag reversal signal, so this is the rigorous comparand for a
    near-band model operating point — unlike a pooled ``lag=1`` over mixed
    offsets, which does not isolate a 1-trading-day reversal.

    Parameters
    ----------
    target, ids, dates : torch.Tensor
        Equal-length 1-D tensors of return, ticker id, and integer date
        ordinal, one row per (ticker, date).
    k : int, optional
        Inclusive upper bound of the near band, in trading-day leads. Defaults
        to ``5``.

    Returns
    -------
    float
        Mean reversal IC over leads ``1..k`` (``nan`` if every lead is ``nan``).
    """
    curve = signal_decay_curve(
        target, ids, dates, leads=list(range(1, k + 1)), kind="reversal"
    )
    values = [v for v in curve.values() if v == v]  # drop nan leads
    if not values:
        return float("nan")
    return sum(values) / len(values)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ceiling.py -k near_band_reversal_ceiling -v`
Expected: PASS.

- [ ] **Step 5: Typecheck, lint, and commit**

Run: `uv run mypy src/ophir && uv run ruff check src/ophir/ceiling.py`
Expected: clean.

Update `CHANGELOG.md` `[Unreleased]`: add bullets for `val_rank_ic_near` (logged each validation pass; drives best-checkpoint selection when identity is present) and `near_band_reversal_ceiling`.

```bash
git add src/ophir/ceiling.py tests/test_ceiling.py CHANGELOG.md
git commit -m "feat: add near_band_reversal_ceiling clean benchmark + changelog"
```

---

### Task 5: Run the operating-point experiment and record the decision (manual GPU)

**This task is not TDD — it is a manual GPU runbook plus analysis.** It needs the RTX 3090 via `uv run` (free any local `llama-server` holding VRAM first). No pytest changes.

**Files:**
- Modify: `docs/forecast-ceiling-results.md` (append an operating-point section)
- Modify: the `forecast-ceiling-investigation` memory (record the decision)
- Reuse (gitignored scratch): `.superpowers/sdd/gpu/harvest.py`, `.superpowers/sdd/gpu/eval_pooled_near.py`

- [ ] **Step 1: Train the short-horizon model, 3 seeds**

`response_size ≈ 10` calendar days covers trading-day leads 1–5 (the response block is calendar-dense, `ticker.py:398`). Run for seeds 0, 1, 2:

```bash
uv run ophir train --emb-dim 128 --num-heads 8 --num-layers 6 \
  --max-steps 10000 --response-size 10 --seed 0 \
  --val-identity --log-offset-ic --val-batches 200
# repeat with --seed 1 and --seed 2
```

Note: with `--val-identity`, the best checkpoint is now selected on `val_rank_ic_near` (Task 3). Record each run's logged `val_rank_ic_near` (and the per-offset `val_rank_ic_h*`) from the CSV logger.

- [ ] **Step 2: Compute pooled near-IC per seed**

Harvest the validation cross-section and evaluate pooled offsets 1–5, reusing the confirmation patterns:

```bash
uv run python .superpowers/sdd/gpu/harvest.py     # CPU val cross-section harvest (model-free)
uv run python .superpowers/sdd/gpu/eval_pooled_near.py   # loads checkpoints on GPU, pooled-near IC
```

Record the short-horizon pooled near-IC (offsets 1–5) for each of the 3 seeds.

- [ ] **Step 3: Compute the clean near-band reversal ceiling**

Using the same harvested validation `target`/`ids`/`dates`, call the new helper:

```python
from ophir.ceiling import near_band_reversal_ceiling
ceiling = near_band_reversal_ceiling(target, ids, dates, k=5)
```

This replaces the unreliable ~0.119 figure.

- [ ] **Step 4: Apply the decision rule**

- 90-day **near-slice** baseline ≈ **0.066** (the confirmation number; offsets 1–5 of the existing 90-day model).
- Adopt the short-horizon **retrain** only if its pooled near-IC beats 0.066 by a **seed-stable margin** — a clearly higher mean across all three seeds. Otherwise default to **read-near** (operate offset-1 of the existing 90-day model).
- Independently, compare the chosen operating point to the clean ceiling from Step 3: model **>** ceiling ⇒ operating-point fix is the whole story; model **≤** ceiling ⇒ architectural headroom remains (response-block masking denies the 1-day feature, `models.py:434-460`) — note this for a future architecture spec.

- [ ] **Step 5: Record the results**

Append to `docs/forecast-ceiling-results.md`: the 3-seed short-horizon pooled near-IC table, the clean near-band reversal ceiling, the 90-day near-slice baseline, and the operating-point decision (read-near vs retrain) with its justification. Update the `forecast-ceiling-investigation` memory with the chosen operating point and whether architectural headroom remains.

- [ ] **Step 6: Commit the docs**

```bash
git add docs/forecast-ceiling-results.md
git commit -m "docs: record operating-point experiment results and decision"
```

(The `forecast-ceiling-investigation` memory lives outside the repo — update it via the memory tooling, not a git commit.)

---

## Self-Review

**Spec coverage:**
- A1 `val_rank_ic_near` helper → Task 1. ✓
- A2 log ungated → Task 2 (Step 3b). ✓
- A3 `near_offset_k` param default 5 → Task 2 (Step 3a). ✓
- A4 checkpoint monitor switch gated on identity → Task 3. ✓
- B1 short-horizon 3-seed runs → Task 5 (Steps 1-2). ✓
- B2 clean reversal ceiling helper (productionized + tested) → Task 4. ✓
- B3 decision rule → Task 5 (Step 4). ✓
- B4 record results + memory → Task 5 (Step 5). ✓
- Testing section (near-IC, ceiling, register wiring) → Tasks 1-4 tests. ✓
- C deferred → no task, consistent with spec. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code. ✓

**Type consistency:** `val_rank_ic_near(pred, target, ids, dates, offsets, k=5)` is defined in Task 1 and called identically in Task 2; `near_offset_k` attribute used consistently; `_best_checkpoint_callback(file_name, monitor_near_ic)` and `fetch_base_trainer(..., monitor_near_ic=...)` consistent across Task 3; `near_band_reversal_ceiling(target, ids, dates, k=5)` consistent in Task 4 and called in Task 5. ✓
