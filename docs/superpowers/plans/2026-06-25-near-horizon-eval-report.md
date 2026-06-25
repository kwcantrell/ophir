# Near-horizon Operating Point in the Eval Report — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ophir evaluate` report the near-horizon cross-sectional rank-IC (the operating point the trader and training logs already use) as the headline number, with the per-offset decay curve, while leaving the pooled metric unchanged.

**Architecture:** Measurement-only change confined to `src/ophir/evaluate.py` plus tests. The eval accumulator gains a per-row forecast-offset tensor (built with the *same* `trading_day_offsets` helper training uses, lazy-imported to avoid an import cycle). `evaluate_model` then computes a near-band IC (offsets `1..k`) and a per-offset curve from already-existing helpers (`dedupe_by_ticker_date`, `rank_ic`, `rank_ic_by_offset`). The report renders the near number as headline and the curve as a separate section. No model retrain, no training-logging change, no `trading/forecast.py` change.

**Tech Stack:** Python 3.10+ (`from __future__ import annotations` already in file), PyTorch (CPU for tests), Typer, pytest, mypy strict, ruff.

## Global Constraints

- **mypy `strict = True`, targets Python 3.10.** Keep `src/ophir` fully typed; new functions need complete annotations. Run `uv run mypy src/ophir`.
- **ruff targets Python 3.12.** Run `uv run ruff check . && uv run ruff format --check .`.
- **pytest runs `filterwarnings = error`.** Any project-owned warning fails the suite. New code must not emit warnings (empty offset bands must short-circuit to `nan` *before* calling `rank_ic`).
- **Tests must never touch the network, CUDA, or the package `.ophir/` layout.** Use the existing `_FakeModel` (its `.cuda()` returns `self`) and plain CPU tensors.
- **NumPy-style docstrings** on every new public function, matching the density in `evaluate.py`.
- **Single source of truth for offset constants.** The near-band default and the offset buckets must derive from one place each — `_NEAR_OFFSET_K` in `evaluate.py` and `_OFFSET_BUCKETS` in `training_models.py`. Do not hardcode `(1, 2, 5, ...)` a second time.
- **Update `CHANGELOG.md` `[Unreleased]`** for this change (folded into the final task).

---

### Task 1: Accumulate per-row forecast offsets in the eval path

**Files:**
- Modify: `src/ophir/evaluate.py` (`AccumulatedEval` dataclass ~L56-63; `accumulate_targets` ~L319-400)
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `trading_day_offsets(trade_occured: torch.Tensor) -> torch.Tensor` from `ophir.training_models` (returns `trade_occured.long().cumsum(dim=1)` — the 1-based cumulative trading-day rank within the response window).
- Produces: `AccumulatedEval.r_close_offsets: torch.Tensor | None` — 1-D CPU int tensor of 1-based forecast offsets, aligned row-for-row with `r_close_ids` / `r_close_dates` (present iff the loader carries identity, else `None`).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_evaluate.py` (place the helper near `_toy_identity_batch`):

```python
def _toy_offset_batch() -> dict[str, object]:
    # B=2 tickers, S=3, response_size=2 -> response cols 1,2 give offsets 1 and 2.
    # r_close (channel 0) ranks ticker 5 above ticker 6 on both response days.
    targets = torch.tensor(
        [
            [[0.0, 0.1, 0.1], [0.2, 0.1, 0.1], [0.3, 0.1, 0.1]],
            [[0.0, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]],
        ]
    )
    return {
        "feature_input": torch.zeros(2, 3, 12),
        "targets": targets,
        "trade_occured": torch.ones(2, 3, dtype=torch.bool),
        "response_size": torch.tensor(2),
        "stock_id": torch.tensor([5, 6]),
        "date_ordinal": torch.tensor([[10, 11, 12], [10, 11, 12]]),
    }


def test_accumulate_targets_carries_offsets() -> None:
    model = _FakeModel()
    acc = accumulate_targets(model, [_toy_offset_batch()], max_batches=1)  # type: ignore[arg-type]

    assert acc.r_close_offsets is not None
    pred, _ = acc.channels["r_close"]
    # One offset per r_close prediction row, aligned with ids.
    assert acc.r_close_offsets.shape == pred.shape
    assert acc.r_close_ids is not None
    # Row-major over (B, R): ticker5@off1, ticker5@off2, ticker6@off1, ticker6@off2.
    assert acc.r_close_ids.tolist() == [5, 5, 6, 6]
    assert acc.r_close_offsets.tolist() == [1, 2, 1, 2]


def test_accumulate_targets_offsets_none_without_identity() -> None:
    model = _FakeModel()
    acc = accumulate_targets(model, [_toy_batch()], max_batches=1)  # type: ignore[arg-type]
    assert acc.r_close_offsets is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_evaluate.py::test_accumulate_targets_carries_offsets tests/test_evaluate.py::test_accumulate_targets_offsets_none_without_identity -v`
Expected: FAIL — `AttributeError: 'AccumulatedEval' object has no attribute 'r_close_offsets'`.

- [ ] **Step 3: Add the dataclass field**

In `src/ophir/evaluate.py`, extend `AccumulatedEval` (after `r_close_dates`):

```python
@dataclass
class AccumulatedEval:
    """Masked predictions/targets, persistence baselines, and r_close identity."""

    channels: dict[str, tuple[torch.Tensor, torch.Tensor]]
    baselines: dict[str, torch.Tensor] = field(default_factory=dict)
    r_close_ids: torch.Tensor | None = None
    r_close_dates: torch.Tensor | None = None
    r_close_offsets: torch.Tensor | None = None
```

- [ ] **Step 4: Populate offsets in `accumulate_targets`**

In `src/ophir/evaluate.py`, inside `accumulate_targets`, add an offsets accumulator next to the id/date lists:

```python
    id_lists: list[torch.Tensor] = []
    date_lists: list[torch.Tensor] = []
    offset_lists: list[torch.Tensor] = []
```

Within the identity block (where `resp_dates` / `ids_br` are built), append the per-row offset using the same `trading_day_offsets` helper training uses (lazy import mirrors how `training_models` imports `evaluate`):

```python
            # Opt-in identity, collected parallel to the r_close pred/target above.
            if output.stock_id is not None and output.date_ordinal is not None:
                from ophir.training_models import trading_day_offsets

                resp_dates = output.date_ordinal[:, -rs:]  # (B, R)
                ids_br = output.stock_id.view(-1, 1).expand(-1, rs)  # (B, R)
                offsets = trading_day_offsets(mask)  # (B, R) cumulative trading-day rank
                id_lists.append(ids_br[mask].reshape(-1).cpu())
                date_lists.append(resp_dates[mask].reshape(-1).cpu())
                offset_lists.append(offsets[mask].reshape(-1).cpu())
```

Then add the field to the returned `AccumulatedEval`:

```python
    return AccumulatedEval(
        channels={
            name: (torch.cat(preds), torch.cat(targets))
            for name, (preds, targets) in collected.items()
        },
        baselines={name: torch.cat(b) for name, b in baseline_lists.items()},
        r_close_ids=torch.cat(id_lists) if id_lists else None,
        r_close_dates=torch.cat(date_lists) if date_lists else None,
        r_close_offsets=torch.cat(offset_lists) if offset_lists else None,
    )
```

Also extend the `accumulate_targets` docstring `Returns` section to mention `r_close_offsets` ("the per-prediction 1-based forecast offset, aligned with `r_close_ids`").

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_evaluate.py::test_accumulate_targets_carries_offsets tests/test_evaluate.py::test_accumulate_targets_offsets_none_without_identity -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Typecheck and commit**

Run: `uv run mypy src/ophir && uv run ruff check src/ophir/evaluate.py`
Expected: no errors.

```bash
git add src/ophir/evaluate.py tests/test_evaluate.py
git commit -m "feat(evaluate): accumulate per-row forecast offsets in eval path"
```

---

### Task 2: `rank_ic_near` pure helper (shared near-band math)

**Files:**
- Modify: `src/ophir/evaluate.py` (add `_NEAR_OFFSET_K` constant near `_METRIC_ORDER` ~L53; add `rank_ic_near` after `rank_ic_by_offset` ~L316)
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `dedupe_by_ticker_date`, `rank_ic` (both already in `evaluate.py`); `val_rank_ic_near` from `ophir.training_models` (for the reconciliation test only).
- Produces: `rank_ic_near(pred, target, ids, dates, offsets, k: int = _NEAR_OFFSET_K) -> float` and module constant `_NEAR_OFFSET_K: int = 5`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_evaluate.py` (and add `rank_ic_near` to the `from ophir.evaluate import (...)` block at the top):

```python
def test_rank_ic_near_matches_val_rank_ic_near() -> None:
    from ophir.evaluate import rank_ic_near
    from ophir.training_models import val_rank_ic_near

    # offsets 1 (perfect rank) and 2 (anti-rank) are in-band; offset 6 is out.
    pred = torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 5.0, 6.0])
    target = torch.tensor([1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0, 2.0])
    ids = torch.tensor([1, 2, 3, 1, 2, 3, 1, 2])
    dates = torch.tensor([1, 1, 1, 2, 2, 2, 3, 3])
    offsets = torch.tensor([1, 1, 1, 2, 2, 2, 6, 6])

    near = rank_ic_near(pred, target, ids, dates, offsets, k=5)
    # day1 IC = +1, day2 IC = -1 -> mean 0; offset-6 rows excluded.
    assert near == pytest.approx(0.0, abs=1e-6)
    # Reconciles exactly with the live training-side metric.
    assert near == pytest.approx(val_rank_ic_near(pred, target, ids, dates, offsets, k=5))


def test_rank_ic_near_empty_band_is_nan_without_warning() -> None:
    from ophir.evaluate import rank_ic_near

    pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([1.0, 2.0])
    ids = torch.tensor([1, 2])
    dates = torch.tensor([1, 1])
    offsets = torch.tensor([10, 10])  # all outside the 1..5 band
    assert math.isnan(rank_ic_near(pred, target, ids, dates, offsets, k=5))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_evaluate.py::test_rank_ic_near_matches_val_rank_ic_near tests/test_evaluate.py::test_rank_ic_near_empty_band_is_nan_without_warning -v`
Expected: FAIL — `ImportError: cannot import name 'rank_ic_near'`.

- [ ] **Step 3: Add the constant and the helper**

In `src/ophir/evaluate.py`, add the constant after `_METRIC_ORDER`:

```python
#: Inclusive upper bound (in 1-based trading-day forecast offsets) of the near
#: band where cross-sectional skill concentrates. Mirrors the training-side
#: ``LightningOHLCPredictor.near_offset_k`` default so the offline report and the
#: live ``val_rank_ic_near`` metric agree.
_NEAR_OFFSET_K = 5
```

Add the helper after `rank_ic_by_offset`:

```python
def rank_ic_near(
    pred: torch.Tensor,
    target: torch.Tensor,
    ids: torch.Tensor,
    dates: torch.Tensor,
    offsets: torch.Tensor,
    k: int = _NEAR_OFFSET_K,
) -> float:
    """Pooled cross-sectional rank-IC over near forecast offsets ``1..k``.

    Restricts the rows to 1-based trading-day offsets ``1 <= offset <= k`` (the
    near band where cross-sectional skill concentrates), dedupes to one
    prediction per ``(ticker, date)``, and averages the daily Spearman rank-IC.
    Reuses :func:`dedupe_by_ticker_date` and :func:`rank_ic` so the offline
    report and the live :func:`ophir.training_models.val_rank_ic_near` metric
    share identical math.

    Parameters
    ----------
    pred, target, ids, dates : torch.Tensor
        Equal-length 1-D tensors, as accumulated for :func:`rank_ic`.
    offsets : torch.Tensor
        Same-length integer tensor of 1-based trading-day forecast offsets.
    k : int, optional
        Inclusive upper bound of the near band. Defaults to ``_NEAR_OFFSET_K``.

    Returns
    -------
    float
        Mean daily rank-IC over the near band, or ``nan`` when no rows fall in it.
    """
    if pred.numel() == 0:
        return float("nan")
    sel = (offsets >= 1) & (offsets <= k)
    if not bool(sel.any()):
        return float("nan")
    dp, dt, dd = dedupe_by_ticker_date(pred[sel], target[sel], ids[sel], dates[sel])
    return rank_ic(dp, dt, dd)["ic_mean"]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_evaluate.py::test_rank_ic_near_matches_val_rank_ic_near tests/test_evaluate.py::test_rank_ic_near_empty_band_is_nan_without_warning -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Typecheck and commit**

Run: `uv run mypy src/ophir`
Expected: no errors.

```bash
git add src/ophir/evaluate.py tests/test_evaluate.py
git commit -m "feat(evaluate): add rank_ic_near near-band metric reconciled with training"
```

---

### Task 3: Wire near-band + per-offset curve into `evaluate_model`

**Files:**
- Modify: `src/ophir/evaluate.py` (`evaluate_model` ~L403-447)
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `AccumulatedEval.r_close_offsets` (Task 1); `rank_ic_near` (Task 2); `rank_ic_by_offset` (existing); `_OFFSET_BUCKETS` from `ophir.training_models`.
- Produces: `evaluate_model` results where `results["r_close"]` additionally carries `rank_ic_near: float` and one `h{offset}: float` key per bucket in `_OFFSET_BUCKETS` — present iff the loader carried identity (offsets available).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_evaluate.py`:

```python
def test_evaluate_model_reports_near_and_offset_curve() -> None:
    from ophir import evaluate as ev

    model = _FakeModel()  # perfect predictions
    out = ev.evaluate_model(model, [_toy_offset_batch()], max_batches=1)  # type: ignore[arg-type]
    rc = out["r_close"]

    # Headline operating point and the near offsets resolve to perfect IC.
    assert rc["rank_ic_near"] == pytest.approx(1.0, abs=1e-5)
    assert rc["h1"] == pytest.approx(1.0, abs=1e-5)
    assert rc["h2"] == pytest.approx(1.0, abs=1e-5)
    # Offsets with no rows in this toy batch are nan.
    assert math.isnan(rc["h5"])


def test_evaluate_model_pooled_rank_ic_unchanged_with_offsets() -> None:
    from ophir import evaluate as ev

    out = ev.evaluate_model(_FakeModel(), [_toy_identity_batch()], max_batches=1)  # type: ignore[arg-type]
    # The pooled metric is computed exactly as before adding offsets.
    assert abs(out["r_close"]["rank_ic_mean"] - 1.0) < 1e-6
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_evaluate.py::test_evaluate_model_reports_near_and_offset_curve -v`
Expected: FAIL — `KeyError: 'rank_ic_near'`.

- [ ] **Step 3: Extend `evaluate_model`**

In `src/ophir/evaluate.py`, after the existing pooled `rank_ic` block in `evaluate_model`, add the near + curve computation. The existing block ends:

```python
    if acc.r_close_ids is not None and acc.r_close_dates is not None:
        pred, target = acc.channels["r_close"]
        dp, dt, dd = dedupe_by_ticker_date(pred, target, acc.r_close_ids, acc.r_close_dates)
        ic = rank_ic(dp, dt, dd)
        results["r_close"]["rank_ic_mean"] = ic["ic_mean"]
        results["r_close"]["rank_ic_ir"] = ic["ic_ir"]
```

Append immediately after it:

```python
    if acc.r_close_offsets is not None:
        from ophir.training_models import _OFFSET_BUCKETS

        assert acc.r_close_ids is not None and acc.r_close_dates is not None
        pred, target = acc.channels["r_close"]
        results["r_close"]["rank_ic_near"] = rank_ic_near(
            pred, target, acc.r_close_ids, acc.r_close_dates, acc.r_close_offsets
        )
        results["r_close"].update(
            rank_ic_by_offset(
                pred,
                target,
                acc.r_close_ids,
                acc.r_close_dates,
                acc.r_close_offsets,
                _OFFSET_BUCKETS,
            )
        )
```

Update the `evaluate_model` docstring `Returns` note to mention that `r_close` additionally carries `rank_ic_near` and per-offset `h{n}` keys when the loader carries identity.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_evaluate.py::test_evaluate_model_reports_near_and_offset_curve tests/test_evaluate.py::test_evaluate_model_pooled_rank_ic_unchanged_with_offsets -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Typecheck and commit**

Run: `uv run mypy src/ophir`
Expected: no errors.

```bash
git add src/ophir/evaluate.py tests/test_evaluate.py
git commit -m "feat(evaluate): surface near-band + per-offset IC from evaluate_model"
```

---

### Task 4: Render the operating point in the report + CHANGELOG

**Files:**
- Modify: `src/ophir/evaluate.py` (`_METRIC_ORDER` ~L43-53; add `format_offset_decay` after `format_report` ~L494; `evaluate` CLI echo ~L601)
- Modify: `CHANGELOG.md` (`[Unreleased]`)
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `evaluate_model` results carrying `rank_ic_near` + `h{n}` (Task 3); `_OFFSET_BUCKETS` from `ophir.training_models`; `_format_metric` (existing).
- Produces: `format_offset_decay(results_by_label: dict[str, dict[str, dict[str, float]]]) -> str` (empty string when no curve present); `rank_ic_near` added to `_METRIC_ORDER`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_evaluate.py`:

```python
def test_format_report_includes_rank_ic_near() -> None:
    from ophir.evaluate import format_report

    md = format_report(
        {"best-val": {"r_close": {"rank_ic_near": 0.066}, "upside": {}, "downside": {}}}
    )
    assert "rank_ic_near" in md
    assert "0.06600" in md


def test_format_offset_decay_renders_curve() -> None:
    from ophir.evaluate import format_offset_decay

    results = {
        "best-val": {
            "r_close": {"rank_ic_near": 0.066, "h1": 0.08, "h2": 0.07, "h5": float("nan")}
        }
    }
    md = format_offset_decay(results)
    assert "Near-horizon IC decay" in md
    assert "h1" in md
    assert "0.08000" in md
    assert "n/a" in md  # h5 is nan


def test_format_offset_decay_empty_without_curve() -> None:
    from ophir.evaluate import format_offset_decay

    assert format_offset_decay({"best-val": {"r_close": {"rank_ic_mean": 0.02}}}) == ""
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_evaluate.py::test_format_offset_decay_renders_curve tests/test_evaluate.py::test_format_report_includes_rank_ic_near -v`
Expected: FAIL — `ImportError: cannot import name 'format_offset_decay'` and the report assertion fails (no `rank_ic_near` row).

- [ ] **Step 3: Add `rank_ic_near` to the metric table order**

In `src/ophir/evaluate.py`, extend `_METRIC_ORDER` (after `rank_ic_ir`):

```python
_METRIC_ORDER = (
    "n",
    "mae",
    "rmse",
    "bias",
    "directional_accuracy",
    "skill_score",
    "skill_vs_persistence",
    "rank_ic_mean",
    "rank_ic_ir",
    "rank_ic_near",
)
```

- [ ] **Step 4: Add the decay renderer**

In `src/ophir/evaluate.py`, add after `format_report`:

```python
def format_offset_decay(results_by_label: dict[str, dict[str, dict[str, float]]]) -> str:
    """Render the per-offset r_close IC decay as a Markdown table.

    One row per forecast offset in ``_OFFSET_BUCKETS``, one column per evaluated
    checkpoint, so the near-horizon concentration (and its dilution at longer
    leads) is visible at a glance. Returns the empty string when no checkpoint
    carries the per-offset ``h{n}`` keys (e.g. the loader had no identity).

    Parameters
    ----------
    results_by_label : dict[str, dict[str, dict[str, float]]]
        ``{label: {channel: metrics}}`` as returned by :func:`evaluate_model`.

    Returns
    -------
    str
        A Markdown section, or ``""`` when there is no curve to show.
    """
    from ophir.training_models import _OFFSET_BUCKETS

    labels = list(results_by_label)
    keys = [f"h{int(b)}" for b in _OFFSET_BUCKETS]
    present = [
        key
        for key in keys
        if any(key in results_by_label[label].get("r_close", {}) for label in labels)
    ]
    if not present:
        return ""
    lines = ["## Near-horizon IC decay (r_close)", ""]
    lines.append("| offset | " + " | ".join(labels) + " |")
    lines.append("| --- |" + " --- |" * len(labels))
    for key in present:
        cells = []
        for label in labels:
            metrics = results_by_label[label].get("r_close", {})
            cells.append(_format_metric(key, metrics[key]) if key in metrics else "n/a")
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    return "\n".join(lines).rstrip() + "\n"
```

- [ ] **Step 5: Print the decay section from the CLI**

In `src/ophir/evaluate.py`, at the end of `evaluate`, replace the final echo:

```python
    typer.echo(format_report(results_by_label))
```

with:

```python
    typer.echo(format_report(results_by_label))
    decay = format_offset_decay(results_by_label)
    if decay:
        typer.echo(decay)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_evaluate.py::test_format_report_includes_rank_ic_near tests/test_evaluate.py::test_format_offset_decay_renders_curve tests/test_evaluate.py::test_format_offset_decay_empty_without_curve -v`
Expected: PASS (3 passed).

- [ ] **Step 7: Update the CHANGELOG**

In `CHANGELOG.md`, under `## [Unreleased]` → `### Added`, add:

```markdown
- `ophir evaluate` now reports the near-horizon operating point: a `rank_ic_near`
  headline (pooled cross-sectional rank-IC over forecast offsets 1..5, the band
  where the model's skill concentrates) alongside the existing pooled
  `rank_ic_mean`, plus a "Near-horizon IC decay" table showing rank-IC per
  forecast offset. The near metric reuses the same math as the training-side
  `val_rank_ic_near`, so the offline report and the live validation metric agree.
  Measurement only — no model, training, or trading-path change.
```

- [ ] **Step 8: Full verification and commit**

Run: `uv run pytest && uv run mypy src/ophir && uv run ruff check . && uv run ruff format --check .`
Expected: full suite passes, no type or lint errors.

```bash
git add src/ophir/evaluate.py tests/test_evaluate.py CHANGELOG.md
git commit -m "feat(evaluate): report near-horizon operating point and IC decay curve"
```

---

## Self-Review

**Spec coverage:**
- Spec Component A (accumulate offsets) → Task 1. ✓
- Spec A2 (offset construction matches training exactly) → Task 1 Step 4 reuses `trading_day_offsets`; Task 2's reconciliation test pins equality. ✓
- Spec Component B1 (near + per-offset in `evaluate_model`) → Tasks 2 (helper) + 3 (wiring). ✓
- Spec B2 (report formatting: near headline + curve) → Task 4. ✓
- Spec B3 (near-k single source, `k=5`) → Task 2 `_NEAR_OFFSET_K = 5`. ✓
- Spec C1 (accumulator carries offsets) → Task 1 tests. ✓
- Spec C2 (reconciliation vs `val_rank_ic_near`) → Task 2 `test_rank_ic_near_matches_val_rank_ic_near`. ✓
- Spec C3 (per-offset wiring) → Task 3 `test_evaluate_model_reports_near_and_offset_curve`. ✓
- Spec C4 (pooled regression guard) → Task 3 `test_evaluate_model_pooled_rank_ic_unchanged_with_offsets`. ✓
- Spec C5 (empty-band nan, no warning) → Task 2 `test_rank_ic_near_empty_band_is_nan_without_warning`. ✓
- Spec acceptance criterion (near > pooled on the real checkpoint) is a CUDA-only manual check, noted here for the operator to run after merge: `uv run ophir evaluate` and confirm `rank_ic_near` exceeds `rank_ic_mean`, matching the training logs' `val_rank_ic_near`.

**Placeholder scan:** No TBD/TODO; every code step shows complete code. ✓

**Type consistency:** `rank_ic_near` signature identical across Tasks 2/3; `AccumulatedEval.r_close_offsets` named consistently in Tasks 1/3; `format_offset_decay` signature identical in Task 4 and tests; `_OFFSET_BUCKETS` / `trading_day_offsets` / `val_rank_ic_near` referenced with their real signatures from `training_models`. ✓
