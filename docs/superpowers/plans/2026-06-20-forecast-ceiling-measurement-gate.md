# Forecasting-ceiling measurement gate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a trustworthy ruler for ophir's forecasting skill and run the three gating experiments (E0 re-measure existing runs, E1 naive baselines, E2 full-budget reality check) that determine where the cross-sectional rank-IC ceiling lives.

**Architecture:** Add one focused, pure, offline module `src/ophir/ceiling.py` holding all investigation math: training-run IC-trajectory parsing, multi-seed aggregation + minimum-detectable-effect (MDE), and naive cross-sectional baselines that reuse the production `evaluate.rank_ic` / `dedupe_by_ticker_date` so methodology is identical. The experiments themselves (E0 analysis, E1 harvest+score, E2 GPU runs) are *operational* tasks that drive those pure helpers over real artifacts and record findings in a living results log.

**Tech Stack:** Python 3.10+ (mypy floor), PyTorch, pandas, numpy, pytest, run via `uv`. Pure helpers are CPU/offline; experiment runs use the RTX 3090 via `uv run`.

## Global Constraints

- **mypy is `strict = True`, targets Python 3.10**; ruff targets 3.12. `src/ophir/ceiling.py` must be fully typed. (Do not change either version floor.)
- **pytest runs `filterwarnings = error`** — any warning the project owns fails the suite.
- **Tests must never touch network, CUDA, or the package `.ophir/` layout.** Use `tmp_path` and synthetic fixtures. The pure `ceiling.py` helpers are unit-tested offline; the operational E0/E1/E2 steps that read real run artifacts or run training are **not** pytest tests — they run via `uv run` and record results in a doc.
- **NumPy-style docstrings** throughout `src/ophir/ceiling.py`, matching existing density.
- Imports: `known-first-party = ["ophir"]`.
- Update `[Unreleased]` in `CHANGELOG.md` for the new module.
- **Leave alone:** `main`'s unpushed commits, the uncommitted `.claude/settings.json`, and the modified `docs/rezero-init-sweep-runbook.md`. Stage only files this plan creates/edits.
- Reuse production IC math (`ophir.evaluate.rank_ic`, `dedupe_by_ticker_date`) — never reimplement Spearman/day-grouping.

## File Structure

- `src/ophir/ceiling.py` *(new)* — all pure investigation helpers. One responsibility: turn run-metric logs and (signal, target, id, date) arrays into IC summaries, aggregates, MDEs, and naive baselines. No data loading, no model, no CUDA.
- `tests/test_ceiling.py` *(new)* — offline unit tests with synthetic CSVs and arrays.
- `docs/forecast-ceiling-results.md` *(new)* — living results log; the operational tasks append tables + the GATE verdict here.
- `CHANGELOG.md` *(modify)* — `[Unreleased]` entry.

Existing files **read but not modified**: `src/ophir/evaluate.py` (`rank_ic`, `dedupe_by_ticker_date`), `src/ophir/train.py` (`build_split_handlers`, `build_dataloader`), `src/ophir/register.py` (`load_base_model_ckpt`), `src/ophir/training_models.py` (validation accumulation pattern, lines ~429-450).

---

### Task 1: `ceiling.py` — run IC-trajectory summary (E0 core)

**Files:**
- Create: `src/ophir/ceiling.py`
- Test: `tests/test_ceiling.py`

**Interfaces:**
- Consumes: nothing (pandas only).
- Produces: `RunICSummary` (frozen dataclass: `peak_ic: float`, `peak_step: int`, `best_ckpt_ic: float`, `final_ic: float`) and `run_ic_summary(metrics_csv: str | Path) -> RunICSummary`. `best_ckpt_ic` is the `val_rank_ic` on the minimum-`val_loss` validation row (the row whose checkpoint `ModelCheckpoint(monitor="val_loss")` would save); `final_ic` is the last validation row's `val_rank_ic`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ceiling.py
from pathlib import Path

import pandas as pd

from ophir.ceiling import RunICSummary, run_ic_summary


def _write_metrics(tmp_path: Path) -> Path:
    # Mimic a Lightning CSVLogger metrics.csv: train-step rows have NaN val
    # metrics; epoch rows carry both val_loss_epoch and val_rank_ic.
    rows = [
        {"step": 100, "val_loss_epoch": None, "val_rank_ic": None},   # train step
        {"step": 500, "val_loss_epoch": 0.90, "val_rank_ic": 0.010},  # epoch 1
        {"step": 1000, "val_loss_epoch": 0.70, "val_rank_ic": 0.030}, # peak IC, min loss
        {"step": 1500, "val_loss_epoch": 0.75, "val_rank_ic": 0.020}, # later
        {"step": 2000, "val_loss_epoch": 0.80, "val_rank_ic": 0.014}, # final (annealed)
    ]
    path = tmp_path / "metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_run_ic_summary_extracts_peak_best_final(tmp_path: Path) -> None:
    summary = run_ic_summary(_write_metrics(tmp_path))
    assert summary == RunICSummary(
        peak_ic=0.030, peak_step=1000, best_ckpt_ic=0.030, final_ic=0.014
    )


def test_run_ic_summary_best_ckpt_differs_from_peak(tmp_path: Path) -> None:
    # min val_loss at a row that is NOT the IC peak.
    rows = [
        {"step": 500, "val_loss_epoch": 0.60, "val_rank_ic": 0.012},  # min loss
        {"step": 1000, "val_loss_epoch": 0.95, "val_rank_ic": 0.040}, # peak IC
    ]
    path = tmp_path / "m.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    summary = run_ic_summary(path)
    assert summary.peak_ic == 0.040
    assert summary.best_ckpt_ic == 0.012
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ceiling.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ophir.ceiling'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ophir/ceiling.py
"""Pure, offline helpers for the forecasting-ceiling investigation.

See ``docs/superpowers/specs/2026-06-20-forecast-ceiling-investigation-design.md``.
Everything here is CPU-only and dependency-light: it parses training-run metric
logs and computes cross-sectional rank-IC baselines, reusing the production IC
math in :mod:`ophir.evaluate` so the offline analysis and the live
``val_rank_ic`` metric agree. No model, no CUDA, no ``.ophir/`` layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    """Return the first of ``candidates`` present in ``df``.

    Lightning's CSVLogger names a metric logged with both ``on_step`` and
    ``on_epoch`` as ``<name>_epoch``; one logged only ``on_epoch`` keeps its bare
    name. This tolerates either spelling.
    """
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"none of {candidates} present in {list(df.columns)}")


@dataclass(frozen=True)
class RunICSummary:
    """Peak / saved-checkpoint / final ``val_rank_ic`` for one training run.

    Attributes
    ----------
    peak_ic, peak_step : float, int
        The maximum ``val_rank_ic`` over the run and the step it occurred at.
    best_ckpt_ic : float
        ``val_rank_ic`` on the minimum-``val_loss`` validation row — the row
        whose checkpoint ``ModelCheckpoint(monitor="val_loss")`` would persist.
    final_ic : float
        ``val_rank_ic`` on the last validation row (the fully-annealed value).
    """

    peak_ic: float
    peak_step: int
    best_ckpt_ic: float
    final_ic: float


def run_ic_summary(metrics_csv: str | Path) -> RunICSummary:
    """Summarise a run's ``val_rank_ic`` trajectory from its ``metrics.csv``.

    Parameters
    ----------
    metrics_csv : str or Path
        Path to a Lightning CSVLogger ``metrics.csv``.

    Returns
    -------
    RunICSummary
        Peak, saved-checkpoint, and final ``val_rank_ic``.

    Raises
    ------
    ValueError
        If no validation rows carry ``val_rank_ic``.
    """
    df = pd.read_csv(metrics_csv)
    ic_col = _pick_column(df, ("val_rank_ic",))
    step_col = _pick_column(df, ("step",))
    val = df.dropna(subset=[ic_col])
    if val.empty:
        raise ValueError(f"no {ic_col} rows in {metrics_csv}")
    peak = val.loc[val[ic_col].idxmax()]
    loss_col = _pick_column(df, ("val_loss_epoch", "val_loss"))
    with_loss = val.dropna(subset=[loss_col])
    best = with_loss.loc[with_loss[loss_col].idxmin()] if not with_loss.empty else peak
    final = val.iloc[-1]
    return RunICSummary(
        peak_ic=float(peak[ic_col]),
        peak_step=int(peak[step_col]),
        best_ckpt_ic=float(best[ic_col]),
        final_ic=float(final[ic_col]),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ceiling.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Lint, format, type-check**

Run: `uv run ruff check src/ophir/ceiling.py tests/test_ceiling.py && uv run ruff format src/ophir/ceiling.py tests/test_ceiling.py && uv run mypy src/ophir`
Expected: all pass, no errors.

- [ ] **Step 6: Commit**

```bash
git add src/ophir/ceiling.py tests/test_ceiling.py
git commit -m "Add run_ic_summary for ceiling investigation"
```

---

### Task 2: `ceiling.py` — multi-seed aggregation + MDE (E0 core)

**Files:**
- Modify: `src/ophir/ceiling.py`
- Test: `tests/test_ceiling.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `ICAggregate` (frozen dataclass: `mean: float`, `min: float`, `max: float`, `std: float`, `n: int`).
  - `aggregate_ic(values: Sequence[float]) -> ICAggregate`.
  - `mde_for_group_difference(replicates: Sequence[float], *, seeds_per_group: int, sigmas: float = 2.0) -> float` — the smallest difference between two `seeds_per_group`-seed config means worth believing, defined as `sigmas * s * sqrt(2 / seeds_per_group)` where `s` is the across-seed sample std (`ddof=1`) of same-config `replicates`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_ceiling.py
import math

import pytest

from ophir.ceiling import ICAggregate, aggregate_ic, mde_for_group_difference


def test_aggregate_ic_basic() -> None:
    agg = aggregate_ic([0.0139, 0.0109, 0.0171])
    assert agg.n == 3
    assert agg.min == pytest.approx(0.0109)
    assert agg.max == pytest.approx(0.0171)
    assert agg.mean == pytest.approx((0.0139 + 0.0109 + 0.0171) / 3)


def test_aggregate_ic_empty_raises() -> None:
    with pytest.raises(ValueError):
        aggregate_ic([])


def test_mde_matches_formula() -> None:
    reps = [0.0139, 0.0109, 0.0171]
    s = float(pd.Series(reps).std(ddof=1))
    expected = 2.0 * s * math.sqrt(2.0 / 3)
    assert mde_for_group_difference(reps, seeds_per_group=3) == pytest.approx(expected)


def test_mde_needs_two_replicates() -> None:
    with pytest.raises(ValueError):
        mde_for_group_difference([0.01], seeds_per_group=3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ceiling.py -k "aggregate or mde" -v`
Expected: FAIL — `ImportError: cannot import name 'aggregate_ic'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/ophir/ceiling.py
from collections.abc import Sequence

import numpy as np


@dataclass(frozen=True)
class ICAggregate:
    """Mean / min / max / sample-std / count over a config's seed replicates."""

    mean: float
    min: float
    max: float
    std: float
    n: int


def aggregate_ic(values: Sequence[float]) -> ICAggregate:
    """Aggregate one config's per-seed IC values.

    Parameters
    ----------
    values : sequence of float
        Per-seed IC values for a single configuration.

    Returns
    -------
    ICAggregate
        ``std`` is the sample standard deviation (``ddof=1``), or ``0.0`` for a
        single value.

    Raises
    ------
    ValueError
        If ``values`` is empty.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("need at least one IC value")
    return ICAggregate(
        mean=float(arr.mean()),
        min=float(arr.min()),
        max=float(arr.max()),
        std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        n=int(arr.size),
    )


def mde_for_group_difference(
    replicates: Sequence[float], *, seeds_per_group: int, sigmas: float = 2.0
) -> float:
    """Minimum detectable effect for a difference of two seed-mean ICs.

    Estimates the seed-noise scale ``s`` from same-config ``replicates`` and
    returns ``sigmas * s * sqrt(2 / seeds_per_group)`` — the half-width below
    which a gap between two ``seeds_per_group``-seed config means is consistent
    with seed noise. Two configs whose mean IC differ by less than this should
    not be called different.

    Raises
    ------
    ValueError
        If fewer than two ``replicates`` are supplied.
    """
    arr = np.asarray(replicates, dtype=float)
    if arr.size < 2:
        raise ValueError("need >= 2 replicates to estimate seed noise")
    s = float(arr.std(ddof=1))
    return sigmas * s * float(np.sqrt(2.0 / seeds_per_group))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ceiling.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Lint, format, type-check**

Run: `uv run ruff check src/ophir/ceiling.py tests/test_ceiling.py && uv run ruff format src/ophir/ceiling.py tests/test_ceiling.py && uv run mypy src/ophir`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/ophir/ceiling.py tests/test_ceiling.py
git commit -m "Add IC aggregation and MDE helpers"
```

---

### Task 3: E0 — re-measure the 14 on-disk runs (operational, zero GPU)

**Files:**
- Create: `docs/forecast-ceiling-results.md`

**Interfaces:**
- Consumes: `run_ic_summary`, `aggregate_ic`, `mde_for_group_difference` (Tasks 1-2).
- Produces: the **MDE number** and a peak-IC re-read of the rezero conclusion, recorded in the results log. No code.

> This task reads real run artifacts under `src/ophir/.ophir/model/csv-logger/`. It is operational (touches the package layout), so it is **not** a pytest test — run it via `uv run` and paste the output into the results doc.

- [ ] **Step 1: Confirm the run directories exist**

Run: `ls -d src/ophir/.ophir/model/csv-logger/version_{246..259} 2>/dev/null | wc -l`
Expected: `14`. If fewer, list what exists and record which versions are missing in the results doc before continuing.

- [ ] **Step 2: Summarise peak / best-ckpt / final IC for every run**

Run:

```bash
uv run python - <<'PY'
from pathlib import Path
from ophir.ceiling import run_ic_summary
base = Path("src/ophir/.ophir/model/csv-logger")
for v in range(246, 260):
    csv = base / f"version_{v}" / "metrics.csv"
    if not csv.exists():
        print(f"version_{v}: MISSING")
        continue
    s = run_ic_summary(csv)
    print(f"version_{v}: peak={s.peak_ic:.4f}@{s.peak_step} "
          f"best_ckpt={s.best_ckpt_ic:.4f} final={s.final_ic:.4f}")
PY
```

Expected: 14 lines. Sanity check against the known value — `version_247` final should read ≈0.0139 and its peak ≈0.030 (the Finding-A discrepancy). Record the full table in `docs/forecast-ceiling-results.md`.

- [ ] **Step 3: Compute the MDE from same-config seed replicates**

The `rezero_init=0.0` baseline ran at seeds 0/1/2 = `version_247` / `256` / `258`. Use their **peak** ICs as the noise replicates:

```bash
uv run python - <<'PY'
from pathlib import Path
from ophir.ceiling import run_ic_summary, aggregate_ic, mde_for_group_difference
base = Path("src/ophir/.ophir/model/csv-logger")
reps = [run_ic_summary(base / f"version_{v}" / "metrics.csv").peak_ic
        for v in (247, 256, 258)]
print("baseline peak ICs (s0,s1,s2):", [round(r, 4) for r in reps])
print("aggregate:", aggregate_ic(reps))
print("MDE (3 seeds/group, 2 sigma):",
      round(mde_for_group_difference(reps, seeds_per_group=3), 4))
PY
```

Expected: prints the three baseline peak ICs, their aggregate, and a single MDE number. Record the MDE prominently in the results doc — it gates every later "is this a real win" call.

- [ ] **Step 4: Re-read the rezero conclusion on peak IC**

Compare `rezero_init=0.0` (247/256/258) vs `0.1` (252/257/259) on **peak** IC:

```bash
uv run python - <<'PY'
from pathlib import Path
from ophir.ceiling import run_ic_summary, aggregate_ic, mde_for_group_difference
base = Path("src/ophir/.ophir/model/csv-logger")
def peaks(vs): return [run_ic_summary(base / f"version_{v}" / "metrics.csv").peak_ic for v in vs]
b, i = peaks((247, 256, 258)), peaks((252, 257, 259))
ab, ai = aggregate_ic(b), aggregate_ic(i)
mde = mde_for_group_difference(b, seeds_per_group=3)
print(f"init0.0 peak: mean={ab.mean:.4f} min={ab.min:.4f}")
print(f"init0.1 peak: mean={ai.mean:.4f} min={ai.min:.4f}")
print(f"delta={ai.mean - ab.mean:+.4f}  MDE={mde:.4f}  "
      f"real={'YES' if abs(ai.mean - ab.mean) > mde else 'NO (within noise)'}")
PY
```

Expected: a verdict line. **Falsifiable fork:** if `delta` exceeds the MDE the prior final-step conclusion was a measurement artifact (record that rezero must be revisited on peak IC); if within MDE, the rezero conclusion stands on the better ruler too.

- [ ] **Step 5: Write the E0 section of the results log**

Create `docs/forecast-ceiling-results.md` with: a header linking the spec; the full per-run table (Step 2); the MDE (Step 3); the peak-IC rezero re-read + verdict (Step 4); and a one-line decision ("ruler = peak IC; MDE = X; rezero conclusion holds/flips on peak IC").

- [ ] **Step 6: Commit**

```bash
git add docs/forecast-ceiling-results.md
git commit -m "Record E0 re-measurement: peak-IC table, MDE, rezero re-read"
```

---

### Task 4: `ceiling.py` — naive cross-sectional baselines + null (E1 core)

**Files:**
- Modify: `src/ophir/ceiling.py`
- Test: `tests/test_ceiling.py`

**Interfaces:**
- Consumes: `ophir.evaluate.rank_ic`, `dedupe_by_ticker_date`.
- Produces (all operate on 1-D torch tensors of equal length, one row per response observation):
  - `dedupe_rows(target, ids, dates) -> tuple[Tensor, Tensor, Tensor]` — keep first row per `(ticker, date)`.
  - `lagged_target_signal(target, ids, dates, *, lag=1) -> tuple[Tensor, Tensor]` — per-ticker previous-by-date target as a naive AR signal; returns `(signal, valid)` where `valid` is `False` for rows lacking `lag` priors. (Momentum = use as-is; reversal = negate.)
  - `cross_sectional_ic(signal, target, ids, dates, *, valid=None) -> dict[str, float]` — daily cross-sectional rank-IC via the production `dedupe_by_ticker_date` + `rank_ic`, optionally restricted to `valid` rows.
  - `shuffle_within_day(target, dates, *, generator) -> Tensor` — permute targets within each day for the null control.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_ceiling.py
import torch

from ophir.ceiling import (
    cross_sectional_ic,
    dedupe_rows,
    lagged_target_signal,
    shuffle_within_day,
)


def test_dedupe_rows_keeps_first_per_ticker_date() -> None:
    target = torch.tensor([1.0, 2.0, 3.0])
    ids = torch.tensor([10, 10, 11])
    dates = torch.tensor([1, 1, 1])  # (10,1) duplicated
    t, i, d = dedupe_rows(target, ids, dates)
    assert t.tolist() == [1.0, 3.0]
    assert i.tolist() == [10, 11]
    assert d.tolist() == [1, 1]


def test_lagged_signal_uses_prior_date_per_ticker() -> None:
    # ticker 10 on dates 1,2,3 with targets 0.1,0.2,0.3
    target = torch.tensor([0.1, 0.2, 0.3])
    ids = torch.tensor([10, 10, 10])
    dates = torch.tensor([1, 2, 3])
    signal, valid = lagged_target_signal(target, ids, dates, lag=1)
    assert valid.tolist() == [False, True, True]
    assert signal[valid].tolist() == [0.1, 0.2]  # yesterday's target


def test_cross_sectional_ic_perfect_rank_is_one() -> None:
    # two days, signal ranks tickers identically to target each day
    signal = torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    ids = torch.tensor([1, 2, 3, 1, 2, 3])
    dates = torch.tensor([1, 1, 1, 2, 2, 2])
    out = cross_sectional_ic(signal, target, ids, dates)
    assert out["ic_mean"] == pytest.approx(1.0)
    assert out["n_days"] == 2.0


def test_shuffle_within_day_preserves_per_day_multiset() -> None:
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    dates = torch.tensor([1, 1, 2, 2])
    g = torch.Generator().manual_seed(0)
    shuffled = shuffle_within_day(target, dates, generator=g)
    assert sorted(shuffled[dates == 1].tolist()) == [1.0, 2.0]
    assert sorted(shuffled[dates == 2].tolist()) == [3.0, 4.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ceiling.py -k "dedupe or lagged or cross_sectional or shuffle" -v`
Expected: FAIL — `ImportError: cannot import name 'dedupe_rows'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/ophir/ceiling.py
import torch

from ophir.evaluate import dedupe_by_ticker_date, rank_ic


def dedupe_rows(
    target: torch.Tensor, ids: torch.Tensor, dates: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep the first row per ``(ticker, date)`` (stable order).

    Overlapping windows emit several rows per name per day; baselines need one.
    """
    seen: set[tuple[int, int]] = set()
    keep: list[int] = []
    for k, (sid, day) in enumerate(zip(ids.tolist(), dates.tolist(), strict=True)):
        key = (int(sid), int(day))
        if key not in seen:
            seen.add(key)
            keep.append(k)
    idx = torch.tensor(keep, dtype=torch.long)
    return target[idx], ids[idx], dates[idx]


def lagged_target_signal(
    target: torch.Tensor, ids: torch.Tensor, dates: torch.Tensor, *, lag: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-ticker previous-by-date target as a naive autoregressive signal.

    For each row, the signal is that ticker's target ``lag`` observations earlier
    in date order. Rows without ``lag`` priors are flagged invalid.

    Returns
    -------
    signal, valid : torch.Tensor, torch.Tensor
        ``signal`` holds the lagged target (``nan`` where invalid); ``valid`` is
        a boolean mask. Use ``signal`` directly for a momentum baseline or
        negate it for reversal.
    """
    t = target.detach().cpu().numpy()
    i = ids.detach().cpu().numpy()
    d = dates.detach().cpu().numpy()
    order = np.lexsort((d, i))  # primary key = id, secondary = date
    sid = i[order]
    st = t[order]
    lagged = np.full(st.shape, np.nan, dtype=float)
    for k in range(lag, len(order)):
        if sid[k] == sid[k - lag]:
            lagged[k] = st[k - lag]
    signal = np.full(t.shape, np.nan, dtype=float)
    signal[order] = lagged
    valid = ~np.isnan(signal)
    return torch.from_numpy(signal), torch.from_numpy(valid)


def cross_sectional_ic(
    signal: torch.Tensor,
    target: torch.Tensor,
    ids: torch.Tensor,
    dates: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
) -> dict[str, float]:
    """Daily cross-sectional rank-IC of ``signal`` vs ``target``.

    Mirrors the production metric exactly: dedupe to one row per ``(ticker,
    date)`` then average the per-day Spearman correlation via
    :func:`ophir.evaluate.rank_ic`. Optionally restrict to ``valid`` rows first.
    """
    if valid is not None:
        signal, target, ids, dates = signal[valid], target[valid], ids[valid], dates[valid]
    dp, dt, dd = dedupe_by_ticker_date(signal, target, ids, dates)
    return rank_ic(dp, dt, dd)


def shuffle_within_day(
    target: torch.Tensor, dates: torch.Tensor, *, generator: torch.Generator
) -> torch.Tensor:
    """Permute ``target`` within each day — a null whose expected IC is ~0."""
    out = target.clone()
    for day in torch.unique(dates):
        idx = (dates == day).nonzero(as_tuple=True)[0]
        perm = idx[torch.randperm(idx.numel(), generator=generator)]
        out[idx] = target[perm]
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ceiling.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Lint, format, type-check**

Run: `uv run ruff check src/ophir/ceiling.py tests/test_ceiling.py && uv run ruff format src/ophir/ceiling.py tests/test_ceiling.py && uv run mypy src/ophir`
Expected: all pass. (If mypy flags numpy-array typing on `from_numpy`, annotate the locals as needed; do not loosen strictness.)

- [ ] **Step 6: Commit**

```bash
git add src/ophir/ceiling.py tests/test_ceiling.py
git commit -m "Add naive cross-sectional baselines and within-day null"
```

---

### Task 5: E1 — naive-baseline calibration (operational, CPU/minutes)

**Files:**
- Modify: `docs/forecast-ceiling-results.md`

**Interfaces:**
- Consumes: `lagged_target_signal`, `cross_sectional_ic`, `shuffle_within_day`, `dedupe_rows` (Task 4); `build_split_handlers`/`build_dataloader` (`ophir.train`); `load_base_model_ckpt` (`ophir.register`).
- Produces: a naive-baseline IC table + null-control IC, recorded in the results log. No package code.

> Operational: builds the real validation loader and harvests `(target, ids, dates)`. Uses the proven accumulation pattern from `training_models.py:429-450`. Runs via `uv run` on the 3090 (the forward pass needs CUDA flex-attention); the model's *predictions* are discarded — only the input-derived `target_r_close` / `stock_id` / `date_ordinal` are used, so any existing checkpoint works.

- [ ] **Step 1: Harvest `(target, ids, dates)` from the validation set**

```bash
uv run python - <<'PY'
import torch
from ophir import register
from ophir.train import build_split_handlers, build_dataloader

base = f"{register.get_default_data_days_dir()}/stocks"
_, val_handler = build_split_handlers(
    base_path=base, seq_len=365, offset=90, min_volume=1000.0,
    train_min_year=None, train_max_year=2023, val_min_year=2024, val_max_year=None,
    use_sp500=False, use_quality_allowlist=False, clean_rows=False, max_abs_r_close=0.75,
)
val_dl = build_dataloader(val_handler, 90, 32, 0, 8)
model = register.load_base_model_ckpt(strict=False).eval()

tgt, ids, dts = [], [], []
with torch.no_grad():
    for batch in val_dl:
        prepared = model._input_obj(batch)
        out = model.forward(prepared)
        rs = int(out.response_size)
        mask = out.trade_occured[:, -rs:]
        tgt.append(out.target_r_close[:, -rs:][mask].reshape(-1).cpu())
        ids.append(out.stock_id.view(-1, 1).expand(-1, rs)[mask].reshape(-1).cpu())
        dts.append(out.date_ordinal[:, -rs:][mask].reshape(-1).cpu())

torch.save({"target": torch.cat(tgt), "ids": torch.cat(ids), "dates": torch.cat(dts)},
           "/tmp/ophir_val_rows.pt")
print("rows:", torch.cat(tgt).numel())
PY
```

Expected: prints a positive row count and writes `/tmp/ophir_val_rows.pt`. (If `load_base_model_ckpt` finds no checkpoint, point it at a 10k run checkpoint under `src/ophir/.ophir/model/` instead — any will do; predictions are unused.)

- [ ] **Step 2: Score the naive baselines + null**

```bash
uv run python - <<'PY'
import torch
from ophir.ceiling import (cross_sectional_ic, dedupe_rows, lagged_target_signal,
                           shuffle_within_day)

d = torch.load("/tmp/ophir_val_rows.pt")
target, ids, dates = dedupe_rows(d["target"], d["ids"], d["dates"])

mom, valid = lagged_target_signal(target, ids, dates, lag=1)
print("momentum (prev-day return):",
      round(cross_sectional_ic(mom, target, ids, dates, valid=valid)["ic_mean"], 4))
print("reversal (-prev-day return):",
      round(cross_sectional_ic(-mom, target, ids, dates, valid=valid)["ic_mean"], 4))

g = torch.Generator().manual_seed(0)
nulls = [cross_sectional_ic(mom, shuffle_within_day(target, dates, generator=g),
                            ids, dates, valid=valid)["ic_mean"] for _ in range(20)]
print("null (shuffled target, 20 draws): mean",
      round(sum(nulls) / len(nulls), 4), "max", round(max(nulls), 4))
PY
```

Expected: a momentum IC, a reversal IC, and a null mean ≈ 0 (|null| should be well under the Task-3 MDE — if not, the metric is biased and that must be investigated before trusting any IC).

- [ ] **Step 3: Record the E1 verdict**

Append to `docs/forecast-ceiling-results.md`: the naive-baseline IC table, the null mean/max, and the **calibration verdict** vs the model's peak IC (from Task 3) and the MDE. **Falsifiable fork:** if a naive signal's IC is within the MDE of the model's peak IC, the model barely beats naive → ceiling is target/features (favor jumping to E6 in the follow-up plan); if naive ≈ 0 and the model's peak IC clears the MDE, there is real learned skill and headroom → proceed to structural experiments.

- [ ] **Step 4: Commit**

```bash
git add docs/forecast-ceiling-results.md
git commit -m "Record E1 naive-baseline calibration"
```

---

### Task 6: E2 — full-budget reality check (operational, GPU; the big fork)

**Files:**
- Modify: `docs/forecast-ceiling-results.md`

**Interfaces:**
- Consumes: `run_ic_summary`, `aggregate_ic` (Tasks 1-2); `ophir train` CLI.
- Produces: full-budget vs 10k-proxy IC comparison + a proxy-fidelity verdict that sets the budget regime for the deferred E3-E6 plan. No package code.

> Operational GPU runs ("hours-long single-GPU" each). Compute is not the constraint, so run ≥3 seeds. Use the **epoch-driven full budget** (omit `--max-steps`), keeping every other knob at the diagnostic defaults so the only change from the proxy is the budget.

- [ ] **Step 1: Launch the full-budget baseline at three seeds**

```bash
for SEED in 0 1 2; do
  uv run ophir train --emb-dim 128 --num-heads 8 --num-layers 6 \
    --rezero-init 0.0 --seed $SEED --val-identity --log-rezero-gates
done
```

Expected: three runs complete; record each new `version_N` (newest dirs in `src/ophir/.ophir/model/csv-logger/`). Confirm each `hparams.yaml` shows the epoch-driven `max_steps` (hundreds of thousands), not 10000.

- [ ] **Step 2: Summarise full-budget IC and compare to the proxy peak**

```bash
uv run python - <<'PY'
from pathlib import Path
from ophir.ceiling import run_ic_summary, aggregate_ic
base = Path("src/ophir/.ophir/model/csv-logger")
FULL = (___, ___, ___)   # the three version_N from Step 1
peaks = [run_ic_summary(base / f"version_{v}" / "metrics.csv").peak_ic for v in FULL]
print("full-budget peak ICs:", [round(p, 4) for p in peaks])
print("full-budget aggregate:", aggregate_ic(peaks))
# proxy baseline peak aggregate (247/256/258) for reference:
proxy = [run_ic_summary(base / f"version_{v}" / "metrics.csv").peak_ic for v in (247, 256, 258)]
print("proxy aggregate:", aggregate_ic(proxy))
PY
```

Expected: full-budget peak-IC aggregate vs the proxy aggregate. Fill in the three `version_N` before running.

- [ ] **Step 3: Check proxy ranking fidelity on a contrasting pair**

Re-run one knob the proxy ranked (e.g. `rezero_lr` high vs default) at full budget, one seed each, and check the sign of the gap matches the proxy's:

```bash
uv run ophir train --emb-dim 128 --num-heads 8 --num-layers 6 --rezero-lr 3e-3 \
  --seed 0 --val-identity --log-rezero-gates
# default rezero_lr full-budget seed 0 is already Step 1's seed-0 run.
```

Expected: record the high-`rezero_lr` `version_N` and its peak IC; compare to the Step-1 seed-0 default. Note whether the proxy's "high `rezero_lr` hurts" ranking survives at full budget.

- [ ] **Step 4: Record the E2 verdict + GATE decision**

Append to `docs/forecast-ceiling-results.md`: the full-vs-proxy IC table, the ranking-fidelity note, and the **fork verdict** — (a) if full-budget peak IC clears the proxy peak by more than the MDE, the proxy was understating skill: all proxy-based conclusions must be revisited at budget, and the structural experiments run at full budget; (b) if it plateaus within the MDE, the proxy ranking is trustworthy and the ceiling is structural. Close with the **GATE decision**: which experiments the follow-up plan (E3-E6) should run, at which budget, in priority order, given E0+E1+E2.

- [ ] **Step 5: Commit**

```bash
git add docs/forecast-ceiling-results.md
git commit -m "Record E2 full-budget reality check and GATE decision"
```

---

### Task 7: Changelog + suite-wide verification

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add the `[Unreleased]` entry**

Under `[Unreleased]`, add (matching the file's existing style):

```markdown
### Added
- `ophir.ceiling`: offline helpers for the forecasting-ceiling investigation —
  run IC-trajectory summary (peak / best-checkpoint / final `val_rank_ic`),
  multi-seed aggregation + minimum-detectable-effect, and naive cross-sectional
  baselines reusing the production rank-IC math.
```

- [ ] **Step 2: Run the full gate**

Run: `uv run pytest && uv run ruff check . && uv run ruff format --check . && uv run mypy src/ophir`
Expected: all green. (`pytest` must stay offline + CPU-only; the new `test_ceiling.py` uses only synthetic fixtures.)

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "Note ceiling investigation helpers in changelog"
```

---

## Deferred to a follow-up plan: structural fan-out (E3-E6)

The spec's structural experiments are intentionally **not** in this plan — their parameters are outputs of this gate:

- **E3 — horizon/response structure:** sweep `response_size ∈ {1 or 5, 20, 90}`.
- **E4 — cross-sectional information:** add universe-relative features (needs date-aligning the per-ticker streaming pipeline).
- **E5 — feature-content battery:** `open`/overnight-gap; drop calendar zero-padding; winsorize-not-drop the 0.75 spike days; fix high/low split-adjust.
- **E6 — target/metric alignment:** audit what the trading core consumes; test more-predictable targets.

Write that plan after Task 6's GATE decision, using the MDE (Task 3) as the win threshold and the budget regime (Task 6) as the run budget. The GATE may also redirect — e.g. straight to E6 if E1 shows the model barely beats naive.

## Self-Review

- **Spec coverage:** E0 → Tasks 1-3; E1 → Tasks 4-5; E2 → Task 6; measurement-discipline foundation (peak-IC ruler + MDE) → Tasks 1-3; out-of-scope leakage → untouched as intended; E3-E6 → explicitly deferred with hand-off conditions. The "audit what `signals.py`/`forecast.py` consume" (E6) lives in the follow-up plan, consistent with the deferral.
- **Placeholder scan:** no TBD/TODO; the only fill-ins are the literal `version_N` run IDs that *don't exist until their runs complete* (Task 6 Steps 2-3), each flagged inline. All code steps carry complete code.
- **Type consistency:** `run_ic_summary`/`RunICSummary`, `aggregate_ic`/`ICAggregate`, `mde_for_group_difference`, `lagged_target_signal`→`(signal, valid)`, `cross_sectional_ic(..., valid=)`, `dedupe_rows`, `shuffle_within_day(generator=)` are used with identical names/signatures across tasks and match `evaluate.rank_ic`/`dedupe_by_ticker_date` real signatures (verified in source).
