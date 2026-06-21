# Forecast-horizon diagnostic (E3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the tooling and instrumentation to decide whether ophir's cross-sectional rank-IC ceiling is a near-free operating-point problem or a real architectural one — via a free signal-decay curve (Step A) and per-offset model-IC instrumentation (Step B).

**Architecture:** Two pure helpers extend the existing `ophir.ceiling` (Step A) and `ophir.evaluate` (Step B) modules, reusing the production rank-IC math. One gated `--log-offset-ic` flag (mirroring `--log-rezero-gates`) adds per-offset IC logging to the validation loop without touching the default path. The actual measurements are two operational tasks driving those tools over the val set and one 10k training run.

**Tech Stack:** Python 3.10+ (mypy floor), PyTorch, pandas, numpy, pytest, run via `uv`. Pure helpers are CPU/offline; the Step-B run uses the RTX 3090.

## Global Constraints

- **mypy `strict = True`, targets Python 3.10**; ruff targets 3.12. New code fully typed.
- **pytest runs `filterwarnings = error`** — tests must not emit project-owned warnings.
- **Tests stay offline + CPU-only**: synthetic `tmp_path`/tensor fixtures; no network, CUDA, or `.ophir/` access. The `--log-offset-ic` gate must leave the default validation path (and the existing offline suite) unchanged when off.
- **Reuse production IC math** (`ophir.evaluate.rank_ic`, `dedupe_by_ticker_date`); never reimplement Spearman/day-grouping.
- **NumPy-style docstrings** throughout, matching existing density.
- Imports: `known-first-party = ["ophir"]` (ruff/isort).
- Update `[Unreleased]` in `CHANGELOG.md`.
- **Leave alone:** `main`'s unpushed commits, the uncommitted `.claude/settings.json`, and the modified `docs/rezero-init-sweep-runbook.md`. Every commit stages ONLY the files it creates/edits — never `git add -A`/`.`/`-u`.
- Offset/lead convention: response-position offset `h` (1-based, first response day = 1) equals forecast lead `h`, so Step-A leads and Step-B offsets are directly comparable.

## File Structure

- `src/ophir/ceiling.py` *(modify)* — add `signal_decay_curve`, `pooled_baseline_ceiling` (Step A). Pure.
- `src/ophir/evaluate.py` *(modify)* — add `rank_ic_by_offset` (Step B). Pure.
- `src/ophir/training_models.py` *(modify)* — offset accumulation + gated `log_offset_ic` per-offset logging.
- `src/ophir/train.py` *(modify)* — thread `log_offset_ic` through `run_training` and the `train` CLI.
- `tests/test_ceiling.py` *(modify)* — Step-A tests.
- `tests/test_evaluate.py` *(modify, or create if absent)* — `rank_ic_by_offset` tests.
- `tests/test_training_models.py` *(modify)* — flag/buffer offline tests.
- `CHANGELOG.md` *(modify)* — `[Unreleased]` entry.
- `docs/forecast-ceiling-results.md` *(modify)* — E3 results + verdict (operational tasks).

Existing references (read, not modified): `model_data.py` (`OHLCMulitClassPredictorInput`), `train.py` (`build_split_handlers`, `build_dataloader`), `register.py` (`get_default_data_days_dir`).

---

### Task 1: Step A helpers — `signal_decay_curve` + `pooled_baseline_ceiling`

**Files:**
- Modify: `src/ophir/ceiling.py`
- Test: `tests/test_ceiling.py`

**Interfaces:**
- Consumes: `lagged_target_signal`, `cross_sectional_ic` (already in `ceiling.py`).
- Produces:
  - `signal_decay_curve(target, ids, dates, leads, *, kind="reversal") -> dict[int, float]` — `{lead: ic_mean}` of the lagged-return signal (negated when `kind="reversal"`) vs current return, per lead. Raises `ValueError` on unknown `kind`.
  - `pooled_baseline_ceiling(decay, response_size) -> float` — mean of finite `decay` values whose lead is in `1..response_size`; `nan` if none.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_ceiling.py
from ophir.ceiling import pooled_baseline_ceiling, signal_decay_curve


def test_signal_decay_curve_perfect_reversal() -> None:
    # 3 tickers over 3 days; each day's returns are the rank-reversal of the
    # prior day, so a 1-lead reversal signal perfectly predicts the cross-section.
    #            d1(t1,t2,t3)  d2(reversed) d3(reversed again)
    target = torch.tensor([1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0])
    ids = torch.tensor([1, 2, 3, 1, 2, 3, 1, 2, 3])
    dates = torch.tensor([1, 1, 1, 2, 2, 2, 3, 3, 3])
    curve = signal_decay_curve(target, ids, dates, leads=(1,), kind="reversal")
    assert curve == pytest.approx({1: 1.0})
    mom = signal_decay_curve(target, ids, dates, leads=(1,), kind="momentum")
    assert mom[1] == pytest.approx(-1.0)


def test_signal_decay_curve_rejects_bad_kind() -> None:
    target = torch.tensor([1.0, 2.0]); ids = torch.tensor([1, 2]); dates = torch.tensor([1, 1])
    with pytest.raises(ValueError):
        signal_decay_curve(target, ids, dates, leads=(1,), kind="trend")


def test_pooled_baseline_ceiling_means_in_range() -> None:
    decay = {1: 0.05, 5: 0.03, 90: 0.0}
    assert pooled_baseline_ceiling(decay, response_size=10) == pytest.approx(0.04)
    assert pooled_baseline_ceiling(decay, response_size=90) == pytest.approx((0.05 + 0.03 + 0.0) / 3)


def test_pooled_baseline_ceiling_empty_is_nan() -> None:
    assert math.isnan(pooled_baseline_ceiling({90: 0.1}, response_size=10))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ceiling.py -k "decay or pooled" -v`
Expected: FAIL — `ImportError: cannot import name 'signal_decay_curve'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/ophir/ceiling.py (after the existing baseline helpers)
def signal_decay_curve(
    target: torch.Tensor,
    ids: torch.Tensor,
    dates: torch.Tensor,
    leads: Sequence[int],
    *,
    kind: str = "reversal",
) -> dict[int, float]:
    """Cross-sectional IC of a lagged-return signal at each forecast lead.

    For each lead ``L`` in ``leads``, uses that ticker's return ``L`` observations
    earlier as the signal (negated when ``kind="reversal"``) and correlates it
    against the current return cross-sectionally via the production rank-IC. The
    result is the achievable signal at each forecast lead — the ceiling a model
    predicting ``L`` days ahead could reach from price history alone.

    Parameters
    ----------
    target, ids, dates : torch.Tensor
        Equal-length 1-D tensors of return, ticker id, and integer date ordinal,
        one row per (ticker, date).
    leads : sequence of int
        Forecast leads (in trading-day observations) to evaluate.
    kind : {"reversal", "momentum"}, optional
        ``"reversal"`` negates the lagged signal; ``"momentum"`` uses it as-is.

    Returns
    -------
    dict[int, float]
        ``{lead: ic_mean}`` for each requested lead.
    """
    if kind not in ("reversal", "momentum"):
        raise ValueError(f"kind must be 'reversal' or 'momentum', got {kind!r}")
    sign = -1.0 if kind == "reversal" else 1.0
    curve: dict[int, float] = {}
    for lead in leads:
        sig, valid = lagged_target_signal(target, ids, dates, lag=lead)
        ic = cross_sectional_ic(sign * sig, target, ids, dates, valid=valid)
        curve[int(lead)] = ic["ic_mean"]
    return curve


def pooled_baseline_ceiling(decay: dict[int, float], response_size: int) -> float:
    """Matched-horizon-mix ceiling: mean IC over sampled leads in ``1..response_size``.

    Approximates the horizon mix that ``val_rank_ic`` pools (offsets
    1..``response_size``) by averaging the decay curve over its sampled leads in
    that range. This is the fair comparand for a model whose pooled metric mixes
    those leads — not an exact replica of the metric's per-(ticker, date) dedup.

    Returns
    -------
    float
        Mean of finite ``decay`` values with lead in ``1..response_size``; ``nan``
        if none qualify.
    """
    vals = [v for lead, v in decay.items() if 1 <= lead <= response_size and v == v]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ceiling.py -v`
Expected: PASS (all, including prior `ceiling` tests).

- [ ] **Step 5: Lint, format, type-check**

Run: `uv run ruff check src/ophir/ceiling.py tests/test_ceiling.py && uv run ruff format src/ophir/ceiling.py tests/test_ceiling.py && uv run mypy src/ophir`
Expected: all pass. (`Sequence` is already imported in `ceiling.py` under `TYPE_CHECKING`; if mypy reports it unused/missing, fix the import position — do not loosen strictness.)

- [ ] **Step 6: Commit**

```bash
git add src/ophir/ceiling.py tests/test_ceiling.py
git commit -m "Add signal_decay_curve and pooled_baseline_ceiling (E3 Step A)"
```

---

### Task 2: Step B helper — `rank_ic_by_offset`

**Files:**
- Modify: `src/ophir/evaluate.py`
- Test: `tests/test_evaluate.py` (create if it does not exist)

**Interfaces:**
- Consumes: `dedupe_by_ticker_date`, `rank_ic` (same module).
- Produces: `rank_ic_by_offset(pred, target, ids, dates, offsets, buckets) -> dict[str, float]` — for each `h` in `buckets`, the cross-sectional `ic_mean` over rows with `offsets == h` (deduped per ticker-date within the bucket), keyed `"h{h}"`; `nan` for an empty/insufficient bucket.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py  (add this test; create the file with these imports if absent)
import math

import pytest
import torch

from ophir.evaluate import rank_ic_by_offset


def test_rank_ic_by_offset_buckets_independently() -> None:
    # One day, 3 tickers. offset 1 rows rank perfectly with target; offset 2 rows
    # are perfectly anti-ranked; offset 5 has no rows.
    pred = torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 3.0, 3.0, 2.0, 1.0])
    ids = torch.tensor([1, 2, 3, 1, 2, 3])
    dates = torch.tensor([1, 1, 1, 1, 1, 1])
    offsets = torch.tensor([1, 1, 1, 2, 2, 2])
    out = rank_ic_by_offset(pred, target, ids, dates, offsets, buckets=(1, 2, 5))
    assert out["h1"] == pytest.approx(1.0)
    assert out["h2"] == pytest.approx(-1.0)
    assert math.isnan(out["h5"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: FAIL — `ImportError: cannot import name 'rank_ic_by_offset'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/ophir/evaluate.py, immediately after rank_ic / dedupe_by_ticker_date.
# Ensure `from collections.abc import Sequence` is imported at the top of the file
# (add it in the correct stdlib import group if absent).
def rank_ic_by_offset(
    pred: torch.Tensor,
    target: torch.Tensor,
    ids: torch.Tensor,
    dates: torch.Tensor,
    offsets: torch.Tensor,
    buckets: Sequence[int],
) -> dict[str, float]:
    """Daily cross-sectional rank-IC resolved by response-position offset.

    Splits predictions by forecast offset (1-based response-position lead) and
    computes each offset's pooled-day rank-IC independently, reusing
    :func:`dedupe_by_ticker_date` and :func:`rank_ic`. A single validation pass can
    then reveal whether skill concentrates at near horizons.

    Parameters
    ----------
    pred, target, ids, dates : torch.Tensor
        Equal-length 1-D tensors, as accumulated for :func:`rank_ic`.
    offsets : torch.Tensor
        Same-length integer tensor: each row's response-position offset (1-based).
    buckets : sequence of int
        Offsets to report. For each ``h`` only rows with ``offsets == h`` are used.

    Returns
    -------
    dict[str, float]
        ``{"h{offset}": ic_mean}`` per bucket; ``nan`` for a bucket with no rows or
        no day having at least two observations.
    """
    out: dict[str, float] = {}
    for h in buckets:
        sel = offsets == h
        if not bool(sel.any()):
            out[f"h{int(h)}"] = float("nan")
            continue
        dp, dt, dd = dedupe_by_ticker_date(pred[sel], target[sel], ids[sel], dates[sel])
        out[f"h{int(h)}"] = rank_ic(dp, dt, dd)["ic_mean"]
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: PASS.

- [ ] **Step 5: Lint, format, type-check**

Run: `uv run ruff check src/ophir/evaluate.py tests/test_evaluate.py && uv run ruff format src/ophir/evaluate.py tests/test_evaluate.py && uv run mypy src/ophir`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/ophir/evaluate.py tests/test_evaluate.py
git commit -m "Add rank_ic_by_offset for per-horizon IC decomposition (E3 Step B)"
```

---

### Task 3: Gated per-offset IC instrumentation + `--log-offset-ic` flag

**Files:**
- Modify: `src/ophir/training_models.py`
- Modify: `src/ophir/train.py`
- Test: `tests/test_training_models.py`

**Interfaces:**
- Consumes: `rank_ic_by_offset` (Task 2).
- Produces: `LightningOHLCPredictor(..., log_offset_ic: bool = False)` storing `self.log_offset_ic`; a `"offsets"` key in `_val_ic_buffers`; when the flag is set, validation logs `val_rank_ic_h{1,2,5,10,20,40,90}`. `run_training(..., log_offset_ic=False)` and `train(..., log_offset_ic=False)` (CLI `--log-offset-ic`) thread it through.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_training_models.py — match the construction args an existing
# test in this file already uses for LightningOHLCPredictor; the kwargs below are a
# valid minimal config (emb_dim multiple of 4 and divisible by num_heads).
from ophir.training_models import LightningOHLCPredictor


def test_log_offset_ic_flag_stored_and_buffer_present() -> None:
    model = LightningOHLCPredictor(emb_dim=8, num_layers=1, num_heads=2, log_offset_ic=True)
    assert model.log_offset_ic is True
    assert "offsets" in model._val_ic_buffers


def test_log_offset_ic_defaults_off() -> None:
    model = LightningOHLCPredictor(emb_dim=8, num_layers=1, num_heads=2)
    assert model.log_offset_ic is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_training_models.py -k log_offset_ic -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'log_offset_ic'`.

- [ ] **Step 3: Add the constructor flag, module constant, and buffer key**

In `src/ophir/training_models.py`:

3a. Near the top of the module (with other module-level constants), add:

```python
_OFFSET_BUCKETS = (1, 2, 5, 10, 20, 40, 90)
```

3b. In `LightningOHLCPredictor.__init__`, add the parameter right after
`log_rezero_gates: bool = False,`:

```python
        log_offset_ic: bool = False,
```

3c. In the `__init__` body, immediately after `self.log_rezero_gates = log_rezero_gates`:

```python
        self.log_offset_ic = log_offset_ic
```

(Leave the existing `self.save_hyperparameters()` call where it is — it captures the
new arg automatically.)

3d. Add `"offsets"` to the `_val_ic_buffers` dict literal:

```python
        self._val_ic_buffers: dict[str, list[torch.Tensor]] = {
            "pred": [],
            "target": [],
            "ids": [],
            "dates": [],
            "offsets": [],
        }
```

- [ ] **Step 4: Run the flag test to verify it passes**

Run: `uv run pytest tests/test_training_models.py -k log_offset_ic -v`
Expected: PASS (both).

- [ ] **Step 5: Accumulate offsets in `validation_step` and log per-offset IC at epoch end**

5a. In `validation_step`, inside the identity block, right after
`ids_br = model_output.stock_id.view(-1, 1).expand(-1, rs)`, build the offset grid:

```python
            offsets = (
                torch.arange(1, rs + 1, device=mask.device)
                .unsqueeze(0)
                .expand(mask.shape[0], rs)
            )
```

and, right after the existing `self._val_ic_buffers["dates"].append(...)` line, add:

```python
            self._val_ic_buffers["offsets"].append(offsets[mask].reshape(-1).cpu())
```

5b. In `on_validation_epoch_end`, after the existing `if preds:` block that logs
`val_rank_ic` and BEFORE the `for buf in self._val_ic_buffers.values(): buf.clear()`
loop, add:

```python
        if self.log_offset_ic and preds:
            from .evaluate import rank_ic_by_offset

            offset_ics = rank_ic_by_offset(
                torch.cat(self._val_ic_buffers["pred"]),
                torch.cat(self._val_ic_buffers["target"]),
                torch.cat(self._val_ic_buffers["ids"]),
                torch.cat(self._val_ic_buffers["dates"]),
                torch.cat(self._val_ic_buffers["offsets"]),
                _OFFSET_BUCKETS,
            )
            for key, ic in offset_ics.items():
                self.log(f"val_rank_ic_{key}", ic, on_epoch=True, logger=True)
```

- [ ] **Step 6: Thread the flag through `train.py`**

In `src/ophir/train.py`, mirror `log_rezero_gates` in all four spots:

6a. `run_training` signature — after its `log_rezero_gates: bool = False,`:
```python
    log_offset_ic: bool = False,
```
6b. `run_training` model construction — after its `log_rezero_gates=log_rezero_gates,`:
```python
        log_offset_ic=log_offset_ic,
```
6c. `train` CLI signature — after its `log_rezero_gates: bool = False,`:
```python
    log_offset_ic: bool = False,
```
6d. `train` CLI call to `run_training` — after its `log_rezero_gates=log_rezero_gates,`:
```python
        log_offset_ic=log_offset_ic,
```

- [ ] **Step 7: Run tests + full type/lint gate**

Run: `uv run pytest tests/test_training_models.py -v && uv run ruff check src/ophir/training_models.py src/ophir/train.py tests/test_training_models.py && uv run ruff format src/ophir/training_models.py src/ophir/train.py tests/test_training_models.py && uv run mypy src/ophir`
Expected: all pass. The default-path tests (flag off) must still pass unchanged.

- [ ] **Step 8: Commit**

```bash
git add src/ophir/training_models.py src/ophir/train.py tests/test_training_models.py
git commit -m "Add gated --log-offset-ic per-horizon validation IC instrumentation"
```

---

### Task 4: Changelog + full-suite verification

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add the `[Unreleased]` entry**

Under `[Unreleased] → ### Added` (matching the file's existing style; append to the
existing `### Added` list if one is already present from prior work):

```markdown
- E3 forecast-horizon diagnostic: `ophir.ceiling.signal_decay_curve` /
  `pooled_baseline_ceiling` (reversal IC vs forecast lead + matched-horizon
  ceiling), `ophir.evaluate.rank_ic_by_offset` (per-horizon IC decomposition), and
  a gated `ophir train --log-offset-ic` flag that logs `val_rank_ic_h{N}`.
```

- [ ] **Step 2: Run the full gate**

Run: `uv run pytest && uv run ruff check . && uv run ruff format --check . && uv run mypy src/ophir`
Expected: all green; the suite stays offline + CPU-only.

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "Note E3 forecast-horizon diagnostic helpers in changelog"
```

---

### Task 5: Step A — signal-decay curve + matched-horizon ceiling (operational, CPU)

**Files:**
- Modify: `docs/forecast-ceiling-results.md`

**Interfaces:**
- Consumes: `signal_decay_curve`, `pooled_baseline_ceiling` (Task 1); `dedupe_rows`, `lagged_target_signal` (existing); `build_split_handlers`/`build_dataloader` (`ophir.train`); `OHLCMulitClassPredictorInput` (`ophir.model_data`).
- Produces: the decay curve, 1-day and 90-day ceilings vs the model's 0.027 peak, recorded in the results log. No package code.

> Operational (reads real val data on CPU; no model, no CUDA). Not a pytest test. Reuses the proven E1 harvest. Run via `uv run`.

- [ ] **Step 1: Harvest the val cross-section and compute the decay curve**

```bash
uv run python - <<'PY'
import torch
from ophir import register
from ophir.train import build_split_handlers, build_dataloader
from ophir.model_data import OHLCMulitClassPredictorInput
from ophir.ceiling import (dedupe_rows, signal_decay_curve, pooled_baseline_ceiling)

base = f"{register.get_default_data_days_dir()}/stocks"
_, val_handler = build_split_handlers(
    base_path=base, seq_len=365, offset=90, min_volume=1000.0,
    train_min_year=None, train_max_year=2023, val_min_year=2024, val_max_year=None,
    use_sp500=False, use_quality_allowlist=False, clean_rows=False, max_abs_r_close=0.75,
)
val_dl = build_dataloader(val_handler, 90, 32, 0, 8, return_identity=True)

tgt, ids, dts = [], [], []
for batch in val_dl:
    batch["response_size"] = batch["response_size"][0].squeeze()
    inp = OHLCMulitClassPredictorInput(**batch)
    rs = int(inp.response_size)
    mask = inp.trade_occured[:, -rs:]
    tgt.append(inp.target_r_close[mask].reshape(-1))
    ids.append(inp.stock_id.view(-1, 1).expand(-1, rs)[mask].reshape(-1))
    dts.append(inp.date_ordinal[:, -rs:][mask].reshape(-1))
target, ids, dates = dedupe_rows(torch.cat(tgt), torch.cat(ids), torch.cat(dts))
print(f"{target.numel()} rows, {len(torch.unique(dates))} days")

leads = (1, 2, 3, 5, 10, 20, 40, 90)
curve = signal_decay_curve(target, ids, dates, leads=leads, kind="reversal")
for L in leads:
    print(f"reversal IC @ lead {L:>2}: {curve[L]:+.4f}")
print(f"1-day ceiling:           {curve[1]:+.4f}")
print(f"90-day pooled ceiling:   {pooled_baseline_ceiling(curve, 90):+.4f}")
print("model baseline peak IC = 0.0271 ; MDE = 0.0069")
PY
```

Expected: a decay curve (reversal IC by lead), the 1-day ceiling (~0.05), and the matched-horizon pooled ceiling. Record all in `docs/forecast-ceiling-results.md`.

- [ ] **Step 2: Write the E3 Step-A section**

Append an `## E3 — horizon diagnostic` section to `docs/forecast-ceiling-results.md`
with: the decay-curve table; the 1-day vs 90-day-pooled ceilings beside the model's
0.027 peak; and the **Step-A fork** — does the pooled ceiling fall below 0.027 (model
already beats matched-horizon naive; horizon confound) or sit near ~0.05 (signal
broadly available)?

- [ ] **Step 3: Commit**

```bash
git add docs/forecast-ceiling-results.md
git commit -m "Record E3 Step A: signal-decay curve and matched-horizon ceiling"
```

---

### Task 6: Step B — per-offset model IC at peak + E3 verdict (operational, GPU)

**Files:**
- Modify: `docs/forecast-ceiling-results.md`

**Interfaces:**
- Consumes: the `--log-offset-ic` flag (Task 3); `ophir.ceiling.run_ic_summary` (existing); the Step-A ceiling (Task 5).
- Produces: per-offset model IC at the peak pooled-IC step, overlaid on the Step-A ceiling, and the E3 verdict (world 1 vs world 2 + the path-preserve-vs-collapse decision). No package code.

> Operational GPU run (one 10k training, the diagnostic regime). Run via `uv run` on the 3090.

- [ ] **Step 1: Train one 10k model with per-offset IC logging**

```bash
uv run ophir train --emb-dim 128 --num-heads 8 --num-layers 6 \
  --max-steps 10000 --seed 0 --val-identity --log-offset-ic
```

Expected: run completes; note the new `version_N` (newest dir under
`src/ophir/.ophir/model/csv-logger/`). Confirm its `metrics.csv` header contains
`val_rank_ic` and `val_rank_ic_h1 … val_rank_ic_h90` columns.

- [ ] **Step 2: Read per-offset IC at the peak pooled-IC step**

```bash
uv run python - <<'PY'
import pandas as pd
from pathlib import Path
from ophir.ceiling import run_ic_summary
base = Path("src/ophir/.ophir/model/csv-logger")
V = ___   # the version_N from Step 1
csv = base / f"version_{V}" / "metrics.csv"
peak_step = run_ic_summary(csv).peak_step
df = pd.read_csv(csv)
row = df[df["step"] == peak_step].iloc[0]
print(f"peak pooled val_rank_ic={run_ic_summary(csv).peak_ic:.4f} @ step {peak_step}")
for h in (1, 2, 5, 10, 20, 40, 90):
    col = f"val_rank_ic_h{h}"
    print(f"  model IC @ offset {h:>2}: {row[col]:+.4f}" if col in df.columns else f"  {col} MISSING")
PY
```

Expected: the model's IC at each horizon offset, at its peak pooled-IC step. Fill in
`V` before running.

- [ ] **Step 3: Overlay, decide, and write the E3 verdict**

Append to the E3 section of `docs/forecast-ceiling-results.md`: a table putting
**model IC @ offset h** beside **Step-A ceiling @ lead h** for h ∈ {1,2,5,10,20,40,90},
then the **verdict**:
- Model IC at h=1 approaches the ceiling and decays with offset like it → **world 1
  (diluted-but-captured)**; recommend the operating-point fix (operate short /
  IC-checkpoint), likely **collapse to short horizon**.
- Model IC stays ~flat near 0.027 even at h=1 while the ceiling is ~0.05 → **world 2
  (not-captured)**; recommend an **architectural fix** (per-day legitimate
  conditioning), and **path-preservation becomes live**.

Close with the explicit **path-preserve-vs-collapse decision** and the hand-off to
the next (fix) spec.

- [ ] **Step 4: Commit**

```bash
git add docs/forecast-ceiling-results.md
git commit -m "Record E3 Step B: per-offset model IC and horizon verdict"
```

---

## Self-Review

- **Spec coverage:** Step A → Tasks 1 + 5; Step B helper → Task 2; validation instrumentation + flag → Task 3; changelog/gate → Task 4; Step-B run + verdict + deferred path decision → Task 6. Decision criterion (world 1 vs world 2) → Task 6 Step 3. Matched-horizon-ceiling-as-central-job → Task 1 `pooled_baseline_ceiling` + Task 5. Out-of-scope items (the fix, IC-checkpoint, retrain sweep) are not tasked, as intended.
- **Placeholder scan:** no TBD/TODO; the only fill-in is `V` = the run's `version_N`, which does not exist until Task 6 Step 1 completes, flagged inline. All code steps carry complete code.
- **Type consistency:** `signal_decay_curve(...) -> dict[int, float]` feeds `pooled_baseline_ceiling(decay: dict[int, float], ...)`; `rank_ic_by_offset(..., buckets) -> dict[str, float]` keyed `h{offset}`, consumed by Task 3's logging as `val_rank_ic_{key}` → `val_rank_ic_h{N}`, matched by Task 6's reader. `_OFFSET_BUCKETS = (1,2,5,10,20,40,90)` is the single source for the logged offsets and the Step-2 reader list. Offset/lead convention (offset h ↔ lead h) is stated in Global Constraints and used consistently in Tasks 5/6.
