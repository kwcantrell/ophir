# Forecast seam (offset-1 inference) + minor cleanups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `ophir.trading.forecast.load_forecasts` to return per-symbol offset-1 forecasts from the IC-best checkpoint, and land three minor cleanups from the prior review.

**Architecture:** A CPU helper `ticker.build_latest_inputs` builds the most-recent `response_size=1` model-input window per requested symbol (unit-testable offline via the `parquet_dir` fixture). `load_forecasts` orchestrates: it keeps the graceful-degrade `{}` contract (no `model_dir` / no checkpoint / no CUDA / no inputs), and on a CUDA host loads the IC-best checkpoint, forwards each window, and emits the raw log-space day-1 channels as `OphirForecast`. The CUDA forward is runtime-only (not unit-tested), exactly as `dashboard.run_evaluation`. Cleanups are independent single-file changes.

**Tech Stack:** Python 3.10+, PyTorch, PyTorch-Lightning, pandas, pytest, uv.

## Global Constraints

- mypy is `strict = True`, targets Python 3.10; run `uv run mypy src/ophir`.
- ruff targets Python 3.12; run `uv run ruff check . && uv run ruff format --check .`.
- pytest runs `filterwarnings = error` — any project-owned warning fails the suite.
- Tests must never touch the network, CUDA, or the package `.ophir/` layout; use `tmp_path` and the seeded fixtures in `tests/conftest.py`.
- NumPy-style docstrings throughout `src/ophir`, matching existing density.
- Imports: `known-first-party = ["ophir"]` (ruff/isort ordering).
- Paper-only; the safety gate (`trading/safety.py`) is non-overridable and untouched — this work only *produces* forecasts.
- `OphirForecast` carries the model's **raw log-space** day-1 channels (no exp/reconstruction in the seam).
- Update the `[Unreleased]` section of `CHANGELOG.md` for notable changes.
- Run commands with `uv run`.

---

### Task 1: `build_latest_inputs` (CPU window builder)

**Files:**
- Modify: `src/ophir/ticker.py` (add helper after `extract_model_data`, which ends at line 815)
- Test: `tests/test_ticker_forecast_inputs.py` (create)

**Interfaces:**
- Consumes: `StockHanlder` (dataclass), `StockStreamer` (`.size`, `.starts`, `__getitem__(i)` → `seq_len`-row window), `extract_model_data(df, response_size, stock_id=None)` — all already in `ticker.py`. `register.get_default_data_days_dir()` (lazy import to avoid a circular import).
- Produces: `build_latest_inputs(symbols: Sequence[str], seq_len: int = 365, base_path: str | None = None) -> dict[str, dict[str, Any]]` — one `extract_model_data` payload per known symbol at `response_size=1`, from that symbol's most-recent window; unknown or too-short symbols are skipped.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ticker_forecast_inputs.py`:

```python
"""Tests for the live-inference window builder."""

from ophir.ticker import build_latest_inputs


def test_build_latest_inputs_known_skips_unknown_and_short(parquet_dir) -> None:
    # parquet_dir (conftest): AAA = full history, BBB = ~5-day span (too short
    # to form a seq_len window), CCC = constant volume. Plus a decoy _logs dir.
    base_path, _ = parquet_dir
    out = build_latest_inputs(["AAA", "BBB", "ZZZ"], seq_len=15, base_path=base_path)

    assert "AAA" in out  # full history -> one most-recent window
    assert "BBB" not in out  # ~5 days < seq_len=15 -> no window
    assert "ZZZ" not in out  # unknown symbol -> skipped, no raise

    inp = out["AAA"]
    assert {"feature_input", "targets", "trade_occured", "response_size"} <= set(inp)
    assert inp["feature_input"].shape[0] == 15
    assert int(inp["response_size"].squeeze()) == 1


def test_build_latest_inputs_empty_for_all_unknown(parquet_dir) -> None:
    base_path, _ = parquet_dir
    assert build_latest_inputs(["NOPE", "ALSONOPE"], seq_len=15, base_path=base_path) == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ticker_forecast_inputs.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_latest_inputs'`.

- [ ] **Step 3: Implement the helper**

In `src/ophir/ticker.py`, after `extract_model_data` (after line 815), add:

```python
def build_latest_inputs(
    symbols: Sequence[str],
    seq_len: int = 365,
    base_path: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the most-recent inference window per symbol at ``response_size=1``.

    For each requested symbol, loads its history from the parquet tree, takes the
    single most-recent ``seq_len``-row feature window, and packages it with
    :func:`extract_model_data` for an offset-1 (day-1) forecast. Symbols absent
    from the tree, or with too little history to form one window, are silently
    skipped — the live forecast seam degrades rather than raising.

    Parameters
    ----------
    symbols : sequence of str
        Ticker symbols to build inference windows for.
    seq_len : int, optional
        Window length the model consumes. Defaults to ``365``.
    base_path : str, optional
        Root of the Hive-partitioned parquet tree. When ``None``, defaults to
        ``register.get_default_data_days_dir()/stocks``.

    Returns
    -------
    dict[str, dict[str, Any]]
        ``{symbol: extract_model_data payload}`` for each symbol that produced a
        window. Empty when no requested symbol is available.
    """
    if base_path is None:
        import os

        from ophir import register

        base_path = os.path.join(register.get_default_data_days_dir(), "stocks")

    handler = StockHanlder(
        seq_len=seq_len,
        base_path=base_path,
        return_stock_id=False,
        return_streamer=True,
    )
    out: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        try:
            streamer = handler[symbol]
        except ValueError:
            continue  # symbol not in the tree
        if not isinstance(streamer, StockStreamer) or streamer.size == 0:
            continue  # too little history to form a window
        window = streamer[int(streamer.starts[-1])]  # most-recent window
        out[symbol] = extract_model_data(window, 1)
    return out
```

Note: `handler[symbol]` raises `ValueError` for an unknown symbol (`self.stocks.index(symbol)`); `streamer.size == 0` covers too-short history. Direct per-symbol indexing is used instead of `keep_stocks` to avoid that method's stdout `print`. If `Sequence` / `Any` are not already imported in `ticker.py`, add them (`from collections.abc import Sequence`, `from typing import Any`) in the existing import block, keeping isort order.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ticker_forecast_inputs.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Typecheck, lint, commit**

Run: `uv run mypy src/ophir && uv run ruff check src/ophir/ticker.py tests/test_ticker_forecast_inputs.py && uv run ruff format --check src/ophir/ticker.py tests/test_ticker_forecast_inputs.py`
Expected: clean.

```bash
git add src/ophir/ticker.py tests/test_ticker_forecast_inputs.py
git commit -m "feat: add build_latest_inputs for offset-1 live inference windows"
```

---

### Task 2: `load_forecasts` offset-1 inference

**Files:**
- Modify: `src/ophir/trading/forecast.py` (rewrite `load_forecasts`, lines 30-37)
- Modify: `CHANGELOG.md` (`[Unreleased]`)
- Test: `tests/test_trading_forecast.py` (existing — keep its 3 tests passing; add one)

**Interfaces:**
- Consumes: `build_latest_inputs` (Task 1); `register.load_base_model_ckpt(strict=False, time_version=False)` → `LightningOHLCPredictor`; `OphirForecast` (same module); `model(batch_dict)` returns an `OHLCMulitClassPredictorInput` exposing `predicted_r_close` / `predicted_upside` / `predicted_downside` of shape `(batch, response_size)`.
- Produces: `load_forecasts(symbols, model_dir) -> dict[str, OphirForecast]` — offset-1 forecasts on a CUDA host; `{}` otherwise.

- [ ] **Step 1: Write the failing test**

The existing three tests in `tests/test_trading_forecast.py` already assert the degrade contract (no model_dir → `{}`, missing ckpt → `{}`, present ckpt → `dict` without raising). On a CUDA-less test host the present-ckpt case still returns `{}`. Add an explicit guard test:

```python
def test_present_checkpoint_returns_empty_without_cuda(tmp_path: Path) -> None:
    # On a CUDA-less host (CI/test invariant) a present checkpoint must still
    # degrade to {} without attempting the flex-attention forward.
    import torch

    if torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA present: this test asserts the no-CUDA degrade path")
    (tmp_path / "base.ckpt").write_bytes(b"")
    assert load_forecasts(["AAPL"], tmp_path) == {}
```

- [ ] **Step 2: Run test to verify it passes-as-guard / fails meaningfully**

Run: `uv run pytest tests/test_trading_forecast.py -v`
Expected: the new test PASSES already only if the current stub returns `{}` — it does (stub returns `{}`). To make this a true TDD failure first, implement Step 3 in a way that would break the no-CUDA path if the guard were missing; the guard test then locks it in. Run the full file after Step 3 and confirm all 4 pass.

- [ ] **Step 3: Rewrite `load_forecasts`**

Replace `load_forecasts` in `src/ophir/trading/forecast.py` (lines 30-37) with:

```python
def load_forecasts(
    symbols: Sequence[str], model_dir: str | Path | None
) -> dict[str, OphirForecast]:
    """Return per-symbol offset-1 ophir forecasts, or ``{}`` if unavailable.

    Builds the most-recent inference window per symbol, loads the IC-best base
    checkpoint, and runs the model's day-1 (``response_size=1``) forward,
    returning the raw log-space ``r_close`` / ``upside`` / ``downside`` channels.
    Never raises: returns ``{}`` when the model directory has no checkpoint, when
    CUDA is unavailable (the flex-attention forward is CUDA-only), or when no
    requested symbol has data — so the trading loop degrades to non-ophir signals.

    Parameters
    ----------
    symbols : sequence of str
        Ticker symbols to forecast.
    model_dir : str or pathlib.Path or None
        Directory holding the base checkpoint. ``None`` (or a directory with no
        ``*.ckpt``) yields ``{}``.

    Returns
    -------
    dict[str, OphirForecast]
        One forecast per symbol that produced a prediction; empty when forecasts
        are unavailable.
    """
    if model_dir is None or not _has_checkpoint(model_dir):
        return {}

    import torch

    if not torch.cuda.is_available():
        return {}

    from ophir import register
    from ophir.ticker import build_latest_inputs

    inputs = build_latest_inputs(symbols)
    if not inputs:
        return {}

    model = register.load_base_model_ckpt(strict=False, time_version=False)
    model = model.cuda().eval()
    out: dict[str, OphirForecast] = {}
    with torch.no_grad():
        for symbol, payload in inputs.items():
            batch = {key: value.unsqueeze(0) for key, value in payload.items()}
            prediction = model(batch)
            out[symbol] = OphirForecast(
                symbol=symbol,
                r_close=float(prediction.predicted_r_close[0, 0]),
                upside=float(prediction.predicted_upside[0, 0]),
                downside=float(prediction.predicted_downside[0, 0]),
            )
    return out
```

Add `from collections.abc import Sequence` and `from pathlib import Path` to the imports if not present (both `Sequence` and `Path` are already imported in the current file — verify and keep isort order). The CUDA forward (everything after the `torch.cuda.is_available()` guard) is runtime-only and not unit-tested, mirroring `dashboard.run_evaluation`.

- [ ] **Step 4: Run the forecast tests**

Run: `uv run pytest tests/test_trading_forecast.py -v`
Expected: PASS (4 tests — the 3 existing + the new no-CUDA guard).

- [ ] **Step 5: Update CHANGELOG**

In `CHANGELOG.md` under `[Unreleased] ### Added`, add:

```markdown
- `ophir.trading.forecast.load_forecasts` now returns per-symbol offset-1
  forecasts (raw log-space `r_close`/`upside`/`downside`) from the IC-best
  checkpoint when CUDA and data are available; still degrades to `{}` otherwise.
- `ophir.ticker.build_latest_inputs` builds the most-recent `response_size=1`
  inference window per symbol.
```

- [ ] **Step 6: Typecheck, lint, commit**

Run: `uv run mypy src/ophir && uv run ruff check src/ophir/trading/forecast.py tests/test_trading_forecast.py`
Expected: clean.

```bash
git add src/ophir/trading/forecast.py tests/test_trading_forecast.py CHANGELOG.md
git commit -m "feat: wire load_forecasts to offset-1 inference from the IC-best checkpoint"
```

---

### Task 3: Cleanup 1 — checkpoint filename reflects monitored metric

**Files:**
- Modify: `src/ophir/register.py` (`_best_checkpoint_callback`, the `filename=` argument)
- Modify: `CHANGELOG.md` (`[Unreleased]`)
- Test: `tests/test_register.py` (existing — extend)

**Interfaces:**
- Consumes: nothing new.
- Produces: `_best_checkpoint_callback(file_name, monitor_near_ic)` now sets `filename = file_name + "best-{epoch:02d}-{val_rank_ic_near:.5f}"` when `monitor_near_ic` else `file_name + EPOCH_MODIFIER` (the `val_loss` form).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_register.py`:

```python
def test_best_checkpoint_filename_embeds_near_ic_when_flagged() -> None:
    cb = _best_checkpoint_callback("model", monitor_near_ic=True)
    assert "val_rank_ic_near" in cb.filename
    assert "val_loss" not in cb.filename


def test_best_checkpoint_filename_embeds_val_loss_by_default() -> None:
    cb = _best_checkpoint_callback("model", monitor_near_ic=False)
    assert "val_loss" in cb.filename
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_register.py -k filename -v`
Expected: FAIL — the `monitor_near_ic=True` filename currently still contains `val_loss` (and not `val_rank_ic_near`).

- [ ] **Step 3: Implement the filename branch**

In `src/ophir/register.py`, inside `_best_checkpoint_callback`, replace the single `filename=file_name + EPOCH_MODIFIER` usage so the suffix tracks the monitor. Current body builds `monitor, mode` then returns the `ModelCheckpoint`. Change it to:

```python
    monitor, mode = ("val_rank_ic_near", "max") if monitor_near_ic else ("val_loss", "min")
    suffix = "best-{epoch:02d}-{val_rank_ic_near:.5f}" if monitor_near_ic else EPOCH_MODIFIER
    return ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        dirpath=MODEL_DIR,
        filename=file_name + suffix,
        save_top_k=1,
        save_on_train_epoch_end=False,
    )
```

`EPOCH_MODIFIER` (`"best-{epoch:02d}-{val_loss:.5f}"`) is unchanged and still used for the default branch and by `load_base_model_ckpt`/`_latest_base_ckpt`, which strip on the pre-`{` prefix (`file_name.split("{")[0]`), so checkpoint discovery is unaffected by either suffix.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_register.py -v`
Expected: PASS (all — the 2 prior monitor/mode tests plus the 2 new filename tests).

- [ ] **Step 5: Update CHANGELOG, lint, commit**

In `CHANGELOG.md` under `[Unreleased] ### Changed`, add:

```markdown
- Best-checkpoint filename now embeds the monitored metric
  (`val_rank_ic_near` when selecting on near-IC, else `val_loss`) instead of
  always labelling it `val_loss`.
```

Run: `uv run mypy src/ophir && uv run ruff check src/ophir/register.py tests/test_register.py`
Expected: clean.

```bash
git add src/ophir/register.py tests/test_register.py CHANGELOG.md
git commit -m "fix: best-checkpoint filename reflects the monitored metric"
```

---

### Task 4: Cleanup 2 — de-duplicate `torch.cat` in `on_validation_epoch_end`

**Files:**
- Modify: `src/ophir/training_models.py` (`on_validation_epoch_end`, lines 478-515)
- Test: none new — the existing `tests/test_training_models.py` validation-epoch tests cover behavior; values must not change.

**Interfaces:**
- Consumes/Produces: no public surface change. Internal refactor only.

- [ ] **Step 1: Confirm the current behavior is green (baseline)**

Run: `uv run pytest tests/test_training_models.py -q`
Expected: PASS (baseline before refactor).

- [ ] **Step 2: Hoist the concatenations**

In `src/ophir/training_models.py`, `on_validation_epoch_end`, the `if preds:` block currently re-`cat`s buffers in three places (the `val_rank_ic` call, the `val_rank_ic_near` call, and the `if self.log_offset_ic and preds:` block). Compute each `cat` once at the top of `if preds:` and reuse:

```python
        preds = self._val_ic_buffers["pred"]
        if preds:
            cat_pred = torch.cat(preds)
            cat_target = torch.cat(self._val_ic_buffers["target"])
            cat_ids = torch.cat(self._val_ic_buffers["ids"])
            cat_dates = torch.cat(self._val_ic_buffers["dates"])
            cat_offsets = torch.cat(self._val_ic_buffers["offsets"])

            ic = val_rank_ic(cat_pred, cat_target, cat_ids, cat_dates)
            self.log("val_rank_ic", ic, prog_bar=False, on_epoch=True, logger=True)

            near_ic = val_rank_ic_near(
                cat_pred, cat_target, cat_ids, cat_dates, cat_offsets, self.near_offset_k
            )
            self.log("val_rank_ic_near", near_ic, prog_bar=False, on_epoch=True, logger=True)

            if self.log_offset_ic:
                from .evaluate import rank_ic_by_offset

                offset_ics = rank_ic_by_offset(
                    cat_pred, cat_target, cat_ids, cat_dates, cat_offsets, _OFFSET_BUCKETS
                )
                for key, off_ic in offset_ics.items():
                    self.log(f"val_rank_ic_{key}", off_ic, on_epoch=True, logger=True)
        for buf in self._val_ic_buffers.values():
            buf.clear()
```

Preserve the surrounding `log_rezero_gates` block (above) and the final buffer-clear loop (which must still run unconditionally, including when `preds` is empty). Keep variable name `ic` for the offset loop distinct (`off_ic`) so it does not shadow the `val_rank_ic` result.

- [ ] **Step 3: Run tests to verify behavior unchanged**

Run: `uv run pytest tests/test_training_models.py -q`
Expected: PASS (same as the baseline; logged values unchanged).

- [ ] **Step 4: Typecheck, lint, commit**

Run: `uv run mypy src/ophir && uv run ruff check src/ophir/training_models.py`
Expected: clean.

```bash
git add src/ophir/training_models.py
git commit -m "refactor: hoist val buffer cats once in on_validation_epoch_end"
```

---

### Task 5: Cleanup 3 — remove import-time `print()`s in `register.py`

**Files:**
- Modify: `src/ophir/register.py` (the module-level `print(...)` statements emitting `current file path` / `current dir path`)
- Test: none new — verified by the full suite staying green and lint.

**Interfaces:**
- Consumes/Produces: no surface change.

- [ ] **Step 1: Locate the prints**

Run: `uv run python -c "import ophir.register"` and confirm it prints the `current file path` / `current dir path` lines to stdout (the noise being removed).

- [ ] **Step 2: Remove the statements**

In `src/ophir/register.py`, delete the two module-level `print(...)` calls that emit `current file path ...` and `current dir path ...`. Do not change the path computations themselves (the variables they print from are still used to compute `OPHIR_DIR` / `MODEL_DIR`); only remove the `print` lines.

- [ ] **Step 3: Verify the import is now quiet and the suite is green**

Run: `uv run python -c "import ophir.register"` (expected: no stdout output) then `uv run pytest -q`
Expected: import prints nothing; full suite passes.

- [ ] **Step 4: Lint, commit**

Run: `uv run ruff check src/ophir/register.py && uv run mypy src/ophir`
Expected: clean.

```bash
git add src/ophir/register.py
git commit -m "chore: drop import-time print() noise in register"
```

---

## Self-Review

**Spec coverage:**
- C1 `build_latest_inputs` (CPU, injectable base_path) → Task 1. ✓
- C2 `load_forecasts` orchestration (guards, IC-best ckpt, raw log channels, degrade) → Task 2. ✓
- Cleanup 1 (filename metric) → Task 3. ✓
- Cleanup 2 (dedupe cat) → Task 4. ✓
- Cleanup 3 (remove prints) → Task 5. ✓
- Testing (build_latest_inputs via parquet_dir; load_forecasts guard paths; filename test; cleanup-2 behavior unchanged) → Tasks 1-4. ✓
- CUDA forward not unit-tested → stated in Task 2. ✓
- Consumer wiring deferred → no task, matches spec. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code. ✓

**Type consistency:** `build_latest_inputs(symbols, seq_len=365, base_path=None) -> dict[str, dict[str, Any]]` defined in Task 1, called with defaults in Task 2. `load_forecasts(symbols, model_dir) -> dict[str, OphirForecast]` consistent. `_best_checkpoint_callback(file_name, monitor_near_ic)` matches the existing signature from the prior branch; only the `filename` suffix changes (Task 3). `predicted_r_close/upside/downside[0, 0]` indexing matches the `(batch, response_size=1)` shape. ✓
