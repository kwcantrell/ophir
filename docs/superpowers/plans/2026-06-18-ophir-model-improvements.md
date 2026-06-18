# Ophir Model Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the unanimous, no-retrain-required correctness and measurement improvements from the three-reviewer panel (quant-finance + ML + devil's advocate) as TDD'd, independently-reviewable tasks.

**Architecture:** Each task is a red-green-refactor cycle landing one pure, CPU-testable change to the loss, the data pipeline, or the evaluation/metrics core — mirroring the existing `tests/test_evaluate.py` / `tests/conftest.py` style (deterministic, seeded, no network, no CUDA). Phase 1 is the executable plan. Phase 2 is a backlog of architecture experiments that change model behavior and therefore need a training run to validate efficacy — each is specced as its own plan, not executed here.

**Tech Stack:** Python 3.10 floor / 3.12 lint target, PyTorch + Lightning, pandas/numpy, Typer, pytest, `uv` for env, ruff + mypy (strict on new files).

## Global Constraints

- Live code is **only** under `src/ophir/`. Never touch top-level `ophir/`, `oldcode/`, or `old_*.py`.
- **New files stay strict** for ruff `ANN` and mypy — do not add them to the suppression blocks in `pyproject.toml` (lines 73–77, 111–120). New test/helper code must type-check under `uv run mypy src/ophir`.
- **Do not regress the forecast-masking contract:** the whole response block is replaced with the learned `mask_token` before the transformer (`models.py:378` `_apply_response_mask`). No task may feed contemporaneous response-block features. Pinned by `tests/test_models_leakage.py` — it must stay green.
- **Do not regress the by-date train/val split** with embargo ≥ `seq_len` (`train.py:125`).
- Tests must be CPU-safe and network-free (the metric/loss/feature cores already are; keep it that way). The CUDA forward (`accumulate_targets`) is not unit-tested — test the pure helpers it feeds.
- Run `uv run ruff check . && uv run ruff format --check . && uv run mypy src/ophir && uv run pytest` clean before each commit. Add a CHANGELOG entry per task in the existing format.

---

## File Structure

| File | Role in this plan |
| --- | --- |
| `src/ophir/model_data.py` | Fix the `pca_projection` double-mean bug (Task 1). |
| `src/ophir/training_models.py` | Remove the broken `use_cache` setter (Task 2); per-channel data-derived smooth-L1 beta (Task 4). |
| `src/ophir/ticker.py` | Separate the warm-up-NaN sentinel from the padding sentinel + emit a validity flag (Task 3). |
| `src/ophir/models.py` | Add a pure output-activation step enforcing target positivity (Task 5); pool the prefix for `stock_embeddings` (Task 8). |
| `src/ophir/evaluate.py` | Cross-sectional rank-IC metric + identity threading (Task 6); persistence/EWMA baseline skill (Task 7). |
| `tests/test_model_data.py` | New — Task 1. |
| `tests/test_training_models.py` | New — Tasks 2, 4. |
| `tests/test_ticker_features.py` | Existing — extend for Task 3. |
| `tests/test_models_output.py` | New — Tasks 5, 8. |
| `tests/test_evaluate.py` | Existing — extend for Tasks 6, 7. |

---

## Task 1: Fix `pca_projection` double-mean bug

The model already pools to `(B, emb_dim)` (`models.py:452` `response_embeddings.mean(dim=1)`), so the extra `.mean(1)` in `pca_projection` (`model_data.py:141`) collapses the embedding axis to a scalar per stock — the UI PCA then runs on a degenerate `(B, 1)` input. Drop the redundant mean.

**Files:**
- Modify: `src/ophir/model_data.py:140-144`
- Test: `tests/test_model_data.py` (create)

**Interfaces:**
- Consumes: `OHLCMulitClassPredictorInput(feature_input, response_size, trade_occured, targets)` with `stock_embeddings` set to a `(B, emb_dim)` tensor.
- Produces: `pca_projection() -> np.ndarray` of shape `(B, 3)` computed from the full `(B, emb_dim)` embeddings.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model_data.py
"""Tests for OHLCMulitClassPredictorInput projection/reconstruction helpers."""

import numpy as np
import torch

from ophir.model_data import OHLCMulitClassPredictorInput


def _make_input(stock_embeddings: torch.Tensor) -> OHLCMulitClassPredictorInput:
    b = stock_embeddings.shape[0]
    obj = OHLCMulitClassPredictorInput(
        feature_input=torch.zeros(b, 4, 13),
        response_size=torch.tensor(2),
        trade_occured=torch.ones(b, 4, dtype=torch.bool),
        targets=torch.zeros(b, 4, 3),
    )
    obj.stock_embeddings = stock_embeddings
    return obj


def test_pca_projection_uses_full_embedding_dimension():
    torch.manual_seed(0)
    # 6 stocks, 8-d embeddings already pooled by the model to (B, emb_dim).
    embeddings = torch.randn(6, 8)
    obj = _make_input(embeddings)

    projected = obj.pca_projection()

    assert projected.shape == (6, 3)
    # A non-degenerate projection has spread on every component; the old
    # double-mean collapsed embeddings to (B, 1), making components 2 and 3 zero.
    assert np.all(projected.std(axis=0) > 1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model_data.py::test_pca_projection_uses_full_embedding_dimension -v`
Expected: FAIL — the current `.mean(1)` makes `stock_embeddings` `(B,)`, so `pca_lowrank(q=3)` errors or the projection has zero-variance trailing components.

- [ ] **Step 3: Write minimal implementation**

In `src/ophir/model_data.py`, drop the redundant mean:

```python
    def pca_projection(self) -> np.ndarray[Any, Any]:
        """Project the stock embeddings onto their top 3 principal components.

        Returns
        -------
        numpy.ndarray
            A ``(B, 3)`` array of the per-stock embeddings projected into the
            leading 3-D PCA subspace.
        """
        assert self.stock_embeddings is not None
        with torch.no_grad():
            stock_embeddings = self.stock_embeddings
            _u, _s, v = torch.pca_lowrank(stock_embeddings, q=3)
            transformed_data = torch.matmul(stock_embeddings, v)
        return transformed_data.cpu().numpy()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model_data.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_model_data.py src/ophir/model_data.py CHANGELOG.md
git commit -m "Fix pca_projection collapsing the stock-embedding dimension"
```

---

## Task 2: Remove the broken `use_cache` setter

The `use_cache` setter (`training_models.py:400-407`) assigns to `self.ohlc_predictor.ohlc_percentage_change` / `volume_percentage_change`, which do not exist on `OHLCMulitClassPredictor` — calling it raises `AttributeError`. It is dead, broken API surface; the model has no percentage-change cache. Remove both the getter and setter (and the `_use_cache` backing field if unused elsewhere).

**Files:**
- Modify: `src/ophir/training_models.py` (remove the `use_cache` property/setter; grep and remove `_use_cache` if unreferenced)
- Test: `tests/test_training_models.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `LightningOHLCPredictor` no longer exposes a `use_cache` attribute.

- [ ] **Step 1: Confirm `_use_cache` usage before deleting**

Run: `rg "_use_cache|use_cache" src/ophir`
Expected: only the property, setter, and (possibly) an `__init__` assignment in `training_models.py`. If any other module reads `use_cache`, STOP and report — the deletion is no longer safe and this task needs re-scoping.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_training_models.py
"""Tests for the LightningOHLCPredictor wrapper."""

from ophir.training_models import LightningOHLCPredictor


def test_use_cache_attribute_is_removed():
    assert not hasattr(LightningOHLCPredictor, "use_cache")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_training_models.py::test_use_cache_attribute_is_removed -v`
Expected: FAIL — `use_cache` is still a class-level property.

- [ ] **Step 4: Remove the property/setter (and backing field if unused)**

Delete the `@property def use_cache` and `@use_cache.setter` block (`training_models.py:395-407`). If Step 1 showed `self._use_cache = ...` in `__init__` with no other readers, delete that assignment too.

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run pytest tests/test_training_models.py -v && uv run mypy src/ophir`
Expected: PASS, mypy clean.

- [ ] **Step 6: Commit**

```bash
git add tests/test_training_models.py src/ophir/training_models.py CHANGELOG.md
git commit -m "Remove dead, raising use_cache property from the Lightning wrapper"
```

---

## Task 3: Separate the warm-up-NaN sentinel from the padding sentinel (F11)

`extract_features` (`ticker.py:359-391`) zero-fills both calendar-padding rows **and** the first `window-1` rows where rolling std/mean are undefined, using the same `pad_value = 0.0`. A literal `0.0` volatility is an impossible, out-of-distribution value injected systematically at every ticker's history start, on the *prefix* (input) side where no mask protects it. Emit an explicit `feature_valid` flag so warm-up rows are distinguishable from genuine zeros, and the downstream window builder can drop them.

**Files:**
- Modify: `src/ophir/ticker.py:381-393` (`extract_features` padding/fill block)
- Test: `tests/test_ticker_features.py` (extend)

**Interfaces:**
- Consumes: a raw OHLCV `DataFrame` (the existing `ohlcv_df` fixture).
- Produces: `extract_features(df)` returns the existing feature columns + `trade_occured` **plus** a new boolean `feature_valid` column that is `False` for the first 59 rows (the 60-window warm-up) and for calendar-padding rows, `True` otherwise. The 13 feature column ordering is unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ticker_features.py (add)
def test_extract_features_flags_rolling_warmup_as_invalid(ohlcv_df):
    out = extract_features(ohlcv_df)

    assert "feature_valid" in out.columns
    assert out["feature_valid"].dtype == np.bool_
    # The 60-day rolling features are undefined for the first 59 trading rows;
    # those must be flagged invalid rather than silently zero-filled.
    valid = out["feature_valid"]
    trading = out["trade_occured"]
    first_valid_trading_day = valid[trading].idxmax()
    assert valid[trading].iloc[:59].sum() == 0
    assert valid.loc[first_valid_trading_day]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ticker_features.py::test_extract_features_flags_rolling_warmup_as_invalid -v`
Expected: FAIL — no `feature_valid` column exists.

- [ ] **Step 3: Write minimal implementation**

Replace the fill block at `ticker.py:381-393`:

```python
    if df.empty:
        return df

    # A row is "valid" only once every rolling feature is defined: the largest
    # rolling window is 60, so the first 59 rows are warm-up. Capture this on the
    # trading-day frame, before calendar padding adds non-trading rows.
    largest_window = 60
    valid = pd.Series(True, index=df.index)
    valid.iloc[: largest_window - 1] = False

    calendar = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df_pad = df.reindex(index=calendar)

    df_pad = add_feature("trade_occured", df_pad["close"].notna(), df_pad)
    # Padding rows are never valid; warm-up rows were flagged above. Anything not
    # explicitly valid (padding gaps) defaults to False.
    df_pad["feature_valid"] = valid.reindex(calendar, fill_value=False)

    pad_value = 0.0
    for col in feature_cols:
        df_pad[col] = df_pad[col].fillna(pad_value)

    return df_pad[[*feature_cols, "trade_occured", "feature_valid"]]
```

Note: `feature_cols` already excludes `trade_occured`; the final selection now appends both flags. Verify the existing `test_extract_features_column_layout` test — update its expectation to include the trailing `feature_valid` column.

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_ticker_features.py -v`
Expected: PASS (including the updated column-layout test).

- [ ] **Step 5: Refactor — consume the flag in the window builder**

In the `StockStreamer` window logic (`ticker.py:396+`), a window whose response block or prefix contains `feature_valid == False` rows still feeds zero-filled warm-up features to the model. Minimal safe change: require a window's start to be at or after the first `feature_valid` row so no warm-up zeros enter any window. Add a test asserting the first emitted window's start index is ≥ the first valid row, then implement. (If `StockStreamer` already starts past row 59 via `seq_len`, assert that and note no change is needed.)

- [ ] **Step 6: Commit**

```bash
git add tests/test_ticker_features.py src/ophir/ticker.py CHANGELOG.md
git commit -m "Flag rolling-feature warm-up rows instead of zero-filling them"
```

---

## Task 4: Per-channel data-derived smooth-L1 beta (A2)

`compute_loss` (`training_models.py:217-253`) hardcodes `beta=0.01` for `r_close` and `beta=0.02` for `upside`/`downside`. Daily log-return std is ~0.01-0.02, so `beta=0.01` keeps `r_close` almost entirely in the quadratic (MSE-like) regime whose minimizer shrinks toward zero, and the fixed `1 / 0.5 / 0.5` channel weights are arbitrary across the channels' different scales. Derive each channel's `beta` from a robust scale of its target so Huber's transition sits at the noise scale. (Per-channel `beta`, not target standardization — avoids touching the `get_ohlcs` / `to_pandas` reconstruction.)

**Files:**
- Modify: `src/ophir/training_models.py` (add a pure `robust_scale` helper; use it in `compute_loss`)
- Test: `tests/test_training_models.py` (extend)

**Interfaces:**
- Consumes: the masked target tensor for a channel.
- Produces: `robust_scale(x: torch.Tensor, floor: float = 1e-4) -> float` — `max(floor, 1.4826 * median(|x - median(x)|))` (MAD → Gaussian-equivalent std), `floor` when `x` is empty.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_training_models.py (add)
import torch

from ophir.training_models import robust_scale


def test_robust_scale_recovers_gaussian_std():
    torch.manual_seed(0)
    x = torch.randn(10_000) * 0.02
    # MAD-based scale of a Gaussian approximates its std (~0.02 here).
    assert abs(robust_scale(x) - 0.02) < 0.002


def test_robust_scale_floors_on_empty_or_constant():
    assert robust_scale(torch.tensor([])) == 1e-4
    assert robust_scale(torch.zeros(100)) == 1e-4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_training_models.py -k robust_scale -v`
Expected: FAIL — `robust_scale` does not exist.

- [ ] **Step 3: Write minimal implementation**

Add to `training_models.py` (module level):

```python
def robust_scale(x: torch.Tensor, floor: float = 1e-4) -> float:
    """Gaussian-equivalent scale of ``x`` via the median absolute deviation.

    ``1.4826 * MAD`` matches the standard deviation for normal data but is
    robust to the fat tails of daily returns. Returns ``floor`` for an empty or
    zero-variance input so it is always safe as a smooth-L1 ``beta``.
    """
    if x.numel() == 0:
        return floor
    median = x.median()
    mad = (x - median).abs().median()
    return max(floor, float(1.4826 * mad.item()))
```

In `compute_loss`, replace the hardcoded betas with per-channel scales computed from the masked targets, e.g.:

```python
        masked_close_target = target_r_close[mask]
        close_loss = F.smooth_l1_loss(
            predicted_r_close,
            target_r_close,
            beta=robust_scale(masked_close_target),
            reduction="none",
        )
```

and likewise for `upside`/`downside` with their masked targets. Keep the `1 / 0.5 / 0.5` combine for now (a separate task can revisit weights once Task 6's rank-IC exists to measure them).

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_training_models.py tests/test_training_loss_decay.py -v`
Expected: PASS — the loss-decay test must stay green (this task changes `beta`, not the decay weighting).

- [ ] **Step 5: Commit**

```bash
git add tests/test_training_models.py src/ophir/training_models.py CHANGELOG.md
git commit -m "Derive smooth-L1 beta per channel from a robust target scale"
```

---

## Task 5: Enforce target positivity on the upside/downside heads (A1, positivity half)

`out_ff = nn.Linear(emb_dim, 3)` (`models.py:376`) is unconstrained, but `upside = log(high/close) >= 0` and `downside = log(close/low) >= 0` (`ticker.py:378-379`), and reconstruction does `predicted_upside.exp()` / `(-predicted_downside).exp()` (`model_data.py:172-178`) — a negative predicted log-magnitude yields `high < close` (or `low > close`), an impossible candle. Pass the two magnitude channels through `softplus` so they are non-negative by construction. (The full quantile/NLL head is deferred to Phase 2 — its benefit is unmeasurable until Task 6's rank-IC exists.)

**Files:**
- Modify: `src/ophir/models.py` (add a pure `apply_output_activations`; call it in `forward` after `out_ff`)
- Test: `tests/test_models_output.py` (create — CPU only, does not touch flex-attention)

**Interfaces:**
- Consumes: raw `out_ff` output of shape `(B, R, 3)` with channel order `r_close=0, upside=1, downside=2`.
- Produces: `apply_output_activations(raw: torch.Tensor) -> torch.Tensor` — same shape, `r_close` passed through unchanged, `upside`/`downside` passed through `softplus`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models_output.py
"""Tests for the pure output-activation step (CPU, no flex-attention)."""

import torch

from ophir.models import apply_output_activations


def test_upside_downside_are_non_negative():
    raw = torch.tensor([[[-0.5, -3.0, -2.0], [0.5, -0.1, 4.0]]])  # (1, 2, 3)
    out = apply_output_activations(raw)

    # r_close (channel 0) is left signed/unchanged.
    torch.testing.assert_close(out[..., 0], raw[..., 0])
    # upside (1) and downside (2) are forced non-negative.
    assert torch.all(out[..., 1] >= 0)
    assert torch.all(out[..., 2] >= 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models_output.py::test_upside_downside_are_non_negative -v`
Expected: FAIL — `apply_output_activations` does not exist.

- [ ] **Step 3: Write minimal implementation**

Add to `models.py`:

```python
def apply_output_activations(raw: torch.Tensor) -> torch.Tensor:
    """Constrain the upside/downside channels to be non-negative.

    ``upside``/``downside`` are log-magnitudes (``log(high/close)`` and
    ``log(close/low)``), both ``>= 0`` by construction, while ``r_close`` is a
    signed return. Softplus keeps the magnitude channels in their valid range so
    the ``.exp()`` reconstruction can never invert the candle.
    """
    r_close = raw[..., 0:1]
    upside = F.softplus(raw[..., 1:2])
    downside = F.softplus(raw[..., 2:3])
    return torch.cat([r_close, upside, downside], dim=-1)
```

Wire it into `forward` (`models.py:451`):

```python
        input.model_output = apply_output_activations(
            cast("torch.Tensor", self.out_ff(response_embeddings))
        )
```

Ensure `torch.nn.functional as F` is imported in `models.py` (add if absent).

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_models_output.py tests/test_models_leakage.py -v`
Expected: PASS — leakage test must stay green (this changes outputs, not the masking).

- [ ] **Step 5: Note for execution**

This changes the model's output distribution, so existing checkpoints will not be optimal under it — efficacy must be confirmed by a retrain (tracked in Phase 2's measurement loop). The *correctness* guarantee (non-negativity) is what this task lands and tests.

- [ ] **Step 6: Commit**

```bash
git add tests/test_models_output.py src/ophir/models.py CHANGELOG.md
git commit -m "Constrain upside/downside forecast heads to be non-negative"
```

---

## Task 6: Cross-sectional rank-IC metric + identity threading (F1)

Every metric pools `(pred, target)` across all tickers and days (`evaluate.py:166-171`), destroying the per-day cross-section — nothing measures whether the model *ranks names* on a given day, the only quantity that makes a cross-sectional return model tradeable. Add a pure daily rank-IC metric, then thread `(date, ticker)` identity through the eval collection so it can be fed.

**Files:**
- Modify: `src/ophir/evaluate.py` (add `rank_ic`; extend `accumulate_targets`/`evaluate_model` to carry identity)
- Test: `tests/test_evaluate.py` (extend)

**Interfaces:**
- Consumes: per-prediction `r_close` predictions/targets plus a `date` key and `ticker` key per prediction.
- Produces: `rank_ic(pred, target, dates) -> dict[str, float]` returning `{"ic_mean", "ic_std", "ic_ir", "n_days"}` — the mean / std / information-ratio of the daily cross-sectional Spearman correlation between `pred` and `target`, grouping rows by `dates`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py (add)
import torch

from ophir.evaluate import rank_ic


def test_rank_ic_perfect_daily_ranking():
    # Two days, three names each; predictions rank names identically to targets.
    dates = ["d1", "d1", "d1", "d2", "d2", "d2"]
    target = torch.tensor([0.03, 0.01, -0.02, -0.01, 0.04, 0.00])
    pred = torch.tensor([3.0, 2.0, 1.0, 1.0, 3.0, 2.0])  # same within-day order

    result = rank_ic(pred, target, dates)

    assert result["n_days"] == 2
    assert abs(result["ic_mean"] - 1.0) < 1e-6
    assert result["ic_std"] < 1e-6


def test_rank_ic_inverted_ranking_is_negative():
    dates = ["d1", "d1", "d1"]
    target = torch.tensor([0.03, 0.01, -0.02])
    pred = torch.tensor([-3.0, -2.0, -1.0])  # exactly reversed

    result = rank_ic(pred, target, dates)

    assert abs(result["ic_mean"] + 1.0) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluate.py -k rank_ic -v`
Expected: FAIL — `rank_ic` does not exist.

- [ ] **Step 3: Write minimal implementation**

Add to `evaluate.py`:

```python
def _spearman(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Spearman rank correlation of two 1-D tensors (nan if < 2 points)."""
    if pred.numel() < 2:
        return float("nan")
    pr = pred.argsort().argsort().float()
    tr = target.argsort().argsort().float()
    pr = pr - pr.mean()
    tr = tr - tr.mean()
    denom = pr.norm() * tr.norm()
    if denom == 0:
        return float("nan")
    return float((pr @ tr / denom).item())


def rank_ic(
    pred: torch.Tensor, target: torch.Tensor, dates: list[str]
) -> dict[str, float]:
    """Daily cross-sectional rank information coefficient.

    Groups predictions/targets by ``dates`` and computes the Spearman rank
    correlation within each day, then summarises across days. ``ic_ir`` is the
    information ratio ``ic_mean / ic_std`` (annualisable by the caller).
    """
    by_day: dict[str, list[int]] = {}
    for i, d in enumerate(dates):
        by_day.setdefault(d, []).append(i)
    ics: list[float] = []
    for idx in by_day.values():
        sel = torch.tensor(idx)
        ic = _spearman(pred[sel], target[sel])
        if ic == ic:  # skip nan days
            ics.append(ic)
    if not ics:
        return {"ic_mean": float("nan"), "ic_std": float("nan"), "ic_ir": float("nan"), "n_days": 0.0}
    t = torch.tensor(ics)
    mean = float(t.mean().item())
    std = float(t.std(unbiased=False).item())
    return {
        "ic_mean": mean,
        "ic_std": std,
        "ic_ir": mean / std if std > 0 else float("nan"),
        "n_days": float(len(ics)),
    }
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: PASS.

- [ ] **Step 5: Thread identity through collection**

`accumulate_targets` currently discards date/ticker. Extend it (and `build_dataloader`/`StockHanlder` via `return_stock_id=True` + `return_date=True`) so each pooled `r_close` prediction carries a parallel list of `(date, ticker)` keys, then add an `r_close` rank-IC line to `evaluate_model`'s report. Write a focused test first: a fake model + a two-day, two-ticker loader asserting the report contains an `ic_mean`. This step is **plumbing-heavy** (the dataset must emit a ticker id — `StockHanlder(return_stock_id=False)` at `train.py:144` and `extract_model_data`'s `return_date`); if the ticker id cannot be threaded without a larger refactor, land Steps 1-4 (the pure metric) and split the plumbing into its own follow-up task rather than forcing it here.

- [ ] **Step 6: Commit**

```bash
git add tests/test_evaluate.py src/ophir/evaluate.py CHANGELOG.md
git commit -m "Add cross-sectional rank-IC metric for r_close forecasts"
```

---

## Task 7: Persistence/EWMA baseline skill for all three channels (A6)

`skill_score` exists only for `r_close` (`evaluate.py:200-202`); `upside`/`downside` report raw MAE/RMSE with no baseline, so those numbers are uninterpretable in isolation. Add a baseline-relative skill score so every channel is judged against a trivial forecaster.

**Files:**
- Modify: `src/ophir/evaluate.py` (add `skill_score_vs_baseline`; use it per channel)
- Test: `tests/test_evaluate.py` (extend)

**Interfaces:**
- Consumes: `pred`, `target`, and a `baseline` tensor (the trailing persistence/EWMA forecast) of equal length.
- Produces: `skill_score_vs_baseline(pred, target, baseline) -> float` — `1 - rmse(pred, target) / rmse(baseline, target)`; `nan` on empty input or zero baseline RMSE.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py (add)
from ophir.evaluate import skill_score_vs_baseline


def test_skill_vs_baseline_positive_when_model_beats_baseline():
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    pred = target.clone()  # perfect
    baseline = torch.tensor([0.0, 0.0, 0.0, 0.0])  # naive

    assert abs(skill_score_vs_baseline(pred, target, baseline) - 1.0) < 1e-6


def test_skill_vs_baseline_is_nan_when_baseline_is_perfect():
    target = torch.tensor([1.0, 2.0, 3.0])
    assert skill_score_vs_baseline(target, target, target) != skill_score_vs_baseline(
        target, target, target
    )  # nan != nan
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluate.py -k skill_vs_baseline -v`
Expected: FAIL — `skill_score_vs_baseline` does not exist.

- [ ] **Step 3: Write minimal implementation**

Add to `evaluate.py`:

```python
def skill_score_vs_baseline(
    pred: torch.Tensor, target: torch.Tensor, baseline: torch.Tensor
) -> float:
    """RMSE skill score of ``pred`` against an arbitrary ``baseline`` forecast.

    ``1 - rmse(pred) / rmse(baseline)``: positive means the model beats the
    baseline, ``0`` ties it, negative is worse. ``nan`` for empty input or a
    zero-RMSE baseline. Lets the non-negative ``upside``/``downside`` channels be
    scored against a persistence/EWMA forecast instead of having no reference.
    """
    if pred.numel() == 0:
        return float("nan")
    rmse_model = (pred - target).pow(2).mean().sqrt().item()
    rmse_baseline = (baseline - target).pow(2).mean().sqrt().item()
    if rmse_baseline == 0:
        return float("nan")
    return float(1.0 - rmse_model / rmse_baseline)
```

Then, in `evaluate_model`, build a persistence baseline per channel from the prefix (the last observed in-window value carried forward, available via the collected response-block tensors) and report `skill_score_vs_baseline` for `upside`/`downside`. Write the baseline-construction test first.

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_evaluate.py src/ophir/evaluate.py CHANGELOG.md
git commit -m "Add baseline-relative skill score for all forecast channels"
```

---

## Task 8: Pool the prefix for `stock_embeddings` (A11, trivial tweak)

`stock_embeddings = response_embeddings.mean(dim=1)` (`models.py:452`) pools the *masked* forecast block — the positions with no observed features — for the UI's PCA visualization. Pool the *prefix* (the signal-bearing observed history) instead. UI-only; zero forecast-loss impact.

**Files:**
- Modify: `src/ophir/models.py:449-452` (`forward` embedding pooling)
- Test: `tests/test_models_output.py` (extend) — or assert via a small CPU helper if the pooling is extracted.

- [ ] **Step 1: Extract the pooling as a pure, testable helper**

Add to `models.py`:

```python
def pool_prefix_embedding(x: torch.Tensor, response_size: int) -> torch.Tensor:
    """Mean-pool the prefix (observed-history) positions into one vector/example.

    Pools ``x[:, :-response_size]`` — the positions that carry real features —
    rather than the masked forecast block, giving a more grounded per-stock
    embedding for the UI projection.
    """
    return x[:, :-response_size].mean(dim=1)
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_models_output.py (add)
from ophir.models import pool_prefix_embedding


def test_pool_prefix_embedding_ignores_response_block():
    # 1 example, 4 positions, 2-d; prefix=first 2 rows, response=last 2.
    x = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0], [99.0, 99.0]]])
    pooled = pool_prefix_embedding(x, response_size=2)
    torch.testing.assert_close(pooled, torch.tensor([[2.0, 2.0]]))
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_models_output.py -k pool_prefix -v`
Expected: FAIL — `pool_prefix_embedding` does not exist.

- [ ] **Step 4: Wire it into `forward` and verify pass**

Replace `models.py:452` with `input.stock_embeddings = pool_prefix_embedding(x, int(input.response_size))`. (Use the post-encoder `x`, not `response_embeddings`.)

Run: `uv run pytest tests/test_models_output.py tests/test_models_leakage.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_models_output.py src/ophir/models.py CHANGELOG.md
git commit -m "Pool the prefix instead of the masked block for stock embeddings"
```

---

## Phase 1 completion review

After Task 8, run the full gate and request a code review (superpowers:requesting-code-review) before starting Phase 2:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy src/ophir && uv run pytest
```

---

## Phase 2 — Experiment backlog (each needs its own spec + a training run)

These are the unanimous-but-behavior-changing items. They cannot be validated by unit tests alone: efficacy is only visible after a CUDA training run scored with the Phase 1 metrics (rank-IC, baseline skill). **Do not execute them from this plan** — spec each as its own `docs/superpowers/plans/` file via superpowers:writing-plans, gated on a measurable before/after. Listed in recommended order.

### Follow-ups discovered during Phase 1 execution

Unlike the experiment backlog below, these are **deterministic plumbing** (unit-testable, no training run needed). They came out of Phase 1 task reviews — two pure metrics landed but are not yet fed into the report, and one warm-up edge case needs a guard.

- **6b + 7b — wire rank-IC and baseline skill into `evaluate_model`.** Phase 1 landed the pure `rank_ic` (Task 6) and `skill_score_vs_baseline` (Task 7) but split their *wiring* (Task 6 needs per-prediction `(date, ticker)` identity; Task 7 needs the prefix's last observed value). **Now fully specced** in [`2026-06-18-ophir-eval-metrics-wiring.md`](2026-06-18-ophir-eval-metrics-wiring.md) — 3 TDD tasks threading identity as opt-in integer/int64 tensors (no custom `collate_fn`, training path untouched). Execute this **before** backlog items F10 and F4, which depend on the `(date, ticker)` panel it produces.
- **3b — thin-ticker warm-up guard.** The final whole-branch review flagged that a sub-60-trading-bar ticker that still exceeds `seq_len` after `freq="D"` padding yields an all-`False` `feature_valid`; `numpy.argmax` then returns `0` (`ticker.py:452`) and `StockStreamer` emits warm-up-polluted windows from row 0 — silently defeating Task 3's F11 fix for exactly those tickers. Non-regressive (old code zero-filled them too) and narrow. Fix: guard the `argmax` — when `feature_valid.any()` is `False`, set `starts = []` so the streamer emits no windows. Needs its own TDD cycle (a test that a thin ticker yields zero windows). The UI path (`ui.py` `<250`-history skip) is unaffected.

1. **Reproducibility + deterministic validation (A4).** `seed_everything(seed, workers=True)` in `train`/`finetune`; build the val handler with `shuffle=False` (split the single `shuffle=True` at `train.py:153`); seed the dataset RNGs (`ticker.py:516`, `:858`). Unblocks honest checkpoint selection. *Mostly config; the testable slice is "`build_split_handlers` passes `shuffle=False` to the val handler" — could be pulled into Phase 1 if desired.*
2. **Deterministic, non-overlapping validation eval (F10).** Eval with `offset = seq_len` (or dedup to one prediction per `(ticker, date)`); cover the full val period deterministically; block-bootstrap CIs as a follow-up. Depends on A4 and on the 6b identity panel (see Follow-ups above).
3. **Tradeable long-short backtest (F4).** Daily cross-sectional decile long-short from `r_close` predictions; report gross/net (configurable bps) Sharpe, turnover, drawdown, holding-period sweep. Depends on the 6b `(date, ticker)` panel (see Follow-ups above). *The Sharpe/turnover computation is itself unit-testable as a pure function — TDD that core, validate the signal empirically.*
4. **Additive market/sector context (F5, additive half only).** Append a daily market-return feature + sector dummies/embedding. Defer the market-demeaned excess-return *target* redefinition (interacts with the masking contract + reconstruction). Needs a training run to confirm rank-IC lift.
5. **Ablate the redundant learned positional encoding (A3).** Init `self.pe` (`models.py:369`) near zero / downscale and ablate against ALiBi-only; it currently double-counts position with ALiBi and at unit-variance init swamps the 13 projected features. Scale-down, not deletion (deletion invalidates checkpoints). Pure experiment.
6. **Overfitting diagnostics (A10, diagnostic half only).** Expose dropout as a hyperparameter (currently hardcoded `0.1`, `models.py:295`) and log the train/val gap. Gate any DropPath/attention-dropout/EarlyStopping on what the gap shows **and** on A4 (else EarlyStopping fires on val noise).
7. **Batch decorrelation (A7, cheap half only).** Bump `cache_size` (default 8) and add a cross-stock shuffle buffer so a batch isn't dominated by a few stocks' ~75%-overlapping windows. Defer the throughput-risky one-window-per-stock constraint.
8. **Padding-row NaN guard (A9, verify-then-guard).** First write a test constructing a fully-padded query row and confirm flex-attention actually NaNs (it may emit 0); only if it reproduces, add a self-allow fallback + NaN check. Don't refactor attention on a hypothesised failure.
9. **Trading-day calendar migration (F6 + F8).** Index features on the trading-day calendar instead of `freq="D"` calendar padding (`ticker.py:384`); recompute `time_delta` as the gap between consecutive trading bars (F8 is the same root cause). **Highest blast radius:** touches the masking contract, embargo math (`required_gap`, `train.py:125`), reconstruction, and invalidates every checkpoint. Must ship with a `tests/test_models_leakage.py` re-run and a fresh embargo proof. Spec deliberately; not an "Effort M" item.
10. **Point-in-time universe (F3 cheap half + F12).** Document survivorship bias (`get_sp_500_symbols` scrapes *current* constituents — but `use_sp500` defaults `False`, so the default full-parquet universe already sidesteps it); make the liquidity gate trailing/PIT (`ticker.py:654` full-history `volume.mean()`). Defer full PIT-membership infra (high effort, off by default). Low priority.

### Explicitly killed by the panel (do not implement)

- **F2** cumulative-horizon head — premise was an architecture misread (each response position is individually conditioned via its own PE + ALiBi); cumulative return is already a `cumsum` of the existing daily head (`ticker.py:484`), computable in `evaluate.py` with no model change.
- **F7** dividend adjustment — effect ~0.3-0.6% × ~4/year per name, swamped by daily vol; needs a new data feed; won't move rank-IC.
- **A5** split ReZero scalars — unmeasurable micro-architecture preference, no observed symptom, breaks checkpoints.
- **A8** signed-ALiBi + slope unit test — symmetric bias is defensible for the bidirectional prefix; the non-power-of-2 slope branch (`models.py:116-119`) is dead code at the shipped `num_heads=8`.
- **A1 full quantile/NLL head**, **F5 demeaned-target redefinition**, **F9 select-checkpoints-on-Sharpe** — *deferred, not rejected*: revisit once Phase 1's rank-IC/backtest make their benefit measurable.
