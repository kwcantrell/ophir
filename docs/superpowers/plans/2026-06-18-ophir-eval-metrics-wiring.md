# Ophir Eval-Metrics Wiring Implementation Plan (Phase 2a: 6b + 7b)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the two pure metrics landed in Phase 1 — `rank_ic` (cross-sectional rank IC) and `skill_score_vs_baseline` — into `evaluate_model` so the validation report actually shows daily rank-IC for `r_close` and a persistence-baseline skill score for `upside`/`downside`.

**Architecture:** Phase 1 split the *wiring* of these metrics into follow-ups (tasks 6b, 7b) because feeding them needs per-prediction `(date, ticker)` identity (rank-IC) and the prefix's last observed value (baseline), neither of which `accumulate_targets` currently keeps. This plan threads that data through **without touching the shared training collation**: identity travels as integer `stock_id` + int64 `date_ordinal` tensors carried as optional fields on `OHLCMulitClassPredictorInput` (mirroring the existing optional `time` field), so the default `DataLoader` collate stacks them and the training path is byte-for-byte unaffected. The baseline (7b) needs no dataloader change at all — it is built from `output.targets` inside `accumulate_targets`.

**Tech Stack:** Python 3.10 floor / 3.12 lint target, PyTorch + Lightning, pandas/numpy, Typer, pytest, `uv`, ruff + mypy (strict on new files and on `evaluate.py`).

## Global Constraints

- Live code is **only** under `src/ophir/`. Never touch top-level `ophir/`, `oldcode/`, or `old_*.py`.
- **The training path must not change behavior.** Identity threading is **opt-in** (off by default): `StockHandlerDataset`/`build_dataloader` default `return_identity=False`, and `extract_model_data` defaults `stock_id=None`. The training `DataLoader` keeps the default collate — do **not** add a custom `collate_fn`.
- **`OHLCMulitClassPredictorInput(**batch)` is how batches become model inputs** (`training_models.py:129`). It is a `slots=True` dataclass, so every batch dict key MUST be a declared field. New identity fields are added as optional (`= None`), exactly like the existing `time` field — extra non-field keys would raise `TypeError`.
- **All of `src/ophir` is mypy-strict** (the old `ignore_errors` legacy block is gone; `strict = true`, `warn_unused_ignores = true`). Every added line in `evaluate.py`, `model_data.py`, `ticker.py`, and `train.py` must type-check under `uv run mypy src/ophir`. The only ANN exemption is `tests/**`; do not add suppression-block entries. Because `warn_unused_ignores` is on, add a `# type: ignore[...]` only where the error actually fires.
- **Do not regress** the forecast-masking contract (`tests/test_models_leakage*.py`), the by-date split, or any existing test. Baseline before this plan: **152 passing**.
- `rank_ic(pred, target, dates: list[str])` and `skill_score_vs_baseline(pred, target, baseline)` already exist in `evaluate.py` and are tested — **consume them; do not modify their signatures.**
- Run `uv run ruff check . && uv run ruff format --check . && uv run mypy src/ophir && uv run pytest` clean before each commit. Add a CHANGELOG entry per task in the existing format. Use `uv`, not pip.

---

## File Structure

| File | Role |
| --- | --- |
| `src/ophir/evaluate.py` | `prefix_last_observed` helper + `AccumulatedEval` result type; `accumulate_targets` collects baseline (Task 1) then identity (Task 3); `evaluate_model` reports baseline skill (Task 1) and rank-IC (Task 3). |
| `src/ophir/model_data.py` | Optional `stock_id` / `date_ordinal` fields on `OHLCMulitClassPredictorInput` (Task 2). |
| `src/ophir/ticker.py` | `StockStreamer.symbol`; `extract_model_data(stock_id=...)`; `StockHandlerDataset(return_identity=...)` carrying `(stock_id, streamer)` in its cache (Task 2). |
| `src/ophir/train.py` | `build_dataloader(return_identity=False)` passthrough (Task 2). |
| `tests/test_evaluate.py` | Tasks 1, 3 — pure helpers + a small CUDA-free `evaluate_model` wiring test with a fake model. |
| `tests/test_model_data.py` | Task 2 — optional fields default to `None` and round-trip. |
| `tests/test_ticker_datasets.py` | Task 2 — `StockHandlerDataset` identity payload (uses the existing `parquet_dir` fixture). |

**Sequencing:** Task 1 (7b, self-contained) → Task 2 (6b identity plumbing) → Task 3 (6b rank-IC wiring, depends on Tasks 1+2). Tasks 1 and 3 both edit `accumulate_targets`/`evaluate_model`; Task 1 introduces the `AccumulatedEval` result type and Task 3 extends it.

---

## Task 1: Persistence-baseline skill for upside/downside (7b)

`skill_score` exists only for `r_close`; `upside`/`downside` report raw MAE/RMSE with no baseline. Build a persistence baseline (each example's last *observed* prefix value, carried across the horizon) inside `accumulate_targets`, and report `skill_score_vs_baseline` for the two magnitude channels. No dataloader change — the baseline comes from `output.targets`.

**Files:**
- Modify: `src/ophir/evaluate.py` — add `prefix_last_observed`, add `AccumulatedEval` dataclass, populate baselines in `accumulate_targets`, report in `evaluate_model`.
- Test: `tests/test_evaluate.py` (extend).

**Interfaces:**
- Produces: `prefix_last_observed(values: torch.Tensor, trade_occured: torch.Tensor, response_size: int) -> torch.Tensor` — given `values` `(B, S)` for one channel and `trade_occured` `(B, S)` bool, returns `(B,)` = each row's channel value at the **last prefix position (the first `S - response_size` columns) where a trade occurred** (falls back to prefix position 0 if a row has no observed prefix day).
- Produces: `@dataclass AccumulatedEval` with `channels: dict[str, tuple[torch.Tensor, torch.Tensor]]` and `baselines: dict[str, torch.Tensor]`. `accumulate_targets` now returns `AccumulatedEval`; `evaluate_model` consumes `.channels` / `.baselines`.

- [ ] **Step 1: Write the failing test for `prefix_last_observed`**

```python
# tests/test_evaluate.py (add)
from ophir.evaluate import prefix_last_observed


def test_prefix_last_observed_picks_last_traded_prefix_day():
    # B=1, S=5, response_size=2 -> prefix is columns 0..2.
    values = torch.tensor([[10.0, 20.0, 30.0, 99.0, 99.0]])
    trade = torch.tensor([[True, True, False, True, True]])  # col 2 is a pad day
    out = prefix_last_observed(values, trade, response_size=2)
    # Last traded prefix column is 1 (col 2 did not trade), so value 20.0.
    torch.testing.assert_close(out, torch.tensor([20.0]))


def test_prefix_last_observed_falls_back_to_position_zero():
    values = torch.tensor([[5.0, 6.0, 7.0, 8.0]])
    trade = torch.tensor([[False, False, False, True]])  # no traded prefix day (S-rs=2)
    out = prefix_last_observed(values, trade, response_size=2)
    torch.testing.assert_close(out, torch.tensor([5.0]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluate.py -k prefix_last_observed -v`
Expected: FAIL — `prefix_last_observed` does not exist.

- [ ] **Step 3: Implement `prefix_last_observed`**

```python
# src/ophir/evaluate.py (add near the other pure helpers)
def prefix_last_observed(
    values: torch.Tensor, trade_occured: torch.Tensor, response_size: int
) -> torch.Tensor:
    """Each row's channel value at its last traded prefix position.

    The persistence baseline carries the last *observed* (pre-horizon) value
    forward across the forecast block. ``values``/``trade_occured`` are ``(B, S)``;
    only the first ``S - response_size`` columns (the prefix) are considered.
    Rows with no traded prefix day fall back to prefix position 0.
    """
    _b, seq_len = values.shape
    prefix_len = seq_len - response_size
    prefix_trade = trade_occured[:, :prefix_len].to(values.dtype)
    positions = torch.arange(prefix_len, device=values.device, dtype=values.dtype)
    # argmax over trade*position selects the largest column index that traded;
    # all-False rows yield argmax 0 (the fallback).
    last_idx = (prefix_trade * positions).argmax(dim=1)
    rows = torch.arange(values.shape[0], device=values.device)
    return values[:, :prefix_len][rows, last_idx]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_evaluate.py -k prefix_last_observed -v`
Expected: PASS.

- [ ] **Step 5: Introduce `AccumulatedEval` and populate baselines**

Replace the return type of `accumulate_targets`. Add at module scope (after the imports):

```python
from dataclasses import dataclass, field


@dataclass
class AccumulatedEval:
    """Masked predictions/targets and per-channel persistence baselines."""

    channels: dict[str, tuple[torch.Tensor, torch.Tensor]]
    baselines: dict[str, torch.Tensor] = field(default_factory=dict)
```

In `accumulate_targets`, alongside the existing per-channel `pred`/`target` collection, collect a baseline for `upside`/`downside`. The channel-to-index map is `{"upside": OHLCMulitClassPredictorInput.upside_index, "downside": OHLCMulitClassPredictorInput.downside_index}`. For each batch:

```python
            rs = int(output.response_size)
            for name, idx in (("upside", 1), ("downside", 2)):
                channel_values = output.targets[..., idx]  # (B, S)
                base = prefix_last_observed(channel_values, output.trade_occured, rs)  # (B,)
                base = base.unsqueeze(1).expand(-1, rs)  # (B, R)
                baseline_lists[name].append(base[mask].reshape(-1).cpu())
```

where `baseline_lists = {"upside": [], "downside": []}` is initialised next to `collected`, and `mask = output.trade_occured[:, -rs:]` (already computed). Return:

```python
    return AccumulatedEval(
        channels={name: (torch.cat(p), torch.cat(t)) for name, (p, t) in collected.items()},
        baselines={name: torch.cat(b) for name, b in baseline_lists.items()},
    )
```

Update `evaluate_model` to consume the new type and add the baseline skill score:

```python
    acc = accumulate_targets(model, dataloader, max_batches)
    results: dict[str, dict[str, float]] = {}
    for name, (pred, target) in acc.channels.items():
        metrics = target_metrics(pred, target)
        if name == "r_close":
            metrics["directional_accuracy"] = directional_accuracy(pred, target)
            metrics["skill_score"] = skill_score(pred, target)
        if name in acc.baselines:
            metrics["skill_vs_persistence"] = skill_score_vs_baseline(
                pred, target, acc.baselines[name]
            )
        results[name] = metrics
    return results
```

Add `"skill_vs_persistence"` to `_METRIC_ORDER` so it renders in the report table.

- [ ] **Step 6: Write a failing test for the wiring (fake model, no CUDA)**

`accumulate_targets` calls `model.cuda()`, so the wiring test must use a fake that ignores `.cuda()`/`.eval()` and returns a prepared output.

**Typing note (applies to every fake-model call in this file):** `accumulate_targets`/`evaluate_model` are annotated `model: LightningOHLCPredictor` and `dataloader: DataLoader[...]`, but the tests pass a `_FakeModel` and a plain `list` of batches as intentional test doubles. The test file is strict-typed, so add a targeted `# type: ignore[arg-type]` on each `accumulate_targets(...)` / `evaluate_model(...)` call in the tests (inline per-line ignores are fine — the prohibition is only on `pyproject.toml` suppression blocks). Do **not** widen the production signatures to accommodate the test. Add:

```python
# tests/test_evaluate.py (add)
from ophir.evaluate import accumulate_targets
from ophir.model_data import OHLCMulitClassPredictorInput


class _FakeModel:
    """Returns its batch as a populated forward output, ignoring device moves."""

    def cuda(self) -> "_FakeModel":
        return self

    def eval(self) -> "_FakeModel":
        return self

    def __call__(self, batch: dict[str, object]) -> OHLCMulitClassPredictorInput:
        obj = OHLCMulitClassPredictorInput(
            feature_input=batch["feature_input"],
            response_size=batch["response_size"],
            trade_occured=batch["trade_occured"],
            targets=batch["targets"],
        )
        # Perfect predictions so error metrics are trivially checkable.
        obj.model_output = obj.targets.clone()
        return obj


def _toy_batch(response_size: int = 2) -> dict[str, object]:
    # B=1, S=4, 3 channels; prefix cols 0..1, response cols 2..3, all traded.
    targets = torch.tensor([[[0.0, 0.1, 0.2], [0.0, 0.3, 0.4], [0.0, 0.5, 0.6], [0.0, 0.7, 0.8]]])
    return {
        "feature_input": torch.zeros(1, 4, 13),
        "targets": targets,
        "trade_occured": torch.ones(1, 4, dtype=torch.bool),
        "response_size": torch.tensor(response_size),
    }


def test_accumulate_targets_reports_persistence_baseline():
    model = _FakeModel()
    acc = accumulate_targets(model, [_toy_batch()], max_batches=1)

    assert "upside" in acc.baselines
    # Last traded prefix upside value is col 1 -> 0.3, broadcast over the horizon.
    torch.testing.assert_close(acc.baselines["upside"], torch.tensor([0.3, 0.3]))
```

Run: `uv run pytest tests/test_evaluate.py -k persistence -v` → expected FAIL first (baselines absent), then implement Step 5 wiring, then PASS. (If you implemented Step 5 before this test, temporarily assert the old behavior to observe RED, or trust the helper RED in Steps 1-4 and document that the wiring test was added GREEN — prefer writing this test before the Step 5 edit.)

- [ ] **Step 7: Run the gate and commit**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy src/ophir && uv run pytest`
Expected: all green (154+ passing).

```bash
git add src/ophir/evaluate.py tests/test_evaluate.py CHANGELOG.md
git commit -m "Report persistence-baseline skill score for upside/downside"
```

---

## Task 2: Thread integer stock_id + int64 date through the dataset (6b plumbing)

Carry per-window identity so the eval path can later group predictions by `(ticker, date)`. Identity travels as tensors via optional dataclass fields, opt-in, leaving the training path untouched.

**Files:**
- Modify: `src/ophir/model_data.py` — add optional `stock_id` / `date_ordinal` fields.
- Modify: `src/ophir/ticker.py` — `StockStreamer.symbol`; `extract_model_data(stock_id=...)`; `StockHandlerDataset(return_identity=...)`.
- Modify: `src/ophir/train.py` — `build_dataloader(return_identity=False)`.
- Test: `tests/test_model_data.py`, `tests/test_ticker_datasets.py`.

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `extract_model_data(df, response_size, return_date=False, stock_id=None)` includes, when `stock_id is not None`, `"stock_id"` = `torch.tensor(stock_id, dtype=torch.long)` (0-dim → `(B,)` after collate) and `"date_ordinal"` = int64 tensor of shape `(seq_len,)` from `df.index` (`(B, seq_len)` after collate). `StockHandlerDataset(..., return_identity=True)` yields those keys. `OHLCMulitClassPredictorInput` gains `stock_id: torch.Tensor | None = None` and `date_ordinal: torch.Tensor | None = None`.

- [ ] **Step 1: Failing test — optional dataclass fields default to None**

```python
# tests/test_model_data.py (add)
def test_optional_identity_fields_default_to_none():
    obj = OHLCMulitClassPredictorInput(
        feature_input=torch.zeros(1, 4, 13),
        response_size=torch.tensor(2),
        trade_occured=torch.ones(1, 4, dtype=torch.bool),
        targets=torch.zeros(1, 4, 3),
    )
    assert obj.stock_id is None
    assert obj.date_ordinal is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_model_data.py -k identity_fields -v`
Expected: FAIL — `stock_id`/`date_ordinal` are not attributes.

- [ ] **Step 3: Add the optional fields (mirror `time`)**

In `src/ophir/model_data.py`, in the `OHLCMulitClassPredictorInput` dataclass body, after the `time` field add:

```python
    stock_id: torch.Tensor | None = None
    date_ordinal: torch.Tensor | None = None
```

(No `__post_init__` handling — they arrive pre-batched from collate in the eval path and default to `None` everywhere else. `to_cuda` intentionally leaves them on CPU; the eval reads them on CPU.)

Run: `uv run pytest tests/test_model_data.py -k identity_fields -v` → PASS.

- [ ] **Step 4: Failing test — `extract_model_data` emits identity tensors**

```python
# tests/test_ticker_features.py (add — extract_model_data lives in ticker)
def test_extract_model_data_includes_identity_when_stock_id_given(ohlcv_df):
    window = extract_features(ohlcv_df).iloc[:30]
    payload = extract_model_data(window, response_size=5, stock_id=7)

    assert payload["stock_id"].item() == 7
    assert payload["stock_id"].dtype == torch.long
    assert payload["date_ordinal"].shape[0] == len(window)
    assert payload["date_ordinal"].dtype == torch.int64
```

Run: `uv run pytest tests/test_ticker_features.py -k identity -v` → FAIL (no `stock_id` key).

- [ ] **Step 5: Implement the `extract_model_data` identity params**

Change the signature and add the identity block (after the existing `return_date` block):

```python
def extract_model_data(
    df: pd.DataFrame,
    response_size: int | np.ndarray[Any, Any],
    return_date: bool = False,
    stock_id: int | None = None,
) -> dict[str, Any]:
    ...
    if return_date:
        model_data["time"] = df.index.to_numpy()
    if stock_id is not None:
        model_data["stock_id"] = torch.tensor(stock_id, dtype=torch.long)
        ordinals = df.index.to_numpy().astype("datetime64[D]").astype(np.int64)
        model_data["date_ordinal"] = torch.from_numpy(ordinals)
    return model_data
```

Run: `uv run pytest tests/test_ticker_features.py -k identity -v` → PASS.

- [ ] **Step 6: Add `symbol` to `StockStreamer` and set it in `stock_streamer`**

In `StockStreamer` (dataclass, `ticker.py:411`), add an optional field:

```python
    symbol: str | None = None
```

In `StockHanlder.stock_streamer` (`ticker.py:718`), pass it:

```python
        return StockStreamer(
            ohlc_df=self.stock_df(stock),
            seq_len=self.seq_len,
            offset=self.offset,
            shuffle=self.shuffle,
            stock_split=stock_split,
            symbol=stock,
        )
```

(`symbol` defaults to `None`, so the many direct `StockStreamer(...)` constructions in tests are unaffected.)

- [ ] **Step 7: Failing test — `StockHandlerDataset` yields identity when enabled**

```python
# tests/test_ticker_datasets.py (add; reuse the existing `parquet_dir` fixture + handler pattern)
def test_handler_dataset_yields_identity_when_enabled(parquet_dir, mocker):
    mocker.patch("numpy.random.randint", return_value=0)
    handler = StockHanlder(
        seq_len=20, base_path=str(parquet_dir), return_stock_id=False,
        return_streamer=True, offset=20,
    )
    ds = StockHandlerDataset(handler, response_size=5, cache_size=1, return_identity=True)
    payload = next(iter(ds))

    assert "stock_id" in payload and "date_ordinal" in payload
    assert payload["stock_id"].dtype == torch.long
    # stock_id is a valid index into the handler's stock list.
    assert 0 <= int(payload["stock_id"]) < len(handler)
```

Run: `uv run pytest tests/test_ticker_datasets.py -k yields_identity -v` → FAIL (`return_identity` is not a parameter).

- [ ] **Step 8: Implement `return_identity` in `StockHandlerDataset`**

Add `return_identity: bool = False` to `StockHandlerDataset.__init__` and store it. In `__iter__`, store the stock index alongside the streamer in the cache and pass it through:

```python
        cache: list[tuple[int, StockStreamer]] = []
        ...
            if len(cache) < self.cache_size and cur_stock < len(shard_stock_indices):
                stock_ind = int(shard_stock_indices[cur_stock])
                streamer = self.stock_hanlder[stock_ind]
                cache.append((stock_ind, streamer))
                cur_stock += 1

            cache_index = np.random.randint(len(cache))
            stock_ind, streamer = cache[cache_index]
            try:
                df = streamer.next()
                sid = stock_ind if self.return_identity else None
                yield extract_model_data(df, self.response_size, stock_id=sid)
            except StopIteration:
                processed_stocks += 1
                cache.pop(cache_index)
```

Run: `uv run pytest tests/test_ticker_datasets.py -v` → PASS (including the existing dataset tests — the cache is now a tuple list; verify no other test indexes the cache directly).

- [ ] **Step 9: Thread `return_identity` through `build_dataloader`**

In `src/ophir/train.py`:

```python
def build_dataloader(
    handler: StockHanlder,
    response_size: int,
    batch_size: int,
    num_workers: int,
    cache_size: int,
    return_identity: bool = False,
) -> DataLoader[dict[str, Any]]:
    ...
    dataset = StockHandlerDataset(
        handler, response_size=response_size, cache_size=cache_size,
        return_identity=return_identity,
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
```

- [ ] **Step 10: Run the full gate and commit**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy src/ophir && uv run pytest`
Expected: all green. Confirm `tests/test_models_leakage*.py` and every `tests/test_ticker_*.py` still pass.

```bash
git add src/ophir/model_data.py src/ophir/ticker.py src/ophir/train.py tests/test_model_data.py tests/test_ticker_features.py tests/test_ticker_datasets.py CHANGELOG.md
git commit -m "Thread opt-in stock_id and date identity through the eval dataset"
```

---

## Task 3: Wire cross-sectional rank-IC into the eval report (6b)

With identity now available, collect per-prediction `(stock_id, date)` for `r_close` in `accumulate_targets`, dedupe to one prediction per `(ticker, date)`, and feed the existing `rank_ic` in `evaluate_model`.

**Files:**
- Modify: `src/ophir/evaluate.py` — extend `AccumulatedEval`, collect identity, add `dedupe_by_ticker_date`, report rank-IC.
- Modify: `src/ophir/evaluate.py` `evaluate` (the Typer command) to pass `return_identity=True` to `build_dataloader`.
- Test: `tests/test_evaluate.py`.

**Interfaces:**
- Consumes: `AccumulatedEval` (Task 1), the identity tensors (Task 2), and the existing `rank_ic` (signature unchanged).
- Produces: `dedupe_by_ticker_date(pred, target, ids, dates) -> tuple[torch.Tensor, torch.Tensor, list[str]]` — keeps the first prediction per `(id, date)` pair (stable order) and returns deduped `pred`, `target`, and the per-row date as a `list[str]` ready for `rank_ic`. `AccumulatedEval` gains `r_close_ids: torch.Tensor | None = None` and `r_close_dates: torch.Tensor | None = None`.

- [ ] **Step 1: Failing test for `dedupe_by_ticker_date`**

```python
# tests/test_evaluate.py (add)
from ophir.evaluate import dedupe_by_ticker_date


def test_dedupe_keeps_first_per_ticker_date():
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([10.0, 20.0, 30.0])
    ids = torch.tensor([5, 5, 6])
    dates = torch.tensor([100, 100, 100])  # ticker 5 appears twice on day 100

    dp, dt, dd = dedupe_by_ticker_date(pred, target, ids, dates)

    assert dp.tolist() == [1.0, 3.0]      # second (5,100) dropped
    assert dt.tolist() == [10.0, 30.0]
    assert dd == ["100", "100"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_evaluate.py -k dedupe -v`
Expected: FAIL — `dedupe_by_ticker_date` does not exist.

- [ ] **Step 3: Implement `dedupe_by_ticker_date`**

```python
def dedupe_by_ticker_date(
    pred: torch.Tensor, target: torch.Tensor, ids: torch.Tensor, dates: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Keep one prediction per (ticker, date) so a day's cross-section is unique.

    Overlapping windows can emit several predictions for the same ticker on the
    same calendar day; rank-IC needs one row per name per day. Keeps the first
    occurrence (stable order) and returns the per-row date as strings for
    :func:`rank_ic`.
    """
    seen: set[tuple[int, int]] = set()
    keep: list[int] = []
    id_list = ids.tolist()
    date_list = dates.tolist()
    for i, (sid, day) in enumerate(zip(id_list, date_list, strict=True)):
        key = (int(sid), int(day))
        if key not in seen:
            seen.add(key)
            keep.append(i)
    index = torch.tensor(keep, dtype=torch.long)
    kept_dates = [str(int(date_list[i])) for i in keep]
    return pred[index], target[index], kept_dates
```

Run: `uv run pytest tests/test_evaluate.py -k dedupe -v` → PASS.

- [ ] **Step 4: Collect r_close identity in `accumulate_targets`**

Extend `AccumulatedEval`:

```python
    r_close_ids: torch.Tensor | None = None
    r_close_dates: torch.Tensor | None = None
```

In `accumulate_targets`, when `output.stock_id is not None`, collect per-kept-`r_close`-prediction ids and dates (parallel to the `r_close` pred/target collection):

```python
            if output.stock_id is not None and output.date_ordinal is not None:
                rs = int(output.response_size)
                resp_dates = output.date_ordinal[:, -rs:]  # (B, R)
                ids_bR = output.stock_id.view(-1, 1).expand(-1, rs)  # (B, R)
                id_lists.append(ids_bR[mask].reshape(-1).cpu())
                date_lists.append(resp_dates[mask].reshape(-1).cpu())
```

with `id_lists`/`date_lists` initialised as `[]`. In the return, set `r_close_ids`/`r_close_dates` to `torch.cat(...)` when non-empty, else `None`.

- [ ] **Step 5: Report rank-IC in `evaluate_model`**

After building `results`, when identity is present, compute and attach rank-IC to the `r_close` entry:

```python
    if acc.r_close_ids is not None and acc.r_close_dates is not None:
        pred, target = acc.channels["r_close"]
        dp, dt, dd = dedupe_by_ticker_date(pred, target, acc.r_close_ids, acc.r_close_dates)
        ic = rank_ic(dp, dt, dd)
        results["r_close"]["rank_ic_mean"] = ic["ic_mean"]
        results["r_close"]["rank_ic_ir"] = ic["ic_ir"]
```

Add `"rank_ic_mean"` and `"rank_ic_ir"` to `_METRIC_ORDER`. In the `evaluate` Typer command, pass `return_identity=True` to the `build_dataloader(...)` call so the report is populated in real runs.

- [ ] **Step 6: Failing test — end-to-end wiring with a fake model across two tickers/days**

```python
# tests/test_evaluate.py (add; reuses _FakeModel from Task 1)
def _toy_identity_batch() -> dict[str, object]:
    # B=2 (two tickers), S=3, response_size=1. Same date so they form one day's
    # cross-section; predictions rank the two names in target order.
    targets = torch.tensor(
        [[[0.0, 0.1, 0.1], [0.0, 0.1, 0.1], [0.02, 0.1, 0.1]],
         [[0.0, 0.1, 0.1], [0.0, 0.1, 0.1], [-0.01, 0.1, 0.1]]]
    )
    return {
        "feature_input": torch.zeros(2, 3, 13),
        "targets": targets,
        "trade_occured": torch.ones(2, 3, dtype=torch.bool),
        "response_size": torch.tensor(1),
        "stock_id": torch.tensor([5, 6]),
        "date_ordinal": torch.tensor([[10, 11, 12], [10, 11, 12]]),
    }


def test_evaluate_model_reports_rank_ic(monkeypatch):
    from ophir import evaluate as ev

    model = _FakeModel()  # perfect predictions
    out = ev.evaluate_model(model, [_toy_identity_batch()], max_batches=1)

    assert "rank_ic_mean" in out["r_close"]
    # One day, two names, perfectly ranked -> IC is 1.0 (or nan if a single
    # name survives dedupe — here two distinct tickers on day 12).
    assert out["r_close"]["rank_ic_mean"] == 1.0
```

Note: `_FakeModel.__call__` must forward `stock_id`/`date_ordinal` from the batch into the returned `OHLCMulitClassPredictorInput`. Update the Task 1 `_FakeModel` to pass them through (`stock_id=batch.get("stock_id")`, `date_ordinal=batch.get("date_ordinal")`).

Run: `uv run pytest tests/test_evaluate.py -k rank_ic -v` → FAIL first, then implement Steps 4-5, then PASS.

- [ ] **Step 7: Run the full gate and commit**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy src/ophir && uv run pytest`
Expected: all green.

```bash
git add src/ophir/evaluate.py tests/test_evaluate.py CHANGELOG.md
git commit -m "Report cross-sectional rank-IC for r_close in the eval report"
```

---

## Self-review checklist (run before final review)

- **Training path untouched:** `git diff main -- src/ophir/training_models.py` is empty; `build_dataloader`/`StockHandlerDataset` default `return_identity=False`; no custom `collate_fn` added.
- **Strict typing:** `evaluate.py` and `model_data.py` changes pass `uv run mypy src/ophir`; no new entries in `pyproject.toml` suppression blocks.
- **Signatures frozen:** `rank_ic` and `skill_score_vs_baseline` are unchanged (only called).
- **No regressions:** `tests/test_models_leakage*.py` and all `tests/test_ticker_*.py` green.
- **Real-run wiring:** the `evaluate` Typer command passes `return_identity=True`; confirm the report now lists `skill_vs_persistence`, `rank_ic_mean`, `rank_ic_ir`.

## Notes / out of scope

- The persistence baseline uses the last traded prefix day carried flat across the horizon. An EWMA variant and a per-response-day-offset breakdown (to expose horizon skill decay) are deferred — they reuse the same plumbing.
- `date_ordinal` is calendar-day based; if Phase 2's trading-day-calendar migration (F6) lands later, the ordinal source stays valid (it reads `df.index`).
- Multi-worker `DataLoader` collate of the new tensor fields is standard; the eval path in `evaluate.py` already runs with `num_workers` from its CLI — no custom collate needed because every field is now a fixed-shape tensor.
