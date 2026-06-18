# Hyperparameter Sweep Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an Optuna-based hyperparameter sweep harness (`ophir sweep`) that searches optimizer, loss-weight, and architecture-tier hyperparameters by mean cross-sectional rank-IC on `r_close`, with proxy-budget search → full-budget confirm.

**Architecture:** Three layers. (1) Plumbing in `training_models.py`: configurable loss weights and a new `val_rank_ic` validation metric reusing `evaluate.py` helpers. (2) A reusable `run_training` engine extracted from `train.train`, accepting injected Lightning callbacks + a seed and returning the fitted model; the Typer `train` command becomes a thin wrapper. (3) A new `sweep.py` harness (search space, custom Optuna pruning callback, top-K selection, objective/run/confirm) wired into the CLI as `ophir sweep`.

**Tech Stack:** Python 3.10+, PyTorch, PyTorch-Lightning, Optuna (new dependency), Typer, pytest.

## Global Constraints

- **Live code lives in `src/ophir/` only.** Ignore top-level `ophir/`, `oldcode/`, `old_*.py`.
- **Backward-compat contract:** a default `ophir train` must stay byte-for-byte unchanged. Every newly exposed param defaults to its current value; `val_rank_ic` and identity-carrying val loaders are opt-in (`val_identity=False` by default).
- **Strict typing:** `mypy --strict` over `src/ophir` with `warn_unused_ignores=true` — a stray `# type: ignore` fails. Only override is `ignore_missing_imports` for stubless packages. mypy `python_version = "3.10"`; ruff `target-version = "py312"` — do not unify.
- **Tests:** deterministic, seeded, **CPU-safe and network-free**. pytest runs with `filterwarnings = ["error", …]` and `--strict-config` — tests and `src/ophir` must run warning-clean. CUDA-only paths (the real `fit`/forward) are exercised via pure helpers or monkeypatched doubles, never a live GPU.
- **Identity tensors are int64**, threaded opt-in; never strings (default collate can't stack them).
- **Style:** comments only for non-obvious *why*; no removed-code stubs or compat shims. CHANGELOG entries match existing format.
- Run after each task: `uv run ruff check . && uv run ruff format --check . && uv run mypy src/ophir && uv run pytest`.

---

### Task 1: Configurable loss weights in `LightningOHLCPredictor`

Replace the hardcoded `close + 0.5·upside + 0.5·downside` combination with constructor-configurable `upside_weight` / `downside_weight` (defaults `0.5`, `0.5` — current behavior).

**Files:**
- Modify: `src/ophir/training_models.py` (`__init__` ~46-58, `compute_loss` return ~293)
- Test: `tests/test_training_models.py`

**Interfaces:**
- Produces: `LightningOHLCPredictor(..., upside_weight: float = 0.5, downside_weight: float = 0.5)`; `compute_loss` returns `close_loss + upside_weight * upside_loss + downside_weight * downside_loss`. The component scalars are still logged under `{state}_r_close_loss` / `{state}_upside_loss` / `{state}_downside_loss`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_training_models.py`:

```python
def _toy_model_output() -> object:
    """A populated OHLCMulitClassPredictorInput with known targets/predictions."""
    from ophir.model_data import OHLCMulitClassPredictorInput

    # 2 examples, seq 4, response 2, 3 channels. Predictions deliberately offset
    # from targets so each channel has a non-zero loss.
    targets = torch.zeros(2, 4, 3)
    targets[..., 0] = 0.10  # r_close
    targets[..., 1] = 0.20  # upside
    targets[..., 2] = 0.30  # downside
    model_output = torch.zeros(2, 4, 3)  # all-zero predictions
    trade = torch.ones(2, 4, dtype=torch.bool)
    return OHLCMulitClassPredictorInput(
        feature_input=torch.zeros(2, 4, 13),
        response_size=torch.tensor(2),
        trade_occured=trade,
        targets=targets,
        model_output=model_output,
    )


def _build_predictor(**kwargs: float) -> LightningOHLCPredictor:
    return LightningOHLCPredictor(emb_dim=16, num_layers=1, num_heads=2, **kwargs)


def test_loss_weights_combine_components() -> None:
    logged: dict[str, float] = {}
    model = _build_predictor(upside_weight=0.4, downside_weight=0.7)
    model.loss_state = "val"
    model.log = lambda name, value, **kw: logged.__setitem__(name, float(value))  # type: ignore[method-assign]

    loss = model.compute_loss(_toy_model_output())  # type: ignore[arg-type]

    expected = (
        logged["val_r_close_loss"]
        + 0.4 * logged["val_upside_loss"]
        + 0.7 * logged["val_downside_loss"]
    )
    assert abs(float(loss) - expected) < 1e-6


def test_loss_weights_default_to_half() -> None:
    model = _build_predictor()
    assert model.upside_weight == 0.5
    assert model.downside_weight == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_training_models.py::test_loss_weights_combine_components tests/test_training_models.py::test_loss_weights_default_to_half -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'upside_weight'`.

- [ ] **Step 3: Add the constructor params**

In `src/ophir/training_models.py` `__init__`, add two params after `loss_decay` and store them. Signature becomes:

```python
    def __init__(
        self,
        emb_dim: int,
        num_layers: int,
        num_heads: int,
        lr: float = 2e-4,
        rezero_lr: float = 3e-4,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.95),
        warmup_ratio: float = 0.03,
        max_steps: int = 100000,
        loss_decay: float = 0.6,
        upside_weight: float = 0.5,
        downside_weight: float = 0.5,
    ) -> None:
```

Add to the docstring's Parameters section (match the existing NumPy style):

```
        upside_weight : float, optional
            Weight of the upside channel in the combined loss. Defaults to
            ``0.5``.
        downside_weight : float, optional
            Weight of the downside channel in the combined loss. Defaults to
            ``0.5``.
```

After the existing `self.loss_decay = loss_decay` assignment (before `self.save_hyperparameters()`), add:

```python
        self.upside_weight = upside_weight
        self.downside_weight = downside_weight
```

- [ ] **Step 4: Use the weights in `compute_loss`**

Replace the return line (currently `return close_loss + 0.5 * upside_loss + 0.5 * downside_loss`) with:

```python
        return close_loss + self.upside_weight * upside_loss + self.downside_weight * downside_loss
```

Also update the `compute_loss` docstring phrase "combined as ``close + 0.5·upside + 0.5·downside``" to "combined as ``close + upside_weight·upside + downside_weight·downside``".

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_training_models.py -v`
Expected: PASS (new tests + existing `robust_scale` tests).

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff check src/ophir/training_models.py tests/test_training_models.py
uv run mypy src/ophir
git add src/ophir/training_models.py tests/test_training_models.py
git commit -m "Make loss combination weights configurable"
```

---

### Task 2: `val_rank_ic` validation metric

Compute mean cross-sectional rank-IC on `r_close` each validation pass, but **only** when the val loader carries identity (`stock_id` / `date_ordinal`). Reuses `evaluate.dedupe_by_ticker_date` + `evaluate.rank_ic`. No identity ⇒ nothing logged ⇒ training path unchanged.

**Files:**
- Modify: `src/ophir/training_models.py` (add module-level helper, `__init__` buffer init, `validation_step`, new `on_validation_epoch_end`)
- Test: `tests/test_training_models.py`

**Interfaces:**
- Consumes: `evaluate.dedupe_by_ticker_date(pred, target, ids, dates)`, `evaluate.rank_ic(pred, target, dates) -> dict` (`ic_mean` key). `OHLCMulitClassPredictorInput.predicted_r_close` / `.target_r_close` (already trimmed to the response block), `.stock_id` `(B,)`, `.date_ordinal` `(B, S)`, `.trade_occured` `(B, S)`, `.response_size`.
- Produces: module-level `val_rank_ic(pred, target, ids, dates) -> float`; logs `val_rank_ic` scalar each validation epoch when identity is present.

- [ ] **Step 1: Write the failing test for the pure helper**

Add to `tests/test_training_models.py` (top: `from ophir.training_models import LightningOHLCPredictor, robust_scale, val_rank_ic`):

```python
def test_val_rank_ic_perfect_ranking_is_positive() -> None:
    # Two days (date ordinals 10 and 11), three tickers each. Predictions rank
    # the same way as targets within each day -> rank-IC == 1.0.
    pred = torch.tensor([3.0, 2.0, 1.0, 1.0, 2.0, 3.0])
    target = torch.tensor([0.3, 0.2, 0.1, 0.1, 0.2, 0.3])
    ids = torch.tensor([1, 2, 3, 1, 2, 3])
    dates = torch.tensor([10, 10, 10, 11, 11, 11])
    assert val_rank_ic(pred, target, ids, dates) > 0.99


def test_val_rank_ic_empty_is_nan() -> None:
    empty = torch.tensor([])
    result = val_rank_ic(empty, empty, empty.long(), empty.long())
    assert result != result  # NaN
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_training_models.py::test_val_rank_ic_perfect_ranking_is_positive -v`
Expected: FAIL — `ImportError: cannot import name 'val_rank_ic'`.

- [ ] **Step 3: Implement the pure helper**

In `src/ophir/training_models.py`, after `robust_scale` (around line 33), add:

```python
def val_rank_ic(
    pred: torch.Tensor,
    target: torch.Tensor,
    ids: torch.Tensor,
    dates: torch.Tensor,
) -> float:
    """Mean cross-sectional rank-IC of ``r_close`` over a validation epoch.

    Dedupes to one prediction per ``(ticker, date)`` and averages the daily
    Spearman rank correlation, reusing the eval module's helpers so the
    validation metric and the offline report agree. Returns ``nan`` for an
    empty input (no identity collected).
    """
    from .evaluate import dedupe_by_ticker_date, rank_ic

    if pred.numel() == 0:
        return float("nan")
    dp, dt, dd = dedupe_by_ticker_date(pred, target, ids, dates)
    return rank_ic(dp, dt, dd)["ic_mean"]
```

(The `from .evaluate import …` is a function-local import: `evaluate` imports `LightningOHLCPredictor` only under `TYPE_CHECKING`, so a module-level import would also be cycle-free, but keeping it local matches the repo's lazy-import style and keeps `evaluate`'s `typer` import off the hot path.)

- [ ] **Step 4: Run the helper tests**

Run: `uv run pytest tests/test_training_models.py::test_val_rank_ic_perfect_ranking_is_positive tests/test_training_models.py::test_val_rank_ic_empty_is_nan -v`
Expected: PASS.

- [ ] **Step 5: Wire accumulation into the validation loop**

In `__init__`, after `self.loss_state = "train"`, initialize per-epoch buffers:

```python
        self._val_ic_buffers: dict[str, list[torch.Tensor]] = {
            "pred": [],
            "target": [],
            "ids": [],
            "dates": [],
        }
```

In `validation_step`, after `loss = self.compute_loss(model_output)` and before `return loss`, accumulate identity-aligned predictions (mirrors `evaluate.accumulate_targets` masking):

```python
        if model_output.stock_id is not None and model_output.date_ordinal is not None:
            rs = int(model_output.response_size)
            mask = model_output.trade_occured[:, -rs:]
            resp_dates = model_output.date_ordinal[:, -rs:]
            ids_br = model_output.stock_id.view(-1, 1).expand(-1, rs)
            self._val_ic_buffers["pred"].append(model_output.predicted_r_close[mask].reshape(-1).cpu())
            self._val_ic_buffers["target"].append(model_output.target_r_close[mask].reshape(-1).cpu())
            self._val_ic_buffers["ids"].append(ids_br[mask].reshape(-1).cpu())
            self._val_ic_buffers["dates"].append(resp_dates[mask].reshape(-1).cpu())
```

Add a new method after `validation_step`:

```python
    def on_validation_epoch_end(self) -> None:
        """Log ``val_rank_ic`` from accumulated identity, then reset buffers.

        Only fires when the validation loader carries identity (``stock_id`` /
        ``date_ordinal``); without it the buffers stay empty and nothing is
        logged, so the default training path is unchanged.
        """
        preds = self._val_ic_buffers["pred"]
        if preds:
            ic = val_rank_ic(
                torch.cat(preds),
                torch.cat(self._val_ic_buffers["target"]),
                torch.cat(self._val_ic_buffers["ids"]),
                torch.cat(self._val_ic_buffers["dates"]),
            )
            self.log("val_rank_ic", ic, prog_bar=False, on_epoch=True, logger=True)
        for buf in self._val_ic_buffers.values():
            buf.clear()
```

- [ ] **Step 6: Run the full test module**

Run: `uv run pytest tests/test_training_models.py -v`
Expected: PASS.

- [ ] **Step 7: Lint, type-check, commit**

```bash
uv run ruff check src/ophir/training_models.py tests/test_training_models.py
uv run mypy src/ophir
git add src/ophir/training_models.py tests/test_training_models.py
git commit -m "Log val_rank_ic when the validation loader carries identity"
```

---

### Task 3: Extract `run_training` engine; thin `train` CLI wrapper

Split `train.train` into a reusable `run_training(...)` engine (accepts `betas`, `upside_weight`, `downside_weight`, `rezero_lr`, `val_identity`, injected `callbacks`, `seed`; **returns the fitted model**) and a thin Typer `train` command exposing the new knobs as CLI-friendly scalars. CLI defaults unchanged.

**Files:**
- Modify: `src/ophir/register.py` (`fetch_base_trainer` — add `extra_callbacks`)
- Modify: `src/ophir/train.py` (`train` body → `run_training`; new thin `train`)
- Test: `tests/test_train.py` (create)

**Interfaces:**
- Consumes: `LightningOHLCPredictor(..., rezero_lr, betas, upside_weight, downside_weight)` (Tasks 1–2), `build_split_handlers`, `build_dataloader(..., return_identity)`, `estimate_windows`, `count_windows`, `steps_for_epochs`, `register.fetch_base_trainer`.
- Produces:
  - `register.fetch_base_trainer(..., extra_callbacks: list[Callback] | None = None) -> L.Trainer`
  - `train.run_training(*, emb_dim, num_layers, num_heads, seq_len, offset, response_size, batch_size, num_workers, cache_size, min_volume, train_min_year, train_max_year, val_min_year, val_max_year, data_dir, use_sp500, use_quality_allowlist, clean_rows, max_abs_r_close, epochs, max_steps, window_sample, val_every_steps, val_batches, lr, rezero_lr, weight_decay, betas, warmup_ratio, loss_decay, upside_weight, downside_weight, val_identity=False, callbacks=None, seed=None) -> LightningOHLCPredictor`
  - `train.train(...)` Typer command exposing `rezero_lr`, `beta1`, `beta2`, `upside_weight`, `downside_weight`, `val_identity`, `seed` (returns `None`).

- [ ] **Step 1: Write the failing test (hparam forwarding, CPU, monkeypatched fit)**

Create `tests/test_train.py`:

```python
"""Tests for the training engine wiring (CPU-safe, no CUDA, no data)."""

from typing import Any

import pytest

from ophir import train
from ophir.training_models import LightningOHLCPredictor


class _FakeTrainer:
    def __init__(self) -> None:
        self.fitted_model: LightningOHLCPredictor | None = None

    def fit(self, model: LightningOHLCPredictor, **_: Any) -> None:
        self.fitted_model = model


@pytest.fixture
def patched_engine(monkeypatch: pytest.MonkeyPatch) -> _FakeTrainer:
    """Stub out data + trainer so run_training builds a real CPU model only."""
    trainer = _FakeTrainer()
    monkeypatch.setattr(train, "build_split_handlers", lambda **_: ("train_h", "val_h"))
    monkeypatch.setattr(train, "build_dataloader", lambda *a, **k: "loader")
    monkeypatch.setattr(train, "estimate_windows", lambda *a, **k: 1000)
    monkeypatch.setattr(train, "_validate_dims", lambda *a, **k: None)
    from ophir import register

    monkeypatch.setattr(register, "fetch_base_trainer", lambda **_: trainer)
    monkeypatch.setattr(register, "get_default_data_days_dir", lambda: "/tmp")
    return trainer


def test_run_training_forwards_hyperparameters(patched_engine: _FakeTrainer) -> None:
    model = train.run_training(
        emb_dim=16,
        num_layers=1,
        num_heads=2,
        lr=1e-3,
        rezero_lr=5e-4,
        weight_decay=0.05,
        betas=(0.9, 0.98),
        upside_weight=0.4,
        downside_weight=0.7,
        loss_decay=0.5,
    )
    assert model is patched_engine.fitted_model
    assert model.lr == 1e-3
    assert model.rezero_lr == 5e-4
    assert model.betas == (0.9, 0.98)
    assert model.upside_weight == 0.4
    assert model.downside_weight == 0.7


def test_run_training_passes_val_identity(monkeypatch: pytest.MonkeyPatch, patched_engine: _FakeTrainer) -> None:
    calls: list[bool] = []

    def fake_loader(*_a: Any, return_identity: bool = False, **_k: Any) -> str:
        calls.append(return_identity)
        return "loader"

    monkeypatch.setattr(train, "build_dataloader", fake_loader)
    train.run_training(emb_dim=16, num_layers=1, num_heads=2, val_identity=True)
    # Two loaders built (train, val); the val one carries identity.
    assert calls == [False, True]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train.py -v`
Expected: FAIL — `AttributeError: module 'ophir.train' has no attribute 'run_training'`.

- [ ] **Step 3: Add `extra_callbacks` to `fetch_base_trainer`**

In `src/ophir/register.py`, change the signature (line ~53) to add `extra_callbacks`:

```python
def fetch_base_trainer(
    file_name: str | None = None,
    max_steps: int = 100000,
    val_check_interval: int | float = 1.0,
    limit_val_batches: int | float = 1.0,
    extra_callbacks: list[L.Callback] | None = None,
) -> L.Trainer:
```

Add a one-line Parameters entry to the docstring:

```
    extra_callbacks : list[lightning.Callback], optional
        Additional callbacks appended to the default set (e.g. a sweep's
        pruning callback). Defaults to ``None``.
```

In the body, replace the inline `callbacks=[...]` list in the `L.Trainer(...)` call with a pre-built list that extends with the extras:

```python
    callbacks: list[L.Callback] = [
        time_checkpoint_callback,
        epoch_checkpoint_callback,
        LearningRateMonitor("step"),
    ]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    trainer = L.Trainer(
        max_steps=max_steps,
        precision="16-mixed",
        default_root_dir=MODEL_DIR,
        accelerator="cuda",
        callbacks=callbacks,
        logger=[
            TensorBoardLogger(MODEL_DIR, name="tensorboard-logger"),
            CSVLogger(MODEL_DIR, name="csv-logger", flush_logs_every_n_steps=10),
        ],
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=None if isinstance(val_check_interval, int) else 1,
        limit_val_batches=limit_val_batches,
        gradient_clip_val=1,
        gradient_clip_algorithm="norm",
    )
```

`L` is already imported at the top of `fetch_base_trainer` (`import lightning as L`); the `L.Callback` annotation resolves there.

- [ ] **Step 4: Extract `run_training` and rewrite `train`**

In `src/ophir/train.py`, replace the entire `def train(...)` function (lines ~290-435) with the engine plus a thin command. First the engine:

```python
def run_training(
    *,
    emb_dim: int = 128,
    num_layers: int = 6,
    num_heads: int = 8,
    seq_len: int = 365,
    offset: int = 90,
    response_size: int = 90,
    batch_size: int = 32,
    num_workers: int = 4,
    cache_size: int = 8,
    min_volume: float = 1000.0,
    train_min_year: int | None = None,
    train_max_year: int = 2023,
    val_min_year: int = 2024,
    val_max_year: int | None = None,
    data_dir: str | None = None,
    use_sp500: bool = False,
    use_quality_allowlist: bool = False,
    clean_rows: bool = False,
    max_abs_r_close: float = 0.75,
    epochs: int = 10,
    max_steps: int | None = None,
    window_sample: int = 256,
    val_every_steps: int = 500,
    val_batches: int = 50,
    lr: float = 2e-4,
    rezero_lr: float = 3e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.95),
    warmup_ratio: float = 0.03,
    loss_decay: float = 0.6,
    upside_weight: float = 0.5,
    downside_weight: float = 0.5,
    val_identity: bool = False,
    callbacks: list[Any] | None = None,
    seed: int | None = None,
) -> LightningOHLCPredictor:
    """Build the date-split dataloaders, fit a model, and return it.

    The reusable engine behind ``ophir train`` and the sweep harness. When
    ``val_identity`` is set the validation loader carries opt-in identity so the
    model logs ``val_rank_ic``; ``callbacks`` are appended to the base trainer
    (e.g. a sweep pruning callback); ``seed`` seeds Lightning for reproducible
    trials. Requires CUDA.
    """
    import lightning as L

    from ophir import register
    from ophir.training_models import LightningOHLCPredictor

    if seed is not None:
        L.seed_everything(seed, workers=True)

    _validate_dims(emb_dim, num_heads, seq_len, response_size)

    base_path = os.path.join(data_dir or register.get_default_data_days_dir(), "stocks")
    train_handler, val_handler = build_split_handlers(
        base_path=base_path,
        seq_len=seq_len,
        offset=offset,
        min_volume=min_volume,
        train_min_year=train_min_year,
        train_max_year=train_max_year,
        val_min_year=val_min_year,
        val_max_year=val_max_year,
        use_sp500=use_sp500,
        use_quality_allowlist=use_quality_allowlist,
        clean_rows=clean_rows,
        max_abs_r_close=max_abs_r_close,
    )

    if max_steps is None:
        if window_sample > 0:
            typer.echo(f"Estimating dataset size from a {window_sample}-stock sample…")
            num_windows = estimate_windows(train_handler, sample_size=window_sample)
            prefix = "~"
        else:
            typer.echo("Counting all training windows (one data pass)…")
            num_windows = count_windows(train_handler)
            prefix = ""
        max_steps = steps_for_epochs(num_windows, batch_size, epochs)
        typer.echo(
            f"{prefix}{num_windows} train windows / batch {batch_size} = "
            f"{max_steps // epochs} steps/epoch x {epochs} epochs = {max_steps} max_steps"
        )

    train_dl = build_dataloader(train_handler, response_size, batch_size, num_workers, cache_size)
    val_dl = build_dataloader(
        val_handler,
        response_size,
        batch_size,
        num_workers,
        cache_size,
        return_identity=val_identity,
    )

    model = LightningOHLCPredictor(
        emb_dim=emb_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        lr=lr,
        rezero_lr=rezero_lr,
        weight_decay=weight_decay,
        betas=betas,
        warmup_ratio=warmup_ratio,
        max_steps=max_steps,
        loss_decay=loss_decay,
        upside_weight=upside_weight,
        downside_weight=downside_weight,
    )
    trainer = register.fetch_base_trainer(
        max_steps=max_steps,
        val_check_interval=val_every_steps,
        limit_val_batches=val_batches,
        extra_callbacks=callbacks,
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    return model
```

Add the needed import at the top of `train.py` — under `if TYPE_CHECKING:` add `from ophir.training_models import LightningOHLCPredictor` (return annotation only; the runtime import stays inside `run_training`). Confirm `Any` is already imported (it is, line 24).

Then the thin Typer command (keep the rich existing docstring; append entries for the new params):

```python
def train(
    emb_dim: int = 128,
    num_layers: int = 6,
    num_heads: int = 8,
    seq_len: int = 365,
    offset: int = 90,
    response_size: int = 90,
    batch_size: int = 32,
    num_workers: int = 4,
    cache_size: int = 8,
    min_volume: float = 1000.0,
    train_min_year: int | None = None,
    train_max_year: int = 2023,
    val_min_year: int = 2024,
    val_max_year: int | None = None,
    data_dir: str | None = None,
    use_sp500: bool = False,
    use_quality_allowlist: bool = False,
    clean_rows: bool = False,
    max_abs_r_close: float = 0.75,
    epochs: int = 10,
    max_steps: int | None = None,
    window_sample: int = 256,
    val_every_steps: int = 500,
    val_batches: int = 50,
    lr: float = 2e-4,
    rezero_lr: float = 3e-4,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.95,
    warmup_ratio: float = 0.03,
    loss_decay: float = 0.6,
    upside_weight: float = 0.5,
    downside_weight: float = 0.5,
    val_identity: bool = False,
    seed: int | None = None,
) -> None:
    """Pre-train a base :class:`LightningOHLCPredictor` from scratch.

    Thin CLI wrapper over :func:`run_training`; see it for the engine details.
    Validation runs every ``val_every_steps`` optimizer steps over at most
    ``val_batches`` batches. Pass ``--val-identity`` to also log ``val_rank_ic``.
    Requires CUDA.
    """
    run_training(
        emb_dim=emb_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        seq_len=seq_len,
        offset=offset,
        response_size=response_size,
        batch_size=batch_size,
        num_workers=num_workers,
        cache_size=cache_size,
        min_volume=min_volume,
        train_min_year=train_min_year,
        train_max_year=train_max_year,
        val_min_year=val_min_year,
        val_max_year=val_max_year,
        data_dir=data_dir,
        use_sp500=use_sp500,
        use_quality_allowlist=use_quality_allowlist,
        clean_rows=clean_rows,
        max_abs_r_close=max_abs_r_close,
        epochs=epochs,
        max_steps=max_steps,
        window_sample=window_sample,
        val_every_steps=val_every_steps,
        val_batches=val_batches,
        lr=lr,
        rezero_lr=rezero_lr,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
        warmup_ratio=warmup_ratio,
        loss_decay=loss_decay,
        upside_weight=upside_weight,
        downside_weight=downside_weight,
        val_identity=val_identity,
        seed=seed,
    )
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_train.py -v`
Expected: PASS (both forwarding tests).

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff check src/ophir/train.py src/ophir/register.py tests/test_train.py
uv run mypy src/ophir
uv run pytest
git add src/ophir/train.py src/ophir/register.py tests/test_train.py
git commit -m "Extract run_training engine and expose new training knobs"
```

---

### Task 4: `sweep.py` harness — search space, pruning, objective, confirm

The Optuna harness: size-tier presets, search-space sampler, a small custom Lightning pruning callback (avoids the `optuna_integration` packaging churn), top-K selection, and the CUDA objective/run/confirm orchestration.

**Files:**
- Create: `src/ophir/sweep.py`
- Test: `tests/test_sweep.py` (create)

**Interfaces:**
- Consumes: `optuna` (`Trial`, `Study`, `TPESampler`, `SuccessiveHalvingPruner`, `TrialPruned`); `train.run_training`; `evaluate.evaluate_model`, `train.build_split_handlers`, `train.build_dataloader`; `lightning.pytorch.callbacks.Callback`.
- Produces:
  - `SIZE_TIERS: dict[str, dict[str, int]]`
  - `sample_config(trial: optuna.Trial) -> dict[str, Any]`
  - `class _OptunaPruning(Callback)` monitoring `val_rank_ic`
  - `select_top_configs(study: optuna.Study, k: int) -> list[dict[str, Any]]`
  - `objective(trial, *, proxy_kwargs: dict[str, Any], base_seed: int) -> float`
  - `run_sweep(*, n_trials, study_name, storage, base_seed, proxy_kwargs) -> optuna.Study`
  - `confirm_top(study, *, k, full_kwargs, val_batches) -> list[dict[str, Any]]`

- [ ] **Step 1: Write failing tests for the pure pieces**

Create `tests/test_sweep.py`:

```python
"""CPU-safe tests for the sweep harness pure helpers (no CUDA, no Optuna run)."""

import optuna

from ophir import sweep


def test_size_tiers_satisfy_head_divisibility() -> None:
    for name, arch in sweep.SIZE_TIERS.items():
        emb, heads = arch["emb_dim"], arch["num_heads"]
        assert emb % 4 == 0, name
        assert emb % heads == 0, name
        assert emb // heads >= 16, name  # flex-attention floor


def test_sample_config_returns_valid_arch_and_ranges() -> None:
    # FixedTrial supplies deterministic values for each suggested param.
    trial = optuna.trial.FixedTrial(
        {
            "size_tier": "base",
            "lr": 1e-3,
            "rezero_lr": 5e-4,
            "weight_decay": 0.02,
            "warmup_ratio": 0.05,
            "loss_decay": 0.6,
            "beta2": 0.95,
            "upside_weight": 0.5,
            "downside_weight": 0.5,
        }
    )
    config = sample_config_value = sweep.sample_config(trial)
    assert config["emb_dim"] == sweep.SIZE_TIERS["base"]["emb_dim"]
    assert config["num_heads"] == sweep.SIZE_TIERS["base"]["num_heads"]
    assert config["betas"] == (0.9, 0.95)
    assert config["lr"] == 1e-3
    assert "size_tier" not in config  # mapped into train kwargs, not passed raw


def test_select_top_configs_orders_by_value_desc() -> None:
    study = optuna.create_study(direction="maximize")
    for i, value in enumerate([0.1, 0.5, 0.3]):
        trial = optuna.trial.create_trial(
            params={"x": float(i)},
            distributions={"x": optuna.distributions.FloatDistribution(0, 10)},
            value=value,
            user_attrs={"config": {"lr": value}},
        )
        study.add_trial(trial)
    top = sweep.select_top_configs(study, k=2)
    assert [c["lr"] for c in top] == [0.5, 0.3]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sweep.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ophir.sweep'` (or `optuna` if not yet installed; Task 5 adds the dep — install it now for local dev with `uv add optuna` is done in Task 5 Step 3, but to run this test, install first: `uv run --with optuna pytest ...`, or do Task 5 Step 3 before this. If `optuna` import fails, jump to Task 5 Step 3, then return.)

- [ ] **Step 3: Implement `sweep.py`**

Create `src/ophir/sweep.py`:

```python
"""Optuna hyperparameter sweep for the OHLC forecaster.

Searches optimizer, loss-weight, and architecture-tier hyperparameters by mean
cross-sectional rank-IC on ``r_close`` (the model logs ``val_rank_ic`` when the
validation loader carries identity). Each trial runs a reduced-budget *proxy*
training with a pruning callback; the best configs are then retrained at full
budget and scored with the offline eval report (:func:`confirm_top`).

Requires CUDA for the actual trials; the pure helpers (search space, top-K
selection) are CPU-safe and unit-tested.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from lightning.pytorch.callbacks import Callback

if TYPE_CHECKING:
    import optuna
    from lightning.pytorch import LightningModule, Trainer

#: Architecture presets; each satisfies emb_dim % 4 == 0, emb_dim % num_heads
#: == 0, and head_dim >= 16 (the flex-attention CUDA floor).
SIZE_TIERS: dict[str, dict[str, int]] = {
    "small": {"emb_dim": 64, "num_layers": 4, "num_heads": 4},
    "base": {"emb_dim": 128, "num_layers": 6, "num_heads": 8},
    "large": {"emb_dim": 192, "num_layers": 8, "num_heads": 12},
}


def sample_config(trial: optuna.Trial) -> dict[str, Any]:
    """Sample one hyperparameter configuration as ``run_training`` kwargs."""
    tier = trial.suggest_categorical("size_tier", list(SIZE_TIERS))
    arch = SIZE_TIERS[tier]
    beta2 = trial.suggest_float("beta2", 0.9, 0.999)
    return {
        "emb_dim": arch["emb_dim"],
        "num_layers": arch["num_layers"],
        "num_heads": arch["num_heads"],
        "lr": trial.suggest_float("lr", 5e-5, 2e-3, log=True),
        "rezero_lr": trial.suggest_float("rezero_lr", 5e-5, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.1),
        "loss_decay": trial.suggest_float("loss_decay", 0.3, 1.0),
        "betas": (0.9, beta2),
        "upside_weight": trial.suggest_float("upside_weight", 0.25, 1.0),
        "downside_weight": trial.suggest_float("downside_weight", 0.25, 1.0),
    }


class _OptunaPruning(Callback):
    """Report ``val_rank_ic`` to an Optuna trial and prune unpromising runs."""

    def __init__(self, trial: optuna.Trial) -> None:
        self._trial = trial

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        import optuna

        metric = trainer.callback_metrics.get("val_rank_ic")
        if metric is None:
            return
        step = trainer.global_step
        self._trial.report(float(metric), step)
        if self._trial.should_prune():
            raise optuna.TrialPruned(f"pruned at step {step}")


def select_top_configs(study: optuna.Study, k: int) -> list[dict[str, Any]]:
    """Return the configs of the top-``k`` completed trials, best first."""
    import optuna

    completed = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    completed.sort(key=lambda t: t.value, reverse=True)
    return [t.user_attrs["config"] for t in completed[:k]]


def objective(trial: optuna.Trial, *, proxy_kwargs: dict[str, Any], base_seed: int) -> float:
    """Run one proxy trial; return its best ``val_rank_ic`` (maximize)."""
    import optuna

    config = sample_config(trial)
    trial.set_user_attr("config", config)
    from ophir.train import run_training

    run_training(
        **proxy_kwargs,
        **config,
        val_identity=True,
        seed=base_seed + trial.number,
        callbacks=[_OptunaPruning(trial)],
    )
    if not trial.intermediate_values:
        raise optuna.TrialPruned("no val_rank_ic was reported")
    return max(trial.intermediate_values.values())


def run_sweep(
    *,
    n_trials: int,
    study_name: str,
    storage: str,
    base_seed: int,
    proxy_kwargs: dict[str, Any],
) -> optuna.Study:
    """Create/resume the SQLite study and run ``n_trials`` proxy trials."""
    import optuna

    sampler = optuna.samplers.TPESampler(seed=base_seed)
    pruner = optuna.pruners.SuccessiveHalvingPruner()
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, proxy_kwargs=proxy_kwargs, base_seed=base_seed),
        n_trials=n_trials,
    )
    return study


def confirm_top(
    study: optuna.Study,
    *,
    k: int,
    full_kwargs: dict[str, Any],
    val_batches: int,
) -> list[dict[str, Any]]:
    """Retrain the top-``k`` configs at full budget and score with the eval report.

    Returns one record per config: its hyperparameters plus the authoritative
    ``rank_ic_mean`` and per-channel skill scores from
    :func:`ophir.evaluate.evaluate_model`. Requires CUDA.
    """
    from ophir import register
    from ophir.evaluate import evaluate_model
    from ophir.train import build_dataloader, build_split_handlers, run_training

    base_path = os.path.join(
        full_kwargs.get("data_dir") or register.get_default_data_days_dir(), "stocks"
    )
    results: list[dict[str, Any]] = []
    for config in select_top_configs(study, k):
        model = run_training(**full_kwargs, **config, val_identity=True)
        _, val_handler = build_split_handlers(
            base_path=base_path,
            seq_len=full_kwargs["seq_len"],
            offset=full_kwargs["offset"],
            min_volume=full_kwargs["min_volume"],
            train_min_year=full_kwargs["train_min_year"],
            train_max_year=full_kwargs["train_max_year"],
            val_min_year=full_kwargs["val_min_year"],
            val_max_year=full_kwargs["val_max_year"],
            use_sp500=full_kwargs["use_sp500"],
        )
        val_dl = build_dataloader(
            val_handler,
            full_kwargs["response_size"],
            full_kwargs["batch_size"],
            full_kwargs["num_workers"],
            full_kwargs["cache_size"],
            return_identity=True,
        )
        report = evaluate_model(model, val_dl, val_batches)
        results.append({"config": config, "report": report})
    results.sort(
        key=lambda r: r["report"]["r_close"].get("rank_ic_mean", float("-inf")),
        reverse=True,
    )
    return results
```

In `tests/test_sweep.py`, fix the stray assignment from Step 1 (`config = sample_config_value = sweep.sample_config(trial)` → `config = sweep.sample_config(trial)`).

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_sweep.py -v`
Expected: PASS (three pure-helper tests).

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check src/ophir/sweep.py tests/test_sweep.py
uv run mypy src/ophir
git add src/ophir/sweep.py tests/test_sweep.py
git commit -m "Add Optuna sweep harness (search space, pruning, confirm)"
```

(If mypy reports `Cannot find implementation or library stub for module named "optuna"`, complete Task 5 Step 3's override edit first, then re-run.)

---

### Task 5: CLI command, dependency, docs

Wire `ophir sweep` into the CLI (thin wrapper, lazy `optuna`/`sweep` import — mirrors `serve`), add the `optuna` dependency, and update CHANGELOG + CLAUDE.md.

**Files:**
- Modify: `pyproject.toml` (deps; mypy override if needed)
- Modify: `src/ophir/cli.py` (new `sweep` command)
- Modify: `CHANGELOG.md`, `CLAUDE.md`
- Test: `tests/test_cli.py` (create)

**Interfaces:**
- Consumes: `sweep.run_sweep`, `sweep.confirm_top`.
- Produces: `ophir sweep [--trials N] [--study NAME] [--storage URL] [--confirm-top K] [--proxy-steps N] [--base-seed N] [data/window opts…]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli.py`:

```python
"""Smoke tests for the Typer CLI wiring (no command execution)."""

from typer.testing import CliRunner

from ophir.cli import app

runner = CliRunner()


def test_sweep_command_is_registered() -> None:
    names = {cmd.name for cmd in app.registered_commands}
    assert "sweep" in names


def test_sweep_help_lists_key_options() -> None:
    result = runner.invoke(app, ["sweep", "--help"])
    assert result.exit_code == 0
    assert "--trials" in result.output
    assert "--confirm-top" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — `assert "sweep" in names` (command not registered).

- [ ] **Step 3: Add the `optuna` dependency**

```bash
uv add optuna
```

Confirm `pyproject.toml`'s `[project].dependencies` now lists `optuna` (alphabetically near the others). Then run mypy:

```bash
uv run mypy src/ophir
```

If — and only if — mypy reports a missing stub for `optuna`, add it to the override list in `pyproject.toml` (the `module = [...]` block at line ~100):

```toml
module = [
    "massive.*",
    "optuna.*",
    "plotly.*",
    "tqdm.*",
    "yfinance.*",
]
```

- [ ] **Step 4: Add the `sweep` command to `cli.py`**

In `src/ophir/cli.py`, after the `dashboard` command, add (lazy-imports `sweep`/`optuna`, mirroring `serve`):

```python
@app.command()
def sweep(
    trials: int = typer.Option(50, help="Number of proxy trials to run"),
    study: str = typer.Option("ophir-sweep", help="Optuna study name (resumed if it exists)"),
    storage: str | None = typer.Option(None, help="Optuna storage URL; defaults to a SQLite db under the model dir"),
    confirm_top: int = typer.Option(5, help="Retrain and eval the top-K configs at full budget (0 to skip)"),
    proxy_steps: int = typer.Option(2000, help="max_steps per proxy trial"),
    proxy_val_batches: int = typer.Option(20, help="Validation batches per proxy validation pass"),
    full_steps: int = typer.Option(20000, help="max_steps for the confirm-phase full runs"),
    val_batches: int = typer.Option(50, help="Validation batches for the confirm-phase eval"),
    base_seed: int = typer.Option(0, help="Base seed; trial N uses base_seed + N"),
    seq_len: int = typer.Option(365, help="Window length (fixed across the sweep)"),
    offset: int = typer.Option(90, help="Window stride"),
    response_size: int = typer.Option(90, help="Forecast horizon"),
    batch_size: int = typer.Option(32, help="Batch size"),
    use_sp500: bool = typer.Option(False, help="Restrict to S&P 500 symbols"),
    data_dir: str | None = typer.Option(None, help="Override the data directory"),
) -> None:
    """Run an Optuna hyperparameter sweep, then confirm the best configs.

    Each proxy trial runs a reduced-budget training scored on ``val_rank_ic``;
    Optuna's ASHA pruner kills unpromising trials early. The study is persisted
    to SQLite and resumed if ``--study`` already exists. After the search, the
    top ``--confirm-top`` configs are retrained at full budget and scored with
    the offline eval report. Requires CUDA.
    """
    import os

    from ophir import register, sweep as sweep_mod

    if storage is None:
        storage = f"sqlite:///{os.path.join(register.MODEL_DIR, study + '.db')}"

    shared = {
        "seq_len": seq_len,
        "offset": offset,
        "response_size": response_size,
        "batch_size": batch_size,
        "use_sp500": use_sp500,
        "data_dir": data_dir,
    }
    proxy_kwargs = {**shared, "max_steps": proxy_steps, "val_batches": proxy_val_batches}

    study_obj = sweep_mod.run_sweep(
        n_trials=trials,
        study_name=study,
        storage=storage,
        base_seed=base_seed,
        proxy_kwargs=proxy_kwargs,
    )
    typer.echo(f"Best proxy val_rank_ic: {study_obj.best_value:.5f}")
    typer.echo(f"Best params: {study_obj.best_params}")

    if confirm_top > 0:
        from ophir.evaluate import format_report

        full_kwargs = {
            **shared,
            "max_steps": full_steps,
            "num_workers": 4,
            "cache_size": 8,
            "min_volume": 1000.0,
            "train_min_year": None,
            "train_max_year": 2023,
            "val_min_year": 2024,
            "val_max_year": None,
            "use_quality_allowlist": False,
            "clean_rows": False,
            "max_abs_r_close": 0.75,
            "epochs": 10,
            "window_sample": 256,
            "val_every_steps": 500,
        }
        results = sweep_mod.confirm_top(
            study_obj, k=confirm_top, full_kwargs=full_kwargs, val_batches=val_batches
        )
        for rank, record in enumerate(results, start=1):
            typer.echo(f"\n## Rank {rank}: {record['config']}")
            typer.echo(format_report({"confirm": record["report"]}))
```

Confirm `register.MODEL_DIR` is importable (it is a module-level constant in `register.py`). `str | None` Typer options are valid (the existing `dashboard` command uses `model_dir: str | None`).

- [ ] **Step 5: Run the CLI tests**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS.

- [ ] **Step 6: Update CHANGELOG and CLAUDE.md**

In `CHANGELOG.md`, add an entry under the current unreleased/next section matching the existing format, e.g.:

```markdown
- Add `ophir sweep`: an Optuna hyperparameter sweep harness that searches
  optimizer, loss-weight, and architecture-tier hyperparameters by mean
  cross-sectional rank-IC on `r_close`, with proxy-budget search (ASHA pruning,
  resumable SQLite study) and a full-budget confirm phase. Exposes the
  previously-buried `rezero_lr`, `betas`, and loss-weight knobs on `ophir train`
  and adds an opt-in `val_rank_ic` validation metric.
```

In `CLAUDE.md`, add `sweep.py` to the module map table and note the `train.py`/`run_training` engine split (one row each), keeping the existing terse style.

- [ ] **Step 7: Full verification and commit**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/ophir
uv run pytest
git add pyproject.toml uv.lock src/ophir/cli.py tests/test_cli.py CHANGELOG.md CLAUDE.md
git commit -m "Wire ophir sweep CLI command and add optuna dependency"
```

---

## Self-Review

**Spec coverage:**
- Optuna harness, TPE + ASHA, SQLite/resumable → Task 4 (`run_sweep`), Task 5 (CLI storage/resume). ✓
- Objective = mean cross-sectional rank-IC on `r_close`; `val_loss`/skill secondary → Task 2 (`val_rank_ic`), Task 4 objective returns it; confirm phase reports the full eval (val_loss + skill via `evaluate_model`/`format_report`). ✓
- `val_rank_ic` plumbing reusing `rank_ic` → Task 2. ✓
- Expose `rezero_lr`, `betas`, `upside_weight`, `downside_weight`, defaults unchanged → Tasks 1 & 3. ✓
- Search space: optimizer/schedule + loss weights + arch tiers; windowing fixed → Task 4 `sample_config` + Task 5 fixed shared kwargs. ✓
- Proxy → confirm strategy → Task 4 (`objective` proxy, `confirm_top`) + Task 5 wiring. ✓
- Per-trial seed `base_seed + trial.number` → Task 4 `objective`. ✓
- Backward-compat byte-for-byte → opt-in `val_identity`/`val_rank_ic`, default-preserving params (Tasks 1–3). ✓
- Tests CPU-safe/network-free; CUDA mocked → Tasks 1–5 test the pure helpers + monkeypatched engine. ✓
- Optuna dep + mypy override contingency → Task 5. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code; test bodies are concrete. The one conditional (mypy `optuna` override) provides the exact block to add and the exact trigger. ✓

**Type consistency:** `run_training` returns `LightningOHLCPredictor` (Task 3) and is called by `objective`/`confirm_top` (Task 4). `sample_config` returns kwargs consumed by `run_training` (keys match: `emb_dim`, `num_layers`, `num_heads`, `lr`, `rezero_lr`, `weight_decay`, `warmup_ratio`, `loss_decay`, `betas`, `upside_weight`, `downside_weight`). `select_top_configs`/`confirm_top` read `trial.user_attrs["config"]` set in `objective`. `_OptunaPruning` monitors `val_rank_ic`, the exact key logged in Task 2. `fetch_base_trainer(extra_callbacks=…)` (Task 3) consumed by `run_training`. ✓

## Notes on what is NOT covered by automated tests

The real proxy/confirm runs (`objective`, `run_sweep`, `confirm_top`, and the `sweep` CLI body past argument parsing) execute CUDA training and cannot run in CI. They are covered by: strict mypy over `src/ophir`, ruff, the pure-helper unit tests (`sample_config`, `select_top_configs`, `val_rank_ic`, hparam forwarding via monkeypatched fit), and CLI registration/help smoke tests. End-to-end validation requires a GPU box and is a manual step (run `ophir sweep --trials 4 --confirm-top 1 --proxy-steps 50` against real data).
