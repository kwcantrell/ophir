# ReZero Gate-Opening Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add gate instrumentation and two opt-in architecture knobs (`rezero_init`, `decouple_rezero_schedule`) plus an analysis helper, so a fair-budget experiment matrix can determine whether the transformer depth is actually helping.

**Architecture:** Small, additive changes across `models.py` (a pure gate-stats helper + a configurable ReZero init), `training_models.py` (opt-in gate logging + a per-group LR schedule), `train.py` (thread three new knobs to the CLI), and `dashboard.py` (a CSV-reading summary helper). Every knob defaults to current behavior. A runbook documents the five experiment arms; the GPU runs are the user's to launch.

**Tech Stack:** Python 3.10+ (runtime floor) / py312 lint target, PyTorch, PyTorch-Lightning, transformers (cosine schedule), pandas (lazy), Typer, pytest.

## Global Constraints

- **Live code is `src/ophir/` only.** Ignore top-level `ophir/`, `oldcode/`, `old_*.py`.
- **Strict typing.** `mypy --strict` over `src/ophir`; `warn_unused_ignores = true` — a stray `# type: ignore` fails mypy. Annotate all new functions. Third-party `pandas` import uses `# type: ignore[import-untyped]` (existing pattern in `dashboard.py`).
- **Tests are CPU-safe, seeded, network-free.** No CUDA, no training, no network. The model constructs on CPU; the CUDA-only forward path is never invoked in tests.
- **pytest runs with `filterwarnings = ["error", …]`** — code and tests must run warning-clean.
- **Ruff target `py312`, mypy `python_version = 3.10`** stay split.
- **Default training must be byte-for-byte unchanged when the new knobs are unset** (`rezero_init=0.0`, `decouple_rezero_schedule=False`, `log_rezero_gates=False`).
- **Do not regress the forecast-masking contract** (`_apply_response_mask`, pinned by `tests/test_models_leakage.py`).
- **Style:** comments only for non-obvious *why*; no removed-code stubs or back-compat shims; CHANGELOG entries match existing format.
- **Validate after each task:** `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy src/ophir`, `uv run pytest`. (pytest now runs without the CUDA `LD_LIBRARY_PATH` workaround — the venv was repaired.)

---

## Task 1: `rezero_gate_stats` helper

**Files:**
- Modify: `src/ophir/models.py` (add a module-level helper near `pool_prefix_embedding`, ~line 379)
- Test: `tests/test_models_output.py`

**Interfaces:**
- Produces: `rezero_gate_stats(model: nn.Module) -> dict[str, float | list[float]]` returning `{"mean_abs": float, "max_abs": float, "per_layer": list[float]}` over every parameter whose name contains `"rezero"`. Empty model → `{"mean_abs": 0.0, "max_abs": 0.0, "per_layer": []}`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_models_output.py`:

```python
def test_rezero_gate_stats_aggregates_per_layer():
    import torch.nn as nn

    from ophir.models import OHLCMulitClassParameters, OHLCMulitClassPredictor, rezero_gate_stats

    torch.manual_seed(0)
    model = OHLCMulitClassPredictor(
        OHLCMulitClassParameters(emb_dim=16, num_layers=3, num_heads=2)
    )
    # Force known gate values.
    vals = [0.1, -0.2, 0.3]
    for block, v in zip(model.encoder, vals):
        with torch.no_grad():
            block._rezero.fill_(v)
    stats = rezero_gate_stats(model)
    assert stats["per_layer"] == [0.1, -0.2, 0.3]
    assert abs(stats["max_abs"] - 0.3) < 1e-6
    assert abs(stats["mean_abs"] - 0.2) < 1e-6  # mean(|0.1|,|0.2|,|0.3|)


def test_rezero_gate_stats_empty_is_zero():
    import torch.nn as nn

    from ophir.models import rezero_gate_stats

    assert rezero_gate_stats(nn.Linear(2, 2)) == {"mean_abs": 0.0, "max_abs": 0.0, "per_layer": []}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_models_output.py -k rezero_gate_stats -v`
Expected: FAIL — `cannot import name 'rezero_gate_stats'`.

- [ ] **Step 3: Implement the helper**

In `src/ophir/models.py`, add after `pool_prefix_embedding` (~line 379):

```python
def rezero_gate_stats(model: nn.Module) -> dict[str, float | list[float]]:
    """Per-layer and aggregate magnitudes of the ReZero gate scalars.

    Reads every parameter whose name contains ``"rezero"`` (one scalar per
    :class:`TransformerBlock`) and reports the raw per-layer values plus the mean
    and max of their absolute values. Used to see whether the residual gates have
    opened during training. Returns zeros for a model with no such parameters.
    """
    per_layer = [float(p) for name, p in model.named_parameters() if "rezero" in name]
    if not per_layer:
        return {"mean_abs": 0.0, "max_abs": 0.0, "per_layer": []}
    abs_vals = [abs(v) for v in per_layer]
    return {
        "mean_abs": sum(abs_vals) / len(abs_vals),
        "max_abs": max(abs_vals),
        "per_layer": per_layer,
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_models_output.py -k rezero_gate_stats -v`
Expected: PASS.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check src/ophir/models.py tests/test_models_output.py
uv run mypy src/ophir
git add src/ophir/models.py tests/test_models_output.py
git commit -m "Add rezero_gate_stats helper for inspecting ReZero gates"
```

---

## Task 2: Configurable `rezero_init`

**Files:**
- Modify: `src/ophir/models.py` (`OHLCMulitClassParameters` ~48-60; `TransformerBlock.__init__` ~322)
- Modify: `src/ophir/training_models.py` (`__init__` ~67-140; `reset_rezero` ~496-501)
- Test: `tests/test_models_output.py`, `tests/test_training_models.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `OHLCMulitClassParameters(..., rezero_init: float = 0.0)`; `TransformerBlock` initializes `_rezero` to `hparams.rezero_init`; `LightningOHLCPredictor(..., rezero_init: float = 0.0)` stores `self.rezero_init`, passes it into the params, and `reset_rezero` fills with it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_models_output.py`:

```python
def test_rezero_init_sets_gate_values():
    from ophir.models import OHLCMulitClassParameters, OHLCMulitClassPredictor, rezero_gate_stats

    model = OHLCMulitClassPredictor(
        OHLCMulitClassParameters(emb_dim=16, num_layers=2, num_heads=2, rezero_init=0.1)
    )
    assert rezero_gate_stats(model)["per_layer"] == [0.1, 0.1]


def test_rezero_init_defaults_to_zero():
    from ophir.models import OHLCMulitClassParameters, OHLCMulitClassPredictor, rezero_gate_stats

    model = OHLCMulitClassPredictor(OHLCMulitClassParameters(emb_dim=16, num_layers=2, num_heads=2))
    assert rezero_gate_stats(model)["per_layer"] == [0.0, 0.0]
```

Append to `tests/test_training_models.py`:

```python
def test_reset_rezero_restores_configured_init() -> None:
    import torch

    model = LightningOHLCPredictor(emb_dim=16, num_layers=2, num_heads=2, rezero_init=0.1)
    for block in model.ohlc_predictor.encoder:
        with torch.no_grad():
            block._rezero.fill_(0.5)
    model.reset_rezero()
    for block in model.ohlc_predictor.encoder:
        assert abs(float(block._rezero) - 0.1) < 1e-6
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_models_output.py -k rezero_init tests/test_training_models.py -k reset_rezero -v`
Expected: FAIL — `OHLCMulitClassParameters.__init__() got an unexpected keyword argument 'rezero_init'` / reset restores `0.0`.

- [ ] **Step 3: Add `rezero_init` to the parameters dataclass**

In `src/ophir/models.py`, add the field to `OHLCMulitClassParameters` (after `num_heads`, ~line 59):

```python
    emb_dim: int
    num_layers: int
    num_heads: int
    rezero_init: float = 0.0
```

- [ ] **Step 4: Use it in `TransformerBlock`**

In `src/ophir/models.py`, replace the hardcoded init (~line 322):

```python
        self._rezero = nn.Parameter(torch.tensor(0.0, dtype=torch.float))
```

with:

```python
        self._rezero = nn.Parameter(torch.tensor(hparams.rezero_init, dtype=torch.float))
```

- [ ] **Step 5: Thread it through the Lightning wrapper**

In `src/ophir/training_models.py` `__init__`, add the parameter (after `num_heads`, ~line 71):

```python
        num_heads: int,
        rezero_init: float = 0.0,
        lr: float = 2e-4,
```

Add a docstring entry (near the other arch params):

```python
        rezero_init : float, optional
            Initial value for every ReZero gate scalar. ``0.0`` (default) starts
            each block as the identity; a positive value starts the residual
            branches partially open. An experiment knob for the depth diagnostic.
```

Pass it into the params construction (~line 125):

```python
        hparams: OHLCMulitClassParameters = OHLCMulitClassParameters(
            emb_dim=emb_dim, num_layers=num_layers, num_heads=num_heads, rezero_init=rezero_init
        )
```

Store it (next to the other assignments, ~line 130):

```python
        self.rezero_init = rezero_init
```

Update `reset_rezero` (~line 496) to fill with the configured init:

```python
    def reset_rezero(self) -> None:
        """Reset every ReZero scalar to the configured ``rezero_init``, in place."""
        with torch.no_grad():
            for name, param in self.ohlc_predictor.named_parameters():
                if "rezero" in name:
                    param.fill_(self.rezero_init)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_models_output.py -k rezero_init tests/test_training_models.py -k reset_rezero tests/test_models_leakage.py -v`
Expected: PASS (leakage tests still pass — `rezero_init` defaults to 0.0).

- [ ] **Step 7: Lint, type-check, commit**

```bash
uv run ruff check src/ophir/models.py src/ophir/training_models.py tests/
uv run mypy src/ophir
git add src/ophir/models.py src/ophir/training_models.py tests/test_models_output.py tests/test_training_models.py
git commit -m "Add configurable rezero_init for the ReZero gates"
```

---

## Task 3: Opt-in gate logging (`log_rezero_gates`)

**Files:**
- Modify: `src/ophir/training_models.py` (`__init__` ~67-140; `on_validation_epoch_end` ~405-422)
- Test: `tests/test_training_models.py`

**Interfaces:**
- Consumes: `rezero_gate_stats` from Task 1.
- Produces: `LightningOHLCPredictor(..., log_rezero_gates: bool = False)`; when set, `on_validation_epoch_end` logs scalars `rezero_mean_abs` and `rezero_max_abs`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_training_models.py`:

```python
def test_log_rezero_gates_logs_when_enabled() -> None:
    import torch

    logged: dict[str, float] = {}
    model = LightningOHLCPredictor(
        emb_dim=16, num_layers=2, num_heads=2, rezero_init=0.1, log_rezero_gates=True
    )
    model.log = lambda name, value, **kw: logged.__setitem__(name, float(value))  # type: ignore[method-assign]
    # Bypass the val_rank_ic branch (needs a trainer); it is guarded by empty buffers.
    model.on_validation_epoch_end()
    assert abs(logged["rezero_mean_abs"] - 0.1) < 1e-6
    assert abs(logged["rezero_max_abs"] - 0.1) < 1e-6


def test_log_rezero_gates_silent_when_disabled() -> None:
    logged: dict[str, float] = {}
    model = LightningOHLCPredictor(emb_dim=16, num_layers=2, num_heads=2)
    model.log = lambda name, value, **kw: logged.__setitem__(name, float(value))  # type: ignore[method-assign]
    model.on_validation_epoch_end()
    assert "rezero_mean_abs" not in logged
```

> Note: `on_validation_epoch_end`'s existing `val_rank_ic` branch only runs when `self._val_ic_buffers["pred"]` is non-empty (it is empty on a fresh model), so the test reaches the gate-logging block without a trainer.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_training_models.py -k log_rezero_gates -v`
Expected: FAIL — unexpected kwarg `log_rezero_gates` / `rezero_mean_abs` not logged.

- [ ] **Step 3: Add the flag**

In `src/ophir/training_models.py` `__init__`, add the parameter (after `downside_weight`, ~line 122):

```python
        downside_weight: float = 0.5,
        log_rezero_gates: bool = False,
```

Docstring entry:

```python
        log_rezero_gates : bool, optional
            When ``True``, log ``rezero_mean_abs`` / ``rezero_max_abs`` each
            validation pass so the gate magnitudes are visible. Defaults to
            ``False`` (default CSV columns unchanged).
```

Store it (~line 138):

```python
        self.log_rezero_gates = log_rezero_gates
```

Add the import at the top of the file (with the other `.models` import, ~line 18):

```python
from .models import OHLCMulitClassParameters, OHLCMulitClassPredictor, rezero_gate_stats
```

- [ ] **Step 4: Log the gate stats**

In `on_validation_epoch_end` (~line 405), add at the start of the method body (before the `preds = …` block):

```python
        if self.log_rezero_gates:
            stats = rezero_gate_stats(self.ohlc_predictor)
            self.log("rezero_mean_abs", stats["mean_abs"], on_epoch=True, logger=True)
            self.log("rezero_max_abs", stats["max_abs"], on_epoch=True, logger=True)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_training_models.py -k log_rezero_gates -v`
Expected: PASS.

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff check src/ophir/training_models.py tests/test_training_models.py
uv run mypy src/ophir
git add src/ophir/training_models.py tests/test_training_models.py
git commit -m "Add opt-in ReZero gate-magnitude logging"
```

---

## Task 4: Decoupled ReZero LR schedule (`decouple_rezero_schedule`)

**Files:**
- Modify: `src/ophir/training_models.py` (add `_cosine_factor`/`_flat_factor` module-level fns; `__init__` ~67-140; `configure_optimizers` ~448-494)
- Test: `tests/test_training_models.py`

**Interfaces:**
- Produces:
  - `_cosine_factor(step: int, warmup: int, total: int) -> float` — replicates the transformers cosine-with-warmup LR factor (num_cycles=0.5).
  - `_flat_factor(step: int, warmup: int) -> float` — linear warmup then constant `1.0` (no decay).
  - `LightningOHLCPredictor(..., decouple_rezero_schedule: bool = False)`; when `True`, `configure_optimizers` returns a per-group `LambdaLR` with `[cosine, cosine, flat]` factors (the rezero group is index 2).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_training_models.py`:

```python
def test_lr_factor_helpers() -> None:
    from ophir.training_models import _cosine_factor, _flat_factor

    # Warmup ramps linearly for both.
    assert abs(_cosine_factor(0, 10, 100) - 0.0) < 1e-9
    assert abs(_cosine_factor(5, 10, 100) - 0.5) < 1e-9
    assert abs(_flat_factor(5, 10) - 0.5) < 1e-9
    # End of training: cosine decays to ~0, flat stays at 1.0.
    assert _cosine_factor(100, 10, 100) < 1e-6
    assert abs(_flat_factor(100, 10) - 1.0) < 1e-9
    # Start of decay (just past warmup): cosine ~1.0.
    assert abs(_cosine_factor(10, 10, 100) - 1.0) < 1e-6


def test_decoupled_schedule_keeps_rezero_flat() -> None:
    import torch

    model = LightningOHLCPredictor(
        emb_dim=16, num_layers=2, num_heads=2, warmup_ratio=0.1, decouple_rezero_schedule=True
    )
    # configure_optimizers calls self._total_training_steps() (which reads the
    # trainer); override it so the test needs no Trainer.
    model._total_training_steps = lambda: 100  # type: ignore[method-assign]
    cfg = model.configure_optimizers()
    sched = cfg["lr_scheduler"]["scheduler"]
    opt = cfg["optimizer"]
    base = [g["lr"] for g in opt.param_groups]
    for _ in range(100):
        opt.step()
        sched.step()
    final = [g["lr"] for g in opt.param_groups]
    # Groups 0/1 (cosine) have decayed to ~0; group 2 (rezero, flat) holds its base lr.
    assert final[0] < base[0] * 0.05
    assert abs(final[2] - base[2]) < base[2] * 0.05
```

> The rezero group is the third AdamW group (index 2) in `configure_optimizers`, and it carries its own `lr=self.rezero_lr`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_training_models.py -k "lr_factor or decoupled_schedule" -v`
Expected: FAIL — `cannot import name '_cosine_factor'` / unexpected kwarg.

- [ ] **Step 3: Add the factor helpers**

In `src/ophir/training_models.py`, add at module level (below `robust_scale`, ~line 33):

```python
def _cosine_factor(step: int, warmup: int, total: int) -> float:
    """Cosine-with-warmup LR factor matching ``get_cosine_schedule_with_warmup``."""
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def _flat_factor(step: int, warmup: int) -> float:
    """Linear warmup then a constant ``1.0`` (no decay) — for the ReZero gates."""
    if step < warmup:
        return step / max(1, warmup)
    return 1.0
```

(`math` is already imported at the top of the file.)

- [ ] **Step 4: Add the flag and the per-group schedule**

In `__init__`, add the parameter (after `log_rezero_gates`):

```python
        log_rezero_gates: bool = False,
        decouple_rezero_schedule: bool = False,
```

Docstring entry:

```python
        decouple_rezero_schedule : bool, optional
            When ``True``, the ReZero param group uses a flat (warmup-then-
            constant) LR while the other groups keep the cosine decay, so the
            gates keep growing late in training. Defaults to ``False`` (single
            cosine schedule over all groups, unchanged).
```

Store it (~line 138):

```python
        self.decouple_rezero_schedule = decouple_rezero_schedule
```

In `configure_optimizers`, replace the scheduler construction (~line 482-486):

```python
        total_steps = self._total_training_steps()
        warmup_steps = int(self.warmup_ratio * total_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, num_training_steps=total_steps
        )
```

with:

```python
        total_steps = self._total_training_steps()
        warmup_steps = int(self.warmup_ratio * total_steps)
        if self.decouple_rezero_schedule:
            from functools import partial

            cosine = partial(_cosine_factor, warmup=warmup_steps, total=total_steps)
            flat = partial(_flat_factor, warmup=warmup_steps)
            scheduler: object = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=[cosine, cosine, flat]
            )
        else:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, warmup_steps, num_training_steps=total_steps
            )
```

> The `lr_lambda` list has one entry per AdamW param group, in declaration order: decay (0), no_decay (1), rezero (2). Keep this aligned with the group order in `configure_optimizers`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_training_models.py -k "lr_factor or decoupled_schedule" -v`
Expected: PASS.

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff check src/ophir/training_models.py tests/test_training_models.py
uv run mypy src/ophir
git add src/ophir/training_models.py tests/test_training_models.py
git commit -m "Add opt-in decoupled (flat) LR schedule for the ReZero gates"
```

---

## Task 5: Thread the three knobs through `run_training` and `train`

**Files:**
- Modify: `src/ophir/train.py` (`run_training` ~316-353 + model construction ~416-429; `train` ~450-486 + call ~494-529)
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `LightningOHLCPredictor(..., rezero_init=…, log_rezero_gates=…, decouple_rezero_schedule=…)` from Tasks 2-4.
- Produces: `run_training(..., rezero_init=0.0, log_rezero_gates=False, decouple_rezero_schedule=False, …)` forwards all three to the model; `train` exposes them as CLI flags and forwards to `run_training`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_train.py`, mirroring the existing `test_run_training_forwards_close_weight` exactly (it uses the `patched_engine` fixture + a `_CapturingPredictor` subclass that records constructor kwargs):

```python
def test_run_training_forwards_rezero_knobs(
    monkeypatch: pytest.MonkeyPatch, patched_engine: _FakeTrainer
) -> None:
    captured: dict[str, Any] = {}

    import ophir.training_models as tm

    _orig_predictor = tm.LightningOHLCPredictor

    class _CapturingPredictor(_orig_predictor):  # type: ignore[misc, valid-type]
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(tm, "LightningOHLCPredictor", _CapturingPredictor)
    train.run_training(
        emb_dim=16,
        num_layers=1,
        num_heads=2,
        rezero_init=0.1,
        log_rezero_gates=True,
        decouple_rezero_schedule=True,
    )
    assert captured["rezero_init"] == 0.1
    assert captured["log_rezero_gates"] is True
    assert captured["decouple_rezero_schedule"] is True
```

> `patched_engine`, `_FakeTrainer`, and the `_CapturingPredictor` idiom already exist in this file (added for the `close_weight` test). The `# type: ignore[misc, valid-type]` on the subclass is required and correct — match it verbatim.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_train.py -k rezero_knobs -v`
Expected: FAIL — unexpected keyword argument `rezero_init`.

- [ ] **Step 3: Add the knobs to `run_training`**

In `src/ophir/train.py` `run_training` signature, add (after `downside_weight`, ~line 349):

```python
    downside_weight: float = 0.5,
    rezero_init: float = 0.0,
    log_rezero_gates: bool = False,
    decouple_rezero_schedule: bool = False,
```

Pass them into the `LightningOHLCPredictor(...)` construction (after `downside_weight=…`, ~line 428):

```python
        downside_weight=downside_weight,
        rezero_init=rezero_init,
        log_rezero_gates=log_rezero_gates,
        decouple_rezero_schedule=decouple_rezero_schedule,
```

- [ ] **Step 4: Add the knobs to the `train` CLI and forward them**

In `train` signature, add (after `downside_weight`, ~line 483):

```python
    downside_weight: float = 0.5,
    rezero_init: float = 0.0,
    log_rezero_gates: bool = False,
    decouple_rezero_schedule: bool = False,
```

In `train`'s call to `run_training`, add (after `downside_weight=…`, ~line 526):

```python
        downside_weight=downside_weight,
        rezero_init=rezero_init,
        log_rezero_gates=log_rezero_gates,
        decouple_rezero_schedule=decouple_rezero_schedule,
```

- [ ] **Step 5: Run the test + a CLI help smoke check**

Run: `uv run pytest tests/test_train.py -k rezero_knobs -v`
Expected: PASS.

Run: `uv run ophir train --help`
Expected: lists `--rezero-init`, `--log-rezero-gates`, `--decouple-rezero-schedule`; exit 0.

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff check src/ophir/train.py tests/test_train.py
uv run mypy src/ophir
git add src/ophir/train.py tests/test_train.py
git commit -m "Thread rezero_init, gate logging, and decoupled schedule through the CLI"
```

---

## Task 6: `summarize_rezero_runs` analysis helper

**Files:**
- Modify: `src/ophir/dashboard.py` (add a helper near `_latest_metrics_csv`)
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Produces: `summarize_rezero_runs(versions: dict[str, str]) -> "pandas.DataFrame"` — `versions` maps an arm label to a `version_*` directory; reads each `metrics.csv` and returns one row per arm with columns `arm`, `val_rank_ic`, `rezero_mean_abs`, `rezero_max_abs` (each the last non-NaN value in its column, or `NaN` if absent).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dashboard.py`:

```python
def test_summarize_rezero_runs_tabulates_final_values(tmp_path: Path) -> None:
    def _arm(name: str, body: str) -> str:
        d = tmp_path / name / "version_0"
        d.mkdir(parents=True)
        (d / "metrics.csv").write_text(body)
        return str(d)

    versions = {
        "shallow": _arm(
            "shallow",
            "step,val_rank_ic,rezero_mean_abs,rezero_max_abs\n10,0.02,,\n20,0.03,,\n",
        ),
        "deep_open": _arm(
            "deep_open",
            "step,val_rank_ic,rezero_mean_abs,rezero_max_abs\n10,0.05,0.4,0.6\n20,0.07,0.5,0.7\n",
        ),
    }
    df = dashboard.summarize_rezero_runs(versions)
    row = df.set_index("arm")
    assert abs(row.loc["deep_open", "val_rank_ic"] - 0.07) < 1e-9
    assert abs(row.loc["deep_open", "rezero_mean_abs"] - 0.5) < 1e-9
    assert abs(row.loc["shallow", "val_rank_ic"] - 0.03) < 1e-9
    # shallow logged no gate stats -> NaN
    assert row.loc["shallow", "rezero_mean_abs"] != row.loc["shallow", "rezero_mean_abs"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_dashboard.py -k summarize_rezero -v`
Expected: FAIL — `module 'ophir.dashboard' has no attribute 'summarize_rezero_runs'`.

- [ ] **Step 3: Implement the helper**

In `src/ophir/dashboard.py`, add after `_latest_metrics_csv` (~line 60):

```python
def summarize_rezero_runs(versions: dict[str, str]) -> "pd.DataFrame":
    """Tabulate the final val_rank_ic and ReZero gate magnitudes per arm.

    ``versions`` maps an arm label to its ``CSVLogger`` ``version_*`` directory.
    Each ``metrics.csv`` is read and the last non-NaN value of ``val_rank_ic``,
    ``rezero_mean_abs``, and ``rezero_max_abs`` is taken (``NaN`` if a column is
    absent). Returns one row per arm for side-by-side comparison.
    """
    import pandas as pd  # type: ignore[import-untyped]

    cols = ["val_rank_ic", "rezero_mean_abs", "rezero_max_abs"]
    rows = []
    for arm, version_dir in versions.items():
        path = os.path.join(version_dir, "metrics.csv")
        df = pd.read_csv(path)
        row: dict[str, object] = {"arm": arm}
        for col in cols:
            series = df[col].dropna() if col in df.columns else pd.Series(dtype="float64")
            row[col] = float(series.iloc[-1]) if not series.empty else float("nan")
        rows.append(row)
    return pd.DataFrame(rows, columns=["arm", *cols])
```

Add the `TYPE_CHECKING` import so the annotation resolves under strict mypy. In the existing `if TYPE_CHECKING:` block (~line 29), add:

```python
    import pandas as pd
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_dashboard.py -k summarize_rezero -v`
Expected: PASS.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check src/ophir/dashboard.py tests/test_dashboard.py
uv run mypy src/ophir
git add src/ophir/dashboard.py tests/test_dashboard.py
git commit -m "Add summarize_rezero_runs to compare diagnostic arms"
```

---

## Task 7: Runbook, CHANGELOG, and final full-suite check

**Files:**
- Create: `docs/rezero-diagnostic-runbook.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Write the experiment runbook**

Create `docs/rezero-diagnostic-runbook.md`:

````markdown
# ReZero gate-opening diagnostic — runbook

Determines whether the transformer depth is helping, by comparing a near-linear
baseline against deep models with the ReZero gates forced open. Requires CUDA.
All arms use the base tier (`emb_dim=128`, `num_heads=8`), `--seed 0`, a fixed
10,000-step budget, and log both `val_rank_ic` and the ReZero gate magnitudes.

Run each arm (each creates a new `csv-logger/version_N` under the model dir —
record which version is which):

```bash
# A. Shallow baseline (near-linear floor)
ophir train --emb-dim 128 --num-heads 8 --num-layers 1 \
  --max-steps 10000 --seed 0 --val-identity --log-rezero-gates

# B. Deep, default (gates expected to stay closed)
ophir train --emb-dim 128 --num-heads 8 --num-layers 6 \
  --max-steps 10000 --seed 0 --val-identity --log-rezero-gates

# C. Deep, high rezero_lr (open gates via LR)
ophir train --emb-dim 128 --num-heads 8 --num-layers 6 --rezero-lr 3e-3 \
  --max-steps 10000 --seed 0 --val-identity --log-rezero-gates

# D. Deep, non-zero init (gates start open)
ophir train --emb-dim 128 --num-heads 8 --num-layers 6 --rezero-init 0.1 \
  --max-steps 10000 --seed 0 --val-identity --log-rezero-gates

# E. Deep, un-decayed rezero schedule (gates keep growing)
ophir train --emb-dim 128 --num-heads 8 --num-layers 6 --decouple-rezero-schedule \
  --max-steps 10000 --seed 0 --val-identity --log-rezero-gates
```

Then compare (point each label at its `version_N` directory under
`<model_dir>/csv-logger/`):

```python
from ophir.dashboard import summarize_rezero_runs

print(summarize_rezero_runs({
    "A_shallow":   "<model_dir>/csv-logger/version_0",
    "B_deep":      "<model_dir>/csv-logger/version_1",
    "C_high_lr":   "<model_dir>/csv-logger/version_2",
    "D_init":      "<model_dir>/csv-logger/version_3",
    "E_undecayed": "<model_dir>/csv-logger/version_4",
}))
```

**Verdict:** if any deep arm (C/D/E) beats the shallow floor (A) on
`val_rank_ic` by a meaningful margin, depth helps once the gates open. If not,
the secondary `rezero_mean_abs` column separates "gates opened but depth didn't
help" (depth genuinely inert) from "gates never opened" (re-run at a longer
budget before concluding).
````

- [ ] **Step 2: Inspect the CHANGELOG format**

Run: `sed -n '1,20p' CHANGELOG.md`
Expected: Keep-a-Changelog `## [Unreleased]` with `### Added` / `### Changed`.

- [ ] **Step 3: Add a CHANGELOG entry**

Under `## [Unreleased] / ### Added`, add (match the existing bullet style):

```
- ReZero depth diagnostic: opt-in `rezero_init`, `--decouple-rezero-schedule`,
  and `--log-rezero-gates` training knobs (all default to current behavior),
  a `rezero_gate_stats` helper, and `dashboard.summarize_rezero_runs` to compare
  experiment arms. See `docs/rezero-diagnostic-runbook.md`.
```

- [ ] **Step 4: Run the full validation suite**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/ophir
uv run pytest
```
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add docs/rezero-diagnostic-runbook.md CHANGELOG.md
git commit -m "Add ReZero diagnostic runbook and CHANGELOG entry"
```

---

## Self-Review Notes

- **Spec coverage:** Part 1 (instrumentation) → Tasks 1 + 3. Part 2a (`rezero_init`) → Task 2. Part 2b (`decouple_rezero_schedule`) → Task 4. Knob threading → Task 5. Part 4 (`summarize_rezero_runs` + orchestration) → Tasks 6 + 7. Testing section → tests embedded per task. Files-touched + CHANGELOG → Task 7.
- **Type consistency:** `rezero_gate_stats(model) -> dict[str, float | list[float]]` with keys `mean_abs`/`max_abs`/`per_layer` is produced in Task 1 and consumed in Task 3. The three knobs `rezero_init: float`, `log_rezero_gates: bool`, `decouple_rezero_schedule: bool` keep identical names/types across `LightningOHLCPredictor`, `run_training`, and `train`. `_cosine_factor`/`_flat_factor` signatures match their use in `configure_optimizers`. The rezero param group is index 2 everywhere it's referenced.
- **Default-behavior safety:** every new knob defaults to current behavior; the decoupled-schedule off-path keeps the exact `get_cosine_schedule_with_warmup` call, and `rezero_init=0.0` keeps `torch.tensor(0.0)`. Leakage contract untouched.
