# Loss-Weight Normalization & Sweep Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Normalize the three multi-target loss weights so they control task balance (not loss scale), add a clean sweep-diagnostics path (random sampler / no pruner + an importances command), and fix two minor robustness issues in `models.py`.

**Architecture:** Five focused changes across the existing modules: (A) a scale-invariant weighted-mean loss in `training_models.py` with a new `close_weight` field threaded through `train.py`; (B) a configurable sampler/pruner in `sweep.py` plus importance helpers and a CLI command in `cli.py`; (C) a padding-masked prefix pool and a `response_size` guard in `models.py`. Every change ships with CPU-safe, seeded tests.

**Tech Stack:** Python 3.10+ (runtime floor) / py312 lint target, PyTorch, PyTorch-Lightning, Optuna, Typer, pytest.

## Global Constraints

- **Live code is `src/ophir/` only.** Ignore top-level `ophir/`, `oldcode/`, `old_*.py`.
- **Strict typing.** `mypy --strict` over `src/ophir`; `warn_unused_ignores = true` — a stray `# type: ignore` fails mypy. Add type annotations to all new functions.
- **Tests are CPU-safe, seeded, network-free.** No CUDA, no Optuna optimization runs, no network. Use `FixedTrial` / `optuna.trial.create_trial` for study fixtures.
- **pytest runs with `filterwarnings = ["error", …]`** — code and tests must run warning-clean.
- **Ruff target `py312`, mypy `python_version = 3.10`** stay split — don't unify.
- **Do not regress the forecast-masking contract** (`_apply_response_mask`, pinned by `tests/test_models_leakage.py`).
- **Style:** comments only for non-obvious *why*; no removed-code stubs or back-compat shims; CHANGELOG entries match existing format.
- **Validate after each task:** `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy src/ophir`, `uv run pytest`.

---

## Task 1: Normalize the loss weights (add `close_weight`)

**Files:**
- Modify: `src/ophir/training_models.py` (`LightningOHLCPredictor.__init__` ~67-140; `compute_loss` return ~330)
- Test: `tests/test_training_models.py`

**Interfaces:**
- Produces: `LightningOHLCPredictor(..., close_weight: float = 1.0, upside_weight: float = 0.5, downside_weight: float = 0.5)`; `compute_loss` returns the **weight-normalized** mean `(cw·close + uw·upside + dw·downside) / (cw + uw + dw)`.

- [ ] **Step 1: Update the existing combine test to expect normalization**

In `tests/test_training_models.py`, replace the body of `test_loss_weights_combine_components` (the `expected = …` block, ~57-62) with:

```python
    expected = (
        logged["val_r_close_loss"]
        + 0.4 * logged["val_upside_loss"]
        + 0.7 * logged["val_downside_loss"]
    ) / (1.0 + 0.4 + 0.7)
    assert abs(float(loss) - expected) < 1e-6
```

- [ ] **Step 2: Add a scale-invariance test and a default-weight assertion**

Append to `tests/test_training_models.py`:

```python
def test_loss_is_invariant_to_uniform_weight_scaling() -> None:
    out_a = _toy_model_output()
    out_b = _toy_model_output()
    base = _build_predictor(close_weight=1.0, upside_weight=0.5, downside_weight=0.5)
    scaled = _build_predictor(close_weight=3.0, upside_weight=1.5, downside_weight=1.5)
    base.loss_state = scaled.loss_state = "val"
    base.log = lambda *a, **k: None  # type: ignore[method-assign]
    scaled.log = lambda *a, **k: None  # type: ignore[method-assign]
    loss_a = base.compute_loss(out_a)  # type: ignore[arg-type]
    loss_b = scaled.compute_loss(out_b)  # type: ignore[arg-type]
    assert abs(float(loss_a) - float(loss_b)) < 1e-6


def test_close_weight_defaults_to_one() -> None:
    model = _build_predictor()
    assert model.close_weight == 1.0
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_training_models.py -k "combine or invariant or close_weight_defaults" -v`
Expected: FAIL — `test_loss_is_invariant…` and `test_close_weight_defaults…` error on unexpected `close_weight` kwarg / missing attribute; `test_loss_weights_combine_components` fails the normalized assertion.

- [ ] **Step 4: Add the `close_weight` constructor parameter**

In `src/ophir/training_models.py`, add the parameter to `__init__` (place it immediately before `upside_weight` in the signature, ~78):

```python
        close_weight: float = 1.0,
        upside_weight: float = 0.5,
        downside_weight: float = 0.5,
```

Add its docstring entry (above the `upside_weight` entry, ~117):

```python
        close_weight : float, optional
            Weight of the r_close channel in the combined loss. The three loss
            weights are normalized by their sum, so only their ratios matter.
            Defaults to ``1.0``.
```

And store it (next to `self.upside_weight`, ~137):

```python
        self.close_weight = close_weight
        self.upside_weight = upside_weight
        self.downside_weight = downside_weight
```

- [ ] **Step 5: Normalize the combined-loss return**

In `compute_loss`, replace the final return (~330):

```python
        return close_loss + self.upside_weight * upside_loss + self.downside_weight * downside_loss
```

with:

```python
        total_weight = self.close_weight + self.upside_weight + self.downside_weight
        combined = (
            self.close_weight * close_loss
            + self.upside_weight * upside_loss
            + self.downside_weight * downside_loss
        )
        return combined / max(total_weight, 1e-8)
```

Also update the `compute_loss` docstring line that reads `combined as ``close + upside_weight·upside + downside_weight·downside``` (~252) to:

```python
        combined as a sum-normalized weighted mean
        ``(close_weight·close + upside_weight·upside + downside_weight·downside)
        / (close_weight + upside_weight + downside_weight)``.
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_training_models.py -v`
Expected: PASS (all, including the unchanged `test_loss_weights_default_to_half`).

- [ ] **Step 7: Lint, type-check, commit**

```bash
uv run ruff check src/ophir/training_models.py tests/test_training_models.py
uv run mypy src/ophir
git add src/ophir/training_models.py tests/test_training_models.py
git commit -m "Normalize multi-target loss weights and add close_weight"
```

---

## Task 2: Thread `close_weight` through `run_training` and `train`

**Files:**
- Modify: `src/ophir/train.py` (`run_training` signature ~316-353 + model construction ~416-429; `train` signature ~450-486 + call ~494-529)
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `LightningOHLCPredictor(..., close_weight=…)` from Task 1.
- Produces: `run_training(..., close_weight: float = 1.0, …)` forwards `close_weight` to the model; `train(..., close_weight: float = 1.0, …)` forwards it to `run_training`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_train.py` (this monkeypatches `LightningOHLCPredictor` and asserts the kwarg is forwarded — no CUDA). First inspect the top of `tests/test_train.py` to reuse its existing monkeypatch pattern for `run_training`; if it already patches `build_split_handlers` / `build_dataloader` / `fetch_base_trainer`, mirror that. Add:

```python
def test_run_training_forwards_close_weight(monkeypatch) -> None:
    captured: dict[str, float] = {}

    class _FakeModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import ophir.train as train_mod
    import ophir.training_models as tm

    monkeypatch.setattr(tm, "LightningOHLCPredictor", _FakeModel)
    # Reuse the module's existing fakes for data + trainer; see the other
    # run_training tests in this file for the exact monkeypatch targets.
    _patch_run_training_io(monkeypatch)  # helper defined alongside existing tests

    train_mod.run_training(close_weight=2.0, max_steps=1, window_sample=0)
    assert captured["close_weight"] == 2.0
```

> If `tests/test_train.py` has no shared IO-patching helper, inline the same `monkeypatch.setattr` calls the existing `run_training` tests use (for `build_split_handlers`, `build_dataloader`, `register.fetch_base_trainer`, and the trainer's `.fit`). Do not introduce CUDA or real data.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_train.py -k close_weight -v`
Expected: FAIL — `run_training` got an unexpected keyword argument `close_weight`.

- [ ] **Step 3: Add `close_weight` to `run_training`**

In `src/ophir/train.py`, add to the `run_training` signature immediately before `upside_weight` (~348):

```python
    close_weight: float = 1.0,
    upside_weight: float = 0.5,
    downside_weight: float = 0.5,
```

And pass it into the model construction (in the `LightningOHLCPredictor(...)` call, before `upside_weight=…`, ~427):

```python
        close_weight=close_weight,
        upside_weight=upside_weight,
        downside_weight=downside_weight,
```

- [ ] **Step 4: Add `close_weight` to `train` and forward it**

In the `train` signature, before `upside_weight` (~482):

```python
    close_weight: float = 1.0,
    upside_weight: float = 0.5,
    downside_weight: float = 0.5,
```

And in `train`'s call to `run_training`, before `upside_weight=…` (~525):

```python
        close_weight=close_weight,
        upside_weight=upside_weight,
        downside_weight=downside_weight,
```

> `finetune` restores weights from a checkpoint and takes no loss-weight args — leave it unchanged.

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_train.py -k close_weight -v`
Expected: PASS.

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff check src/ophir/train.py tests/test_train.py
uv run mypy src/ophir
git add src/ophir/train.py tests/test_train.py
git commit -m "Thread close_weight through run_training and the train CLI"
```

---

## Task 3: Anchor `close_weight` in the sweep search space

**Files:**
- Modify: `src/ophir/sweep.py` (`sample_config` ~33-50)
- Test: `tests/test_sweep.py` (`test_sample_config_returns_valid_arch_and_ranges` ~16-36)

**Interfaces:**
- Consumes: `run_training(..., close_weight=…)` from Task 2.
- Produces: `sample_config(trial)` returns a config dict containing `"close_weight": 1.0` (fixed, not sampled) alongside the existing sampled `upside_weight` / `downside_weight`.

- [ ] **Step 1: Extend the existing config test to assert the anchor**

In `tests/test_sweep.py`, add to `test_sample_config_returns_valid_arch_and_ranges`, after the existing assertions (~35):

```python
    assert config["close_weight"] == 1.0  # anchored: normalization removes its scale DOF
    assert config["upside_weight"] == 0.5
    assert config["downside_weight"] == 0.5
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_sweep.py -k sample_config -v`
Expected: FAIL — `KeyError: 'close_weight'`.

- [ ] **Step 3: Add the anchored `close_weight` to `sample_config`**

In `src/ophir/sweep.py`, add to the returned dict in `sample_config` (next to the weight entries, ~48):

```python
        "close_weight": 1.0,
        "upside_weight": trial.suggest_float("upside_weight", 0.25, 1.0),
        "downside_weight": trial.suggest_float("downside_weight", 0.25, 1.0),
```

Add a brief comment above the return explaining the anchor:

```python
    # close_weight is fixed at 1.0: compute_loss normalizes the three weights by
    # their sum, so sampling all three would only add a redundant scale axis.
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_sweep.py -k sample_config -v`
Expected: PASS.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check src/ophir/sweep.py tests/test_sweep.py
uv run mypy src/ophir
git add src/ophir/sweep.py tests/test_sweep.py
git commit -m "Anchor close_weight at 1.0 in the sweep search space"
```

---

## Task 4: Make the sweep sampler and pruner configurable

**Files:**
- Modify: `src/ophir/sweep.py` (`run_sweep` ~109-134; add `_build_sampler` / `_build_pruner`)
- Modify: `src/ophir/cli.py` (`sweep` command ~83-138)
- Test: `tests/test_sweep.py`

**Interfaces:**
- Produces:
  - `_build_sampler(name: str, seed: int) -> optuna.samplers.BaseSampler` — `"tpe"` → `TPESampler(seed=…)`, `"random"` → `RandomSampler(seed=…)`, else `ValueError`.
  - `_build_pruner(prune: bool) -> optuna.pruners.BasePruner` — `True` → `SuccessiveHalvingPruner()`, `False` → `NopPruner()`.
  - `run_sweep(..., sampler: str = "tpe", prune: bool = True)` uses them.

- [ ] **Step 1: Write the failing builder tests**

Append to `tests/test_sweep.py`:

```python
def test_build_sampler_selects_type() -> None:
    assert isinstance(sweep._build_sampler("tpe", 0), optuna.samplers.TPESampler)
    assert isinstance(sweep._build_sampler("random", 0), optuna.samplers.RandomSampler)


def test_build_sampler_rejects_unknown() -> None:
    import pytest

    with pytest.raises(ValueError, match="sampler"):
        sweep._build_sampler("nope", 0)


def test_build_pruner_toggles() -> None:
    assert isinstance(sweep._build_pruner(True), optuna.pruners.SuccessiveHalvingPruner)
    assert isinstance(sweep._build_pruner(False), optuna.pruners.NopPruner)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_sweep.py -k "build_sampler or build_pruner" -v`
Expected: FAIL — `module 'ophir.sweep' has no attribute '_build_sampler'`.

- [ ] **Step 3: Add the builders and wire them into `run_sweep`**

In `src/ophir/sweep.py`, add (above `run_sweep`):

```python
def _build_sampler(name: str, seed: int) -> optuna.samplers.BaseSampler:
    """Construct the Optuna sampler selected by ``name`` (``"tpe"`` | ``"random"``)."""
    import optuna

    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    raise ValueError(f"unknown sampler {name!r}; expected 'tpe' or 'random'")


def _build_pruner(prune: bool) -> optuna.pruners.BasePruner:
    """ASHA pruner when ``prune``; a no-op pruner otherwise (clean control runs)."""
    import optuna

    return optuna.pruners.SuccessiveHalvingPruner() if prune else optuna.pruners.NopPruner()
```

> `sweep.py` imports `optuna` lazily inside functions and only under `TYPE_CHECKING` at module scope. Keep that pattern: the `import optuna` lines above stay inside the functions. The return annotations reference `optuna.samplers...`, which resolves via the existing `if TYPE_CHECKING: import optuna` block plus `from __future__ import annotations` (already at the top of the file).

Then change `run_sweep`'s signature to add the two parameters (after `proxy_kwargs`, ~115):

```python
    proxy_kwargs: dict[str, Any],
    sampler: str = "tpe",
    prune: bool = True,
```

and replace the `sampler = …` / `pruner = …` lines (~120-121) with:

```python
    sampler_obj = _build_sampler(sampler, base_seed)
    pruner = _build_pruner(prune)
```

and update the `optuna.create_study(...)` call to use `sampler=sampler_obj` (it currently passes `sampler=sampler`, ~125).

- [ ] **Step 4: Run the builder tests to verify they pass**

Run: `uv run pytest tests/test_sweep.py -k "build_sampler or build_pruner" -v`
Expected: PASS.

- [ ] **Step 5: Add `--sampler` / `--no-prune` to the CLI and forward them**

In `src/ophir/cli.py`, add two options to the `sweep` command signature (after `data_dir`, ~102):

```python
    sampler: str = typer.Option("tpe", help="Optuna sampler: 'tpe' (default) or 'random'"),
    prune: bool = typer.Option(True, help="Enable ASHA pruning (use --no-prune for a clean control study)"),
```

and forward them in the `run_sweep(...)` call (~132-138):

```python
    study_obj = sweep_mod.run_sweep(
        n_trials=trials,
        study_name=study,
        storage=storage,
        base_seed=base_seed,
        proxy_kwargs=proxy_kwargs,
        sampler=sampler,
        prune=prune,
    )
```

- [ ] **Step 6: Verify the CLI help renders (smoke test)**

Run: `uv run ophir sweep --help`
Expected: help text lists `--sampler` and `--no-prune`; exit 0.

- [ ] **Step 7: Lint, type-check, run the sweep tests, commit**

```bash
uv run ruff check src/ophir/sweep.py src/ophir/cli.py tests/test_sweep.py
uv run mypy src/ophir
uv run pytest tests/test_sweep.py tests/test_cli.py -v
git add src/ophir/sweep.py src/ophir/cli.py tests/test_sweep.py
git commit -m "Make sweep sampler and pruner configurable for clean control studies"
```

---

## Task 5: Importance helpers (`compute_importances` + `format_importances`)

**Files:**
- Modify: `pyproject.toml` (add `scikit-learn` to `dependencies`)
- Modify: `src/ophir/sweep.py` (add two functions)
- Test: `tests/test_sweep.py`

**Interfaces:**
- Produces:
  - `compute_importances(study: optuna.Study) -> dict[str, Any]` → `{"fanova": dict[str, float], "mdi": dict[str, float], "n_completed": int}`; returns empty importance dicts when fewer than 2 trials completed.
  - `format_importances(result: dict[str, Any], *, sampler: str, pruned: bool) -> str` — renders both rankings (desc) and prepends a warning line when `sampler != "random"` or `pruned` is `True` or `n_completed < 8`.

- [ ] **Step 0: Add the scikit-learn dependency**

Optuna's `FanovaImportanceEvaluator` and `MeanDecreaseImpurityImportanceEvaluator`
both require scikit-learn (RandomForest internals), which is not yet a project
dependency. Our code never imports `sklearn` directly — Optuna does — so no mypy
`ignore_missing_imports` entry is needed.

Add `"scikit-learn>=1.5"` to the `dependencies` list in `pyproject.toml`
(alphabetical position, between `pyarrow` and `tensorboard`), then sync:

```bash
uv add "scikit-learn>=1.5"   # or hand-edit pyproject.toml then: uv sync --group dev
```

Verify the evaluators now import:

Run: `uv run python -c "from optuna.importance import FanovaImportanceEvaluator, MeanDecreaseImpurityImportanceEvaluator; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 1: Write the failing formatter tests (no Optuna run)**

Append to `tests/test_sweep.py`:

```python
def test_format_importances_warns_for_tpe_or_pruned() -> None:
    result = {"fanova": {"downside_weight": 0.6, "lr": 0.4}, "mdi": {"downside_weight": 0.5}, "n_completed": 30}
    txt = sweep.format_importances(result, sampler="tpe", pruned=True)
    assert "WARNING" in txt
    assert "downside_weight" in txt
    # highest-importance param is listed first in the fANOVA section
    assert txt.index("downside_weight") < txt.index("lr")


def test_format_importances_clean_study_has_no_warning() -> None:
    result = {"fanova": {"lr": 1.0}, "mdi": {"lr": 1.0}, "n_completed": 40}
    txt = sweep.format_importances(result, sampler="random", pruned=False)
    assert "WARNING" not in txt


def test_format_importances_warns_on_few_trials() -> None:
    result = {"fanova": {"lr": 1.0}, "mdi": {"lr": 1.0}, "n_completed": 3}
    txt = sweep.format_importances(result, sampler="random", pruned=False)
    assert "WARNING" in txt
```

- [ ] **Step 2: Write the failing `compute_importances` test using a fake study**

Append to `tests/test_sweep.py`:

```python
def _study_with_completed_trials(n: int) -> optuna.Study:
    study = optuna.create_study(direction="maximize")
    dist = optuna.distributions.FloatDistribution(0.25, 1.0)
    for i in range(n):
        # Two varying params so fANOVA has variance to decompose.
        w = 0.25 + 0.75 * (i / max(n - 1, 1))
        study.add_trial(
            optuna.trial.create_trial(
                params={"downside_weight": w, "upside_weight": 1.0 - 0.5 * w},
                distributions={"downside_weight": dist, "upside_weight": dist},
                value=w,  # objective tracks downside_weight => high importance
            )
        )
    return study


def test_compute_importances_reports_completed_and_params() -> None:
    result = sweep.compute_importances(_study_with_completed_trials(20))
    assert result["n_completed"] == 20
    assert "downside_weight" in result["fanova"]
    assert set(result["fanova"]) == {"downside_weight", "upside_weight"}


def test_compute_importances_handles_too_few_trials() -> None:
    result = sweep.compute_importances(_study_with_completed_trials(1))
    assert result["n_completed"] == 1
    assert result["fanova"] == {}
    assert result["mdi"] == {}
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_sweep.py -k importances -v`
Expected: FAIL — `module 'ophir.sweep' has no attribute 'compute_importances'`.

- [ ] **Step 4: Implement `compute_importances`**

In `src/ophir/sweep.py`, add:

```python
def compute_importances(study: optuna.Study) -> dict[str, Any]:
    """Hyperparameter importances over the study's completed trials.

    Returns fANOVA and mean-decrease-impurity (MDI) importances plus the number
    of completed trials. fANOVA needs at least two completed trials with varying
    parameters; when there are too few, the importance maps come back empty
    rather than raising, so callers can still report ``n_completed``.
    """
    import optuna

    completed = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if len(completed) < 2:
        return {"fanova": {}, "mdi": {}, "n_completed": len(completed)}

    fanova = optuna.importance.get_param_importances(
        study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=0)
    )
    mdi = optuna.importance.get_param_importances(
        study, evaluator=optuna.importance.MeanDecreaseImpurityImportanceEvaluator(seed=0)
    )
    return {"fanova": dict(fanova), "mdi": dict(mdi), "n_completed": len(completed)}
```

- [ ] **Step 5: Implement `format_importances`**

In `src/ophir/sweep.py`, add:

```python
def format_importances(result: dict[str, Any], *, sampler: str, pruned: bool) -> str:
    """Render fANOVA + MDI importances, warning when the estimate is unreliable.

    fANOVA assumes a roughly i.i.d. design over the search space. A TPE sampler
    concentrates sampling, and pruning leaves only an early-success-biased subset
    of completed trials, so importances from such a study are biased. A small
    completed-trial count is also unreliable. Any of these prepends a WARNING.
    """
    lines: list[str] = []
    n = int(result["n_completed"])
    reasons: list[str] = []
    if sampler != "random":
        reasons.append(f"sampler={sampler!r} (non-random designs bias fANOVA)")
    if pruned:
        reasons.append("pruning enabled (completed trials are selection-biased)")
    if n < 8:
        reasons.append(f"only {n} completed trials")
    if reasons:
        lines.append("WARNING: importances may be unreliable — " + "; ".join(reasons) + ".")

    lines.append(f"Completed trials: {n}")
    for title, key in (("fANOVA", "fanova"), ("MDI", "mdi")):
        lines.append(f"\n{title} importances:")
        ranked = sorted(result[key].items(), key=lambda kv: kv[1], reverse=True)
        if not ranked:
            lines.append("  (too few completed trials to estimate)")
        for name, importance in ranked:
            lines.append(f"  {name:<18} {importance:.4f}")
    return "\n".join(lines)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_sweep.py -k importances -v`
Expected: PASS.

- [ ] **Step 7: Lint, type-check, commit**

```bash
uv run ruff check src/ophir/sweep.py tests/test_sweep.py
uv run mypy src/ophir
git add pyproject.toml uv.lock src/ophir/sweep.py tests/test_sweep.py
git commit -m "Add fANOVA + MDI importance helpers with reliability warnings"
```

---

## Task 6: `ophir importances` CLI command

**Files:**
- Modify: `src/ophir/cli.py` (add command after `sweep`, ~172)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `sweep.compute_importances`, `sweep.format_importances` from Task 5; `register.MODEL_DIR` for storage defaulting (mirrors the `sweep` command, ~118).
- Produces: `ophir importances <study> [--storage URL]` — loads the study and prints `format_importances`.

- [ ] **Step 1: Write the failing CLI test**

In `tests/test_cli.py`, mirror the existing command-registration / help smoke tests. Add:

```python
def test_importances_command_is_registered() -> None:
    from typer.testing import CliRunner

    from ophir.cli import app

    result = CliRunner().invoke(app, ["importances", "--help"])
    assert result.exit_code == 0
    assert "study" in result.output.lower()
```

> Match the exact import style the other tests in `tests/test_cli.py` use (some use `CliRunner`, some call the registered function directly). Reuse whichever the file already establishes.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_cli.py -k importances -v`
Expected: FAIL — `importances` is not a registered command (`exit_code != 0`).

- [ ] **Step 3: Add the command**

In `src/ophir/cli.py`, add after the `sweep` command (~172):

```python
@app.command()
def importances(
    study: str = typer.Argument(..., help="Optuna study name to analyze"),
    storage: str | None = typer.Option(
        None, help="Optuna storage URL; defaults to a SQLite db under the model dir"
    ),
    sampler: str = typer.Option("tpe", help="Sampler the study used (controls the reliability warning)"),
    pruned: bool = typer.Option(True, help="Whether the study used pruning (controls the reliability warning)"),
) -> None:
    """Print fANOVA + MDI hyperparameter importances for a completed sweep study.

    Loads the persisted study and reports both importance estimates. Pass the
    ``--sampler``/``--pruned`` the study was run with so the output can warn when
    the estimate is biased (TPE or ASHA produce non-i.i.d. completed-trial sets).
    """
    import os

    import optuna

    from ophir import register
    from ophir import sweep as sweep_mod

    if storage is None:
        storage = f"sqlite:///{os.path.join(register.MODEL_DIR, study + '.db')}"
    study_obj = optuna.load_study(study_name=study, storage=storage)
    result = sweep_mod.compute_importances(study_obj)
    typer.echo(sweep_mod.format_importances(result, sampler=sampler, pruned=pruned))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_cli.py -k importances -v`
Expected: PASS.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check src/ophir/cli.py tests/test_cli.py
uv run mypy src/ophir
git add src/ophir/cli.py tests/test_cli.py
git commit -m "Add ophir importances command to report sweep param importances"
```

---

## Task 7: Padding-masked prefix pool

**Files:**
- Modify: `src/ophir/models.py` (`pool_prefix_embedding` ~371-378; caller ~483)
- Test: `tests/test_models_output.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `pool_prefix_embedding(x: torch.Tensor, response_size: int, trade_occured: torch.Tensor) -> torch.Tensor` — masked mean over prefix positions where `trade_occured` is `True`; rows with no valid prefix position fall back to the unmasked prefix mean.

- [ ] **Step 1: Update the existing test for the new signature and add masking tests**

In `tests/test_models_output.py`, replace `test_pool_prefix_embedding_ignores_response_block` (~8-12) with:

```python
def test_pool_prefix_embedding_ignores_response_block():
    # 1 example, 4 positions, 2-d; prefix=first 2 rows, response=last 2.
    x = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0], [99.0, 99.0]]])
    trade = torch.ones(1, 4, dtype=torch.bool)
    pooled = pool_prefix_embedding(x, response_size=2, trade_occured=trade)
    torch.testing.assert_close(pooled, torch.tensor([[2.0, 2.0]]))


def test_pool_prefix_embedding_masks_padded_prefix_positions():
    # Prefix rows are [1,1] (padded) and [3,3] (valid); only the valid row counts.
    x = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0], [99.0, 99.0]]])
    trade = torch.tensor([[False, True, True, True]])
    pooled = pool_prefix_embedding(x, response_size=2, trade_occured=trade)
    torch.testing.assert_close(pooled, torch.tensor([[3.0, 3.0]]))


def test_pool_prefix_embedding_all_padded_falls_back_to_mean():
    x = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0], [99.0, 99.0]]])
    trade = torch.tensor([[False, False, True, True]])  # no valid prefix positions
    pooled = pool_prefix_embedding(x, response_size=2, trade_occured=trade)
    torch.testing.assert_close(pooled, torch.tensor([[2.0, 2.0]]))  # unmasked prefix mean
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_models_output.py -k pool_prefix -v`
Expected: FAIL — `pool_prefix_embedding()` missing required `trade_occured` / masking not applied.

- [ ] **Step 3: Implement the masked pool**

In `src/ophir/models.py`, replace `pool_prefix_embedding` (~371-378) with:

```python
def pool_prefix_embedding(
    x: torch.Tensor, response_size: int, trade_occured: torch.Tensor
) -> torch.Tensor:
    """Padding-masked mean of the prefix (observed-history) positions per example.

    Pools ``x[:, :-response_size]`` — the positions that carry real features,
    excluding the masked forecast block — and averages only the positions where a
    trade occurred, so padded (no-trade) rows do not contaminate the per-stock
    embedding used for the UI projection. Rows with no valid prefix position fall
    back to the unmasked prefix mean.
    """
    prefix = x[:, :-response_size]
    valid = trade_occured[:, : prefix.shape[1]].unsqueeze(-1).to(prefix.dtype)
    count = valid.sum(dim=1)
    masked_mean = (prefix * valid).sum(dim=1) / count.clamp_min(1.0)
    fallback = prefix.mean(dim=1)
    has_valid = count.squeeze(-1) > 0
    return torch.where(has_valid.unsqueeze(-1), masked_mean, fallback)
```

- [ ] **Step 4: Update the caller to pass `trade_occured`**

In `OHLCMulitClassPredictor.forward`, update the assignment (~483):

```python
        input.stock_embeddings = pool_prefix_embedding(
            x, int(input.response_size), input.trade_occured
        )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_models_output.py tests/test_models_leakage.py -v`
Expected: PASS (leakage tests unaffected — they don't touch `pool_prefix_embedding`).

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff check src/ophir/models.py tests/test_models_output.py
uv run mypy src/ophir
git add src/ophir/models.py tests/test_models_output.py
git commit -m "Padding-mask the prefix pool so no-trade rows don't skew embeddings"
```

---

## Task 8: Guard `response_size` in the model forward

**Files:**
- Modify: `src/ophir/models.py` (`OHLCMulitClassPredictor.forward` top, ~453)
- Test: `tests/test_models_leakage.py` (reuses its CPU `_predictor()` helper)

**Interfaces:**
- Produces: `OHLCMulitClassPredictor.forward` raises `ValueError` when `response_size` is not in `1 .. seq_len-1`.

- [ ] **Step 1: Write the failing guard tests**

Append to `tests/test_models_leakage.py`:

```python
import pytest

from ophir.model_data import OHLCMulitClassPredictorInput


def _input_with_response(response_size: int) -> OHLCMulitClassPredictorInput:
    return OHLCMulitClassPredictorInput(
        feature_input=torch.zeros(BATCH, SEQ_LEN, 12),
        response_size=torch.tensor(response_size),
        trade_occured=torch.ones(BATCH, SEQ_LEN, dtype=torch.bool),
        targets=torch.zeros(BATCH, SEQ_LEN, 3),
    )


def test_forward_rejects_response_size_ge_seq_len() -> None:
    model = _predictor()
    with pytest.raises(ValueError, match="response_size"):
        model(_input_with_response(SEQ_LEN))


def test_forward_rejects_zero_response_size() -> None:
    model = _predictor()
    with pytest.raises(ValueError, match="response_size"):
        model(_input_with_response(0))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_models_leakage.py -k "rejects_response or rejects_zero" -v`
Expected: FAIL — no `ValueError` raised (the code reaches the CUDA attention path or produces NaN instead).

- [ ] **Step 3: Add the guard at the top of `forward`**

In `src/ophir/models.py`, at the very start of `forward` (before `feature = input.feature_input`, ~453), insert:

```python
        seq_len = input.feature_input.shape[1]
        response_size = int(input.response_size)
        if not 0 < response_size < seq_len:
            raise ValueError(
                f"response_size must be in 1..{seq_len - 1}, got {response_size}"
            )
```

Then reuse the already-computed `seq_len` below by replacing the existing unpacking line `_, seq_len, _ = x.shape` (~459) with `_, _, _ = x.shape` removed entirely — i.e., delete that line, since `seq_len` is now defined at the top (the value is identical: `feature_input.shape[1]` equals `x.shape[1]` after the `feature_mlp` projection, which preserves the sequence dim).

> Verify nothing else between the old unpacking and its first use of `seq_len` depends on the deleted line. `seq_len` is next used at the `pe_slice = self.pe[:, :seq_len]` line; the top-level definition covers it.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_models_leakage.py -v`
Expected: PASS (existing leakage tests still pass; new guard tests pass).

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check src/ophir/models.py tests/test_models_leakage.py
uv run mypy src/ophir
git add src/ophir/models.py tests/test_models_leakage.py
git commit -m "Guard response_size range at the model forward entry"
```

---

## Task 9: CHANGELOG and final full-suite check

**Files:**
- Modify: `CHANGELOG.md` (top entry, matching existing format)

- [ ] **Step 1: Inspect the existing CHANGELOG format**

Run: `sed -n '1,30p' CHANGELOG.md`
Expected: see the heading style and bullet format of the most recent entry.

- [ ] **Step 2: Add an entry matching that format**

Add a new entry (matching the observed style) covering:

```
- Normalized the three multi-target loss weights by their sum and added a
  tunable `close_weight` (default 1.0), so the weights control task balance only
  and no longer co-vary with total loss scale. Note: existing default configs see
  a halved loss magnitude (effective LR rescale); prior checkpoints/LRs are not
  directly comparable.
- Added `--sampler {tpe,random}` and `--no-prune` to `ophir sweep`, and a new
  `ophir importances <study>` command reporting fANOVA + MDI importances with a
  reliability warning for biased (TPE/ASHA) designs.
- Padding-masked the UI prefix-pool embedding and guarded `response_size` range
  in the model forward.
```

> Adjust wording/heading to match the exact existing CHANGELOG convention (e.g. an `## [Unreleased]` section or dated heading), per the file you inspected.

- [ ] **Step 3: Run the full validation suite**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/ophir
uv run pytest
```
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md
git commit -m "Document loss-weight normalization and sweep diagnostics in CHANGELOG"
```

---

## Self-Review Notes

- **Spec coverage:** Part A → Tasks 1-3 (normalize + thread + anchor). Part B → Tasks 4-6 (configurable sampler/pruner, importance helpers, CLI command). Part C → Task 7 (masked pool) + Task 8 (guard). Testing section → tests embedded in each task. CHANGELOG/back-compat note → Task 9.
- **Type consistency:** `close_weight` is `float` everywhere (model, `run_training`, `train`, anchored `1.0` in `sample_config`). `compute_importances` returns `{"fanova", "mdi", "n_completed"}`, consumed verbatim by `format_importances` and the CLI. `pool_prefix_embedding` signature `(x, response_size, trade_occured)` matches its single caller in `forward`.
- **Anchored close_weight:** the model API keeps `close_weight` fully tunable; only the *sweep search space* fixes it at `1.0` (per approved design) to avoid the redundant scale axis introduced by normalization.
