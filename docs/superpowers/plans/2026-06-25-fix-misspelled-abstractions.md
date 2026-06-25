# Fix Misspelled Core Abstractions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename five misspelled core Python identifiers (`Mulit`→`Multi`, `Hanlder`→`Handler`, `fintuned`→`finetuned`) across `src/`, `tests/`, and `scripts/` with zero behavioral change.

**Architecture:** This is a mechanical, behavior-preserving rename, not a feature. The regression gate is the **existing** test + type-check suite, not new tests. Each of the three typo substrings has been verified unique (it appears only inside its intended identifier), so a global substring replace is collateral-free. Each task: replace one substring, prove the suite still passes and `mypy --strict` is clean, prove the typo is gone, commit.

**Tech Stack:** Python 3.10+ runtime, `uv` for env/commands, `pytest` (offline + CPU-only, `filterwarnings = error`), `mypy` (strict, targets 3.10), `ruff` (lint+format, targets 3.12), GNU `sed`/`grep` on Linux.

## Global Constraints

- mypy targets Python 3.10 and runs with `strict = True`; ruff targets 3.12. Do not "fix" one to match the other.
- `pytest` runs `filterwarnings = error`; it must never touch the network, CUDA, or the package `.ophir/` layout.
- Keep `src/ophir` fully typed (NumPy-style docstrings throughout).
- **Out of scope — never modify these persisted on-disk strings** (renaming them breaks resolution of existing checkpoints): `FINETUNE_NAME = "ophire-ohlc-finetuned"` (`src/ophir/register.py:38`), `BASE_NAME`, and any `ModelCheckpoint(filename=...)` string literals. The `ophire`→`ophir` wart is deliberately retained.
- Scope is Python identifiers in `src/`, `tests/`, `scripts/` only. Do not edit historical `docs/` plan/spec files that quote the old names.
- `load_fintuned_ckpt` → `load_finetuned_ckpt` is a hard rename with **no** deprecated alias.

---

### Task 1: Establish a green baseline

Capture proof that the suite is green *before* any edit, so later "still green" claims are meaningful.

**Files:**
- Modify: none (verification only)

- [ ] **Step 1: Confirm a clean working tree**

Run: `git status --porcelain`
Expected: empty output (no uncommitted changes). If non-empty, stop and resolve first.

- [ ] **Step 2: Run the type checker (baseline)**

Run: `uv run mypy src/ophir`
Expected: `Success: no issues found` (some files). Record that it passed.

- [ ] **Step 3: Run the full test suite (baseline)**

Run: `uv run pytest -q`
Expected: all tests pass, 0 failures, 0 errors. Record the passing test count.

- [ ] **Step 4: Confirm the three typo substrings are present and unique**

Run:
```bash
grep -rhoE '[A-Za-z_]*(Mulit|Hanlder|fintuned)[A-Za-z_]*' src tests scripts --include='*.py' | sort -u
```
Expected exactly these five identifiers, nothing else:
```
OHLCMulitClassParameters
OHLCMulitClassPredictor
OHLCMulitClassPredictorInput
StockHanlder
load_fintuned_ckpt
```
If any other identifier appears, stop — a substring replace would be unsafe and the plan must be revised to whole-word renames.

No commit (verification-only task).

---

### Task 2: Rename `StockHanlder` → `StockHandler`

**Files (all `.py` containing `Hanlder`):**
- Modify: `src/ophir/ticker.py` (class def at `:550`), `src/ophir/ui.py`, `src/ophir/dashboard.py`, `src/ophir/curation.py`, `src/ophir/train.py`, `src/ophir/trading/momentum.py`, `tests/conftest.py`, `tests/test_trading_momentum.py`, `tests/test_models_leakage_realdata.py`, `tests/test_ticker_handler.py`, `tests/test_ticker_datasets.py`, `tests/test_train.py`, `scripts/leakage_viz.py`

**Interfaces:**
- Consumes: nothing (independent of other tasks).
- Produces: the public class `StockHandler` (was `StockHanlder`). `StockHandlerDataset` already exists and is spelled correctly — it is unaffected (it does not contain the substring `Hanlder`) and will simply reference the renamed class.

- [ ] **Step 1: Replace the substring everywhere it occurs**

Run:
```bash
grep -rl 'Hanlder' src tests scripts --include='*.py' | xargs -r sed -i 's/Hanlder/Handler/g'
```

- [ ] **Step 2: Verify the typo is gone**

Run: `grep -rn 'Hanlder' src tests scripts --include='*.py'`
Expected: no output (exit code 1).

- [ ] **Step 3: Verify the rename landed**

Run: `grep -rn 'class StockHandler\b' src/ophir/ticker.py`
Expected: one hit — `class StockHandler:` near line 550.

- [ ] **Step 4: Type-check (catches any missed reference)**

Run: `uv run mypy src/ophir`
Expected: `Success: no issues found`. (`strict` mode would flag a dangling `StockHanlder` reference as an error.)

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: same passing count as Task 1 Step 3, 0 failures.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: rename StockHanlder -> StockHandler"
```

---

### Task 3: Rename `OHLCMulitClass*` → `OHLCMultiClass*`

One substring replace (`Mulit`→`Multi`) atomically renames all three classes: `OHLCMulitClassPredictor`, `OHLCMulitClassPredictorInput`, `OHLCMulitClassParameters`.

**Files (all `.py` containing `Mulit`):**
- Modify: `src/ophir/models.py` (class defs `OHLCMulitClassParameters:54`, `OHLCMulitClassPredictor:417`), `src/ophir/model_data.py` (class def `OHLCMulitClassPredictorInput:12`), `src/ophir/leakage.py`, `src/ophir/ticker.py`, `src/ophir/ui.py`, `src/ophir/training_models.py`, `tests/test_leakage_score.py`, `tests/test_training_models.py`, `tests/test_models_leakage.py`, `tests/test_evaluate.py`, `tests/test_model_data.py`, `tests/test_models_output.py`, `scripts/leakage_viz.py`

**Interfaces:**
- Consumes: nothing (independent of other tasks).
- Produces: public classes `OHLCMultiClassPredictor`, `OHLCMultiClassPredictorInput`, `OHLCMultiClassParameters`. The LightningModule attribute `self.ohlc_predictor` is **not** renamed (it does not contain `Mulit`), so checkpoint `state_dict` keys are unchanged.

- [ ] **Step 1: Replace the substring everywhere it occurs**

Run:
```bash
grep -rl 'Mulit' src tests scripts --include='*.py' | xargs -r sed -i 's/Mulit/Multi/g'
```

- [ ] **Step 2: Verify the typo is gone**

Run: `grep -rn 'Mulit' src tests scripts --include='*.py'`
Expected: no output (exit code 1).

- [ ] **Step 3: Verify all three classes renamed**

Run: `grep -rnE 'class OHLCMultiClass(Predictor|PredictorInput|Parameters)\b' src/ophir`
Expected: three hits across `models.py` (Parameters, Predictor) and `model_data.py` (PredictorInput).

- [ ] **Step 4: Type-check**

Run: `uv run mypy src/ophir`
Expected: `Success: no issues found`.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: same passing count as baseline, 0 failures.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: rename OHLCMulitClass* -> OHLCMultiClass*"
```

---

### Task 4: Rename `load_fintuned_ckpt` → `load_finetuned_ckpt`

**Files (all `.py` containing `fintuned`):**
- Modify: `src/ophir/register.py` (the `@overload` declarations + impl around `:563-577`), `src/ophir/evaluate.py` (import + call site)

**Interfaces:**
- Consumes: nothing (independent of other tasks).
- Produces: public function `load_finetuned_ckpt(...)` (hard rename, no alias). Aligns with the already-correct `_latest_finetuned_ckpt` and the `--finetuned` CLI flag. The substring `fintuned` does not occur inside the correctly-spelled `finetuned`, so existing correct spellings are untouched.

- [ ] **Step 1: Replace the substring everywhere it occurs**

Run:
```bash
grep -rl 'fintuned' src tests scripts --include='*.py' | xargs -r sed -i 's/fintuned/finetuned/g'
```

- [ ] **Step 2: Verify the typo is gone**

Run: `grep -rn 'fintuned' src tests scripts --include='*.py'`
Expected: no output (exit code 1).

- [ ] **Step 3: Verify the rename landed and the persisted string is untouched**

Run: `grep -rn 'def load_finetuned_ckpt' src/ophir/register.py`
Expected: the function defs (overloads + impl) now read `load_finetuned_ckpt`.

Run: `grep -n 'ophire-ohlc-finetuned' src/ophir/register.py`
Expected: still present — `FINETUNE_NAME = "ophire-ohlc-finetuned"` is intentionally unchanged.

- [ ] **Step 4: Type-check**

Run: `uv run mypy src/ophir`
Expected: `Success: no issues found`.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: same passing count as baseline, 0 failures.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: rename load_fintuned_ckpt -> load_finetuned_ckpt"
```

---

### Task 5: Final verification, lint, and changelog

**Files:**
- Modify: `CHANGELOG.md` (append under the existing `## [Unreleased]` → `### Fixed` section)

- [ ] **Step 1: Confirm zero remaining Python-identifier typos**

Run: `grep -rn 'Mulit\|Hanlder\|fintuned' src tests scripts --include='*.py'`
Expected: no output (exit code 1).

- [ ] **Step 2: Confirm the only retained `fintuned`-family string is the persisted checkpoint name**

Run: `grep -rn 'ophire' src/ophir`
Expected: only `FINETUNE_NAME = "ophire-ohlc-finetuned"` in `register.py` (the documented, intentional exception).

- [ ] **Step 3: Lint and format check**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: `All checks passed!` and no files would be reformatted. If `ruff format --check` reports diffs, run `uv run ruff format .`, then re-run this step.

- [ ] **Step 4: Add the changelog entry**

In `CHANGELOG.md`, under `## [Unreleased]` → `### Fixed`, add this bullet as the first item of the `### Fixed` list:

```markdown
- Corrected long-standing spelling errors in core public identifiers:
  `OHLCMulitClassPredictor`/`OHLCMulitClassPredictorInput`/`OHLCMulitClassParameters`
  → `OHLCMultiClass*`, `StockHanlder` → `StockHandler`, and `load_fintuned_ckpt`
  → `load_finetuned_ckpt`. Behavior is unchanged and existing checkpoints still
  load (state-dict keys and saved hyperparameters are unaffected). The persisted
  `"ophire-ohlc-finetuned"` checkpoint filename is intentionally left as-is to
  avoid breaking resolution of already-saved checkpoints.
```

- [ ] **Step 5: Final full-suite + type-check pass**

Run: `uv run pytest -q && uv run mypy src/ophir`
Expected: all tests pass; `Success: no issues found`.

- [ ] **Step 6: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: changelog for misspelled-abstraction rename"
```

---

## Self-Review

**1. Spec coverage:**
- Rename `OHLCMulitClass*` (3 classes) → Task 3. ✓
- Rename `StockHanlder` → Task 2. ✓
- Rename `load_fintuned_ckpt` (hard, no alias) → Task 4. ✓
- Persisted strings out of scope / `ophire` retained → Global Constraints + Task 4 Step 3 + Task 5 Step 2. ✓
- Checkpoint-safety rationale → encoded as "attribute `ohlc_predictor` not renamed" (Task 3 Interfaces) and changelog text. ✓
- Baseline-green gate → Task 1. ✓
- Per-rename `mypy` + `pytest` gate → Steps 4-5 of Tasks 2-4. ✓
- Final zero-hit `grep` + `ruff` + CHANGELOG → Task 5. ✓

**2. Placeholder scan:** No TBD/TODO/"handle edge cases"/vague steps. Every code/command step shows the exact command and expected output, and the changelog text is given verbatim.

**3. Type consistency:** New names are consistent across tasks — `StockHandler`, `OHLCMultiClassPredictor`/`OHLCMultiClassPredictorInput`/`OHLCMultiClassParameters`, `load_finetuned_ckpt`. The unchanged attribute `self.ohlc_predictor` is named identically where referenced. No task references a name another task fails to define.
