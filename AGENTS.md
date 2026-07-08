# AGENTS.md

Guidance for agents working in this repository. See `README.md` for the full
command and project-layout reference; this file covers how to work here without
breaking things.

## What ophir is

A BERT-style masked transformer that predicts three forward OHLC targets per day
(relative close return, intraday upside, intraday downside), plus a
deterministic trading core (`ophir.trading`) that proposes, gates, records, and
scores paper trades. The package is built and run with `uv`.

## Commands

```bash
uv sync --group dev                          # environment + dev tooling
uv run pytest                                # full suite (must stay offline + CPU-only)
uv run ruff check . && uv run ruff format --check .
uv run mypy src/ophir                        # strict mode
uv run pre-commit install                    # one-time: installs the mypy/ruff/gitleaks hooks
```

Run a single test file with `uv run pytest tests/test_<name>.py`.

CI (`.github/workflows/ci.yml`) runs the same four checks (ruff check, ruff
format, mypy, pytest) on every PR and on pushes to `main`.

## Hard constraints

- **mypy targets Python 3.10, ruff targets 3.12.** This is intentional (3.10 is
  the lowest supported runtime; 3.12 enables newer lint rules). Do not "fix" one
  to match the other.
- **pytest runs `filterwarnings = error`.** Any warning the project owns fails
  the suite. (Third-party `DeprecationWarning`/`FutureWarning` are downgraded to
  `default`; do not broaden that.)
- **Tests must never touch the network, CUDA, or the package `.ophir/`
  layout.** Use `tmp_path` and the seeded, deterministic fixtures in
  `tests/conftest.py`. Model and UI *runtime* paths require CUDA, but tests
  must not.
- **mypy is `strict = True`** and runs as a pre-commit hook against the project
  venv. Keep `src/ophir` fully typed.

## Architecture

**Model pipeline**

- `models.py` — transformer architecture (ALiBi bias, flex-attention block mask, ReZero).
- `training_models.py` — PyTorch-Lightning wrapper (AdamW, cosine warmup, weighted loss).
- `ticker.py` — data ingestion, split adjustment, feature extraction, streaming datasets.
- `model_data.py` — structured input/output container; reconstructs candles and PCA projections.
- `register.py` — `.ophir/` filesystem layout, checkpoint loaders, `Trainer` factories.
- `train.py` — `train` / `finetune` entrypoints.
- `evaluate.py` — validation scoring and the eval report (cross-sectional rank-IC).
- `sweep.py` — Optuna sweep harness + fANOVA/MDI importance helpers.
- `curation.py` — high-quality dataset curation.
- `leakage.py` — response-block target-leakage diagnostics.
- `sqlite_store.py` — single-file SQLite store for per-ticker data.
- `ui.py` / `dashboard.py` — Gradio UI and live training dashboard.
- `cli.py` — Typer app wiring all commands together.

**Trading core (`ophir.trading`)**

- `types.py` — frozen domain types (no logic).
- `config.py` — load/validate `config.json` (`account_mode`, guardrail limits).
- `safety.py` — the single non-overridable pre-trade gate.
- `ledger.py` — append-only JSONL decision ledger (outcome-attribution source of truth).
- `signals.py` / `metrics.py` / `exposure.py` / `outcomes.py` / `forecast.py` / `memory.py` — signal blending, performance metrics, exposure aggregation, outcome scoring, the ophir-forecast seam, and entity-memory editing.
- `cli.py` — the `ophir trade` subcommands.

## Trading constraints

- The safety gate (`safety.py`) is **non-overridable**. Never weaken it or route
  around it; honor `reject` (skip) and `resize` (use the smaller approved
  notional).
- The system is **paper-only** in intent. `account_mode` is validated in
  `config.py`; do not add a path that bypasses it.

## How we work

Non-trivial work follows the superpowers SDLC — **don't jump straight to code**:

**brainstorm → spec (`docs/superpowers/specs/`) → plan (`docs/superpowers/plans/`) →
implement.** Small, obvious fixes can skip ahead, but design-bearing changes get a
spec first.

## Conventions

- **NumPy-style docstrings** throughout `src/ophir` — match the existing density
  when adding code.
- Imports: `known-first-party = ["ophir"]` (ruff/isort ordering).
- Update the `[Unreleased]` section of `CHANGELOG.md` for notable changes.

## Git conventions

- **Conventional Commits**: `type(scope): description` — types in use: `feat`,
  `fix`, `docs`, `refactor`, `test`, `chore`, `style`.
- **Branch naming**: `category/short-description` (`feat/`, `fix/`, `docs/`,
  `refactor/`, `chore/`).
- **Merging**: PRs into `main` with linear history (fast-forward or rebase —
  no merge commits). `main` requires the `checks` CI job to pass.

## Dev workflow

Preferred approach for non-trivial work: adversarial review panel → TDD plan →
subagent-driven execution.
