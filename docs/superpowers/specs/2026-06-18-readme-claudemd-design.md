# Design: Agent-facing README.md and CLAUDE.md

Date: 2026-06-18
Status: Approved

## Goal

Regenerate `README.md` (deleted in the working tree) and create a new
`CLAUDE.md`, both written for an **agent audience**, both derived purely from
the **current** `src/` source. Remove stale references to the retired Sphinx
documentation site. Do not touch `.claude/*`.

## Constraints and decisions

- **Audience: agents.** Operational and high-signal, not marketing.
- **No duplication.** README answers "what is this / which command do I run";
  CLAUDE.md answers "how do I work in this repo without breaking it" and links
  to README for the command/architecture reference.
- **No Ollama anywhere.** The `serve` UI currently embeds an Ollama-backed chat
  panel, but it is documented generically as a "chat panel"; Ollama is not named
  and is not listed as a requirement.
- **Source of truth is `src/`.** Trading behavior is described from
  `ophir.trading` (`safety.py`, `config.py`), not from the `.claude` skill.
- **`.claude/*` is out of scope.** No edits there (including the ruff-hook path).

## Deliverables

### 1. `README.md`

Sections:

- **Title + one-line description** — masked-transformer OHLC predictor plus a
  deterministic trading core.
- **What it is** — two subsystems: (a) BERT-style masked transformer predicting
  three forward targets per day (relative close return, intraday upside,
  intraday downside); (b) `ophir.trading`, a deterministic trading core.
- **Requirements** — Python >= 3.10; CUDA GPU for the model/UI runtime paths;
  network access at UI startup (S&P 500 constituents + split history). No
  Ollama.
- **Install** — `uv sync`; `uv sync --group dev` for development.
- **CLI reference** — table covering every command: `serve`, `dashboard`,
  `sweep`, `importances`, `migrate-sqlite`, `train`, `finetune`, `evaluate`,
  `curate`, `register massive-key`, and `trade gate/record/close/performance`.
- **UI** — `serve` launches a Gradio app (predicted-vs-actual candlesticks, a
  3-D PCA stock-embedding cloud, a chat panel). `dashboard` shows live training
  curves and an on-demand leakage check.
- **Project layout** — table of `src/ophir/*` modules, plus `src/ophir/trading/`,
  `memories/`, and `tests/`.
- **Data and checkpoints** — the `.ophir/{data,model}` package layout;
  `register massive-key <KEY>` stores the MASSIVE API key.
- **Trading** — short subsection: deterministic core; the safety gate
  (`safety.py`) is the authority; `account_mode` (paper/live) is validated in
  `config.py`; paper-only intent; "not financial advice."
- **License** — BSD 3-Clause (per `LICENSE`).

No references to a Sphinx/GitHub Pages documentation site.

### 2. `CLAUDE.md`

Lean operational guide. Sections:

- **What ophir is** — three sentences, then a pointer to README.md for the full
  command and layout reference (no duplication).
- **Commands** — `uv sync`; `uv run pytest` (289 tests; must stay offline and
  CPU-only); `uv run ruff check . && uv run ruff format --check .`;
  `uv run mypy src/ophir`; `uv run pre-commit install`.
- **Hard constraints**
  - mypy targets Python 3.10, ruff targets 3.12 — intentional; do not unify.
  - pytest runs `filterwarnings = error`; warnings fail the suite.
  - Tests must never touch the network, CUDA, or the package `.ophir/` layout;
    use `tmp_path` and the seeded fixtures in `tests/conftest.py`.
  - Model and UI runtime paths require CUDA, but tests must not.
- **Architecture map** — one line per module, grouped into model pipeline and
  trading core.
- **Trading constraints** — the safety gate is non-overridable; paper-only
  intent; enforcement lives in `safety.py` / `config.py`.
- **Conventions** — NumPy-style docstrings; `known-first-party = ["ophir"]`
  import ordering; update the CHANGELOG `[Unreleased]` section for notable
  changes.
- **Dev workflow** — adversarial review panel -> TDD plan -> subagent-driven
  execution.

### 3. Cleanup

- Delete `.github/workflows/docs.yml` (it builds a Sphinx site that no longer
  exists in `docs/`).
- Remove the `[dependency-groups] docs` block (`sphinx`, `furo`,
  `sphinx-autodoc-typehints`) from `pyproject.toml`.

## Out of scope

- No changes under `src/`, `tests/`, or `.claude/`.
- No new documentation site.
