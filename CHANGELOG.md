# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.7] - 2026-05-20

### Fixed

- The chat panel's system prompt was constructed as a 4-tuple of strings
  (`SystemMessage(content=(str, str, str, str,))`) due to a trailing comma
  per line, sending a tuple to the LLM instead of a single string.
  Concatenated into one string so the chat panel now receives the intended
  system prompt.

### Changed

- Type-annotate `ophir.ui` and drop it from the mypy `ignore_errors`
  overrides — the override block is now empty and removed. Drop the
  `gradio.*` `ignore_missing_imports` override (gradio 6.4 ships
  `py.typed`); add `plotly.*` to the consolidated `ignore_missing_imports`
  block (plotly ships no stubs). Add `warn_unused_ignores = true` to
  `[tool.mypy]`. Make `return_ckpt_path: Literal[True]` keyword-only in
  both `load_base_model_ckpt` / `load_fintuned_ckpt` overloads so callers
  can pass it with `strict` defaulted.

## [0.1.6] - 2026-05-20

### Changed

- Type-annotate `ophir.models` and `ophir.ticker`; drop them from the ruff
  `ANN` per-file-ignores and the mypy `ignore_errors` overrides. Widen
  `extract_model_data(response_size)` from `int` to `int | numpy.ndarray`
  to match existing call sites in `StockStreamerDataset` and
  `StockHandlerDataset` (non-breaking — both forms already worked at
  runtime). Widen `get_splits` return to `dict[str, StockSplit | None]`
  to reflect the sentinel passthrough already exercised by tests.
  Consolidate the third-party `ignore_missing_imports` mypy override into
  one block covering `gradio`, `massive`, `tqdm`, and `yfinance`.

## [0.1.5] - 2026-05-20

### Changed

- Type-annotate `ophir.model_data`, `ophir.register`, and
  `ophir.training_models` and drop them from the coupled ruff `ANN`
  per-file-ignores and the mypy `ignore_errors` overrides. No user-facing
  behavior change. `models.py`, `ticker.py`, and `ui.py` remain suppressed.

## [0.1.4] - 2026-05-19

### Added

- Initial `pytest` configuration (`[tool.pytest.ini_options]`) and a
  deterministic, network-free test suite for `ophir.ticker` (73 tests across
  helpers, network, features, handler, streamer, and datasets). `pd.read_html`
  and `yfinance.Ticker` are mocked; fixtures never touch the package's
  `.ophir/` layout. Adds `pytest>=8.0` and `pytest-mock>=3.14` to the dev
  group.

### Fixed

- `StockHanlder.narrow_stocks` previously printed `stocks not found: 0`
  regardless of input (`len(stock_list) - len(stock_list)`). Now reports the
  actual count of unmatched symbols.

## [0.1.3] - 2026-05-19

### Added

- `CLAUDE.md` agent-bootstrap guide at the repo root: `src/ophir/` module
  map, dev workflow, type-checking tech-debt layout, and runtime
  requirements. Lets new agent sessions ramp up without re-exploring. Repo
  tooling; no runtime impact on the `ophir` package.

## [0.1.2] - 2026-05-15

### Added

- Static type checking via `mypy` (strict; configured in `pyproject.toml`),
  enforced by a local `pre-commit` hook that runs `mypy` against the project
  venv. The 5 legacy torch/lightning modules and `ophir.ui` are relaxed via
  `ignore_errors` overrides that mirror the existing ruff `ANN`
  per-file-ignores as documented tech debt. Repo tooling; no runtime impact on
  the `ophir` package.

## [0.1.1] - 2026-05-14

### Added

- `commit` Claude Code skill (`.claude/skills/commit/SKILL.md`) that automates
  release-hygiene commits: groups the working tree into ordered commit units,
  infers SemVer bumps, updates the changelog and version files, and commits
  only after explicit confirmation. Repo tooling; no runtime impact on the
  `ophir` package.
- Project README, changelog, and a Sphinx + autodoc documentation site,
  published to GitHub Pages via a `Docs` GitHub Actions workflow.

## [0.1.0] - 2026-05-14

### Added

- Initial project scaffold, `.gitignore`, and `pyproject.toml`.
- Core transformer models for OHLC prediction.
- Multi-stock price predictor support and training/residual notebooks.
- Pip-installable package with CLI commands to download datasets.
- `ophir` console script with `serve` and `register` subcommands.
- Model validation utilities.

### Changed

- Converted the architecture from a GPT-style causal decoder LLM to a
  BERT-style full-encoder masked LLM.
- Migrated to a `src/` layout and removed the abandoned top-level `ophir/`.
- Made the entire codebase ruff-clean.
- Track the shared `.claude/settings.json`; ignore local settings.

### Fixed

- Corrected the base exponent in `RotationNumberEmbeddings` so the maximum
  value yields a rotation of π.
- Model validation and minor fixes.

[Unreleased]: https://github.com/kwcantrell/ophir/compare/v0.1.7...HEAD
[0.1.7]: https://github.com/kwcantrell/ophir/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/kwcantrell/ophir/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/kwcantrell/ophir/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/kwcantrell/ophir/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/kwcantrell/ophir/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/kwcantrell/ophir/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/kwcantrell/ophir/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/kwcantrell/ophir/releases/tag/v0.1.0
