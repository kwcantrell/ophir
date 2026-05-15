# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/kwcantrell/ophir/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/kwcantrell/ophir/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/kwcantrell/ophir/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/kwcantrell/ophir/releases/tag/v0.1.0
