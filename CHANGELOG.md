# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

[Unreleased]: https://github.com/kwcantrell/ophir/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kwcantrell/ophir/releases/tag/v0.1.0
