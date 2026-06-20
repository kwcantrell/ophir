# Ophir

> A BERT-style masked transformer for stock OHLC prediction, paired with a
> deterministic trading core.

Ophir has two subsystems:

1. **Forecaster** — a full-encoder (BERT-style) masked transformer over
   sequences of daily OHLC candles. It predicts three forward targets per day:
   relative close return, intraday upside, and intraday downside. Predictions
   are reconstructed back into candlesticks and explored in a Gradio UI that
   also projects a learned stock-embedding space into an interactive 3-D point
   cloud.
2. **Trading core** (`ophir.trading`) — deterministic, side-effect-free logic
   for proposing, sizing, gating, recording, and scoring paper trades. A
   non-overridable safety gate is the single authority on whether an order is
   approved, resized, or rejected.

## Requirements

- **Python >= 3.10.**
- **A CUDA GPU** for the model and UI runtime paths. Training and inference
  configure Lightning with `accelerator="cuda"` and move tensors with `.cuda()`;
  these paths will not run on CPU.
- **A trained base checkpoint.** `serve` loads the latest base checkpoint from
  the package model directory at startup; without one it will not start.
- **Network access at UI startup.** The S&P 500 constituent list and split
  history are fetched on import (results are cached).

## Install

[uv](https://docs.astral.sh/uv/) is the supported workflow:

```bash
uv sync                 # create the environment and install ophir
uv sync --group dev     # add dev tooling (ruff, mypy, pytest, pre-commit)
```

## CLI

All commands are exposed through the `ophir` entry point (`ophir.cli:app`).

| Command | Purpose |
| --- | --- |
| `ophir train` | Train the base forecaster. |
| `ophir finetune` | Finetune from an existing checkpoint. |
| `ophir evaluate` | Score a checkpoint on the held-out validation set (includes cross-sectional rank-IC). |
| `ophir sweep` | Run an Optuna hyperparameter sweep (proxy-budget search with ASHA pruning), then confirm the top configs at full budget. Requires CUDA. |
| `ophir importances <study>` | Report fANOVA + MDI hyperparameter importances for a completed sweep study. |
| `ophir curate` | Build the high-quality dataset allowlist. |
| `ophir migrate-sqlite` | Convert the per-ticker parquet tree into a single-file SQLite store. |
| `ophir serve` | Launch the Gradio UI (see below). |
| `ophir dashboard` | Launch the live training dashboard. |
| `ophir register massive-key <KEY>` | Store a [MASSIVE](https://pypi.org/project/massive/) API key for data fetching. |
| `ophir trade gate` | Run a proposed order through the safety gate (exit non-zero on reject). |
| `ophir trade record` | Append one decision to the ledger. |
| `ophir trade close` | Mark a decision closed/scored with its realized P&L. |
| `ophir trade performance` | Compute portfolio metrics and write a `performance.md` snapshot. |

### UI

- `ophir serve` launches a Gradio app with predicted-vs-actual candlesticks, a
  3-D PCA stock-embedding cloud colored by predicted return, and a chat panel.
- `ophir dashboard` shows per-target loss curves read live from `metrics.csv`
  plus an on-demand response-block leakage check.

## Data and checkpoints

The package owns an on-disk layout under `src/ophir/.ophir/`:

- `.ophir/data/` — datasets, the `days/` stock store, and symbol allowlists.
- `.ophir/model/` — checkpoints, TensorBoard logs, and CSV training metrics.

`ophir register massive-key <KEY>` writes the MASSIVE API key under `.ophir/`
for later data fetching.

## Trading

The trading core is deterministic and isolated from the model:

- `safety.py` is the single non-overridable pre-trade gate. It returns
  `approve` / `resize` / `reject`; honoring its verdict is mandatory.
- `config.py` validates `account_mode` (`paper` or `live`) and guardrail limits.
  The system is intended for **paper** trading.
- `ledger.py` is an append-only JSON-Lines ledger — the source of truth for
  outcome attribution. Do not hand-edit.
- `memories/` holds the entity-organized knowledge base (per-ticker, per-sector,
  patterns, lessons, ledger, performance); see `memories/README.md`.

This is research tooling, **not financial advice**.

## Project layout

| Path | Purpose |
| --- | --- |
| `src/ophir/cli.py` | Typer CLI app (the `ophir` entry point). |
| `src/ophir/models.py` | Core transformer architecture (ALiBi bias, flex-attention block mask, ReZero). |
| `src/ophir/training_models.py` | PyTorch-Lightning training wrapper. |
| `src/ophir/model_data.py` | Structured model input/output container. |
| `src/ophir/ticker.py` | Stock data ingestion, split adjustment, feature extraction, datasets. |
| `src/ophir/register.py` | Filesystem, checkpoint, and Lightning `Trainer` helpers. |
| `src/ophir/evaluate.py` | Validation scoring and the eval report. |
| `src/ophir/sweep.py` | Optuna sweep harness and importance helpers. |
| `src/ophir/curation.py` | High-quality dataset curation. |
| `src/ophir/leakage.py` | Response-block target-leakage diagnostics. |
| `src/ophir/sqlite_store.py` | Single-file SQLite store for per-ticker data. |
| `src/ophir/train.py` | Training/finetuning entrypoints. |
| `src/ophir/dashboard.py` | Live training dashboard. |
| `src/ophir/ui.py` | Gradio UI. |
| `src/ophir/trading/` | Deterministic trading core (gate, ledger, signals, metrics, exposure, outcomes). |
| `memories/` | Trading knowledge base. |
| `tests/` | Test suite. |

## Development

```bash
uv sync --group dev
uv run pre-commit install      # one-time: install the mypy git hook
uv run pytest                  # full suite
uv run ruff check .
uv run ruff format --check .
uv run mypy src/ophir
```

## License

BSD 3-Clause. See [LICENSE](LICENSE).
