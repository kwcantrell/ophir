# Ophir

> BERT-style masked transformer for stock OHLC prediction, with a Gradio UI and
> a local-LLM chat panel.

Ophir trains a full-encoder (BERT-style) masked transformer over sequences of
daily OHLC (Open / High / Low / Close) candles and predicts three forward
targets per day — relative close return, intraday upside, and intraday
downside. Predictions are reconstructed back into candlesticks and explored in
an interactive Gradio dashboard that also embeds every S&P 500 stock into a
learned representation space and projects it into an interactive 3-D point
cloud.

## Features

- **Masked transformer model** (`ophir.models`) — ALiBi positional bias,
  `torch` flex-attention with a cached causal + prefix block mask, and ReZero
  residual scaling.
- **Lightning training/finetuning wrapper** (`ophir.training_models`) — AdamW
  with cosine warmup, per-group learning rates, weighted smooth-L1 loss over
  the three targets.
- **Stock data pipeline** (`ophir.ticker`) — parquet ingestion, stock-split
  adjustment, 13-feature technical extraction (log returns, rolling volatility,
  normalized volume, upside/downside), and streaming datasets.
- **Structured I/O dataclass** (`ophir.model_data`) — `OHLCMulitClassPredictorInput`
  carries features, targets, and predictions, and converts them back to
  candles and PCA projections.
- **Gradio UI + local-LLM chat** (`ophir.ui`) — candlestick comparison
  (predicted vs. actual), a 3-D PCA stock-embedding cloud colored by predicted
  return, and an Ollama-backed chat panel.
- **Checkpoint / data-dir management** (`ophir.register`) — Lightning
  `Trainer` factories and checkpoint loaders.
- **Data ingestion** (`ophir.agent.ingest`) — pull a ticker's daily OHLC from
  Yahoo Finance into a model-ready dataset (`ophir ingest`), reusing the
  `ophir.ticker` feature pipeline.
- **Model prediction** (`ophir.agent.predict`) — load the trained checkpoint and
  forecast a ticker's next 90 days, ranking candidates by predicted return
  (`ophir predict` / `ophir rank`).
- **`ophir` CLI** (Typer) — `serve`, `ingest`, `predict`, `rank`, and `register`
  subcommands.

## Requirements

- **Python >= 3.10**.
- **A CUDA GPU.** Training and inference paths call `.cuda()` and configure
  Lightning with `accelerator="cuda"`; the UI will not run on CPU.
- **A trained base checkpoint.** `ophir serve` loads the latest base checkpoint
  from the package data directory at startup; without one it will not start.
- **Ollama running locally**, serving the `gpt-oss:20b` model, for the chat
  panel.
- **Network access at UI startup** — the S&P 500 constituent list is fetched
  from Wikipedia and split history from Yahoo Finance (results are cached).

## Installation

Using [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync                 # create the environment and install ophir
```

Using pip:

```bash
pip install .           # or: pip install -e .   (editable, for development)
```

## CLI usage

```bash
ophir serve [--port 7860] [--share/--no-share] [--debug/--no-debug]
ophir ingest <SYMBOL> [--days 730]
ophir predict <SYMBOL>
ophir rank <SYMBOL> [<SYMBOL> ...] [--top-k 5]
ophir register massive-key <KEY>
```

- `ophir serve` launches the Gradio UI (`ophir.ui.serve`). `--share` exposes a
  public link; `--debug` (default on) launches Gradio in debug mode.
- `ophir ingest <SYMBOL>` pulls ~2 years of daily OHLC from Yahoo Finance into
  a model-ready parquet (no GPU required); `--days` overrides the lookback.
- `ophir predict <SYMBOL>` forecasts the next 90 days with the trained model;
  `ophir rank <SYMBOLS> [--top-k 5]` ranks several by predicted return (both need
  a CUDA GPU + checkpoint).
- `ophir register massive-key <KEY>` stores a [MASSIVE](https://pypi.org/project/massive/)
  API key (used for data fetching) under the package's `.ophir/` directory.

## Quickstart

```bash
uv sync
ophir register massive-key <YOUR_KEY>
# Ensure a trained base checkpoint exists in the package .ophir/model directory,
# a CUDA GPU is available, and Ollama is serving gpt-oss:20b.
ophir serve
# open the local URL printed by Gradio
```

## Documentation

The full documentation (overview, installation, CLI reference, architecture,
and an autodoc-generated API reference) is published to GitHub Pages at
**https://kwcantrell.github.io/ophir/** and is rebuilt and deployed
automatically by the `Docs` GitHub Actions workflow on every push to `main`.

To build it locally:

```bash
uv sync --group docs
uv run --group docs sphinx-build -b html docs docs/_build/html
# open docs/_build/html/index.html
```

## Development setup

```bash
uv sync --group dev
uv run pre-commit install   # one-time: install the mypy git hook
uv run ruff check .
uv run ruff format --check .
uv run mypy src/ophir
uv run --group docs sphinx-build -W -b html docs docs/_build/html
```

## Project layout

| Path | Purpose |
| --- | --- |
| `src/ophir/__init__.py` | Package root. |
| `src/ophir/cli.py` | Typer CLI app (`ophir` entry point). |
| `src/ophir/models.py` | Core transformer architecture. |
| `src/ophir/training_models.py` | PyTorch-Lightning training wrapper. |
| `src/ophir/model_data.py` | Structured model input/output dataclass. |
| `src/ophir/ticker.py` | Stock data loading, splits, feature extraction, datasets. |
| `src/ophir/register.py` | Trainer factories, checkpoint loaders, data dirs. |
| `src/ophir/ui.py` | Gradio UI and local-LLM chat. |
| `src/ophir/agent/` | Trading-agent layers (data ingestion, prediction) built on the model. |

## License

BSD 3-Clause. Copyright (c) 2025, kwcantrell. See [LICENSE](LICENSE).
