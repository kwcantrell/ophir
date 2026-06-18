# CLAUDE.md — agent bootstrap for the ophir repo

A short orientation for Claude / agent sessions. User-facing docs are in `README.md`
and `docs/`; this file exists to skip the rediscovery step.

## What this project is

Ophir is a BERT-style masked transformer for stock OHLC prediction, packaged with
a Gradio dashboard and a local-Ollama chat panel. It predicts three forward
targets per day (relative close return, intraday upside, intraday downside)
across S&P 500 tickers.

## Where the live code lives

- **Live code: `src/ophir/` only.**
- **Ignore** the top-level `ophir/`, `oldcode/`, and any `old_*.py` — abandoned
  refactor leftovers. Don't propose fixes or analysis for them unless asked.
- The `.env` file is empty and unused; `dotenv` is in deps but the code does not
  read env vars at runtime. The MASSIVE API key is stored at
  `src/ophir/.ophir/.massive_key` via `ophir register massive-key`.

## Module map (`src/ophir/`)

| File | What it does |
| --- | --- |
| `cli.py` | Typer app exposed as the `ophir` console script (`ophir.cli:app`). Mounts `register.app`; lazy-imports `ui.serve`. |
| `register.py` | Owns `.ophir/{data,model}` layout, Lightning `Trainer` factories, checkpoint loaders, MASSIVE-key storage. **Prints on import** (lines 20, 24) — leftover debug, expected. |
| `models.py` | Transformer architecture: ALiBi positional bias, `torch` flex-attention with cached causal+prefix block mask, ReZero residual scaling. |

> **Forecast-masking contract (do not regress):** every input feature at a
> response-block position is contemporaneous with that day's targets
> (`r_close`/`upside`/`downside` and the rolling features derived from them), so
> feeding them leaks the answer. `OHLCMulitClassPredictor._apply_response_mask`
> overwrites the last `response_size` positions with a learned `mask_token`
> before the transformer; the model must forecast the horizon from the prefix
> only. Masking just the three target columns is **not** enough — mask the whole
> block. Pinned by `tests/test_models_leakage.py`.
>
> **Train/val split (for the off-repo training driver):** split **by date**, not
> randomly (overlapping `offset` windows otherwise straddle the boundary), and
> leave an embargo gap ≥ `seq_len` between train-end and val-start. Rolling
> features are trailing/self-normalizing, so there is no global-stat leak.
| `training_models.py` | PyTorch-Lightning wrapper: AdamW + cosine warmup, per-group LRs, weighted smooth-L1 over the three targets. |
| `model_data.py` | `OHLCMulitClassPredictorInput` dataclass; converts features/targets/predictions back into candles and PCA projections. |
| `ticker.py` | Parquet ingest, stock-split adjustment, 13-feature extraction (log returns, rolling vol, normalized volume, upside/downside), streaming datasets. |
| `ui.py` | Gradio dashboard + LangChain–Ollama chat. Heavy import — fetches S&P 500 list, loads a CUDA checkpoint. Docs build mocks it. |

## Entry points

- Console script (`pyproject.toml:32`): `ophir = "ophir.cli:app"`.
- `ophir serve [--port 7860] [--share] [--debug]` — launches Gradio.
- `ophir register massive-key <KEY>` — persists the MASSIVE API key.

## Dev workflow

```bash
uv sync --group dev                 # set up env (use uv, not pip, for dev)
uv run pre-commit install           # one-time: installs the mypy git hook
uv run ruff check .
uv run ruff format --check .
uv run mypy src/ophir
uv run pytest
uv run --group docs sphinx-build -W -b html docs docs/_build/html
```

- Pre-commit runs `uv run --frozen mypy src/ophir` against the **project venv**
  (not an isolated pre-commit venv) so `torch`/`lightning`/`numpy` py.typed
  packages resolve. Config: `.pre-commit-config.yaml`.
- Ruff `target-version = "py312"` and mypy `python_version = "3.10"` are both
  intentional (lowest supported runtime vs. lint target). Don't unify them.

## Tests

`tests/` only covers `ticker` (6 files: handler, helpers, features, datasets,
streamer, network). `models.py`, `training_models.py`, `register.py`, `ui.py`,
`cli.py` are untested — add tests for new code in those areas where reasonable.

## Type-checking tech debt (biggest gotcha)

Two coupled suppression blocks in `pyproject.toml`:

- `[tool.ruff.lint.per-file-ignores]` (lines 73–77) suppresses `ANN` on five
  legacy files: `models.py`, `ticker.py`, `register.py`, `training_models.py`,
  `model_data.py`.
- `[[tool.mypy.overrides]]` (lines 111–120) sets `ignore_errors = true` on
  those five **plus `ophir.ui`** (which has real strict violations and a latent
  bug). A scoped `gradio.*` `ignore_missing_imports` override (lines 122–124)
  exists because gradio ships no stubs.

These blocks are meant to be removed together. **New files (`cli.py`,
`__init__.py`) stay strict** — do not extend the suppression list to make new
code type-check.

## Runtime requirements (often a surprise)

- **CUDA GPU required.** `.cuda()` and `accelerator="cuda"` are hardcoded in
  training/inference and the UI; CPU runs will not work.
- **Ollama** running locally serving `gpt-oss:20b` for the chat panel.
- **Network at UI startup** — Wikipedia (S&P 500 list) and Yahoo Finance
  (split history).
- **A trained base checkpoint** under `src/ophir/.ophir/model/`.
- Implication: a sandboxed agent usually **cannot** `ophir serve` end-to-end.
  Validate changes with `pytest`, `mypy`, and `ruff` instead.

## CI / docs

- One workflow: `.github/workflows/docs.yml` builds Sphinx with `-W` on PRs and
  deploys to GitHub Pages on `main`.
- Published at <https://kwcantrell.github.io/ophir/>.

## Style notes

- Don't add comments unless they explain non-obvious *why*; well-named code
  speaks for itself.
- Don't leave removed-code stubs, backward-compat shims, or "// removed" notes.
- New CHANGELOG entries should match the existing format.
