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
| `train.py` | `ophir train` / `ophir finetune` entry points; `run_training` is the shared engine used by both the CLI and the sweep harness. |
| `sweep.py` | Optuna hyperparameter sweep: TPE + ASHA proxy search over optimizer/loss-weight/arch-tier knobs scored by `val_rank_ic`, then full-budget confirm phase via `confirm_top`. |
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

`tests/` now covers `ticker` (handler, helpers, features, datasets, streamer,
network), `evaluate`, `curation`, `dashboard`, `leakage`, `optimizer`,
`models` (leakage + output activations), `model_data`, `training_models`
(loss decay + the wrapper), `sweep` (pure helpers + monkeypatched fit), `train`
(run_training), and `cli` (command registration + help smoke tests). Still
untested: `register.py`, `ui.py` — add tests for new code in those areas where
reasonable.

**Established test convention:** deterministic, seeded, **CPU-safe and
network-free**. Shared fixtures live in `tests/conftest.py` (`make_ohlcv`,
`ohlcv_df`, `parquet_dir`, `stock_split`, …). The metric/loss/feature cores are
pure module-level functions tested directly; see "Pipeline & testing gotchas"
below for testing the CUDA-only forward path.

## Type-checking (all strict — the old tech-debt suppressions are gone)

`pyproject.toml` is fully strict now (the per-file `ANN` ignores and the
`ignore_errors` legacy block were dropped in commits 39941a6 / 4816d0d /
eb9f0ae):

- `[tool.mypy] strict = true` over `files = ["src/ophir"]`, with
  `warn_unused_ignores = true` — so a stray/unneeded `# type: ignore` **fails**
  mypy. Only override is `ignore_missing_imports` for stubless third-party
  packages (`massive`, `plotly`, `tqdm`, `yfinance`).
- `[tool.ruff.lint.per-file-ignores]` only exempts `"tests/**"` from `ANN`.
- pytest runs with `filterwarnings = ["error", …]` and `--strict-config`, so
  **all of `src/ophir` and every test must type-check and run warning-clean.**
- Ruff `target-version = "py312"` and mypy `python_version = "3.10"` stay split
  (lowest supported runtime vs. lint target) — don't unify them.

## Pipeline & testing gotchas (non-obvious; learned the hard way)

- **Batch dict → model input is a `slots=True` dataclass splat.**
  `LightningOHLCPredictor._input_obj` does `OHLCMulitClassPredictorInput(**batch)`
  (`training_models.py`), so **every key a dataset puts in the batch dict must be a
  declared field** on `OHLCMulitClassPredictorInput`. To add data through the
  pipeline (e.g. `time`, `stock_id`, `date_ordinal`), add it as an *optional*
  field (`= None`), mirroring the existing `time` field — an undeclared key
  raises `TypeError`.
- **The default `DataLoader` collate is shared by training and eval.** It can't
  stack `datetime64`/`str` fields. Thread any new per-window identity as
  **integer / int64 tensors**, not strings — adding a custom `collate_fn` would
  also touch the training path. Keep such additions **opt-in** (off by default)
  so training is byte-for-byte unaffected.
- **Testing the CUDA-only forward path without a GPU:** the model forward uses
  flex-attention (CUDA-only), but its consumers can be unit-tested with a fake.
  Extract behavior into pure module-level helpers (e.g. `apply_output_activations`,
  `pool_prefix_embedding`, `rank_ic`, `robust_scale`) and test those on CPU; to
  test `accumulate_targets` / `evaluate_model`, pass a fake model whose
  `cuda()`/`eval()` return `self` and whose `__call__` returns a populated
  `OHLCMulitClassPredictorInput`. (Strict mypy + `warn_unused_ignores` means such
  test doubles may need a *precise* `# type: ignore[arg-type]` on the call — only
  where the error actually fires.)
- **Response-block positions are individually conditioned — not a single repeated
  mean.** Although `_apply_response_mask` fills the whole horizon with the *same*
  `mask_token`, each position then gets its own positional encoding (`self.pe`)
  **plus** ALiBi distance bias before the transformer, and `out_ff` decodes each
  position independently. So the model forecasts a position-varying curve over the
  horizon; do **not** reason about it as "one masked token ⇒ the unconditional
  mean" (a tempting but wrong read when reviewing the architecture).

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
