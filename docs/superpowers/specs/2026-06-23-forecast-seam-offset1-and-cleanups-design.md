# Forecast seam (offset-1 inference) + minor cleanups — design

**Date:** 2026-06-23
**Branch:** `forecast-seam-offset1-and-cleanups`
**Status:** approved design; implementation plan to follow.

## Background

The forecast-ceiling investigation concluded with an operating-point fix
(Components A + B, merged): the model carries seed-stable near-band
cross-sectional skill, the operating point is **read offset-1 of the 90-day
model** (no short-horizon retrain), and the best checkpoint is now selected on
`val_rank_ic_near`. Component C — wiring the trading forecast seam to that
offset-1 prediction — was deferred to this spec. The final whole-branch review
of A+B also logged three minor cleanups, bundled here.

`ophir.trading.forecast.load_forecasts` currently returns `{}` (inference not
wired). It has **no consumer yet** — the signals layer exposes a `blend(ophir=…)`
slot but nothing connects them. This spec implements inference into the seam so
it becomes real and callable; wiring a consumer (signals/cli feeding `blend`) is
left to a later step.

## Scope

- **Component C (inference only):** implement `load_forecasts` to return per-symbol
  offset-1 forecasts from the IC-best checkpoint. No consumer wiring.
- **Cleanups (separate task group):** three independent minor fixes from the A+B
  final review.

Out of scope: consumer wiring, signal normalization (`OphirForecast` → `[-1,1]`),
any trading-execution path. `safety.py` is untouched (the seam only *produces*
forecasts).

## Component C — `load_forecasts` offset-1 inference

Signature unchanged:

```
load_forecasts(symbols: Sequence[str], model_dir: str | Path | None)
    -> dict[str, OphirForecast]
```

Contract preserved: **never raises**; returns `{}` whenever forecasts are
unavailable, so the trading loop degrades to non-ophir signals.

The function splits into a CPU half (testable offline) and a CUDA half (gated,
runtime-only), mirroring how `dashboard.run_evaluation` gates its forward.

### C1. `build_latest_inputs` (CPU, unit-testable)

New helper in `ticker.py`, beside the existing windowing code:

```
build_latest_inputs(
    symbols: Sequence[str], seq_len: int = 365, base_path: str | None = None
) -> dict[str, dict[str, Any]]
```

- Builds a `StockHanlder` rooted at `base_path` (a Hive parquet tree). When
  `base_path` is `None`, defaults to `register.get_default_data_days_dir()/stocks`.
  The injectable `base_path` is what makes the helper unit-testable against the
  `parquet_dir` fixture without touching the package tree.
- Calls `StockHanlder.keep_stocks(symbols)` to restrict to the requested symbols
  and drop unknowns.
- For each kept symbol, produces the **most-recent window** model-input dict at
  `response_size=1`, reusing the existing streamer feature pipeline (no
  reimplemented feature extraction; DRY).
- Returns `{symbol: input_dict}`, silently skipping symbols absent from the data
  tree or failing the handler's volume/history filters.

Reuses `extract_model_data(df, response_size=1)`. Testable offline with the
`parquet_dir` conftest fixture (CPU; no CUDA, no network).

### C2. `load_forecasts` orchestration (CUDA-gated)

Guards, in order (each returns `{}` without raising):

1. `model_dir` is `None` or has no `*.ckpt` (existing `_has_checkpoint`).
2. `torch.cuda.is_available()` is `False` — degrade. Keeps offline/CI tests on
   the `{}` path.
3. `build_latest_inputs(symbols)` is empty (no requested symbol is in the tree).

Otherwise:

- Load the **IC-best** checkpoint:
  `register.load_base_model_ckpt(strict=False, time_version=False)`, then
  `.cuda().eval()`.
- For each symbol's input: wrap in `OHLCMulitClassPredictorInput`, forward under
  `torch.no_grad()`, and read the day-1 (`response_size=1`) predictions
  `predicted_r_close` / `predicted_upside` / `predicted_downside`.
- Emit the **raw model channels in log space** (the model's native target
  convention — least-lossy, no transform baked into the seam; a future consumer
  converts as needed).
- Return `{symbol: OphirForecast(symbol, r_close, upside, downside)}` for symbols
  that produced a forecast.

### Data flow

```
symbols
  -> build_latest_inputs (CPU)        -> {symbol: input_dict}
  -> IC-best model.forward (CUDA)     -> day-1 log-space channels
  -> {symbol: OphirForecast(...)}
```

## Minor cleanups (separate task group; 3 independent fixes)

### Cleanup 1 — checkpoint filename reflects the monitored metric

In `register._best_checkpoint_callback`, choose the filename suffix alongside the
monitor: `…best-{epoch:02d}-{val_rank_ic_near:.5f}` when `monitor_near_ic` is set,
else the current `…best-{epoch:02d}-{val_loss:.5f}`. The monitored metric is in
`callback_metrics` at save time, and `load_base_model_ckpt` / `_latest_base_ckpt`
match on the pre-`{` prefix (`file_name.split("{")[0]`), so checkpoint discovery
is unaffected. Fixes the misleading filename (a best-by-IC checkpoint currently
labelled with its drooped `val_loss`).

### Cleanup 2 — de-duplicate the `torch.cat` in `on_validation_epoch_end`

Hoist the buffer concatenations once (pred/target/ids/dates/offsets) and reuse
the resulting tensors across the `val_rank_ic`, `val_rank_ic_near`, and
`log_offset_ic` blocks, instead of re-`cat`-ing the same buffers up to three
times. Pure readability/DRY; behavior and logged values identical.

### Cleanup 3 — remove import-time `print()`s in `register.py`

Remove the module-import `print()` calls (the `current file path` /
`current dir path` lines) that emit to stdout on every `import ophir.register`.
Drop them, or route to `logging` at debug level.

## Testing

CPU-only, offline, deterministic (`filterwarnings=error`, mypy strict,
NumPy-style docstrings):

- **`build_latest_inputs`** via `parquet_dir`: returns one input per known symbol
  at `response_size=1`; unknown symbols are skipped; window length is `seq_len`;
  the per-symbol dict has the `extract_model_data` keys.
- **`load_forecasts` guard paths:** `model_dir=None` → `{}`; missing checkpoint →
  `{}`; present checkpoint on a CUDA-less host → `{}` without raising (the three
  existing `tests/test_trading_forecast.py` tests keep passing unchanged).
- **Cleanup 1:** `_best_checkpoint_callback(..., monitor_near_ic=True).filename`
  contains `val_rank_ic_near`; the `False` case contains `val_loss`.
- **Cleanup 2:** the existing `on_validation_epoch_end` tests
  (`test_on_validation_epoch_end_logs_near_ic_ungated`, the `val_rank_ic` tests)
  keep passing — values unchanged.

Not unit-tested (CUDA runtime): the model forward and day-1 extraction in
`load_forecasts`, exactly as `dashboard.run_evaluation`'s CUDA path is untested.

## Constraints

mypy strict / Python 3.10 floor; ruff 3.12; NumPy-style docstrings. pytest stays
offline + CPU-only. The system is paper-only; the safety gate (`trading/safety.py`)
is non-overridable and untouched — this spec only produces forecasts, it does not
gate or execute trades.

## Pointers

- Prior operating-point work: `docs/forecast-ceiling-results.md` (2026-06-23
  operating-point section); spec/plan
  `docs/superpowers/specs/2026-06-23-forecast-ceiling-operating-point-fix-design.md`,
  `docs/superpowers/plans/2026-06-23-forecast-ceiling-operating-point-fix.md`.
- Seam + consumer: `src/ophir/trading/forecast.py`, `src/ophir/trading/signals.py`.
- Inference pattern reference: `dashboard.run_evaluation`,
  `register.load_base_model_ckpt`, `ticker.extract_model_data` /
  `StockHanlder.keep_stocks`.
- Memory: `forecast-ceiling-investigation`.
