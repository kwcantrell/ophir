# Forecast consumer wiring — design

Wire the now-callable ophir forecast seam into the trading signal flow so a
forecast actually produces a `ProposedOrder`. Scope for this cycle:
**normalization + a thin orchestration command**, with momentum/sentiment
stubbed neutral and the safety gate left as a separate, explicit step.

## Background

`load_forecasts(symbols, model_dir)` returns per-symbol offset-1 forecasts
(`OphirForecast(symbol, r_close, upside, downside)` — raw log-space day-1
channels), but nothing consumes them. `blend_signals` / `normalize` exist in
`trading/signals.py` and are referenced only by their own unit tests. No
momentum/sentiment producers exist, and the trading CLI has no
signal-production / propose path. The job is to connect
`load_forecasts → normalize → blend_signals → ProposedOrder` end to end.

The model's measured skill is **cross-sectional** (rank-IC on near-horizon
`r_close`), which drives the normalization choice below.

## Components

Two units with a clean boundary:

1. **`signals.ophir_signals(forecasts) -> dict[str, float]`** — the pure, CPU,
   cross-sectional normalizer. Lives beside `blend_signals`/`normalize` in
   `trading/signals.py`. This is the genuinely testable core.
2. **`ophir trade propose` command** in `trading/cli.py` — the thin
   orchestrator. No new module; follows the existing file-in / JSON-out command
   shape.

## `ophir_signals` — the normalizer

Signature:

```python
def ophir_signals(forecasts: Mapping[str, OphirForecast]) -> dict[str, float]:
```

Algorithm — cross-sectional, demean + scale + clamp, ranking on `r_close`
alone:

```
xs   = [f.r_close for f in forecasts.values()]
mean = mean(xs)
std  = pstd(xs)                                   # population std
if std == 0:
    return {sym: 0.0 for sym in forecasts}        # no dispersion -> no signal
return {
    sym: clamp((f.r_close - mean) / std, -1.0, 1.0)
    for sym, f in forecasts.items()
}
```

- Empty input → `{}`.
- The `std == 0` guard subsumes the degenerate cases (single candidate, or an
  all-identical day): zero cross-sectional dispersion means no signal, so every
  score is `0.0`. No special-case branch beyond this guard.
- `clamp(z, -1, 1)` keeps the mapping **parameter-free** for the MVP: ±1σ maps
  to the extremes. If ±1σ saturation later proves too aggressive, the single
  documented spot to introduce a dispersion constant `k` is `z / k`.

### Deliberately deferred

- **`upside` / `downside` asymmetry.** `r_close` is the directional channel
  rank-IC was measured on. The up/down channels are raw log-space with
  conventions that need care (`upside.exp()` → high/close ratio ≥ 1;
  `downside` is stored *negated*, so `(-downside).exp()` → low/close ratio ≤ 1).
  Folding them into the score is an unproven refinement — out of scope.
- **Absolute / fixed-band calibration.** Cross-sectional matches the evidence;
  an absolute `normalize(value, lo, hi)` mapping with calibrated bounds is not
  built here.

## `ophir trade propose` — the orchestrator

CLI options:

- `--config` (Path) — `config.json` (loaded for consistency with the other
  commands; not used for sizing in the MVP).
- `--symbols` — comma-separated list, or a path to a file of symbols.
- `--model-dir` (Path) — checkpoint directory passed to `load_forecasts`.
- `--base-notional` (float, dollars) — sizing base (see Sizing).
- `--sleeve` (default `core`).
- `--min-abs-signal` (float, default `0.0`) — skip orders whose `|blended|` is
  at or below this; default skips only exact-neutral signals.

Flow:

1. `forecasts = load_forecasts(symbols, model_dir)` — degrades to `{}` (logged)
   when CUDA / checkpoint / data are unavailable.
2. `osig = ophir_signals(forecasts)`.
3. Per symbol in the requested set:
   `blended = blend_signals(ophir=osig.get(sym), momentum=0.0, sentiment=0.0,
   weights=CORE_WEIGHTS)`. A symbol absent from `osig` → `ophir=None` →
   `blend_signals` renormalizes onto the (neutral) momentum/sentiment → `0.0`
   → skipped by the `min_abs_signal` guard.
4. **Order construction** for surviving symbols:
   - `side = BUY if blended > 0 else SELL`
   - `notional = base_notional * abs(blended)`
   - `asset_class = EQUITY`, `sector = None`,
     `is_defined_risk = True`, `is_short_option = False`
   - skip if `abs(blended) <= min_abs_signal`
5. Emit a JSON array of `ProposedOrder` dicts to stdout — each dict is exactly
   what `gate --order` consumes. The command **does not call the gate or write
   the ledger**; that keeps the non-overridable safety gate a separate,
   explicit step.

### A documented property (not a bug)

With momentum/sentiment stubbed to neutral-but-present (`0.0` value, active
weight), `blend_signals` shrinks the ophir score by its CORE weight: the
denominator still includes the inactive sleeves, so `blended = 0.6 * ophir`
under `CORE_WEIGHTS`. This dampening is conservative and acceptable for the
MVP; it is documented rather than hidden, and resolves naturally when real
momentum/sentiment producers land.

## Sizing

`notional = base_notional * abs(blended)`.

This keeps the proposer **decoupled from the account snapshot** and leaves all
limit enforcement to the gate (which resizes/rejects against equity). The
rejected alternative — sizing as a fraction of equity — would force the
proposer to take the snapshot too, duplicating the gate's authority over
limits.

## Degradation & gate boundary

- Forecasts unavailable → `load_forecasts` returns `{}` → every `ophir` is
  `None` → every `blended` is `0.0` → no orders emitted (empty JSON array).
  The command never raises on unavailability.
- The safety gate (`trading/safety.py`) stays non-overridable and is invoked
  separately (`ophir trade gate`). The propose command produces input *for* the
  gate; it does not bypass or pre-empt it.

## Testing

- `ophir_signals` — real unit tests: cross-sectional spread/sign, `std == 0`
  → all zeros, single symbol → `0.0`, empty → `{}`, clamping beyond ±1σ.
- `propose` command — tested with `load_forecasts` mocked (and the `{}` path)
  so it stays **offline + CPU-only**; the CUDA forward is never invoked in
  tests. Cover: order construction (side from sign, notional from
  `base_notional * |blended|`), `min_abs_signal` skipping, and the empty-forecast
  → empty-array path.
- End-to-end CUDA inference stays out of the suite, matching the existing seam
  boundary.

## Out of scope

- Momentum / sentiment producers (stubbed neutral here).
- Calling the gate, writing the ledger, real sizing-vs-equity.
- `upside`/`downside` asymmetry and absolute-band calibration.

## Runtime precondition (not a test blocker)

Live forecasts require a **90-day IC-best checkpoint** in `register.MODEL_DIR`
(`load_forecasts` loads via `register.load_base_model_ckpt(time_version=False)`).
Task-5 short-horizon runs (`response_size=10`) are not the 90-day production
model. The command degrades cleanly to "no orders" without a suitable
checkpoint, so this does not gate the implementation — but a
`ophir train --val-identity` 90-day run may be needed before relying on live
forecasts.

## Constraints

- mypy strict / Python 3.10 floor; ruff 3.12; NumPy-style docstrings.
- pytest stays offline + CPU-only (`filterwarnings = error`); never touch
  network / CUDA / `.ophir/`.
- Update `CHANGELOG.md` `[Unreleased]`.
