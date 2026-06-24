# Deterministic momentum producer — design

Build the momentum signal component as a deterministic, offline-testable
`ophir.trading` primitive and wire it into `ophir trade propose`, replacing the
`momentum=0.0` stub. Sentiment stays the alpaca-skill's LLM judgment (out of
scope); the 90-day checkpoint precondition is a separate operational task (out
of scope).

## Background

`ophir trade propose` (just landed) blends `ophir + momentum + sentiment` via
`signals.blend_signals`, but momentum and sentiment are stubbed `0.0`, so the
emitted order is effectively `0.6 * ophir` (CORE) and `0.0` whenever no
checkpoint is present. No momentum producer exists anywhere in the deterministic
core. Price bars *are* available (the Hive-partitioned parquet tree under
`register.get_default_data_days_dir()/stocks`, the same source
`ticker.build_latest_inputs` reads).

### Architecture decision — layered, not competing

The deterministic `ophir.trading` core and the LLM-driven `alpaca-trader` skill
(`workflows/morning.js`) are two layers, not rival proposers:

- **Objective signals (ophir, momentum)** → deterministic, testable
  `ophir.trading` primitives. Momentum is a mechanical function of price bars;
  it should be reproducible, not eyeballed.
- **Subjective signal (sentiment)** → stays the skill's LLM judgment over
  `get_news` (`lib/signals.md`: "soft signal from `get_news`"). No deterministic
  sentiment unit is built; a Python sentiment producer would need an external
  feed or an embedded model and could not stay offline/CPU-testable.

The momentum number is a *shared primitive*: `morning.js` can later consume the
reproducible value instead of reading bars by eye (`lib/signals.md` already
describes momentum as "from recent bars"). Surfacing it to the skill is out of
scope here, but the seam makes it possible.

## Components

New module `src/ophir/trading/momentum.py`, mirroring the forecast seam
(`trading/forecast.py`): a data loader plus pure metrics. Cross-sectional
normalization is shared with `ophir_signals` via a helper extracted into
`signals.py`.

### 1. `momentum_score` (pure)

```python
def momentum_score(
    closes: Sequence[float], lookback: int = 63, skip: int = 5
) -> float | None:
```

Over the window that **ends `skip` bars before the latest close and spans
`lookback` daily returns** (i.e. `lookback + 1` consecutive closes ending at
index `len - 1 - skip`), compute the **information ratio of daily log returns**:
`mean(daily_logret) / std(daily_logret)` (sample std, `ddof=1`).

- `closes` is oldest → newest, split-adjusted.
- The `skip` excludes the most-recent `skip` bars, whose returns are
  reversal-prone (the ceiling work measured naive 1-day reversal IC ≈ +0.05;
  including them would load the signal on reversal — anti-momentum).
- Returns `None` when there is too little history (`len(closes) <
  lookback + skip + 1`, the `+1` because daily returns need a predecessor) or
  when the window's return std is `0.0` (no variance → undefined ratio).

**Why a ratio, not a raw return:** dividing by each symbol's *own* realized
volatility controls for a high-vol name posting large returns from noise.
Cross-sectional normalization (below) only normalizes the spread *across* names,
not each name's own vol — so the two steps are complementary, not redundant.

### 2. `cross_sectional_normalize` (shared, extracted from `ophir_signals`)

```python
def cross_sectional_normalize(values: Mapping[str, float]) -> dict[str, float]:
```

The demean / divide-by-population-std / clamp-to-`[-1,1]` mapping, lifted out of
the current `ophir_signals` body. Contract unchanged: empty input → `{}`;
zero cross-sectional dispersion (`pstdev == 0.0`: single value or all-identical)
→ every value `0.0`. `ophir_signals` is refactored to build `{symbol: r_close}`
and delegate to this helper — behavior identical, existing tests stay green.
This is the targeted DRY improvement to code we are already touching. Lives in
`signals.py` (the home of normalization).

### 3. `momentum_signals` (pure)

```python
def momentum_signals(
    closes_by_symbol: Mapping[str, Sequence[float]],
    lookback: int = 63,
    skip: int = 5,
) -> dict[str, float]:
```

Computes `momentum_score` per symbol, drops the `None`s (insufficient history /
zero variance), and runs the survivors through `cross_sectional_normalize`.
A symbol that scores `None` is omitted from the result (the caller treats a
missing symbol as neutral). Empty input or all-`None` → `{}`.

### 4. `load_recent_closes` (data seam, CPU/offline)

```python
def load_recent_closes(
    symbols: Sequence[str], base_path: str | None = None
) -> dict[str, list[float]]:
```

Reads daily closes per symbol (oldest → newest) by **reusing the exact read
path the model's inference seam uses** — `ticker.StockHanlder.stock_df(symbol)`,
the same accessor reached by `build_latest_inputs` — and taking its `close`
column. That path reads the parquet, daily-aggregates (`groupby("date").agg(close="last")`),
and applies `StockHanlder`'s default cleaning (`clean_rows=False`, matching
`build_latest_inputs`). **Correctness principle: momentum must see the same
prices the model does, not a separately-adjusted series.** The tree is read
as-stored; no live `get_splits`/`StockSplit` call is made (that hits the network
and is *not* part of the inference read path). Any split artifact is therefore a
known limitation shared identically with the model — not something momentum
corrects here.

`base_path=None` resolves to `register.get_default_data_days_dir()/stocks`,
matching `build_latest_inputs`. Lazy-imports `ophir.ticker`/`register` inside the
function (keeps module import cheap and offline, like `forecast.py`). Degrades to
`{}` when the tree is absent, and skips individual symbols that are absent from
the tree or yield an empty frame (e.g. `StockHanlder`'s history/volume filters);
it does not raise on missing data. Unlike `load_forecasts` (CUDA), this is
CPU + offline, so it is unit-tested against the `parquet_dir` conftest fixture.

## Wiring into `ophir trade propose`

In `trading/cli.py`, the `propose` command gains:

- `--base-path` (Path, optional) — parquet tree root for momentum closes;
  `None` → default tree.
- `--momentum-lookback` (int, default `63`).
- `--momentum-skip` (int, default `5`).

Flow change — momentum is no longer stubbed:

```
closes = load_recent_closes(names, base_path)
msig   = momentum_signals(closes, lookback=momentum_lookback, skip=momentum_skip)
...
blended = blend_signals(
    ophir=scores.get(symbol),
    momentum=msig.get(symbol, 0.0),   # missing/insufficient -> neutral
    sentiment=0.0,                    # still the skill's job
    weights=weights,
)
```

Degradation is preserved: closes unavailable → `msig == {}` → momentum neutral
→ the command behaves exactly as today (ophir-only, or empty when ophir is also
absent). Sentiment remains `0.0`.

## Error handling & degradation

- `momentum_score` never raises on thin/flat data — it returns `None`.
- `load_recent_closes` degrades to `{}` / skips symbols on missing data; it does
  not raise on the absence of the tree or a symbol.
- `propose` continues to emit `[]` when nothing produces a non-neutral signal.
- No path bypasses or weakens the safety gate; `propose` still does not call the
  gate or write the ledger.

## Testing

All CPU + offline (`filterwarnings = error`); never touch network/CUDA/`.ophir/`.

- `momentum_score`: rising series → positive, falling → negative, a constant
  (zero-variance) series → `None`, a zero-drift noisy series → `~0`, insufficient
  history → `None`, and a test proving the `skip` excludes a planted recent spike
  (a reversal spike in the last `skip` bars must not change the score).
- `cross_sectional_normalize`: the relocated demean/scale/clamp tests (empty,
  single, all-identical, cross-sectional sign, clamp); `ophir_signals` tests
  stay green unchanged (behavior-preserving refactor).
- `momentum_signals`: cross-sectional sign across symbols, `None`-drop (a
  short-history symbol is absent from the result), empty/all-`None` → `{}`.
- `load_recent_closes`: the `parquet_dir` conftest fixture → closes for a
  full-history symbol (`AAA`, oldest→newest, matching its daily-aggregated
  `close` column); a symbol absent from the tree (e.g. `"ZZZ"`) is skipped while
  present symbols still load; a missing/absent tree → `{}`. (Defaults match
  `build_latest_inputs` — no volume/history filtering at load; a too-short series
  still loads and is dropped downstream by `momentum_score` → `None`.)
- `propose`: monkeypatch `load_recent_closes` (and `load_forecasts`) so a symbol
  with rising prices and **no** ophir forecast still produces a correctly-signed,
  correctly-sized order driven by momentum alone; and the all-unavailable path
  still emits `[]`.

## Out of scope

- Sentiment producer (stays the alpaca-skill's LLM judgment).
- The 90-day IC-best checkpoint (separate operational task — verify
  `register.MODEL_DIR`, else a short `ophir train --val-identity` run).
- Feeding the momentum number back into `morning.js` (seam enables it later).

## Constraints

- mypy `strict = True`, Python 3.10 floor; ruff 3.12; NumPy-style docstrings.
- pytest `filterwarnings = error`; offline + CPU-only; never touch
  network/CUDA/`.ophir/`.
- The safety gate is non-overridable; the system is paper-only. Nothing here
  touches either.
- Update `CHANGELOG.md` `[Unreleased]`.
