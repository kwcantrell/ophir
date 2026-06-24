# Momentum Producer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic momentum signal producer in `ophir.trading` and wire it into `ophir trade propose`, replacing the `momentum=0.0` stub.

**Architecture:** A new `trading/momentum.py` module mirrors the forecast seam: a CPU/offline data loader (`load_recent_closes`) plus pure metrics (`momentum_score`, `momentum_signals`). Cross-sectional normalization is shared with `ophir_signals` via a `cross_sectional_normalize` helper extracted into `signals.py`. Sentiment stays the alpaca-skill's LLM judgment (out of scope); the 90-day checkpoint is a separate ops task (out of scope).

**Tech Stack:** Python 3.10+, Typer CLI, pytest + `typer.testing.CliRunner`, `statistics`/`math` stdlib, pandas (only inside the data loader, via `ticker`).

## Global Constraints

- mypy is `strict = True`, targets Python 3.10 — keep `src/ophir` fully typed.
- ruff targets 3.12; run `uv run ruff check .` and `uv run ruff format --check .`.
- pytest runs `filterwarnings = error`; tests must stay **offline + CPU-only** and never touch network/CUDA/`.ophir/`. Use `tmp_path`, `monkeypatch`, and the seeded `tests/conftest.py` fixtures.
- NumPy-style docstrings throughout `src/ophir`, matching existing density.
- Imports: `known-first-party = ["ophir"]` ordering.
- Update the `[Unreleased]` section of `CHANGELOG.md`.
- Run tests with `uv run pytest`; single file via `uv run pytest tests/test_<name>.py`.
- Data correctness principle: momentum must see the **same prices the model does** — reuse `ticker.StockHanlder.stock_df`, never call network `get_splits`.

---

### Task 1: Extract `cross_sectional_normalize`; refactor `ophir_signals`

**Files:**
- Modify: `src/ophir/trading/signals.py`
- Test: `tests/test_trading_signals.py` (append)

**Interfaces:**
- Produces: `cross_sectional_normalize(values: Mapping[str, float]) -> dict[str, float]` — demean / divide by population std / clamp to `[-1, 1]`; empty → `{}`; std == 0 → all `0.0`.
- `ophir_signals(forecasts: Mapping[str, OphirForecast]) -> dict[str, float]` keeps its existing signature and behavior, now delegating to the helper.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_trading_signals.py`. Add `cross_sectional_normalize` to the existing `from ophir.trading.signals import (...)` block, then add:

```python
def test_cross_sectional_normalize_empty() -> None:
    assert cross_sectional_normalize({}) == {}


def test_cross_sectional_normalize_single_is_neutral() -> None:
    assert cross_sectional_normalize({"A": 0.05}) == {"A": 0.0}


def test_cross_sectional_normalize_all_identical_is_neutral() -> None:
    assert cross_sectional_normalize({"A": 0.01, "B": 0.01}) == {"A": 0.0, "B": 0.0}


def test_cross_sectional_normalize_sign_and_clamp() -> None:
    out = cross_sectional_normalize({"HI": 0.05, "MID": 0.0, "LO": -0.05})
    assert out["HI"] > 0.0
    assert out["LO"] < 0.0
    assert out["MID"] == pytest.approx(0.0)
    assert out["HI"] == pytest.approx(-out["LO"])
    assert all(-1.0 <= v <= 1.0 for v in out.values())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_trading_signals.py -v`
Expected: FAIL — `ImportError: cannot import name 'cross_sectional_normalize'`.

- [ ] **Step 3: Implement the helper and refactor `ophir_signals`**

In `src/ophir/trading/signals.py`, add `cross_sectional_normalize` (place it just above `ophir_signals`) and replace the body of `ophir_signals` to delegate. The final two functions read:

```python
def cross_sectional_normalize(values: Mapping[str, float]) -> dict[str, float]:
    """Cross-sectionally map per-key values into ``[-1, 1]``.

    Demeans by the cross-sectional mean, divides by the population standard
    deviation, and clamps to ``[-1, 1]``.

    Parameters
    ----------
    values : mapping of str to float
        One raw score per key (e.g. per symbol) for the day's candidate set.

    Returns
    -------
    dict[str, float]
        Per-key score in ``[-1, 1]``. Empty input yields ``{}``. When the
        cross-sectional dispersion is zero (a single key, or all-identical
        values), every score is ``0.0`` — no dispersion, no signal.
    """
    if not values:
        return {}
    xs = list(values.values())
    mean = fmean(xs)
    std = pstdev(xs)
    if std == 0.0:
        return dict.fromkeys(values, 0.0)
    return {key: max(-1.0, min(1.0, (value - mean) / std)) for key, value in values.items()}


def ophir_signals(forecasts: Mapping[str, OphirForecast]) -> dict[str, float]:
    """Cross-sectionally score per-symbol forecasts into ``[-1, 1]``.

    Ranks the day's candidates on ``r_close`` via
    :func:`cross_sectional_normalize`. The model's measured skill is
    cross-sectional (rank-IC), so the score is relative to the other candidates
    rather than an absolute return.

    Parameters
    ----------
    forecasts : mapping of str to OphirForecast
        Per-symbol forecasts for the day's candidate set.

    Returns
    -------
    dict[str, float]
        Per-symbol score in ``[-1, 1]``; ``{}`` for empty input, all-``0.0`` when
        the cross-sectional dispersion is zero.
    """
    return cross_sectional_normalize({symbol: f.r_close for symbol, f in forecasts.items()})
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_trading_signals.py -v`
Expected: PASS — the new `cross_sectional_normalize` tests AND every pre-existing `ophir_signals` test (behavior is preserved).

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy src/ophir && uv run ruff check src/ophir/trading/signals.py tests/test_trading_signals.py && uv run ruff format --check src/ophir/trading/signals.py tests/test_trading_signals.py`
Expected: no errors. (If format check fails, run `uv run ruff format <file>` and re-run.)

- [ ] **Step 6: Commit**

```bash
git add src/ophir/trading/signals.py tests/test_trading_signals.py
git commit -m "refactor: extract cross_sectional_normalize from ophir_signals"
```

---

### Task 2: `momentum_score` (pure metric)

**Files:**
- Create: `src/ophir/trading/momentum.py`
- Test: `tests/test_trading_momentum.py` (new)

**Interfaces:**
- Produces: `momentum_score(closes: Sequence[float], lookback: int = 63, skip: int = 5) -> float | None` — information ratio (`mean/std`, sample std) of daily log returns over the window of `lookback` returns ending `skip` bars before the latest close. `None` on insufficient history (`len < lookback + skip + 1`) or zero return-variance.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_trading_momentum.py`:

```python
from ophir.trading.momentum import momentum_score


def _series(n: int, drift: float, base: float = 100.0, noise: float = 0.003) -> list[float]:
    """Deterministic price path with ``drift`` mean daily return plus alternating
    noise, so daily returns *vary* (nonzero variance — a pure geometric path has
    constant returns and an undefined information ratio)."""
    closes = [base]
    for i in range(1, n):
        ret = drift + (noise if i % 2 == 0 else -noise)
        closes.append(closes[-1] * (1.0 + ret))
    return closes


def test_momentum_score_rising_is_positive() -> None:
    assert (momentum_score(_series(80, 0.01), lookback=63, skip=5) or 0.0) > 0.0


def test_momentum_score_falling_is_negative() -> None:
    assert (momentum_score(_series(80, -0.01), lookback=63, skip=5) or 0.0) < 0.0


def test_momentum_score_constant_series_is_none() -> None:
    # Zero variance -> undefined information ratio.
    assert momentum_score([100.0] * 80, lookback=63, skip=5) is None


def test_momentum_score_insufficient_history_is_none() -> None:
    # Need lookback + skip + 1 = 69 closes; 68 is too few.
    assert momentum_score(_series(68, 0.01), lookback=63, skip=5) is None


def test_momentum_score_skip_excludes_recent_spike() -> None:
    base = _series(80, 0.01)
    spiked = base[:-3] + [base[-3] * 0.5, base[-2] * 0.5, base[-1] * 0.5]
    # The crash lands inside the skipped 5-bar tail, so the score is unchanged.
    assert momentum_score(spiked, lookback=63, skip=5) == momentum_score(
        base, lookback=63, skip=5
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_trading_momentum.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ophir.trading.momentum'`.

- [ ] **Step 3: Implement `momentum_score`**

Create `src/ophir/trading/momentum.py`:

```python
"""Deterministic momentum signal producer for the trading core.

Mirrors the forecast seam (:mod:`ophir.trading.forecast`): a CPU/offline data
loader plus pure metrics. Momentum is a mechanical function of recent price
bars, so it lives as a reproducible primitive rather than an LLM judgment.
"""

from collections.abc import Sequence
from math import log
from statistics import fmean, stdev


def momentum_score(
    closes: Sequence[float], lookback: int = 63, skip: int = 5
) -> float | None:
    """Information ratio of daily log returns over a skip-adjusted window.

    Over the window of ``lookback`` daily returns ending ``skip`` bars before the
    latest close, returns ``mean / std`` of the daily log returns (sample std,
    ``ddof=1``). The ``skip`` excludes the most-recent ``skip`` bars, whose
    returns are reversal-prone; including them would load the signal on reversal.

    Parameters
    ----------
    closes : sequence of float
        Positive daily closing prices, oldest first.
    lookback : int, optional
        Number of daily returns in the window. Defaults to ``63``.
    skip : int, optional
        Number of most-recent bars to exclude. Defaults to ``5``.

    Returns
    -------
    float or None
        The information ratio, or ``None`` when there is too little history
        (``len(closes) < lookback + skip + 1``) or the window's return variance
        is zero.
    """
    if len(closes) < lookback + skip + 1:
        return None
    end = len(closes) - skip
    start = end - lookback - 1
    window = closes[start:end]
    rets = [log(window[i] / window[i - 1]) for i in range(1, len(window))]
    spread = stdev(rets)
    if spread == 0.0:
        return None
    return fmean(rets) / spread
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_trading_momentum.py -v`
Expected: PASS (all five tests).

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy src/ophir && uv run ruff check src/ophir/trading/momentum.py tests/test_trading_momentum.py && uv run ruff format --check src/ophir/trading/momentum.py tests/test_trading_momentum.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/ophir/trading/momentum.py tests/test_trading_momentum.py
git commit -m "feat: add momentum_score information-ratio metric"
```

---

### Task 3: `momentum_signals` (cross-sectional)

**Files:**
- Modify: `src/ophir/trading/momentum.py`
- Test: `tests/test_trading_momentum.py` (append)

**Interfaces:**
- Consumes: `momentum_score` (Task 2); `cross_sectional_normalize` from `ophir.trading.signals` (Task 1).
- Produces: `momentum_signals(closes_by_symbol: Mapping[str, Sequence[float]], lookback: int = 63, skip: int = 5) -> dict[str, float]` — per-symbol momentum scored cross-sectionally into `[-1, 1]`; symbols whose `momentum_score` is `None` are omitted; empty / all-`None` → `{}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_trading_momentum.py`. Add this import at the top of the file:

```python
from ophir.trading.momentum import momentum_score, momentum_signals
```

(replace the existing single-name import), then append:

```python
def test_momentum_signals_cross_sectional_sign() -> None:
    out = momentum_signals(
        {"UP": _series(80, 0.01), "DOWN": _series(80, -0.01)}, lookback=63, skip=5
    )
    assert out["UP"] > 0.0
    assert out["DOWN"] < 0.0
    assert out["UP"] == pytest.approx(-out["DOWN"])


def test_momentum_signals_drops_short_history() -> None:
    short = _series(40, 0.01)  # < 69 closes -> momentum_score is None -> dropped
    out = momentum_signals({"UP": _series(80, 0.01), "SHORT": short}, lookback=63, skip=5)
    assert "SHORT" not in out
    # Only one survivor -> zero cross-sectional dispersion -> neutral.
    assert out == {"UP": 0.0}


def test_momentum_signals_empty_and_all_none() -> None:
    assert momentum_signals({}, lookback=63, skip=5) == {}
    assert momentum_signals({"SHORT": _series(40, 0.01)}, lookback=63, skip=5) == {}
```

Add `import pytest` at the top of the test file if it is not already imported.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_trading_momentum.py -v`
Expected: FAIL — `ImportError: cannot import name 'momentum_signals'`.

- [ ] **Step 3: Implement `momentum_signals`**

In `src/ophir/trading/momentum.py`, add the import and the function. Update the imports block to add `Mapping` and the signals helper:

```python
from collections.abc import Mapping, Sequence
from math import log
from statistics import fmean, stdev

from ophir.trading.signals import cross_sectional_normalize
```

Then append:

```python
def momentum_signals(
    closes_by_symbol: Mapping[str, Sequence[float]],
    lookback: int = 63,
    skip: int = 5,
) -> dict[str, float]:
    """Cross-sectionally score per-symbol momentum into ``[-1, 1]``.

    Computes :func:`momentum_score` per symbol, drops symbols whose score is
    ``None`` (insufficient history or zero variance), and normalizes the
    survivors with :func:`ophir.trading.signals.cross_sectional_normalize`.

    Parameters
    ----------
    closes_by_symbol : mapping of str to sequence of float
        Per-symbol daily closing prices, oldest first.
    lookback : int, optional
        Window length in daily returns. Defaults to ``63``.
    skip : int, optional
        Most-recent bars to exclude. Defaults to ``5``.

    Returns
    -------
    dict[str, float]
        Per-symbol momentum score in ``[-1, 1]``; symbols without a defined
        score are omitted. Empty input or all-``None`` yields ``{}``.
    """
    raw: dict[str, float] = {}
    for symbol, closes in closes_by_symbol.items():
        score = momentum_score(closes, lookback=lookback, skip=skip)
        if score is not None:
            raw[symbol] = score
    return cross_sectional_normalize(raw)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_trading_momentum.py -v`
Expected: PASS (all eight tests).

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy src/ophir && uv run ruff check src/ophir/trading/momentum.py tests/test_trading_momentum.py && uv run ruff format --check src/ophir/trading/momentum.py tests/test_trading_momentum.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/ophir/trading/momentum.py tests/test_trading_momentum.py
git commit -m "feat: add momentum_signals cross-sectional scorer"
```

---

### Task 4: `load_recent_closes` (data seam)

**Files:**
- Modify: `src/ophir/trading/momentum.py`
- Test: `tests/test_trading_momentum.py` (append)

**Interfaces:**
- Produces: `load_recent_closes(symbols: Sequence[str], base_path: str | None = None) -> dict[str, list[float]]` — per-symbol daily closes (oldest first) read via `ticker.StockHanlder.stock_df`; `{}` when the tree is absent; symbols absent from the tree (or yielding an empty frame) are skipped.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_trading_momentum.py`. Add this import at the top of the file:

```python
from ophir.trading.momentum import load_recent_closes, momentum_score, momentum_signals
```

(replace the existing momentum import), then append:

```python
def test_load_recent_closes_reads_full_history(parquet_dir) -> None:
    from ophir.ticker import StockHanlder

    base_path, _paths = parquet_dir
    result = load_recent_closes(["AAA", "ZZZ"], base_path)
    # Present symbol loads; absent symbol is skipped.
    assert "AAA" in result
    assert "ZZZ" not in result
    # Matches the model's own read path exactly.
    handler = StockHanlder(
        seq_len=365, base_path=base_path, return_stock_id=False, return_streamer=False
    )
    expected = [float(c) for c in handler.stock_df("AAA")["close"].tolist()]
    assert result["AAA"] == expected
    assert all(isinstance(c, float) for c in result["AAA"])


def test_load_recent_closes_missing_tree_returns_empty(tmp_path) -> None:
    missing = tmp_path / "nope" / "stocks"
    assert load_recent_closes(["AAA"], str(missing)) == {}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_trading_momentum.py -v`
Expected: FAIL — `ImportError: cannot import name 'load_recent_closes'`.

- [ ] **Step 3: Implement `load_recent_closes`**

In `src/ophir/trading/momentum.py`, add `import os` to the top of the module (above the `from collections.abc` line), then append the function:

```python
def load_recent_closes(
    symbols: Sequence[str], base_path: str | None = None
) -> dict[str, list[float]]:
    """Load per-symbol daily closes from the model's parquet read path.

    Reuses :meth:`ophir.ticker.StockHanlder.stock_df` — the same accessor the
    inference seam (:func:`ophir.ticker.build_latest_inputs`) reaches — so
    momentum sees the identical daily-aggregated, cleaned closes the model does.
    No split adjustment / ``get_splits`` is performed (that is network-bound and
    not part of the inference read path).

    Parameters
    ----------
    symbols : sequence of str
        Ticker symbols to load.
    base_path : str or None, optional
        Root of the Hive-partitioned parquet tree. ``None`` resolves to
        ``register.get_default_data_days_dir()/stocks``.

    Returns
    -------
    dict[str, list[float]]
        ``{symbol: [close, ...]}`` (oldest first) for each available symbol.
        ``{}`` when the tree is absent; symbols missing from the tree or yielding
        an empty frame are skipped.
    """
    if base_path is None:
        from ophir import register

        base_path = os.path.join(register.get_default_data_days_dir(), "stocks")
    if not os.path.isdir(base_path):
        return {}

    from ophir.ticker import StockHanlder

    handler = StockHanlder(
        seq_len=365, base_path=base_path, return_stock_id=False, return_streamer=False
    )
    out: dict[str, list[float]] = {}
    for symbol in symbols:
        try:
            frame = handler.stock_df(symbol)
        except (KeyError, ValueError, FileNotFoundError, OSError):
            continue
        if frame is None or len(frame) == 0 or "close" not in frame.columns:
            continue
        out[symbol] = [float(close) for close in frame["close"].tolist()]
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_trading_momentum.py -v`
Expected: PASS (all ten tests).

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy src/ophir && uv run ruff check src/ophir/trading/momentum.py tests/test_trading_momentum.py && uv run ruff format --check src/ophir/trading/momentum.py tests/test_trading_momentum.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/ophir/trading/momentum.py tests/test_trading_momentum.py
git commit -m "feat: add load_recent_closes reusing the model parquet read path"
```

---

### Task 5: Wire momentum into `ophir trade propose`

**Files:**
- Modify: `src/ophir/trading/cli.py`
- Modify: `CHANGELOG.md`
- Test: `tests/test_trading_propose.py` (append)

**Interfaces:**
- Consumes: `momentum.load_recent_closes`, `momentum.momentum_signals` (Tasks 3–4); existing `blend_signals`, `ophir_signals`, `forecast.load_forecasts`, `ProposedOrder`, etc.
- Produces: `ophir trade propose` now blends a real momentum component; new options `--base-path`, `--momentum-lookback` (default `63`), `--momentum-skip` (default `5`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_trading_propose.py`. At the top of the file add a helper near `_fc`, and an **autouse fixture** that keeps every pre-existing propose test hermetic — after this task `propose` calls `load_recent_closes`, which without a stub would read a machine-dependent data tree:

```python
def _series(n: int, drift: float, base: float = 100.0, noise: float = 0.003) -> list[float]:
    closes = [base]
    for i in range(1, n):
        ret = drift + (noise if i % 2 == 0 else -noise)
        closes.append(closes[-1] * (1.0 + ret))
    return closes


@pytest.fixture(autouse=True)
def _no_momentum_data(monkeypatch: pytest.MonkeyPatch) -> None:
    # Default: no momentum closes (offline + deterministic). The momentum test
    # below overrides this with its own monkeypatch, which takes precedence.
    monkeypatch.setattr(
        "ophir.trading.momentum.load_recent_closes", lambda symbols, base_path: {}
    )
```

Then append the tests:

```python
def test_propose_momentum_drives_order_when_ophir_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config_file(tmp_path)
    # No ophir forecasts at all -> ophir component is None.
    monkeypatch.setattr(
        "ophir.trading.forecast.load_forecasts", lambda symbols, model_dir: {}
    )
    monkeypatch.setattr(
        "ophir.trading.momentum.load_recent_closes",
        lambda symbols, base_path: {
            "UP": _series(80, 0.01),
            "DOWN": _series(80, -0.01),
        },
    )
    result = runner.invoke(
        app,
        [
            "propose",
            "--symbols", "UP,DOWN",
            "--model-dir", str(tmp_path),
            "--base-notional", "1000",
            "--config", str(cfg),
        ],
    )
    assert result.exit_code == 0, result.output
    orders = {o["symbol"]: o for o in json.loads(result.stdout)}
    assert orders["UP"]["side"] == "buy"
    assert orders["DOWN"]["side"] == "sell"
    # ophir=None -> blend over momentum(0.25)+sentiment(0.15); momentum=+/-1 ->
    # blended = 0.25/0.40 = 0.625 -> notional = 1000 * 0.625.
    assert orders["UP"]["notional"] == pytest.approx(625.0)


def test_propose_empty_when_no_signals_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config_file(tmp_path)
    monkeypatch.setattr(
        "ophir.trading.forecast.load_forecasts", lambda symbols, model_dir: {}
    )
    monkeypatch.setattr(
        "ophir.trading.momentum.load_recent_closes", lambda symbols, base_path: {}
    )
    result = runner.invoke(
        app,
        [
            "propose",
            "--symbols", "UP,DOWN",
            "--model-dir", str(tmp_path),
            "--base-notional", "1000",
            "--config", str(cfg),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_trading_propose.py -v`
Expected: FAIL — `test_propose_momentum_drives_order_when_ophir_absent` fails because momentum is still stubbed `0.0`, so with `ophir=None` the blend is `0.0` and no orders are emitted.

- [ ] **Step 3: Wire momentum into the command**

In `src/ophir/trading/cli.py`:

(a) Add to the imports near the other `from ophir.trading ...` lines (if `from ophir.trading import forecast` already exists, replace it with the combined line), and add `Optional` from `typing` at the top of the file:

```python
from typing import Optional

from ophir.trading import forecast, momentum
```

(b) Add three options to the `propose` signature, after `min_abs_signal`. Use `Optional[Path]` (Typer-compatible; matches the strict-mypy/3.10 floor):

```python
    base_path: Optional[Path] = typer.Option(
        None, help="Parquet tree root for momentum closes; default tree if unset"
    ),
    momentum_lookback: int = typer.Option(63, help="Momentum window length (daily returns)"),
    momentum_skip: int = typer.Option(5, help="Most-recent bars excluded from momentum"),
```

(c) After the `scores = ophir_signals(forecasts)` line, load and score momentum:

```python
    closes = momentum.load_recent_closes(
        names, None if base_path is None else str(base_path)
    )
    msig = momentum.momentum_signals(
        closes, lookback=momentum_lookback, skip=momentum_skip
    )
```

(d) Replace the `momentum=0.0,` argument in the `blend_signals(...)` call with:

```python
            momentum=msig.get(symbol, 0.0),
```

(e) Update the command docstring's "blend with neutral momentum/sentiment" phrasing to reflect that momentum is now real:

```python
    """Emit ProposedOrder JSON from ophir + momentum signals (no gate, no ledger).

    Runs the seam end to end: load forecasts and recent closes, cross-sectionally
    normalize each, blend (sentiment stubbed neutral), and size by
    ``base_notional``. Prints a JSON array of proposed orders for piping into
    ``gate``. Degrades to an empty array when no signals are available; never
    invokes the safety gate or writes the ledger.
    """
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_trading_propose.py -v`
Expected: PASS (all tests, including the pre-existing ones). The `_no_momentum_data` autouse fixture stubs `load_recent_closes` to `{}` for every test, so the pre-existing `0.6 * ophir` assertions (e.g. `test_propose_emits_orders_with_side_and_notional` → `notional == 600`) are unaffected by the new momentum component and stay hermetic regardless of the host's data tree.

- [ ] **Step 5: Update the CHANGELOG**

In `CHANGELOG.md`, under `## [Unreleased]` → `### Added`, add at the top of the list:

```markdown
- Deterministic momentum signal producer (`ophir.trading.momentum`): an
  information-ratio momentum metric over recent price bars (`momentum_score`),
  cross-sectional scoring (`momentum_signals`), and a `load_recent_closes` data
  seam reusing the model's parquet read path. Wired into `ophir trade propose`
  (new `--base-path` / `--momentum-lookback` / `--momentum-skip` options),
  replacing the neutral momentum stub. Sentiment remains the skill's judgment.
  Also extracts `signals.cross_sectional_normalize`, now shared by the ophir and
  momentum scorers.
```

- [ ] **Step 6: Full suite, typecheck, lint**

Run: `uv run pytest && uv run mypy src/ophir && uv run ruff check . && uv run ruff format --check .`
Expected: all pass. (If format check fails, run `uv run ruff format .` and re-run.)

- [ ] **Step 7: Commit**

```bash
git add src/ophir/trading/cli.py tests/test_trading_propose.py CHANGELOG.md
git commit -m "feat: blend real momentum into 'ophir trade propose'"
```

---

## Notes for the implementer

- **Why `0.625` in the propose test:** with `ophir=None`, `blend_signals` renormalizes over the active momentum (0.25) and sentiment (0.15) weights only; momentum cross-sectionally saturates at `±1` for two symbols, so `blended = 0.25/0.40 = 0.625`. Sentiment stays `0.0`.
- **No CUDA / network in tests:** every propose test monkeypatches `ophir.trading.forecast.load_forecasts` AND `ophir.trading.momentum.load_recent_closes`. The `momentum.py` data loader is offline (parquet via `StockHanlder`) and is tested directly against the `parquet_dir` conftest fixture.
- **Behavior-preserving refactor (Task 1):** `ophir_signals` must keep passing its existing tests unchanged — only its internals move into `cross_sectional_normalize`.
- **`StockHanlder` spelling is intentional** (the class is `StockHanlder`, not `StockHandler`) — match the existing code.
- **Out of scope:** sentiment producer (stays in the skill), the 90-day checkpoint (separate ops task), feeding momentum back into `morning.js`.
