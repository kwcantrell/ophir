# Forecast Consumer Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the ophir forecast seam into the trading signal flow so forecasts produce `ProposedOrder` JSON via a new `ophir trade propose` command.

**Architecture:** Two units with a clean boundary. (1) `signals.ophir_signals` — a pure, CPU, cross-sectional normalizer (demean + scale + clamp on `r_close`). (2) `ophir trade propose` — a thin orchestrator that runs `load_forecasts → ophir_signals → blend_signals → ProposedOrder` and emits JSON for the existing `gate` command. The non-overridable safety gate stays a separate, explicit step; momentum/sentiment are stubbed neutral.

**Tech Stack:** Python 3.10+, Typer CLI, pytest + `typer.testing.CliRunner`, `statistics` stdlib.

## Global Constraints

- mypy is `strict = True`, targets Python 3.10 — keep `src/ophir` fully typed.
- ruff targets 3.12; run `uv run ruff check . && uv run ruff format --check .`.
- pytest runs `filterwarnings = error`; tests must stay **offline + CPU-only** and never touch network / CUDA / `.ophir/`. Use `tmp_path` and `monkeypatch`.
- NumPy-style docstrings throughout `src/ophir`, matching existing density.
- Imports: `known-first-party = ["ophir"]` ordering.
- Update the `[Unreleased]` section of `CHANGELOG.md`.
- Run tests with `uv run pytest`; single file via `uv run pytest tests/test_<name>.py`.

---

### Task 1: `ophir_signals` cross-sectional normalizer

**Files:**
- Modify: `src/ophir/trading/signals.py`
- Test: `tests/test_trading_signals.py` (append to existing file)

**Interfaces:**
- Consumes: `OphirForecast` from `ophir.trading.forecast` (fields `symbol`, `r_close`, `upside`, `downside`; all `float` except `symbol`).
- Produces: `ophir_signals(forecasts: Mapping[str, OphirForecast]) -> dict[str, float]` — per-symbol score in `[-1, 1]`; `{}` for empty input; all-`0.0` when cross-sectional std is `0`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_trading_signals.py`. Add `OphirForecast` and `ophir_signals` to the imports at the top of the file, then add:

```python
from ophir.trading.forecast import OphirForecast
from ophir.trading.signals import ophir_signals


def _fc(symbol: str, r_close: float) -> OphirForecast:
    return OphirForecast(symbol=symbol, r_close=r_close, upside=0.0, downside=0.0)


def test_ophir_signals_empty_returns_empty() -> None:
    assert ophir_signals({}) == {}


def test_ophir_signals_single_symbol_is_neutral() -> None:
    # One candidate has no cross-sectional dispersion -> no signal.
    assert ophir_signals({"AAPL": _fc("AAPL", 0.05)}) == {"AAPL": 0.0}


def test_ophir_signals_all_identical_is_neutral() -> None:
    out = ophir_signals({"A": _fc("A", 0.01), "B": _fc("B", 0.01)})
    assert out == {"A": 0.0, "B": 0.0}


def test_ophir_signals_cross_sectional_sign() -> None:
    out = ophir_signals(
        {"HI": _fc("HI", 0.05), "MID": _fc("MID", 0.0), "LO": _fc("LO", -0.05)}
    )
    assert out["HI"] > 0.0
    assert out["LO"] < 0.0
    assert out["MID"] == pytest.approx(0.0)
    assert out["HI"] == pytest.approx(-out["LO"])  # symmetric around the mean


def test_ophir_signals_clamps_to_unit_interval() -> None:
    # An outlier saturates at +1; every score stays within [-1, 1].
    out = ophir_signals(
        {f"S{i}": _fc(f"S{i}", v) for i, v in enumerate([0.0, 0.0, 0.0, 1.0])}
    )
    assert all(-1.0 <= s <= 1.0 for s in out.values())
    assert out["S3"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_trading_signals.py -v`
Expected: FAIL — `ImportError: cannot import name 'ophir_signals'`.

- [ ] **Step 3: Implement `ophir_signals`**

In `src/ophir/trading/signals.py`, add the new imports. The file already has `from ophir.trading.types import SignalWeights` — do **not** duplicate it. Add the two stdlib imports above the first-party block, and add the `forecast` import alphabetically before the existing `types` import, so the import region reads:

```python
from collections.abc import Mapping
from statistics import fmean, pstdev

from ophir.trading.forecast import OphirForecast
from ophir.trading.types import SignalWeights
```

(The `OphirForecast` import does not create a cycle: `forecast.py` has no top-level `ophir` imports.)

Then append the function:

```python
def ophir_signals(forecasts: Mapping[str, OphirForecast]) -> dict[str, float]:
    """Cross-sectionally score per-symbol forecasts into ``[-1, 1]``.

    Ranks the day's candidates on ``r_close`` by demeaning, dividing by the
    cross-sectional (population) standard deviation, and clamping to
    ``[-1, 1]``. The model's measured skill is cross-sectional (rank-IC), so the
    score is relative to the other candidates rather than an absolute return.

    Parameters
    ----------
    forecasts : mapping of str to OphirForecast
        Per-symbol forecasts for the day's candidate set.

    Returns
    -------
    dict[str, float]
        Per-symbol score in ``[-1, 1]``. Empty input yields ``{}``. When the
        cross-sectional dispersion is zero (a single candidate, or an
        all-identical day), every score is ``0.0`` — no dispersion, no signal.
    """
    if not forecasts:
        return {}
    closes = [f.r_close for f in forecasts.values()]
    mean = fmean(closes)
    std = pstdev(closes)
    if std == 0.0:
        return dict.fromkeys(forecasts, 0.0)
    return {
        symbol: max(-1.0, min(1.0, (f.r_close - mean) / std))
        for symbol, f in forecasts.items()
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_trading_signals.py -v`
Expected: PASS (all tests, including the pre-existing ones).

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy src/ophir && uv run ruff check src/ophir/trading/signals.py && uv run ruff format --check src/ophir/trading/signals.py`
Expected: no errors. (If format check fails, run `uv run ruff format src/ophir/trading/signals.py` and re-run.)

- [ ] **Step 6: Commit**

```bash
git add src/ophir/trading/signals.py tests/test_trading_signals.py
git commit -m "feat: add ophir_signals cross-sectional forecast normalizer"
```

---

### Task 2: `ophir trade propose` orchestration command

**Files:**
- Modify: `src/ophir/trading/cli.py`
- Modify: `CHANGELOG.md`
- Test: `tests/test_trading_propose.py` (new)

**Interfaces:**
- Consumes: `ophir_signals` and `blend_signals` / `CORE_WEIGHTS` / `TACTICAL_WEIGHTS` from `ophir.trading.signals`; `load_forecasts` from `ophir.trading.forecast`; `ProposedOrder`, `Side`, `Sleeve`, `AssetClass` from `ophir.trading.types`; `load_config` from `ophir.trading.config`.
- Produces: a Typer command `propose` on the existing `app`, emitting a JSON array of `ProposedOrder` dicts (keys: `symbol`, `side`, `sleeve`, `asset_class`, `notional`, `sector`, `is_defined_risk`, `is_short_option`) — exactly the shape `gate`'s `_order_from` consumes.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_trading_propose.py`:

```python
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ophir.trading.cli import app
from ophir.trading.forecast import OphirForecast

runner = CliRunner()

CONFIG = {
    "account_mode": "paper",
    "depth": "lean",
    "shortlist_size": 15,
    "verify_votes": 1,
    "limits": {
        "max_position_pct": 0.05,
        "max_option_premium_pct": 0.02,
        "halt_new_entries_day_loss_pct": 0.02,
        "flatten_tactical_day_loss_pct": 0.04,
        "max_deployed_pct": 0.80,
        "min_cash_pct": 0.20,
        "max_core_pct": 0.50,
        "max_tactical_pct": 0.30,
        "max_sector_pct": 0.25,
        "max_open_positions": 15,
        "max_total_option_premium_pct": 0.10,
    },
}


def _fc(symbol: str, r_close: float) -> OphirForecast:
    return OphirForecast(symbol=symbol, r_close=r_close, upside=0.0, downside=0.0)


def _config_file(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(CONFIG))
    return cfg


def test_propose_emits_orders_with_side_and_notional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config_file(tmp_path)
    monkeypatch.setattr(
        "ophir.trading.forecast.load_forecasts",
        lambda symbols, model_dir: {"AAA": _fc("AAA", 0.05), "BBB": _fc("BBB", -0.05)},
    )
    result = runner.invoke(
        app,
        [
            "propose",
            "--symbols", "AAA,BBB",
            "--model-dir", str(tmp_path),
            "--base-notional", "1000",
            "--config", str(cfg),
        ],
    )
    assert result.exit_code == 0, result.output
    orders = {o["symbol"]: o for o in json.loads(result.stdout)}
    assert orders["AAA"]["side"] == "buy"
    assert orders["BBB"]["side"] == "sell"
    # blended = 0.6 * ophir (CORE weight); ophir saturates at +/-1 for two names.
    assert orders["AAA"]["notional"] == pytest.approx(600.0)
    assert orders["AAA"]["asset_class"] == "equity"
    assert orders["AAA"]["sleeve"] == "core"
    assert orders["AAA"]["sector"] is None
    assert orders["AAA"]["is_defined_risk"] is True
    assert orders["AAA"]["is_short_option"] is False


def test_propose_emits_empty_when_forecasts_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config_file(tmp_path)
    monkeypatch.setattr(
        "ophir.trading.forecast.load_forecasts", lambda symbols, model_dir: {}
    )
    result = runner.invoke(
        app,
        [
            "propose",
            "--symbols", "AAA,BBB",
            "--model-dir", str(tmp_path),
            "--base-notional", "1000",
            "--config", str(cfg),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == []


def test_propose_skips_neutral_signals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config_file(tmp_path)
    # Identical forecasts -> zero dispersion -> neutral -> skipped (|0| <= 0.0).
    monkeypatch.setattr(
        "ophir.trading.forecast.load_forecasts",
        lambda symbols, model_dir: {"AAA": _fc("AAA", 0.01), "BBB": _fc("BBB", 0.01)},
    )
    result = runner.invoke(
        app,
        [
            "propose",
            "--symbols", "AAA,BBB",
            "--model-dir", str(tmp_path),
            "--base-notional", "1000",
            "--config", str(cfg),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == []


def test_propose_reads_symbols_from_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config_file(tmp_path)
    sym_file = tmp_path / "syms.txt"
    sym_file.write_text("AAA\nBBB\n")
    monkeypatch.setattr(
        "ophir.trading.forecast.load_forecasts",
        lambda symbols, model_dir: {"AAA": _fc("AAA", 0.05), "BBB": _fc("BBB", -0.05)},
    )
    result = runner.invoke(
        app,
        [
            "propose",
            "--symbols", str(sym_file),
            "--model-dir", str(tmp_path),
            "--base-notional", "1000",
            "--config", str(cfg),
        ],
    )
    assert result.exit_code == 0, result.output
    assert {o["symbol"] for o in json.loads(result.stdout)} == {"AAA", "BBB"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_trading_propose.py -v`
Expected: FAIL — `propose` is not a registered command (non-zero exit / usage error).

- [ ] **Step 3: Implement the `propose` command**

In `src/ophir/trading/cli.py`, extend the imports. Add to the `ophir.trading.types` import block the names already present plus nothing new (all of `ProposedOrder`, `Side`, `Sleeve`, `AssetClass` are imported). Add two new imports near the top with the other `from ophir.trading...` lines:

```python
from ophir.trading import forecast
from ophir.trading.signals import (
    CORE_WEIGHTS,
    TACTICAL_WEIGHTS,
    blend_signals,
    ophir_signals,
)
```

Add a symbol-parsing helper and an order-serializer near the existing `_order_from` helper:

```python
def _parse_symbols(value: str) -> list[str]:
    """Parse ``--symbols`` as a path to a file of symbols, else a comma list."""
    candidate = Path(value)
    text = candidate.read_text() if candidate.is_file() else value
    return [s.strip() for s in text.replace("\n", ",").split(",") if s.strip()]


def _order_to_dict(order: ProposedOrder) -> dict[str, object]:
    """Serialize a proposed order to the dict shape ``gate`` consumes."""
    return {
        "symbol": order.symbol,
        "side": order.side.value,
        "sleeve": order.sleeve.value,
        "asset_class": order.asset_class.value,
        "notional": order.notional,
        "sector": order.sector,
        "is_defined_risk": order.is_defined_risk,
        "is_short_option": order.is_short_option,
    }
```

Then add the command:

```python
@app.command()
def propose(
    symbols: str = typer.Option(
        ..., help="Comma-separated symbols, or a path to a file of symbols"
    ),
    model_dir: Path = typer.Option(..., help="Directory holding the base checkpoint"),
    base_notional: float = typer.Option(
        ..., help="Sizing base in dollars; notional = base_notional * |blended|"
    ),
    config: Path = typer.Option(..., help="Path to config.json"),
    sleeve: Sleeve = typer.Option(Sleeve.CORE, help="Strategy sleeve for the orders"),
    min_abs_signal: float = typer.Option(
        0.0, help="Skip orders whose |blended signal| is at or below this"
    ),
) -> None:
    """Emit ProposedOrder JSON from ophir forecasts (no gate, no ledger).

    Runs the seam end to end: load forecasts, cross-sectionally normalize them,
    blend with neutral momentum/sentiment, and size by ``base_notional``. Prints
    a JSON array of proposed orders for piping into ``gate``. Degrades to an
    empty array when forecasts are unavailable; never invokes the safety gate or
    writes the ledger.
    """
    load_config(config)  # validate account_mode/limits; not used for sizing here
    names = _parse_symbols(symbols)
    forecasts = forecast.load_forecasts(names, model_dir)
    scores = ophir_signals(forecasts)
    weights = CORE_WEIGHTS if sleeve is Sleeve.CORE else TACTICAL_WEIGHTS
    orders: list[dict[str, object]] = []
    for symbol in names:
        blended = blend_signals(
            ophir=scores.get(symbol),
            momentum=0.0,
            sentiment=0.0,
            weights=weights,
        )
        if abs(blended) <= min_abs_signal:
            continue
        order = ProposedOrder(
            symbol=symbol,
            side=Side.BUY if blended > 0 else Side.SELL,
            sleeve=sleeve,
            asset_class=AssetClass.EQUITY,
            notional=base_notional * abs(blended),
            sector=None,
            is_defined_risk=True,
            is_short_option=False,
        )
        orders.append(_order_to_dict(order))
    typer.echo(json.dumps(orders))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_trading_propose.py -v`
Expected: PASS (all four tests).

- [ ] **Step 5: Update the CHANGELOG**

In `CHANGELOG.md`, under `## [Unreleased]` → `### Added`, add a bullet at the top of the list:

```markdown
- `ophir trade propose`: orchestration command wiring the ophir forecast seam
  into the trading signal flow. Loads per-symbol offset-1 forecasts, scores them
  cross-sectionally (`signals.ophir_signals` — demean/scale/clamp on `r_close`),
  blends with neutral momentum/sentiment, sizes by `--base-notional`, and emits
  `ProposedOrder` JSON for the existing `gate` command. Degrades to an empty
  array when forecasts are unavailable; does not call the gate or write the
  ledger.
```

- [ ] **Step 6: Full suite, typecheck, lint**

Run: `uv run pytest && uv run mypy src/ophir && uv run ruff check . && uv run ruff format --check .`
Expected: all pass. (If format check fails, run `uv run ruff format .` and re-run.)

- [ ] **Step 7: Commit**

```bash
git add src/ophir/trading/cli.py tests/test_trading_propose.py CHANGELOG.md
git commit -m "feat: add 'ophir trade propose' to emit orders from the forecast seam"
```

---

## Notes for the implementer

- **Why `blended = 0.6 * ophir`:** `blend_signals` always includes the momentum
  and sentiment weights in its denominator even when those values are `0.0`, so
  under `CORE_WEIGHTS` (ophir=0.6) the blended score is the ophir score scaled by
  0.6. This dampening is intentional and conservative for the MVP; it resolves
  when real momentum/sentiment producers land. Do not "fix" it by bypassing
  `blend_signals`.
- **No CUDA in tests:** every test mocks `ophir.trading.forecast.load_forecasts`.
  Never let a test reach the real forward pass — the patch target is the module
  attribute `ophir.trading.forecast.load_forecasts`, which is why the command
  calls `forecast.load_forecasts(...)` (module-qualified) rather than importing
  the function by name.
- **Runtime precondition (not a test concern):** live forecasts need a 90-day
  IC-best checkpoint in `register.MODEL_DIR`. Without one, `load_forecasts`
  returns `{}` and `propose` emits `[]` — correct degradation, no error.
- **Out of scope:** momentum/sentiment producers, calling the gate, writing the
  ledger, `upside`/`downside` asymmetry, absolute-band calibration.
