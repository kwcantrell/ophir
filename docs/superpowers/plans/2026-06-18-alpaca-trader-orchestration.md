# Alpaca-Trader Skill & Orchestration — Implementation Plan (Plan 2 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Depends on Plan 1** (`2026-06-18-alpaca-trader-core.md`): the deterministic core in `src/ophir/trading/` (types, `evaluate_order`, ledger, metrics, signals, memory, `ophir trade gate`) must be complete and green before starting here.

**Goal:** Wire the deterministic core into a daily, manually-triggered paper-trading loop: a `.claude/skills/alpaca-trader/` skill with two `Workflow`-tool scripts (morning = act, evening = learn) over the Alpaca MCP, plus the tested Python glue (exposure join, outcome scoring, extra CLI verbs) the skill shells out to, plus the `memories/` knowledge base.

**Architecture:** The **main agent** (driven by `SKILL.md`) owns all I/O and order placement; it gathers account/market state from the Alpaca MCP, joins it with the ledger into an `AccountSnapshot` via `ophir trade snapshot`, fetches ophir forecasts (or a graceful "unavailable" stub), then runs a `Workflow` script that fans out **read-only analysis** agents and returns structured trade *proposals* (morning) or thesis *scores* (evening). Every proposal passes through `ophir trade gate` before the main agent places it; placement and ledger writes never happen inside parallel Workflow agents (single chokepoint). The Workflow runtime does no filesystem I/O and runs no GPU code — that all stays in the main session.

**Tech Stack:** Python 3.10+ (extends `ophir.trading`), `typer`, `pytest`; the harness `Workflow` tool (JavaScript ESM, `agent()`/`pipeline()`/`parallel()`); Alpaca MCP tools; Markdown skill + memory files.

## Global Constraints

- **Same repo rules as Plan 1:** new Python only under `src/ophir/trading/`; tests under `tests/`; strict mypy (py3.10, `warn_unused_ignores`); ruff line-length 100 with `ANN`/`N`/`UP`/`I`/`B`/`SIM`; pytest `filterwarnings=["error", …]`; Python 3.10 syntax (`X | None`, no `StrEnum`).
- **Account-mode safety:** the skill operates on a **paper** Alpaca account; `config.json` ships `account_mode: "paper"`. The gate's account-mode interlock (Plan 1, Task 3) blocks any mismatch.
- **No order placement inside Workflow agents.** Workflow scripts return proposals/scores only; the main agent gates and places.
- **No filesystem / `Date.now` / `Math.random` inside Workflow scripts** (runtime forbids them) — pass dates and any seeds via `args`.
- **ophir signal is optional:** the loop must run start-to-finish when ophir inference is unavailable (no CUDA/checkpoint), degrading to momentum + sentiment per Plan 1's `blend_signals`.
- **Depth knobs honored:** `shortlist_size` and `verify_votes` from `config.json` drive Workflow fan-out; default tier is **lean**.
- **MCP tool names referenced** (verify availability at run time): `get_clock`, `get_account_info`, `get_all_positions`, `get_most_active_stocks`, `get_market_movers`, `get_news`, `get_stock_bars`, `get_stock_snapshot`, `get_option_chain`, `get_option_snapshot`, `place_stock_order`, `place_option_order`, `get_orders`, `get_portfolio_history`, `get_account_activities`, `close_position`.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/ophir/trading/exposure.py` | `PositionInput`, `build_snapshot(...)`: join Alpaca positions + ledger sleeve tags → `AccountSnapshot`. |
| `src/ophir/trading/outcomes.py` | `ScoredOutcome`, `score_record(record, mark_price)`: evening thesis scoring + calibration. |
| `src/ophir/trading/forecast.py` | `OphirForecast`, `load_forecasts(symbols, model_dir)`: ophir adapter; returns `{}` (unavailable) when no checkpoint — never raises. |
| `src/ophir/trading/cli.py` | **Modify:** add `snapshot`, `record`, `close`, `score`, `memory-update`, `performance` commands. |
| `.claude/skills/alpaca-trader/config.json` | Real default config (paper, lean, the §6 numeric limits). |
| `.claude/skills/alpaca-trader/SKILL.md` | When-to-use + the morning/evening SOP the main agent follows. |
| `.claude/skills/alpaca-trader/lib/safety.md` | Human-readable description of the gate contract (points at the code as source of truth). |
| `.claude/skills/alpaca-trader/lib/signals.md` | How the blended signal + sleeves work; weights reference. |
| `.claude/skills/alpaca-trader/workflows/morning.js` | Act: screen → per-candidate analyst → verify → return proposals. |
| `.claude/skills/alpaca-trader/workflows/evening.js` | Learn: per-thesis outcome scoring + memory-update suggestions. |
| `memories/README.md` | Layout doc + `.gitkeep`ed subdirs (`tickers/`, `sectors/`, `ledger/`). |
| `tests/test_trading_exposure.py`, `tests/test_trading_outcomes.py`, `tests/test_trading_forecast.py`, `tests/test_trading_cli_glue.py` | Tests for the new Python. |

**Data-flow recap (morning):** MCP read → `ophir trade snapshot` (build `AccountSnapshot`) → `ophir trade forecast`/stub → `Workflow morning.js` (analysis, returns proposals) → per proposal `ophir trade gate` → `place_stock_order`/`place_option_order` → `ophir trade record`.
**Data-flow recap (evening):** MCP read (orders/positions/history) → `Workflow evening.js` (scores + memory suggestions) → `ophir trade close`/`score` (update ledger) → `ophir trade memory-update` (upsert entity files) → `ophir trade performance` (refresh `performance.md`).

---

## Task 1: Exposure join → `AccountSnapshot`

**Files:**
- Create: `src/ophir/trading/exposure.py`
- Test: `tests/test_trading_exposure.py`

**Interfaces:**
- Consumes: `AccountSnapshot`, `AssetClass`, `Sleeve`, `DecisionRecord` (Plan 1).
- Produces:
  - `@dataclass(frozen=True) PositionInput(symbol: str, market_value: float, asset_class: AssetClass, sector: str | None)`.
  - `def build_snapshot(*, equity: float, cash: float, day_pl: float, account_mode: str, positions: Sequence[PositionInput], ledger_records: Sequence[DecisionRecord]) -> AccountSnapshot`.
- Semantics: `held_symbols` = all position symbols; `open_position_count` = `len(positions)`; `symbol_exposure`/`sector_exposure` sum `abs(market_value)`; a symbol's sleeve = the sleeve of its **most recent** ledger record (records are appended in order, so the last match wins) and only such symbols contribute to `sleeve_exposure`; `option_premium_at_risk` = sum `abs(market_value)` over `AssetClass.OPTION` positions.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trading_exposure.py
from ophir.trading.exposure import PositionInput, build_snapshot
from ophir.trading.types import AssetClass, DecisionRecord, Side, Sleeve


def _rec(symbol: str, sleeve: Sleeve) -> DecisionRecord:
    return DecisionRecord(
        date="2026-06-18",
        symbol=symbol,
        sleeve=sleeve,
        side=Side.BUY,
        asset_class=AssetClass.EQUITY,
        notional=1000.0,
        sector="Technology",
        thesis="t",
        signals={},
        entry_price=100.0,
        target=None,
        stop=None,
        order_id=None,
        status="open",
        realized_pl=None,
        scored=False,
    )


def test_build_snapshot_aggregates_and_tags_sleeve() -> None:
    positions = [
        PositionInput("AAPL", 5_000.0, AssetClass.EQUITY, "Technology"),
        PositionInput("XOM", 3_000.0, AssetClass.EQUITY, "Energy"),
        PositionInput("AAPL260116C", 1_000.0, AssetClass.OPTION, "Technology"),
    ]
    ledger = [_rec("AAPL", Sleeve.TACTICAL), _rec("AAPL", Sleeve.CORE)]  # latest wins -> CORE
    snap = build_snapshot(
        equity=100_000.0,
        cash=50_000.0,
        day_pl=-200.0,
        account_mode="paper",
        positions=positions,
        ledger_records=ledger,
    )
    assert snap.held_symbols == frozenset({"AAPL", "XOM", "AAPL260116C"})
    assert snap.open_position_count == 3
    assert snap.symbol_exposure["AAPL"] == 5_000.0
    assert snap.sector_exposure["Technology"] == 6_000.0
    assert snap.sleeve_exposure[Sleeve.CORE] == 5_000.0  # only ledger-known symbol
    assert snap.option_premium_at_risk == 1_000.0
    assert snap.account_mode == "paper"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_trading_exposure.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ophir.trading.exposure'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ophir/trading/exposure.py
"""Join live Alpaca positions with the ledger into an AccountSnapshot.

The safety gate consumes pre-aggregated exposures; this module is where the
Alpaca position list (which carries no sleeve/sector tags of ours) is merged
with the decision ledger to produce them.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from ophir.trading.types import AccountSnapshot, AssetClass, DecisionRecord, Sleeve


@dataclass(frozen=True)
class PositionInput:
    """A single open position as gathered from the Alpaca MCP."""

    symbol: str
    market_value: float
    asset_class: AssetClass
    sector: str | None


def build_snapshot(
    *,
    equity: float,
    cash: float,
    day_pl: float,
    account_mode: str,
    positions: Sequence[PositionInput],
    ledger_records: Sequence[DecisionRecord],
) -> AccountSnapshot:
    """Aggregate positions + ledger sleeve tags into an :class:`AccountSnapshot`."""
    sleeve_by_symbol: dict[str, Sleeve] = {}
    for record in ledger_records:
        sleeve_by_symbol[record.symbol] = record.sleeve

    symbol_exposure: dict[str, float] = {}
    sector_exposure: dict[str, float] = {}
    sleeve_exposure: dict[Sleeve, float] = {}
    option_premium = 0.0

    for pos in positions:
        value = abs(pos.market_value)
        symbol_exposure[pos.symbol] = symbol_exposure.get(pos.symbol, 0.0) + value
        if pos.sector is not None:
            sector_exposure[pos.sector] = sector_exposure.get(pos.sector, 0.0) + value
        sleeve = sleeve_by_symbol.get(pos.symbol)
        if sleeve is not None:
            sleeve_exposure[sleeve] = sleeve_exposure.get(sleeve, 0.0) + value
        if pos.asset_class is AssetClass.OPTION:
            option_premium += value

    return AccountSnapshot(
        equity=equity,
        cash=cash,
        day_pl=day_pl,
        open_position_count=len(positions),
        held_symbols=frozenset(p.symbol for p in positions),
        symbol_exposure=symbol_exposure,
        sector_exposure=sector_exposure,
        sleeve_exposure=sleeve_exposure,
        option_premium_at_risk=option_premium,
        account_mode=account_mode,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_trading_exposure.py -v && uv run mypy src/ophir && uv run ruff check src/ophir/trading tests/test_trading_exposure.py`
Expected: PASS, mypy clean, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/ophir/trading/exposure.py tests/test_trading_exposure.py
git commit -m "feat(trading): add exposure join to AccountSnapshot"
```

---

## Task 2: Evening outcome scoring

**Files:**
- Create: `src/ophir/trading/outcomes.py`
- Test: `tests/test_trading_outcomes.py`

**Interfaces:**
- Consumes: `DecisionRecord`, `Side` (Plan 1).
- Produces:
  - `@dataclass(frozen=True) ScoredOutcome(symbol: str, correct: bool, realized_return: float, predicted_ophir: float | None, abs_calibration_error: float | None)`.
  - `def score_record(record: DecisionRecord, mark_price: float) -> ScoredOutcome`.
- Semantics: requires `record.entry_price` not None (raise `ValueError` otherwise). `realized_return` = `(mark - entry)/entry` for `BUY`, `(entry - mark)/entry` for `SELL`. `correct = realized_return > 0`. `predicted_ophir = record.signals.get("ophir")`. `abs_calibration_error = abs(predicted_ophir - realized_return)` when the ophir signal is present, else `None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trading_outcomes.py
import pytest

from ophir.trading.outcomes import score_record
from ophir.trading.types import AssetClass, DecisionRecord, Side, Sleeve


def _rec(side: Side, signals: dict[str, float]) -> DecisionRecord:
    return DecisionRecord(
        date="2026-06-18",
        symbol="AAPL",
        sleeve=Sleeve.CORE,
        side=side,
        asset_class=AssetClass.EQUITY,
        notional=1000.0,
        sector="Technology",
        thesis="t",
        signals=signals,
        entry_price=100.0,
        target=110.0,
        stop=90.0,
        order_id="x",
        status="open",
        realized_pl=None,
        scored=False,
    )


def test_buy_winner() -> None:
    out = score_record(_rec(Side.BUY, {"ophir": 0.08}), mark_price=110.0)
    assert out.correct is True
    assert out.realized_return == pytest.approx(0.10)
    assert out.abs_calibration_error == pytest.approx(0.02)


def test_buy_loser() -> None:
    out = score_record(_rec(Side.BUY, {}), mark_price=95.0)
    assert out.correct is False
    assert out.realized_return == pytest.approx(-0.05)
    assert out.predicted_ophir is None
    assert out.abs_calibration_error is None


def test_sell_inverts_direction() -> None:
    out = score_record(_rec(Side.SELL, {}), mark_price=90.0)
    assert out.correct is True
    assert out.realized_return == pytest.approx(0.10)


def test_missing_entry_raises() -> None:
    rec = _rec(Side.BUY, {})
    rec_no_entry = DecisionRecord(**{**rec.__dict__, "entry_price": None})
    with pytest.raises(ValueError, match="entry_price"):
        score_record(rec_no_entry, mark_price=100.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_trading_outcomes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ophir.trading.outcomes'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ophir/trading/outcomes.py
"""Score a closed/marked decision against its thesis and ophir calibration."""

from dataclasses import dataclass

from ophir.trading.types import DecisionRecord, Side


@dataclass(frozen=True)
class ScoredOutcome:
    """Outcome attribution for one ledger decision."""

    symbol: str
    correct: bool
    realized_return: float
    predicted_ophir: float | None
    abs_calibration_error: float | None


def score_record(record: DecisionRecord, mark_price: float) -> ScoredOutcome:
    """Score ``record`` at ``mark_price`` (exit price or current mark)."""
    if record.entry_price is None:
        raise ValueError("record.entry_price is required to score an outcome")
    entry = record.entry_price
    move = (mark_price - entry) / entry
    realized_return = move if record.side is Side.BUY else -move
    predicted = record.signals.get("ophir")
    cal_err = None if predicted is None else abs(predicted - realized_return)
    return ScoredOutcome(
        symbol=record.symbol,
        correct=realized_return > 0,
        realized_return=realized_return,
        predicted_ophir=predicted,
        abs_calibration_error=cal_err,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_trading_outcomes.py -v && uv run mypy src/ophir && uv run ruff check src/ophir/trading tests/test_trading_outcomes.py`
Expected: PASS, mypy clean, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/ophir/trading/outcomes.py tests/test_trading_outcomes.py
git commit -m "feat(trading): add evening outcome scoring"
```

---

## Task 3: ophir forecast adapter (graceful when unavailable)

**Files:**
- Create: `src/ophir/trading/forecast.py`
- Test: `tests/test_trading_forecast.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `@dataclass(frozen=True) OphirForecast(symbol: str, r_close: float, upside: float, downside: float)`.
  - `def load_forecasts(symbols: Sequence[str], model_dir: str | Path | None) -> dict[str, OphirForecast]`.
- Semantics: this is the **seam**, not the model. If `model_dir` is `None` or no checkpoint file exists under it, return `{}` (unavailable) — never raise. Real CUDA inference is a future enhancement wired in behind this same signature; for now the function only reports availability. A checkpoint is considered present if `model_dir` contains any `*.ckpt` file.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trading_forecast.py
from pathlib import Path

from ophir.trading.forecast import load_forecasts


def test_no_model_dir_returns_empty() -> None:
    assert load_forecasts(["AAPL", "MSFT"], None) == {}


def test_missing_checkpoint_returns_empty(tmp_path: Path) -> None:
    assert load_forecasts(["AAPL"], tmp_path) == {}


def test_present_checkpoint_does_not_raise(tmp_path: Path) -> None:
    (tmp_path / "base.ckpt").write_bytes(b"")
    # Real inference is not wired yet; the adapter must still return a dict
    # (empty is acceptable) without raising.
    result = load_forecasts(["AAPL"], tmp_path)
    assert isinstance(result, dict)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_trading_forecast.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ophir.trading.forecast'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ophir/trading/forecast.py
"""Adapter seam for ophir model forecasts.

This module defines the contract the trading loop uses to obtain per-symbol
forecasts. Actual CUDA inference (loading a checkpoint and running the model) is
a future enhancement implemented behind :func:`load_forecasts`; until then the
function reports availability only and never raises, so the loop degrades to the
non-ophir signals when no checkpoint is present.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OphirForecast:
    """One symbol's three forward targets from the ophir model."""

    symbol: str
    r_close: float
    upside: float
    downside: float


def _has_checkpoint(model_dir: str | Path) -> bool:
    path = Path(model_dir)
    return path.is_dir() and any(path.glob("*.ckpt"))


def load_forecasts(
    symbols: Sequence[str], model_dir: str | Path | None
) -> dict[str, OphirForecast]:
    """Return per-symbol ophir forecasts, or ``{}`` if the model is unavailable."""
    if model_dir is None or not _has_checkpoint(model_dir):
        return {}
    # Inference not yet wired; report availability without fabricating forecasts.
    return {}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_trading_forecast.py -v && uv run mypy src/ophir && uv run ruff check src/ophir/trading tests/test_trading_forecast.py`
Expected: PASS, mypy clean, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/ophir/trading/forecast.py tests/test_trading_forecast.py
git commit -m "feat(trading): add ophir forecast adapter seam"
```

---

## Task 4: CLI glue commands

**Files:**
- Modify: `src/ophir/trading/cli.py`
- Test: `tests/test_trading_cli_glue.py`

**Interfaces (new Typer commands on the existing `app`):**
- `record --ledger-dir <dir> --month <YYYY-MM> --decision <json-file>` — append one decision; the JSON matches `record_to_dict` shape. Prints `{"appended": true}`.
- `close --ledger-dir <dir> --month <YYYY-MM> --order-id <id> --status <s> --realized-pl <float>` — rewrite the month file, setting `status`, `realized_pl`, and `scored=true` on the matching record; prints `{"updated": <count>}`.
- `performance --ledger-dir <dir> --equity-curve <json-file> --out <path>` — read an equity-curve JSON array, compute metrics (Plan 1 `metrics.py`), and write a `performance.md` snapshot; prints the metrics as JSON.

> `snapshot` and `score`/`memory-update` are intentionally **out of this task**: `snapshot` is exercised end-to-end by the skill smoke run (Task 10) and would need a large fixture here; memory upserts are simple enough to call in-process from the skill via `python -m`. Keeping this task to the three ledger/metrics verbs keeps it reviewable. The skill (Task 9) documents the `python -m ophir.trading...` one-liners for `build_snapshot` and `upsert_section`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trading_cli_glue.py
import json
from pathlib import Path

from typer.testing import CliRunner

from ophir.trading.cli import app

runner = CliRunner()

DECISION = {
    "date": "2026-06-18",
    "symbol": "AAPL",
    "sleeve": "core",
    "side": "buy",
    "asset_class": "equity",
    "notional": 1000.0,
    "sector": "Technology",
    "thesis": "t",
    "signals": {"ophir": 0.05},
    "entry_price": 100.0,
    "target": 110.0,
    "stop": 90.0,
    "order_id": "oid-1",
    "status": "open",
    "realized_pl": None,
    "scored": False,
}


def test_record_then_close(tmp_path: Path) -> None:
    dpath = tmp_path / "d.json"
    dpath.write_text(json.dumps(DECISION))
    r = runner.invoke(
        app,
        ["record", "--ledger-dir", str(tmp_path), "--month", "2026-06", "--decision", str(dpath)],
    )
    assert r.exit_code == 0
    assert json.loads(r.stdout)["appended"] is True

    r2 = runner.invoke(
        app,
        [
            "close",
            "--ledger-dir", str(tmp_path),
            "--month", "2026-06",
            "--order-id", "oid-1",
            "--status", "closed",
            "--realized-pl", "150.0",
        ],
    )
    assert r2.exit_code == 0
    assert json.loads(r2.stdout)["updated"] == 1
    lines = (tmp_path / "2026-06.jsonl").read_text().splitlines()
    rec = json.loads(lines[0])
    assert rec["status"] == "closed"
    assert rec["realized_pl"] == 150.0
    assert rec["scored"] is True


def test_performance_writes_markdown(tmp_path: Path) -> None:
    curve = tmp_path / "curve.json"
    curve.write_text(json.dumps([100000.0, 101000.0, 100500.0, 102000.0]))
    out = tmp_path / "performance.md"
    r = runner.invoke(
        app,
        ["performance", "--equity-curve", str(curve), "--out", str(out)],
    )
    assert r.exit_code == 0
    payload = json.loads(r.stdout)
    assert "total_return" in payload
    assert "sharpe" in payload
    assert "max_drawdown" in payload
    assert out.exists()
    assert "Total return" in out.read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_trading_cli_glue.py -v`
Expected: FAIL — Typer reports `No such command 'record'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/ophir/trading/cli.py` (keep the existing `gate` command). Add imports at the top alongside the current ones:

```python
from ophir.trading import metrics
from ophir.trading.ledger import append_decision, load_decisions, ledger_path, record_from_dict, record_to_dict
```

(Adjust the import to satisfy ruff isort ordering; the names used are `append_decision`, `load_decisions`, `ledger_path`, `record_from_dict`, `record_to_dict`.)

Then add the commands:

```python
@app.command()
def record(
    ledger_dir: Path = typer.Option(..., help="Ledger directory"),
    month: str = typer.Option(..., help="YYYY-MM"),
    decision: Path = typer.Option(..., help="Decision JSON file"),
) -> None:
    """Append one decision record to the ledger."""
    rec = record_from_dict(json.loads(decision.read_text()))
    append_decision(ledger_dir, month, rec)
    typer.echo(json.dumps({"appended": True}))


@app.command()
def close(
    ledger_dir: Path = typer.Option(..., help="Ledger directory"),
    month: str = typer.Option(..., help="YYYY-MM"),
    order_id: str = typer.Option(..., help="order_id to update"),
    status: str = typer.Option(..., help="new status"),
    realized_pl: float = typer.Option(..., help="realized P&L"),
) -> None:
    """Mark a decision closed/scored with its realized P&L."""
    records = load_decisions(ledger_dir, month)
    updated = 0
    new_lines: list[str] = []
    for rec in records:
        data = record_to_dict(rec)
        if rec.order_id == order_id:
            data["status"] = status
            data["realized_pl"] = realized_pl
            data["scored"] = True
            updated += 1
        new_lines.append(json.dumps(data))
    ledger_path(ledger_dir, month).write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
    typer.echo(json.dumps({"updated": updated}))


@app.command()
def performance(
    equity_curve: Path = typer.Option(..., help="JSON array of equity values"),
    out: Path = typer.Option(..., help="Output performance.md path"),
) -> None:
    """Compute portfolio metrics and write a performance.md snapshot."""
    curve = [float(x) for x in json.loads(equity_curve.read_text())]
    payload = {
        "total_return": metrics.total_return(curve),
        "sharpe": metrics.sharpe(metrics.daily_returns(curve)),
        "max_drawdown": metrics.max_drawdown(curve),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "# Performance\n\n"
        f"- Total return: {payload['total_return']:.4f}\n"
        f"- Sharpe: {payload['sharpe']:.4f}\n"
        f"- Max drawdown: {payload['max_drawdown']:.4f}\n"
    )
    typer.echo(json.dumps(payload))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_trading_cli_glue.py -v && uv run mypy src/ophir && uv run ruff check src/ophir/trading tests/test_trading_cli_glue.py`
Expected: PASS, mypy clean, ruff clean. (Run `uv run ruff format src/ophir/trading/cli.py` if format check complains.)

- [ ] **Step 5: Commit**

```bash
git add src/ophir/trading/cli.py tests/test_trading_cli_glue.py
git commit -m "feat(trading): add record/close/performance CLI verbs"
```

---

## Task 5: Default `config.json`

**Files:**
- Create: `.claude/skills/alpaca-trader/config.json`

**Interface:** must load cleanly via `ophir`'s `load_config` (Plan 1, Task 2) — same keys, valid ranges, `account_mode: "paper"`, `depth: "lean"`.

- [ ] **Step 1: Write the config file**

```json
{
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
    "max_total_option_premium_pct": 0.10
  }
}
```

- [ ] **Step 2: Validate it loads**

Run: `uv run python -c "from ophir.trading.config import load_config; print(load_config('.claude/skills/alpaca-trader/config.json'))"`
Expected: prints a `TradingConfig(...)` repr with `account_mode='paper'`, no exception.

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/alpaca-trader/config.json
git commit -m "feat(skill): add alpaca-trader default config"
```

---

## Task 6: Reference docs (`lib/safety.md`, `lib/signals.md`)

**Files:**
- Create: `.claude/skills/alpaca-trader/lib/safety.md`
- Create: `.claude/skills/alpaca-trader/lib/signals.md`

These are human/agent-readable explainers. The **code is the source of truth**; these point at it.

- [ ] **Step 1: Write `lib/safety.md`**

````markdown
# Safety contract

Every order MUST pass `ophir trade gate` before placement. The gate is
implemented in `src/ophir/trading/safety.py::evaluate_order` — that code, not
this doc, is authoritative.

## Hard limits (from `config.json -> limits`)

| Limit | Default | Meaning |
| --- | --- | --- |
| `max_position_pct` | 5% | Max equity in one symbol at entry. |
| `max_option_premium_pct` | 2% | Max premium-at-risk per option order. |
| `halt_new_entries_day_loss_pct` | 2% | Halt new BUYs once the day is down this much. |
| `flatten_tactical_day_loss_pct` | 4% | Evening/intraday: flatten the tactical sleeve. |
| `max_deployed_pct` | 80% | Max total deployed. |
| `min_cash_pct` | 20% | Cash floor the agent may never spend below. |
| `max_core_pct` / `max_tactical_pct` | 50% / 30% | Per-sleeve exposure caps. |
| `max_sector_pct` | 25% | Per-sector exposure cap. |
| `max_open_positions` | 15 | Position count cap. |
| `max_total_option_premium_pct` | 10% | Aggregate option premium-at-risk cap. |

## Non-negotiables

- Account-mode interlock: the gate REJECTS if the live account's mode does not
  match `config.account_mode`. This guards the paper→live switch.
- No naked short options (must be defined-risk).
- SELLs (risk-reducing) are always allowed at full size.
- The agent cannot override a REJECT or a RESIZE. A RESIZE means place the
  smaller `approved_notional`, not the requested size.

## Calling the gate

```bash
ophir trade gate --config <config.json> --order <order.json> --snapshot <snapshot.json>
```
Exit code is non-zero on REJECT. Parse stdout JSON for `action` /
`approved_notional` / `reasons`.
````

- [ ] **Step 2: Write `lib/signals.md`**

````markdown
# Signals & sleeves

Each candidate gets a blended score in [-1, 1] from three components, each first
normalized to [-1, 1] (`src/ophir/trading/signals.py`):

- **ophir** — model forecast (relative close return). May be absent (no
  checkpoint / uncovered name); the blend renormalizes over the remaining
  weights. Never fabricate an ophir value when unavailable.
- **momentum** — from recent bars (e.g. trend / rate-of-change).
- **sentiment** — soft signal from `get_news`; never the sole basis for a trade.

## Sleeve weights (`signals.py`)

- **Core** (S&P 500, weeks–months): `CORE_WEIGHTS` = ophir 0.6 / momentum 0.25 /
  sentiment 0.15. ophir-led.
- **Tactical** (movers/news discovery, days–weeks): `TACTICAL_WEIGHTS` =
  ophir 0.2 / momentum 0.5 / sentiment 0.3. Technicals-led.

## Sleeve allocation

Core ≤ 50% of equity, tactical ≤ 30%, cash ≥ 20% — enforced by the gate, not by
the analyst. Analysts propose; the gate sizes.
````

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/alpaca-trader/lib/
git commit -m "docs(skill): add safety and signals reference docs"
```

---

## Task 7: Morning workflow script

**Files:**
- Create: `.claude/skills/alpaca-trader/workflows/morning.js`

**Contract:** invoked via the `Workflow` tool with `args = { tradingDate, depth, shortlistSize, verifyVotes, sleeves: {core, tactical}, ophirForecasts, seedCandidates: {core:[...], tactical:[...]}, memoryNotes }`. Returns `{ proposals: [ {symbol, side, sleeve, asset_class, notional, sector, is_defined_risk, is_short_option, thesis, signals} ] }`. Does **no** file I/O and places **no** orders.

- [ ] **Step 1: Write `workflows/morning.js`**

```javascript
export const meta = {
  name: 'alpaca-morning',
  description: 'Screen candidates, analyze each, adversarially verify, return trade proposals (no placement).',
  phases: [
    { title: 'Analyze', detail: 'one analyst per shortlisted candidate' },
    { title: 'Verify', detail: 'devil\'s-advocate vote per proposed trade' },
  ],
}

const a = args || {}
const shortlistSize = a.shortlistSize || 15
const verifyVotes = a.verifyVotes || 1
const seeds = a.seedCandidates || { core: [], tactical: [] }
const forecasts = a.ophirForecasts || {}

const PROPOSAL_SCHEMA = {
  type: 'object',
  required: ['recommend', 'symbol', 'sleeve'],
  properties: {
    recommend: { type: 'boolean' },
    symbol: { type: 'string' },
    sleeve: { type: 'string', enum: ['core', 'tactical'] },
    side: { type: 'string', enum: ['buy', 'sell'] },
    asset_class: { type: 'string', enum: ['equity', 'option'] },
    notional: { type: 'number' },
    sector: { type: ['string', 'null'] },
    is_defined_risk: { type: 'boolean' },
    is_short_option: { type: 'boolean' },
    thesis: { type: 'string' },
    signals: { type: 'object' },
  },
}

const VERDICT_SCHEMA = {
  type: 'object',
  required: ['survives'],
  properties: { survives: { type: 'boolean' }, reason: { type: 'string' } },
}

// Build the shortlist from the seed candidates the main agent gathered.
const candidates = [
  ...seeds.core.map((c) => ({ ...c, sleeve: 'core' })),
  ...seeds.tactical.map((c) => ({ ...c, sleeve: 'tactical' })),
].slice(0, shortlistSize)

log(`Analyzing ${candidates.length} candidates (verifyVotes=${verifyVotes})`)

const analyzed = await pipeline(
  candidates,
  (c) =>
    agent(
      `You are a trading analyst. Symbol ${c.symbol} (sleeve: ${c.sleeve}).
Use Alpaca MCP read tools (get_stock_snapshot, get_stock_bars, get_news, and for
options get_option_chain/get_option_snapshot) to assess a ${c.sleeve} trade.
ophir forecast for this symbol (may be absent): ${JSON.stringify(forecasts[c.symbol] || null)}.
Blend ophir + momentum + sentiment per the sleeve weighting (core is ophir-led,
tactical is technicals-led). Propose at most ONE order. Set recommend=false if no
edge. notional is the dollar size you want (premium-at-risk for options). Do NOT
place any order. Return the proposal object.`,
      { label: `analyze:${c.symbol}`, phase: 'Analyze', schema: PROPOSAL_SCHEMA },
    ),
  (proposal, c, _i) => {
    if (!proposal || !proposal.recommend) return null
    return parallel(
      Array.from({ length: verifyVotes }, (_v) => () =>
        agent(
          `Adversarially review this proposed ${proposal.sleeve} trade and try to
REFUTE it. Default survives=false if the thesis is weak, the signal is thin, or
risk is unclear. Proposal: ${JSON.stringify(proposal)}.`,
          { label: `verify:${proposal.symbol}`, phase: 'Verify', schema: VERDICT_SCHEMA },
        ),
      ),
    ).then((votes) => {
      const ok = votes.filter(Boolean).filter((v) => v.survives).length
      const need = Math.ceil(verifyVotes / 2)
      return ok >= need ? proposal : null
    })
  },
)

const proposals = analyzed.flat ? analyzed.flat().filter(Boolean) : analyzed.filter(Boolean)
log(`Surviving proposals: ${proposals.length}`)
return { proposals }
```

- [ ] **Step 2: Syntax-check the script**

Run: `node --check .claude/skills/alpaca-trader/workflows/morning.js`
Expected: no output, exit 0. (If `node` is unavailable, validate by visual inspection that braces/parens balance and the `agent`/`pipeline`/`parallel` calls match the Workflow API.)

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/alpaca-trader/workflows/morning.js
git commit -m "feat(skill): add morning analysis workflow"
```

---

## Task 8: Evening workflow script

**Files:**
- Create: `.claude/skills/alpaca-trader/workflows/evening.js`

**Contract:** invoked with `args = { tradingDate, openTheses: [ {order_id, symbol, sleeve, thesis, entry_price, mark_price, realized_return, predicted_ophir, correct} ], }` where the main agent has already computed numeric outcomes via `score_record`. The workflow turns each scored thesis into a distilled memory-update suggestion. Returns `{ updates: [ {scope, name, heading, body} ] }` where `scope` ∈ `ticker|sector|pattern|lesson`. No file I/O.

- [ ] **Step 1: Write `workflows/evening.js`**

```javascript
export const meta = {
  name: 'alpaca-evening',
  description: 'Turn scored theses into distilled memory-update suggestions (no file writes).',
  phases: [{ title: 'Reflect', detail: 'one reflection per scored thesis' }],
}

const a = args || {}
const theses = a.openTheses || []

const UPDATE_SCHEMA = {
  type: 'object',
  required: ['updates'],
  properties: {
    updates: {
      type: 'array',
      items: {
        type: 'object',
        required: ['scope', 'name', 'heading', 'body'],
        properties: {
          scope: { type: 'string', enum: ['ticker', 'sector', 'pattern', 'lesson'] },
          name: { type: 'string' },
          heading: { type: 'string' },
          body: { type: 'string' },
        },
      },
    },
  },
}

log(`Reflecting on ${theses.length} scored theses`)

const perThesis = await parallel(
  theses.map((t) => () =>
    agent(
      `Reflect on this closed/marked trade and produce concise memory updates.
Trade: ${JSON.stringify(t)}.
Was the thesis right? Did ophir's prediction calibrate to the realized return?
Return 1-3 updates: a 'ticker' note for ${t.symbol}, optionally a 'sector' note,
and (only if a generalizable rule emerged) a 'pattern' or 'lesson' note. Keep
each body to a few sentences; write durable knowledge, not a play-by-play.`,
      { label: `reflect:${t.symbol}`, phase: 'Reflect', schema: UPDATE_SCHEMA },
    ),
  ),
)

const updates = perThesis
  .filter(Boolean)
  .flatMap((r) => (r.updates || []))
log(`Produced ${updates.length} memory updates`)
return { updates }
```

- [ ] **Step 2: Syntax-check**

Run: `node --check .claude/skills/alpaca-trader/workflows/evening.js`
Expected: no output, exit 0 (or visual inspection if `node` absent).

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/alpaca-trader/workflows/evening.js
git commit -m "feat(skill): add evening reflection workflow"
```

---

## Task 9: `SKILL.md` (the SOP)

**Files:**
- Create: `.claude/skills/alpaca-trader/SKILL.md`

This is the procedure the main agent follows. It ties MCP gathering, the CLI glue, the Workflow scripts, and memory writes together. It must encode the **single-gate** and **no-placement-in-workflow** rules.

- [ ] **Step 1: Write `SKILL.md`**

````markdown
---
name: alpaca-trader
description: >-
  Run the daily paper-trading loop over the Alpaca MCP for the ophir account:
  a morning pass that screens/analyzes/verifies and places gated orders, and an
  evening pass that scores outcomes and updates the memories knowledge base. Use
  when the user says "run the morning trade pass", "run the evening review",
  "trade today", or "do the daily trading routine".
---

# alpaca-trader

Mixed-strategy (core + tactical) AI trading on an Alpaca **paper** account, with
a non-overridable safety gate and an entity-organized memory base. Deterministic
logic lives in `ophir.trading` (called via `ophir trade …`); analysis fans out
through two `Workflow` scripts in `workflows/`.

## Invariants (never violate)

1. **Paper only** for now: `config.json` has `account_mode: "paper"`. The gate
   rejects on any account-mode mismatch — do not edit config to bypass it.
2. **Every order passes `ophir trade gate`** before placement. Honor REJECT
   (skip) and RESIZE (place the smaller `approved_notional`).
3. **No order placement inside Workflow agents.** Workflows return proposals;
   only the main agent (you) places orders via the Alpaca MCP.
4. Read `lib/safety.md` and `lib/signals.md` before acting.

## Arguments

Parse `$ARGUMENTS`: `morning` runs the act pass, `evening` runs the learn pass.
If neither is given, ask which pass to run.

## Morning pass (act)

1. **Confirm trading day:** `get_clock`. If the market is closed today, stop.
2. **Account + positions:** `get_account_info` (equity, cash, day P&L),
   `get_all_positions`. Resolve each position's sector (use `get_asset` or a
   cached map).
3. **Load ledger sleeve tags:** read this month's ledger via
   `python -m ophir.trading.ledger`-style load, or just pass records to the
   snapshot builder:
   ```bash
   uv run python -c "import json,sys; from ophir.trading.exposure import build_snapshot, PositionInput; from ophir.trading.types import AssetClass; from ophir.trading.ledger import load_decisions; \
   # build PositionInput list from the MCP positions you gathered, then:
   # print(json.dumps(<AccountSnapshot as dict>))"
   ```
   In practice: construct a `snapshot.json` (matching `AccountSnapshot` fields)
   using `build_snapshot` so the gate can consume it.
4. **ophir forecasts:** attempt `load_forecasts(symbols, model_dir)`; if it
   returns `{}` (no checkpoint), proceed without the ophir component.
5. **Screen → seed candidates:** core = top/bottom ophir names if available else
   liquid S&P 500 names of interest; tactical = `get_most_active_stocks` /
   `get_market_movers` / `get_news`. Trim to `shortlist_size`.
6. **Run analysis workflow:** call the `Workflow` tool with
   `scriptPath: ".claude/skills/alpaca-trader/workflows/morning.js"` and `args`
   per its contract (date, depth knobs, sleeves, ophirForecasts, seedCandidates,
   memoryNotes). It returns `{ proposals }`.
7. **Gate + place each proposal:** for every proposal, write `order.json` and the
   current `snapshot.json`, then:
   ```bash
   uv run ophir trade gate --config .claude/skills/alpaca-trader/config.json \
     --order order.json --snapshot snapshot.json
   ```
   - REJECT (non-zero exit) → skip, note the reason.
   - APPROVE/RESIZE → place via `place_stock_order` / `place_option_order` using
     `approved_notional` (convert to qty/contracts via latest price). Update the
     running snapshot's exposures so the next proposal is gated against the new
     state.
8. **Record:** for each placed order, append a `DecisionRecord` to the ledger:
   ```bash
   uv run ophir trade record --ledger-dir memories/ledger --month <YYYY-MM> \
     --decision decision.json
   ```
9. **Summarize** what was placed, skipped (with gate reasons), and current
   exposure vs. the caps.

## Evening pass (learn)

1. **Pull results:** `get_orders` (fills), `get_all_positions`,
   `get_portfolio_history`, `get_account_activities`.
2. **Score open/closed theses:** for each open ledger record, get a mark price
   (`get_stock_snapshot`) and compute the outcome with `score_record`
   (`ophir.trading.outcomes`). Build the `openTheses` array (include
   `realized_return`, `predicted_ophir`, `correct`).
3. **Run reflection workflow:** call `Workflow` with
   `scriptPath: ".claude/skills/alpaca-trader/workflows/evening.js"` and
   `args.openTheses`. It returns `{ updates }`.
4. **Apply ledger closures:** for positions that closed, update the ledger:
   ```bash
   uv run ophir trade close --ledger-dir memories/ledger --month <YYYY-MM> \
     --order-id <id> --status closed --realized-pl <pnl>
   ```
5. **Apply memory updates:** for each update, upsert the section into the right
   file (`memories/tickers/<SYM>.md`, `memories/sectors/<sector>.md`,
   `memories/patterns.md`, or `memories/lessons.md`):
   ```bash
   uv run python -c "from ophir.trading.memory import read_memory, write_memory, upsert_section; \
   p='memories/tickers/AAPL.md'; write_memory(p, upsert_section(read_memory(p), 'Thesis review', '<body>'))"
   ```
6. **Refresh performance:** build an equity curve from `get_portfolio_history`
   and run:
   ```bash
   uv run ophir trade performance --equity-curve curve.json --out memories/performance.md
   ```
7. **Summarize** hit-rate, calibration, and notable lessons. Track Phase-1
   progress (hit-rate + calibration) toward the paper→live graduation bar.

## Depth tiers

`config.json -> depth` (`lean`/`balanced`/`deep`) and `shortlist_size` /
`verify_votes` scale the morning fan-out. Start `lean`; raise once the loop is
proven. See the design spec for cost ballparks.
````

- [ ] **Step 2: Validate frontmatter + reference integrity**

Run: `uv run python -c "import pathlib,yaml,sys; t=pathlib.Path('.claude/skills/alpaca-trader/SKILL.md').read_text(); fm=t.split('---')[1]; d=yaml.safe_load(fm); assert d['name']=='alpaca-trader' and 'description' in d; print('ok')"`
Expected: prints `ok`. (PyYAML ships with the env via other deps; if missing, validate the frontmatter by inspection.)

Confirm the referenced files exist:

Run: `ls .claude/skills/alpaca-trader/workflows/morning.js .claude/skills/alpaca-trader/workflows/evening.js .claude/skills/alpaca-trader/config.json .claude/skills/alpaca-trader/lib/safety.md .claude/skills/alpaca-trader/lib/signals.md`
Expected: all five listed, no "No such file".

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/alpaca-trader/SKILL.md
git commit -m "feat(skill): add alpaca-trader SKILL.md SOP"
```

---

## Task 10: Memories seed + end-to-end dry run + docs

**Files:**
- Create: `memories/README.md`
- Create: `memories/tickers/.gitkeep`, `memories/sectors/.gitkeep`, `memories/ledger/.gitkeep`
- Modify: `CLAUDE.md` (module map + tests note), `CHANGELOG.md`

- [ ] **Step 1: Seed the memories tree**

```bash
mkdir -p memories/tickers memories/sectors memories/ledger
touch memories/tickers/.gitkeep memories/sectors/.gitkeep memories/ledger/.gitkeep
```

Write `memories/README.md`:

```markdown
# Trading memories

Knowledge base maintained by the `alpaca-trader` skill.

- `tickers/<SYM>.md` — per-company distilled knowledge (theses, what worked).
- `sectors/<sector>.md` — per-industry knowledge.
- `patterns.md` — generalizable trading patterns that have repeated.
- `lessons.md` — mistakes and their corrections.
- `ledger/<YYYY-MM>.jsonl` — append-only decision ledger (machine source of
  truth for outcome attribution). Do not hand-edit.
- `performance.md` — rolling return vs SPY, Sharpe, drawdown, hit-rate.

Entity files are edited by section via `ophir.trading.memory.upsert_section`.
```

- [ ] **Step 2: Full automated verification**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy src/ophir && uv run pytest`
Expected: all clean / green. (Run `uv run ruff format .` and re-commit if the format check fails.)

Run: `uv run ophir trade --help`
Expected: lists `gate`, `record`, `close`, `performance`.

- [ ] **Step 3: Manual paper dry-run checklist** (requires Alpaca MCP creds; not automated)

Document the result of each in the commit message or a scratch note. This is the integration gate the unit tests cannot cover:

- [ ] `get_clock` reachable through the Alpaca MCP and `account_mode` reads `paper`.
- [ ] Morning pass on a **tiny** `shortlist_size` (set to 2 temporarily) produces proposals, each is gated, and at least one places on the paper account.
- [ ] A `DecisionRecord` lands in `memories/ledger/<month>.jsonl`.
- [ ] Evening pass scores it and writes/updates a `memories/tickers/<SYM>.md` section and `memories/performance.md`.
- [ ] Reset `shortlist_size` back to 15.

- [ ] **Step 4: Update docs**

- Add a `trading` row to the `CLAUDE.md` module map: "`trading/` | Deterministic core for the `alpaca-trader` skill: safety gate, ledger, metrics, signals, memory, exposure join, outcome scoring; CLI under `ophir trade`."
- Add to the `CLAUDE.md` Tests section: "`trading` (types, config, safety, ledger, metrics, signals, memory, exposure, outcomes, forecast, cli)."
- Add a CHANGELOG entry in the existing format describing the alpaca-trader skill + trading core.

- [ ] **Step 5: Commit**

```bash
git add memories/ CLAUDE.md CHANGELOG.md
git commit -m "feat(skill): seed memories tree and document alpaca-trader"
```

---

## Self-Review (completed during planning)

**Spec coverage vs. design §1–§11 (orchestration parts):**
- §2 file layout (skill dir, memories tree) → Tasks 5–10. ✓
- §3 sleeves → encoded in `morning.js` seeds + `signals.md` + gate caps. ✓
- §4 morning workflow (preflight, screen, analyze, verify, gate, place, record) → `morning.js` (Task 7) + `SKILL.md` morning SOP (Task 9) + CLI `gate`/`record`. ✓
- §5 evening workflow (pull, attribute, update memories, performance) → `outcomes.py` (Task 2) + `evening.js` (Task 8) + `SKILL.md` evening SOP + CLI `close`/`performance` + `upsert_section`. ✓
- §6 safety interlock (paper↔live) → enforced by Plan 1 gate; `config.json` paper (Task 5); invariant in `SKILL.md`. ✓
- §7 config knobs → `config.json` (Task 5), consumed by `morning.js`. ✓
- §8 signal blend + graceful ophir-absent → `signals.py` (Plan 1) + `forecast.py` (Task 3) + `signals.md`. ✓
- §9 graduation metrics → `performance` CLI + Phase-1 tracking note in `SKILL.md`. ✓
- §10 caveats (paper fidelity, ophir optional, sentiment soft, fuzzy attribution) → encoded in `signals.md`, `forecast.py`, `SKILL.md`. ✓

**Testability honesty:** Python tasks (1–4) are TDD with real unit tests. Artifact tasks (5–9) are verified by load/parse/syntax checks (`load_config`, `node --check`, YAML parse) plus the **Task 10 manual paper dry-run**, which is the genuine integration gate and is explicitly called out as not automatable here (needs live MCP creds + paper account).

**Placeholder scan:** the `SKILL.md` snapshot-construction step intentionally describes a `python -c` pattern rather than a fixed one-liner because the exact field values come from live MCP data at run time; the *function* used (`build_snapshot`) and the output contract (`AccountSnapshot`-shaped `snapshot.json` the gate consumes) are concrete. No code task contains TBD/TODO.

**Type/interface consistency:** `morning.js` proposal keys match `ProposedOrder` fields and the `gate` CLI's expected `order.json`. `evening.js` `openTheses` keys match what `score_record` produces. CLI `record`/`close` JSON matches `record_to_dict`. ✓
````
