# Alpaca-Trader Deterministic Core — Implementation Plan (Plan 1 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the deterministic, unit-tested Python core (`src/ophir/trading/`) that backs the `alpaca-trader` skill: typed config loading, a non-overridable pre-trade safety gate, a decision ledger, performance metrics, signal blending, and entity-memory section editing — exposed to the skill through an `ophir trade` CLI subapp.

**Architecture:** A new `ophir.trading` subpackage of pure, side-effect-light functions over frozen dataclasses. The safety gate is the centerpiece: a single pure function `evaluate_order(order, snapshot, config) -> GateDecision` that every order must pass. All filesystem/network/Alpaca interaction lives *outside* this core (in the skill's Workflow scripts and the main agent, built in Plan 2); the core only consumes already-gathered snapshots and returns decisions/records/strings. A thin Typer subapp (`ophir trade …`) lets the skill invoke the core via Bash with JSON in / JSON out.

**Tech Stack:** Python 3.10+, dataclasses + `enum`, `typer` (CLI), `pytest` + `pytest-mock`, strict `mypy`, `ruff`. No new third-party dependencies.

## Global Constraints

- **Live code path:** all new modules under `src/ophir/trading/`; tests under `tests/`. (Verbatim repo rule: live code is `src/ophir/` only.)
- **Strict typing:** `[tool.mypy] strict = true`, `python_version = "3.10"`, `warn_unused_ignores = true` over `files = ["src/ophir"]` — every function fully annotated; no unneeded `# type: ignore`.
- **Ruff:** `target-version = "py312"`, `line-length = 100`, rule sets include `ANN` (annotations), `N` (naming), `UP`, `I`, `B`, `SIM`. `tests/**` is exempt from `ANN` only.
- **Tests:** `testpaths = ["tests"]`, `addopts = "-q -ra --strict-markers --strict-config"`, `filterwarnings = ["error", …]` — every test must run warning-clean.
- **Python floor 3.10:** use `X | None`, `frozenset[str]`, `dict[str, float]` syntax (no `typing.Optional`/`Dict`). `enum.StrEnum` is 3.11+, so use `class Foo(str, Enum)` mixins, NOT `StrEnum`.
- **No `Date.now`/wall-clock in core logic:** any date/month is passed in as an explicit `str` argument so functions stay deterministic and testable.
- **Determinism:** core functions are pure where possible; the only I/O functions are the explicit `read_*`/`write_*`/`append_*` helpers in `ledger.py` and `memory.py`, which take explicit paths.
- **Style:** brief numpy-style docstrings on public functions (match existing modules); no inline comments unless they explain a non-obvious *why*.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/ophir/trading/__init__.py` | Package marker; re-export the public domain types and `evaluate_order`. |
| `src/ophir/trading/types.py` | Frozen dataclasses + enums: `Sleeve`, `Side`, `AssetClass`, `GateAction`, `ProposedOrder`, `AccountSnapshot`, `GateDecision`, `GuardrailLimits`, `TradingConfig`, `DecisionRecord`, `SignalWeights`. No logic. |
| `src/ophir/trading/config.py` | `load_config(path) -> TradingConfig`: parse + validate `config.json`; range-check limits; raise `ConfigError` on bad input. |
| `src/ophir/trading/safety.py` | `evaluate_order(order, snapshot, config) -> GateDecision`: the single non-overridable pre-trade gate. |
| `src/ophir/trading/ledger.py` | `append_decision`, `load_decisions`, `DecisionRecord` (de)serialization over `ledger/<YYYY-MM>.jsonl`. |
| `src/ophir/trading/metrics.py` | Pure metrics: `total_return`, `sharpe`, `max_drawdown`, `hit_rate`, `calibration_error`. |
| `src/ophir/trading/signals.py` | `normalize`, `blend_signals` (graceful when ophir signal absent). |
| `src/ophir/trading/memory.py` | `upsert_section` (pure string edit), `read_memory`, `write_memory`. |
| `src/ophir/trading/cli.py` | Typer subapp `app` with `gate`, `record`, `score`, `memory-update` commands (JSON in/out via files or stdin). |
| `src/ophir/cli.py` | **Modify:** mount the trading subapp as `app.add_typer(trading_cli.app, name="trade")`. |
| `tests/test_trading_types.py` … `tests/test_trading_cli.py` | One test module per source module. |

**Exposure-join note (read before Task 3):** Alpaca positions do **not** carry our `Sleeve`/sector tags. The *caller* (Plan 2's agent) is responsible for joining live Alpaca positions with the ledger to produce the pre-aggregated exposure maps inside `AccountSnapshot`. The safety gate consumes those maps and never queries Alpaca itself. This keeps the gate pure and unit-testable.

---

## Task 1: Package scaffold + domain types

**Files:**
- Create: `src/ophir/trading/__init__.py`
- Create: `src/ophir/trading/types.py`
- Test: `tests/test_trading_types.py`

**Interfaces:**
- Consumes: nothing.
- Produces (used by every later task):
  - `class Sleeve(str, Enum)`: `CORE = "core"`, `TACTICAL = "tactical"`.
  - `class Side(str, Enum)`: `BUY = "buy"`, `SELL = "sell"`.
  - `class AssetClass(str, Enum)`: `EQUITY = "equity"`, `OPTION = "option"`.
  - `class GateAction(str, Enum)`: `APPROVE = "approve"`, `RESIZE = "resize"`, `REJECT = "reject"`.
  - `@dataclass(frozen=True) ProposedOrder(symbol: str, side: Side, sleeve: Sleeve, asset_class: AssetClass, notional: float, sector: str | None, is_defined_risk: bool, is_short_option: bool)`.
  - `@dataclass(frozen=True) AccountSnapshot(equity: float, cash: float, day_pl: float, open_position_count: int, held_symbols: frozenset[str], symbol_exposure: Mapping[str, float], sector_exposure: Mapping[str, float], sleeve_exposure: Mapping[Sleeve, float], option_premium_at_risk: float, account_mode: str)`.
  - `@dataclass(frozen=True) GateDecision(action: GateAction, approved_notional: float, reasons: tuple[str, ...])`.
  - `@dataclass(frozen=True) GuardrailLimits(max_position_pct, max_option_premium_pct, halt_new_entries_day_loss_pct, flatten_tactical_day_loss_pct, max_deployed_pct, min_cash_pct, max_core_pct, max_tactical_pct, max_sector_pct, max_open_positions: int, max_total_option_premium_pct: float)` — all `float` except `max_open_positions: int`.
  - `@dataclass(frozen=True) TradingConfig(account_mode: str, limits: GuardrailLimits, shortlist_size: int, verify_votes: int, depth: str)`.
  - `@dataclass(frozen=True) SignalWeights(ophir: float, momentum: float, sentiment: float)`.
  - `@dataclass(frozen=True) DecisionRecord(date: str, symbol: str, sleeve: Sleeve, side: Side, asset_class: AssetClass, notional: float, sector: str | None, thesis: str, signals: Mapping[str, float], entry_price: float | None, target: float | None, stop: float | None, order_id: str | None, status: str, realized_pl: float | None, scored: bool)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trading_types.py
from ophir.trading.types import (
    AccountSnapshot,
    AssetClass,
    GateAction,
    GateDecision,
    ProposedOrder,
    Side,
    Sleeve,
)


def test_enums_have_string_values() -> None:
    assert Sleeve.CORE.value == "core"
    assert Side.BUY.value == "buy"
    assert AssetClass.OPTION.value == "option"
    assert GateAction.RESIZE.value == "resize"


def test_proposed_order_is_frozen() -> None:
    order = ProposedOrder(
        symbol="AAPL",
        side=Side.BUY,
        sleeve=Sleeve.CORE,
        asset_class=AssetClass.EQUITY,
        notional=1000.0,
        sector="Technology",
        is_defined_risk=True,
        is_short_option=False,
    )
    assert order.symbol == "AAPL"
    import dataclasses

    try:
        order.notional = 2.0  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("ProposedOrder must be frozen")


def test_snapshot_and_decision_construct() -> None:
    snap = AccountSnapshot(
        equity=100_000.0,
        cash=40_000.0,
        day_pl=-500.0,
        open_position_count=3,
        held_symbols=frozenset({"AAPL"}),
        symbol_exposure={"AAPL": 5_000.0},
        sector_exposure={"Technology": 5_000.0},
        sleeve_exposure={Sleeve.CORE: 5_000.0},
        option_premium_at_risk=0.0,
        account_mode="paper",
    )
    assert snap.sleeve_exposure[Sleeve.CORE] == 5_000.0
    dec = GateDecision(action=GateAction.APPROVE, approved_notional=1000.0, reasons=())
    assert dec.action is GateAction.APPROVE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_trading_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ophir.trading'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ophir/trading/__init__.py
"""Deterministic core for the alpaca-trader skill.

Pure, side-effect-light building blocks: domain types, config loading, the
pre-trade safety gate, the decision ledger, performance metrics, signal
blending, and entity-memory editing. All Alpaca/MCP and filesystem orchestration
lives outside this package (in the skill's Workflow scripts and main agent).
"""

from ophir.trading.types import (
    AccountSnapshot,
    AssetClass,
    DecisionRecord,
    GateAction,
    GateDecision,
    GuardrailLimits,
    ProposedOrder,
    Side,
    SignalWeights,
    Sleeve,
    TradingConfig,
)

__all__ = [
    "AccountSnapshot",
    "AssetClass",
    "DecisionRecord",
    "GateAction",
    "GateDecision",
    "GuardrailLimits",
    "ProposedOrder",
    "Side",
    "SignalWeights",
    "Sleeve",
    "TradingConfig",
]
```

```python
# src/ophir/trading/types.py
"""Frozen domain types for the trading core. No logic lives here."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum


class Sleeve(str, Enum):
    """Which strategy book an order/position belongs to."""

    CORE = "core"
    TACTICAL = "tactical"


class Side(str, Enum):
    """Order direction."""

    BUY = "buy"
    SELL = "sell"


class AssetClass(str, Enum):
    """Instrument class the gate distinguishes."""

    EQUITY = "equity"
    OPTION = "option"


class GateAction(str, Enum):
    """Outcome of the pre-trade safety gate."""

    APPROVE = "approve"
    RESIZE = "resize"
    REJECT = "reject"


@dataclass(frozen=True)
class ProposedOrder:
    """A trade the agent wants to place, before the safety gate runs.

    ``notional`` is the intended dollar exposure; for options it is the premium
    at risk. ``is_defined_risk`` is ``True`` for equities and defined-risk option
    structures; ``is_short_option`` flags a short option leg.
    """

    symbol: str
    side: Side
    sleeve: Sleeve
    asset_class: AssetClass
    notional: float
    sector: str | None
    is_defined_risk: bool
    is_short_option: bool


@dataclass(frozen=True)
class AccountSnapshot:
    """Pre-aggregated account state the gate reasons over.

    Exposure maps are produced upstream by joining live Alpaca positions with the
    decision ledger; the gate never queries Alpaca. ``day_pl`` is dollars
    (negative on a losing day). Exposures are absolute dollar values.
    """

    equity: float
    cash: float
    day_pl: float
    open_position_count: int
    held_symbols: frozenset[str]
    symbol_exposure: Mapping[str, float]
    sector_exposure: Mapping[str, float]
    sleeve_exposure: Mapping[Sleeve, float]
    option_premium_at_risk: float
    account_mode: str


@dataclass(frozen=True)
class GateDecision:
    """Result of :func:`ophir.trading.safety.evaluate_order`."""

    action: GateAction
    approved_notional: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class GuardrailLimits:
    """Hard, non-overridable risk limits (fractions of equity unless noted)."""

    max_position_pct: float
    max_option_premium_pct: float
    halt_new_entries_day_loss_pct: float
    flatten_tactical_day_loss_pct: float
    max_deployed_pct: float
    min_cash_pct: float
    max_core_pct: float
    max_tactical_pct: float
    max_sector_pct: float
    max_open_positions: int
    max_total_option_premium_pct: float


@dataclass(frozen=True)
class TradingConfig:
    """Top-level skill configuration loaded from ``config.json``."""

    account_mode: str
    limits: GuardrailLimits
    shortlist_size: int
    verify_votes: int
    depth: str


@dataclass(frozen=True)
class SignalWeights:
    """Per-sleeve weighting of the blended signal components."""

    ophir: float
    momentum: float
    sentiment: float


@dataclass(frozen=True)
class DecisionRecord:
    """One row of the decision ledger (the outcome-attribution source of truth)."""

    date: str
    symbol: str
    sleeve: Sleeve
    side: Side
    asset_class: AssetClass
    notional: float
    sector: str | None
    thesis: str
    signals: Mapping[str, float]
    entry_price: float | None
    target: float | None
    stop: float | None
    order_id: str | None
    status: str
    realized_pl: float | None
    scored: bool
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_trading_types.py -v && uv run mypy src/ophir && uv run ruff check src/ophir/trading tests/test_trading_types.py`
Expected: tests PASS, mypy clean, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/ophir/trading/__init__.py src/ophir/trading/types.py tests/test_trading_types.py
git commit -m "feat(trading): add trading-core domain types"
```

---

## Task 2: Config loader

**Files:**
- Create: `src/ophir/trading/config.py`
- Test: `tests/test_trading_config.py`

**Interfaces:**
- Consumes: `GuardrailLimits`, `TradingConfig` (Task 1).
- Produces:
  - `class ConfigError(ValueError)`.
  - `def load_config(path: str | Path) -> TradingConfig`.
  - Expected JSON shape: top-level keys `account_mode` (`"paper"|"live"`), `depth` (`"lean"|"balanced"|"deep"`), `shortlist_size` (int > 0), `verify_votes` (int > 0), and a nested `limits` object with all eleven `GuardrailLimits` fields.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trading_config.py
import json
from pathlib import Path

import pytest

from ophir.trading.config import ConfigError, load_config

VALID = {
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


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data))
    return p


def test_loads_valid_config(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, VALID))
    assert cfg.account_mode == "paper"
    assert cfg.limits.max_position_pct == 0.05
    assert cfg.shortlist_size == 15


def test_rejects_bad_account_mode(tmp_path: Path) -> None:
    bad = {**VALID, "account_mode": "real"}
    with pytest.raises(ConfigError, match="account_mode"):
        load_config(_write(tmp_path, bad))


def test_rejects_missing_limit(tmp_path: Path) -> None:
    bad = {**VALID, "limits": {k: v for k, v in VALID["limits"].items() if k != "min_cash_pct"}}
    with pytest.raises(ConfigError, match="min_cash_pct"):
        load_config(_write(tmp_path, bad))


def test_rejects_non_positive_shortlist(tmp_path: Path) -> None:
    bad = {**VALID, "shortlist_size": 0}
    with pytest.raises(ConfigError, match="shortlist_size"):
        load_config(_write(tmp_path, bad))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_trading_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ophir.trading.config'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ophir/trading/config.py
"""Load and validate the alpaca-trader ``config.json`` into typed objects."""

import json
from dataclasses import fields
from pathlib import Path

from ophir.trading.types import GuardrailLimits, TradingConfig

_ACCOUNT_MODES = {"paper", "live"}
_DEPTHS = {"lean", "balanced", "deep"}


class ConfigError(ValueError):
    """Raised when ``config.json`` is missing keys or has invalid values."""


def _limits_from(data: dict[str, object]) -> GuardrailLimits:
    names = [f.name for f in fields(GuardrailLimits)]
    missing = [n for n in names if n not in data]
    if missing:
        raise ConfigError(f"limits missing keys: {', '.join(sorted(missing))}")
    kwargs: dict[str, object] = {}
    for name in names:
        value = data[name]
        if name == "max_open_positions":
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ConfigError(f"limits.{name} must be a positive int")
            kwargs[name] = value
        else:
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                raise ConfigError(f"limits.{name} must be a non-negative number")
            kwargs[name] = float(value)
    return GuardrailLimits(**kwargs)  # type: ignore[arg-type]


def load_config(path: str | Path) -> TradingConfig:
    """Parse and validate the trading config file.

    Parameters
    ----------
    path : str or Path
        Location of ``config.json``.

    Returns
    -------
    TradingConfig
        The validated configuration.

    Raises
    ------
    ConfigError
        If a key is missing or a value is out of range.
    """
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a JSON object")
    mode = raw.get("account_mode")
    if mode not in _ACCOUNT_MODES:
        raise ConfigError(f"account_mode must be one of {sorted(_ACCOUNT_MODES)}")
    depth = raw.get("depth")
    if depth not in _DEPTHS:
        raise ConfigError(f"depth must be one of {sorted(_DEPTHS)}")
    for key in ("shortlist_size", "verify_votes"):
        value = raw.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConfigError(f"{key} must be a positive int")
    limits_raw = raw.get("limits")
    if not isinstance(limits_raw, dict):
        raise ConfigError("limits must be a JSON object")
    return TradingConfig(
        account_mode=str(mode),
        limits=_limits_from(limits_raw),
        shortlist_size=int(raw["shortlist_size"]),
        verify_votes=int(raw["verify_votes"]),
        depth=str(depth),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_trading_config.py -v && uv run mypy src/ophir && uv run ruff check src/ophir/trading tests/test_trading_config.py`
Expected: tests PASS, mypy clean, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/ophir/trading/config.py tests/test_trading_config.py
git commit -m "feat(trading): add config loader with validation"
```

---

## Task 3: Safety gate (the centerpiece)

**Files:**
- Create: `src/ophir/trading/safety.py`
- Test: `tests/test_trading_safety.py`

**Interfaces:**
- Consumes: `ProposedOrder`, `AccountSnapshot`, `TradingConfig`, `GuardrailLimits`, `GateDecision`, `GateAction`, `Sleeve`, `Side`, `AssetClass` (Task 1).
- Produces: `def evaluate_order(order: ProposedOrder, snapshot: AccountSnapshot, config: TradingConfig) -> GateDecision`.

**Gate semantics (implement exactly):**
1. **Account-mode interlock** — if `snapshot.account_mode != config.account_mode` → `REJECT` (reason `"account-mode mismatch: snapshot=… config=…"`).
2. **Naked short option** — if option, `is_short_option` and not `is_defined_risk` → `REJECT` (`"naked short option not allowed"`).
3. **Sells are risk-reducing** — if `side is SELL` → `APPROVE` at full `notional` (after checks 1–2).
4. **Daily kill-switch** — for a BUY, `day_loss_frac = max(0, -day_pl) / equity`; if `day_loss_frac >= halt_new_entries_day_loss_pct` → `REJECT` (`"daily kill-switch: new entries halted"`).
5. **Max open positions** — BUY of a symbol not in `held_symbols` while `open_position_count >= max_open_positions` → `REJECT` (`"max open positions reached"`).
6. **Sizing caps** (BUY) — compute each remaining-headroom cap, take the minimum, and remember which one bound:
   - per-position: `max_position_pct*equity - symbol_exposure[symbol]`
   - sleeve: `(max_core_pct if CORE else max_tactical_pct)*equity - sleeve_exposure[sleeve]`
   - sector (if `sector` not None): `max_sector_pct*equity - sector_exposure[sector]`
   - deployment: `max_deployed_pct*equity - sum(symbol_exposure.values())`
   - cash floor: `cash - min_cash_pct*equity`
   - option-per-contract (if option): `max_option_premium_pct*equity`
   - option-total (if option): `max_total_option_premium_pct*equity - option_premium_at_risk`
   - `allowed = min(order.notional, *caps)`
   - if `allowed <= 0` → `REJECT` (reason names the binding cap)
   - elif `allowed < order.notional` → `RESIZE` to `allowed` (reason names the binding cap)
   - else → `APPROVE` at `order.notional`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trading_safety.py
from ophir.trading.safety import evaluate_order
from ophir.trading.types import (
    AccountSnapshot,
    AssetClass,
    GateAction,
    GuardrailLimits,
    ProposedOrder,
    Side,
    Sleeve,
    TradingConfig,
)

LIMITS = GuardrailLimits(
    max_position_pct=0.05,
    max_option_premium_pct=0.02,
    halt_new_entries_day_loss_pct=0.02,
    flatten_tactical_day_loss_pct=0.04,
    max_deployed_pct=0.80,
    min_cash_pct=0.20,
    max_core_pct=0.50,
    max_tactical_pct=0.30,
    max_sector_pct=0.25,
    max_open_positions=15,
    max_total_option_premium_pct=0.10,
)
CONFIG = TradingConfig(
    account_mode="paper", limits=LIMITS, shortlist_size=15, verify_votes=1, depth="lean"
)


def _snapshot(**overrides: object) -> AccountSnapshot:
    base: dict[str, object] = dict(
        equity=100_000.0,
        cash=50_000.0,
        day_pl=0.0,
        open_position_count=0,
        held_symbols=frozenset(),
        symbol_exposure={},
        sector_exposure={},
        sleeve_exposure={},
        option_premium_at_risk=0.0,
        account_mode="paper",
    )
    base.update(overrides)
    return AccountSnapshot(**base)  # type: ignore[arg-type]


def _order(**overrides: object) -> ProposedOrder:
    base: dict[str, object] = dict(
        symbol="AAPL",
        side=Side.BUY,
        sleeve=Sleeve.CORE,
        asset_class=AssetClass.EQUITY,
        notional=1_000.0,
        sector="Technology",
        is_defined_risk=True,
        is_short_option=False,
    )
    base.update(overrides)
    return ProposedOrder(**base)  # type: ignore[arg-type]


def test_approves_small_order() -> None:
    decision = evaluate_order(_order(), _snapshot(), CONFIG)
    assert decision.action is GateAction.APPROVE
    assert decision.approved_notional == 1_000.0


def test_account_mode_mismatch_rejects() -> None:
    decision = evaluate_order(_order(), _snapshot(account_mode="live"), CONFIG)
    assert decision.action is GateAction.REJECT
    assert any("account-mode" in r for r in decision.reasons)


def test_naked_short_option_rejected() -> None:
    order = _order(asset_class=AssetClass.OPTION, is_short_option=True, is_defined_risk=False)
    decision = evaluate_order(order, _snapshot(), CONFIG)
    assert decision.action is GateAction.REJECT
    assert any("naked short" in r for r in decision.reasons)


def test_sell_always_approved_full() -> None:
    decision = evaluate_order(
        _order(side=Side.SELL, notional=999_999.0), _snapshot(), CONFIG
    )
    assert decision.action is GateAction.APPROVE
    assert decision.approved_notional == 999_999.0


def test_kill_switch_halts_buys() -> None:
    decision = evaluate_order(_order(), _snapshot(day_pl=-2_500.0), CONFIG)
    assert decision.action is GateAction.REJECT
    assert any("kill-switch" in r for r in decision.reasons)


def test_max_open_positions_blocks_new_symbol() -> None:
    snap = _snapshot(open_position_count=15, held_symbols=frozenset({"MSFT"}))
    decision = evaluate_order(_order(symbol="AAPL"), snap, CONFIG)
    assert decision.action is GateAction.REJECT
    assert any("max open positions" in r for r in decision.reasons)


def test_resizes_to_per_position_cap() -> None:
    # per-position cap = 5% * 100k = 5_000; existing 4_000 -> 1_000 headroom
    snap = _snapshot(symbol_exposure={"AAPL": 4_000.0})
    decision = evaluate_order(_order(notional=3_000.0), snap, CONFIG)
    assert decision.action is GateAction.RESIZE
    assert decision.approved_notional == 1_000.0
    assert any("per-position" in r for r in decision.reasons)


def test_rejects_when_cash_floor_binds_to_zero() -> None:
    # cash floor headroom = cash - 20% equity = 20_000 - 20_000 = 0
    decision = evaluate_order(_order(notional=1_000.0), _snapshot(cash=20_000.0), CONFIG)
    assert decision.action is GateAction.REJECT
    assert any("cash floor" in r for r in decision.reasons)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_trading_safety.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ophir.trading.safety'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ophir/trading/safety.py
"""The single non-overridable pre-trade safety gate.

Every order an agent proposes must pass through :func:`evaluate_order`. The
function is pure: it reads a pre-aggregated :class:`AccountSnapshot` and the
:class:`TradingConfig` limits and returns a :class:`GateDecision` of approve,
resize, or reject. It never touches Alpaca, the network, or the filesystem.
"""

from ophir.trading.types import (
    AccountSnapshot,
    AssetClass,
    GateAction,
    GateDecision,
    ProposedOrder,
    Side,
    Sleeve,
    TradingConfig,
)


def evaluate_order(
    order: ProposedOrder, snapshot: AccountSnapshot, config: TradingConfig
) -> GateDecision:
    """Approve, resize, or reject ``order`` against the hard guardrails.

    See the plan's "Gate semantics" for the exact ordering of checks.
    """
    limits = config.limits

    if snapshot.account_mode != config.account_mode:
        return GateDecision(
            GateAction.REJECT,
            0.0,
            (
                f"account-mode mismatch: snapshot={snapshot.account_mode} "
                f"config={config.account_mode}",
            ),
        )

    if (
        order.asset_class is AssetClass.OPTION
        and order.is_short_option
        and not order.is_defined_risk
    ):
        return GateDecision(GateAction.REJECT, 0.0, ("naked short option not allowed",))

    if order.side is Side.SELL:
        return GateDecision(GateAction.APPROVE, order.notional, ())

    equity = snapshot.equity
    day_loss_frac = max(0.0, -snapshot.day_pl) / equity if equity > 0 else 0.0
    if day_loss_frac >= limits.halt_new_entries_day_loss_pct:
        return GateDecision(GateAction.REJECT, 0.0, ("daily kill-switch: new entries halted",))

    if order.symbol not in snapshot.held_symbols and (
        snapshot.open_position_count >= limits.max_open_positions
    ):
        return GateDecision(GateAction.REJECT, 0.0, ("max open positions reached",))

    sleeve_cap_pct = (
        limits.max_core_pct if order.sleeve is Sleeve.CORE else limits.max_tactical_pct
    )
    deployed = sum(snapshot.symbol_exposure.values())

    caps: list[tuple[str, float]] = [
        (
            "per-position cap",
            limits.max_position_pct * equity - snapshot.symbol_exposure.get(order.symbol, 0.0),
        ),
        (
            f"{order.sleeve.value} sleeve cap",
            sleeve_cap_pct * equity - snapshot.sleeve_exposure.get(order.sleeve, 0.0),
        ),
        ("deployment cap", limits.max_deployed_pct * equity - deployed),
        ("cash floor", snapshot.cash - limits.min_cash_pct * equity),
    ]
    if order.sector is not None:
        caps.append(
            (
                "sector cap",
                limits.max_sector_pct * equity - snapshot.sector_exposure.get(order.sector, 0.0),
            )
        )
    if order.asset_class is AssetClass.OPTION:
        caps.append(("option per-contract cap", limits.max_option_premium_pct * equity))
        caps.append(
            (
                "total option premium cap",
                limits.max_total_option_premium_pct * equity - snapshot.option_premium_at_risk,
            )
        )

    binding_name, binding_value = min(caps, key=lambda kv: kv[1])
    allowed = min(order.notional, binding_value)

    if allowed <= 0:
        return GateDecision(GateAction.REJECT, 0.0, (f"{binding_name} leaves no headroom",))
    if allowed < order.notional:
        return GateDecision(GateAction.RESIZE, allowed, (f"resized to {binding_name}",))
    return GateDecision(GateAction.APPROVE, order.notional, ())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_trading_safety.py -v && uv run mypy src/ophir && uv run ruff check src/ophir/trading tests/test_trading_safety.py`
Expected: all tests PASS, mypy clean, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/ophir/trading/safety.py tests/test_trading_safety.py
git commit -m "feat(trading): add non-overridable pre-trade safety gate"
```

---

## Task 4: Decision ledger

**Files:**
- Create: `src/ophir/trading/ledger.py`
- Test: `tests/test_trading_ledger.py`

**Interfaces:**
- Consumes: `DecisionRecord`, `Sleeve`, `Side`, `AssetClass` (Task 1).
- Produces:
  - `def record_to_dict(record: DecisionRecord) -> dict[str, object]`
  - `def record_from_dict(data: Mapping[str, object]) -> DecisionRecord`
  - `def ledger_path(ledger_dir: str | Path, month: str) -> Path` (`<ledger_dir>/<month>.jsonl`, `month` = `"YYYY-MM"`)
  - `def append_decision(ledger_dir: str | Path, month: str, record: DecisionRecord) -> None` (creates dir, appends one JSON line)
  - `def load_decisions(ledger_dir: str | Path, month: str) -> list[DecisionRecord]` (returns `[]` if file absent)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trading_ledger.py
from pathlib import Path

from ophir.trading.ledger import append_decision, load_decisions, record_from_dict, record_to_dict
from ophir.trading.types import AssetClass, DecisionRecord, Side, Sleeve

REC = DecisionRecord(
    date="2026-06-18",
    symbol="AAPL",
    sleeve=Sleeve.CORE,
    side=Side.BUY,
    asset_class=AssetClass.EQUITY,
    notional=1_000.0,
    sector="Technology",
    thesis="ophir bullish + momentum confirm",
    signals={"ophir": 0.7, "momentum": 0.4, "sentiment": 0.1},
    entry_price=195.0,
    target=210.0,
    stop=185.0,
    order_id="abc-123",
    status="open",
    realized_pl=None,
    scored=False,
)


def test_roundtrip_dict() -> None:
    assert record_from_dict(record_to_dict(REC)) == REC


def test_append_then_load(tmp_path: Path) -> None:
    append_decision(tmp_path, "2026-06", REC)
    append_decision(tmp_path, "2026-06", REC)
    loaded = load_decisions(tmp_path, "2026-06")
    assert len(loaded) == 2
    assert loaded[0] == REC


def test_load_missing_returns_empty(tmp_path: Path) -> None:
    assert load_decisions(tmp_path, "1999-01") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_trading_ledger.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ophir.trading.ledger'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ophir/trading/ledger.py
"""Append-only JSONL decision ledger — the outcome-attribution source of truth."""

import json
from collections.abc import Mapping
from pathlib import Path

from ophir.trading.types import AssetClass, DecisionRecord, Side, Sleeve


def record_to_dict(record: DecisionRecord) -> dict[str, object]:
    """Serialize a record to a JSON-safe dict (enums become their values)."""
    return {
        "date": record.date,
        "symbol": record.symbol,
        "sleeve": record.sleeve.value,
        "side": record.side.value,
        "asset_class": record.asset_class.value,
        "notional": record.notional,
        "sector": record.sector,
        "thesis": record.thesis,
        "signals": dict(record.signals),
        "entry_price": record.entry_price,
        "target": record.target,
        "stop": record.stop,
        "order_id": record.order_id,
        "status": record.status,
        "realized_pl": record.realized_pl,
        "scored": record.scored,
    }


def record_from_dict(data: Mapping[str, object]) -> DecisionRecord:
    """Inverse of :func:`record_to_dict`."""
    signals_raw = data["signals"]
    assert isinstance(signals_raw, dict)
    return DecisionRecord(
        date=str(data["date"]),
        symbol=str(data["symbol"]),
        sleeve=Sleeve(str(data["sleeve"])),
        side=Side(str(data["side"])),
        asset_class=AssetClass(str(data["asset_class"])),
        notional=float(data["notional"]),  # type: ignore[arg-type]
        sector=None if data["sector"] is None else str(data["sector"]),
        thesis=str(data["thesis"]),
        signals={str(k): float(v) for k, v in signals_raw.items()},
        entry_price=None if data["entry_price"] is None else float(data["entry_price"]),  # type: ignore[arg-type]
        target=None if data["target"] is None else float(data["target"]),  # type: ignore[arg-type]
        stop=None if data["stop"] is None else float(data["stop"]),  # type: ignore[arg-type]
        order_id=None if data["order_id"] is None else str(data["order_id"]),
        status=str(data["status"]),
        realized_pl=None if data["realized_pl"] is None else float(data["realized_pl"]),  # type: ignore[arg-type]
        scored=bool(data["scored"]),
    )


def ledger_path(ledger_dir: str | Path, month: str) -> Path:
    """Return the JSONL path for a ``YYYY-MM`` month."""
    return Path(ledger_dir) / f"{month}.jsonl"


def append_decision(ledger_dir: str | Path, month: str, record: DecisionRecord) -> None:
    """Append one decision as a JSON line, creating the directory if needed."""
    path = ledger_path(ledger_dir, month)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record_to_dict(record)) + "\n")


def load_decisions(ledger_dir: str | Path, month: str) -> list[DecisionRecord]:
    """Load all decisions for a month, or ``[]`` if the file does not exist."""
    path = ledger_path(ledger_dir, month)
    if not path.exists():
        return []
    records: list[DecisionRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(record_from_dict(json.loads(line)))
    return records
```

> **Note on the `# type: ignore[arg-type]` markers:** `warn_unused_ignores=true` means an ignore that doesn't fire will *fail* mypy. Run mypy in Step 4; if any ignore is flagged unused, delete that specific one (it means mypy already narrowed `data[...]` acceptably). Conversely if mypy reports an `arg-type` error on a line lacking one, add it there. Adjust until mypy is clean.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_trading_ledger.py -v && uv run mypy src/ophir && uv run ruff check src/ophir/trading tests/test_trading_ledger.py`
Expected: tests PASS; mypy clean (fix ignore markers per the note); ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/ophir/trading/ledger.py tests/test_trading_ledger.py
git commit -m "feat(trading): add JSONL decision ledger"
```

---

## Task 5: Performance metrics

**Files:**
- Create: `src/ophir/trading/metrics.py`
- Test: `tests/test_trading_metrics.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (operates on plain sequences).
- Produces:
  - `def total_return(equity_curve: Sequence[float]) -> float` — `(last/first) - 1`; `0.0` if fewer than 2 points or `first == 0`.
  - `def daily_returns(equity_curve: Sequence[float]) -> list[float]` — period-over-period simple returns.
  - `def sharpe(returns: Sequence[float], periods_per_year: int = 252) -> float` — `mean/std * sqrt(periods_per_year)`; `0.0` if `<2` points or `std == 0`.
  - `def max_drawdown(equity_curve: Sequence[float]) -> float` — most-negative peak-to-trough fraction (returned as a non-positive float); `0.0` if empty.
  - `def hit_rate(outcomes: Sequence[bool]) -> float` — fraction `True`; `0.0` if empty.
  - `def calibration_error(predicted: Sequence[float], realized: Sequence[float]) -> float` — mean absolute error; raises `ValueError` on length mismatch; `0.0` if empty.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trading_metrics.py
import math

import pytest

from ophir.trading.metrics import (
    calibration_error,
    daily_returns,
    hit_rate,
    max_drawdown,
    sharpe,
    total_return,
)


def test_total_return() -> None:
    assert total_return([100.0, 110.0]) == pytest.approx(0.10)
    assert total_return([100.0]) == 0.0
    assert total_return([]) == 0.0


def test_daily_returns() -> None:
    assert daily_returns([100.0, 110.0, 99.0]) == pytest.approx([0.10, -0.10])


def test_sharpe_zero_variance_is_zero() -> None:
    assert sharpe([0.01, 0.01, 0.01]) == 0.0


def test_sharpe_positive() -> None:
    s = sharpe([0.01, -0.005, 0.02, 0.0], periods_per_year=252)
    assert s > 0.0
    assert math.isfinite(s)


def test_max_drawdown() -> None:
    assert max_drawdown([100.0, 120.0, 90.0, 110.0]) == pytest.approx(-0.25)
    assert max_drawdown([]) == 0.0


def test_hit_rate() -> None:
    assert hit_rate([True, False, True, True]) == pytest.approx(0.75)
    assert hit_rate([]) == 0.0


def test_calibration_error() -> None:
    assert calibration_error([0.1, 0.2], [0.0, 0.4]) == pytest.approx(0.15)
    with pytest.raises(ValueError, match="length"):
        calibration_error([0.1], [0.1, 0.2])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_trading_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ophir.trading.metrics'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ophir/trading/metrics.py
"""Pure performance metrics over equity curves and decision outcomes."""

import math
from collections.abc import Sequence


def total_return(equity_curve: Sequence[float]) -> float:
    """Fractional return from first to last equity point."""
    if len(equity_curve) < 2 or equity_curve[0] == 0:
        return 0.0
    return equity_curve[-1] / equity_curve[0] - 1.0


def daily_returns(equity_curve: Sequence[float]) -> list[float]:
    """Period-over-period simple returns."""
    out: list[float] = []
    for prev, cur in zip(equity_curve[:-1], equity_curve[1:], strict=False):
        out.append(0.0 if prev == 0 else cur / prev - 1.0)
    return out


def sharpe(returns: Sequence[float], periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio (zero risk-free rate); 0.0 if undefined."""
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    return mean / std * math.sqrt(periods_per_year)


def max_drawdown(equity_curve: Sequence[float]) -> float:
    """Most-negative peak-to-trough drawdown as a non-positive fraction."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    worst = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - 1.0)
    return worst


def hit_rate(outcomes: Sequence[bool]) -> float:
    """Fraction of ``True`` outcomes; 0.0 if empty."""
    if not outcomes:
        return 0.0
    return sum(1 for o in outcomes if o) / len(outcomes)


def calibration_error(predicted: Sequence[float], realized: Sequence[float]) -> float:
    """Mean absolute error between predicted and realized values."""
    if len(predicted) != len(realized):
        raise ValueError("predicted and realized must have equal length")
    if not predicted:
        return 0.0
    return sum(abs(p - r) for p, r in zip(predicted, realized, strict=True)) / len(predicted)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_trading_metrics.py -v && uv run mypy src/ophir && uv run ruff check src/ophir/trading tests/test_trading_metrics.py`
Expected: tests PASS, mypy clean, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/ophir/trading/metrics.py tests/test_trading_metrics.py
git commit -m "feat(trading): add performance metrics"
```

---

## Task 6: Signal blend

**Files:**
- Create: `src/ophir/trading/signals.py`
- Test: `tests/test_trading_signals.py`

**Interfaces:**
- Consumes: `SignalWeights` (Task 1).
- Produces:
  - `def normalize(value: float, lo: float, hi: float) -> float` — map `value` into `[-1, 1]` linearly over `[lo, hi]`, clamped; raises `ValueError` if `lo >= hi`.
  - `def blend_signals(ophir: float | None, momentum: float, sentiment: float, weights: SignalWeights) -> float` — weighted average of the available components (each already in `[-1, 1]`). When `ophir is None`, drop the ophir term and renormalize the remaining weights so the result stays in `[-1, 1]`. Raises `ValueError` if the active weights sum to 0.
  - `CORE_WEIGHTS: SignalWeights` (`ophir=0.6, momentum=0.25, sentiment=0.15`) and `TACTICAL_WEIGHTS: SignalWeights` (`ophir=0.2, momentum=0.5, sentiment=0.3`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trading_signals.py
import pytest

from ophir.trading.signals import (
    CORE_WEIGHTS,
    TACTICAL_WEIGHTS,
    blend_signals,
    normalize,
)
from ophir.trading.types import SignalWeights


def test_normalize_clamps() -> None:
    assert normalize(5.0, 0.0, 10.0) == pytest.approx(0.0)
    assert normalize(10.0, 0.0, 10.0) == pytest.approx(1.0)
    assert normalize(0.0, 0.0, 10.0) == pytest.approx(-1.0)
    assert normalize(-5.0, 0.0, 10.0) == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="lo"):
        normalize(1.0, 1.0, 1.0)


def test_blend_all_present() -> None:
    w = SignalWeights(ophir=0.5, momentum=0.3, sentiment=0.2)
    assert blend_signals(1.0, 1.0, 1.0, w) == pytest.approx(1.0)
    assert blend_signals(0.0, 0.0, 0.0, w) == pytest.approx(0.0)


def test_blend_ophir_absent_renormalizes() -> None:
    w = SignalWeights(ophir=0.6, momentum=0.25, sentiment=0.15)
    # momentum=1, sentiment=-1, weights renormalize over 0.25/0.15
    expected = (0.25 * 1.0 + 0.15 * -1.0) / (0.25 + 0.15)
    assert blend_signals(None, 1.0, -1.0, w) == pytest.approx(expected)


def test_preset_weights_exist() -> None:
    assert CORE_WEIGHTS.ophir == pytest.approx(0.6)
    assert TACTICAL_WEIGHTS.momentum == pytest.approx(0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_trading_signals.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ophir.trading.signals'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ophir/trading/signals.py
"""Normalize and blend the per-candidate signal components into one score.

Each component is expected in ``[-1, 1]``. The ophir forecast may be ``None``
(no CUDA/checkpoint, or the name is uncovered); the blend then degrades to the
remaining signals by renormalizing their weights.
"""

from ophir.trading.types import SignalWeights

CORE_WEIGHTS = SignalWeights(ophir=0.6, momentum=0.25, sentiment=0.15)
TACTICAL_WEIGHTS = SignalWeights(ophir=0.2, momentum=0.5, sentiment=0.3)


def normalize(value: float, lo: float, hi: float) -> float:
    """Linearly map ``value`` over ``[lo, hi]`` into ``[-1, 1]``, clamped."""
    if lo >= hi:
        raise ValueError("lo must be < hi")
    frac = (value - lo) / (hi - lo)
    scaled = 2.0 * frac - 1.0
    return max(-1.0, min(1.0, scaled))


def blend_signals(
    ophir: float | None, momentum: float, sentiment: float, weights: SignalWeights
) -> float:
    """Weighted blend of the available signal components, result in ``[-1, 1]``."""
    pairs: list[tuple[float, float]] = [
        (weights.momentum, momentum),
        (weights.sentiment, sentiment),
    ]
    if ophir is not None:
        pairs.append((weights.ophir, ophir))
    total_weight = sum(w for w, _ in pairs)
    if total_weight == 0:
        raise ValueError("active signal weights sum to zero")
    return sum(w * v for w, v in pairs) / total_weight
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_trading_signals.py -v && uv run mypy src/ophir && uv run ruff check src/ophir/trading tests/test_trading_signals.py`
Expected: tests PASS, mypy clean, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/ophir/trading/signals.py tests/test_trading_signals.py
git commit -m "feat(trading): add signal normalization and blending"
```

---

## Task 7: Entity-memory section editing

**Files:**
- Create: `src/ophir/trading/memory.py`
- Test: `tests/test_trading_memory.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `def upsert_section(markdown: str, heading: str, body: str) -> str` — given a markdown doc, replace the contents under the `## {heading}` section with `body` (preserving other sections and their order); if the heading does not exist, append a new `## {heading}` section at the end. Headings are matched on a line equal to `## {heading}`. Returns the new document text. Pure (no I/O).
  - `def read_memory(path: str | Path) -> str` — return file text, or `""` if absent.
  - `def write_memory(path: str | Path, text: str) -> None` — create parent dirs and write text.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trading_memory.py
from pathlib import Path

from ophir.trading.memory import read_memory, upsert_section, write_memory


def test_upsert_replaces_existing_section() -> None:
    doc = "# AAPL\n\n## Thesis\n\nold thesis\n\n## Notes\n\nkeep me\n"
    out = upsert_section(doc, "Thesis", "new thesis")
    assert "new thesis" in out
    assert "old thesis" not in out
    assert "keep me" in out  # other sections preserved


def test_upsert_appends_new_section() -> None:
    doc = "# AAPL\n\n## Thesis\n\nt\n"
    out = upsert_section(doc, "Risks", "earnings next week")
    assert "## Risks" in out
    assert "earnings next week" in out
    assert "## Thesis" in out


def test_upsert_on_empty_doc() -> None:
    out = upsert_section("", "Thesis", "first")
    assert "## Thesis" in out
    assert "first" in out


def test_read_missing_returns_empty(tmp_path: Path) -> None:
    assert read_memory(tmp_path / "nope.md") == ""


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "tickers" / "AAPL.md"
    write_memory(target, "# AAPL\n")
    assert read_memory(target) == "# AAPL\n"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_trading_memory.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ophir.trading.memory'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ophir/trading/memory.py
"""Edit entity-memory markdown files by section (upsert), plus thin file I/O."""

from pathlib import Path


def upsert_section(markdown: str, heading: str, body: str) -> str:
    """Replace the ``## {heading}`` section's body, or append the section.

    Sections are delimited by lines equal to ``## <name>``. The body of the
    target section is replaced with ``body``; all other sections keep their
    order and content. If the heading is absent, a new section is appended.
    """
    marker = f"## {heading}"
    lines = markdown.splitlines()
    section_block = ["", marker, "", body.rstrip("\n"), ""]

    start: int | None = None
    for i, line in enumerate(lines):
        if line.strip() == marker:
            start = i
            break

    if start is None:
        prefix = markdown.rstrip("\n")
        joined = "\n".join(section_block).strip("\n")
        return (prefix + "\n\n" + joined + "\n") if prefix else joined + "\n"

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break

    new_lines = lines[:start] + [marker, "", body.rstrip("\n"), ""] + lines[end:]
    return "\n".join(new_lines).rstrip("\n") + "\n"


def read_memory(path: str | Path) -> str:
    """Return the file's text, or ``""`` if it does not exist."""
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else ""


def write_memory(path: str | Path, text: str) -> None:
    """Write ``text`` to ``path``, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_trading_memory.py -v && uv run mypy src/ophir && uv run ruff check src/ophir/trading tests/test_trading_memory.py`
Expected: tests PASS, mypy clean, ruff clean.

> If `test_upsert_appends_new_section` or `test_upsert_replaces_existing_section` fails on exact spacing, adjust the assertions to the actual output (they check substring membership, so they should hold) — do not loosen the implementation's section-preservation behavior.

- [ ] **Step 5: Commit**

```bash
git add src/ophir/trading/memory.py tests/test_trading_memory.py
git commit -m "feat(trading): add entity-memory section upsert"
```

---

## Task 8: `ophir trade` CLI subapp

**Files:**
- Create: `src/ophir/trading/cli.py`
- Modify: `src/ophir/cli.py` (mount the subapp)
- Test: `tests/test_trading_cli.py`

**Interfaces:**
- Consumes: `load_config` (Task 2), `evaluate_order` (Task 3), all types (Task 1).
- Produces a Typer app `app` with one command used by the skill:
  - `gate --config <path> --order <json-file> --snapshot <json-file>` → prints a JSON object `{"action": ..., "approved_notional": ..., "reasons": [...]}` to stdout and exits non-zero when the action is `reject`. This is the skill's Bash-callable chokepoint.
- Mounted in `src/ophir/cli.py` as `app.add_typer(trading_cli.app, name="trade")` so the command is `ophir trade gate …`.

> Scope note: `gate` is the only command needed for the paper loop's pre-trade check from Bash. Ledger/metrics/memory are called in-process by Plan 2's Workflow-result handling (the agent runs short `python -c` / `python -m` snippets), so they do not each need a CLI verb in this task. Keeping the CLI surface to the one safety-critical command keeps the attack surface small.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trading_cli.py
import json
from pathlib import Path

from typer.testing import CliRunner

from ophir.trading.cli import app

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
ORDER = {
    "symbol": "AAPL",
    "side": "buy",
    "sleeve": "core",
    "asset_class": "equity",
    "notional": 1000.0,
    "sector": "Technology",
    "is_defined_risk": True,
    "is_short_option": False,
}
SNAPSHOT = {
    "equity": 100000.0,
    "cash": 50000.0,
    "day_pl": 0.0,
    "open_position_count": 0,
    "held_symbols": [],
    "symbol_exposure": {},
    "sector_exposure": {},
    "sleeve_exposure": {},
    "option_premium_at_risk": 0.0,
    "account_mode": "paper",
}


def _files(tmp_path: Path) -> tuple[Path, Path, Path]:
    c = tmp_path / "config.json"
    o = tmp_path / "order.json"
    s = tmp_path / "snap.json"
    c.write_text(json.dumps(CONFIG))
    o.write_text(json.dumps(ORDER))
    s.write_text(json.dumps(SNAPSHOT))
    return c, o, s


def test_gate_approves(tmp_path: Path) -> None:
    c, o, s = _files(tmp_path)
    result = runner.invoke(
        app, ["gate", "--config", str(c), "--order", str(o), "--snapshot", str(s)]
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["action"] == "approve"
    assert payload["approved_notional"] == 1000.0


def test_gate_rejects_nonzero_exit(tmp_path: Path) -> None:
    bad_snap = {**SNAPSHOT, "account_mode": "live"}
    c, o, s = _files(tmp_path)
    s.write_text(json.dumps(bad_snap))
    result = runner.invoke(
        app, ["gate", "--config", str(c), "--order", str(o), "--snapshot", str(s)]
    )
    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["action"] == "reject"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_trading_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ophir.trading.cli'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ophir/trading/cli.py
"""``ophir trade`` subcommands. Currently the Bash-callable pre-trade gate."""

import json
from pathlib import Path

import typer

from ophir.trading.config import load_config
from ophir.trading.safety import evaluate_order
from ophir.trading.types import (
    AssetClass,
    GateAction,
    ProposedOrder,
    Side,
    Sleeve,
)

app = typer.Typer(help="Deterministic trading-core commands.")


def _order_from(data: dict[str, object]) -> ProposedOrder:
    return ProposedOrder(
        symbol=str(data["symbol"]),
        side=Side(str(data["side"])),
        sleeve=Sleeve(str(data["sleeve"])),
        asset_class=AssetClass(str(data["asset_class"])),
        notional=float(data["notional"]),  # type: ignore[arg-type]
        sector=None if data["sector"] is None else str(data["sector"]),
        is_defined_risk=bool(data["is_defined_risk"]),
        is_short_option=bool(data["is_short_option"]),
    )


def _snapshot_from(data: dict[str, object]):  # type: ignore[no-untyped-def]
    from ophir.trading.types import AccountSnapshot

    sleeve_exposure = {Sleeve(k): float(v) for k, v in dict(data["sleeve_exposure"]).items()}  # type: ignore[arg-type]
    return AccountSnapshot(
        equity=float(data["equity"]),  # type: ignore[arg-type]
        cash=float(data["cash"]),  # type: ignore[arg-type]
        day_pl=float(data["day_pl"]),  # type: ignore[arg-type]
        open_position_count=int(data["open_position_count"]),  # type: ignore[arg-type]
        held_symbols=frozenset(str(s) for s in list(data["held_symbols"])),  # type: ignore[arg-type]
        symbol_exposure={str(k): float(v) for k, v in dict(data["symbol_exposure"]).items()},  # type: ignore[arg-type]
        sector_exposure={str(k): float(v) for k, v in dict(data["sector_exposure"]).items()},  # type: ignore[arg-type]
        sleeve_exposure=sleeve_exposure,
        option_premium_at_risk=float(data["option_premium_at_risk"]),  # type: ignore[arg-type]
        account_mode=str(data["account_mode"]),
    )


@app.command()
def gate(
    config: Path = typer.Option(..., help="Path to config.json"),
    order: Path = typer.Option(..., help="Path to the proposed-order JSON"),
    snapshot: Path = typer.Option(..., help="Path to the account-snapshot JSON"),
) -> None:
    """Run a proposed order through the safety gate; exit non-zero on reject."""
    cfg = load_config(config)
    decision = evaluate_order(
        _order_from(json.loads(order.read_text())),
        _snapshot_from(json.loads(snapshot.read_text())),
        cfg,
    )
    typer.echo(
        json.dumps(
            {
                "action": decision.action.value,
                "approved_notional": decision.approved_notional,
                "reasons": list(decision.reasons),
            }
        )
    )
    if decision.action is GateAction.REJECT:
        raise typer.Exit(code=1)
```

> **mypy note:** the `_snapshot_from` helper is annotated loosely (`# type: ignore[no-untyped-def]`) to keep the JSON-parsing boilerplate readable. If strict mypy rejects that ignore as unused, give the function the explicit `-> AccountSnapshot` return type and import `AccountSnapshot` at module top instead, then drop the ignore. Resolve until `uv run mypy src/ophir` is clean.

- [ ] **Step 4: Mount the subapp in the root CLI**

In `src/ophir/cli.py`, update the import line and registrations:

```python
from ophir import curation, evaluate, register, train
from ophir.trading import cli as trading_cli
```

and after the existing `app.add_typer(register.app, name="register")` line, add:

```python
app.add_typer(trading_cli.app, name="trade")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_trading_cli.py -v && uv run mypy src/ophir && uv run ruff check src/ophir && uv run pytest -q`
Expected: new CLI tests PASS, mypy clean, ruff clean, **full suite green**.

Also smoke-test the mounted command:

Run: `uv run ophir trade --help`
Expected: shows the `gate` command under `ophir trade`.

- [ ] **Step 6: Commit**

```bash
git add src/ophir/trading/cli.py src/ophir/cli.py tests/test_trading_cli.py
git commit -m "feat(trading): expose ophir trade gate CLI"
```

---

## Final verification (run after Task 8)

- [ ] `uv run ruff check .` — clean
- [ ] `uv run ruff format --check .` — clean (run `uv run ruff format .` first if needed, then re-commit)
- [ ] `uv run mypy src/ophir` — clean
- [ ] `uv run pytest` — full suite green
- [ ] Update `CLAUDE.md` Tests section + module map to mention the new `trading` subpackage and its tests (one line each), and add a CHANGELOG entry matching the existing format. Commit as `docs(trading): document trading core`.

---

## Self-Review (completed during planning)

**Spec coverage vs. design §1–§11:**
- Safety layer §6 → Task 3 (`evaluate_order`) + Task 8 (CLI gate) + Task 2 (limits in config). Account-mode interlock → Task 3 check #1. ✓
- Decision ledger §4/§5 → Task 4. ✓
- Performance metrics (return vs SPY, Sharpe, drawdown, hit-rate, calibration) §5/§9 → Task 5. ✓
- Signal blend with graceful ophir-absent §8 → Task 6. ✓
- Entity memory upsert §2/§5 → Task 7. ✓
- Config knobs (account_mode, sleeve %, limits, shortlist_size, verify_votes, depth) §7 → Tasks 1–2. ✓
- **Deferred to Plan 2 (orchestration):** `SKILL.md`, `config.json` content + the actual numeric defaults file, `morning.js`/`evening.js`, Alpaca-MCP exposure-join, ophir inference adapter, memories/ seed, `/schedule` wrapping. These are integration concerns, not core logic.

**Placeholder scan:** no TBD/TODO; every code step has complete code. ✓

**Type consistency:** `evaluate_order`, `AccountSnapshot` field names, `GateDecision` shape, `DecisionRecord` fields, `SignalWeights` fields are used identically across Tasks 1/3/4/6/8. CLI JSON keys match `ProposedOrder`/`AccountSnapshot` field names exactly. ✓

**Note on `# type: ignore` markers:** several JSON-deserialization sites carry precise `# type: ignore[arg-type]`. Because the repo sets `warn_unused_ignores = true`, each must be verified against `uv run mypy src/ophir` and removed if unused or relocated if a different line errors — called out inline in Tasks 4 and 8.
