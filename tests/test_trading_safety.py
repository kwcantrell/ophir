"""Tests for the pre-trade safety gate."""

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
    base: dict[str, object] = {
        "equity": 100_000.0,
        "cash": 50_000.0,
        "day_pl": 0.0,
        "open_position_count": 0,
        "held_symbols": frozenset(),
        "symbol_exposure": {},
        "sector_exposure": {},
        "sleeve_exposure": {},
        "option_premium_at_risk": 0.0,
        "account_mode": "paper",
    }
    base.update(overrides)
    return AccountSnapshot(**base)  # type: ignore[arg-type]


def _order(**overrides: object) -> ProposedOrder:
    base: dict[str, object] = {
        "symbol": "AAPL",
        "side": Side.BUY,
        "sleeve": Sleeve.CORE,
        "asset_class": AssetClass.EQUITY,
        "notional": 1_000.0,
        "sector": "Technology",
        "is_defined_risk": True,
        "is_short_option": False,
    }
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
    decision = evaluate_order(_order(side=Side.SELL, notional=999_999.0), _snapshot(), CONFIG)
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
