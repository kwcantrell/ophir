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
