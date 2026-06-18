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
    assert snap.symbol_exposure["XOM"] == 3_000.0
    assert snap.symbol_exposure["AAPL260116C"] == 1_000.0
    assert snap.sector_exposure["Technology"] == 6_000.0
    assert snap.sleeve_exposure[Sleeve.CORE] == 5_000.0  # only ledger-known symbol
    # ledger-unknown symbols absent from sleeve_exposure
    assert set(snap.sleeve_exposure) == {Sleeve.CORE}
    assert Sleeve.TACTICAL not in snap.sleeve_exposure
    assert snap.option_premium_at_risk == 1_000.0
    assert snap.account_mode == "paper"


def test_sector_none_skipped() -> None:
    positions = [
        PositionInput("CASHX", 2_000.0, AssetClass.EQUITY, None),
        PositionInput("AAPL", 5_000.0, AssetClass.EQUITY, "Technology"),
    ]
    snap = build_snapshot(
        equity=100_000.0,
        cash=50_000.0,
        day_pl=0.0,
        account_mode="live",
        positions=positions,
        ledger_records=[],
    )
    assert snap.symbol_exposure["CASHX"] == 2_000.0
    assert None not in snap.sector_exposure
    assert "CASHX" in snap.held_symbols


def test_negative_market_value_short_positions() -> None:
    positions = [
        PositionInput("SHORTY", -4_000.0, AssetClass.EQUITY, "Energy"),
        PositionInput("LONG", 3_000.0, AssetClass.EQUITY, "Technology"),
    ]
    snap = build_snapshot(
        equity=100_000.0,
        cash=50_000.0,
        day_pl=0.0,
        account_mode="live",
        positions=positions,
        ledger_records=[],
    )
    # negative market_value converted to positive exposure
    assert snap.symbol_exposure["SHORTY"] == 4_000.0
    assert snap.symbol_exposure["LONG"] == 3_000.0
    assert snap.sector_exposure["Energy"] == 4_000.0
    assert snap.sector_exposure["Technology"] == 3_000.0
