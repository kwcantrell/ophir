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


def test_roundtrip_none_optionals() -> None:
    sparse = DecisionRecord(
        date="2026-01-01",
        symbol="SPY",
        sleeve=Sleeve.TACTICAL,
        side=Side.SELL,
        asset_class=AssetClass.OPTION,
        notional=500.0,
        sector=None,
        thesis="hedge",
        signals={},
        entry_price=None,
        target=None,
        stop=None,
        order_id=None,
        status="closed",
        realized_pl=None,
        scored=True,
    )
    assert record_from_dict(record_to_dict(sparse)) == sparse
