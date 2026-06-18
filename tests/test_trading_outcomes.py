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
