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
