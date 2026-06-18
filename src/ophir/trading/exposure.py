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
