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
