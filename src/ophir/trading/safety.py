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

    sleeve_cap_pct = limits.max_core_pct if order.sleeve is Sleeve.CORE else limits.max_tactical_pct
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
