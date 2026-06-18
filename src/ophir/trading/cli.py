"""``ophir trade`` subcommands. Currently the Bash-callable pre-trade gate."""

import json
from pathlib import Path

import typer

from ophir.trading.config import load_config
from ophir.trading.safety import evaluate_order
from ophir.trading.types import (
    AccountSnapshot,
    AssetClass,
    GateAction,
    ProposedOrder,
    Side,
    Sleeve,
)

app = typer.Typer(help="Deterministic trading-core commands.")


@app.callback()
def _callback() -> None:
    """Deterministic trading-core commands."""


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


def _snapshot_from(data: dict[str, object]) -> AccountSnapshot:
    sleeve_exposure = {
        Sleeve(k): float(v)
        for k, v in dict(data["sleeve_exposure"]).items()  # type: ignore[call-overload]
    }
    return AccountSnapshot(
        equity=float(data["equity"]),  # type: ignore[arg-type]
        cash=float(data["cash"]),  # type: ignore[arg-type]
        day_pl=float(data["day_pl"]),  # type: ignore[arg-type]
        open_position_count=int(data["open_position_count"]),  # type: ignore[call-overload]
        held_symbols=frozenset(str(s) for s in list(data["held_symbols"])),  # type: ignore[call-overload]
        symbol_exposure={
            str(k): float(v)
            for k, v in dict(data["symbol_exposure"]).items()  # type: ignore[call-overload]
        },
        sector_exposure={
            str(k): float(v)
            for k, v in dict(data["sector_exposure"]).items()  # type: ignore[call-overload]
        },
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
