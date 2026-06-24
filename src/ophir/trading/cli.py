"""``ophir trade`` subcommands. Currently the Bash-callable pre-trade gate."""

import json
from pathlib import Path

import typer

from ophir.trading import forecast, metrics
from ophir.trading.config import load_config
from ophir.trading.ledger import (
    append_decision,
    ledger_path,
    load_decisions,
    record_from_dict,
    record_to_dict,
)
from ophir.trading.safety import evaluate_order
from ophir.trading.signals import (
    CORE_WEIGHTS,
    TACTICAL_WEIGHTS,
    blend_signals,
    ophir_signals,
)
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


def _parse_symbols(value: str) -> list[str]:
    """Parse ``--symbols`` as a path to a file of symbols, else a comma list."""
    candidate = Path(value)
    text = candidate.read_text() if candidate.is_file() else value
    parsed = [s.strip() for s in text.replace("\n", ",").split(",") if s.strip()]
    return list(dict.fromkeys(parsed))  # de-duplicate, preserving first-seen order


def _order_to_dict(order: ProposedOrder) -> dict[str, object]:
    """Serialize a proposed order to the dict shape ``gate`` consumes."""
    return {
        "symbol": order.symbol,
        "side": order.side.value,
        "sleeve": order.sleeve.value,
        "asset_class": order.asset_class.value,
        "notional": order.notional,
        "sector": order.sector,
        "is_defined_risk": order.is_defined_risk,
        "is_short_option": order.is_short_option,
    }


@app.command()
def propose(
    symbols: str = typer.Option(
        ..., help="Comma-separated symbols, or a path to a file of symbols"
    ),
    model_dir: Path = typer.Option(..., help="Directory holding the base checkpoint"),
    base_notional: float = typer.Option(
        ..., help="Sizing base in dollars; notional = base_notional * |blended|"
    ),
    config: Path = typer.Option(..., help="Path to config.json"),
    sleeve: Sleeve = typer.Option(Sleeve.CORE, help="Strategy sleeve for the orders"),
    min_abs_signal: float = typer.Option(
        0.0, help="Skip orders whose |blended signal| is at or below this"
    ),
) -> None:
    """Emit ProposedOrder JSON from ophir forecasts (no gate, no ledger).

    Runs the seam end to end: load forecasts, cross-sectionally normalize them,
    blend with neutral momentum/sentiment, and size by ``base_notional``. Prints
    a JSON array of proposed orders for piping into ``gate``. Degrades to an
    empty array when forecasts are unavailable; never invokes the safety gate or
    writes the ledger.
    """
    load_config(config)  # validate account_mode/limits; not used for sizing here
    names = _parse_symbols(symbols)
    forecasts = forecast.load_forecasts(names, model_dir)
    scores = ophir_signals(forecasts)
    weights = CORE_WEIGHTS if sleeve is Sleeve.CORE else TACTICAL_WEIGHTS
    orders: list[dict[str, object]] = []
    for symbol in names:
        blended = blend_signals(
            ophir=scores.get(symbol),
            momentum=0.0,
            sentiment=0.0,
            weights=weights,
        )
        if abs(blended) <= min_abs_signal:
            continue
        order = ProposedOrder(
            symbol=symbol,
            side=Side.BUY if blended > 0 else Side.SELL,
            sleeve=sleeve,
            asset_class=AssetClass.EQUITY,
            notional=base_notional * abs(blended),
            sector=None,
            is_defined_risk=True,
            is_short_option=False,
        )
        orders.append(_order_to_dict(order))
    typer.echo(json.dumps(orders))


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
    try:
        order_obj = _order_from(json.loads(order.read_text()))
        snap = _snapshot_from(json.loads(snapshot.read_text()))
    except (KeyError, ValueError, TypeError) as exc:
        typer.echo(
            json.dumps(
                {
                    "action": "reject",
                    "approved_notional": 0.0,
                    "reasons": [f"malformed input: {exc}"],
                }
            )
        )
        raise typer.Exit(code=1) from exc
    decision = evaluate_order(order_obj, snap, cfg)
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


@app.command()
def record(
    ledger_dir: Path = typer.Option(..., help="Ledger directory"),
    month: str = typer.Option(..., help="YYYY-MM"),
    decision: Path = typer.Option(..., help="Decision JSON file"),
) -> None:
    """Append one decision record to the ledger."""
    rec = record_from_dict(json.loads(decision.read_text()))
    append_decision(ledger_dir, month, rec)
    typer.echo(json.dumps({"appended": True}))


@app.command()
def close(
    ledger_dir: Path = typer.Option(..., help="Ledger directory"),
    month: str = typer.Option(..., help="YYYY-MM"),
    order_id: str = typer.Option(..., help="order_id to update"),
    status: str = typer.Option(..., help="new status"),
    realized_pl: float = typer.Option(..., help="realized P&L"),
) -> None:
    """Mark a decision closed/scored with its realized P&L."""
    records = load_decisions(ledger_dir, month)
    updated = 0
    new_lines: list[str] = []
    for rec in records:
        data = record_to_dict(rec)
        if rec.order_id == order_id:
            data["status"] = status
            data["realized_pl"] = realized_pl
            data["scored"] = True
            updated += 1
        new_lines.append(json.dumps(data))
    ledger_path(ledger_dir, month).write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
    typer.echo(json.dumps({"updated": updated}))


@app.command()
def performance(
    equity_curve: Path = typer.Option(..., help="JSON array of equity values"),
    out: Path = typer.Option(..., help="Output performance.md path"),
) -> None:
    """Compute portfolio metrics and write a performance.md snapshot."""
    curve = [float(x) for x in json.loads(equity_curve.read_text())]
    payload = {
        "total_return": metrics.total_return(curve),
        "sharpe": metrics.sharpe(metrics.daily_returns(curve)),
        "max_drawdown": metrics.max_drawdown(curve),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "# Performance\n\n"
        f"- Total return: {payload['total_return']:.4f}\n"
        f"- Sharpe: {payload['sharpe']:.4f}\n"
        f"- Max drawdown: {payload['max_drawdown']:.4f}\n"
    )
    typer.echo(json.dumps(payload))
