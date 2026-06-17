"""Paper execution: reconcile a target portfolio into orders and (optionally) place them.

Turns the Phase 6 :class:`~ophir.agent.manage.Portfolio` into broker orders and, by
default, only *plans* them (dry run) -- nothing is submitted unless the caller
explicitly opts in. A simulated in-process :class:`PaperBroker` backs tests and dry
runs; :class:`AlpacaPaperBroker` targets a real Alpaca **paper** account.

Per the ``trading-best-practices`` skill: paper-first, dry-run default, ``Decimal``
money, deterministic ``client_order_id`` idempotency, sells before buys, a no-trade
band to cut churn, and reconciling against broker truth. Everything is audit-logged.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import TYPE_CHECKING, Any, Literal, Protocol

from ophir.agent import audit
from ophir.agent.config import get_settings

if TYPE_CHECKING:
    from ophir.agent.config import AgentSettings
    from ophir.agent.manage import Portfolio

OrderSide = Literal["buy", "sell"]
_CENTS = Decimal("0.01")


def _money(amount: Decimal) -> Decimal:
    """Quantize a dollar amount to cents (Alpaca rejects notionals past 2 dp).

    Rounds down so a buy never overshoots available buying power.
    """
    return amount.quantize(_CENTS, rounding=ROUND_DOWN)


@dataclass(frozen=True, slots=True)
class Order:
    """A market order in dollar (notional) terms with an idempotency key.

    ``close`` marks a full-position exit: it is filled by liquidating the entire
    held quantity (the broker's native close), not by a notional order. A notional
    sell of the whole position converts back to a hair more shares than are held
    (price rounding) and Alpaca rejects it with "insufficient qty"; closing by
    quantity avoids that. ``notional`` is still the position's market value, used
    for reporting and the no-trade band.
    """

    symbol: str
    side: OrderSide
    notional: Decimal
    client_order_id: str
    close: bool = False


@dataclass(frozen=True, slots=True)
class Account:
    """A broker account snapshot used for sizing and the kill-switch."""

    equity: Decimal
    cash: Decimal
    drawdown: float = 0.0
    daily_loss: float = 0.0


class Broker(Protocol):
    """The minimal broker surface the execution layer needs."""

    def get_account(self) -> Account: ...
    def get_positions(self) -> dict[str, Decimal]: ...
    def submit_order(self, order: Order) -> None: ...


class PaperBroker:
    """In-memory simulated paper account (deterministic, no network).

    Positions are tracked by market value in dollars; equity is held constant (no
    price simulation), so drawdown / daily-loss are ``0`` -- the kill-switch never
    fires for a fresh sim. Ephemeral: state lives only for the process. Use it for
    dry runs and tests; :class:`AlpacaPaperBroker` is the persistent paper account.
    """

    def __init__(
        self,
        equity: Decimal | float = Decimal("100000"),
        positions: dict[str, Decimal] | None = None,
    ) -> None:
        self._equity = Decimal(str(equity))
        self._positions: dict[str, Decimal] = dict(positions or {})
        self._cash = self._equity - sum(self._positions.values(), Decimal(0))

    def get_account(self) -> Account:
        return Account(equity=self._equity, cash=self._cash, drawdown=0.0, daily_loss=0.0)

    def get_positions(self) -> dict[str, Decimal]:
        return dict(self._positions)

    def submit_order(self, order: Order) -> None:
        if order.side == "buy":
            self._positions[order.symbol] = (
                self._positions.get(order.symbol, Decimal(0)) + order.notional
            )
            self._cash -= order.notional
        else:
            held = self._positions.get(order.symbol, Decimal(0))
            sold = held if order.close else min(held, order.notional)
            self._positions[order.symbol] = held - sold
            if self._positions[order.symbol] <= 0:
                self._positions.pop(order.symbol, None)
            self._cash += sold
        audit.log_event(
            "paper_fill", symbol=order.symbol, side=order.side, notional=float(order.notional)
        )


class AlpacaPaperBroker:
    """Adapter over a real Alpaca **paper** account (alpaca-py).

    The ``alpaca`` SDK is imported dynamically so this module never depends on its
    exact API at type-check time; the broker is constructed only when explicitly
    selected and credentials are present.
    """

    def __init__(self, settings: AgentSettings | None = None) -> None:
        settings = settings or get_settings()
        if settings.alpaca_key_id is None or settings.alpaca_secret_key is None:
            raise ValueError(
                "Alpaca paper trading requires AGENT_ALPACA_KEY_ID and AGENT_ALPACA_SECRET_KEY"
            )
        client_mod = importlib.import_module("alpaca.trading.client")
        self._client: Any = client_mod.TradingClient(
            settings.alpaca_key_id.get_secret_value(),
            settings.alpaca_secret_key.get_secret_value(),
            paper=True,
        )

    def get_account(self) -> Account:
        acct = self._client.get_account()
        equity = Decimal(str(acct.equity))
        last = Decimal(str(acct.last_equity)) if getattr(acct, "last_equity", None) else equity
        daily_loss = float((last - equity) / last) if last > 0 else 0.0
        return Account(
            equity=equity,
            cash=Decimal(str(acct.cash)),
            drawdown=max(0.0, daily_loss),
            daily_loss=max(0.0, daily_loss),
        )

    def get_positions(self) -> dict[str, Decimal]:
        return {
            pos.symbol: Decimal(str(pos.market_value)) for pos in self._client.get_all_positions()
        }

    def submit_order(self, order: Order) -> None:
        if order.close:
            # Full exit: liquidate the exact held quantity (a notional sell of the
            # whole position rounds to slightly more shares than held and is rejected).
            self._client.close_position(order.symbol)
            audit.log_event(
                "order_submitted",
                symbol=order.symbol,
                side="sell",
                notional=float(order.notional),
                close=True,
            )
            return
        requests_mod = importlib.import_module("alpaca.trading.requests")
        enums_mod = importlib.import_module("alpaca.trading.enums")
        side = enums_mod.OrderSide.BUY if order.side == "buy" else enums_mod.OrderSide.SELL
        request = requests_mod.MarketOrderRequest(
            symbol=order.symbol,
            notional=float(order.notional),
            side=side,
            time_in_force=enums_mod.TimeInForce.DAY,
            client_order_id=order.client_order_id,
        )
        self._client.submit_order(request)
        audit.log_event(
            "order_submitted", symbol=order.symbol, side=order.side, notional=float(order.notional)
        )


def _client_order_id(asof: str, symbol: str, side: str) -> str:
    """Deterministic idempotency key: re-running the same day re-derives it."""
    return f"ophir:{asof}:{symbol}:{side}"


def reconcile(
    portfolio: Portfolio,
    account: Account,
    positions: dict[str, Decimal],
    *,
    no_trade_frac: float = 0.005,
) -> list[Order]:
    """Delta the current positions to the portfolio's target weights into orders.

    Targets are ``weight * equity`` in dollars. Names held below target (or no
    longer wanted) are **sold first**; names below target are then bought. Deltas
    smaller than ``no_trade_frac * equity`` are skipped (a no-trade band). All money
    math is ``Decimal``; each order carries a deterministic ``client_order_id``.
    """
    equity = account.equity
    band = Decimal(str(no_trade_frac)) * equity
    targets = {p.symbol: Decimal(str(p.weight)) * equity for p in portfolio.positions}

    sells: list[Order] = []
    for symbol, current in positions.items():
        target = targets.get(symbol, Decimal(0))
        delta = target - current
        if delta < 0 and -delta >= band:
            sells.append(
                Order(
                    symbol,
                    "sell",
                    _money(min(current, -delta)),
                    _client_order_id(portfolio.asof, symbol, "sell"),
                    close=target == 0,  # exiting the name entirely -> liquidate by quantity
                )
            )

    buys: list[Order] = []
    for symbol, target in targets.items():
        delta = target - positions.get(symbol, Decimal(0))
        if delta > 0 and delta >= band:
            buys.append(
                Order(symbol, "buy", _money(delta), _client_order_id(portfolio.asof, symbol, "buy"))
            )

    return sells + buys


def place_orders(orders: list[Order], broker: Broker, *, dry_run: bool = True) -> list[str]:
    """Plan (dry run, default) or submit orders; returns one summary line each.

    With ``dry_run`` (the default) nothing is submitted -- each order is logged as
    planned and returned as a ``DRY-RUN`` line. Otherwise each is submitted via the
    broker (idempotent through its ``client_order_id``).
    """
    plan: list[str] = []
    for order in orders:
        line = f"{order.side.upper():<4} {order.symbol:<6} ${order.notional:,.2f}"
        if dry_run:
            audit.log_event(
                "order_planned",
                symbol=order.symbol,
                side=order.side,
                notional=float(order.notional),
                client_order_id=order.client_order_id,
            )
            plan.append(f"DRY-RUN  {line}")
        else:
            try:
                broker.submit_order(order)
                plan.append(f"SUBMITTED {line}")
            except Exception as exc:
                # Fail safe: a single rejected order must not abort the rebalance
                # (e.g. an already-filled idempotent order, or a liquidity reject).
                audit.log_event(
                    "order_failed",
                    symbol=order.symbol,
                    side=order.side,
                    notional=float(order.notional),
                    error=str(exc),
                )
                plan.append(f"FAILED   {line} -- {type(exc).__name__}")
    return plan


def _template_report(portfolio: Portfolio, orders: list[Order], *, note: str = "") -> str:
    """A plain, no-LLM end-of-day summary (the fail-safe report)."""
    parts = [
        f"As of {portfolio.asof}: {len(portfolio.positions)} target position(s), "
        f"gross {portfolio.gross_exposure:.0%}, cash {portfolio.cash_weight:.0%}.",
    ]
    if portfolio.halted:
        parts.append("Risk gate HALTED -- holding all cash.")
    parts.append(f"{len(orders)} order(s) planned.")
    if note:
        parts.append(note)
    return " ".join(parts)


def daily_report(
    portfolio: Portfolio,
    orders: list[Order],
    *,
    llm: Any = None,
    settings: AgentSettings | None = None,
) -> str:
    """Write a short end-of-day report from the portfolio + planned orders.

    A reporting-only LLM use (no decisions). Fails safe to a templated summary when
    the model is unavailable.
    """
    settings = settings or get_settings()
    facts = {
        "asof": portfolio.asof,
        "halted": portfolio.halted,
        "gross_exposure": round(portfolio.gross_exposure, 4),
        "cash_weight": round(portfolio.cash_weight, 4),
        "positions": [
            {"symbol": p.symbol, "weight": round(p.weight, 4), "rationale": p.rationale}
            for p in portfolio.positions
        ],
        "manager_rationale": portfolio.rationale,
        "gate_notes": portfolio.gate_notes,
        "orders": [
            {"side": o.side, "symbol": o.symbol, "notional": float(o.notional)} for o in orders
        ],
    }
    if llm is None:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=settings.ollama_model,
            temperature=0,
            base_url=settings.ollama_base_url,
            num_ctx=settings.ollama_num_ctx,
        )
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        system = SystemMessage(
            content=(
                "You are a trading-operations assistant. Write a brief (<=120 words) "
                "end-of-day report from ONLY the data provided -- the target positions, "
                "planned orders, manager rationale, and any risk-gate notes. Do not invent "
                "any new figures."
            )
        )
        human = HumanMessage(content=json.dumps(facts, indent=2, default=str))
        text = llm.invoke([system, human]).content
        report = text if isinstance(text, str) else str(text)
    except Exception as exc:  # fail safe: a plain templated report
        report = _template_report(
            portfolio, orders, note=f"(LLM report unavailable: {type(exc).__name__})"
        )
    audit.log_event(
        "daily_report", asof=portfolio.asof, n_orders=len(orders), halted=portfolio.halted
    )
    return report
