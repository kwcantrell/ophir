"""Tests for the paper-execution layer (in-process broker + mocked LLM; no network)."""

import types
from decimal import Decimal

from ophir.agent import execute as execute_mod
from ophir.agent.execute import Account, Order, PaperBroker, daily_report, place_orders, reconcile
from ophir.agent.manage import Portfolio, Position


class _FakeLLM:
    def __init__(self, content):
        self._content = content

    def invoke(self, _messages):
        return types.SimpleNamespace(content=self._content)


class _BadLLM:
    def invoke(self, _messages):
        raise RuntimeError("boom")


def _portfolio(weighted, asof="2026-06-13"):
    gross = sum(w for _, w in weighted)
    return Portfolio(
        asof=asof,
        positions=[
            Position(symbol=s, weight=w, conviction=0.5, rationale="r") for s, w in weighted
        ],
        cash_weight=1 - gross,
        gross_exposure=gross,
        rationale="manager rationale",
    )


def test_reconcile_sells_before_buys():
    account = Account(equity=Decimal("100000"), cash=Decimal("100000"))
    positions = {"OLD": Decimal("5000")}  # held, not in target -> sell
    pf = _portfolio([("AAPL", 0.04)])  # target 4000 -> buy
    orders = reconcile(pf, account, positions)
    assert [o.side for o in orders] == ["sell", "buy"]
    assert orders[0].symbol == "OLD"
    assert orders[0].notional == Decimal("5000")
    assert orders[1].symbol == "AAPL"
    assert orders[1].notional == Decimal("4000")
    assert orders[1].client_order_id == "ophir:2026-06-13:AAPL:buy"


def test_reconcile_no_trade_band():
    account = Account(equity=Decimal("100000"), cash=Decimal("100000"))
    positions = {"AAPL": Decimal("4000")}
    pf = _portfolio([("AAPL", 0.0402)])  # target 4020; delta 20 < band (0.5% * 100000 = 500)
    assert reconcile(pf, account, positions) == []


def test_place_orders_dry_run_then_execute(monkeypatch):
    monkeypatch.setattr(execute_mod.audit, "log_event", lambda *a, **k: None)
    broker = PaperBroker(equity=Decimal("100000"))
    orders = [Order("AAPL", "buy", Decimal("4000"), "ophir:x:AAPL:buy")]

    plan = place_orders(orders, broker, dry_run=True)
    assert plan[0].startswith("DRY-RUN")
    assert broker.get_positions() == {}  # nothing submitted in dry-run

    plan2 = place_orders(orders, broker, dry_run=False)
    assert plan2[0].startswith("SUBMITTED")
    assert broker.get_positions()["AAPL"] == Decimal("4000")


def test_paper_broker_sell_clears_position(monkeypatch):
    monkeypatch.setattr(execute_mod.audit, "log_event", lambda *a, **k: None)
    broker = PaperBroker(equity=Decimal("100000"), positions={"AAPL": Decimal("4000")})
    broker.submit_order(Order("AAPL", "sell", Decimal("4000"), "x"))
    assert "AAPL" not in broker.get_positions()


def test_daily_report_with_llm(monkeypatch):
    monkeypatch.setattr(execute_mod.audit, "log_event", lambda *a, **k: None)
    text = daily_report(_portfolio([("AAPL", 0.04)]), [], llm=_FakeLLM("Holding AAPL at 4%."))
    assert "Holding AAPL" in text


def test_daily_report_failsafe_template(monkeypatch):
    monkeypatch.setattr(execute_mod.audit, "log_event", lambda *a, **k: None)
    text = daily_report(_portfolio([("AAPL", 0.04)]), [], llm=_BadLLM())
    assert "2026-06-13" in text
