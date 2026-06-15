"""Tests for the decision layer (quant pure; Ollama LLM mocked, no server)."""

import types

from ophir.agent import decide as decide_mod
from ophir.agent.config import AgentSettings
from ophir.agent.decide import (
    Decision,
    DecisionComparison,
    compare_decisions,
    ollama_decision,
    quant_decision,
)
from ophir.agent.predict import Forecast


def _forecast(symbol="AAPL", cum_return=0.0, n=90):
    return Forecast(
        symbol=symbol,
        asof="2026-06-03",
        horizon=n,
        r_close=[cum_return / n] * n,
        upside=[0.01] * n,
        downside=[0.01] * n,
        cum_return=cum_return,
        score=cum_return,
    )


def _settings():
    return AgentSettings(buy_threshold=0.02, sell_threshold=-0.02, decision_downside_penalty=0.0)


class _FakeLLM:
    """Minimal stand-in for ChatOllama: ``.invoke`` returns canned ``.content``."""

    def __init__(self, content):
        self._content = content

    def invoke(self, _messages):
        return types.SimpleNamespace(content=self._content)


def test_quant_decision_buy_sell_hold(monkeypatch):
    monkeypatch.setattr(decide_mod.audit, "log_event", lambda *a, **k: None)
    s = _settings()
    assert quant_decision(_forecast(cum_return=0.05), settings=s).action == "BUY"
    assert quant_decision(_forecast(cum_return=-0.05), settings=s).action == "SELL"
    assert quant_decision(_forecast(cum_return=0.0), settings=s).action == "HOLD"


def test_quant_decision_fields_and_confidence(monkeypatch):
    monkeypatch.setattr(decide_mod.audit, "log_event", lambda *a, **k: None)
    d = quant_decision(_forecast(symbol="MSFT", cum_return=0.02), settings=_settings())
    assert isinstance(d, Decision)
    assert d.symbol == "MSFT"
    assert d.source == "quant"
    assert d.action == "BUY"  # exactly at the threshold counts as BUY (>=)
    assert d.confidence == 1.0
    assert d.asof == "2026-06-03"


def test_quant_decision_non_finite_holds(monkeypatch):
    monkeypatch.setattr(decide_mod.audit, "log_event", lambda *a, **k: None)
    d = quant_decision(_forecast(cum_return=float("nan")), settings=_settings())
    assert d.action == "HOLD"
    assert d.confidence == 0.0


def test_quant_decision_downside_penalty_flips_to_hold(monkeypatch):
    monkeypatch.setattr(decide_mod.audit, "log_event", lambda *a, **k: None)
    s = AgentSettings(buy_threshold=0.02, sell_threshold=-0.02, decision_downside_penalty=4.0)
    # cum_return 0.05 alone clears BUY; the downside penalty (4 * 0.01) pulls the
    # score to 0.01, back inside the HOLD band.
    d = quant_decision(_forecast(cum_return=0.05), settings=s)
    assert d.action == "HOLD"


def test_ollama_decision_parses_json(monkeypatch):
    monkeypatch.setattr(decide_mod.audit, "log_event", lambda *a, **k: None)
    llm = _FakeLLM('Sure: {"action": "buy", "confidence": 0.8, "rationale": "strong uptrend"}')
    d = ollama_decision(_forecast(symbol="NVDA", cum_return=0.1), llm=llm, settings=_settings())
    assert d.source == "ollama"
    assert d.action == "BUY"  # lowercase "buy" is normalised
    assert d.confidence == 0.8
    assert "uptrend" in d.rationale


def test_ollama_decision_failsafe_on_garbage(monkeypatch):
    monkeypatch.setattr(decide_mod.audit, "log_event", lambda *a, **k: None)
    d = ollama_decision(
        _forecast(cum_return=0.1), llm=_FakeLLM("I'm not sure, sorry."), settings=_settings()
    )
    assert d.action == "HOLD"
    assert d.confidence == 0.0


def test_ollama_decision_failsafe_on_unknown_action(monkeypatch):
    monkeypatch.setattr(decide_mod.audit, "log_event", lambda *a, **k: None)
    d = ollama_decision(
        _forecast(cum_return=0.1),
        llm=_FakeLLM('{"action": "YOLO", "confidence": 0.9, "rationale": "x"}'),
        settings=_settings(),
    )
    assert d.action == "HOLD"


def test_compare_decisions_agreement(monkeypatch):
    monkeypatch.setattr(decide_mod.audit, "log_event", lambda *a, **k: None)
    llm = _FakeLLM('{"action": "BUY", "confidence": 0.7, "rationale": "ok"}')
    forecasts = [
        _forecast(symbol="AAPL", cum_return=0.05),
        _forecast(symbol="T", cum_return=-0.05),
    ]
    comps = compare_decisions(forecasts, llm=llm, settings=_settings())
    assert len(comps) == 2
    assert all(isinstance(c, DecisionComparison) for c in comps)
    # AAPL: quant BUY vs ollama BUY -> agree
    assert comps[0].symbol == "AAPL"
    assert comps[0].quant.action == "BUY"
    assert comps[0].agree is True
    # T: quant SELL vs ollama BUY -> differ
    assert comps[1].quant.action == "SELL"
    assert comps[1].ollama.action == "BUY"
    assert comps[1].agree is False
