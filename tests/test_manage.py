"""Tests for the manager pick + risk gate (LLM mocked, no network/GPU)."""

import types

from ophir.agent import manage as manage_mod
from ophir.agent.config import AgentSettings
from ophir.agent.debate import Debate, Thesis
from ophir.agent.decide import Decision, DecisionComparison
from ophir.agent.manage import Candidate, Position, apply_risk_gate, manage
from ophir.agent.predict import Forecast
from ophir.agent.research import ResearchAnalysis, ResearchBrief


class _FakeLLM:
    def __init__(self, content):
        self._content = content

    def invoke(self, _messages):
        return types.SimpleNamespace(content=self._content)


def _forecast(symbol, cum=0.05, n=90):
    return Forecast(
        symbol=symbol,
        asof="2026-06-13",
        horizon=n,
        r_close=[cum / n] * n,
        upside=[0.012] * n,
        downside=[0.009] * n,
        cum_return=cum,
        score=cum,
    )


def _decision(symbol):
    quant = Decision(
        symbol=symbol,
        action="BUY",
        confidence=0.8,
        rationale="q",
        source="quant",
        cum_return=0.05,
        asof="2026-06-13",
    )
    ollama = Decision(
        symbol=symbol,
        action="BUY",
        confidence=0.7,
        rationale="o",
        source="ollama",
        cum_return=0.05,
        asof="2026-06-13",
    )
    return DecisionComparison(symbol=symbol, quant=quant, ollama=ollama, agree=True)


def _candidate(symbol, vol_60d=0.015):
    return Candidate(
        symbol=symbol,
        forecast=_forecast(symbol),
        decision=_decision(symbol),
        brief=ResearchBrief(
            symbol=symbol,
            asof="2026-06-13",
            fundamentals={"sector": "Tech"},
            news=[],
            technicals={"vol_60d": vol_60d},
            analysis=ResearchAnalysis(overall_stance="bullish", overall_summary="ok"),
            sources=[],
            llm_ok=True,
        ),
        debate=Debate(
            symbol=symbol,
            asof="2026-06-13",
            bull=Thesis(summary="up", stance_strength=0.7),
            bear=Thesis(summary="down", stance_strength=0.4),
            llm_ok=True,
        ),
    )


_PICKS_JSON = (
    '{"picks":[{"symbol":"AAPL","conviction":0.9,"rationale":"a"},'
    '{"symbol":"MSFT","conviction":0.5,"rationale":"m"},'
    '{"symbol":"NVDA","conviction":0.3,"rationale":"n"}],"overall_rationale":"balanced"}'
)


def test_manage_orders_and_sizes_by_conviction(monkeypatch):
    monkeypatch.setattr(manage_mod.audit, "log_event", lambda *a, **k: None)
    candidates = [_candidate("AAPL"), _candidate("MSFT"), _candidate("NVDA")]
    # loose per-name cap so the conviction ordering survives un-clamped
    settings = AgentSettings(max_position_weight=0.5, target_gross_exposure=0.95)
    pf = manage(candidates, llm=_FakeLLM(_PICKS_JSON), settings=settings)
    assert pf.llm_ok is True
    assert pf.halted is False
    assert [p.symbol for p in pf.positions] == ["AAPL", "MSFT", "NVDA"]
    weights = [p.weight for p in pf.positions]
    assert weights == sorted(weights, reverse=True)
    assert pf.gross_exposure <= 0.95 + 1e-9
    assert abs(pf.cash_weight - (1 - pf.gross_exposure)) < 1e-9
    assert pf.rationale == "balanced"


def test_manage_enforces_per_name_cap(monkeypatch):
    monkeypatch.setattr(manage_mod.audit, "log_event", lambda *a, **k: None)
    candidates = [_candidate("AAPL"), _candidate("MSFT"), _candidate("NVDA")]
    settings = AgentSettings()  # default max_position_weight=0.05
    pf = manage(candidates, llm=_FakeLLM(_PICKS_JSON), settings=settings)
    assert all(p.weight <= 0.05 + 1e-9 for p in pf.positions)


def test_manage_drops_unknown_symbol(monkeypatch):
    monkeypatch.setattr(manage_mod.audit, "log_event", lambda *a, **k: None)
    llm = _FakeLLM(
        '{"picks":[{"symbol":"AAPL","conviction":0.8,"rationale":"a"},'
        '{"symbol":"FAKE","conviction":0.9,"rationale":"x"}],"overall_rationale":"r"}'
    )
    pf = manage([_candidate("AAPL")], llm=llm, settings=AgentSettings(max_position_weight=0.5))
    assert [p.symbol for p in pf.positions] == ["AAPL"]


def test_risk_gate_killswitch_halts_to_cash():
    settings = AgentSettings()
    positions = [Position(symbol="AAPL", weight=0.04, conviction=0.9, rationale="a")]
    kept, notes, halted = apply_risk_gate(
        positions, settings=settings, current_drawdown=settings.max_drawdown_kill
    )
    assert halted is True
    assert kept == []
    assert any("HALT" in n for n in notes)


def test_risk_gate_caps_and_gross():
    settings = AgentSettings()
    positions = [
        Position(symbol=f"S{i}", weight=0.5, conviction=0.5, rationale="r") for i in range(30)
    ]
    kept, _notes, halted = apply_risk_gate(positions, settings=settings)
    assert halted is False
    assert all(p.weight <= settings.max_position_weight + 1e-9 for p in kept)
    assert sum(p.weight for p in kept) <= settings.target_gross_exposure + 1e-9


def test_manage_failsafe_all_cash(monkeypatch):
    monkeypatch.setattr(manage_mod.audit, "log_event", lambda *a, **k: None)
    pf = manage([_candidate("AAPL")], llm=_FakeLLM("not json"), settings=AgentSettings())
    assert pf.llm_ok is False
    assert pf.positions == []
    assert pf.cash_weight == 1.0
