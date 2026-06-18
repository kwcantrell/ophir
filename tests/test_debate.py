"""Tests for the bull/bear debate (LLM mocked, no network/GPU)."""

import json
import types

from ophir.agent import debate as debate_mod
from ophir.agent.debate import Debate, Thesis, debate_ticker
from ophir.agent.research import ResearchAnalysis, ResearchBrief

_BULL_JSON = (
    '{"summary":"upside ahead","key_points":["revenue growth","margin expansion"],'
    '"key_risks":["rich valuation"],"stance_strength":0.7}'
)
_BEAR_JSON = (
    '{"summary":"downside risk","key_points":["macro headwinds"],'
    '"key_risks":["short squeeze"],"stance_strength":1.4}'
)


class _SideLLM:
    """Returns bull vs bear JSON based on the system message's wording."""

    def invoke(self, messages):
        text = " ".join(getattr(m, "content", "") for m in messages).lower()
        return types.SimpleNamespace(content=_BULL_JSON if "bullish" in text else _BEAR_JSON)


class _BadLLM:
    def invoke(self, _messages):
        return types.SimpleNamespace(content="no json here")


def _brief(symbol="AAPL"):
    return ResearchBrief(
        symbol=symbol,
        asof="2026-06-13",
        fundamentals={"sector": "Technology", "trailingPE": 30.0},
        news=[{"title": "Record quarter", "publisher": "Reuters", "link": "x", "published": ""}],
        technicals={"last_close": 200.0, "forecast_cum_return": 0.05},
        analysis=ResearchAnalysis(overall_stance="neutral", overall_summary="mixed"),
        sources=["yfinance:info"],
        llm_ok=True,
    )


def test_debate_ticker_two_theses(monkeypatch):
    monkeypatch.setattr(debate_mod.audit, "log_event", lambda *a, **k: None)
    d = debate_ticker(_brief(), llm=_SideLLM())
    assert isinstance(d, Debate)
    assert d.llm_ok is True
    assert d.bull.summary == "upside ahead"
    assert "revenue growth" in d.bull.key_points
    assert d.bull.stance_strength == 0.7
    assert d.bear.summary == "downside risk"
    assert d.bear.stance_strength == 1.0  # clamped from 1.4
    assert d.bull.key_points != d.bear.key_points


def test_debate_ticker_failsafe(monkeypatch):
    monkeypatch.setattr(debate_mod.audit, "log_event", lambda *a, **k: None)
    d = debate_ticker(_brief(), llm=_BadLLM())
    assert d.llm_ok is False
    assert d.bull.stance_strength == 0.0
    assert d.bear.stance_strength == 0.0
    assert "unavailable" in d.bull.summary
    assert "unavailable" in d.bear.summary


def test_thesis_splits_blob_key_points():
    # The model returns one string with several bullets instead of an array.
    t = Thesis(key_points="- alpha\n- beta")
    assert t.key_points == ["alpha", "beta"]


def test_thesis_drops_leaked_field_fragment():
    # Observed GOOGL failure: bullets crammed into one string with a leaked key_risks tail.
    blob = "- Near 52-week high\n- Elevated P/E\nkey_risks':['Forward P/E may improve']"
    t = Thesis(key_points=blob, key_risks=[])
    assert t.key_points == ["Near 52-week high", "Elevated P/E"]
    assert all("key_risks" not in p for p in t.key_points)
    assert t.key_risks == []


def test_thesis_valid_array_unchanged():
    t = Thesis(key_points=["revenue growth", "margin expansion"], key_risks=["rich valuation"])
    assert t.key_points == ["revenue growth", "margin expansion"]
    assert t.key_risks == ["rich valuation"]


def test_debate_ticker_sanitizes_malformed_thesis(monkeypatch):
    monkeypatch.setattr(debate_mod.audit, "log_event", lambda *a, **k: None)
    payload = json.dumps(
        {
            "summary": "bearish",
            "key_points": "- Near high\n- Elevated P/E\nkey_risks':['leaked risk']",
            "key_risks": [],
            "stance_strength": 0.5,
        }
    )

    class _MalformedLLM:
        def invoke(self, _messages):
            return types.SimpleNamespace(content=payload)

    d = debate_ticker(_brief(), llm=_MalformedLLM())
    assert d.bull.key_points == ["Near high", "Elevated P/E"]
    assert all("key_risks" not in p for p in d.bull.key_points)
