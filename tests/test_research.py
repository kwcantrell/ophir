"""Tests for the research layer (yfinance + LLM mocked, no network/GPU)."""

import types

from ophir.agent import research as research_mod
from ophir.agent.predict import Forecast
from ophir.agent.research import (
    ResearchBrief,
    gather_fundamentals,
    gather_news,
    gather_technicals,
    research_ticker,
)


class _FakeLLM:
    """Minimal stand-in for ChatOllama: ``.invoke`` returns canned ``.content``."""

    def __init__(self, content):
        self._content = content

    def invoke(self, _messages):
        return types.SimpleNamespace(content=self._content)


_FAKE_INFO = {
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "marketCap": 3_000_000_000_000,
    "trailingPE": 30.5,
    "forwardPE": 28.0,
    "profitMargins": 0.25,
    "beta": 1.2,
    "dividendYield": 0.005,
    "fiftyTwoWeekHigh": 250.0,
    "fiftyTwoWeekLow": 150.0,
    "currentPrice": 200.0,
    "irrelevant": "dropped",
}

_FAKE_NEWS_NESTED = [
    {
        "content": {
            "title": "Company ships record units",
            "provider": {"displayName": "Reuters"},
            "canonicalUrl": {"url": "https://example.com/a"},
            "pubDate": "2026-06-10T00:00:00Z",
        }
    }
]

_FAKE_NEWS_FLAT = [
    {
        "title": "Analyst upgrades stock",
        "publisher": "Bloomberg",
        "link": "https://example.com/b",
        "providerPublishTime": 1_700_000_000,
    }
]


def _fake_ticker(news):
    class _FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        @property
        def info(self):
            return dict(_FAKE_INFO)

        @property
        def news(self):
            return news

    return _FakeTicker


def _forecast(symbol="AAPL", cum_return=0.05, n=90):
    return Forecast(
        symbol=symbol,
        asof="2026-06-13",
        horizon=n,
        r_close=[cum_return / n] * n,
        upside=[0.012] * n,
        downside=[0.009] * n,
        cum_return=cum_return,
        score=cum_return,
    )


def test_gather_fundamentals_subset(monkeypatch):
    monkeypatch.setattr("yfinance.Ticker", _fake_ticker(_FAKE_NEWS_NESTED))
    f = gather_fundamentals("AAPL")
    assert f["sector"] == "Technology"
    assert f["marketCap"] == 3_000_000_000_000
    assert "irrelevant" not in f


def test_gather_news_nested_shape(monkeypatch):
    monkeypatch.setattr("yfinance.Ticker", _fake_ticker(_FAKE_NEWS_NESTED))
    news = gather_news("AAPL")
    assert len(news) == 1
    assert news[0]["title"] == "Company ships record units"
    assert news[0]["publisher"] == "Reuters"
    assert news[0]["link"] == "https://example.com/a"


def test_gather_news_flat_shape(monkeypatch):
    monkeypatch.setattr("yfinance.Ticker", _fake_ticker(_FAKE_NEWS_FLAT))
    news = gather_news("AAPL")
    assert news[0]["title"] == "Analyst upgrades stock"
    assert news[0]["publisher"] == "Bloomberg"
    assert news[0]["link"] == "https://example.com/b"
    assert news[0]["published"]  # epoch converted to an ISO date


def test_gather_technicals_with_forecast(monkeypatch, make_ohlcv):
    monkeypatch.setattr(research_mod, "load_history", lambda *a, **k: make_ohlcv(n_days=300))
    tech = gather_technicals("AAPL", _forecast(cum_return=0.07))
    assert tech["last_close"] > 0
    assert tech["vol_20d"] is not None
    assert tech["forecast_cum_return"] == 0.07
    assert "forecast_mean_upside" in tech


def test_research_ticker_builds_brief(monkeypatch, make_ohlcv):
    monkeypatch.setattr("yfinance.Ticker", _fake_ticker(_FAKE_NEWS_NESTED))
    monkeypatch.setattr(research_mod, "load_history", lambda *a, **k: make_ohlcv(n_days=300))
    monkeypatch.setattr(research_mod.audit, "log_event", lambda *a, **k: None)
    llm = _FakeLLM(
        '{"fundamentals_summary": "solid margins", "news_summary": "positive",'
        ' "technicals_summary": "uptrend", "overall_stance": "Bullish",'
        ' "overall_summary": "constructive"}'
    )
    brief = research_ticker("aapl", forecast=_forecast(symbol="AAPL"), llm=llm)
    assert isinstance(brief, ResearchBrief)
    assert brief.symbol == "AAPL"
    assert brief.asof == "2026-06-13"
    assert brief.llm_ok is True
    assert brief.analysis.overall_stance == "bullish"  # normalized to lowercase
    assert brief.fundamentals["sector"] == "Technology"
    assert len(brief.news) == 1
    assert "yfinance:info" in brief.sources
    assert "ophir:forecast" in brief.sources


def test_research_ticker_failsafe_on_bad_llm(monkeypatch, make_ohlcv):
    monkeypatch.setattr("yfinance.Ticker", _fake_ticker(_FAKE_NEWS_NESTED))
    monkeypatch.setattr(research_mod, "load_history", lambda *a, **k: make_ohlcv(n_days=300))
    monkeypatch.setattr(research_mod.audit, "log_event", lambda *a, **k: None)
    brief = research_ticker("AAPL", forecast=_forecast(), llm=_FakeLLM("I cannot answer"))
    assert brief.llm_ok is False
    assert brief.analysis.overall_stance == "neutral"
    assert brief.fundamentals["sector"] == "Technology"  # grounded data still intact
    assert "synthesis unavailable" in brief.analysis.overall_summary
