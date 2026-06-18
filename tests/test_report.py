"""Tests for the per-stock dossier report writer (no network/GPU; files go to tmp_path)."""

from decimal import Decimal

from ophir.agent import report as report_mod
from ophir.agent.debate import Debate, Thesis
from ophir.agent.decide import Decision, DecisionComparison
from ophir.agent.execute import Order
from ophir.agent.manage import Candidate, Portfolio, Position
from ophir.agent.predict import Forecast
from ophir.agent.report import write_reports
from ophir.agent.research import ResearchAnalysis, ResearchBrief


def _candidate(symbol, cum=0.05, stance="bullish"):
    return Candidate(
        symbol=symbol,
        forecast=Forecast(
            symbol=symbol,
            asof="2026-06-16",
            horizon=90,
            r_close=[cum / 90] * 90,
            upside=[0.012] * 90,
            downside=[0.009] * 90,
            cum_return=cum,
            score=cum,
        ),
        decision=DecisionComparison(
            symbol=symbol,
            quant=Decision(
                symbol=symbol,
                action="BUY",
                confidence=0.8,
                rationale="quant-reason",
                source="quant",
                cum_return=cum,
                asof="2026-06-16",
            ),
            ollama=Decision(
                symbol=symbol,
                action="HOLD",
                confidence=0.5,
                rationale="ollama-reason",
                source="ollama",
                cum_return=cum,
                asof="2026-06-16",
            ),
            agree=False,
        ),
        brief=ResearchBrief(
            symbol=symbol,
            asof="2026-06-16",
            fundamentals={},
            news=[],
            technicals={"vol_60d": 0.015},
            analysis=ResearchAnalysis(overall_stance=stance, overall_summary="research summary"),
            sources=[],
            llm_ok=True,
        ),
        debate=Debate(
            symbol=symbol,
            asof="2026-06-16",
            bull=Thesis(
                summary="bull case", key_points=["bp1"], key_risks=["br1"], stance_strength=0.7
            ),
            bear=Thesis(
                summary="bear case", key_points=["xp1"], key_risks=["xr1"], stance_strength=0.4
            ),
            llm_ok=True,
        ),
    )


def _portfolio(**kw):
    base = {
        "asof": "2026-06-16",
        "positions": [
            Position(symbol="AAPL", weight=0.042, conviction=0.9, rationale="strong conviction")
        ],
        "cash_weight": 0.958,
        "gross_exposure": 0.042,
        "rationale": "kept AAPL, passed on MSFT",
        "gate_notes": ["dropped MSFT: non-finite/non-positive weight"],
        "halted": False,
        "llm_ok": True,
    }
    return Portfolio(**{**base, **kw})


def _no_audit(monkeypatch):
    monkeypatch.setattr(report_mod.audit, "log_event", lambda *a, **k: None)


def test_writes_per_stock_and_index(tmp_path, monkeypatch):
    _no_audit(monkeypatch)
    candidates = [_candidate("AAPL"), _candidate("MSFT", cum=0.01, stance="neutral")]
    out = write_reports(candidates, _portfolio(), [], [], out_dir=tmp_path)

    assert out == tmp_path / "2026-06-16"
    assert (out / "AAPL.md").exists()
    assert (out / "MSFT.md").exists()
    assert (out / "INDEX.md").exists()

    aapl = (out / "AAPL.md").read_text(encoding="utf-8")
    assert "SELECTED" in aapl and "strong conviction" in aapl
    assert "Bull thesis" in aapl and "bull case" in aapl
    assert "Bear thesis" in aapl and "bear case" in aapl
    assert "2026-06-16" in aapl  # as-of (last closed session)

    msft = (out / "MSFT.md").read_text(encoding="utf-8")
    assert "NOT SELECTED" in msft
    assert "dropped MSFT" in msft  # gate note surfaced on the right name

    index = (out / "INDEX.md").read_text(encoding="utf-8")
    assert "[AAPL](AAPL.md)" in index
    assert "[MSFT](MSFT.md)" in index


def test_halted_banner(tmp_path, monkeypatch):
    _no_audit(monkeypatch)
    pf = _portfolio(positions=[], halted=True, gross_exposure=0.0, cash_weight=1.0)
    out = write_reports([_candidate("AAPL")], pf, [], [], out_dir=tmp_path)
    aapl = (out / "AAPL.md").read_text(encoding="utf-8")
    assert "HALTED" in aapl and "NOT SELECTED" in aapl


def test_llm_unavailable_thesis(tmp_path, monkeypatch):
    _no_audit(monkeypatch)
    c = _candidate("AAPL")
    cand = Candidate(
        symbol="AAPL",
        forecast=c.forecast,
        decision=c.decision,
        brief=c.brief,
        debate=Debate(symbol="AAPL", asof="2026-06-16", bull=Thesis(), bear=Thesis(), llm_ok=False),
    )
    out = write_reports([cand], _portfolio(positions=[]), [], [], out_dir=tmp_path)
    aapl = (out / "AAPL.md").read_text(encoding="utf-8")
    assert "no thesis" in aapl


def test_orders_section(tmp_path, monkeypatch):
    _no_audit(monkeypatch)
    orders = [Order(symbol="AAPL", side="buy", notional=Decimal("4200"), client_order_id="x")]
    plan = ["DRY-RUN  BUY AAPL $4,200.00"]
    out = write_reports([_candidate("AAPL")], _portfolio(), orders, plan, out_dir=tmp_path)
    aapl = (out / "AAPL.md").read_text(encoding="utf-8")
    assert "Orders this run" in aapl and "BUY" in aapl and "4,200.00" in aapl
    index = (out / "INDEX.md").read_text(encoding="utf-8")
    assert "DRY-RUN  BUY AAPL" in index


def test_empty_candidates_writes_index_only(tmp_path, monkeypatch):
    _no_audit(monkeypatch)
    out = write_reports([], _portfolio(asof="", positions=[]), [], [], out_dir=tmp_path)
    assert (out / "INDEX.md").exists()
    assert list(out.glob("*.md")) == [out / "INDEX.md"]


def test_filename_sanitization(tmp_path, monkeypatch):
    _no_audit(monkeypatch)
    out = write_reports([_candidate("BRK.B")], _portfolio(positions=[]), [], [], out_dir=tmp_path)
    assert (out / "BRK.B.md").exists()
