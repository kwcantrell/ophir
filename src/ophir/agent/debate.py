"""Bull/bear debate: two grounded theses per ticker from its research brief.

For each candidate we ask the local ``gpt-oss:20b`` model to argue the strongest
*bullish* case and, separately, the strongest *bearish* case -- each reasoning
ONLY over the ticker's :class:`~ophir.agent.research.ResearchBrief` (fundamentals,
news, technicals, and the neutral analysis), never inventing figures. The Phase 6
manager later weighs the two; here we only produce the structured theses.

Per the ``trading-best-practices`` skill: ground every claim, no fabricated
figures, fail safe (a neutral thesis when the model is unreachable), and audit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from ophir.agent import audit
from ophir.agent.config import get_settings
from ophir.agent.decide import _extract_json_object

if TYPE_CHECKING:
    from ophir.agent.config import AgentSettings
    from ophir.agent.research import ResearchBrief

Side = Literal["bull", "bear"]


class Thesis(BaseModel):
    """One side's validated argument over a research brief."""

    summary: str = ""
    key_points: list[str] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)
    stance_strength: float = 0.5


@dataclass(frozen=True, slots=True)
class Debate:
    """The bull and bear theses for one ticker.

    Attributes
    ----------
    symbol : str
        Ticker symbol.
    asof : str
        ISO date the underlying brief is conditioned on.
    bull, bear : Thesis
        The bullish and bearish arguments.
    llm_ok : bool
        ``True`` when both theses came from the model (not a fail-safe default).
    """

    symbol: str
    asof: str
    bull: Thesis
    bear: Thesis
    llm_ok: bool = True


def _build_messages(brief: ResearchBrief, side: Side) -> tuple[Any, Any]:
    """Build the grounded system + human messages for one side of the debate."""
    from langchain_core.messages import HumanMessage, SystemMessage

    case = "bullish" if side == "bull" else "bearish"
    system = SystemMessage(
        content=(
            f"You are a {side} equity analyst. Argue the strongest {case} case for "
            f"{brief.symbol} using ONLY the research data provided below -- never invent "
            "prices, ratios, headlines, or figures, and do not use outside knowledge. List "
            "the honest risks to your own case under key_risks. Reply with ONLY a JSON object "
            'of the form {"summary": "...", "key_points": ["..."], "key_risks": ["..."], '
            '"stance_strength": 0.0-1.0} and nothing else.'
        )
    )
    news_lines = (
        "\n".join(f"  - {n['title']}" for n in brief.news if n.get("title"))
        or "  (no recent headlines)"
    )
    human = HumanMessage(
        content=(
            f"Ticker: {brief.symbol}\nAs-of date: {brief.asof}\n\n"
            f"FUNDAMENTALS:\n{json.dumps(brief.fundamentals, indent=2, default=str)}\n\n"
            f"NEWS HEADLINES:\n{news_lines}\n\n"
            f"TECHNICALS:\n{json.dumps(brief.technicals, indent=2, default=str)}\n\n"
            "NEUTRAL ANALYST NOTES:\n"
            f"  fundamentals: {brief.analysis.fundamentals_summary}\n"
            f"  news: {brief.analysis.news_summary}\n"
            f"  technicals: {brief.analysis.technicals_summary}\n"
            f"  overall: {brief.analysis.overall_summary}"
        )
    )
    return system, human


def _argue(brief: ResearchBrief, side: Side, llm: Any) -> tuple[Thesis, bool]:
    """Ask the model for one side's thesis; fail safe to a neutral default."""
    system, human = _build_messages(brief, side)
    try:
        content = llm.invoke([system, human]).content
        text = content if isinstance(content, str) else str(content)
        data = json.loads(_extract_json_object(text))
        if not isinstance(data, dict):
            raise ValueError("LLM did not return a JSON object")
        strength = data.get("stance_strength")
        if isinstance(strength, (int, float)):
            data["stance_strength"] = min(1.0, max(0.0, float(strength)))
        return Thesis(**data), True
    except Exception as exc:  # fail safe: a neutral, no-strength thesis
        note = f"{side} thesis unavailable ({type(exc).__name__})"
        return Thesis(summary=note, stance_strength=0.0), False


def debate_ticker(
    brief: ResearchBrief, *, llm: Any = None, settings: AgentSettings | None = None
) -> Debate:
    """Produce independent bull and bear theses for one ticker's brief.

    Parameters
    ----------
    brief : ResearchBrief
        The grounded research brief to argue over.
    llm : object, optional
        A chat model exposing ``.invoke(messages) -> obj.content``; defaults to a
        ``ChatOllama(model=settings.ollama_model, temperature=0)``. Inject a fake
        in tests.
    settings : AgentSettings, optional
        Defaults to the cached :func:`~ophir.agent.config.get_settings`.

    Returns
    -------
    Debate
        The bull and bear theses (fail-safe neutral when the model is unavailable).
    """
    settings = settings or get_settings()
    if llm is None:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=settings.ollama_model,
            temperature=0,
            format="json",
            base_url=settings.ollama_base_url,
            num_ctx=settings.ollama_num_ctx,
        )

    bull, bull_ok = _argue(brief, "bull", llm)
    bear, bear_ok = _argue(brief, "bear", llm)
    llm_ok = bull_ok and bear_ok

    debate = Debate(symbol=brief.symbol, asof=brief.asof, bull=bull, bear=bear, llm_ok=llm_ok)
    audit.log_event(
        "debate",
        symbol=brief.symbol,
        bull_strength=round(bull.stance_strength, 4),
        bear_strength=round(bear.stance_strength, 4),
        llm_ok=llm_ok,
    )
    return debate


def debate_many(
    briefs: list[ResearchBrief], *, llm: Any = None, settings: AgentSettings | None = None
) -> list[Debate]:
    """Run the bull/bear debate for several briefs; failures are logged and skipped."""
    settings = settings or get_settings()
    if llm is None:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=settings.ollama_model,
            temperature=0,
            format="json",
            base_url=settings.ollama_base_url,
            num_ctx=settings.ollama_num_ctx,
        )

    debates: list[Debate] = []
    for brief in briefs:
        try:
            debates.append(debate_ticker(brief, llm=llm, settings=settings))
        except (ValueError, OSError) as exc:
            audit.log_event("debate_failed", symbol=brief.symbol, error=str(exc))
            print(f"[debate] {brief.symbol}: FAILED -- {exc}")
    return debates
