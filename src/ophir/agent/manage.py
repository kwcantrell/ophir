"""Manager pick + deterministic risk gate: the final portfolio decision.

A manager LLM ingests the full per-ticker ensemble (forecast + quant/Ollama
decisions + research brief + bull/bear debate) and returns ranked picks, each
with a *conviction* and rationale -- never raw weights. Deterministic code then
sizes those convictions (inverse-volatility, scaled toward a target annual
volatility) and a **risk gate** enforces the hard limits -- per-name cap, gross
exposure, sanity, and a drawdown / daily-loss kill-switch -- so a hallucinating
or over-eager LLM can never push the book past a configured limit.

Per the ``trading-best-practices`` skill: the LLM advises, deterministic code
sizes, and the gate is the final hard checkpoint. Everything is audit-logged.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ophir.agent import audit
from ophir.agent.config import get_settings
from ophir.agent.decide import _extract_json_object

if TYPE_CHECKING:
    from ophir.agent.config import AgentSettings
    from ophir.agent.debate import Debate
    from ophir.agent.decide import DecisionComparison
    from ophir.agent.predict import Forecast
    from ophir.agent.research import ResearchBrief

_TRADING_DAYS = 252.0


@dataclass(frozen=True, slots=True)
class Candidate:
    """The full ensemble for one ticker, fed to the manager."""

    symbol: str
    forecast: Forecast
    decision: DecisionComparison
    brief: ResearchBrief
    debate: Debate


class Pick(BaseModel):
    """One manager pick (LLM output): a symbol, a conviction, and why."""

    symbol: str = ""
    conviction: float = 0.0
    rationale: str = ""


class ManagerOutput(BaseModel):
    """The manager's validated output."""

    picks: list[Pick] = Field(default_factory=list)
    overall_rationale: str = ""


@dataclass(frozen=True, slots=True)
class Position:
    """A final target position after sizing and the risk gate."""

    symbol: str
    weight: float
    conviction: float
    rationale: str


@dataclass(frozen=True, slots=True)
class Portfolio:
    """The gated target portfolio.

    Attributes
    ----------
    asof : str
        ISO date the underlying forecasts are conditioned on.
    positions : list[Position]
        Target positions (weights), sorted by weight descending.
    cash_weight : float
        ``1 - gross_exposure``.
    gross_exposure : float
        Sum of position weights (``<= target_gross_exposure``).
    rationale : str
        The manager's overall rationale.
    gate_notes : list[str]
        Every adjustment the risk gate made (drops, clamps, scaling, halt).
    halted : bool
        ``True`` if the kill-switch fired (all cash).
    llm_ok : bool
        ``True`` if the manager LLM produced the picks (not a fail-safe default).
    """

    asof: str
    positions: list[Position]
    cash_weight: float
    gross_exposure: float
    rationale: str
    gate_notes: list[str] = field(default_factory=list)
    halted: bool = False
    llm_ok: bool = True


def _dossier(candidate: Candidate) -> dict[str, Any]:
    """Compact grounded summary of one candidate for the manager prompt."""
    quant = candidate.decision.quant
    ollama = candidate.decision.ollama
    return {
        "symbol": candidate.symbol,
        "forecast_cum_return": round(candidate.forecast.cum_return, 4),
        "quant_decision": f"{quant.action} ({quant.confidence:.0%})",
        "ollama_decision": f"{ollama.action} ({ollama.confidence:.0%})",
        "research_stance": candidate.brief.analysis.overall_stance,
        "research_summary": candidate.brief.analysis.overall_summary,
        "bull": f"{candidate.debate.bull.stance_strength:.0%}: {candidate.debate.bull.summary}",
        "bear": f"{candidate.debate.bear.stance_strength:.0%}: {candidate.debate.bear.summary}",
        "daily_vol_60d": candidate.brief.technicals.get("vol_60d"),
    }


def _build_messages(dossiers: list[dict[str, Any]]) -> tuple[Any, Any]:
    """Build the grounded system + human messages for the manager."""
    from langchain_core.messages import HumanMessage, SystemMessage

    system = SystemMessage(
        content=(
            "You are a disciplined paper-trading portfolio manager. From the candidate "
            "dossiers below, choose and rank the names worth holding, each with a conviction "
            "from 0.0 to 1.0 and a one-sentence rationale. Use ONLY the provided data -- never "
            "invent figures. Prefer fewer, higher-conviction names; picking none is acceptable. "
            'Reply with ONLY a JSON object of the form {"picks": [{"symbol": "...", '
            '"conviction": 0.0-1.0, "rationale": "..."}], "overall_rationale": "..."} and '
            "nothing else."
        )
    )
    human = HumanMessage(content="CANDIDATES:\n" + json.dumps(dossiers, indent=2, default=str))
    return system, human


def _propose(dossiers: list[dict[str, Any]], llm: Any) -> tuple[ManagerOutput, bool]:
    """Ask the manager LLM for ranked picks; fail safe to no picks (all cash)."""
    system, human = _build_messages(dossiers)
    try:
        content = llm.invoke([system, human]).content
        text = content if isinstance(content, str) else str(content)
        data = json.loads(_extract_json_object(text))
        if not isinstance(data, dict):
            raise ValueError("LLM did not return a JSON object")
        return ManagerOutput(**data), True
    except Exception as exc:  # fail safe: no picks -> all cash
        return ManagerOutput(overall_rationale=f"manager unavailable ({type(exc).__name__})"), False


def _size(
    picks: list[Pick], candidates: list[Candidate], settings: AgentSettings
) -> list[Position]:
    """Turn convictions into target weights: inverse-vol, scaled to the vol target.

    Only picks naming a known candidate with a finite positive conviction survive
    (a hallucination guard). Weights are conviction-weighted and inversely scaled
    by each name's annualized volatility, then levered toward
    ``annual_vol_target`` (capped at ``target_gross_exposure``). The risk gate
    applies the hard per-name and gross caps afterward.
    """
    by_symbol = {c.symbol.upper(): c for c in candidates}
    chosen: dict[str, Pick] = {}
    for pick in picks:
        symbol = pick.symbol.upper().strip()
        if symbol not in by_symbol or not math.isfinite(pick.conviction) or pick.conviction <= 0:
            continue
        if symbol not in chosen or pick.conviction > chosen[symbol].conviction:
            chosen[symbol] = pick
    if not chosen:
        return []

    annualize = math.sqrt(_TRADING_DAYS)
    sigma: dict[str, float] = {}
    for symbol in chosen:
        vol60 = by_symbol[symbol].brief.technicals.get("vol_60d")
        if isinstance(vol60, (int, float)) and math.isfinite(vol60) and vol60 > 0:
            sigma[symbol] = float(vol60) * annualize
        else:
            sigma[symbol] = settings.annual_vol_target  # neutral fallback

    base = {symbol: chosen[symbol].conviction / sigma[symbol] for symbol in chosen}
    total = sum(base.values())
    if total <= 0:
        return []
    norm = {symbol: base[symbol] / total for symbol in chosen}
    book_vol = sum(norm[symbol] * sigma[symbol] for symbol in chosen)
    leverage = (
        min(settings.annual_vol_target / book_vol, settings.target_gross_exposure)
        if book_vol > 0
        else 0.0
    )

    positions: list[Position] = []
    for symbol, pick in chosen.items():
        positions.append(
            Position(
                symbol=symbol,
                weight=norm[symbol] * leverage,
                conviction=pick.conviction,
                rationale=pick.rationale,
            )
        )
    return positions


def apply_risk_gate(
    positions: list[Position],
    *,
    settings: AgentSettings,
    current_drawdown: float = 0.0,
    daily_loss: float = 0.0,
) -> tuple[list[Position], list[str], bool]:
    """Enforce the hard risk limits; returns (kept positions, notes, halted).

    Halts to all cash on a drawdown / daily-loss breach (the kill-switch; state is
    injected and defaults to safe -- a broker feeds real numbers in a later phase).
    Otherwise drops non-finite/non-positive weights, clamps each to
    ``max_position_weight``, and scales the book down to ``target_gross_exposure``.
    No output can exceed either limit.
    """
    notes: list[str] = []
    if current_drawdown >= settings.max_drawdown_kill:
        notes.append(
            f"HALT: drawdown {current_drawdown:.2%} >= kill {settings.max_drawdown_kill:.2%}"
        )
        return [], notes, True
    if daily_loss >= settings.daily_loss_limit:
        notes.append(f"HALT: daily loss {daily_loss:.2%} >= limit {settings.daily_loss_limit:.2%}")
        return [], notes, True

    kept: list[Position] = []
    for pos in positions:
        if not math.isfinite(pos.weight) or pos.weight <= 0:
            notes.append(f"dropped {pos.symbol}: non-finite/non-positive weight")
            continue
        weight = pos.weight
        if weight > settings.max_position_weight:
            notes.append(
                f"clamped {pos.symbol}: {weight:.2%} -> {settings.max_position_weight:.2%}"
            )
            weight = settings.max_position_weight
        kept.append(
            Position(
                symbol=pos.symbol,
                weight=weight,
                conviction=pos.conviction,
                rationale=pos.rationale,
            )
        )

    gross = sum(p.weight for p in kept)
    if gross > settings.target_gross_exposure and gross > 0:
        scale = settings.target_gross_exposure / gross
        notes.append(f"scaled book x{scale:.3f} to gross {settings.target_gross_exposure:.2%}")
        kept = [
            Position(
                symbol=p.symbol,
                weight=p.weight * scale,
                conviction=p.conviction,
                rationale=p.rationale,
            )
            for p in kept
        ]
    return kept, notes, False


def manage(
    candidates: list[Candidate],
    *,
    llm: Any = None,
    settings: AgentSettings | None = None,
    current_drawdown: float = 0.0,
    daily_loss: float = 0.0,
) -> Portfolio:
    """Run the manager pick + sizing + risk gate over the candidate ensemble.

    Parameters
    ----------
    candidates : list[Candidate]
        The per-ticker ensembles to choose from.
    llm : object, optional
        A chat model exposing ``.invoke(messages) -> obj.content``; defaults to a
        JSON-constrained ``ChatOllama``. Inject a fake in tests.
    settings : AgentSettings, optional
        Defaults to the cached :func:`~ophir.agent.config.get_settings`.
    current_drawdown, daily_loss : float, optional
        Account state for the kill-switch (default ``0.0`` -- no halt).

    Returns
    -------
    Portfolio
        The gated target portfolio (all cash if halted or the manager failed).
    """
    settings = settings or get_settings()
    asof = candidates[0].forecast.asof if candidates else ""
    if llm is None:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=settings.ollama_model,
            temperature=0,
            format="json",
            base_url=settings.ollama_base_url,
        )

    output, llm_ok = _propose([_dossier(c) for c in candidates], llm)
    sized = _size(output.picks, candidates, settings)
    gated, notes, halted = apply_risk_gate(
        sized, settings=settings, current_drawdown=current_drawdown, daily_loss=daily_loss
    )
    gross = sum(p.weight for p in gated)

    portfolio = Portfolio(
        asof=asof,
        positions=sorted(gated, key=lambda p: p.weight, reverse=True),
        cash_weight=max(0.0, 1.0 - gross),
        gross_exposure=gross,
        rationale=output.overall_rationale,
        gate_notes=notes,
        halted=halted,
        llm_ok=llm_ok,
    )
    audit.log_event(
        "manage", n_picks=len(gated), gross=round(gross, 4), halted=halted, llm_ok=llm_ok
    )
    return portfolio
