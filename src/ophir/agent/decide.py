"""Turn a model :class:`~ophir.agent.predict.Forecast` into a buy/sell/hold decision.

Two parallel tracks, meant to be compared:

* **quant** — a deterministic rule over the forecast's cumulative return (with an
  optional downside penalty). Pure and fully testable.
* **ollama** — the local ``gpt-oss:20b`` model (the same setup :mod:`ophir.ui`
  uses), grounded *only* in the forecast numbers and asked for a structured JSON
  verdict.

Per the ``trading-best-practices`` skill these decisions are advisory and
paper-only: the LLM is grounded in real numbers, its output is validated, any
ambiguity falls back to ``HOLD``, and every decision is written to the audit
trail. Portfolio sizing and the deterministic risk gate live in a later phase.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from ophir.agent import audit
from ophir.agent.config import get_settings

if TYPE_CHECKING:
    from ophir.agent.config import AgentSettings
    from ophir.agent.predict import Forecast

Action = Literal["BUY", "SELL", "HOLD"]
Source = Literal["quant", "ollama"]


@dataclass(frozen=True, slots=True)
class Decision:
    """A buy/sell/hold call for one ticker from one track.

    Attributes
    ----------
    symbol : str
        Ticker symbol.
    action : {"BUY", "SELL", "HOLD"}
        The recommended action.
    confidence : float
        Confidence in ``[0, 1]``.
    rationale : str
        Human-readable explanation.
    source : {"quant", "ollama"}
        Which track produced the decision.
    cum_return : float
        The forecast cumulative return the decision was based on.
    asof : str
        ISO date of the forecast's last input bar.
    """

    symbol: str
    action: Action
    confidence: float
    rationale: str
    source: Source
    cum_return: float
    asof: str


@dataclass(frozen=True, slots=True)
class DecisionComparison:
    """The quant and Ollama decisions for one ticker, side by side."""

    symbol: str
    quant: Decision
    ollama: Decision
    agree: bool


class LLMVerdict(BaseModel):
    """Schema the Ollama track must return; anything else falls back to ``HOLD``."""

    action: Action
    confidence: float = 0.5
    rationale: str = ""


def _log(decision: Decision) -> None:
    """Append the decision to the audit trail."""
    audit.log_event(
        "decision",
        symbol=decision.symbol,
        source=decision.source,
        action=decision.action,
        confidence=round(decision.confidence, 4),
        cum_return=round(decision.cum_return, 6),
    )


def quant_decision(forecast: Forecast, settings: AgentSettings | None = None) -> Decision:
    """Decide BUY/SELL/HOLD from the forecast via deterministic thresholds.

    ``BUY`` when the (optionally downside-penalised) cumulative return clears
    ``buy_threshold``, ``SELL`` at or below ``sell_threshold``, else ``HOLD``. A
    non-finite forecast (e.g. from a diverged checkpoint) fails safe to ``HOLD``.

    Parameters
    ----------
    forecast : Forecast
        The model forecast to act on.
    settings : AgentSettings, optional
        Thresholds; defaults to the cached :func:`~ophir.agent.config.get_settings`.

    Returns
    -------
    Decision
        The quant track's decision.
    """
    settings = settings or get_settings()

    if not math.isfinite(forecast.cum_return):
        decision = Decision(
            symbol=forecast.symbol,
            action="HOLD",
            confidence=0.0,
            rationale="non-finite forecast; defaulting to HOLD",
            source="quant",
            cum_return=forecast.cum_return,
            asof=forecast.asof,
        )
        _log(decision)
        return decision

    downside_mean = sum(forecast.downside) / len(forecast.downside) if forecast.downside else 0.0
    score = forecast.cum_return - settings.decision_downside_penalty * downside_mean

    action: Action
    if score >= settings.buy_threshold:
        action = "BUY"
    elif score <= settings.sell_threshold:
        action = "SELL"
    else:
        action = "HOLD"

    if settings.buy_threshold > 0:
        confidence = min(1.0, abs(score) / settings.buy_threshold)
    else:
        confidence = 0.0
    rationale = (
        f"90d predicted {forecast.cum_return:+.2%} (risk-adj score {score:+.2%}); "
        f"thresholds buy>={settings.buy_threshold:+.2%} / sell<={settings.sell_threshold:+.2%} "
        f"-> {action}"
    )
    decision = Decision(
        symbol=forecast.symbol,
        action=action,
        confidence=confidence,
        rationale=rationale,
        source="quant",
        cum_return=forecast.cum_return,
        asof=forecast.asof,
    )
    _log(decision)
    return decision


def _build_messages(forecast: Forecast) -> tuple[Any, Any]:
    """Build the grounded system + human messages for the Ollama track."""
    from langchain_core.messages import HumanMessage, SystemMessage

    system = SystemMessage(
        content=(
            "You are a disciplined paper-trading assistant. Decide BUY, SELL, or "
            "HOLD for the ticker using ONLY the model forecast numbers provided "
            "below -- never invent prices, news, or figures. Be conservative: when "
            "the signal is weak or unclear, prefer HOLD. Reply with ONLY a JSON "
            'object of the form {"action": "BUY|SELL|HOLD", "confidence": 0.0-1.0, '
            '"rationale": "one sentence"} and nothing else.'
        )
    )
    up = sum(forecast.upside) / len(forecast.upside) if forecast.upside else 0.0
    down = sum(forecast.downside) / len(forecast.downside) if forecast.downside else 0.0
    human = HumanMessage(
        content=(
            f"Ticker: {forecast.symbol}\n"
            f"As-of date: {forecast.asof}\n"
            f"Horizon: {forecast.horizon} trading days\n"
            f"Predicted cumulative return: {forecast.cum_return:+.4f} "
            f"({forecast.cum_return:+.2%})\n"
            f"Mean predicted intraday upside (log high/close): {up:+.4f}\n"
            f"Mean predicted intraday downside (log close/low): {down:+.4f}"
        )
    )
    return system, human


def _extract_json_object(text: str) -> str:
    """Return the first brace-balanced ``{...}`` substring of ``text``."""
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in LLM response")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("unbalanced JSON object in LLM response")


def _parse_verdict(text: str) -> LLMVerdict:
    """Parse and validate the LLM's JSON verdict (raises on malformed output)."""
    data = json.loads(_extract_json_object(text))
    if not isinstance(data, dict):
        raise ValueError("LLM did not return a JSON object")
    action_raw = data.get("action")
    if isinstance(action_raw, str):
        data["action"] = action_raw.strip().upper()
    return LLMVerdict(**data)


def ollama_reachable(settings: AgentSettings | None = None, *, timeout: float = 2.0) -> bool:
    """Best-effort check that an Ollama server is responding.

    Parameters
    ----------
    settings : AgentSettings, optional
        Source of ``ollama_base_url``; defaults to the cached settings.
    timeout : float, optional
        Seconds to wait for the probe. Defaults to ``2.0``.

    Returns
    -------
    bool
        ``True`` if ``<base_url>/api/tags`` responds, else ``False``.
    """
    settings = settings or get_settings()
    base = (settings.ollama_base_url or "http://localhost:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=timeout):
            return True
    except (urllib.error.URLError, OSError):
        return False


def ollama_decision(
    forecast: Forecast, *, llm: Any = None, settings: AgentSettings | None = None
) -> Decision:
    """Ask the local Ollama model for a BUY/SELL/HOLD call on the forecast.

    The model sees only the forecast numbers and must answer with a JSON verdict.
    Any failure -- no server, a malformed reply, an unknown action -- fails safe
    to ``HOLD`` so a hallucinating or unreachable LLM can never push a trade.

    Parameters
    ----------
    forecast : Forecast
        The model forecast to act on.
    llm : object, optional
        A chat model exposing ``.invoke(messages) -> obj.content``; defaults to a
        ``ChatOllama(model=settings.ollama_model, temperature=0)``. Inject a fake
        in tests to avoid needing a running Ollama server.
    settings : AgentSettings, optional
        Defaults to the cached :func:`~ophir.agent.config.get_settings`.

    Returns
    -------
    Decision
        The Ollama track's decision.
    """
    settings = settings or get_settings()
    if llm is None:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=settings.ollama_model,
            temperature=0,
            base_url=settings.ollama_base_url,
        )

    system, human = _build_messages(forecast)
    action: Action
    try:
        content = llm.invoke([system, human]).content
        verdict = _parse_verdict(content if isinstance(content, str) else str(content))
        action = verdict.action
        confidence = min(1.0, max(0.0, verdict.confidence))
        rationale = verdict.rationale or "(no rationale)"
    except Exception as exc:  # fail safe: never trade on a bad/unreachable LLM reply
        action = "HOLD"
        confidence = 0.0
        rationale = f"LLM verdict unavailable ({type(exc).__name__}); defaulting to HOLD"

    decision = Decision(
        symbol=forecast.symbol,
        action=action,
        confidence=confidence,
        rationale=rationale,
        source="ollama",
        cum_return=forecast.cum_return,
        asof=forecast.asof,
    )
    _log(decision)
    return decision


def compare_decisions(
    forecasts: list[Forecast], *, llm: Any = None, settings: AgentSettings | None = None
) -> list[DecisionComparison]:
    """Run both tracks on each forecast and pair them up with an agreement flag.

    Constructs one shared Ollama client when ``llm`` is omitted, so a list of
    forecasts reuses a single connection.
    """
    settings = settings or get_settings()
    if llm is None:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=settings.ollama_model,
            temperature=0,
            base_url=settings.ollama_base_url,
        )

    comparisons: list[DecisionComparison] = []
    for forecast in forecasts:
        quant = quant_decision(forecast, settings=settings)
        ollama = ollama_decision(forecast, llm=llm, settings=settings)
        comparisons.append(
            DecisionComparison(
                symbol=forecast.symbol,
                quant=quant,
                ollama=ollama,
                agree=quant.action == ollama.action,
            )
        )
    return comparisons
