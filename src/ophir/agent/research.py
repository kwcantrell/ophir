"""Per-ticker research briefs: grounded data + a cited LLM synthesis.

For each ticker we gather three dimensions of *real* data deterministically --
**fundamentals** (Yahoo Finance ``.info``), **news** (Yahoo Finance ``.news``),
and **technicals** (ophir's computed features plus the model
:class:`~ophir.agent.predict.Forecast`) -- and then ask the local ``gpt-oss:20b``
model to summarize *only that data* into a structured brief. The model never
fetches or invents numbers; if it is unreachable or replies badly the brief
falls back to a neutral analysis while keeping the grounded data intact.

Per the ``trading-best-practices`` skill: ground every claim, no fabricated
figures, fail safe, and audit every brief.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from ophir.agent import audit
from ophir.agent.config import get_settings
from ophir.agent.decide import _extract_json_object
from ophir.agent.feed import load_history

if TYPE_CHECKING:
    from ophir.agent.config import AgentSettings
    from ophir.agent.predict import Forecast

Stance = Literal["bullish", "neutral", "bearish"]

_FUNDAMENTAL_KEYS = [
    "sector",
    "industry",
    "marketCap",
    "trailingPE",
    "forwardPE",
    "profitMargins",
    "beta",
    "dividendYield",
    "fiftyTwoWeekHigh",
    "fiftyTwoWeekLow",
    "currentPrice",
]


class ResearchAnalysis(BaseModel):
    """The LLM's validated synthesis of the grounded research data."""

    fundamentals_summary: str = ""
    news_summary: str = ""
    technicals_summary: str = ""
    overall_stance: Stance = "neutral"
    overall_summary: str = ""


@dataclass(frozen=True, slots=True)
class ResearchBrief:
    """Grounded research data for one ticker plus the LLM analysis.

    Attributes
    ----------
    symbol : str
        Ticker symbol.
    asof : str
        ISO date the data is conditioned on.
    fundamentals : dict
        Deterministically fetched fundamentals (Yahoo Finance ``.info`` subset).
    news : list[dict]
        Recent headlines (``title`` / ``publisher`` / ``link`` / ``published``).
    technicals : dict
        ophir-computed indicators plus the model forecast (when supplied).
    analysis : ResearchAnalysis
        The LLM's summary; a neutral default when ``llm_ok`` is ``False``.
    sources : list[str]
        Provenance of every datum in the brief.
    llm_ok : bool
        ``True`` when the LLM synthesis succeeded.
    """

    symbol: str
    asof: str
    fundamentals: dict[str, Any]
    news: list[dict[str, Any]]
    technicals: dict[str, Any]
    analysis: ResearchAnalysis
    sources: list[str] = field(default_factory=list)
    llm_ok: bool = True


def _num(value: Any) -> float | None:
    """Coerce to a finite float, or ``None`` if missing / non-finite."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and out not in (float("inf"), float("-inf")) else None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def gather_fundamentals(symbol: str) -> dict[str, Any]:
    """Fetch a curated subset of Yahoo Finance fundamentals (best-effort)."""
    import yfinance as yf

    try:
        info = yf.Ticker(symbol).info or {}
    except Exception as exc:  # network / parse failure -> grounded-but-empty
        return {"error": f"fundamentals unavailable ({type(exc).__name__})"}
    return {key: info.get(key) for key in _FUNDAMENTAL_KEYS}


def _normalize_news_item(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize a yfinance news item across its old (flat) and new (nested) shapes."""
    content = item.get("content")
    if isinstance(content, dict):  # newer nested shape
        provider = content.get("provider")
        publisher = provider.get("displayName", "") if isinstance(provider, dict) else ""
        url_obj = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
        link = url_obj.get("url", "") if isinstance(url_obj, dict) else ""
        return {
            "title": content.get("title", ""),
            "publisher": publisher,
            "link": link,
            "published": str(content.get("pubDate", "")),
        }
    ts = item.get("providerPublishTime")  # older flat shape
    published = ""
    if isinstance(ts, (int, float)):
        published = time.strftime("%Y-%m-%d", time.gmtime(ts))
    return {
        "title": item.get("title", ""),
        "publisher": item.get("publisher", ""),
        "link": item.get("link", ""),
        "published": published,
    }


def gather_news(symbol: str, limit: int = 8) -> list[dict[str, Any]]:
    """Fetch recent Yahoo Finance headlines (best-effort, normalized)."""
    import yfinance as yf

    try:
        raw = yf.Ticker(symbol).news or []
    except Exception:  # network / parse failure -> no news
        return []
    return [_normalize_news_item(item) for item in raw[:limit] if isinstance(item, dict)]


def gather_technicals(
    symbol: str, forecast: Forecast | None = None, *, stocks_dir: str | None = None
) -> dict[str, Any]:
    """Compute grounded technical indicators from ingested OHLC (+ the forecast)."""
    from ophir.ticker import extract_features

    df = load_history(symbol, stocks_dir=stocks_dir)
    feats = extract_features(df)
    real = feats[feats["trade_occured"]]
    last = real.iloc[-1]
    close = df["close"]
    recent = close.iloc[-252:]
    last_close = float(close.iloc[-1])
    tech: dict[str, Any] = {
        "last_close": round(last_close, 4),
        "recent_return_1d": _num(last["r_close"]),
        "vol_20d": _num(last["20_volatility"]),
        "vol_60d": _num(last["60_volatility"]),
        "pct_of_52w_high": round(last_close / float(recent.max()), 4),
        "pct_above_52w_low": round(last_close / float(recent.min()), 4),
    }
    if forecast is not None:
        tech["forecast_cum_return"] = round(forecast.cum_return, 5)
        tech["forecast_mean_upside"] = round(_mean(forecast.upside), 5)
        tech["forecast_mean_downside"] = round(_mean(forecast.downside), 5)
    return tech


def _build_messages(
    symbol: str,
    asof: str,
    fundamentals: dict[str, Any],
    news: list[dict[str, Any]],
    technicals: dict[str, Any],
) -> tuple[Any, Any]:
    """Build the grounded system + human messages for the synthesis call."""
    from langchain_core.messages import HumanMessage, SystemMessage

    system = SystemMessage(
        content=(
            "You are an equity research assistant. Summarize ONLY the data provided "
            "below -- never invent prices, ratios, headlines, or figures, and do not "
            "use outside knowledge. Base your stance solely on the provided "
            "fundamentals, news headlines, and technicals. Reply with ONLY a JSON "
            'object of the form {"fundamentals_summary": "...", "news_summary": "...", '
            '"technicals_summary": "...", "overall_stance": "bullish|neutral|bearish", '
            '"overall_summary": "..."} and nothing else.'
        )
    )
    news_lines = (
        "\n".join(
            f"  - {n['title']} ({n['publisher']}, {n['published']})" for n in news if n["title"]
        )
        or "  (no recent headlines)"
    )
    human = HumanMessage(
        content=(
            f"Ticker: {symbol}\nAs-of date: {asof}\n\n"
            f"FUNDAMENTALS:\n{json.dumps(fundamentals, indent=2, default=str)}\n\n"
            f"NEWS HEADLINES:\n{news_lines}\n\n"
            f"TECHNICALS:\n{json.dumps(technicals, indent=2, default=str)}"
        )
    )
    return system, human


def _synthesize(
    symbol: str,
    asof: str,
    fundamentals: dict[str, Any],
    news: list[dict[str, Any]],
    technicals: dict[str, Any],
    llm: Any,
) -> tuple[ResearchAnalysis, bool]:
    """Call the LLM to summarize the grounded data; fail safe to neutral."""
    system, human = _build_messages(symbol, asof, fundamentals, news, technicals)
    try:
        content = llm.invoke([system, human]).content
        text = content if isinstance(content, str) else str(content)
        data = json.loads(_extract_json_object(text))
        if not isinstance(data, dict):
            raise ValueError("LLM did not return a JSON object")
        stance = data.get("overall_stance")
        if isinstance(stance, str):
            data["overall_stance"] = stance.strip().lower()
        return ResearchAnalysis(**data), True
    except Exception as exc:  # fail safe: keep the grounded data, neutral analysis
        note = f"synthesis unavailable ({type(exc).__name__})"
        return (
            ResearchAnalysis(
                fundamentals_summary=note,
                news_summary=note,
                technicals_summary=note,
                overall_stance="neutral",
                overall_summary=note,
            ),
            False,
        )


def research_ticker(
    symbol: str,
    *,
    forecast: Forecast | None = None,
    llm: Any = None,
    settings: AgentSettings | None = None,
    stocks_dir: str | None = None,
) -> ResearchBrief:
    """Build a grounded research brief for one ticker.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    forecast : Forecast, optional
        The model forecast; adds forecast fields to the technicals dimension.
    llm : object, optional
        A chat model exposing ``.invoke(messages) -> obj.content``; defaults to a
        ``ChatOllama(model=settings.ollama_model, temperature=0)``. Inject a fake
        in tests.
    settings : AgentSettings, optional
        Defaults to the cached :func:`~ophir.agent.config.get_settings`.
    stocks_dir : str, optional
        Override for the parquet root.

    Returns
    -------
    ResearchBrief
        Grounded data plus the (possibly fail-safe) LLM analysis.
    """
    settings = settings or get_settings()
    symbol = symbol.upper().strip()

    fundamentals = gather_fundamentals(symbol)
    news = gather_news(symbol, limit=settings.research_news_limit)
    technicals = gather_technicals(symbol, forecast, stocks_dir=stocks_dir)
    asof = (
        forecast.asof
        if forecast is not None
        else str(load_history(symbol, stocks_dir=stocks_dir).index.max().date())
    )

    if llm is None:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=settings.ollama_model,
            temperature=0,
            format="json",
            base_url=settings.ollama_base_url,
            num_ctx=settings.ollama_num_ctx,
        )

    analysis, llm_ok = _synthesize(symbol, asof, fundamentals, news, technicals, llm)

    sources = ["yfinance:info"]
    sources += [n["link"] for n in news if n["link"]]
    sources.append("ophir:features")
    if forecast is not None:
        sources.append("ophir:forecast")

    brief = ResearchBrief(
        symbol=symbol,
        asof=asof,
        fundamentals=fundamentals,
        news=news,
        technicals=technicals,
        analysis=analysis,
        sources=sources,
        llm_ok=llm_ok,
    )
    audit.log_event(
        "research", symbol=symbol, stance=analysis.overall_stance, llm_ok=llm_ok, n_news=len(news)
    )
    return brief


def research_many(
    symbols: list[str],
    *,
    forecasts: list[Forecast] | None = None,
    llm: Any = None,
    settings: AgentSettings | None = None,
    stocks_dir: str | None = None,
) -> list[ResearchBrief]:
    """Build briefs for several tickers; failures are logged and skipped."""
    settings = settings or get_settings()
    by_symbol = {f.symbol.upper(): f for f in forecasts} if forecasts else {}

    if llm is None:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=settings.ollama_model,
            temperature=0,
            format="json",
            base_url=settings.ollama_base_url,
            num_ctx=settings.ollama_num_ctx,
        )

    briefs: list[ResearchBrief] = []
    for symbol in symbols:
        try:
            briefs.append(
                research_ticker(
                    symbol,
                    forecast=by_symbol.get(symbol.upper().strip()),
                    llm=llm,
                    settings=settings,
                    stocks_dir=stocks_dir,
                )
            )
        except (ValueError, FileNotFoundError, OSError) as exc:
            audit.log_event("research_failed", symbol=symbol, error=str(exc))
            print(f"[research] {symbol}: FAILED -- {exc}")
    return briefs
