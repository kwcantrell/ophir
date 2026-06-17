"""Per-stock dossier reports: one Markdown file per considered ticker, plus an index.

At the end of a run (``ophir trade`` / ``ophir manage``) this serializes the full
ensemble the agent weighed for each candidate -- model forecast, quant + Ollama
decisions, grounded research, the bull and bear theses, and the manager's
accept/reject decision -- into a human-readable ``<SYMBOL>.md`` under a dated
directory, with an ``INDEX.md`` roll-up linking them. Everything written is already
in memory at the end of the pipeline; nothing is recomputed or re-fetched.

Output goes to ``<report_dir>/<asof>/`` (``report_dir`` from ``AGENT_REPORT_DIR``,
defaulting to ``<DATA_DIR>/reports``). Re-running the same day overwrites that day's
files.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from ophir.agent import audit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ophir.agent.config import AgentSettings
    from ophir.agent.debate import Thesis
    from ophir.agent.execute import Order
    from ophir.agent.manage import Candidate, Portfolio, Position

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def write_reports(
    candidates: Sequence[Candidate],
    portfolio: Portfolio,
    orders: Sequence[Order],
    plan: Sequence[str],
    *,
    settings: AgentSettings | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Write one ``<SYMBOL>.md`` per candidate plus ``INDEX.md``; return the dated dir.

    Parameters
    ----------
    candidates : sequence of Candidate
        Every ticker that got the full ensemble (the "considered" names).
    portfolio : Portfolio
        The gated target portfolio -- its ``positions`` are the manager's final picks.
    orders : sequence of Order
        Reconciled orders (empty for a decide-only run).
    plan : sequence of str
        Human-readable order plan lines (empty for a decide-only run).
    settings : AgentSettings, optional
        Defaults to the cached :func:`~ophir.agent.config.get_settings`.
    out_dir : pathlib.Path, optional
        Base directory override (mainly for tests); bypasses ``report_dir`` / DATA_DIR.

    Returns
    -------
    pathlib.Path
        The dated directory the files were written to.
    """
    if settings is None:
        from ophir.agent.config import get_settings

        settings = get_settings()

    target = _resolve_dir(portfolio, settings, out_dir)
    positions = {p.symbol.upper(): p for p in portfolio.positions}
    for cand in candidates:
        pos = positions.get(cand.symbol.upper())
        body = _render_stock(cand, portfolio, pos, orders)
        (target / f"{_safe_stem(cand.symbol)}.md").write_text(body, encoding="utf-8", newline="\n")

    (target / "INDEX.md").write_text(
        _render_index(candidates, portfolio, plan), encoding="utf-8", newline="\n"
    )
    audit.log_event(
        "reports_written",
        asof=portfolio.asof,
        dir=str(target),
        n_stocks=len(candidates),
        n_selected=len(portfolio.positions),
        halted=portfolio.halted,
    )
    return target


def _base_report_dir(settings: AgentSettings) -> Path:
    """Resolve the base reports directory (``report_dir`` or ``<DATA_DIR>/reports``)."""
    if settings.report_dir:
        return Path(settings.report_dir)
    from ophir.register import DATA_DIR

    return Path(DATA_DIR) / "reports"


def _resolve_dir(portfolio: Portfolio, settings: AgentSettings, out_dir: Path | None) -> Path:
    """Create and return ``<base>/<asof>`` (today's date if ``asof`` is empty)."""
    base = out_dir if out_dir is not None else _base_report_dir(settings)
    day = _safe_stem(portfolio.asof) if portfolio.asof else date.today().isoformat()
    target = base / day
    target.mkdir(parents=True, exist_ok=True)
    return target


def _safe_stem(symbol: str) -> str:
    """A filesystem-safe, upper-cased file stem for a ticker symbol."""
    cleaned = _UNSAFE.sub("_", symbol.strip()).strip("._")
    return cleaned.upper() or "UNKNOWN"


def _cell(text: str) -> str:
    """Escape a string for use inside a Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ").strip() or "—"


def _notes_for(symbol: str, portfolio: Portfolio) -> list[str]:
    """Risk-gate notes that name ``symbol`` (word-boundary match, no false hits)."""
    pat = re.compile(rf"\b{re.escape(symbol.upper())}\b")
    return [n for n in portfolio.gate_notes if pat.search(n.upper())]


def _orders_for(symbol: str, orders: Sequence[Order]) -> list[Order]:
    """Orders touching ``symbol`` this run."""
    return [o for o in orders if o.symbol.upper() == symbol.upper()]


def _manager_block(symbol: str, portfolio: Portfolio, pos: Position | None) -> str:
    """Render the manager's accept/reject decision for one name."""
    lines = ["## Manager decision"]
    if portfolio.halted:
        lines.append(
            "> ⚠ Portfolio **HALTED** by the risk gate — holding all cash. No name selected."
        )
    if pos is not None:
        lines.append(f"**SELECTED** — weight {pos.weight:.2%}, conviction {pos.conviction:.2f}")
        if pos.rationale.strip():
            lines.append(f"> {pos.rationale.strip()}")
    else:
        lines.append("**NOT SELECTED** — did not make the final book.")
        notes = _notes_for(symbol, portfolio)
        if notes:
            lines.append("\nRisk-gate notes:")
            lines += [f"- {n}" for n in notes]
        if portfolio.rationale.strip():
            lines.append(f"\nManager rationale: {portfolio.rationale.strip()}")
    return "\n".join(lines)


def _thesis_block(label: str, thesis: Thesis, llm_ok: bool) -> str:
    """Render one side of the debate (bull or bear)."""
    lines = [f"## {label} (strength {thesis.stance_strength:.0%})"]
    summary = thesis.summary.strip()
    points = [p.strip() for p in thesis.key_points if p.strip()]
    risks = [r.strip() for r in thesis.key_risks if r.strip()]
    if not summary and not points and not risks:
        lines.append("_(no thesis — LLM unavailable)_" if not llm_ok else "_(no thesis returned)_")
        return "\n".join(lines)
    if summary:
        lines.append(summary)
    if points:
        lines.append("\n**Key points**")
        lines += [f"- {p}" for p in points]
    if risks:
        lines.append("\n**Key risks**")
        lines += [f"- {r}" for r in risks]
    return "\n".join(lines)


def _render_stock(
    cand: Candidate, portfolio: Portfolio, pos: Position | None, orders: Sequence[Order]
) -> str:
    """Render the full per-stock dossier Markdown body."""
    fc, dec, brief, deb = cand.forecast, cand.decision, cand.brief, cand.debate
    parts = [f"# {cand.symbol} — {fc.asof}", _manager_block(cand.symbol, portfolio, pos)]

    parts.append(
        "## Model forecast\n"
        f"- Cumulative return (horizon {fc.horizon}d): {fc.cum_return:+.2%}\n"
        f"- Score: {fc.score:.4f}\n"
        f"- As-of (last closed session): {fc.asof}"
    )

    rows = "\n".join(
        f"| {d.source} | {d.action} | {d.confidence:.0%} | {_cell(d.rationale)} |"
        for d in (dec.quant, dec.ollama)
    )
    parts.append(
        "## Decisions\n"
        "| Track | Action | Confidence | Rationale |\n"
        "| --- | --- | --- | --- |\n"
        f"{rows}\n\nAgreement: {'agree' if dec.agree else 'disagree'}"
    )

    summary = brief.analysis.overall_summary.strip()
    if not summary:
        summary = "_(no analysis — LLM unavailable)_" if not brief.llm_ok else "_(no analysis)_"
    parts.append(f"## Research\n**Stance:** {brief.analysis.overall_stance}\n{summary}")

    parts.append(_thesis_block("Bull thesis", deb.bull, deb.llm_ok))
    parts.append(_thesis_block("Bear thesis", deb.bear, deb.llm_ok))

    sym_orders = _orders_for(cand.symbol, orders)
    if sym_orders:
        order_lines = "\n".join(f"- {o.side.upper()} ${o.notional:,.2f}" for o in sym_orders)
        parts.append(f"## Orders this run\n{order_lines}")

    return "\n\n".join(parts) + "\n"


def _render_index(
    candidates: Sequence[Candidate], portfolio: Portfolio, plan: Sequence[str]
) -> str:
    """Render the daily roll-up index linking every per-stock dossier."""
    asof = portfolio.asof or date.today().isoformat()
    halt = " · **HALTED (all cash)**" if portfolio.halted else ""
    llm = "LLM ok" if portfolio.llm_ok else "LLM fail-safe"
    sections = [
        f"# Daily report — {asof}\n\n"
        f"Considered {len(candidates)} · selected {len(portfolio.positions)} "
        f"· gross {portfolio.gross_exposure:.2%} · cash {portfolio.cash_weight:.2%} · {llm}{halt}"
    ]
    by_sym = {c.symbol.upper(): c for c in candidates}
    selected = {p.symbol.upper() for p in portfolio.positions}

    sel = ["## Selected"]
    if portfolio.positions:
        sel.append("| Symbol | Weight | Conv | Cum ret | Quant | Ollama | Stance |")
        sel.append("| --- | --- | --- | --- | --- | --- | --- |")
        for p in portfolio.positions:
            c = by_sym.get(p.symbol.upper())
            cum = f"{c.forecast.cum_return:+.2%}" if c else "—"
            q = c.decision.quant.action if c else "—"
            o = c.decision.ollama.action if c else "—"
            st = c.brief.analysis.overall_stance if c else "—"
            sel.append(
                f"| [{p.symbol}]({_safe_stem(p.symbol)}.md) | {p.weight:.2%} | {p.conviction:.2f} "
                f"| {cum} | {q} | {o} | {st} |"
            )
    else:
        sel.append("_None._")
    sections.append("\n".join(sel))

    not_sel = [c for c in candidates if c.symbol.upper() not in selected]
    miss = ["## Considered but not selected"]
    if not_sel:
        miss.append("| Symbol | Cum ret | Quant | Ollama | Stance |")
        miss.append("| --- | --- | --- | --- | --- |")
        for c in not_sel:
            miss.append(
                f"| [{c.symbol}]({_safe_stem(c.symbol)}.md) | {c.forecast.cum_return:+.2%} "
                f"| {c.decision.quant.action} | {c.decision.ollama.action} "
                f"| {c.brief.analysis.overall_stance} |"
            )
    else:
        miss.append("_None._")
    sections.append("\n".join(miss))

    if portfolio.rationale.strip():
        sections.append(f"## Manager rationale\n> {portfolio.rationale.strip()}")
    if portfolio.gate_notes:
        sections.append("## Risk-gate notes\n" + "\n".join(f"- {n}" for n in portfolio.gate_notes))
    if plan:
        sections.append("## Orders\n```\n" + "\n".join(plan) + "\n```")
    else:
        sections.append("## Orders\n_No orders (decide-only run or nothing to trade)._")

    return "\n\n".join(sections) + "\n"
