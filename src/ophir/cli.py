"""The ``ophir`` command-line interface.

This module builds the Typer application exposed as the ``ophir`` console
script (entry point ``ophir.cli:app``). It mounts the :mod:`ophir.register`
sub-application under ``register`` and adds the top-level ``serve`` command.
"""

import sys

import typer

from ophir import register
from ophir import train as train_cmd

# Model output can contain Unicode (curly quotes, non-breaking hyphens, ...) that a
# console's default encoding (e.g. Windows cp1252) cannot represent. Reconfigure the
# streams to UTF-8 with replacement so ``typer.echo`` never crashes with a
# UnicodeEncodeError when printing LLM-generated text.
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        _reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(help="Ophir CLI")
app.add_typer(register.app, name="register")
app.command(name="train")(train_cmd.train)


@app.command()
def serve(
    port: int = typer.Option(7860, help="Gradio server port"),
    share: bool = typer.Option(False, help="Expose a public share link"),
    debug: bool = typer.Option(True, help="Launch Gradio in debug mode"),
) -> None:
    """Launch the Ophir Gradio UI.

    Imports :mod:`ophir.ui` lazily and starts the Gradio server. Importing
    ``ophir.ui`` fetches reference data from the network and loads a base
    checkpoint onto a CUDA device, so a GPU, a checkpoint, network access, and
    a local Ollama server are required.

    Parameters
    ----------
    port : int, optional
        Gradio server port. Defaults to ``7860``.
    share : bool, optional
        If ``True``, expose a public Gradio share link. Defaults to ``False``.
    debug : bool, optional
        If ``True``, launch Gradio in debug mode. Defaults to ``True``.
    """
    from ophir import ui

    ui.serve(port=port, share=share, debug=debug)


@app.command()
def ingest(
    symbol: str,
    days: int = typer.Option(730, help="Calendar days of Yahoo Finance history to fetch."),
) -> None:
    """Ingest a ticker's daily OHLC from Yahoo Finance into a model-ready dataset.

    Pulls ``days`` of history (default ~2 years, enough for the model's 365-day
    window plus rolling-feature warmup) and writes
    ``<DATA_DIR>/days/stocks/symbol=<SYMBOL>/data.parquet``. Reuses
    :func:`ophir.ticker.extract_features` downstream; no GPU required.

    Parameters
    ----------
    symbol : str
        Ticker symbol to ingest (e.g. ``AAPL``).
    days : int, optional
        Calendar days of history to fetch. Defaults to ``730``.
    """
    from ophir.agent.ingest import ingest as ingest_ticker

    ingest_ticker(symbol, days=days)


@app.command()
def predict(symbol: str) -> None:
    """Forecast a ticker's next 90 days with the trained model.

    Ingests the ticker if needed, runs the base checkpoint over the latest
    365-day window, and prints the predicted cumulative return. Requires a CUDA
    GPU and a trained checkpoint.

    Parameters
    ----------
    symbol : str
        Ticker symbol to forecast (e.g. ``AAPL``).
    """
    from ophir.agent.predict import predict_ticker

    fc = predict_ticker(symbol)
    typer.echo(
        f"{fc.symbol} asof {fc.asof}: {fc.horizon}-day predicted return {fc.cum_return:+.2%}"
    )


@app.command()
def rank(
    symbols: list[str],
    top_k: int = typer.Option(5, help="Number of top picks to show."),
) -> None:
    """Forecast several tickers and print the top picks by predicted return.

    Requires a CUDA GPU and a trained checkpoint.

    Parameters
    ----------
    symbols : list[str]
        Ticker symbols to forecast and rank.
    top_k : int, optional
        How many top picks to print. Defaults to ``5``.
    """
    from ophir.agent.predict import predict_many
    from ophir.agent.predict import rank as rank_forecasts

    picks = rank_forecasts(predict_many(symbols), top_k=top_k)
    for i, fc in enumerate(picks, start=1):
        typer.echo(f"{i}. {fc.symbol}  {fc.horizon}d {fc.cum_return:+.2%}")


@app.command()
def decide(
    symbols: list[str],
    track: str = typer.Option(
        "both", help="Which decision track(s) to run: 'both', 'quant', or 'ollama'."
    ),
    top_k: int = typer.Option(5, help="Decide on at most this many forecasts."),
) -> None:
    """Turn ticker forecasts into buy/sell/hold decisions and compare two tracks.

    Forecasts each symbol, then runs a deterministic quant rule and the local
    Ollama model over each forecast. ``--track both`` (default) prints them side
    by side with an agreement flag; ``quant`` / ``ollama`` print a single track
    with its rationale. Requires a CUDA GPU and a trained checkpoint; the Ollama
    track also needs a local Ollama server.

    Parameters
    ----------
    symbols : list[str]
        Ticker symbols to decide on.
    track : str, optional
        ``both``, ``quant``, or ``ollama``. Defaults to ``both``.
    top_k : int, optional
        Decide on at most this many forecasts. Defaults to ``5``.
    """
    from ophir.agent.decide import (
        compare_decisions,
        ollama_decision,
        ollama_reachable,
        quant_decision,
    )
    from ophir.agent.predict import predict_many
    from ophir.agent.predict import rank as rank_forecasts

    if track not in {"both", "quant", "ollama"}:
        raise typer.BadParameter("track must be one of: both, quant, ollama")

    if track in {"both", "ollama"} and not ollama_reachable():
        typer.echo(
            "[warning] Ollama unreachable -- the LLM track will HOLD every ticker. "
            "Is the server running with the model pulled? (docs: Setting up Ollama)"
        )

    forecasts = rank_forecasts(predict_many(symbols), top_k=top_k)

    if track == "both":
        for comp in compare_decisions(forecasts):
            flag = "agree" if comp.agree else "DIFFER"
            typer.echo(
                f"{comp.symbol:<6} "
                f"quant={comp.quant.action:<4} ({comp.quant.confidence:.0%})  "
                f"ollama={comp.ollama.action:<4} ({comp.ollama.confidence:.0%})  [{flag}]"
            )
    elif track == "quant":
        for fc in forecasts:
            quant = quant_decision(fc)
            typer.echo(
                f"{quant.symbol:<6} {quant.action:<4} ({quant.confidence:.0%})  {quant.rationale}"
            )
    else:
        for fc in forecasts:
            ollama = ollama_decision(fc)
            typer.echo(
                f"{ollama.symbol:<6} {ollama.action:<4} ({ollama.confidence:.0%})  "
                f"{ollama.rationale}"
            )


@app.command()
def research(
    symbols: list[str],
    top_k: int = typer.Option(5, help="Research at most this many top-ranked tickers."),
) -> None:
    """Build grounded research briefs for the top-ranked tickers.

    Ranks the symbols by model forecast, then for each top pick gathers
    fundamentals (Yahoo Finance), recent news, and technicals, and asks the local
    Ollama model to summarize that data into a cited brief. Requires a CUDA GPU
    and a trained checkpoint; the LLM summaries also need a local Ollama server
    (without it the grounded data is still returned).

    Parameters
    ----------
    symbols : list[str]
        Ticker symbols to research.
    top_k : int, optional
        Research at most this many top-ranked tickers. Defaults to ``5``.
    """
    from ophir.agent.decide import ollama_reachable
    from ophir.agent.predict import predict_many
    from ophir.agent.predict import rank as rank_forecasts
    from ophir.agent.research import research_many

    if not ollama_reachable():
        typer.echo(
            "[warning] Ollama unreachable -- briefs will return grounded data with no LLM "
            "summary. Is the server running with the model pulled? (docs: Setting up Ollama)"
        )

    forecasts = rank_forecasts(predict_many(symbols), top_k=top_k)
    for brief in research_many([fc.symbol for fc in forecasts], forecasts=forecasts):
        typer.echo(
            f"\n=== {brief.symbol}  [{brief.analysis.overall_stance}]  asof {brief.asof} ==="
        )
        typer.echo(f"  {brief.analysis.overall_summary}")
        typer.echo(f"  fundamentals: {brief.analysis.fundamentals_summary}")
        typer.echo(f"  news:         {brief.analysis.news_summary}")
        typer.echo(f"  technicals:   {brief.analysis.technicals_summary}")


@app.command()
def debate(
    symbols: list[str],
    top_k: int = typer.Option(5, help="Debate at most this many top-ranked tickers."),
) -> None:
    """Argue a bull and a bear thesis for each top-ranked ticker.

    Ranks the symbols by model forecast, builds a research brief per top pick, then
    has the local Ollama model argue the strongest bullish and bearish cases from
    that brief. Requires a CUDA GPU and a trained checkpoint; the theses also need a
    local Ollama server.

    Parameters
    ----------
    symbols : list[str]
        Ticker symbols to debate.
    top_k : int, optional
        Debate at most this many top-ranked tickers. Defaults to ``5``.
    """
    from ophir.agent.debate import debate_many
    from ophir.agent.decide import ollama_reachable
    from ophir.agent.predict import predict_many
    from ophir.agent.predict import rank as rank_forecasts
    from ophir.agent.research import research_many

    if not ollama_reachable():
        typer.echo(
            "[warning] Ollama unreachable -- theses will be neutral/empty. Is the server "
            "running with the model pulled? (docs: Setting up Ollama)"
        )

    forecasts = rank_forecasts(predict_many(symbols), top_k=top_k)
    briefs = research_many([fc.symbol for fc in forecasts], forecasts=forecasts)
    for d in debate_many(briefs):
        typer.echo(f"\n=== {d.symbol}  asof {d.asof} ===")
        typer.echo(f"  BULL ({d.bull.stance_strength:.0%}): {d.bull.summary}")
        for point in d.bull.key_points:
            typer.echo(f"    + {point}")
        typer.echo(f"  BEAR ({d.bear.stance_strength:.0%}): {d.bear.summary}")
        for point in d.bear.key_points:
            typer.echo(f"    - {point}")


@app.command()
def manage(
    symbols: list[str],
    top_k: int = typer.Option(5, help="Consider at most this many top-ranked tickers."),
) -> None:
    """Build a gated target portfolio from the full agent ensemble.

    Ranks the symbols by model forecast, runs the decision, research, and debate
    layers per top pick, then a manager LLM chooses and ranks the names (with a
    conviction each). Deterministic code sizes the convictions and a risk gate
    enforces the per-name / exposure caps and kill-switch. Requires a CUDA GPU and
    a trained checkpoint; the manager also needs a local Ollama server.

    Parameters
    ----------
    symbols : list[str]
        Ticker symbols to consider.
    top_k : int, optional
        Consider at most this many top-ranked tickers. Defaults to ``5``.
    """
    from ophir.agent.debate import debate_many
    from ophir.agent.decide import compare_decisions, ollama_reachable
    from ophir.agent.manage import Candidate
    from ophir.agent.manage import manage as run_manage
    from ophir.agent.predict import predict_many
    from ophir.agent.predict import rank as rank_forecasts
    from ophir.agent.report import write_reports
    from ophir.agent.research import research_many

    if not ollama_reachable():
        typer.echo(
            "[warning] Ollama unreachable -- the manager will return an all-cash portfolio. "
            "Is the server running with the model pulled? (docs: Setting up Ollama)"
        )

    forecasts = rank_forecasts(predict_many(symbols), top_k=top_k)
    comparisons = {c.symbol: c for c in compare_decisions(forecasts)}
    briefs = {
        b.symbol: b for b in research_many([fc.symbol for fc in forecasts], forecasts=forecasts)
    }
    debates = {d.symbol: d for d in debate_many(list(briefs.values()))}

    candidates = [
        Candidate(
            symbol=fc.symbol,
            forecast=fc,
            decision=comparisons[fc.symbol],
            brief=briefs[fc.symbol],
            debate=debates[fc.symbol],
        )
        for fc in forecasts
        if fc.symbol in comparisons and fc.symbol in briefs and fc.symbol in debates
    ]
    portfolio = run_manage(candidates)
    report_path = write_reports(candidates, portfolio, [], [])

    typer.echo(f"\n=== Target portfolio  asof {portfolio.asof} ===")
    if portfolio.halted:
        typer.echo("  *** HALTED by risk gate -- all cash ***")
    for pos in portfolio.positions:
        typer.echo(
            f"  {pos.symbol:<6} {pos.weight:6.2%}  conv={pos.conviction:.2f}  {pos.rationale}"
        )
    typer.echo(f"  cash {portfolio.cash_weight:.2%}  |  gross {portfolio.gross_exposure:.2%}")
    if portfolio.rationale:
        typer.echo(f"  manager: {portfolio.rationale}")
    for note in portfolio.gate_notes:
        typer.echo(f"  [gate] {note}")
    typer.echo(f"  per-stock dossiers written to {report_path}")


@app.command()
def trade(
    symbols: list[str],
    top_k: int = typer.Option(5, help="Consider at most this many top-ranked tickers."),
    broker: str = typer.Option(
        "paper", help="Broker: 'paper' (in-process simulator) or 'alpaca' (paper account)."
    ),
    execute: bool = typer.Option(
        False, "--execute/--dry-run", help="Submit orders. Default is dry-run (plan only)."
    ),
) -> None:
    """Reconcile the target portfolio into paper orders and report (dry-run by default).

    Builds the target portfolio from the full ensemble, reconciles it against the
    chosen broker's account into delta orders, and -- unless ``--execute`` is passed
    -- only prints the plan without submitting. The broker's account drawdown / daily
    loss feed the risk-gate kill-switch. Requires a CUDA GPU and a trained checkpoint;
    ``--broker alpaca`` needs ``AGENT_ALPACA_*`` credentials.

    Parameters
    ----------
    symbols : list[str]
        Ticker symbols to consider.
    top_k : int, optional
        Consider at most this many top-ranked tickers. Defaults to ``5``.
    broker : str, optional
        ``paper`` (default, in-process simulator) or ``alpaca`` (real paper account).
    execute : bool, optional
        Submit orders instead of only planning them. Defaults to ``False`` (dry-run).
    """
    from ophir.agent import audit
    from ophir.agent.debate import debate_many
    from ophir.agent.decide import compare_decisions, ollama_reachable
    from ophir.agent.execute import (
        AlpacaPaperBroker,
        PaperBroker,
        daily_report,
        place_orders,
        reconcile,
    )
    from ophir.agent.manage import Candidate
    from ophir.agent.manage import manage as run_manage
    from ophir.agent.market_calendar import is_trading_day
    from ophir.agent.predict import predict_many
    from ophir.agent.predict import rank as rank_forecasts
    from ophir.agent.report import write_reports
    from ophir.agent.research import research_many

    if broker not in {"paper", "alpaca"}:
        raise typer.BadParameter("broker must be 'paper' or 'alpaca'")
    if not is_trading_day():
        audit.log_event("cycle_skipped", reason="not_a_trading_day")
        typer.echo("NYSE is closed today -- skipping the cycle, no orders placed.")
        return
    if not ollama_reachable():
        typer.echo(
            "[warning] Ollama unreachable -- the manager will hold all cash and the report "
            "will be templated. (docs: Setting up Ollama)"
        )

    account_broker = AlpacaPaperBroker() if broker == "alpaca" else PaperBroker()
    account = account_broker.get_account()

    forecasts = rank_forecasts(predict_many(symbols), top_k=top_k)
    comparisons = {c.symbol: c for c in compare_decisions(forecasts)}
    briefs = {
        b.symbol: b for b in research_many([fc.symbol for fc in forecasts], forecasts=forecasts)
    }
    debates = {d.symbol: d for d in debate_many(list(briefs.values()))}
    candidates = [
        Candidate(
            symbol=fc.symbol,
            forecast=fc,
            decision=comparisons[fc.symbol],
            brief=briefs[fc.symbol],
            debate=debates[fc.symbol],
        )
        for fc in forecasts
        if fc.symbol in comparisons and fc.symbol in briefs and fc.symbol in debates
    ]

    portfolio = run_manage(
        candidates, current_drawdown=account.drawdown, daily_loss=account.daily_loss
    )
    orders = reconcile(portfolio, account, account_broker.get_positions())
    plan = place_orders(orders, account_broker, dry_run=not execute)
    report_path = write_reports(candidates, portfolio, orders, plan)

    mode = "EXECUTE" if execute else "dry-run"
    typer.echo(f"\n=== Target portfolio  asof {portfolio.asof}  ({broker} account, {mode}) ===")
    if portfolio.halted:
        typer.echo("  *** HALTED by risk gate -- all cash ***")
    for pos in portfolio.positions:
        typer.echo(f"  {pos.symbol:<6} {pos.weight:6.2%}  conv={pos.conviction:.2f}")
    typer.echo(f"  cash {portfolio.cash_weight:.2%}  |  gross {portfolio.gross_exposure:.2%}")
    typer.echo("  --- orders ---")
    for line in plan or ["(no orders)"]:
        typer.echo(f"  {line}")
    typer.echo("  --- report ---")
    typer.echo(f"  {daily_report(portfolio, orders)}")
    typer.echo(f"  --- per-stock dossiers written to {report_path} ---")


@app.command()
def backtest(
    symbols: list[str],
    start: str = typer.Option("2024-01-01", help="Backtest start date (YYYY-MM-DD)."),
    end: str = typer.Option("2025-12-31", help="Backtest end date (YYYY-MM-DD)."),
    rebalance_days: int = typer.Option(21, help="Trading days between rebalances."),
    cost_bps: float = typer.Option(5.0, help="Cost per unit turnover, in basis points."),
    cv_splits: int = typer.Option(5, help="Purged-CV folds for the signal IC."),
) -> None:
    """Walk-forward backtest of the quant signal vs SPY, with a purged-CV signal IC.

    Forecasts as-of each rebalance date (no look-ahead), builds a BUY book through the
    live risk gate, marks it to the next bar with bps costs, and benchmarks vs SPY
    buy-and-hold. Ingests any missing tickers (and SPY) first. Requires a CUDA GPU and
    a trained checkpoint. NOTE: with the current degenerate model, expect flat /
    ~zero-IC results -- this validates the plumbing, not signal quality.

    Parameters
    ----------
    symbols : list[str]
        Ticker universe to backtest.
    start, end : str, optional
        Backtest date range (ISO ``YYYY-MM-DD``).
    rebalance_days : int, optional
        Trading days between rebalances. Defaults to ``21`` (~monthly).
    cost_bps : float, optional
        Cost per unit turnover in basis points. Defaults to ``5``.
    cv_splits : int, optional
        Purged-CV folds for the signal information coefficient. Defaults to ``5``.
    """
    import datetime as _dt

    from ophir.agent.backtest import walk_forward
    from ophir.agent.feed import load_daily_ohlcv, parquet_path
    from ophir.agent.ingest import ingest
    from ophir.agent.predict import load_predictor

    universe = [s.upper().strip() for s in symbols]
    lookback = (_dt.date.today() - _dt.date.fromisoformat(start)).days + 500
    for sym in [*universe, "SPY"]:
        if not parquet_path(sym).exists():
            typer.echo(f"ingesting {sym} ...")
            ingest(sym, days=lookback)

    benchmark = load_daily_ohlcv("SPY")["close"]
    result = walk_forward(
        universe,
        model=load_predictor(),
        benchmark=benchmark,
        start=start,
        end=end,
        rebalance_days=rebalance_days,
        cost_bps=cost_bps,
        cv_splits=cv_splits,
    )

    m, b = result.metrics, result.benchmark_metrics
    typer.echo(
        f"\n=== Backtest {start} -> {end}  "
        f"({result.n_rebalances} rebalances, {result.n_trades} trades) ==="
    )
    typer.echo(f"  total return   strat {m['total_return']:+.2%}   SPY {b['total_return']:+.2%}")
    typer.echo(f"  ann return     strat {m['ann_return']:+.2%}   SPY {b['ann_return']:+.2%}")
    typer.echo(f"  Sharpe         strat {m['sharpe']:+.2f}    SPY {b['sharpe']:+.2f}")
    typer.echo(f"  Sortino        strat {m['sortino']:+.2f}    SPY {b['sortino']:+.2f}")
    typer.echo(f"  max drawdown   strat {m['max_drawdown']:.2%}   SPY {b['max_drawdown']:.2%}")
    typer.echo(f"  Calmar         strat {m['calmar']:+.2f}    SPY {b['calmar']:+.2f}")
    typer.echo(f"  hit rate {m['hit_rate']:.2%}  |  mean turnover {m['mean_turnover']:.2f}")
    typer.echo(
        f"  signal IC (purged-CV, {result.ic['n_folds']} folds): "
        f"mean {result.ic['mean_ic']:+.3f}  std {result.ic['std_ic']:.3f}"
    )
