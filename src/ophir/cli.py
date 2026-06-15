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
