"""The ``ophir`` command-line interface.

This module builds the Typer application exposed as the ``ophir`` console
script (entry point ``ophir.cli:app``). It mounts the :mod:`ophir.register`
sub-application under ``register`` and adds the top-level ``serve`` command.
"""

import typer

from ophir import register

app = typer.Typer(help="Ophir CLI")
app.add_typer(register.app, name="register")


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
