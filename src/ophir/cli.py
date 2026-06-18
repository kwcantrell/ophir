"""The ``ophir`` command-line interface.

This module builds the Typer application exposed as the ``ophir`` console
script (entry point ``ophir.cli:app``). It mounts the :mod:`ophir.register`
sub-application under ``register`` and adds the top-level ``serve`` command.
"""

import typer

from ophir import evaluate, register, train

app = typer.Typer(help="Ophir CLI")
app.add_typer(register.app, name="register")
app.command()(train.train)
app.command()(train.finetune)
app.command()(evaluate.evaluate)


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
def dashboard(
    port: int = typer.Option(7861, help="Gradio server port"),
    share: bool = typer.Option(False, help="Expose a public share link"),
    debug: bool = typer.Option(True, help="Launch Gradio in debug mode"),
    model_dir: str | None = typer.Option(
        None, help="Directory holding csv-logger metrics and checkpoints"
    ),
) -> None:
    """Launch the live training dashboard.

    Imports :mod:`ophir.dashboard` lazily and starts a Gradio server showing
    per-target loss curves (read live from ``metrics.csv``) and an on-demand
    response-block leakage check. Unlike ``serve`` the module is import-safe;
    the leakage check loads a checkpoint and prefers CUDA when available.

    Parameters
    ----------
    port : int, optional
        Gradio server port. Defaults to ``7861``.
    share : bool, optional
        If ``True``, expose a public Gradio share link. Defaults to ``False``.
    debug : bool, optional
        If ``True``, launch Gradio in debug mode. Defaults to ``True``.
    model_dir : str, optional
        Directory holding the training metrics and checkpoints. Defaults to the
        package model directory.
    """
    from ophir import dashboard as dashboard_ui

    dashboard_ui.launch(port=port, share=share, debug=debug, model_dir=model_dir)
