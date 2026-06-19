"""The ``ophir`` command-line interface.

This module builds the Typer application exposed as the ``ophir`` console
script (entry point ``ophir.cli:app``). It mounts the :mod:`ophir.register`
sub-application under ``register`` and adds the top-level ``serve`` command.
"""

import typer

from ophir import curation, evaluate, register, train
from ophir.trading import cli as trading_cli

app = typer.Typer(help="Ophir CLI")
app.add_typer(register.app, name="register")
app.add_typer(trading_cli.app, name="trade")
app.command()(train.train)
app.command()(train.finetune)
app.command()(evaluate.evaluate)
app.command()(curation.curate)


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


@app.command()
def sweep(
    trials: int = typer.Option(50, help="Number of proxy trials to run"),
    study: str = typer.Option("ophir-sweep", help="Optuna study name (resumed if it exists)"),
    storage: str | None = typer.Option(
        None, help="Optuna storage URL; defaults to a SQLite db under the model dir"
    ),
    confirm_top: int = typer.Option(
        5, help="Retrain and eval the top-K configs at full budget (0 to skip)"
    ),
    proxy_steps: int = typer.Option(2000, help="max_steps per proxy trial"),
    proxy_val_batches: int = typer.Option(20, help="Validation batches per proxy validation pass"),
    full_steps: int = typer.Option(20000, help="max_steps for the confirm-phase full runs"),
    val_batches: int = typer.Option(50, help="Validation batches for the confirm-phase eval"),
    base_seed: int = typer.Option(0, help="Base seed; trial N uses base_seed + N"),
    seq_len: int = typer.Option(365, help="Window length (fixed across the sweep)"),
    offset: int = typer.Option(90, help="Window stride"),
    response_size: int = typer.Option(90, help="Forecast horizon"),
    batch_size: int = typer.Option(32, help="Batch size"),
    use_sp500: bool = typer.Option(False, help="Restrict to S&P 500 symbols"),
    data_dir: str | None = typer.Option(None, help="Override the data directory"),
    sampler: str = typer.Option("tpe", help="Optuna sampler: 'tpe' (default) or 'random'"),
    prune: bool = typer.Option(
        True, help="Enable ASHA pruning (use --no-prune for a clean control study)"
    ),
) -> None:
    """Run an Optuna hyperparameter sweep, then confirm the best configs.

    Each proxy trial runs a reduced-budget training scored on ``val_rank_ic``;
    Optuna's ASHA pruner kills unpromising trials early. The study is persisted
    to SQLite and resumed if ``--study`` already exists. After the search, the
    top ``--confirm-top`` configs are retrained at full budget and scored with
    the offline eval report. Requires CUDA.
    """
    import os

    from ophir import register
    from ophir import sweep as sweep_mod

    if storage is None:
        storage = f"sqlite:///{os.path.join(register.MODEL_DIR, study + '.db')}"

    shared = {
        "seq_len": seq_len,
        "offset": offset,
        "response_size": response_size,
        "batch_size": batch_size,
        "use_sp500": use_sp500,
        "data_dir": data_dir,
    }
    proxy_kwargs = {**shared, "max_steps": proxy_steps, "val_batches": proxy_val_batches}

    import optuna

    study_obj = sweep_mod.run_sweep(
        n_trials=trials,
        study_name=study,
        storage=storage,
        base_seed=base_seed,
        proxy_kwargs=proxy_kwargs,
        sampler=sampler,
        prune=prune,
    )
    completed = [t for t in study_obj.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        typer.echo(f"Best proxy val_rank_ic: {study_obj.best_value:.5f}")
        typer.echo(f"Best params: {study_obj.best_params}")
    else:
        typer.echo("No trials completed (all pruned); inspect the study for details.")

    if confirm_top > 0:
        from ophir.evaluate import format_report

        full_kwargs = {
            **shared,
            "max_steps": full_steps,
            "num_workers": 4,
            "cache_size": 8,
            "min_volume": 1000.0,
            "train_min_year": None,
            "train_max_year": 2023,
            "val_min_year": 2024,
            "val_max_year": None,
            "use_quality_allowlist": False,
            "clean_rows": False,
            "max_abs_r_close": 0.75,
            "epochs": 10,
            "window_sample": 256,
            "val_every_steps": 500,
        }
        results = sweep_mod.confirm_top(
            study_obj, k=confirm_top, full_kwargs=full_kwargs, val_batches=val_batches
        )
        for rank, record in enumerate(results, start=1):
            typer.echo(f"\n## Rank {rank}: {record['config']}")
            typer.echo(format_report({"confirm": record["report"]}))


@app.command()
def importances(
    study: str = typer.Argument(..., help="Optuna study name to analyze"),
    storage: str | None = typer.Option(
        None, help="Optuna storage URL; defaults to a SQLite db under the model dir"
    ),
    sampler: str = typer.Option(
        "tpe", help="Sampler the study used (controls the reliability warning)"
    ),
    pruned: bool = typer.Option(
        True, help="Whether the study used pruning (controls the reliability warning)"
    ),
) -> None:
    """Print fANOVA + MDI hyperparameter importances for a completed sweep study.

    Loads the persisted study and reports both importance estimates. Pass the
    ``--sampler``/``--pruned`` the study was run with so the output can warn when
    the estimate is biased (TPE or ASHA produce non-i.i.d. completed-trial sets).
    """
    import os

    import optuna

    from ophir import register
    from ophir import sweep as sweep_mod

    if storage is None:
        storage = f"sqlite:///{os.path.join(register.MODEL_DIR, study + '.db')}"
    study_obj = optuna.load_study(study_name=study, storage=storage)
    result = sweep_mod.compute_importances(study_obj)
    typer.echo(sweep_mod.format_importances(result, sampler=sampler, pruned=pruned))


@app.command()
def migrate_sqlite(
    src: str = typer.Option(None, help="Parquet base dir (default: <DATA_DIR>/days/stocks)"),
    dst: str = typer.Option(
        None, help="Destination SQLite file (default: <DATA_DIR>/days/stocks.db)"
    ),
    overwrite: bool = typer.Option(False, help="Rewrite tables for tickers already present"),
) -> None:
    """Convert the per-ticker parquet tree into a single SQLite store.

    Builds one table per ticker plus a ``_tickers`` manifest. Idempotent:
    tickers already present are skipped unless ``--overwrite`` is given.

    Parameters
    ----------
    src : str, optional
        Parquet base directory. Defaults to ``<DATA_DIR>/days/stocks``.
    dst : str, optional
        Destination SQLite file path. Defaults to ``<DATA_DIR>/days/stocks.db``.
    overwrite : bool, optional
        If ``True``, rewrite tables for tickers already present. Defaults to
        ``False``.
    """
    import os

    from ophir.register import get_default_data_days_dir
    from ophir.sqlite_store import build_sqlite_store

    days = get_default_data_days_dir()
    src = src or os.path.join(days, "stocks")
    dst = dst or os.path.join(days, "stocks.db")

    written = build_sqlite_store(src, dst, overwrite=overwrite)
    typer.echo(f"{written} tickers written")
