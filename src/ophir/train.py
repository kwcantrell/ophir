"""Model training command for Ophir (``ophir train``).

Reproduces the notebook training loop as a strict-typed CLI command: builds a
full-US-market :class:`~ophir.ticker.StockHanlder` from the shared parquet
layout, date-splits into train (``year < 2024``) and validation
(``year >= 2024``), and fits a
:class:`~ophir.training_models.LightningOHLCPredictor`. Defaults to fine-tuning
the latest base checkpoint; pass ``--finetune-from ""`` to train from scratch.
Saves a new base ``-time-check`` checkpoint that
:func:`~ophir.register.load_base_model_ckpt` auto-selects.

Requires a CUDA GPU and a backfilled parquet dataset (run ``ophir data backfill``
first). The yfinance split layer is bypassed (``stock_splits=None``) because the
parquet written by the data pipeline is already split-adjusted by MASSIVE.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Annotated, cast

import typer

if TYPE_CHECKING:
    from ophir.ticker import StockStreamer

SEQ_LEN = 365
RESPONSE_SIZE = 90
SPLIT_YEAR = 2024
MIN_VOLUME = 1000


def train(
    finetune_from: Annotated[
        str | None,
        typer.Option(
            help='Checkpoint to fine-tune from; omit for the latest base, pass "" for scratch.'
        ),
    ] = None,
    max_steps: Annotated[
        int, typer.Option(help="Maximum optimizer steps; -1 keeps the trainer default")
    ] = -1,
) -> None:
    """Train (or fine-tune) the base OHLC model on the full US market.

    Parameters
    ----------
    finetune_from : str or None, optional
        ``None`` (default) fine-tunes the latest base checkpoint; ``""`` trains
        a fresh model from scratch; any other value is a checkpoint path.
    max_steps : int, optional
        Cap on optimizer steps. ``-1`` (default) keeps the trainer's configured
        maximum; a positive value shortens the run (e.g. the validation gate).
    """
    from torch.utils.data import DataLoader

    from ophir import register
    from ophir.ticker import StockHandlerDataset, StockHanlder, StockStreamerDataset
    from ophir.training_models import LightningOHLCPredictor

    base_path = os.path.join(register.DATA_DIR, "days", "stocks")

    train_handler = StockHanlder(
        seq_len=SEQ_LEN,
        base_path=base_path,
        return_stock_id=False,
        return_streamer=True,
        stock_splits=None,
        offset=RESPONSE_SIZE,
        max_year=SPLIT_YEAR,
        min_volume=MIN_VOLUME,
        shuffle=True,
    )
    train_dataset = StockHandlerDataset(
        train_handler,
        response_size=RESPONSE_SIZE,
        cache_size=len(train_handler.stocks),
    )
    train_loader = DataLoader(train_dataset, batch_size=64, pin_memory=True, num_workers=6)

    val_handler = StockHanlder(
        seq_len=SEQ_LEN,
        base_path=base_path,
        return_stock_id=False,
        return_streamer=True,
        stock_splits=None,
        offset=RESPONSE_SIZE,
        min_year=SPLIT_YEAR,
        min_volume=MIN_VOLUME,
        shuffle=True,
    )
    streamers = [cast("StockStreamer", val_handler[stock]) for stock in val_handler.stocks]
    streamers = [s for s in streamers if s.size > 0]
    val_dataset = StockStreamerDataset(streamers, response_size=RESPONSE_SIZE)
    val_loader = DataLoader(
        val_dataset, batch_size=64, pin_memory=True, num_workers=2, prefetch_factor=10
    )

    if finetune_from == "":
        typer.echo("Training a fresh model from scratch...")
        model = LightningOHLCPredictor(512, 8, 8)
    elif finetune_from is None:
        typer.echo("Fine-tuning the latest base checkpoint...")
        model = register.load_base_model_ckpt()
    else:
        typer.echo(f"Fine-tuning from {finetune_from}...")
        model = LightningOHLCPredictor.load_from_checkpoint(finetune_from)

    trainer = register.fetch_base_trainer(max_steps=max_steps if max_steps > 0 else 100000)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
