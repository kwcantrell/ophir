"""Filesystem, checkpoint, and Lightning ``Trainer`` helpers.

This module owns the package's on-disk layout under ``.ophir/`` (data and
model directories, created on import), factory functions for the base and
finetuning :class:`lightning.Trainer` objects, checkpoint loaders for the
latest base / finetuned :class:`~ophir.training_models.LightningOHLCPredictor`,
and a small Typer sub-application (:data:`app`) for storing the MASSIVE API
key.
"""

from __future__ import annotations

import os
from typing import Annotated

import typer
from massive import RESTClient

from ophir.register.checkpoints import _feature_dim_mismatch as _feature_dim_mismatch
from ophir.register.checkpoints import _latest_base_ckpt as _latest_base_ckpt
from ophir.register.checkpoints import _latest_finetuned_ckpt as _latest_finetuned_ckpt
from ophir.register.checkpoints import _raise_load_error_with_hint as _raise_load_error_with_hint
from ophir.register.checkpoints import _resolve_base_ckpt_path as _resolve_base_ckpt_path
from ophir.register.checkpoints import load_base_model_ckpt, load_finetuned_ckpt
from ophir.register.layout import (
    BASE_BEST_CKPT,
    BASE_MODEL_CKPT,
    BASE_NAME,
    DATA_DIR,
    EPOCH_MODIFIER,
    FINETUNE_NAME,
    MODEL_DIR,
    OPHIR_DIR,
    TIME_MODIFIER,
    get_default_data_days_dir,
    quality_stats_path,
)
from ophir.register.symbols import (
    clear_ignore_symbols,
    clear_quality_symbols,
    fetch_ignore_symbols_list,
    fetch_quality_symbols_list,
    set_ignore_symbols,
    set_quality_symbols,
)
from ophir.register.trainers import _best_checkpoint_callback as _best_checkpoint_callback
from ophir.register.trainers import fetch_base_trainer, fetch_finetune_trainer, predict_trainer

__all__ = [
    "BASE_BEST_CKPT",
    "BASE_MODEL_CKPT",
    "BASE_NAME",
    "DATA_DIR",
    "EPOCH_MODIFIER",
    "FINETUNE_NAME",
    "MODEL_DIR",
    "OPHIR_DIR",
    "TIME_MODIFIER",
    "app",
    "clear_ignore_symbols",
    "clear_quality_symbols",
    "fetch_base_trainer",
    "fetch_finetune_trainer",
    "fetch_ignore_symbols_list",
    "fetch_quality_symbols_list",
    "get_default_data_days_dir",
    "get_massive_client",
    "load_base_model_ckpt",
    "load_finetuned_ckpt",
    "massive_key",
    "predict_trainer",
    "quality_stats_path",
    "set_ignore_symbols",
    "set_quality_symbols",
]


def get_massive_client() -> RESTClient:
    """Construct an authenticated MASSIVE REST client.

    Reads the API key previously stored by the ``ophir register massive-key``
    command (see :func:`massive_key`).

    Returns
    -------
    massive.RESTClient
        A client authenticated with the stored key.

    Raises
    ------
    AssertionError
        If no key file exists under :data:`OPHIR_DIR`.
    """
    assert os.path.exists(os.path.join(OPHIR_DIR, ".massive_key"))
    with open(os.path.join(OPHIR_DIR, ".massive_key")) as f:
        key = f.readline().strip()
    return RESTClient(key)


app = typer.Typer()


@app.command()
def massive_key(
    key: Annotated[str, typer.Argument(help="MASSIVE API key")],
) -> None:
    """Store a MASSIVE API key for later data fetching.

    Backs the ``ophir register massive-key`` CLI command. The key is written
    to ``.massive_key`` under :data:`OPHIR_DIR` and later read by
    :func:`get_massive_client`.

    Parameters
    ----------
    key : str
        The MASSIVE API key to persist.
    """
    with open(os.path.join(OPHIR_DIR, ".massive_key"), "w") as f:
        f.write(f"{key}\n")
