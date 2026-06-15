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
from datetime import timedelta
from typing import TYPE_CHECKING, Annotated, Literal, overload

import typer
from massive import RESTClient

if TYPE_CHECKING:
    from collections.abc import Iterable

    import lightning as L

    from ophir.training_models import LightningOHLCPredictor

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
print(f"current file path {current_file_path}")

# Get the directory containing the current file
current_dir = os.path.dirname(current_file_path)
print(f"current dir path {current_dir}")
OPHIR_DIR = os.path.join(current_dir, ".ophir")
DATA_DIR = os.path.join(OPHIR_DIR, "data")
MODEL_DIR = os.path.join(OPHIR_DIR, "model")
BASE_NAME = "ophir-ohlc-base"
FINETUNE_NAME = "ophire-ohlc-finetuned"
BASE_MODEL_CKPT = os.path.join(MODEL_DIR, f"{BASE_NAME}.ckpt")
TIME_MODIFIER = "-time-check"
EPOCH_MODIFIER = "best-{epoch:02d}-{val_loss:.5f}"

if not os.path.exists(OPHIR_DIR):
    os.makedirs(OPHIR_DIR)

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


def fetch_base_trainer(file_name: str | None = None, max_steps: int = 100000) -> L.Trainer:
    """Build the :class:`lightning.Trainer` used for base pre-training.

    Configures mixed precision, CUDA acceleration, gradient clipping, a
    TensorBoard logger, and two checkpoint callbacks: one that saves on a
    fixed wall-clock interval and one that saves the best ``val_loss`` epoch.

    Parameters
    ----------
    file_name : str, optional
        Base name for the checkpoint files. Defaults to :data:`BASE_NAME`.
    max_steps : int, optional
        Maximum number of optimizer steps to run. Defaults to ``100000``;
        a smaller value shortens the run (e.g. a quick validation gate).

    Returns
    -------
    lightning.Trainer
        A configured trainer ready to fit a
        :class:`~ophir.training_models.LightningOHLCPredictor`.
    """
    import lightning as L
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger

    if file_name is None:
        file_name = BASE_NAME

    # 1. Checkpoint every N minutes
    # Checkpoints will be saved with a format like 'time_check-{time}.ckpt'
    time_checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_DIR,
        filename=file_name + TIME_MODIFIER,
        train_time_interval=timedelta(minutes=1),  # Set N to your desired interval
        save_on_train_epoch_end=False,  # Prevents this callback from also saving at epoch end
    )

    # 2. Checkpoint at the end of every epoch
    # Checkpoints will be saved with a format like 'epoch_check-epoch={epoch}.ckpt'
    epoch_checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=MODEL_DIR,
        filename=file_name + EPOCH_MODIFIER,
        save_top_k=1,  # Saves at the end of every epoch
        # To save all epoch checkpoints, or use 0 to save only the last
        save_on_train_epoch_end=True,
    )

    trainer = L.Trainer(
        max_steps=max_steps,
        precision="16-mixed",
        default_root_dir=MODEL_DIR,
        accelerator="cuda",
        callbacks=[
            time_checkpoint_callback,
            epoch_checkpoint_callback,
            LearningRateMonitor("step"),
        ],
        logger=TensorBoardLogger(MODEL_DIR, name="tensorboard-logger"),
        gradient_clip_val=1,
        gradient_clip_algorithm="norm",
    )
    return trainer


def fetch_finetune_trainer() -> L.Trainer:
    """Build the :class:`lightning.Trainer` used for finetuning.

    Like :func:`fetch_base_trainer` but epoch-driven: it validates every epoch
    and checkpoints every 25 epochs under :data:`FINETUNE_NAME`.

    Returns
    -------
    lightning.Trainer
        A configured trainer for the finetuning stage.
    """
    import lightning as L
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger

    trainer = L.Trainer(
        precision="16-mixed",
        max_epochs=10000,
        default_root_dir=MODEL_DIR,
        accelerator="cuda",
        callbacks=[
            ModelCheckpoint(
                dirpath=MODEL_DIR,
                filename=FINETUNE_NAME,
                every_n_epochs=25,
                save_on_train_epoch_end=True,
            ),
            LearningRateMonitor("epoch"),
        ],
        logger=TensorBoardLogger(MODEL_DIR, name="tensorboard-logger"),
        check_val_every_n_epoch=1,
    )
    return trainer


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


def _latest_ckpt(prefix: str) -> str:
    """Return the newest checkpoint whose filename contains ``prefix``.

    Files carrying a Lightning ``-v<N>`` suffix are ordered by that version (highest
    wins, preserving the time-check behavior). Otherwise -- e.g. the
    ``basebest-epoch=NN-val_loss=X`` files, which have no ``-v<N>`` -- the most
    recently modified file is chosen.

    Parameters
    ----------
    prefix : str
        Substring the checkpoint filename must contain.

    Returns
    -------
    str
        The selected checkpoint filename within :data:`MODEL_DIR`.

    Raises
    ------
    FileNotFoundError
        If no ``.ckpt`` file in :data:`MODEL_DIR` contains ``prefix``.
    """
    paths = sorted(p for p in os.listdir(MODEL_DIR) if prefix in p and p.endswith(".ckpt"))
    if not paths:
        raise FileNotFoundError(f"no checkpoint matching {prefix!r} in {MODEL_DIR}")
    versioned: list[tuple[int, str]] = []
    for path in paths:
        suffix = path.removeprefix(f"{prefix}-v").removesuffix(".ckpt")
        if f"{prefix}-v" in path and suffix.isdigit():
            versioned.append((int(suffix), path))
    if versioned:
        return max(versioned)[1]
    return max(paths, key=lambda p: os.path.getmtime(os.path.join(MODEL_DIR, p)))


def _latest_base_ckpt(filename: str) -> str:
    """Return the filename of the most recent base checkpoint.

    Parameters
    ----------
    filename : str
        Substring that base checkpoint files must contain.

    Returns
    -------
    str
        The latest matching checkpoint filename (highest ``-v<N>`` version, else the
        most recently modified) within :data:`MODEL_DIR`.
    """
    return _latest_ckpt(filename)


def _latest_finetuned_ckpt() -> str:
    """Return the filename of the most recent finetuned checkpoint.

    Returns
    -------
    str
        The latest :data:`FINETUNE_NAME` checkpoint filename (highest ``-v<N>``
        version, else the most recently modified) within :data:`MODEL_DIR`.
    """
    return _latest_ckpt(FINETUNE_NAME)


def get_default_data_days_dir() -> str:
    """Return the default directory holding per-day stock data.

    Returns
    -------
    str
        ``<DATA_DIR>/days``.
    """
    return os.path.join(DATA_DIR, "days")


def clear_ignore_symbols() -> None:
    """Delete the ignore-symbols list, re-enabling all symbols.

    Removes ``<DATA_DIR>/ignore-symbols.txt`` if it exists; a no-op otherwise.
    """
    print("Reseting ignore symbols mode...")
    path = os.path.join(DATA_DIR, "ignore-symbols.txt")
    if os.path.exists(path):
        os.remove(path)


def set_ignore_symbols(symbols: Iterable[str]) -> None:
    """Add symbols to the persistent ignore list.

    The new symbols are unioned with the existing list and written back,
    sorted, to ``<DATA_DIR>/ignore-symbols.txt``.

    Parameters
    ----------
    symbols : Iterable[str]
        Ticker symbols to add to the ignore list.
    """
    merged = sorted(set(fetch_ignore_symbols_list()).union(symbols))
    with open(os.path.join(DATA_DIR, "ignore-symbols.txt"), "w") as f:
        for symbol in merged:
            f.write(f"{symbol}\n")
    print(f"Entering Ignore Symbol mode...currently ignoring {len(merged)} symbols")


def fetch_ignore_symbols_list() -> list[str]:
    """Return the persisted ignore-symbols list.

    Returns
    -------
    list[str]
        Symbols read from ``<DATA_DIR>/ignore-symbols.txt``, or an empty list
        if the file does not exist.
    """
    if not os.path.exists(os.path.join(DATA_DIR, "ignore-symbols.txt")):
        return []

    with open(os.path.join(DATA_DIR, "ignore-symbols.txt")) as f:
        symbols = [symbol.strip() for symbol in f.readlines()]
    return symbols


@overload
def load_base_model_ckpt(
    strict: bool = ...,
    return_ckpt_path: Literal[False] = ...,
    file_name: str | None = ...,
    time_version: bool = ...,
) -> LightningOHLCPredictor: ...


@overload
def load_base_model_ckpt(
    strict: bool = ...,
    *,
    return_ckpt_path: Literal[True],
    file_name: str | None = ...,
    time_version: bool = ...,
) -> tuple[LightningOHLCPredictor, str]: ...


def load_base_model_ckpt(
    strict: bool = True,
    return_ckpt_path: bool = False,
    file_name: str | None = None,
    time_version: bool = True,
) -> LightningOHLCPredictor | tuple[LightningOHLCPredictor, str]:
    """Load the latest base checkpoint.

    Parameters
    ----------
    strict : bool, optional
        Passed to ``load_from_checkpoint``; require an exact ``state_dict``
        match. Defaults to ``True``.
    return_ckpt_path : bool, optional
        If ``True``, also return the resolved checkpoint path. Defaults to
        ``False``.
    file_name : str, optional
        Base name to load. Defaults to :data:`BASE_NAME`.
    time_version : bool, optional
        If ``True``, load the time-interval checkpoint; otherwise the
        best-epoch checkpoint. Defaults to ``True``.

    Returns
    -------
    LightningOHLCPredictor or tuple[LightningOHLCPredictor, str]
        The restored model, and the checkpoint path when
        ``return_ckpt_path`` is ``True``.
    """
    from ophir.training_models import LightningOHLCPredictor

    if file_name is None:
        file_name = BASE_NAME

    if time_version:
        file_name += TIME_MODIFIER
    else:
        file_name += EPOCH_MODIFIER
    file_name = file_name.split("{")[0]

    latest_version = _latest_base_ckpt(filename=file_name)
    print(f"loading {latest_version}")

    last_ckpt = os.path.join(MODEL_DIR, latest_version)
    if not return_ckpt_path:
        return LightningOHLCPredictor.load_from_checkpoint(last_ckpt, strict=strict)
    return LightningOHLCPredictor.load_from_checkpoint(last_ckpt, strict=strict), last_ckpt


@overload
def load_fintuned_ckpt(
    strict: bool = ...,
    return_ckpt_path: Literal[False] = ...,
) -> LightningOHLCPredictor: ...


@overload
def load_fintuned_ckpt(
    strict: bool = ...,
    *,
    return_ckpt_path: Literal[True],
) -> tuple[LightningOHLCPredictor, str]: ...


def load_fintuned_ckpt(
    strict: bool = True,
    return_ckpt_path: bool = False,
) -> LightningOHLCPredictor | tuple[LightningOHLCPredictor, str]:
    """Load the latest finetuned checkpoint.

    Parameters
    ----------
    strict : bool, optional
        Passed to ``load_from_checkpoint``. Defaults to ``True``.
    return_ckpt_path : bool, optional
        If ``True``, also return the resolved checkpoint path. Defaults to
        ``False``.

    Returns
    -------
    LightningOHLCPredictor or tuple[LightningOHLCPredictor, str]
        The restored model, and the checkpoint path when
        ``return_ckpt_path`` is ``True``.
    """
    from ophir.training_models import LightningOHLCPredictor

    latest_version = _latest_finetuned_ckpt()
    print(f"loading {latest_version}")

    last_ckpt = os.path.join(MODEL_DIR, latest_version)
    if not return_ckpt_path:
        return LightningOHLCPredictor.load_from_checkpoint(last_ckpt, strict=strict)
    return LightningOHLCPredictor.load_from_checkpoint(last_ckpt, strict=strict), last_ckpt


def predict_trainer() -> L.Trainer:
    """Build a minimal :class:`lightning.Trainer` for inference.

    Returns
    -------
    lightning.Trainer
        A CUDA, mixed-precision trainer suitable for ``predict``.
    """
    import lightning as L

    trainer = L.Trainer(precision="16-mixed", accelerator="cuda")
    return trainer


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
