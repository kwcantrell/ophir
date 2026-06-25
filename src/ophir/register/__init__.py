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
from typing import TYPE_CHECKING, Annotated, Literal, NoReturn, overload

import typer
from massive import RESTClient

from ophir.register.symbols import (
    clear_ignore_symbols,
    clear_quality_symbols,
    fetch_ignore_symbols_list,
    fetch_quality_symbols_list,
    set_ignore_symbols,
    set_quality_symbols,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint

    from ophir.training_models import LightningOHLCPredictor

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


def _best_checkpoint_callback(file_name: str, monitor_near_ic: bool) -> ModelCheckpoint:
    """Build the best-checkpoint callback, monitoring near-IC or ``val_loss``.

    ``val_loss`` is anti-aligned with cross-sectional IC (IC peaks mid-run then
    droops as the cosine LR anneals), so when the validation loader carries
    identity we select on ``val_rank_ic_near`` (maximising) instead. Without
    identity that metric is never logged, so we fall back to ``val_loss``
    (minimising).

    Parameters
    ----------
    file_name : str
        Base name for the checkpoint files.
    monitor_near_ic : bool
        When ``True`` monitor ``val_rank_ic_near`` (``mode="max"``); otherwise
        monitor ``val_loss`` (``mode="min"``).

    Returns
    -------
    ModelCheckpoint
        The configured best-checkpoint callback.
    """
    from lightning.pytorch.callbacks import ModelCheckpoint

    monitor, mode = ("val_rank_ic_near", "max") if monitor_near_ic else ("val_loss", "min")
    suffix = "best-{epoch:02d}-{val_rank_ic_near:.5f}" if monitor_near_ic else EPOCH_MODIFIER
    return ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        dirpath=os.path.join(MODEL_DIR, "candidates"),
        filename=file_name + suffix,
        save_top_k=1,
        save_on_train_epoch_end=False,
    )


def fetch_base_trainer(
    file_name: str | None = None,
    max_steps: int = 100000,
    val_check_interval: int | float = 1.0,
    limit_val_batches: int | float = 1.0,
    extra_callbacks: list[L.Callback] | None = None,
    monitor_near_ic: bool = False,
) -> L.Trainer:
    """Build the :class:`lightning.Trainer` used for base pre-training.

    Configures mixed precision, CUDA acceleration, gradient clipping, both a
    TensorBoard and a CSV logger, and two checkpoint callbacks: one that saves
    on a fixed wall-clock interval and one that saves the best epoch.

    Parameters
    ----------
    file_name : str, optional
        Base name for the checkpoint files. Defaults to :data:`BASE_NAME`.
    max_steps : int, optional
        Total number of optimizer steps. Should match the ``max_steps``
        hyper-parameter passed to
        :class:`~ophir.training_models.LightningOHLCPredictor` so the trainer
        and the cosine schedule share one horizon. Defaults to ``100000``.
    val_check_interval : int or float, optional
        How often to run validation. An ``int`` validates every N optimizer
        steps (decoupled from the epoch — preferred for the unsized streaming
        dataset); a ``float`` is a fraction of an epoch. Defaults to ``1.0``.
    limit_val_batches : int or float, optional
        Cap on validation batches per validation pass, keeping step-based
        validation cheap. Defaults to ``1.0`` (the whole validation set).
    extra_callbacks : list[lightning.Callback], optional
        Additional callbacks appended to the default set (e.g. a sweep's
        pruning callback). Defaults to ``None``.
    monitor_near_ic : bool, optional
        When ``True`` the best-checkpoint callback selects on
        ``val_rank_ic_near`` (``mode="max"``) instead of ``val_loss``. Only
        valid when the validation loader logs that metric (identity present).
        Defaults to ``False``.

    Returns
    -------
    lightning.Trainer
        A configured trainer ready to fit a
        :class:`~ophir.training_models.LightningOHLCPredictor`.
    """
    import lightning as L
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

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

    # 2. Best-checkpoint callback: monitors ``val_rank_ic_near`` (max) when
    # identity is present, otherwise ``val_loss`` (min).
    epoch_checkpoint_callback = _best_checkpoint_callback(file_name, monitor_near_ic)

    callbacks: list[L.Callback] = [
        time_checkpoint_callback,
        epoch_checkpoint_callback,
        LearningRateMonitor("step"),
    ]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    trainer = L.Trainer(
        max_steps=max_steps,
        precision="bf16-mixed",
        default_root_dir=MODEL_DIR,
        accelerator="cuda",
        callbacks=callbacks,
        logger=[
            TensorBoardLogger(MODEL_DIR, name="tensorboard-logger"),
            CSVLogger(MODEL_DIR, name="csv-logger", flush_logs_every_n_steps=10),
        ],
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=None if isinstance(val_check_interval, int) else 1,
        limit_val_batches=limit_val_batches,
        gradient_clip_val=1,
        gradient_clip_algorithm="norm",
    )
    return trainer


def fetch_finetune_trainer() -> L.Trainer:
    """Build the :class:`lightning.Trainer` used for finetuning.

    Like :func:`fetch_base_trainer` but epoch-driven: it validates every epoch
    and checkpoints every 25 epochs under :data:`FINETUNE_NAME`. Logs to both
    TensorBoard and CSV.

    Returns
    -------
    lightning.Trainer
        A configured trainer for the finetuning stage.
    """
    import lightning as L
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

    trainer = L.Trainer(
        precision="bf16-mixed",
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
        logger=[
            TensorBoardLogger(MODEL_DIR, name="tensorboard-logger"),
            CSVLogger(MODEL_DIR, name="csv-logger", flush_logs_every_n_steps=10),
        ],
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


def _latest_base_ckpt(filename: str) -> str:
    """Return the filename of the most recent base checkpoint.

    Parameters
    ----------
    filename : str
        Substring that base checkpoint files must contain.

    Returns
    -------
    str
        The highest ``-v<N>`` version among matches, or the sorted-last match
        when none carry a ``-v<N>`` suffix.

    Raises
    ------
    FileNotFoundError
        When no file in :data:`MODEL_DIR` contains ``filename``.
    """
    base_paths = sorted(path for path in os.listdir(MODEL_DIR) if filename in path)
    if not base_paths:
        raise FileNotFoundError(f"no checkpoint matching {filename!r} in {MODEL_DIR}")
    versioned = sorted(
        (int(version.removeprefix(f"{filename}-v").removesuffix(".ckpt")), version)
        for version in base_paths
        if f"{filename}-v" in version
    )
    return versioned[-1][1] if versioned else base_paths[-1]


def _latest_finetuned_ckpt() -> str:
    """Return the filename of the most recent finetuned checkpoint.

    Returns
    -------
    str
        The latest :data:`FINETUNE_NAME` checkpoint filename (highest
        ``-v<N>`` version, or the sole match) within :data:`MODEL_DIR`.
    """
    fintune_paths = sorted([path for path in os.listdir(MODEL_DIR) if FINETUNE_NAME in path])
    if len(fintune_paths) > 1:
        base_versions = sorted(
            [
                (
                    int(version.removeprefix(f"{FINETUNE_NAME}-v").removesuffix(".ckpt")),
                    version,
                )
                for version in fintune_paths
                if f"{FINETUNE_NAME}-v" in version
            ]
        )
        latest_version = base_versions[-1][1]
    else:
        latest_version = fintune_paths[0]

    return latest_version


def _resolve_base_ckpt_path(file_name: str | None = None, time_version: bool = True) -> str:
    """Resolve a base checkpoint path without loading the model.

    ``time_version=True`` selects the latest rolling ``-time-check-v<N>``
    checkpoint; ``time_version=False`` selects the explicit canonical
    best-checkpoint ``{MODEL_DIR}/{name}-best.ckpt``.

    Parameters
    ----------
    file_name : str or None, optional
        Base name. ``None`` uses :data:`BASE_NAME`.
    time_version : bool, optional
        Select the rolling time-check checkpoint (``True``) or the canonical
        best checkpoint (``False``). Defaults to ``True``.

    Returns
    -------
    str
        Absolute path to the resolved checkpoint.

    Raises
    ------
    FileNotFoundError
        When no matching checkpoint exists.
    """
    name = file_name if file_name is not None else BASE_NAME
    if time_version:
        return os.path.join(MODEL_DIR, _latest_base_ckpt(name + TIME_MODIFIER))
    path = os.path.join(MODEL_DIR, f"{name}-best.ckpt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"canonical base checkpoint not found: {path}")
    return path


def _feature_dim_mismatch(state_dict: Mapping[str, Any]) -> tuple[int, int] | None:
    """Return ``(found, expected)`` when a checkpoint disagrees on feature width.

    Inspects the ``feature_mlp`` weight in a checkpoint ``state_dict`` and
    compares its input dimension to the current model's
    :data:`ophir.models.FEATURE_DIM`.

    Parameters
    ----------
    state_dict : mapping of str to Any
        A checkpoint ``state_dict`` (parameter name to tensor).

    Returns
    -------
    tuple[int, int] or None
        ``(found, expected)`` when the checkpoint's ``feature_mlp`` input dim
        differs from the current model; ``None`` when it matches or no
        ``feature_mlp`` weight is present (cannot check — do not block).
    """
    from ophir.models import FEATURE_DIM

    key = next((k for k in state_dict if k.endswith("feature_mlp.weight")), None)
    if key is None:
        return None
    found = int(state_dict[key].shape[1])
    return (found, FEATURE_DIM) if found != FEATURE_DIM else None


def _raise_load_error_with_hint(ckpt_path: str, original: RuntimeError) -> NoReturn:
    """Re-raise ``original``, clarifying it first if it is a feature-dim drift.

    A stale checkpoint (trained before a feature-schema change) fails
    ``load_from_checkpoint`` with an opaque ``size mismatch`` error. This reads
    the checkpoint's ``feature_mlp`` width and, on a mismatch, raises a clear,
    actionable error pointing at the promotion runbook; otherwise it re-raises
    the original error unchanged.
    """
    import torch

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception:
        raise original from None
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else {}
    mismatch = _feature_dim_mismatch(state_dict)
    if mismatch is None:
        raise original
    found, expected = mismatch
    raise RuntimeError(
        f"checkpoint {ckpt_path!r} has feature_mlp input dim {found}, but the current "
        f"model expects {expected}; it predates a feature-schema change and must be "
        f"retrained and re-promoted (see docs/checkpoint-promotion.md)."
    ) from original


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

    last_ckpt = _resolve_base_ckpt_path(file_name, time_version)
    print(f"loading {last_ckpt}")

    try:
        model = LightningOHLCPredictor.load_from_checkpoint(last_ckpt, strict=strict)
    except RuntimeError as exc:
        _raise_load_error_with_hint(last_ckpt, exc)

    if not return_ckpt_path:
        return model
    return model, last_ckpt


@overload
def load_finetuned_ckpt(
    strict: bool = ...,
    return_ckpt_path: Literal[False] = ...,
) -> LightningOHLCPredictor: ...


@overload
def load_finetuned_ckpt(
    strict: bool = ...,
    *,
    return_ckpt_path: Literal[True],
) -> tuple[LightningOHLCPredictor, str]: ...


def load_finetuned_ckpt(
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

    trainer = L.Trainer(precision="bf16-mixed", accelerator="cuda")
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
