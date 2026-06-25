"""PyTorch-Lightning ``Trainer`` factories for base / finetune / predict."""

from __future__ import annotations

import os
from datetime import timedelta
from typing import TYPE_CHECKING

from ophir.register import layout

if TYPE_CHECKING:
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint


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
    suffix = "best-{epoch:02d}-{val_rank_ic_near:.5f}" if monitor_near_ic else layout.EPOCH_MODIFIER
    return ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        dirpath=os.path.join(layout.MODEL_DIR, "candidates"),
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
        file_name = layout.BASE_NAME

    # 1. Checkpoint every N minutes
    # Checkpoints will be saved with a format like 'time_check-{time}.ckpt'
    time_checkpoint_callback = ModelCheckpoint(
        dirpath=layout.MODEL_DIR,
        filename=file_name + layout.TIME_MODIFIER,
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
        default_root_dir=layout.MODEL_DIR,
        accelerator="cuda",
        callbacks=callbacks,
        logger=[
            TensorBoardLogger(layout.MODEL_DIR, name="tensorboard-logger"),
            CSVLogger(layout.MODEL_DIR, name="csv-logger", flush_logs_every_n_steps=10),
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
        default_root_dir=layout.MODEL_DIR,
        accelerator="cuda",
        callbacks=[
            ModelCheckpoint(
                dirpath=layout.MODEL_DIR,
                filename=layout.FINETUNE_NAME,
                every_n_epochs=25,
                save_on_train_epoch_end=True,
            ),
            LearningRateMonitor("epoch"),
        ],
        logger=[
            TensorBoardLogger(layout.MODEL_DIR, name="tensorboard-logger"),
            CSVLogger(layout.MODEL_DIR, name="csv-logger", flush_logs_every_n_steps=10),
        ],
        check_val_every_n_epoch=1,
    )
    return trainer


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
