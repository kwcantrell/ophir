"""Agent-mutable training assembly for the autoresearch loop.

THE ONLY FILE THE PROPOSER AGENT MAY EDIT. Everything here — architecture,
loss, optimizer, schedule, data knobs (except the sealed import) — is the
search space. You may inline/override any ophir component (e.g. subclass and
reimplement ``compute_loss``) as long as the harness contract holds:

Contract (loop.py / eval_harness.py depend on it; breaking it wastes the trial):
- CLI: ``--max-steps``, ``--seed``, ``--out-dir``.
- Writes exactly one ``best*.ckpt`` under ``--out-dir``.
- Exposes ``MODEL_CLASS`` for ``eval_harness.py`` to load the checkpoint.
  If you give ``ExperimentPredictor`` its own ``__init__`` it MUST call
  ``super().__init__(...)`` and ``self.save_hyperparameters()`` or the
  checkpoint cannot be reloaded for scoring.
- The sealed import line below stays verbatim; no year literals anywhere.

Requires CUDA (same runtime constraint as ``ophir train``).
"""

from __future__ import annotations  # noqa: I001 - block kept unsorted so the sealed line stays intact

import argparse
import os
from typing import TYPE_CHECKING

from _sealed import ACCEPT_VAL_MAX_YEAR, ACCEPT_VAL_MIN_YEAR, NUM_WORKERS, TRAIN_MAX_YEAR  # SEALED: loop.py enforces this line verbatim  # noqa: E501  # fmt: skip

from ophir.training_models import LightningOHLCPredictor

if TYPE_CHECKING:
    import lightning as L


class ExperimentPredictor(LightningOHLCPredictor):
    """The experiment surface: override loss/optimizer/model pieces here."""


MODEL_CLASS = ExperimentPredictor

# ---- experiment configuration (agent-editable) -----------------------------
MODEL_KWARGS: dict[str, object] = {
    "emb_dim": 128,
    "num_layers": 6,
    "num_heads": 8,
    "lr": 2e-4,
    "rezero_lr": 3e-4,
    "weight_decay": 0.01,
    "betas": (0.9, 0.95),
    "warmup_ratio": 0.03,
    "loss_decay": 0.6,
    "close_weight": 1.0,
    "upside_weight": 0.5,
    "downside_weight": 0.5,
    "rezero_init": 0.0,
}
SEQ_LEN = 365
WINDOW_OFFSET = 90
RESPONSE_SIZE = 90
BATCH_SIZE = 32
CACHE_SIZE = 8
MIN_VOLUME = 1000.0
VAL_EVERY_STEPS = 500
VAL_BATCHES = 50
# -----------------------------------------------------------------------------


def build_trainer(out_dir: str, max_steps: int) -> L.Trainer:
    """Local trainer writing all artifacts under ``out_dir``.

    Deliberately NOT ``register.fetch_base_trainer``: trials must never write
    into the real ``.ophir/model`` registry.
    """
    import lightning as L  # heavy import deferred past --help
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

    best = ModelCheckpoint(
        monitor="val_rank_ic_near",
        mode="max",
        dirpath=out_dir,
        filename="best-{step}-{val_rank_ic_near:.5f}",
        save_top_k=1,
        save_on_train_epoch_end=False,
    )
    return L.Trainer(
        max_steps=max_steps,
        precision="bf16-mixed",
        default_root_dir=out_dir,
        accelerator="cuda",
        callbacks=[best, LearningRateMonitor("step")],
        logger=CSVLogger(out_dir, name="csv"),
        val_check_interval=VAL_EVERY_STEPS,
        check_val_every_n_epoch=None,
        limit_val_batches=VAL_BATCHES,
        gradient_clip_val=1,
        gradient_clip_algorithm="norm",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    import lightning as L
    import torch

    from ophir import register
    from ophir.train import build_dataloader, build_split_handlers

    torch.set_float32_matmul_precision("high")
    L.seed_everything(args.seed, workers=True)

    base_path = os.path.join(register.get_default_data_days_dir(), "stocks")
    train_handler, val_handler = build_split_handlers(
        base_path=base_path,
        seq_len=SEQ_LEN,
        offset=WINDOW_OFFSET,
        min_volume=MIN_VOLUME,
        train_min_year=None,
        train_max_year=TRAIN_MAX_YEAR,
        val_min_year=ACCEPT_VAL_MIN_YEAR,
        val_max_year=ACCEPT_VAL_MAX_YEAR,
        use_sp500=False,
    )
    train_dl = build_dataloader(train_handler, RESPONSE_SIZE, BATCH_SIZE, NUM_WORKERS, CACHE_SIZE)
    val_dl = build_dataloader(
        val_handler, RESPONSE_SIZE, BATCH_SIZE, NUM_WORKERS, CACHE_SIZE, return_identity=True
    )

    model = MODEL_CLASS(max_steps=args.max_steps, **MODEL_KWARGS)
    trainer = build_trainer(args.out_dir, args.max_steps)
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    main()
