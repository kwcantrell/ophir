"""Diagnostic Lightning callbacks for the ophir training loop.

These are observational only — they log signals and never alter optimization, so
they cannot themselves cause a training failure. Attach via the
``extra_callbacks`` argument of :func:`ophir.register.fetch_base_trainer`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from lightning.pytorch.callbacks import Callback

if TYPE_CHECKING:
    import lightning.pytorch as pl
    from torch.optim import Optimizer


class GradNormMonitor(Callback):
    """Log the global gradient L2 norm before each optimizer step.

    The hook fires after ``backward`` but before Lightning applies
    ``gradient_clip_val``, so the logged ``grad_norm`` is the *pre-clip* norm —
    the quantity that reveals a gradient explosion building toward a NaN. (A
    post-clip norm would always read at or below the clip threshold and hide the
    signal.) The norm is accumulated on-device with a single host sync per step.
    """

    def on_before_optimizer_step(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer: Optimizer
    ) -> None:
        total = torch.zeros((), device=pl_module.device)
        for param in pl_module.parameters():
            grad = param.grad
            if grad is not None:
                total = total + grad.detach().pow(2).sum()
        pl_module.log(
            "grad_norm",
            total.sqrt(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
