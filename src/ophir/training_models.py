"""PyTorch-Lightning wrapper for :class:`OHLCMulitClassPredictor`."""

from __future__ import annotations

import math
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities import rank_zero_warn
from transformers import get_cosine_schedule_with_warmup

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import OptimizerLRScheduler

from .model_data import OHLCMulitClassPredictorInput
from .models import OHLCMulitClassParameters, OHLCMulitClassPredictor, rezero_gate_stats

_OFFSET_BUCKETS = (1, 2, 5, 10, 20, 40, 90)


def trading_day_offsets(trade_occured: torch.Tensor) -> torch.Tensor:
    """1-based trading-day rank of each position within its row.

    Counts only real trading days (``trade_occured``), so a surviving response
    row's offset equals its forecast lead in *trading days* — the same unit as
    :func:`ophir.ceiling.signal_decay_curve`'s leads, making the per-offset model
    IC and the signal-decay ceiling directly comparable. Padded non-trading
    positions still receive the running count but are dropped by the response mask.
    """
    return trade_occured.long().cumsum(dim=1)


def _cosine_factor(step: int, warmup: int, total: int) -> float:
    """Cosine-with-warmup LR factor matching ``get_cosine_schedule_with_warmup``."""
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def _flat_factor(step: int, warmup: int) -> float:
    """Linear warmup then a constant ``1.0`` (no decay) — for the ReZero gates."""
    if step < warmup:
        return step / max(1, warmup)
    return 1.0


def robust_scale(x: torch.Tensor, floor: float = 1e-4) -> float:
    """Gaussian-equivalent scale of ``x`` via the median absolute deviation.

    ``1.4826 * MAD`` matches the standard deviation for normal data but is
    robust to the fat tails of daily returns. Returns ``floor`` for an empty or
    zero-variance input so it is always safe as a smooth-L1 ``beta``.
    """
    if x.numel() == 0:
        return floor
    median = x.median()
    mad = (x - median).abs().median()
    return max(floor, float(1.4826 * mad.item()))


def val_rank_ic(
    pred: torch.Tensor,
    target: torch.Tensor,
    ids: torch.Tensor,
    dates: torch.Tensor,
) -> float:
    """Mean cross-sectional rank-IC of ``r_close`` over a validation epoch.

    Dedupes to one prediction per ``(ticker, date)`` and averages the daily
    Spearman rank correlation, reusing the eval module's helpers so the
    validation metric and the offline report agree. Returns ``nan`` for an
    empty input (no identity collected).
    """
    from .evaluate import dedupe_by_ticker_date, rank_ic

    if pred.numel() == 0:
        return float("nan")
    dp, dt, dd = dedupe_by_ticker_date(pred, target, ids, dates)
    return rank_ic(dp, dt, dd)["ic_mean"]


def val_rank_ic_near(
    pred: torch.Tensor,
    target: torch.Tensor,
    ids: torch.Tensor,
    dates: torch.Tensor,
    offsets: torch.Tensor,
    k: int = 5,
) -> float:
    """Pooled cross-sectional rank-IC over near forecast offsets ``1..k``.

    Restricts the validation rows to trading-day offsets ``1 <= offset <= k``
    (the near band where cross-sectional skill concentrates), then dedupes to
    one prediction per ``(ticker, date)`` and averages the daily Spearman
    rank correlation, reusing the eval module's helpers so the live metric and
    the offline report agree. Pooling the near offsets keeps each day's
    cross-section dense (overlapping windows are sparse, so the dedup drops
    almost nothing). Returns ``nan`` when no rows fall in the band.

    Parameters
    ----------
    pred, target, ids, dates : torch.Tensor
        Equal-length 1-D tensors, as accumulated for :func:`val_rank_ic`.
    offsets : torch.Tensor
        Same-length integer tensor of 1-based trading-day forecast offsets.
    k : int, optional
        Inclusive upper bound of the near band. Defaults to ``5``.

    Returns
    -------
    float
        Mean daily rank-IC over the near band, or ``nan`` if empty.
    """
    from .evaluate import dedupe_by_ticker_date, rank_ic

    if pred.numel() == 0:
        return float("nan")
    sel = (offsets >= 1) & (offsets <= k)
    if not bool(sel.any()):
        return float("nan")
    dp, dt, dd = dedupe_by_ticker_date(pred[sel], target[sel], ids[sel], dates[sel])
    return rank_ic(dp, dt, dd)["ic_mean"]


# --------------------------------------------------------------------------- #
#  Lightning wrapper
# --------------------------------------------------------------------------- #
class LightningOHLCPredictor(L.LightningModule):
    """PyTorch-Lightning wrapper around :class:`OHLCMulitClassPredictor`.

    Owns the model, the weighted smooth-L1 loss over the three targets, and
    the AdamW + cosine-warmup optimizer configuration (with a dedicated
    learning rate for the ReZero parameters).
    """

    def __init__(
        self,
        emb_dim: int,
        num_layers: int,
        num_heads: int,
        rezero_init: float = 0.0,
        lr: float = 2e-4,
        rezero_lr: float = 3e-4,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.95),
        warmup_ratio: float = 0.03,
        max_steps: int = 100000,
        loss_decay: float = 0.6,
        close_weight: float = 1.0,
        upside_weight: float = 0.5,
        downside_weight: float = 0.5,
        log_rezero_gates: bool = False,
        log_offset_ic: bool = False,
        near_offset_k: int = 5,
        decouple_rezero_schedule: bool = False,
    ) -> None:
        """Build the wrapped predictor and save hyper-parameters.

        Parameters
        ----------
        emb_dim : int
            Token embedding dimensionality (must be a multiple of 4 and
            divisible by ``num_heads``).
        num_layers : int
            Number of transformer blocks.
        num_heads : int
            Number of attention heads.
        rezero_init : float, optional
            Initial value for every ReZero gate scalar. ``0.0`` (default) starts
            each block as the identity; a positive value starts the residual
            branches partially open. An experiment knob for the depth diagnostic.
        lr : float, optional
            AdamW learning rate for the decayed and bias/norm parameter groups.
            Defaults to ``2e-4``.
        rezero_lr : float, optional
            Dedicated learning rate for the ReZero scalars. Defaults to
            ``3e-4``.
        weight_decay : float, optional
            Weight decay for the decayed (dense weight) group. Defaults to
            ``0.01``.
        betas : tuple[float, float], optional
            AdamW beta coefficients. Defaults to ``(0.9, 0.95)``.
        warmup_ratio : float, optional
            Fraction of total steps spent in linear warmup before the cosine
            decay. Defaults to ``0.03``.
        max_steps : int, optional
            Scheduler horizon used only when the trainer cannot estimate the
            total number of steps (e.g. an unsized ``IterableDataset`` driven by
            ``max_epochs``). Defaults to ``100000``.
        loss_decay : float, optional
            Time-decay weight of the furthest-future forecast day relative to the
            nearest, in ``(0, 1]``. The per-day loss weight decays geometrically
            across the response block, so errors on nearer-term days are punished
            more. ``1.0`` recovers a uniform (unweighted) loss. Defaults to
            ``0.6``.
        close_weight : float, optional
            Weight of the r_close channel in the combined loss. The three loss
            weights are normalized by their sum, so only their ratios matter.
            Defaults to ``1.0``.
        upside_weight : float, optional
            Weight of the upside channel in the combined loss (normalized by the
            weight sum; see ``close_weight``). Defaults to ``0.5``.
        downside_weight : float, optional
            Weight of the downside channel in the combined loss (normalized by
            the weight sum; see ``close_weight``). Defaults to ``0.5``.
        log_rezero_gates : bool, optional
            When ``True``, log ``rezero_mean_abs`` / ``rezero_max_abs`` each
            validation pass so the gate magnitudes are visible. Defaults to
            ``False`` (default CSV columns unchanged).
        log_offset_ic : bool, optional
            When ``True``, log ``val_rank_ic_h{1,2,5,10,20,40,90}`` at each
            validation epoch end, showing rank-IC broken out by forecast
            offset bucket. Defaults to ``False`` (default CSV columns
            unchanged).
        near_offset_k : int, optional
            Inclusive upper bound (in trading-day offsets) of the near band
            scored by ``val_rank_ic_near``. Defaults to ``5`` (the validated
            band where cross-sectional skill concentrates).
        decouple_rezero_schedule : bool, optional
            When ``True``, the ReZero param group uses a flat (warmup-then-
            constant) LR while the other groups keep the cosine decay, so the
            gates keep growing late in training. Defaults to ``False`` (single
            cosine schedule over all groups, unchanged).
        """
        super().__init__()
        hparams: OHLCMulitClassParameters = OHLCMulitClassParameters(
            emb_dim=emb_dim, num_layers=num_layers, num_heads=num_heads, rezero_init=rezero_init
        )
        self.ohlc_predictor = OHLCMulitClassPredictor(hparams=hparams)

        self.rezero_init = rezero_init
        self.lr = lr
        self.rezero_lr = rezero_lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.warmup_ratio = warmup_ratio
        self.max_steps = max_steps
        self.loss_decay = loss_decay
        self.close_weight = close_weight
        self.upside_weight = upside_weight
        self.downside_weight = downside_weight
        self.log_rezero_gates = log_rezero_gates
        self.log_offset_ic = log_offset_ic
        self.near_offset_k = near_offset_k
        self.decouple_rezero_schedule = decouple_rezero_schedule

        self.save_hyperparameters()
        self.loss_state = "train"
        self._val_ic_buffers: dict[str, list[torch.Tensor]] = {
            "pred": [],
            "target": [],
            "ids": [],
            "dates": [],
            "offsets": [],
        }

    def _input_obj(
        self, input: dict[str, Any] | OHLCMulitClassPredictorInput
    ) -> OHLCMulitClassPredictorInput:
        """Coerce a batch into a CUDA :class:`OHLCMulitClassPredictorInput`.

        Parameters
        ----------
        input : dict or OHLCMulitClassPredictorInput
            A raw collated batch (dict) or an already-built input object.

        Returns
        -------
        OHLCMulitClassPredictorInput
            The input object with its tensors moved to CUDA.
        """
        if isinstance(input, dict):
            input["response_size"] = input["response_size"][0].squeeze()
            return OHLCMulitClassPredictorInput(**input).to_cuda()
        return input.to_cuda()

    def forward(
        self, input: dict[str, Any] | OHLCMulitClassPredictorInput
    ) -> OHLCMulitClassPredictorInput:
        """Run the predictor on a batch.

        Parameters
        ----------
        input : dict or OHLCMulitClassPredictorInput
            A raw collated batch or a prepared input object.

        Returns
        -------
        OHLCMulitClassPredictorInput
            The input object with ``model_output`` and ``stock_embeddings``
            populated.
        """
        prepared = self._input_obj(input)
        return cast("OHLCMulitClassPredictorInput", self.ohlc_predictor(prepared))

    def _response_weights(
        self, response_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Build the geometric time-decay weights over the forecast horizon.

        The weight for the ``i``-th response position (``i = 0`` is the earliest
        forecast day) is ``loss_decay ** (i / (response_size - 1))``, so the
        nearest day has weight ``1.0`` and the furthest has weight
        ``loss_decay``. A ``loss_decay`` of ``1.0`` (or a single-position block)
        yields uniform weights.

        Parameters
        ----------
        response_size : int
            Number of trailing forecast positions.
        device, dtype : torch.device, torch.dtype
            Placement of the returned tensor (matched to the loss tensor).

        Returns
        -------
        torch.Tensor
            Shape ``(response_size,)`` weight vector.
        """
        if self.loss_decay == 1.0 or response_size <= 1:
            return torch.ones(response_size, device=device, dtype=dtype)
        steps = torch.arange(response_size, device=device, dtype=dtype)
        return self.loss_decay ** (steps / (response_size - 1))

    @staticmethod
    def _weighted_masked_mean(
        loss: torch.Tensor, weight: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Reduce a per-day loss with time-decay weights and a trade mask.

        Computes ``sum(w * loss * m) / sum(w * m)``, restricting the average to
        days where a trade occurred and down-weighting later forecast days.
        Dividing by the summed weights keeps the result on the same scale as a
        plain masked mean.

        Parameters
        ----------
        loss : torch.Tensor
            Shape ``(B, R, 1)`` per-day loss (``reduction="none"``).
        weight : torch.Tensor
            Shape ``(R,)`` time-decay weights.
        mask : torch.Tensor
            Shape ``(B, R)`` boolean mask (``True`` where a trade occurred).

        Returns
        -------
        torch.Tensor
            The scalar weighted masked mean.
        """
        w = weight.view(1, -1, 1)
        m = mask.unsqueeze(-1).to(loss.dtype)
        weighted = w * m
        return (loss * weighted).sum() / weighted.sum().clamp_min(1e-8)

    def compute_loss(self, model_output: OHLCMulitClassPredictorInput) -> torch.Tensor:
        """Compute the masked, weighted multi-target training loss.

        Each target uses a smooth-L1 loss restricted to days where a trade
        occurred and weighted by a geometric time-decay over the forecast
        horizon (see :meth:`_response_weights`); the components are logged and
        combined as a sum-normalized weighted mean
        ``(close_weight·close + upside_weight·upside + downside_weight·downside)
        / (close_weight + upside_weight + downside_weight)``.

        Parameters
        ----------
        model_output : OHLCMulitClassPredictorInput
            A populated forward output.

        Returns
        -------
        torch.Tensor
            The scalar combined loss.
        """
        mask = model_output.trade_occured[:, -model_output.response_size :]
        target_r_close = model_output.target_r_close
        predicted_r_close = model_output.predicted_r_close
        masked_close_target = target_r_close[mask]
        close_loss = F.smooth_l1_loss(
            predicted_r_close,
            target_r_close,
            beta=robust_scale(masked_close_target),
            reduction="none",
        )
        weight = self._response_weights(
            int(model_output.response_size), close_loss.device, close_loss.dtype
        )

        close_loss = self._weighted_masked_mean(close_loss, weight, mask)
        self.log(
            f"{self.loss_state}_r_close_loss",
            close_loss,
            prog_bar=False,
            on_epoch=True,
            on_step=True,
            logger=True,
        )

        target_upside = model_output.target_upside
        predicted_upside = model_output.predicted_upside
        upside_loss = self._weighted_masked_mean(
            F.smooth_l1_loss(
                predicted_upside,
                target_upside,
                beta=robust_scale(target_upside[mask]),
                reduction="none",
            ),
            weight,
            mask,
        )
        self.log(
            f"{self.loss_state}_upside_loss",
            upside_loss,
            prog_bar=False,
            on_epoch=True,
            on_step=True,
            logger=True,
        )

        target_downside = model_output.target_downside
        predicted_downside = model_output.predicted_downside
        downside_loss = self._weighted_masked_mean(
            F.smooth_l1_loss(
                predicted_downside,
                target_downside,
                beta=robust_scale(target_downside[mask]),
                reduction="none",
            ),
            weight,
            mask,
        )
        self.log(
            f"{self.loss_state}_downside_loss",
            downside_loss,
            prog_bar=False,
            on_epoch=True,
            on_step=True,
            logger=True,
        )

        total_weight = self.close_weight + self.upside_weight + self.downside_weight
        combined = (
            self.close_weight * close_loss
            + self.upside_weight * upside_loss
            + self.downside_weight * downside_loss
        )
        return combined / max(total_weight, 1e-8)

    def training_step(
        self,
        batch: dict[str, Any] | OHLCMulitClassPredictorInput,
        batch_indx: int,
    ) -> torch.Tensor:
        """Run one training step and log ``train_loss``.

        Parameters
        ----------
        batch : dict or OHLCMulitClassPredictorInput
            The collated training batch.
        batch_indx : int
            The batch index (unused; required by the Lightning API).

        Returns
        -------
        torch.Tensor
            The training loss for this batch.
        """
        self.loss_state = "train"
        prepared = self._input_obj(batch)
        model_output = self.forward(prepared)
        loss = self.compute_loss(model_output)

        self.log("train_loss", loss, prog_bar=False, on_epoch=True, on_step=True, logger=True)

        return loss

    def validation_step(
        self,
        batch: dict[str, Any] | OHLCMulitClassPredictorInput,
        batch_indx: int,
    ) -> torch.Tensor:
        """Run one validation step and log ``val_loss``.

        Parameters
        ----------
        batch : dict or OHLCMulitClassPredictorInput
            The collated validation batch.
        batch_indx : int
            The batch index (unused; required by the Lightning API).

        Returns
        -------
        torch.Tensor
            The validation loss for this batch.
        """
        self.loss_state = "val"
        prepared = self._input_obj(batch)
        model_output = self.forward(prepared)
        loss = self.compute_loss(model_output)
        self.log("val_loss", loss, prog_bar=False, on_epoch=True, on_step=True, logger=True)

        if (
            not self.trainer.sanity_checking
            and model_output.stock_id is not None
            and model_output.date_ordinal is not None
        ):
            rs = int(model_output.response_size)
            mask = model_output.trade_occured[:, -rs:]
            resp_dates = model_output.date_ordinal[:, -rs:]
            ids_br = model_output.stock_id.view(-1, 1).expand(-1, rs)
            offsets = trading_day_offsets(mask)
            self._val_ic_buffers["pred"].append(
                model_output.predicted_r_close[mask].reshape(-1).cpu()
            )
            self._val_ic_buffers["target"].append(
                model_output.target_r_close[mask].reshape(-1).cpu()
            )
            self._val_ic_buffers["ids"].append(ids_br[mask].reshape(-1).cpu())
            self._val_ic_buffers["dates"].append(resp_dates[mask].reshape(-1).cpu())
            self._val_ic_buffers["offsets"].append(offsets[mask].reshape(-1).cpu())

        return loss

    def on_validation_epoch_end(self) -> None:
        """Log ``val_rank_ic`` from accumulated identity, then reset buffers.

        Also logs ``rezero_mean_abs`` / ``rezero_max_abs`` when ``log_rezero_gates``
        is set. Only fires rank-IC when the validation loader carries identity
        (``stock_id`` / ``date_ordinal``); without it the buffers stay empty and
        nothing is logged, so the default training path is unchanged.
        """
        if self.log_rezero_gates:
            stats = rezero_gate_stats(self.ohlc_predictor)
            self.log(
                "rezero_mean_abs", cast("float", stats["mean_abs"]), on_epoch=True, logger=True
            )
            self.log("rezero_max_abs", cast("float", stats["max_abs"]), on_epoch=True, logger=True)
        preds = self._val_ic_buffers["pred"]
        if preds:
            ic = val_rank_ic(
                torch.cat(preds),
                torch.cat(self._val_ic_buffers["target"]),
                torch.cat(self._val_ic_buffers["ids"]),
                torch.cat(self._val_ic_buffers["dates"]),
            )
            self.log("val_rank_ic", ic, prog_bar=False, on_epoch=True, logger=True)
            near_ic = val_rank_ic_near(
                torch.cat(preds),
                torch.cat(self._val_ic_buffers["target"]),
                torch.cat(self._val_ic_buffers["ids"]),
                torch.cat(self._val_ic_buffers["dates"]),
                torch.cat(self._val_ic_buffers["offsets"]),
                self.near_offset_k,
            )
            self.log("val_rank_ic_near", near_ic, prog_bar=False, on_epoch=True, logger=True)
        if self.log_offset_ic and preds:
            from .evaluate import rank_ic_by_offset

            offset_ics = rank_ic_by_offset(
                torch.cat(self._val_ic_buffers["pred"]),
                torch.cat(self._val_ic_buffers["target"]),
                torch.cat(self._val_ic_buffers["ids"]),
                torch.cat(self._val_ic_buffers["dates"]),
                torch.cat(self._val_ic_buffers["offsets"]),
                _OFFSET_BUCKETS,
            )
            for key, ic in offset_ics.items():
                self.log(f"val_rank_ic_{key}", ic, on_epoch=True, logger=True)
        for buf in self._val_ic_buffers.values():
            buf.clear()

    def _total_training_steps(self) -> int:
        """Resolve the cosine scheduler's horizon in optimizer steps.

        Prefers the trainer's own estimate (which accounts for ``max_steps`` /
        ``max_epochs``, dataloader length, gradient accumulation, and device
        count). The streaming :class:`~ophir.ticker.StockHandlerDataset` is an
        unsized ``IterableDataset``, so an epoch-driven run (finetuning) cannot
        be estimated; in that case fall back to the ``max_steps`` hyperparameter.

        Returns
        -------
        int
            The number of optimizer steps the cosine schedule decays over.
        """
        estimated = self.trainer.estimated_stepping_batches
        if math.isfinite(estimated):
            return int(estimated)
        rank_zero_warn(
            "Trainer could not estimate the total step count (unsized "
            f"IterableDataset?); falling back to max_steps={self.max_steps} for "
            "the cosine schedule."
        )
        return self.max_steps

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure AdamW with cosine warmup and ReZero-specific grouping.

        Parameters are split into three groups — weight-decayed dense
        weights, non-decayed (bias / norm) weights, and ReZero scalars (which
        get their own learning rate). All rates, decay, betas, and the warmup
        fraction come from the constructor hyper-parameters, and the cosine
        horizon is derived from the trainer (see :meth:`_total_training_steps`).

        Returns
        -------
        dict
            A Lightning optimizer configuration with a step-interval cosine
            warmup scheduler.
        """
        decay, no_decay, rezero_params = [], [], []

        for name, p in self.named_parameters():
            if p.ndim >= 2 and "bias" not in name and "norm" not in name and "rezero" not in name:
                decay.append(p)
            elif "rezero" not in name:
                no_decay.append(p)
            else:
                rezero_params.append(p)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
                {"params": rezero_params, "lr": self.rezero_lr, "weight_decay": 0.0},
            ],
            lr=self.lr,
            betas=self.betas,
        )
        total_steps = self._total_training_steps()
        warmup_steps = int(self.warmup_ratio * total_steps)
        if self.decouple_rezero_schedule:
            cosine = partial(_cosine_factor, warmup=warmup_steps, total=total_steps)
            flat = partial(_flat_factor, warmup=warmup_steps)
            scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=[cosine, cosine, flat]
            )
        else:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, warmup_steps, num_training_steps=total_steps
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def reset_rezero(self) -> None:
        """Reset every ReZero scalar to the configured ``rezero_init``, in place."""
        with torch.no_grad():
            for name, param in self.ohlc_predictor.named_parameters():
                if "rezero" in name:
                    param.fill_(self.rezero_init)
