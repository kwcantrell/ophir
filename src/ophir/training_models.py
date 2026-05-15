"""PyTorch-Lightning wrapper for :class:`OHLCMulitClassPredictor`."""

from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

# Import the dataclass that holds the raw input features.  It is
# defined in ``model_data.py`` - the import is kept at the top of the
# file so that the Lightning module can be used as a standalone
# example.
from .model_data import OHLCMulitClassPredictorInput  # type: ignore
from .models import OHLCMulitClassParameters, OHLCMulitClassPredictor


# --------------------------------------------------------------------------- #
#  Lightning wrapper
# --------------------------------------------------------------------------- #
class LightningOHLCPredictor(L.LightningModule):
    def __init__(self, emb_dim: int, num_layers: int, num_heads: int) -> None:
        super().__init__()
        hparams: OHLCMulitClassParameters = OHLCMulitClassParameters(
            emb_dim=emb_dim, num_layers=num_layers, num_heads=num_heads
        )
        self.ohlc_predictor = OHLCMulitClassPredictor(hparams=hparams)

        self.save_hyperparameters()
        self._use_cache = False
        self.loss_state = "train"

    def _input_obj(self, input):
        if isinstance(input, dict):
            input["response_size"] = input["response_size"][0].squeeze()
            input = OHLCMulitClassPredictorInput(**input).to_cuda()
        return input.to_cuda()

    def forward(self, input: dict | OHLCMulitClassPredictorInput) -> OHLCMulitClassPredictorInput:
        input = self._input_obj(input)
        return self.ohlc_predictor(input)

    def compute_loss(self, model_output: OHLCMulitClassPredictorInput):
        mask = model_output.trade_occured[:, -model_output.response_size :]
        target_r_close = model_output.target_r_close
        predicted_r_close = model_output.predicted_r_close
        close_loss = F.smooth_l1_loss(
            predicted_r_close, target_r_close, beta=0.01, reduction="none"
        )

        close_loss = close_loss[mask].mean()
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
        upside_loss = F.smooth_l1_loss(
            predicted_upside, target_upside, beta=0.02, reduction="none"
        )[mask].mean()
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
        downside_loss = F.smooth_l1_loss(
            predicted_downside, target_downside, beta=0.02, reduction="none"
        )[mask].mean()
        self.log(
            f"{self.loss_state}_downside_loss",
            downside_loss,
            prog_bar=False,
            on_epoch=True,
            on_step=True,
            logger=True,
        )

        return close_loss + 0.5 * upside_loss + 0.5 * downside_loss

    def training_step(self, batch, batch_indx):
        self.loss_state = "train"
        batch = self._input_obj(batch)
        model_output = self.forward(batch)
        loss = self.compute_loss(model_output)

        self.log("train_loss", loss, prog_bar=False, on_epoch=True, on_step=True, logger=True)

        return loss

    def validation_step(self, batch, batch_indx):
        self.loss_state = "val"
        batch = self._input_obj(batch)
        model_output = self.forward(batch)
        loss = self.compute_loss(model_output)
        self.log("val_loss", loss, prog_bar=False, on_epoch=True, on_step=True, logger=True)

        return loss

    def configure_optimizers(self):
        decay, no_decay, rezero_params = [], [], []

        for name, p in self.named_parameters():
            if p.ndim >= 2 and "bias" not in name and "norm" not in name and "rezero" not in name:
                decay.append(p)
            elif "rezero" not in name:
                no_decay.append(p)
            else:
                print(name)
                rezero_params.append(p)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": 0.01},
                {"params": no_decay, "weight_decay": 0.0},
                {"params": rezero_params, "lr": 3e-4, "weight_decay": 0.0},
            ],
            lr=2e-4,
            betas=(0.9, 0.95),
        )
        steps = 100000
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, int(0.03 * steps), num_training_steps=steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    @property
    def use_cache(self):
        return self._use_cache

    @use_cache.setter
    def use_cache(self, value) -> None:
        assert isinstance(value, bool)
        self.ohlc_predictor.ohlc_percentage_change.use_cache = value
        self.ohlc_predictor.volume_percentage_change.use_cache = value

    def reset_rezero(self) -> None:
        with torch.no_grad():
            for name, param in self.ohlc_predictor.named_parameters():
                if "rezero" in name:
                    param.fill_(0)
