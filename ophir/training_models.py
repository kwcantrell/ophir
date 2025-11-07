from typing import Optional

import lightning as L
import torch
import torch.nn.functional as F
from pytorch_optimizer.optimizer import Lamb

from .models import SingleCoinPricePredictor


class LightningPricePredictor(L.LightningModule):
    def __init__(
        self,
        emb_dim: int,
        num_dec_layers: int,
        num_heads: int,
        min_value: float,
        max_value: float,
        elements_in_encoder: int,
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.num_dec_layers = num_dec_layers
        self.num_heads = num_heads
        self.min_value = min_value
        self.max_value = max_value
        self.elements_in_encoder = elements_in_encoder

        self.price_predictor = SingleCoinPricePredictor(
            emb_dim, num_dec_layers, num_heads, min_value, max_value, elements_in_encoder
        )
        self.price_loss = torch.nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, input, random_mask=None):
        model_output = self.price_predictor(*input, random_mask=random_mask)
        return model_output

    def compute_loss(
        self,
        pred_prices: torch.Tensor,
        true_prices: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        if mask is not None:
            mask = F.pad(mask, (0, 0, 0, pred_prices.shape[1] - mask.shape[1]), value=1)
            pred_prices = pred_prices[mask]
            true_prices = true_prices[mask]

        price_loss = self.price_loss(pred_prices, true_prices)
        return price_loss

    def training_step(self, batch, batch_indx):
        input, output = batch

        random_mask = torch.rand_like(input[0]) < 0.15
        model_output = self.forward(input, random_mask=random_mask)
        loss = self.compute_loss(model_output, output, random_mask)

        self.log("train_loss", loss, prog_bar=False, on_epoch=True, on_step=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input, output = batch
        model_output = self.forward(input)
        loss = self.compute_loss(model_output, output)

        self.log("val_loss", loss, on_epoch=True, on_step=True, logger=True)
        return loss

    def predict_step(self, batch, batch_indx):
        (enc_inp, dec_inp), target = batch
        num_enc, num_dec = enc_inp.shape[1], dec_inp.shape[1] + 1
        dec_inp = None
        for _ in torch.arange(num_dec):
            output = self.forward((enc_inp, dec_inp))
            dec_inp = output[:, num_enc:]
        return (output, target)

    def configure_optimizers(self):
        optimizer = Lamb(self.parameters(), lr=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=1000, power=0.9)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
