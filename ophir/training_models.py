from typing import Optional, Tuple

import lightning as L
import torch
import torch.nn.functional as F
from pytorch_optimizer.optimizer import Lamb

from .models import MulitClassPricePredictor, SingleCoinPricePredictor
from .pcgrad import PCGrad


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
        self.year_loss = torch.nn.CrossEntropyLoss()
        self.month_loss = torch.nn.CrossEntropyLoss()
        self.day_loss = torch.nn.CrossEntropyLoss()
        self.sector_loss = torch.nn.CrossEntropyLoss()
        self.industry_loss = torch.nn.CrossEntropyLoss()
        self.stock_loss = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.pcgrad = None

    def forward(self, input, random_mask=None):
        model_output = self.price_predictor(*input, random_mask=random_mask)
        return model_output

    def _extract_class_logit_and_lables(
        self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]
    ):
        _, len, cls = pred.shape
        if mask is not None:
            pred = pred[mask]
            target = target.expand(-1, len)[mask]
        else:
            pred = pred.transpose(2, 1).reshape(-1, cls)
            target = target.expand(-1, len).reshape(-1)
        return pred, target

    def compute_loss(
        self,
        model_outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        stock_info: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ):
        pred_val, pred_year, pred_month, pred_day, pred_sec, pred_ind, pred_stock = model_outputs
        true_val, true_year, true_month, true_day, true_sec, true_ind, true_stock = stock_info

        if mask is not None:
            mask = F.pad(mask, (0, 0, 0, pred_val.shape[1] - mask.shape[1]), value=1)
            pred_val = pred_val[mask]
            true_val = true_val[mask]
            mask = mask.squeeze(-1)

        pred_year, true_year = self._extract_class_logit_and_lables(pred_year, true_year, mask)
        pred_month, true_month = self._extract_class_logit_and_lables(pred_month, true_month, mask)
        pred_day, true_day = self._extract_class_logit_and_lables(pred_day, true_day, mask)
        # pred_sec, true_sec = self._extract_class_logit_and_lables(pred_sec, true_sec, mask)
        # pred_ind, true_ind = self._extract_class_logit_and_lables(pred_ind, true_ind, mask)
        # pred_stock, true_stock = self._extract_class_logit_and_lables(pred_stock, true_stock, mask)
        price_loss = self.price_loss(pred_val, true_val)
        year_loss = self.year_loss(pred_year, true_year)
        month_loss = self.month_loss(pred_month, true_month)
        day_loss = self.day_loss(pred_day, true_day)
        sec_loss = self.sector_loss(pred_sec, true_sec.squeeze(-1))
        ind_loss = self.industry_loss(pred_ind, true_ind.squeeze(-1))
        stock_loss = self.stock_loss(pred_stock, true_stock.squeeze(-1))
        return [price_loss, year_loss, month_loss, day_loss, sec_loss, ind_loss, stock_loss]

    def training_step(self, batch, batch_indx):
        if self.pcgrad is None:
            opt = self.optimizers()
            self.pcgrad = PCGrad(opt)
        input, true_val = batch

        random_mask = torch.rand_like(input[0]) < 0.15
        model_outputs = self.forward(input, random_mask=random_mask)
        loss = self.compute_loss(
            model_outputs,
            (true_val, input[-6], input[-5], input[-4], input[-3], input[-2], input[-1]),
            random_mask,
        )

        self.log("train_loss", loss[0], prog_bar=False, on_epoch=True, on_step=True, logger=True)
        self.log("train_year", loss[1], prog_bar=False, on_epoch=True, on_step=True, logger=True)
        self.log("train_month", loss[2], prog_bar=False, on_epoch=True, on_step=True, logger=True)
        self.log("train_day", loss[3], prog_bar=False, on_epoch=True, on_step=True, logger=True)
        self.log("train_sec", loss[4], prog_bar=False, on_epoch=True, on_step=True, logger=True)
        self.log("train_ind", loss[5], prog_bar=False, on_epoch=True, on_step=True, logger=True)
        self.log("train_stock", loss[6], prog_bar=False, on_epoch=True, on_step=True, logger=True)

        # Optimizer step
        self.pcgrad.pc_backward(loss, self.manual_backward)
        self.pcgrad.step()
        self.pcgrad.zero_grad()

        sch = self.lr_schedulers()
        sch.step()

    def validation_step(self, batch, batch_idx):
        input, true_val = batch
        model_outputs = self.forward(input)
        loss = self.compute_loss(
            model_outputs,
            (true_val, input[-6], input[-5], input[-4], input[-3], input[-2], input[-1]),
        )

        self.log("val_loss", loss[0], on_epoch=True, on_step=True, logger=True)
        self.log("val_year", loss[1], on_epoch=True, on_step=True, logger=True)
        self.log("val_month", loss[2], on_epoch=True, on_step=True, logger=True)
        self.log("val_day", loss[3], on_epoch=True, on_step=True, logger=True)
        self.log("val_sec", loss[4], on_epoch=True, on_step=True, logger=True)
        self.log("val_ind", loss[5], on_epoch=True, on_step=True, logger=True)
        self.log("val_stock", loss[6], on_epoch=True, on_step=True, logger=True)

    def configure_optimizers(self):
        optimizer = Lamb(self.parameters(), lr=1e-3)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=679836)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }


class LightningMulitClassPricePredictor(LightningPricePredictor):
    def __init__(
        self, num_sector: int, num_industry: int, num_stock: int, **kwargs: LightningPricePredictor
    ):
        super().__init__(**kwargs)
        self.num_sector = num_sector
        self.num_industry = num_industry
        self.num_stock = num_stock
        self.price_predictor = MulitClassPricePredictor(
            num_sector, num_industry, num_stock, **kwargs
        )

    def predict_step(self, batch, batch_indx):
        input, target = batch
        return (self.forward(input)[0], target)
