import lightning as L
import torch
from pytorch_optimizer.optimizer import Lamb

from .models import SingleCoinPricePredictor


class LightningPricePredictor(L.LightningModule):
    def __init__(
        self,
        emb_dim: int,
        num_enc_layers: int,
        num_dec_layers: int,
        num_heads: int,
        mean: float,
        std: float,
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.num_heads = num_heads
        self.mean = mean
        self.std = std

        self.price_predictor = SingleCoinPricePredictor(
            emb_dim, num_enc_layers, num_dec_layers, num_heads, mean, std
        )
        # self.price_loss = torch.nn.MSELoss()
        self.price_loss = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, input):
        model_output = self.price_predictor(*input)
        return model_output

    def compute_loss(self, pred_prices, true_prices):
        b, s, n_classes = pred_prices.shape
        pred_prices = pred_prices.reshape((-1, n_classes))
        true_prices = true_prices.reshape((-1, n_classes))
        price_loss = self.price_loss(pred_prices, true_prices)
        return price_loss

    def training_step(self, batch, batch_indx):
        input, output = batch
        model_output = self.forward(input)
        loss = self.compute_loss(model_output[:, :-1], output)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input, output = batch
        model_output = self.forward(input)
        loss = self.compute_loss(model_output[:, :-1], output)

        self.log("val_loss", loss, on_epoch=True, on_step=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = Lamb(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=1000, power=0.9)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
