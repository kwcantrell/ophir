from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass(kw_only=True, slots=True)
class OHLCMulitClassPredictorInput:
    """Structured dataclass for OHLCMultiClassPredictor input/output.
    Args:
        feature_input (torch.FloatTensor): shape (B, S, 13) feature vector that
            will be used as model input.
        response_size (torch.LongTensor): ()  how many days the model will predict.
        trade_occured (torch.BoolTensor): (B, S) padding mask.
        targets (torch.FloatTensor): shape (B, S, 3) targets for model output
        model_output (torch.FloatTensor): shape (B, S, 3) output for model output
        time (np.ndarray): Timestamps for each day in feature_input
        stock_embeddings (torch.FloatTensor): stock embeddings
    """

    feature_input: torch.FloatTensor
    response_size: torch.LongTensor
    trade_occured: torch.BoolTensor
    targets: torch.FloatTensor
    model_output: torch.FloatTensor = None
    time: np.ndarray | None = None
    stock_embeddings: torch.FloatTensor = None
    return_full_targets: bool = False
    r_close_index = 0
    upside_index = 1
    downside_index = 2

    def __post_init__(self):
        if len(self.feature_input.shape) < 3:
            self.feature_input = self.feature_input.unsqueeze(0)

        if len(self.targets.shape) < 3:
            self.targets = self.targets.unsqueeze(0)

        if len(self.trade_occured.shape) < 2:
            self.trade_occured = self.trade_occured.unsqueeze(0)

        if self.time is not None and len(self.time.shape) < 2:
            self.time = np.expand_dims(self.time, axis=0)

    def chunk(self, tensor: torch.Tensor, chunk_index) -> torch.Tensor:
        chunk = tensor.chunk(3, dim=-1)[chunk_index]
        if not self.return_full_targets:
            chunk = chunk[:, -self.response_size :]
        return chunk

    @property
    def target_r_close(self):
        return self.chunk(self.targets, self.r_close_index)

    @property
    def predicted_r_close(self):
        return self.chunk(self.model_output, self.r_close_index)

    @property
    def target_upside(self):
        return self.chunk(self.targets, self.upside_index)

    @property
    def predicted_upside(self):
        return self.chunk(self.model_output, self.upside_index)

    @property
    def target_downside(self):
        return self.chunk(self.targets, self.downside_index)

    @property
    def predicted_downside(self):
        return self.chunk(self.model_output, self.downside_index)

    def to_cuda(self) -> OHLCMulitClassPredictorInput:
        self.feature_input = self.feature_input.cuda()
        self.targets = self.targets.cuda()
        self.trade_occured = self.trade_occured.cuda()
        return self

    def pca_projection(self) -> np.ndarray:
        """PCA project of stock embeddings.
        Returns a list of numpy arrays (B, 3)
        """
        with torch.no_grad():
            stock_embeddings = self.stock_embeddings.mean(1)
            _u, _s, v = torch.pca_lowrank(stock_embeddings, q=3)
            transformed_data = torch.matmul(stock_embeddings, v)
        return transformed_data.cpu().numpy()

    def to_pandas(self) -> pd.DataFrame:
        """Converts model_output to a pandas.DataFrame"""
        b, _s, _ = self.feature_input.shape
        dfs = []

        target_flag = self.return_full_targets
        self.return_full_targets = True
        for i in range(b):
            target_r_close = self.target_r_close[i].reshape(-1).detach().cpu().numpy()
            predicted_r_close = self.predicted_r_close[i].reshape(-1).detach().cpu().numpy()
            predicted_r_close = np.concat(
                [target_r_close[: -self.response_size], predicted_r_close]
            )

            target_upside = self.target_upside[i].exp().reshape(-1).detach().cpu().numpy()
            predicted_upside = self.predicted_upside[i].exp().reshape(-1).detach().cpu().numpy()
            predicted_upside = np.concat([target_upside[: -self.response_size], predicted_upside])

            target_downside = (-self.target_downside[i]).exp().reshape(-1).detach().cpu().numpy()
            predicted_downside = (
                (-self.predicted_downside[i]).exp().reshape(-1).detach().cpu().numpy()
            )
            predicted_downside = np.concat(
                [target_downside[: -self.response_size], predicted_downside]
            )
            df = pd.DataFrame(
                {
                    "target_r_close": target_r_close,
                    "predicted_r_close": predicted_r_close,
                    "target_upside": target_upside,
                    "predicted_upside": predicted_upside,
                    "target_downside": target_downside,
                    "predicted_downside": predicted_downside,
                    "trade_occured": self.trade_occured[i].cpu().reshape(-1),
                    "time": self.time[i].reshape(-1),
                }
            )
            df = df.set_index("time")
            dfs.append(df)
        self.return_full_targets = target_flag
        return dfs
