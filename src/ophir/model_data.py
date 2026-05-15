from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass(kw_only=True, slots=True)
class OHLCMulitClassPredictorInput:
    """Structured input/output container for :class:`OHLCMulitClassPredictor`.

    The same object is threaded through the model: features and targets are
    set by the caller, and ``model_output`` / ``stock_embeddings`` are written
    back by the forward pass.

    Attributes
    ----------
    feature_input : torch.FloatTensor
        Shape ``(B, S, 13)`` feature vector used as model input.
    response_size : torch.LongTensor
        Scalar; how many trailing days the model predicts.
    trade_occured : torch.BoolTensor
        Shape ``(B, S)`` padding mask (``True`` where a trade occurred).
    targets : torch.FloatTensor
        Shape ``(B, S, 3)`` ground-truth targets.
    model_output : torch.FloatTensor
        Shape ``(B, S, 3)`` model predictions (written by the model).
    time : numpy.ndarray, optional
        Timestamps for each day in ``feature_input``.
    stock_embeddings : torch.FloatTensor
        Pooled per-example stock embeddings (written by the model).
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
        """Select one of the three target/output channels.

        Parameters
        ----------
        tensor : torch.Tensor
            A ``(..., 3)`` tensor (typically ``targets`` or ``model_output``).
        chunk_index : int
            Which channel to select: ``r_close_index``, ``upside_index``, or
            ``downside_index``.

        Returns
        -------
        torch.Tensor
            The selected channel. Trimmed to the last ``response_size``
            positions unless ``return_full_targets`` is set.
        """
        chunk = tensor.chunk(3, dim=-1)[chunk_index]
        if not self.return_full_targets:
            chunk = chunk[:, -self.response_size :]
        return chunk

    @property
    def target_r_close(self):
        """Ground-truth relative close return channel of ``targets``."""
        return self.chunk(self.targets, self.r_close_index)

    @property
    def predicted_r_close(self):
        """Predicted relative close return channel of ``model_output``."""
        return self.chunk(self.model_output, self.r_close_index)

    @property
    def target_upside(self):
        """Ground-truth upside channel of ``targets``."""
        return self.chunk(self.targets, self.upside_index)

    @property
    def predicted_upside(self):
        """Predicted upside channel of ``model_output``."""
        return self.chunk(self.model_output, self.upside_index)

    @property
    def target_downside(self):
        """Ground-truth downside channel of ``targets``."""
        return self.chunk(self.targets, self.downside_index)

    @property
    def predicted_downside(self):
        """Predicted downside channel of ``model_output``."""
        return self.chunk(self.model_output, self.downside_index)

    def to_cuda(self) -> OHLCMulitClassPredictorInput:
        """Move ``feature_input``, ``targets``, and ``trade_occured`` to CUDA.

        Returns
        -------
        OHLCMulitClassPredictorInput
            ``self``, with the tensors moved in place.
        """
        self.feature_input = self.feature_input.cuda()
        self.targets = self.targets.cuda()
        self.trade_occured = self.trade_occured.cuda()
        return self

    def pca_projection(self) -> np.ndarray:
        """Project the stock embeddings onto their top 3 principal components.

        Returns
        -------
        numpy.ndarray
            A ``(B, 3)`` array of the embeddings (averaged over the sequence
            axis) projected into the leading 3-D PCA subspace.
        """
        with torch.no_grad():
            stock_embeddings = self.stock_embeddings.mean(1)
            _u, _s, v = torch.pca_lowrank(stock_embeddings, q=3)
            transformed_data = torch.matmul(stock_embeddings, v)
        return transformed_data.cpu().numpy()

    def to_pandas(self) -> pd.DataFrame:
        """Convert targets and predictions to per-stock pandas frames.

        For each example in the batch, builds a time-indexed frame holding the
        target and predicted ``r_close`` / ``upside`` / ``downside`` series
        (with predictions spliced onto the history prefix) plus the
        ``trade_occured`` mask.

        Returns
        -------
        list[pandas.DataFrame]
            One frame per batch element, indexed by ``time``.
        """
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
