from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch


@dataclass(kw_only=True, slots=True)
class OHLCMultiClassPredictorInput:
    """Structured input/output container for :class:`OHLCMultiClassPredictor`.

    The same object is threaded through the model: features and targets are
    set by the caller, and ``model_output`` / ``stock_embeddings`` are written
    back by the forward pass.

    Attributes
    ----------
    feature_input : torch.FloatTensor
        Shape ``(B, S, 12)`` feature vector used as model input.
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
    stock_id : torch.LongTensor, optional
        Per-example integer stock id ``(B,)``, carried only on the eval path
        (opt-in) so predictions can be grouped by ticker. ``None`` in training.
    date_ordinal : torch.LongTensor, optional
        Per-day calendar-day ordinal ``(B, S)``, carried only on the eval path
        (opt-in) so a day's cross-section can be identified. ``None`` in training.
    stock_embeddings : torch.FloatTensor
        Pooled per-example stock embeddings (written by the model).
    """

    feature_input: torch.Tensor
    response_size: torch.Tensor
    trade_occured: torch.Tensor
    targets: torch.Tensor
    model_output: torch.Tensor | None = None
    time: np.ndarray[Any, Any] | None = None
    stock_id: torch.Tensor | None = None
    date_ordinal: torch.Tensor | None = None
    stock_embeddings: torch.Tensor | None = None
    return_full_targets: bool = False
    r_close_index: ClassVar[int] = 0
    upside_index: ClassVar[int] = 1
    downside_index: ClassVar[int] = 2

    def __post_init__(self) -> None:
        if len(self.feature_input.shape) < 3:
            self.feature_input = self.feature_input.unsqueeze(0)

        if len(self.targets.shape) < 3:
            self.targets = self.targets.unsqueeze(0)

        if len(self.trade_occured.shape) < 2:
            self.trade_occured = self.trade_occured.unsqueeze(0)

        if self.time is not None and len(self.time.shape) < 2:
            self.time = np.expand_dims(self.time, axis=0)

    def chunk(self, tensor: torch.Tensor, chunk_index: int) -> torch.Tensor:
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
    def target_r_close(self) -> torch.Tensor:
        """Ground-truth relative close return channel of ``targets``."""
        return self.chunk(self.targets, self.r_close_index)

    @property
    def predicted_r_close(self) -> torch.Tensor:
        """Predicted relative close return channel of ``model_output``."""
        assert self.model_output is not None
        return self.chunk(self.model_output, self.r_close_index)

    @property
    def target_upside(self) -> torch.Tensor:
        """Ground-truth upside channel of ``targets``."""
        return self.chunk(self.targets, self.upside_index)

    @property
    def predicted_upside(self) -> torch.Tensor:
        """Predicted upside channel of ``model_output``."""
        assert self.model_output is not None
        return self.chunk(self.model_output, self.upside_index)

    @property
    def target_downside(self) -> torch.Tensor:
        """Ground-truth downside channel of ``targets``."""
        return self.chunk(self.targets, self.downside_index)

    @property
    def predicted_downside(self) -> torch.Tensor:
        """Predicted downside channel of ``model_output``."""
        assert self.model_output is not None
        return self.chunk(self.model_output, self.downside_index)

    def to_cuda(self) -> OHLCMultiClassPredictorInput:
        """Move ``feature_input``, ``targets``, and ``trade_occured`` to CUDA.

        Returns
        -------
        OHLCMultiClassPredictorInput
            ``self``, with the tensors moved in place.
        """
        self.feature_input = self.feature_input.cuda()
        self.targets = self.targets.cuda()
        self.trade_occured = self.trade_occured.cuda()
        return self

    def pca_projection(self) -> np.ndarray[Any, Any]:
        """Project the stock embeddings onto their top 3 principal components.

        Returns
        -------
        numpy.ndarray
            A ``(B, 3)`` array of the per-stock embeddings projected into the
            leading 3-D PCA subspace.
        """
        assert self.stock_embeddings is not None
        with torch.no_grad():
            stock_embeddings = self.stock_embeddings
            _u, _s, v = torch.pca_lowrank(stock_embeddings, q=3)
            transformed_data = torch.matmul(stock_embeddings, v)
        return transformed_data.cpu().numpy()

    def to_pandas(self) -> list[pd.DataFrame]:
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
        assert self.time is not None
        b, _s, _ = self.feature_input.shape
        dfs: list[pd.DataFrame] = []

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
