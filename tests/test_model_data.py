"""Tests for OHLCMulitClassPredictorInput projection/reconstruction helpers."""

import numpy as np
import torch

from ophir.model_data import OHLCMulitClassPredictorInput


def _make_input(stock_embeddings: torch.Tensor) -> OHLCMulitClassPredictorInput:
    b = stock_embeddings.shape[0]
    obj = OHLCMulitClassPredictorInput(
        feature_input=torch.zeros(b, 4, 13),
        response_size=torch.tensor(2),
        trade_occured=torch.ones(b, 4, dtype=torch.bool),
        targets=torch.zeros(b, 4, 3),
    )
    obj.stock_embeddings = stock_embeddings
    return obj


def test_pca_projection_uses_full_embedding_dimension():
    torch.manual_seed(0)
    # 6 stocks, 8-d embeddings already pooled by the model to (B, emb_dim).
    embeddings = torch.randn(6, 8)
    obj = _make_input(embeddings)

    projected = obj.pca_projection()

    assert projected.shape == (6, 3)
    # A non-degenerate projection has spread on every component; the old
    # double-mean collapsed embeddings to (B, 1), making components 2 and 3 zero.
    assert np.all(projected.std(axis=0) > 1e-6)
