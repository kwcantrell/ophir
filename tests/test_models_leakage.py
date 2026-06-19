"""Leakage regression tests for :class:`OHLCMulitClassPredictor`.

The model forecasts the last ``response_size`` days of each window. Every input
feature at those positions is contemporaneous with that day's targets, so the
response block must be masked before it reaches the transformer -- otherwise the
model could read the answer it is asked to predict. These tests pin that
masking contract; they exercise only :meth:`_apply_response_mask` and the
feature projection, so they run on CPU without the CUDA-only attention path.
"""

import torch

from ophir.models import OHLCMulitClassParameters, OHLCMulitClassPredictor

EMB_DIM = 8
SEQ_LEN = 12
RESPONSE_SIZE = 4
BATCH = 2


def _predictor() -> OHLCMulitClassPredictor:
    torch.manual_seed(0)
    hparams = OHLCMulitClassParameters(emb_dim=EMB_DIM, num_layers=1, num_heads=2)
    return OHLCMulitClassPredictor(hparams)


def test_response_block_replaced_with_mask_token():
    model = _predictor()
    x = torch.randn(BATCH, SEQ_LEN, EMB_DIM)

    masked = model._apply_response_mask(x, RESPONSE_SIZE)

    prefix_len = SEQ_LEN - RESPONSE_SIZE
    # Prefix is untouched.
    torch.testing.assert_close(masked[:, :prefix_len], x[:, :prefix_len])
    # Every response row equals the learned mask token.
    expected = model.mask_token.expand(BATCH, RESPONSE_SIZE, EMB_DIM)
    torch.testing.assert_close(masked[:, prefix_len:], expected)


def test_response_features_cannot_influence_masked_representation():
    """Perturbing the response-block inputs must not change anything downstream."""
    model = _predictor()

    feature_a = torch.randn(BATCH, SEQ_LEN, 12)
    feature_b = feature_a.clone()
    # Arbitrarily corrupt only the response-block features (incl. the target
    # columns r_close/upside/downside); the prefix is identical.
    feature_b[:, SEQ_LEN - RESPONSE_SIZE :] += 1000.0

    with torch.no_grad():
        masked_a = model._apply_response_mask(model.feature_mlp(feature_a), RESPONSE_SIZE)
        masked_b = model._apply_response_mask(model.feature_mlp(feature_b), RESPONSE_SIZE)

    # Identical -> the forecast horizon carries no information from its own day.
    torch.testing.assert_close(masked_a, masked_b)


def test_mask_token_is_a_trainable_parameter():
    model = _predictor()
    assert model.mask_token.requires_grad
    assert model.mask_token.shape == (EMB_DIM,)
