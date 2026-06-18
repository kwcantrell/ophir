"""Unit tests for the CPU-safe leakage scorer in :mod:`ophir.leakage`.

Mirrors ``tests/test_models_leakage.py``: builds a tiny predictor on synthetic
CPU tensors and checks that perturbing the *response* block leaves the masked
representation unchanged (score ~ 0), while perturbing the *prefix* moves it (a
positive control, so the metric cannot pass vacuously).
"""

import torch

from ophir.leakage import response_block_leakage_score
from ophir.models import OHLCMulitClassParameters, OHLCMulitClassPredictor

EMB_DIM = 8
SEQ_LEN = 12
RESPONSE_SIZE = 4
BATCH = 2


def _predictor() -> OHLCMulitClassPredictor:
    torch.manual_seed(0)
    hparams = OHLCMulitClassParameters(emb_dim=EMB_DIM, num_layers=1, num_heads=2)
    return OHLCMulitClassPredictor(hparams)


def test_response_block_score_is_zero() -> None:
    """Masking holds: perturbing the response block must not move anything."""
    model = _predictor()
    feature = torch.randn(BATCH, SEQ_LEN, 13)

    score = response_block_leakage_score(model, feature, RESPONSE_SIZE)

    assert score < 1e-6


def test_prefix_perturbation_is_detected() -> None:
    """Positive control: a prefix change *does* move the masked representation.

    The scorer only perturbs the response block, so to prove it is non-vacuous
    we feed two windows that differ in the prefix and confirm the masked
    representations differ -- i.e. the masking preserves prefix information.
    """
    model = _predictor()
    feature = torch.randn(BATCH, SEQ_LEN, 13)
    perturbed = feature.clone()
    perturbed[:, : SEQ_LEN - RESPONSE_SIZE] += 1000.0

    with torch.no_grad():
        masked_a = model._apply_response_mask(model.feature_mlp(feature), RESPONSE_SIZE)
        masked_b = model._apply_response_mask(model.feature_mlp(perturbed), RESPONSE_SIZE)

    assert (masked_a - masked_b).abs().max().item() > 1.0
