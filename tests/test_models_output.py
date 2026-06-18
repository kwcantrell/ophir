"""Tests for the pure output-activation step (CPU, no flex-attention)."""

import torch

from ophir.models import apply_output_activations, pool_prefix_embedding


def test_pool_prefix_embedding_ignores_response_block():
    # 1 example, 4 positions, 2-d; prefix=first 2 rows, response=last 2.
    x = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0], [99.0, 99.0]]])
    pooled = pool_prefix_embedding(x, response_size=2)
    torch.testing.assert_close(pooled, torch.tensor([[2.0, 2.0]]))


def test_upside_downside_are_non_negative():
    raw = torch.tensor([[[-0.5, -3.0, -2.0], [0.5, -0.1, 4.0]]])  # (1, 2, 3)
    out = apply_output_activations(raw)

    # r_close (channel 0) is left signed/unchanged.
    torch.testing.assert_close(out[..., 0], raw[..., 0])
    # upside (1) and downside (2) are forced non-negative.
    assert torch.all(out[..., 1] >= 0)
    assert torch.all(out[..., 2] >= 0)
