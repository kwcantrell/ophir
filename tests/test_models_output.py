"""Tests for the pure output-activation step (CPU, no flex-attention)."""

import torch

from ophir.models import apply_output_activations, pool_prefix_embedding


def test_pool_prefix_embedding_ignores_response_block():
    # 1 example, 4 positions, 2-d; prefix=first 2 rows, response=last 2.
    x = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0], [99.0, 99.0]]])
    trade = torch.ones(1, 4, dtype=torch.bool)
    pooled = pool_prefix_embedding(x, response_size=2, trade_occured=trade)
    torch.testing.assert_close(pooled, torch.tensor([[2.0, 2.0]]))


def test_pool_prefix_embedding_masks_padded_prefix_positions():
    # Prefix rows are [1,1] (padded) and [3,3] (valid); only the valid row counts.
    x = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0], [99.0, 99.0]]])
    trade = torch.tensor([[False, True, True, True]])
    pooled = pool_prefix_embedding(x, response_size=2, trade_occured=trade)
    torch.testing.assert_close(pooled, torch.tensor([[3.0, 3.0]]))


def test_pool_prefix_embedding_all_padded_falls_back_to_mean():
    x = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0], [99.0, 99.0]]])
    # positions 2-3 are the response block; the prefix (0-1) is all no-trade,
    # so there are no valid prefix positions to pool.
    trade = torch.tensor([[False, False, True, True]])
    pooled = pool_prefix_embedding(x, response_size=2, trade_occured=trade)
    torch.testing.assert_close(pooled, torch.tensor([[2.0, 2.0]]))  # unmasked prefix mean


def test_upside_downside_are_non_negative():
    raw = torch.tensor([[[-0.5, -3.0, -2.0], [0.5, -0.1, 4.0]]])  # (1, 2, 3)
    out = apply_output_activations(raw)

    # r_close (channel 0) is left signed/unchanged.
    torch.testing.assert_close(out[..., 0], raw[..., 0])
    # upside (1) and downside (2) are forced non-negative.
    assert torch.all(out[..., 1] >= 0)
    assert torch.all(out[..., 2] >= 0)
