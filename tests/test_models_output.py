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


def test_rezero_gate_stats_aggregates_per_layer():
    from ophir.models import OHLCMultiClassParameters, OHLCMultiClassPredictor, rezero_gate_stats

    torch.manual_seed(0)
    model = OHLCMultiClassPredictor(OHLCMultiClassParameters(emb_dim=16, num_layers=3, num_heads=2))
    # Force known gate values.
    vals = [0.1, -0.2, 0.3]
    for block, v in zip(model.encoder, vals, strict=True):
        with torch.no_grad():
            block._rezero.fill_(v)
    stats = rezero_gate_stats(model)
    assert len(stats["per_layer"]) == 3
    for actual, expected in zip(stats["per_layer"], vals, strict=True):
        assert abs(actual - expected) < 1e-6
    assert abs(stats["max_abs"] - 0.3) < 1e-6
    assert abs(stats["mean_abs"] - 0.2) < 1e-6  # mean(|0.1|,|0.2|,|0.3|)


def test_rezero_gate_stats_empty_is_zero():
    import torch.nn as nn

    from ophir.models import rezero_gate_stats

    linear = nn.Linear(2, 2)
    assert rezero_gate_stats(linear) == {"mean_abs": 0.0, "max_abs": 0.0, "per_layer": []}


def test_rezero_init_sets_gate_values():
    from ophir.models import OHLCMultiClassParameters, OHLCMultiClassPredictor, rezero_gate_stats

    model = OHLCMultiClassPredictor(
        OHLCMultiClassParameters(emb_dim=16, num_layers=2, num_heads=2, rezero_init=0.1)
    )
    per_layer = rezero_gate_stats(model)["per_layer"]
    assert len(per_layer) == 2
    for v in per_layer:
        assert abs(v - 0.1) < 1e-6


def test_rezero_init_defaults_to_zero():
    from ophir.models import OHLCMultiClassParameters, OHLCMultiClassPredictor, rezero_gate_stats

    model = OHLCMultiClassPredictor(OHLCMultiClassParameters(emb_dim=16, num_layers=2, num_heads=2))
    assert rezero_gate_stats(model)["per_layer"] == [0.0, 0.0]
