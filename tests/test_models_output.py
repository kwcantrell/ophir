"""Tests for the pure output-activation step (CPU, no flex-attention)."""

import torch

from ophir.models import apply_output_activations


def test_upside_downside_are_non_negative():
    raw = torch.tensor([[[-0.5, -3.0, -2.0], [0.5, -0.1, 4.0]]])  # (1, 2, 3)
    out = apply_output_activations(raw)

    # r_close (channel 0) is left signed/unchanged.
    torch.testing.assert_close(out[..., 0], raw[..., 0])
    # upside (1) and downside (2) are forced non-negative.
    assert torch.all(out[..., 1] >= 0)
    assert torch.all(out[..., 2] >= 0)
