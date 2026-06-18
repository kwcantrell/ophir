"""Tests for the time-decay loss weighting in :mod:`ophir.training_models`.

The forecast loss down-weights later (further-future) days so nearer-term
errors are punished more (see :meth:`LightningOHLCPredictor._response_weights`).
These exercise the two pure reduction helpers directly, so they run on CPU
without the CUDA-only forward pass.
"""

from typing import Any

import torch

from ophir.training_models import LightningOHLCPredictor


def _model(**kwargs: Any) -> LightningOHLCPredictor:
    return LightningOHLCPredictor(emb_dim=8, num_layers=1, num_heads=2, **kwargs)


def test_response_weights_decay_monotonically() -> None:
    model = _model(loss_decay=0.6)
    w = model._response_weights(5, torch.device("cpu"), torch.float32)

    assert w.shape == (5,)
    assert torch.all(w[:-1] >= w[1:])  # non-increasing across the horizon
    assert w[0].item() == 1.0  # nearest day unweighted
    assert torch.isclose(w[-1], torch.tensor(0.6))  # furthest day == loss_decay


def test_response_weights_uniform_when_decay_one() -> None:
    model = _model(loss_decay=1.0)
    w = model._response_weights(4, torch.device("cpu"), torch.float32)
    assert torch.allclose(w, torch.ones(4))


def test_response_weights_single_position() -> None:
    model = _model(loss_decay=0.6)
    w = model._response_weights(1, torch.device("cpu"), torch.float32)
    assert torch.allclose(w, torch.ones(1))


def test_weighted_masked_mean_matches_plain_mean_when_uniform() -> None:
    loss = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])  # (1, 4, 1)
    mask = torch.tensor([[True, True, False, True]])  # (1, 4)
    weight = torch.ones(4)

    got = LightningOHLCPredictor._weighted_masked_mean(loss, weight, mask)
    expected = loss[mask].mean()
    assert torch.isclose(got, expected)


def test_weighted_masked_mean_excludes_padding_and_weights() -> None:
    loss = torch.tensor([[[1.0], [2.0], [4.0]]])  # (1, 3, 1)
    mask = torch.tensor([[True, False, True]])  # position 1 padded out
    weight = torch.tensor([1.0, 0.5, 0.25])

    got = LightningOHLCPredictor._weighted_masked_mean(loss, weight, mask)
    # only positions 0 and 2 contribute: (1*1.0 + 4*0.25) / (1.0 + 0.25)
    expected = torch.tensor((1.0 * 1.0 + 4.0 * 0.25) / (1.0 + 0.25))
    assert torch.isclose(got, expected)
