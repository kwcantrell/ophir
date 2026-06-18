"""Tests for the LightningOHLCPredictor wrapper."""

import torch

from ophir.training_models import LightningOHLCPredictor, robust_scale


def test_use_cache_attribute_is_removed() -> None:
    """Verify that the broken use_cache property has been removed."""
    assert not hasattr(LightningOHLCPredictor, "use_cache")


def test_robust_scale_recovers_gaussian_std() -> None:
    torch.manual_seed(0)
    x = torch.randn(10_000) * 0.02
    # MAD-based scale of a Gaussian approximates its std (~0.02 here).
    assert abs(robust_scale(x) - 0.02) < 0.002


def test_robust_scale_floors_on_empty_or_constant() -> None:
    assert robust_scale(torch.tensor([])) == 1e-4
    assert robust_scale(torch.zeros(100)) == 1e-4
