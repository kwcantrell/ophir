"""Tests for the LightningOHLCPredictor wrapper."""

from ophir.training_models import LightningOHLCPredictor


def test_use_cache_attribute_is_removed() -> None:
    """Verify that the broken use_cache property has been removed."""
    assert not hasattr(LightningOHLCPredictor, "use_cache")
