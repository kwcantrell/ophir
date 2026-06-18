"""Tests for the reworked optimizer/scheduler in :mod:`ophir.training_models`.

Confirms the cosine horizon is derived from the trainer (with a ``max_steps``
fallback) rather than a hardcoded constant, and that the AdamW param groups use
the configured hyper-parameters.
"""

import math
from types import SimpleNamespace
from typing import Any

import pytest

from ophir.training_models import LightningOHLCPredictor


def _model(**kwargs: Any) -> LightningOHLCPredictor:
    return LightningOHLCPredictor(emb_dim=8, num_layers=1, num_heads=2, **kwargs)


def test_total_steps_prefers_trainer_estimate() -> None:
    model = _model(max_steps=999)
    model._trainer = SimpleNamespace(estimated_stepping_batches=1234)  # type: ignore[assignment]
    assert model._total_training_steps() == 1234


def test_total_steps_falls_back_to_max_steps_when_unsized() -> None:
    model = _model(max_steps=4242)
    model._trainer = SimpleNamespace(estimated_stepping_batches=math.inf)  # type: ignore[assignment]
    with pytest.warns(UserWarning, match="falling back to max_steps=4242"):
        assert model._total_training_steps() == 4242


def test_optimizer_groups_use_hyperparameters() -> None:
    model = _model(lr=1e-3, rezero_lr=7e-4, weight_decay=0.05, max_steps=500)
    model._trainer = SimpleNamespace(estimated_stepping_batches=500)  # type: ignore[assignment]

    config = model.configure_optimizers()
    assert isinstance(config, dict)
    groups = config["optimizer"].param_groups

    decay, no_decay, rezero = groups
    # The cosine-warmup scheduler scales live ``lr`` (to 0 at step 0), so the
    # configured base rate lives in ``initial_lr``.
    assert decay["initial_lr"] == 1e-3
    assert decay["weight_decay"] == 0.05
    assert no_decay["weight_decay"] == 0.0
    assert rezero["initial_lr"] == 7e-4
    assert rezero["weight_decay"] == 0.0
