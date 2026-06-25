"""Quantify response-block target leakage in the OHLC forecaster.

The model forecasts the last ``response_size`` days of each window. Every input
feature at those positions is contemporaneous with that day's targets, so the
response block must be replaced with a learned mask token before the transformer
(:meth:`~ophir.models.OHLCMultiClassPredictor._apply_response_mask`) — otherwise
the model can read the answer it is asked to predict.

These helpers turn that contract into a *score*: perturb only the response-block
inputs and measure how much the model changes. ``0`` means the masking holds.

* :func:`response_block_leakage_score` is CPU-safe — it exercises only the
  feature projection and the masking helper, mirroring
  ``tests/test_models_leakage.py``.
* :func:`end_to_end_leakage_scores` runs the full forward (CUDA flex-attention)
  and reports the change per target channel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from .model_data import OHLCMultiClassPredictorInput

if TYPE_CHECKING:
    from .models import OHLCMultiClassPredictor

#: Magnitude added to the response-block inputs; large enough that any leakage
#: would dominate floating-point noise in the resulting score.
_PERTURBATION = 1000.0


def response_block_leakage_score(
    model: OHLCMultiClassPredictor,
    feature: torch.Tensor,
    response_size: int,
) -> float:
    """Score leakage from the masking stage alone (CPU-safe).

    Projects and masks both the clean features and a copy whose response block
    has been perturbed, then returns the maximum absolute difference between the
    two masked representations. A correctly masked model yields ``0.0``.

    Parameters
    ----------
    model : OHLCMultiClassPredictor
        The predictor whose ``feature_mlp`` and ``_apply_response_mask`` are
        exercised.
    feature : torch.Tensor
        Input features of shape ``(B, S, 12)``.
    response_size : int
        Number of trailing positions that form the forecast horizon.

    Returns
    -------
    float
        Maximum absolute change in the masked representation; ``0.0`` when the
        response block carries no information past the mask.
    """
    perturbed = feature.clone()
    perturbed[:, -response_size:] += _PERTURBATION
    with torch.no_grad():
        masked_clean = model._apply_response_mask(model.feature_mlp(feature), response_size)
        masked_perturbed = model._apply_response_mask(model.feature_mlp(perturbed), response_size)
    return float((masked_clean - masked_perturbed).abs().max().item())


def end_to_end_leakage_scores(
    model: OHLCMultiClassPredictor,
    feature: torch.Tensor,
    response_size: int,
) -> dict[str, float]:
    """Score leakage through the full forward, per target channel.

    Runs the predictor on the clean inputs and on a copy whose response block is
    perturbed, then reports the maximum absolute change in ``model_output`` for
    each target (``r_close`` / ``upside`` / ``downside``). All-zero means the
    forecast does not depend on its own day's inputs.

    Requires CUDA (the model's flex-attention path is CUDA-only).

    Parameters
    ----------
    model : OHLCMultiClassPredictor
        The predictor to run.
    feature : torch.Tensor
        Input features of shape ``(B, S, 12)`` on a CUDA device.
    response_size : int
        Number of trailing positions that form the forecast horizon.

    Returns
    -------
    dict[str, float]
        Maximum absolute output change keyed by ``"r_close"``, ``"upside"``,
        ``"downside"``.
    """

    def _run(feat: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = feat.shape
        model_input = OHLCMultiClassPredictorInput(
            feature_input=feat,
            targets=torch.zeros((batch, seq_len, 3), device=feat.device),
            trade_occured=torch.ones((batch, seq_len), dtype=torch.bool, device=feat.device),
            response_size=torch.tensor(response_size),
        )
        with torch.no_grad():
            result = cast("OHLCMultiClassPredictorInput", model(model_input))
        output = result.model_output
        assert output is not None
        return output.detach().clone()

    perturbed = feature.clone()
    perturbed[:, -response_size:] += _PERTURBATION
    diff = (_run(feature) - _run(perturbed)).abs()
    return {
        "r_close": float(diff[..., OHLCMultiClassPredictorInput.r_close_index].max().item()),
        "upside": float(diff[..., OHLCMultiClassPredictorInput.upside_index].max().item()),
        "downside": float(diff[..., OHLCMultiClassPredictorInput.downside_index].max().item()),
    }
