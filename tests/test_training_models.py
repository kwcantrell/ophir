"""Tests for the LightningOHLCPredictor wrapper."""

import torch

from ophir.model_data import OHLCMulitClassPredictorInput
from ophir.training_models import LightningOHLCPredictor, robust_scale, val_rank_ic


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


def _toy_model_output() -> object:
    """A populated OHLCMulitClassPredictorInput with known targets/predictions."""
    # 2 examples, seq 4, response 2, 3 channels. Predictions deliberately offset
    # from targets so each channel has a non-zero loss.
    targets = torch.zeros(2, 4, 3)
    targets[..., 0] = 0.10  # r_close
    targets[..., 1] = 0.20  # upside
    targets[..., 2] = 0.30  # downside
    model_output = torch.zeros(2, 4, 3)  # all-zero predictions
    trade = torch.ones(2, 4, dtype=torch.bool)
    return OHLCMulitClassPredictorInput(
        feature_input=torch.zeros(2, 4, 12),
        response_size=torch.tensor(2),
        trade_occured=trade,
        targets=targets,
        model_output=model_output,
    )


def _build_predictor(**kwargs: float) -> LightningOHLCPredictor:
    return LightningOHLCPredictor(emb_dim=16, num_layers=1, num_heads=2, **kwargs)


def test_loss_weights_combine_components() -> None:
    logged: dict[str, float] = {}
    model = _build_predictor(upside_weight=0.4, downside_weight=0.7)
    model.loss_state = "val"
    model.log = lambda name, value, **kw: logged.__setitem__(name, float(value))  # type: ignore[method-assign]

    loss = model.compute_loss(_toy_model_output())  # type: ignore[arg-type]

    expected = (
        logged["val_r_close_loss"]
        + 0.4 * logged["val_upside_loss"]
        + 0.7 * logged["val_downside_loss"]
    ) / (1.0 + 0.4 + 0.7)
    assert abs(float(loss) - expected) < 1e-6


def test_loss_weights_default_to_half() -> None:
    model = _build_predictor()
    assert model.upside_weight == 0.5
    assert model.downside_weight == 0.5


def test_val_rank_ic_perfect_ranking_is_positive() -> None:
    # Two days (date ordinals 10 and 11), three tickers each. Predictions rank
    # the same way as targets within each day -> rank-IC == 1.0.
    pred = torch.tensor([3.0, 2.0, 1.0, 1.0, 2.0, 3.0])
    target = torch.tensor([0.3, 0.2, 0.1, 0.1, 0.2, 0.3])
    ids = torch.tensor([1, 2, 3, 1, 2, 3])
    dates = torch.tensor([10, 10, 10, 11, 11, 11])
    assert val_rank_ic(pred, target, ids, dates) > 0.99


def test_val_rank_ic_empty_is_nan() -> None:
    empty = torch.tensor([])
    result = val_rank_ic(empty, empty, empty.long(), empty.long())
    assert result != result  # NaN


def test_loss_is_invariant_to_uniform_weight_scaling() -> None:
    out_a = _toy_model_output()
    out_b = _toy_model_output()
    base = _build_predictor(close_weight=1.0, upside_weight=0.5, downside_weight=0.5)
    scaled = _build_predictor(close_weight=3.0, upside_weight=1.5, downside_weight=1.5)
    base.loss_state = scaled.loss_state = "val"
    base.log = lambda *a, **k: None  # type: ignore[method-assign]
    scaled.log = lambda *a, **k: None  # type: ignore[method-assign]
    loss_a = base.compute_loss(out_a)  # type: ignore[arg-type]
    loss_b = scaled.compute_loss(out_b)  # type: ignore[arg-type]
    assert abs(float(loss_a) - float(loss_b)) < 1e-6


def test_close_weight_defaults_to_one() -> None:
    model = _build_predictor()
    assert model.close_weight == 1.0
