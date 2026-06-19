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


def test_reset_rezero_restores_configured_init() -> None:
    import torch

    model = LightningOHLCPredictor(emb_dim=16, num_layers=2, num_heads=2, rezero_init=0.1)
    for block in model.ohlc_predictor.encoder:
        with torch.no_grad():
            block._rezero.fill_(0.5)
    model.reset_rezero()
    for block in model.ohlc_predictor.encoder:
        assert abs(float(block._rezero.detach()) - 0.1) < 1e-6


def test_log_rezero_gates_logs_when_enabled() -> None:
    logged: dict[str, float] = {}
    model = LightningOHLCPredictor(
        emb_dim=16, num_layers=2, num_heads=2, rezero_init=0.1, log_rezero_gates=True
    )
    model.log = lambda name, value, **kw: logged.__setitem__(name, float(value))  # type: ignore[method-assign]
    # Bypass the val_rank_ic branch (needs a trainer); it is guarded by empty buffers.
    model.on_validation_epoch_end()
    assert abs(logged["rezero_mean_abs"] - 0.1) < 1e-6
    assert abs(logged["rezero_max_abs"] - 0.1) < 1e-6


def test_log_rezero_gates_silent_when_disabled() -> None:
    logged: dict[str, float] = {}
    model = LightningOHLCPredictor(emb_dim=16, num_layers=2, num_heads=2)
    model.log = lambda name, value, **kw: logged.__setitem__(name, float(value))  # type: ignore[method-assign]
    model.on_validation_epoch_end()
    assert "rezero_mean_abs" not in logged


def test_lr_factor_helpers() -> None:
    from ophir.training_models import _cosine_factor, _flat_factor

    # Warmup ramps linearly for both.
    assert abs(_cosine_factor(0, 10, 100) - 0.0) < 1e-9
    assert abs(_cosine_factor(5, 10, 100) - 0.5) < 1e-9
    assert abs(_flat_factor(5, 10) - 0.5) < 1e-9
    # End of training: cosine decays to ~0, flat stays at 1.0.
    assert _cosine_factor(100, 10, 100) < 1e-6
    assert abs(_flat_factor(100, 10) - 1.0) < 1e-9
    # Start of decay (just past warmup): cosine ~1.0.
    assert abs(_cosine_factor(10, 10, 100) - 1.0) < 1e-6


def test_decoupled_schedule_keeps_rezero_flat() -> None:
    model = LightningOHLCPredictor(
        emb_dim=16, num_layers=2, num_heads=2, warmup_ratio=0.1, decouple_rezero_schedule=True
    )
    # configure_optimizers calls self._total_training_steps() (which reads the
    # trainer); override it so the test needs no Trainer.
    model._total_training_steps = lambda: 100  # type: ignore[method-assign]
    cfg = model.configure_optimizers()
    sched = cfg["lr_scheduler"]["scheduler"]
    opt = cfg["optimizer"]
    # LambdaLR applies the factor immediately on construction (step=0 → factor=0 during warmup),
    # so g["lr"] is already 0 at this point; read initial_lr to get the configured base rates.
    base = [g["initial_lr"] for g in opt.param_groups]
    for _ in range(100):
        opt.step()
        sched.step()
    final = [g["lr"] for g in opt.param_groups]
    # Groups 0/1 (cosine) have decayed to ~0; group 2 (rezero, flat) holds its base lr.
    assert final[0] < base[0] * 0.05
    assert abs(final[2] - base[2]) < base[2] * 0.05
