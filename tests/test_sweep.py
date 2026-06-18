"""CPU-safe tests for the sweep harness pure helpers (no CUDA, no Optuna run)."""

import optuna

from ophir import sweep


def test_size_tiers_satisfy_head_divisibility() -> None:
    for name, arch in sweep.SIZE_TIERS.items():
        emb, heads = arch["emb_dim"], arch["num_heads"]
        assert emb % 4 == 0, name
        assert emb % heads == 0, name
        assert emb // heads >= 16, name  # flex-attention floor


def test_sample_config_returns_valid_arch_and_ranges() -> None:
    # FixedTrial supplies deterministic values for each suggested param.
    trial = optuna.trial.FixedTrial(
        {
            "size_tier": "base",
            "lr": 1e-3,
            "rezero_lr": 5e-4,
            "weight_decay": 0.02,
            "warmup_ratio": 0.05,
            "loss_decay": 0.6,
            "beta2": 0.95,
            "upside_weight": 0.5,
            "downside_weight": 0.5,
        }
    )
    config = sweep.sample_config(trial)
    assert config["emb_dim"] == sweep.SIZE_TIERS["base"]["emb_dim"]
    assert config["num_heads"] == sweep.SIZE_TIERS["base"]["num_heads"]
    assert config["betas"] == (0.9, 0.95)
    assert config["lr"] == 1e-3
    assert "size_tier" not in config  # mapped into train kwargs, not passed raw


def test_select_top_configs_orders_by_value_desc() -> None:
    study = optuna.create_study(direction="maximize")
    for i, value in enumerate([0.1, 0.5, 0.3]):
        trial = optuna.trial.create_trial(
            params={"x": float(i)},
            distributions={"x": optuna.distributions.FloatDistribution(0, 10)},
            value=value,
            user_attrs={"config": {"lr": value}},
        )
        study.add_trial(trial)
    top = sweep.select_top_configs(study, k=2)
    assert [c["lr"] for c in top] == [0.5, 0.3]
