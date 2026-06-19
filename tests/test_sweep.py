"""CPU-safe tests for the sweep harness pure helpers (no CUDA, no Optuna run)."""

import optuna
import pytest

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
    assert config["close_weight"] == 1.0  # anchored: normalization removes its scale DOF
    assert config["upside_weight"] == 0.5
    assert config["downside_weight"] == 0.5


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


def test_build_sampler_selects_type() -> None:
    assert isinstance(sweep._build_sampler("tpe", 0), optuna.samplers.TPESampler)
    assert isinstance(sweep._build_sampler("random", 0), optuna.samplers.RandomSampler)


def test_build_sampler_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="sampler"):
        sweep._build_sampler("nope", 0)


def test_build_pruner_toggles() -> None:
    assert isinstance(sweep._build_pruner(True), optuna.pruners.SuccessiveHalvingPruner)
    assert isinstance(sweep._build_pruner(False), optuna.pruners.NopPruner)


def test_format_importances_warns_for_tpe_or_pruned() -> None:
    result = {
        "fanova": {"downside_weight": 0.6, "lr": 0.4},
        "mdi": {"downside_weight": 0.5},
        "n_completed": 30,
    }
    txt = sweep.format_importances(result, sampler="tpe", pruned=True)
    assert "WARNING" in txt
    assert "downside_weight" in txt
    # within the fANOVA section, params are listed highest-importance first
    fanova_section = txt.split("fANOVA importances:")[1].split("MDI importances:")[0]
    ranked = [ln.split()[0] for ln in fanova_section.splitlines() if ln.startswith("  ")]
    assert ranked == ["downside_weight", "lr"]


def test_format_importances_clean_study_has_no_warning() -> None:
    result = {"fanova": {"lr": 1.0}, "mdi": {"lr": 1.0}, "n_completed": 40}
    txt = sweep.format_importances(result, sampler="random", pruned=False)
    assert "WARNING" not in txt


def test_format_importances_warns_on_few_trials() -> None:
    result = {"fanova": {"lr": 1.0}, "mdi": {"lr": 1.0}, "n_completed": 3}
    txt = sweep.format_importances(result, sampler="random", pruned=False)
    assert "WARNING" in txt


def _study_with_completed_trials(n: int) -> optuna.Study:
    study = optuna.create_study(direction="maximize")
    dist = optuna.distributions.FloatDistribution(0.25, 1.0)
    for i in range(n):
        # Two varying params so fANOVA has variance to decompose.
        w = 0.25 + 0.75 * (i / max(n - 1, 1))
        study.add_trial(
            optuna.trial.create_trial(
                params={"downside_weight": w, "upside_weight": 1.0 - 0.5 * w},
                distributions={"downside_weight": dist, "upside_weight": dist},
                value=w,  # objective tracks downside_weight => high importance
            )
        )
    return study


def test_compute_importances_reports_completed_and_params() -> None:
    result = sweep.compute_importances(_study_with_completed_trials(20))
    assert result["n_completed"] == 20
    assert "downside_weight" in result["fanova"]
    assert set(result["fanova"]) == {"downside_weight", "upside_weight"}


def test_compute_importances_handles_too_few_trials() -> None:
    result = sweep.compute_importances(_study_with_completed_trials(1))
    assert result["n_completed"] == 1
    assert result["fanova"] == {}
    assert result["mdi"] == {}
