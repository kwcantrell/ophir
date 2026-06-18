"""Optuna hyperparameter sweep for the OHLC forecaster.

Searches optimizer, loss-weight, and architecture-tier hyperparameters by mean
cross-sectional rank-IC on ``r_close`` (the model logs ``val_rank_ic`` when the
validation loader carries identity). Each trial runs a reduced-budget *proxy*
training with a pruning callback; the best configs are then retrained at full
budget and scored with the offline eval report (:func:`confirm_top`).

Requires CUDA for the actual trials; the pure helpers (search space, top-K
selection) are CPU-safe and unit-tested.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from lightning.pytorch.callbacks import Callback

if TYPE_CHECKING:
    import optuna
    from lightning.pytorch import LightningModule, Trainer

#: Architecture presets; each satisfies emb_dim % 4 == 0, emb_dim % num_heads
#: == 0, and head_dim >= 16 (the flex-attention CUDA floor).
SIZE_TIERS: dict[str, dict[str, int]] = {
    "small": {"emb_dim": 64, "num_layers": 4, "num_heads": 4},
    "base": {"emb_dim": 128, "num_layers": 6, "num_heads": 8},
    "large": {"emb_dim": 192, "num_layers": 8, "num_heads": 12},
}


def sample_config(trial: optuna.Trial) -> dict[str, Any]:
    """Sample one hyperparameter configuration as ``run_training`` kwargs."""
    tier = trial.suggest_categorical("size_tier", list(SIZE_TIERS))
    arch = SIZE_TIERS[tier]
    beta2 = trial.suggest_float("beta2", 0.9, 0.999)
    return {
        "emb_dim": arch["emb_dim"],
        "num_layers": arch["num_layers"],
        "num_heads": arch["num_heads"],
        "lr": trial.suggest_float("lr", 5e-5, 2e-3, log=True),
        "rezero_lr": trial.suggest_float("rezero_lr", 5e-5, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.1),
        "loss_decay": trial.suggest_float("loss_decay", 0.3, 1.0),
        "betas": (0.9, beta2),
        "upside_weight": trial.suggest_float("upside_weight", 0.25, 1.0),
        "downside_weight": trial.suggest_float("downside_weight", 0.25, 1.0),
    }


class _OptunaPruning(Callback):
    """Report ``val_rank_ic`` to an Optuna trial and prune unpromising runs."""

    def __init__(self, trial: optuna.Trial) -> None:
        self._trial = trial
        self.best_val_rank_ic: float | None = None

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        import optuna

        metric = trainer.callback_metrics.get("val_rank_ic")
        if metric is None:
            return
        value = float(metric)
        if self.best_val_rank_ic is None or value > self.best_val_rank_ic:
            self.best_val_rank_ic = value
        step = trainer.global_step
        self._trial.report(value, step)
        if self._trial.should_prune():
            raise optuna.TrialPruned(f"pruned at step {step}")


def select_top_configs(study: optuna.Study, k: int) -> list[dict[str, Any]]:
    """Return the configs of the top-``k`` completed trials, best first."""
    import optuna

    completed = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    completed.sort(key=lambda t: float(t.value), reverse=True)  # type: ignore[arg-type]
    return [t.user_attrs["config"] for t in completed[:k]]


def objective(trial: optuna.Trial, *, proxy_kwargs: dict[str, Any], base_seed: int) -> float:
    """Run one proxy trial; return its best ``val_rank_ic`` (maximize)."""
    import optuna

    config = sample_config(trial)
    trial.set_user_attr("config", config)
    from ophir.train import run_training

    pruning_cb = _OptunaPruning(trial)
    run_training(
        **proxy_kwargs,
        **config,
        val_identity=True,
        seed=base_seed + trial.number,
        callbacks=[pruning_cb],
    )
    if pruning_cb.best_val_rank_ic is None:
        raise optuna.TrialPruned("no val_rank_ic was reported")
    return pruning_cb.best_val_rank_ic


def run_sweep(
    *,
    n_trials: int,
    study_name: str,
    storage: str,
    base_seed: int,
    proxy_kwargs: dict[str, Any],
) -> optuna.Study:
    """Create/resume the SQLite study and run ``n_trials`` proxy trials."""
    import optuna

    sampler = optuna.samplers.TPESampler(seed=base_seed)
    pruner = optuna.pruners.SuccessiveHalvingPruner()
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, proxy_kwargs=proxy_kwargs, base_seed=base_seed),
        n_trials=n_trials,
    )
    return study


def confirm_top(
    study: optuna.Study,
    *,
    k: int,
    full_kwargs: dict[str, Any],
    val_batches: int,
) -> list[dict[str, Any]]:
    """Retrain the top-``k`` configs at full budget and score with the eval report.

    Returns one record per config: its hyperparameters plus the authoritative
    ``rank_ic_mean`` and per-channel skill scores from
    :func:`ophir.evaluate.evaluate_model`. Requires CUDA.
    """
    from ophir import register
    from ophir.evaluate import evaluate_model
    from ophir.train import build_dataloader, build_split_handlers, run_training

    base_path = os.path.join(
        full_kwargs.get("data_dir") or register.get_default_data_days_dir(), "stocks"
    )
    results: list[dict[str, Any]] = []
    for config in select_top_configs(study, k):
        model = run_training(**full_kwargs, **config, val_identity=True)
        _, val_handler = build_split_handlers(
            base_path=base_path,
            seq_len=full_kwargs["seq_len"],
            offset=full_kwargs["offset"],
            min_volume=full_kwargs["min_volume"],
            train_min_year=full_kwargs["train_min_year"],
            train_max_year=full_kwargs["train_max_year"],
            val_min_year=full_kwargs["val_min_year"],
            val_max_year=full_kwargs["val_max_year"],
            use_sp500=full_kwargs["use_sp500"],
        )
        val_dl = build_dataloader(
            val_handler,
            full_kwargs["response_size"],
            full_kwargs["batch_size"],
            full_kwargs["num_workers"],
            full_kwargs["cache_size"],
            return_identity=True,
        )
        report = evaluate_model(model, val_dl, val_batches)
        results.append({"config": config, "report": report})
    results.sort(
        key=lambda r: r["report"]["r_close"].get("rank_ic_mean", float("-inf")),
        reverse=True,
    )
    return results
