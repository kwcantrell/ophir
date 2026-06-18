"""Score a trained checkpoint on the held-out validation set.

The trainer only logs a single combined ``val_loss``; this module turns a
checkpoint into an interpretable, per-target accuracy report over the date-split
validation set (the same split :mod:`ophir.train` builds, so the windows are
genuinely held out).

Like :mod:`ophir.leakage`, the metric core is **pure and CPU-safe** — the
``target_metrics`` / ``directional_accuracy`` / ``skill_score`` helpers take
plain tensors and are covered by ``tests/test_evaluate.py`` — while
:func:`accumulate_targets` runs the full CUDA forward to collect the masked
predictions/targets the metrics consume.

The forecaster predicts three channels per response-block day: ``r_close``
(relative close return), ``upside`` and ``downside`` (log-space magnitudes).
Predictions and targets are restricted to the response block and to days where a
trade occurred, mirroring the masking in
:meth:`~ophir.training_models.LightningOHLCPredictor.compute_loss`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import typer

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.utils.data import DataLoader

    from ophir.training_models import LightningOHLCPredictor

#: Channels predicted by the model, in report order. ``r_close`` additionally
#: gets a directional accuracy and a zero-baseline skill score (it is a signed,
#: zero-centred return; the others are non-negative log-magnitudes).
_TARGETS = ("r_close", "upside", "downside")

#: Metric columns, in report order. Not every target reports every metric.
_METRIC_ORDER = ("n", "mae", "rmse", "bias", "directional_accuracy", "skill_score")


def target_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Compute error metrics between masked prediction and target tensors.

    Parameters
    ----------
    pred, target : torch.Tensor
        1-D tensors of equal length holding the predicted and ground-truth
        values for one channel (already masked to the response block / trading
        days).

    Returns
    -------
    dict[str, float]
        ``n`` (sample count), ``mae`` (mean absolute error), ``rmse`` (root mean
        squared error), and ``bias`` (mean signed error). The error metrics are
        ``nan`` when there are no samples.
    """
    n = pred.numel()
    if n == 0:
        return {"n": 0.0, "mae": float("nan"), "rmse": float("nan"), "bias": float("nan")}
    error = pred - target
    return {
        "n": float(n),
        "mae": float(error.abs().mean().item()),
        "rmse": float(error.pow(2).mean().sqrt().item()),
        "bias": float(error.mean().item()),
    }


def directional_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Fraction of samples where the prediction's sign matches the target's.

    Parameters
    ----------
    pred, target : torch.Tensor
        1-D tensors of equal length (typically the ``r_close`` channel).

    Returns
    -------
    float
        Mean of ``sign(pred) == sign(target)``; ``nan`` when there are no
        samples.
    """
    if pred.numel() == 0:
        return float("nan")
    return float((pred.sign() == target.sign()).float().mean().item())


def skill_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """RMSE skill score against the zero-return baseline.

    ``1 - rmse(pred) / rmse(0)``: positive means the model beats predicting a
    constant zero return, ``0`` means it ties the baseline, and negative means it
    does worse. Meaningful only for the zero-centred ``r_close`` channel.

    Parameters
    ----------
    pred, target : torch.Tensor
        1-D tensors of equal length.

    Returns
    -------
    float
        The skill score; ``nan`` when there are no samples or the baseline RMSE
        is zero.
    """
    if pred.numel() == 0:
        return float("nan")
    rmse_model = (pred - target).pow(2).mean().sqrt().item()
    rmse_baseline = target.pow(2).mean().sqrt().item()
    if rmse_baseline == 0:
        return float("nan")
    return float(1.0 - rmse_model / rmse_baseline)


def accumulate_targets(
    model: LightningOHLCPredictor,
    dataloader: DataLoader[dict[str, object]],
    max_batches: int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Collect masked predictions and targets per channel over the loader.

    Moves ``model`` to CUDA and runs its forward pass on up to ``max_batches``
    batches under :func:`torch.no_grad`. For each channel the response-block
    predictions and targets are restricted to days where a trade occurred (the
    same mask as
    :meth:`~ophir.training_models.LightningOHLCPredictor.compute_loss`),
    flattened, and gathered onto the CPU.

    Requires CUDA — the model's flex-attention forward is CUDA-only.

    Parameters
    ----------
    model : LightningOHLCPredictor
        The checkpoint to evaluate.
    dataloader : DataLoader
        A validation loader yielding collated batches.
    max_batches : int
        Maximum number of batches to score.

    Returns
    -------
    dict[str, tuple[torch.Tensor, torch.Tensor]]
        ``{channel: (pred, target)}`` of 1-D CPU tensors, keyed by ``r_close`` /
        ``upside`` / ``downside``.
    """
    model = model.cuda().eval()
    collected: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] = {
        name: ([], []) for name in _TARGETS
    }
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            if index >= max_batches:
                break
            output = model(batch)
            mask = output.trade_occured[:, -output.response_size :]
            channels = {
                "r_close": (output.predicted_r_close, output.target_r_close),
                "upside": (output.predicted_upside, output.target_upside),
                "downside": (output.predicted_downside, output.target_downside),
            }
            for name, (pred, target) in channels.items():
                collected[name][0].append(pred[mask].reshape(-1).cpu())
                collected[name][1].append(target[mask].reshape(-1).cpu())
    return {
        name: (torch.cat(preds), torch.cat(targets)) for name, (preds, targets) in collected.items()
    }


def evaluate_model(
    model: LightningOHLCPredictor,
    dataloader: DataLoader[dict[str, object]],
    max_batches: int,
) -> dict[str, dict[str, float]]:
    """Run a checkpoint over the loader and compute per-channel metrics.

    Parameters
    ----------
    model : LightningOHLCPredictor
        The checkpoint to evaluate.
    dataloader : DataLoader
        A validation loader.
    max_batches : int
        Maximum number of batches to score.

    Returns
    -------
    dict[str, dict[str, float]]
        ``{channel: metrics}``; the ``r_close`` entry additionally carries
        ``directional_accuracy`` and ``skill_score``.
    """
    collected = accumulate_targets(model, dataloader, max_batches)
    results: dict[str, dict[str, float]] = {}
    for name, (pred, target) in collected.items():
        metrics = target_metrics(pred, target)
        if name == "r_close":
            metrics["directional_accuracy"] = directional_accuracy(pred, target)
            metrics["skill_score"] = skill_score(pred, target)
        results[name] = metrics
    return results


def _format_metric(key: str, value: float) -> str:
    """Render a single metric cell for the report table."""
    if key == "n":
        return str(int(value))
    if value != value:  # NaN
        return "n/a"
    return f"{value:.5f}"


def format_report(results_by_label: dict[str, dict[str, dict[str, float]]]) -> str:
    """Render evaluation results as a Markdown report.

    Produces one table per target channel with a column per evaluated
    checkpoint, so multiple checkpoints (e.g. the best-``val_loss`` and
    time-interval base checkpoints) appear side by side.

    Parameters
    ----------
    results_by_label : dict[str, dict[str, dict[str, float]]]
        ``{label: {channel: metrics}}`` as returned by :func:`evaluate_model`.

    Returns
    -------
    str
        A Markdown string with a section and table per target channel.
    """
    labels = list(results_by_label)
    lines = ["## Validation evaluation", ""]
    for target in _TARGETS:
        lines.append(f"### {target}")
        lines.append("")
        lines.append("| metric | " + " | ".join(labels) + " |")
        lines.append("| --- |" + " --- |" * len(labels))
        metric_keys = [
            key
            for key in _METRIC_ORDER
            if any(key in results_by_label[label].get(target, {}) for label in labels)
        ]
        for key in metric_keys:
            cells = []
            for label in labels:
                metrics = results_by_label[label].get(target, {})
                cells.append(_format_metric(key, metrics[key]) if key in metrics else "n/a")
            lines.append(f"| {key} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def evaluate(
    seq_len: int = 365,
    offset: int = 90,
    response_size: int = 90,
    batch_size: int = 32,
    num_workers: int = 4,
    cache_size: int = 8,
    min_volume: float = 1000.0,
    train_min_year: int | None = None,
    train_max_year: int = 2023,
    val_min_year: int = 2024,
    val_max_year: int | None = None,
    data_dir: str | None = None,
    use_sp500: bool = False,
    val_batches: int = 50,
    strict: bool = False,
    finetuned: bool = False,
) -> None:
    """Evaluate a checkpoint on the held-out validation set.

    Rebuilds the by-date validation split (see
    :func:`~ophir.train.build_split_handlers`), loads the checkpoint(s), runs
    them over at most ``val_batches`` validation batches, and prints a per-target
    accuracy report. By default both base checkpoints — the best-``val_loss`` and
    the time-interval one — are evaluated and reported side by side (each loaded
    independently, so a missing one is skipped rather than failing the run); pass
    ``--finetuned`` to evaluate the latest finetuned checkpoint instead. Requires
    CUDA.

    Parameters
    ----------
    seq_len, offset, response_size : int
        Window length, stride, and forecast horizon.
    batch_size, num_workers, cache_size : int
        Dataloader configuration.
    min_volume : float
        Minimum average volume filter.
    train_min_year, train_max_year, val_min_year, val_max_year : int or None
        Date-split year ranges (``max_year`` exclusive); only the validation
        handler is used, but the gap must still embargo one window length.
    data_dir : str, optional
        Override the data directory (defaults to the package
        ``.ophir/data/days``).
    use_sp500 : bool
        Restrict to S&P 500 symbols (network fetch). Defaults to ``False``.
    val_batches : int
        Maximum number of validation batches to score. Defaults to ``50``.
    strict : bool
        Passed to the checkpoint loader. Defaults to ``False`` (older
        checkpoints may predate the ``mask_token``).
    finetuned : bool
        Evaluate the latest finetuned checkpoint instead of the base ones.
        Defaults to ``False``.
    """
    from ophir import register
    from ophir.train import build_dataloader, build_split_handlers

    base_path = os.path.join(data_dir or register.get_default_data_days_dir(), "stocks")
    _, val_handler = build_split_handlers(
        base_path=base_path,
        seq_len=seq_len,
        offset=offset,
        min_volume=min_volume,
        train_min_year=train_min_year,
        train_max_year=train_max_year,
        val_min_year=val_min_year,
        val_max_year=val_max_year,
        use_sp500=use_sp500,
    )
    val_dl = build_dataloader(val_handler, response_size, batch_size, num_workers, cache_size)

    loaders: list[tuple[str, Callable[[], LightningOHLCPredictor]]]
    if finetuned:
        loaders = [("finetuned", lambda: register.load_fintuned_ckpt(strict=strict))]
    else:
        loaders = [
            ("best-val", lambda: register.load_base_model_ckpt(strict=strict, time_version=False)),
            (
                "time-interval",
                lambda: register.load_base_model_ckpt(strict=strict, time_version=True),
            ),
        ]

    # Load and score each checkpoint independently so a single missing/unloadable
    # one (e.g. no best-val checkpoint saved yet) does not sink the whole run.
    results_by_label: dict[str, dict[str, dict[str, float]]] = {}
    for label, load in loaders:
        try:
            model = load()
        except (FileNotFoundError, IndexError, OSError) as exc:
            typer.echo(f"Skipping {label}: could not load a checkpoint ({exc}).")
            continue
        results_by_label[label] = evaluate_model(model, val_dl, val_batches)

    if not results_by_label:
        raise typer.BadParameter("No checkpoint could be loaded to evaluate.")

    typer.echo(format_report(results_by_label))
