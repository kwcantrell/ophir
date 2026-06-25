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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import typer

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch.utils.data import DataLoader

    from ophir.training_models import LightningOHLCPredictor

#: Channels predicted by the model, in report order. ``r_close`` additionally
#: gets a directional accuracy and a zero-baseline skill score (it is a signed,
#: zero-centred return; the others are non-negative log-magnitudes).
_TARGETS = ("r_close", "upside", "downside")

#: Metric columns, in report order. Not every target reports every metric.
_METRIC_ORDER = (
    "n",
    "mae",
    "rmse",
    "bias",
    "directional_accuracy",
    "skill_score",
    "skill_vs_persistence",
    "rank_ic_mean",
    "rank_ic_ir",
)

#: Inclusive upper bound (in 1-based trading-day forecast offsets) of the near
#: band where cross-sectional skill concentrates. Mirrors the training-side
#: ``LightningOHLCPredictor.near_offset_k`` default so the offline report and the
#: live ``val_rank_ic_near`` metric agree.
_NEAR_OFFSET_K = 5


@dataclass
class AccumulatedEval:
    """Masked predictions/targets, persistence baselines, and r_close identity."""

    channels: dict[str, tuple[torch.Tensor, torch.Tensor]]
    baselines: dict[str, torch.Tensor] = field(default_factory=dict)
    r_close_ids: torch.Tensor | None = None
    r_close_dates: torch.Tensor | None = None
    r_close_offsets: torch.Tensor | None = None


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


def skill_score_vs_baseline(
    pred: torch.Tensor, target: torch.Tensor, baseline: torch.Tensor
) -> float:
    """RMSE skill score of ``pred`` against an arbitrary ``baseline`` forecast.

    ``1 - rmse(pred) / rmse(baseline)``: positive means the model beats the
    baseline, ``0`` ties it, negative is worse. ``nan`` for empty input or a
    zero-RMSE baseline. Lets the non-negative ``upside``/``downside`` channels be
    scored against a persistence/EWMA forecast instead of having no reference.

    Parameters
    ----------
    pred, target, baseline : torch.Tensor
        1-D tensors of equal length.

    Returns
    -------
    float
        The skill score; ``nan`` when there are no samples or baseline RMSE is zero.
    """
    if pred.numel() == 0:
        return float("nan")
    rmse_model = (pred - target).pow(2).mean().sqrt().item()
    rmse_baseline = (baseline - target).pow(2).mean().sqrt().item()
    if rmse_baseline == 0:
        return float("nan")
    return float(1.0 - rmse_model / rmse_baseline)


def prefix_last_observed(
    values: torch.Tensor, trade_occured: torch.Tensor, response_size: int
) -> torch.Tensor:
    """Each row's channel value at its last traded prefix position.

    The persistence baseline carries the last *observed* (pre-horizon) value
    forward across the forecast block. ``values``/``trade_occured`` are ``(B, S)``;
    only the first ``S - response_size`` columns (the prefix) are considered.
    Rows with no traded prefix day fall back to prefix position 0.
    """
    _b, seq_len = values.shape
    prefix_len = seq_len - response_size
    prefix_trade = trade_occured[:, :prefix_len].to(values.dtype)
    positions = torch.arange(prefix_len, device=values.device, dtype=values.dtype)
    # argmax over trade*position selects the largest column index that traded;
    # all-False rows yield argmax 0 (the fallback).
    last_idx = (prefix_trade * positions).argmax(dim=1)
    rows = torch.arange(values.shape[0], device=values.device)
    return values[:, :prefix_len][rows, last_idx]


def _spearman(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Spearman rank correlation of two 1-D tensors (nan if < 2 points)."""
    if pred.numel() < 2 or pred.std() == 0 or target.std() == 0:
        return float("nan")
    pr = pred.argsort().argsort().float()
    tr = target.argsort().argsort().float()
    pr = pr - pr.mean()
    tr = tr - tr.mean()
    denom = pr.norm() * tr.norm()
    if denom == 0:
        return float("nan")
    return float((pr @ tr / denom).item())


def rank_ic(pred: torch.Tensor, target: torch.Tensor, dates: list[str]) -> dict[str, float]:
    """Daily cross-sectional rank information coefficient.

    Groups predictions/targets by ``dates`` and computes the Spearman rank
    correlation within each day, then summarises across days. ``ic_ir`` is the
    information ratio ``ic_mean / ic_std`` (annualisable by the caller).

    Parameters
    ----------
    pred, target : torch.Tensor
        1-D tensors of equal length holding predictions and ground-truth
        values across all days.
    dates : list[str]
        Day label for each element; rows sharing a label form one cross-section.

    Returns
    -------
    dict[str, float]
        ``ic_mean``, ``ic_std``, ``ic_ir``, and ``n_days`` (the number of days
        with at least two observations and a finite Spearman correlation).
    """
    by_day: dict[str, list[int]] = {}
    for i, d in enumerate(dates):
        by_day.setdefault(d, []).append(i)
    ics: list[float] = []
    for idx in by_day.values():
        sel = torch.tensor(idx)
        ic = _spearman(pred[sel], target[sel])
        if ic == ic:  # skip nan days
            ics.append(ic)
    if not ics:
        return {
            "ic_mean": float("nan"),
            "ic_std": float("nan"),
            "ic_ir": float("nan"),
            "n_days": 0.0,
        }
    t = torch.tensor(ics)
    mean = float(t.mean().item())
    std = float(t.std(unbiased=False).item())
    return {
        "ic_mean": mean,
        "ic_std": std,
        "ic_ir": mean / std if std > 0 else float("nan"),
        "n_days": float(len(ics)),
    }


def dedupe_by_ticker_date(
    pred: torch.Tensor, target: torch.Tensor, ids: torch.Tensor, dates: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Keep one prediction per (ticker, date) so a day's cross-section is unique.

    Overlapping windows can emit several predictions for the same ticker on the
    same calendar day; rank-IC needs one row per name per day. Keeps the first
    occurrence (stable order) and returns the per-row date as strings for
    :func:`rank_ic`.
    """
    seen: set[tuple[int, int]] = set()
    keep: list[int] = []
    id_list = ids.tolist()
    date_list = dates.tolist()
    for i, (sid, day) in enumerate(zip(id_list, date_list, strict=True)):
        key = (int(sid), int(day))
        if key not in seen:
            seen.add(key)
            keep.append(i)
    index = torch.tensor(keep, dtype=torch.long)
    kept_dates = [str(int(date_list[i])) for i in keep]
    return pred[index], target[index], kept_dates


def rank_ic_by_offset(
    pred: torch.Tensor,
    target: torch.Tensor,
    ids: torch.Tensor,
    dates: torch.Tensor,
    offsets: torch.Tensor,
    buckets: Sequence[int],
) -> dict[str, float]:
    """Daily cross-sectional rank-IC resolved by response-position offset.

    Splits predictions by forecast offset (1-based response-position lead) and
    computes each offset's pooled-day rank-IC independently, reusing
    :func:`dedupe_by_ticker_date` and :func:`rank_ic`. A single validation pass can
    then reveal whether skill concentrates at near horizons.

    Parameters
    ----------
    pred, target, ids, dates : torch.Tensor
        Equal-length 1-D tensors, as accumulated for :func:`rank_ic`.
    offsets : torch.Tensor
        Same-length integer tensor: each row's 1-based forecast offset in trading
        days (the trading-day lead within the response block).
    buckets : sequence of int
        Offsets to report. For each ``h`` only rows with ``offsets == h`` are used.

    Returns
    -------
    dict[str, float]
        ``{"h{offset}": ic_mean}`` per bucket; ``nan`` for a bucket with no rows or
        no day having at least two observations.
    """
    out: dict[str, float] = {}
    for h in buckets:
        sel = offsets == h
        if not bool(sel.any()):
            out[f"h{int(h)}"] = float("nan")
            continue
        dp, dt, dd = dedupe_by_ticker_date(pred[sel], target[sel], ids[sel], dates[sel])
        out[f"h{int(h)}"] = rank_ic(dp, dt, dd)["ic_mean"]
    return out


def rank_ic_near(
    pred: torch.Tensor,
    target: torch.Tensor,
    ids: torch.Tensor,
    dates: torch.Tensor,
    offsets: torch.Tensor,
    k: int = _NEAR_OFFSET_K,
) -> float:
    """Pooled cross-sectional rank-IC over near forecast offsets ``1..k``.

    Restricts the rows to 1-based trading-day offsets ``1 <= offset <= k`` (the
    near band where cross-sectional skill concentrates), dedupes to one
    prediction per ``(ticker, date)``, and averages the daily Spearman rank-IC.
    Reuses :func:`dedupe_by_ticker_date` and :func:`rank_ic` so the offline
    report and the live :func:`ophir.training_models.val_rank_ic_near` metric
    share identical math.

    Parameters
    ----------
    pred, target, ids, dates : torch.Tensor
        Equal-length 1-D tensors, as accumulated for :func:`rank_ic`.
    offsets : torch.Tensor
        Same-length integer tensor of 1-based trading-day forecast offsets.
    k : int, optional
        Inclusive upper bound of the near band. Defaults to ``_NEAR_OFFSET_K``.

    Returns
    -------
    float
        Mean daily rank-IC over the near band, or ``nan`` when no rows fall in it.
    """
    if pred.numel() == 0:
        return float("nan")
    sel = (offsets >= 1) & (offsets <= k)
    if not bool(sel.any()):
        return float("nan")
    dp, dt, dd = dedupe_by_ticker_date(pred[sel], target[sel], ids[sel], dates[sel])
    return rank_ic(dp, dt, dd)["ic_mean"]


def accumulate_targets(
    model: LightningOHLCPredictor,
    dataloader: DataLoader[dict[str, object]],
    max_batches: int,
) -> AccumulatedEval:
    """Collect masked predictions, targets, and baselines per channel.

    Moves ``model`` to CUDA and runs its forward pass on up to ``max_batches``
    batches under :func:`torch.no_grad`. For each channel the response-block
    predictions and targets are restricted to days where a trade occurred (the
    same mask as
    :meth:`~ophir.training_models.LightningOHLCPredictor.compute_loss`),
    flattened, and gathered onto the CPU. For the non-negative
    ``upside``/``downside`` magnitude channels a persistence baseline (each
    example's last traded prefix value, carried flat across the horizon) is
    collected in parallel so the report can score them against it.

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
    AccumulatedEval
        ``channels`` maps each of ``r_close`` / ``upside`` / ``downside`` to a
        ``(pred, target)`` pair of 1-D CPU tensors; ``baselines`` maps
        ``upside`` / ``downside`` to their persistence-baseline tensors aligned
        row-for-row with the predictions. When the loader carries identity
        (``return_identity=True``), ``r_close_ids`` / ``r_close_dates`` hold the
        per-prediction ticker id and calendar-day ordinal aligned row-for-row
        with the ``r_close`` channel (else ``None``). ``r_close_offsets`` carries
        each prediction's 1-based forecast offset (trading-day lead within the
        response block), aligned with ``r_close_ids`` (also ``None`` without
        identity).
    """
    model = model.cuda().eval()
    collected: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] = {
        name: ([], []) for name in _TARGETS
    }
    baseline_lists: dict[str, list[torch.Tensor]] = {"upside": [], "downside": []}
    id_lists: list[torch.Tensor] = []
    date_lists: list[torch.Tensor] = []
    offset_lists: list[torch.Tensor] = []
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            if index >= max_batches:
                break
            output = model(batch)
            rs = int(output.response_size)
            mask = output.trade_occured[:, -rs:]
            channels = {
                "r_close": (output.predicted_r_close, output.target_r_close),
                "upside": (output.predicted_upside, output.target_upside),
                "downside": (output.predicted_downside, output.target_downside),
            }
            for name, (pred, target) in channels.items():
                collected[name][0].append(pred[mask].reshape(-1).cpu())
                collected[name][1].append(target[mask].reshape(-1).cpu())
            # upside is target channel 1, downside is channel 2.
            for name, idx in (("upside", 1), ("downside", 2)):
                channel_values = output.targets[..., idx]  # (B, S)
                base = prefix_last_observed(channel_values, output.trade_occured, rs)  # (B,)
                base = base.unsqueeze(1).expand(-1, rs)  # (B, R)
                baseline_lists[name].append(base[mask].reshape(-1).cpu())
            # Opt-in identity, collected parallel to the r_close pred/target above.
            if output.stock_id is not None and output.date_ordinal is not None:
                from ophir.training_models import trading_day_offsets

                resp_dates = output.date_ordinal[:, -rs:]  # (B, R)
                ids_br = output.stock_id.view(-1, 1).expand(-1, rs)  # (B, R)
                offsets = trading_day_offsets(mask)  # (B, R) cumulative trading-day rank
                id_lists.append(ids_br[mask].reshape(-1).cpu())
                date_lists.append(resp_dates[mask].reshape(-1).cpu())
                offset_lists.append(offsets[mask].reshape(-1).cpu())
    return AccumulatedEval(
        channels={
            name: (torch.cat(preds), torch.cat(targets))
            for name, (preds, targets) in collected.items()
        },
        baselines={name: torch.cat(b) for name, b in baseline_lists.items()},
        r_close_ids=torch.cat(id_lists) if id_lists else None,
        r_close_dates=torch.cat(date_lists) if date_lists else None,
        r_close_offsets=torch.cat(offset_lists) if offset_lists else None,
    )


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
        ``directional_accuracy`` and ``skill_score`` (and ``rank_ic_mean`` /
        ``rank_ic_ir`` when the loader carries identity; plus ``rank_ic_near``
        — the near-band operating point — and a per-offset ``h{n}`` IC curve when
        the loader also carries forecast offsets), and the ``upside``/``downside``
        entries carry ``skill_vs_persistence`` (the skill score against the
        persistence baseline).
    """
    acc = accumulate_targets(model, dataloader, max_batches)
    results: dict[str, dict[str, float]] = {}
    for name, (pred, target) in acc.channels.items():
        metrics = target_metrics(pred, target)
        if name == "r_close":
            metrics["directional_accuracy"] = directional_accuracy(pred, target)
            metrics["skill_score"] = skill_score(pred, target)
        if name in acc.baselines:
            metrics["skill_vs_persistence"] = skill_score_vs_baseline(
                pred, target, acc.baselines[name]
            )
        results[name] = metrics
    if acc.r_close_ids is not None and acc.r_close_dates is not None:
        pred, target = acc.channels["r_close"]
        dp, dt, dd = dedupe_by_ticker_date(pred, target, acc.r_close_ids, acc.r_close_dates)
        ic = rank_ic(dp, dt, dd)
        results["r_close"]["rank_ic_mean"] = ic["ic_mean"]
        results["r_close"]["rank_ic_ir"] = ic["ic_ir"]
    if acc.r_close_offsets is not None:
        from ophir.training_models import _OFFSET_BUCKETS

        assert acc.r_close_ids is not None and acc.r_close_dates is not None
        pred, target = acc.channels["r_close"]
        results["r_close"]["rank_ic_near"] = rank_ic_near(
            pred, target, acc.r_close_ids, acc.r_close_dates, acc.r_close_offsets
        )
        results["r_close"].update(
            rank_ic_by_offset(
                pred,
                target,
                acc.r_close_ids,
                acc.r_close_dates,
                acc.r_close_offsets,
                _OFFSET_BUCKETS,
            )
        )
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
    val_dl = build_dataloader(
        val_handler,
        response_size,
        batch_size,
        num_workers,
        cache_size,
        return_identity=True,
    )

    loaders: list[tuple[str, Callable[[], LightningOHLCPredictor]]]
    if finetuned:
        loaders = [("finetuned", lambda: register.load_finetuned_ckpt(strict=strict))]
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
