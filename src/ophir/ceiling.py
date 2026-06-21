"""Pure, offline helpers for the forecasting-ceiling investigation.

See ``docs/superpowers/specs/2026-06-20-forecast-ceiling-investigation-design.md``.
Everything here is CPU-only and dependency-light: it parses training-run metric
logs and computes cross-sectional rank-IC baselines, reusing the production IC
math in :mod:`ophir.evaluate` so the offline analysis and the live
``val_rank_ic`` metric agree. No model, no CUDA, no ``.ophir/`` layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch

from ophir.evaluate import dedupe_by_ticker_date, rank_ic

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    """Return the first of ``candidates`` present in ``df``.

    Lightning's CSVLogger names a metric logged with both ``on_step`` and
    ``on_epoch`` as ``<name>_epoch``; one logged only ``on_epoch`` keeps its bare
    name. This tolerates either spelling.
    """
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"none of {candidates} present in {list(df.columns)}")


@dataclass(frozen=True)
class RunICSummary:
    """Peak / saved-checkpoint / final ``val_rank_ic`` for one training run.

    Attributes
    ----------
    peak_ic, peak_step : float, int
        The maximum ``val_rank_ic`` over the run and the step it occurred at.
    best_ckpt_ic : float
        ``val_rank_ic`` on the minimum-``val_loss`` validation row — the row
        whose checkpoint ``ModelCheckpoint(monitor="val_loss")`` would persist.
    final_ic : float
        ``val_rank_ic`` on the last validation row (the fully-annealed value).
    """

    peak_ic: float
    peak_step: int
    best_ckpt_ic: float
    final_ic: float


@dataclass(frozen=True)
class ICAggregate:
    """Mean / min / max / sample-std / count over a config's seed replicates."""

    mean: float
    min: float
    max: float
    std: float
    n: int


def run_ic_summary(metrics_csv: str | Path) -> RunICSummary:
    """Summarise a run's ``val_rank_ic`` trajectory from its ``metrics.csv``.

    Parameters
    ----------
    metrics_csv : str or Path
        Path to a Lightning CSVLogger ``metrics.csv``.

    Returns
    -------
    RunICSummary
        Peak, saved-checkpoint, and final ``val_rank_ic``.

    Raises
    ------
    ValueError
        If no validation rows carry ``val_rank_ic``.
    """
    df = pd.read_csv(metrics_csv)
    ic_col = _pick_column(df, ("val_rank_ic",))
    step_col = _pick_column(df, ("step",))
    val = df.dropna(subset=[ic_col])
    if val.empty:
        raise ValueError(f"no {ic_col} rows in {metrics_csv}")
    peak = val.loc[val[ic_col].idxmax()]
    loss_col = _pick_column(df, ("val_loss_epoch", "val_loss"))
    with_loss = val.dropna(subset=[loss_col])
    best = with_loss.loc[with_loss[loss_col].idxmin()] if not with_loss.empty else peak
    final = val.iloc[-1]
    return RunICSummary(
        peak_ic=float(peak[ic_col]),
        peak_step=int(peak[step_col]),
        best_ckpt_ic=float(best[ic_col]),
        final_ic=float(final[ic_col]),
    )


def aggregate_ic(values: Sequence[float]) -> ICAggregate:
    """Aggregate one config's per-seed IC values.

    Parameters
    ----------
    values : sequence of float
        Per-seed IC values for a single configuration.

    Returns
    -------
    ICAggregate
        ``std`` is the sample standard deviation (``ddof=1``), or ``0.0`` for a
        single value.

    Raises
    ------
    ValueError
        If ``values`` is empty.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("need at least one IC value")
    return ICAggregate(
        mean=float(arr.mean()),
        min=float(arr.min()),
        max=float(arr.max()),
        std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        n=int(arr.size),
    )


def mde_for_group_difference(
    replicates: Sequence[float], *, seeds_per_group: int, sigmas: float = 2.0
) -> float:
    """Minimum detectable effect for a difference of two seed-mean ICs.

    Estimates the seed-noise scale ``s`` from same-config ``replicates`` and
    returns ``sigmas * s * sqrt(2 / seeds_per_group)`` — the half-width below
    which a gap between two ``seeds_per_group``-seed config means is consistent
    with seed noise. Two configs whose mean IC differ by less than this should
    not be called different.

    Raises
    ------
    ValueError
        If fewer than two ``replicates`` are supplied.
    """
    arr = np.asarray(replicates, dtype=float)
    if arr.size < 2:
        raise ValueError("need >= 2 replicates to estimate seed noise")
    s = float(arr.std(ddof=1))
    return sigmas * s * float(np.sqrt(2.0 / seeds_per_group))


def dedupe_rows(
    target: torch.Tensor, ids: torch.Tensor, dates: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep the first row per ``(ticker, date)`` (stable order).

    Overlapping windows emit several rows per name per day; baselines need one.
    """
    seen: set[tuple[int, int]] = set()
    keep: list[int] = []
    for k, (sid, day) in enumerate(zip(ids.tolist(), dates.tolist(), strict=True)):
        key = (int(sid), int(day))
        if key not in seen:
            seen.add(key)
            keep.append(k)
    idx = torch.tensor(keep, dtype=torch.long)
    return target[idx], ids[idx], dates[idx]


def lagged_target_signal(
    target: torch.Tensor, ids: torch.Tensor, dates: torch.Tensor, *, lag: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-ticker previous-by-date target as a naive autoregressive signal.

    For each row, the signal is that ticker's target ``lag`` observations earlier
    in date order. Rows without ``lag`` priors are flagged invalid.

    Returns
    -------
    signal, valid : torch.Tensor, torch.Tensor
        ``signal`` holds the lagged target (``nan`` where invalid); ``valid`` is
        a boolean mask. Use ``signal`` directly for a momentum baseline or
        negate it for reversal.
    """
    t = target.detach().cpu().numpy()
    i = ids.detach().cpu().numpy()
    d = dates.detach().cpu().numpy()
    order = np.lexsort((d, i))  # primary key = id, secondary = date
    sid = i[order]
    st = t[order]
    lagged = np.full(st.shape, np.nan, dtype=float)
    for k in range(lag, len(order)):
        if sid[k] == sid[k - lag]:
            lagged[k] = st[k - lag]
    signal = np.full(t.shape, np.nan, dtype=float)
    signal[order] = lagged
    valid = ~np.isnan(signal)
    return torch.from_numpy(signal), torch.from_numpy(valid)


def cross_sectional_ic(
    signal: torch.Tensor,
    target: torch.Tensor,
    ids: torch.Tensor,
    dates: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
) -> dict[str, float]:
    """Daily cross-sectional rank-IC of ``signal`` vs ``target``.

    Mirrors the production metric exactly: dedupe to one row per ``(ticker,
    date)`` then average the per-day Spearman correlation via
    :func:`ophir.evaluate.rank_ic`. Optionally restrict to ``valid`` rows first.
    """
    if valid is not None:
        signal, target, ids, dates = signal[valid], target[valid], ids[valid], dates[valid]
    dp, dt, dd = dedupe_by_ticker_date(signal, target, ids, dates)
    return rank_ic(dp, dt, dd)


def shuffle_within_day(
    target: torch.Tensor, dates: torch.Tensor, *, generator: torch.Generator
) -> torch.Tensor:
    """Permute ``target`` within each day — a null whose expected IC is ~0."""
    out = target.clone()
    for day in torch.unique(dates):
        idx = (dates == day).nonzero(as_tuple=True)[0]
        perm = idx[torch.randperm(idx.numel(), generator=generator)]
        out[idx] = target[perm]
    return out
