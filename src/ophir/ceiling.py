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

import pandas as pd  # type: ignore[import-untyped]

if TYPE_CHECKING:
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
