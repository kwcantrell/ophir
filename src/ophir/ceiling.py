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
class OffsetRunIC:
    """Per-offset ``val_rank_ic_h*`` summary for one training run.

    Attributes
    ----------
    snapshot_mean : float
        Mean ``val_rank_ic_h{offset}`` over the run's non-burn-in validation
        snapshots — the denoised headline estimate.
    peak : float
        Max over those snapshots (the ceiling; also flags the E0 mid-run droop).
    n_snapshots : int
        Number of snapshots averaged.
    """

    snapshot_mean: float
    peak: float
    n_snapshots: int


@dataclass(frozen=True)
class ICAggregate:
    """Mean / min / max / sample-std / count over a config's seed replicates."""

    mean: float
    min: float
    max: float
    std: float
    n: int


@dataclass(frozen=True)
class NullBand:
    """Within-day permutation-null IC distribution for one offset bucket.

    Attributes
    ----------
    mean, std : float
        Mean and sample std (``ddof=1``) of the null cross-sectional IC over
        ``n_perms`` within-day target shuffles. ``mean`` is expected near 0.
    p05, p95 : float
        5th / 95th percentiles of the null IC distribution — the chance band a
        real per-offset IC must clear to be called signal.
    n_perms, n_rows : int
        Permutation count and the number of rows in this offset bucket.
    """

    mean: float
    std: float
    p05: float
    p95: float
    n_perms: int
    n_rows: int


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


def run_offset_ic(
    metrics_csv: str | Path,
    buckets: Sequence[int],
    *,
    burn_in_steps: int = 0,
) -> dict[str, OffsetRunIC]:
    """Summarise per-offset ``val_rank_ic_h*`` from a run's ``metrics.csv``.

    Averages each ``val_rank_ic_h{offset}`` column over validation snapshots
    (dropping NaN rows and rows with ``step < burn_in_steps``) and records the
    peak. A bucket whose column is absent or all-NaN yields a ``nan`` summary
    with ``n_snapshots == 0`` rather than raising.

    Parameters
    ----------
    metrics_csv : str or Path
        Lightning CSVLogger ``metrics.csv`` from an ``--log-offset-ic`` run.
    buckets : sequence of int
        Offsets to summarise (``_OFFSET_BUCKETS`` in production).
    burn_in_steps : int, optional
        Exclude validation rows logged before this global step (default 0).

    Returns
    -------
    dict[str, OffsetRunIC]
        One summary per bucket, keyed ``"h{offset}"``.
    """
    df = pd.read_csv(metrics_csv)
    step_col = _pick_column(df, ("step",))
    out: dict[str, OffsetRunIC] = {}
    for h in buckets:
        key = f"h{int(h)}"
        try:
            col = _pick_column(df, (f"val_rank_ic_{key}", f"val_rank_ic_{key}_epoch"))
        except KeyError:
            out[key] = OffsetRunIC(float("nan"), float("nan"), 0)
            continue
        sub = df.dropna(subset=[col])
        sub = sub[sub[step_col] >= burn_in_steps]
        if sub.empty:
            out[key] = OffsetRunIC(float("nan"), float("nan"), 0)
            continue
        vals = sub[col].to_numpy(dtype=float)
        out[key] = OffsetRunIC(
            snapshot_mean=float(vals.mean()),
            peak=float(vals.max()),
            n_snapshots=int(vals.size),
        )
    return out


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
    # Not ophir.evaluate.dedupe_by_ticker_date: that helper is pred/target-shaped
    # and returns dates as a string list for rank_ic; here we keep ids and integer
    # dates so the lagged-signal builder can order rows per ticker by date.
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

    Parameters
    ----------
    target, ids, dates : torch.Tensor
        Equal-length 1-D tensors: the target value, ticker id, and integer date
        ordinal for each response observation.
    lag : int, optional
        Number of prior same-ticker observations to look back (default 1).

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
    Rows with non-finite signal or target are always excluded before scoring.

    Parameters
    ----------
    signal, target, ids, dates : torch.Tensor
        Equal-length 1-D tensors of the signal, target, ticker id, and integer
        date ordinal for each row.
    valid : torch.Tensor, optional
        Boolean mask; when given, only rows flagged ``True`` are scored.
    """
    finite = torch.isfinite(signal) & torch.isfinite(target)
    mask = finite if valid is None else finite & valid
    signal, target, ids, dates = signal[mask], target[mask], ids[mask], dates[mask]
    dp, dt, dd = dedupe_by_ticker_date(signal, target, ids, dates)
    return rank_ic(dp, dt, dd)


def shuffle_within_day(
    target: torch.Tensor, dates: torch.Tensor, *, generator: torch.Generator
) -> torch.Tensor:
    """Permute ``target`` within each day — a null whose expected IC is ~0.

    The ``generator`` is advanced in place.
    """
    out = target.clone()
    for day in torch.unique(dates):
        idx = (dates == day).nonzero(as_tuple=True)[0]
        perm = idx[torch.randperm(idx.numel(), generator=generator)]
        out[idx] = target[perm]
    return out


def per_offset_shuffle_null(
    target: torch.Tensor,
    ids: torch.Tensor,
    dates: torch.Tensor,
    offsets: torch.Tensor,
    buckets: Sequence[int],
    *,
    n_perms: int,
    generator: torch.Generator,
) -> dict[str, NullBand]:
    """Per-offset within-day permutation null for cross-sectional rank-IC.

    For each bucket ``h`` the rows with ``offsets == h`` are isolated and the
    target is permuted within each day ``n_perms`` times (via
    :func:`shuffle_within_day`); each shuffle's cross-sectional IC (via
    :func:`cross_sectional_ic`, the production metric) forms the null. The band
    depends only on the per-day cross-section group sizes, not on the identity
    of the signal, so correlating the target against a within-day shuffle of
    itself yields the same band as shuffling against model predictions — no
    model is needed. Thinner near-offset cross-sections give wider bands.

    Parameters
    ----------
    target, ids, dates, offsets : torch.Tensor
        Equal-length 1-D tensors: target value, ticker id, integer date ordinal,
        and 1-based trading-day offset for each response observation.
    buckets : sequence of int
        Offsets to report; ``"h{offset}"`` keys mirror
        :func:`ophir.evaluate.rank_ic_by_offset`.
    n_perms : int
        Number of within-day shuffles.
    generator : torch.Generator
        Advanced in place; seed it for reproducibility.

    Returns
    -------
    dict[str, NullBand]
        One band per bucket; an empty bucket yields a ``nan`` band.
    """
    out: dict[str, NullBand] = {}
    for h in buckets:
        key = f"h{int(h)}"
        sel = offsets == int(h)
        n_rows = int(sel.sum())
        if n_rows == 0:
            out[key] = NullBand(float("nan"), float("nan"), float("nan"), float("nan"), n_perms, 0)
            continue
        t_h, i_h, d_h = target[sel], ids[sel], dates[sel]
        ics = [
            cross_sectional_ic(shuffle_within_day(t_h, d_h, generator=generator), t_h, i_h, d_h)[
                "ic_mean"
            ]
            for _ in range(n_perms)
        ]
        finite = np.asarray(ics, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            out[key] = NullBand(
                float("nan"), float("nan"), float("nan"), float("nan"), n_perms, n_rows
            )
            continue
        out[key] = NullBand(
            mean=float(finite.mean()),
            std=float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
            p05=float(np.percentile(finite, 5)),
            p95=float(np.percentile(finite, 95)),
            n_perms=n_perms,
            n_rows=n_rows,
        )
    return out


@dataclass(frozen=True)
class OffsetVerdict:
    """Per-offset confirmation verdict joining seed aggregate to the null band.

    Attributes
    ----------
    offset : int
        Trading-day-lead bucket.
    seed_mean, seed_std : float
        Cross-seed mean and sample std of each run's ``snapshot_mean``.
    n_seeds : int
        Number of runs (seeds) contributing a finite value.
    peak : float
        Cross-seed mean of each run's per-offset peak (diagnostic).
    null : NullBand
        The bucket's within-day permutation null.
    clears_null : bool
        ``True`` iff ``seed_mean`` exceeds the null 95th percentile.
    """

    offset: int
    seed_mean: float
    seed_std: float
    n_seeds: int
    peak: float
    null: NullBand
    clears_null: bool


def confirm_offset_skill(
    metrics_csvs: Sequence[str | Path],
    harvest: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    buckets: Sequence[int],
    *,
    n_perms: int = 500,
    burn_in_steps: int = 0,
    seed: int = 0,
) -> list[OffsetVerdict]:
    """Confirm per-offset skill across seeds against a per-offset null.

    Computes the within-day permutation null once from ``harvest`` (model-free),
    aggregates each run's ``snapshot_mean`` across seeds via
    :func:`aggregate_ic`, and flags ``clears_null`` where the seed-mean exceeds
    the null 95th percentile.

    Parameters
    ----------
    metrics_csvs : sequence of str or Path
        One ``metrics.csv`` per seed run.
    harvest : tuple of torch.Tensor
        ``(target, ids, dates, offsets)`` from the CPU validation harvest.
    buckets : sequence of int
        Offsets to report.
    n_perms : int, optional
        Null permutations (default 500).
    burn_in_steps : int, optional
        Snapshot burn-in passed to :func:`run_offset_ic` (default 0).
    seed : int, optional
        Seeds the null's ``torch.Generator`` (default 0).

    Returns
    -------
    list[OffsetVerdict]
        One verdict per bucket, in ``buckets`` order.
    """
    target, ids, dates, offsets = harvest
    generator = torch.Generator().manual_seed(seed)
    null = per_offset_shuffle_null(
        target, ids, dates, offsets, buckets, n_perms=n_perms, generator=generator
    )
    per_run = [run_offset_ic(csv, buckets, burn_in_steps=burn_in_steps) for csv in metrics_csvs]
    verdicts: list[OffsetVerdict] = []
    for h in buckets:
        key = f"h{int(h)}"
        means = [
            r[key].snapshot_mean for r in per_run if r[key].snapshot_mean == r[key].snapshot_mean
        ]
        peaks = [r[key].peak for r in per_run if r[key].peak == r[key].peak]
        band = null[key]
        if means:
            agg = aggregate_ic(means)
            seed_mean, seed_std, n_seeds = agg.mean, agg.std, agg.n
        else:
            seed_mean, seed_std, n_seeds = float("nan"), float("nan"), 0
        peak = float(sum(peaks) / len(peaks)) if peaks else float("nan")
        clears = bool(seed_mean == seed_mean and band.p95 == band.p95 and seed_mean > band.p95)
        verdicts.append(OffsetVerdict(int(h), seed_mean, seed_std, n_seeds, peak, band, clears))
    return verdicts


def format_verdict_table(verdicts: Sequence[OffsetVerdict]) -> str:
    """Render :func:`confirm_offset_skill` verdicts as a fixed-width table."""
    header = (
        f"{'offset':>6} {'seed_mean':>10} {'seed_std':>9} {'n':>3} "
        f"{'peak':>8} {'null_p95':>9} {'clears':>7}"
    )
    lines = [header]
    for v in verdicts:
        lines.append(
            f"{v.offset:>6} {v.seed_mean:>10.4f} {v.seed_std:>9.4f} {v.n_seeds:>3} "
            f"{v.peak:>8.4f} {v.null.p95:>9.4f} {'yes' if v.clears_null else 'no':>7}"
        )
    return "\n".join(lines)


def signal_decay_curve(
    target: torch.Tensor,
    ids: torch.Tensor,
    dates: torch.Tensor,
    leads: Sequence[int],
    *,
    kind: str = "reversal",
) -> dict[int, float]:
    """Cross-sectional IC of a lagged-return signal at each forecast lead.

    For each lead ``L`` in ``leads``, uses that ticker's return ``L`` observations
    earlier as the signal (negated when ``kind="reversal"``) and correlates it
    against the current return cross-sectionally via the production rank-IC. The
    result is the achievable signal at each forecast lead — the ceiling a model
    predicting ``L`` days ahead could reach from price history alone.

    Parameters
    ----------
    target, ids, dates : torch.Tensor
        Equal-length 1-D tensors of return, ticker id, and integer date ordinal,
        one row per (ticker, date).
    leads : sequence of int
        Forecast leads (in trading-day observations) to evaluate.
    kind : {"reversal", "momentum"}, optional
        ``"reversal"`` negates the lagged signal; ``"momentum"`` uses it as-is.

    Returns
    -------
    dict[int, float]
        ``{lead: ic_mean}`` for each requested lead.
    """
    if kind not in ("reversal", "momentum"):
        raise ValueError(f"kind must be 'reversal' or 'momentum', got {kind!r}")
    sign = -1.0 if kind == "reversal" else 1.0
    curve: dict[int, float] = {}
    for lead in leads:
        sig, valid = lagged_target_signal(target, ids, dates, lag=lead)
        ic = cross_sectional_ic(sign * sig, target, ids, dates, valid=valid)
        curve[int(lead)] = ic["ic_mean"]
    return curve


def pooled_baseline_ceiling(decay: dict[int, float], response_size: int) -> float:
    """Matched-horizon-mix ceiling: mean IC over sampled leads in ``1..response_size``.

    Approximates the horizon mix that ``val_rank_ic`` pools (offsets
    1..``response_size``) by averaging the decay curve over its sampled leads in
    that range. This is the fair comparand for a model whose pooled metric mixes
    those leads — not an exact replica of the metric's per-(ticker, date) dedup.

    Returns
    -------
    float
        Mean of finite ``decay`` values with lead in ``1..response_size``; ``nan``
        if none qualify.
    """
    vals = [v for lead, v in decay.items() if 1 <= lead <= response_size and v == v]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)
