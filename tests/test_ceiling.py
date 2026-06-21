import math
from pathlib import Path

import pandas as pd
import pytest
import torch

from ophir.ceiling import (
    ICAggregate,  # noqa: F401
    RunICSummary,
    aggregate_ic,
    cross_sectional_ic,
    dedupe_rows,
    lagged_target_signal,
    mde_for_group_difference,
    run_ic_summary,
    shuffle_within_day,
)


def _write_metrics(tmp_path: Path) -> Path:
    # Mimic a Lightning CSVLogger metrics.csv: train-step rows have NaN val
    # metrics; epoch rows carry both val_loss_epoch and val_rank_ic.
    rows = [
        {"step": 100, "val_loss_epoch": None, "val_rank_ic": None},  # train step
        {"step": 500, "val_loss_epoch": 0.90, "val_rank_ic": 0.010},  # epoch 1
        {"step": 1000, "val_loss_epoch": 0.70, "val_rank_ic": 0.030},  # peak IC, min loss
        {"step": 1500, "val_loss_epoch": 0.75, "val_rank_ic": 0.020},  # later
        {"step": 2000, "val_loss_epoch": 0.80, "val_rank_ic": 0.014},  # final (annealed)
    ]
    path = tmp_path / "metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_run_ic_summary_extracts_peak_best_final(tmp_path: Path) -> None:
    summary = run_ic_summary(_write_metrics(tmp_path))
    assert summary == RunICSummary(
        peak_ic=0.030, peak_step=1000, best_ckpt_ic=0.030, final_ic=0.014
    )


def test_run_ic_summary_best_ckpt_differs_from_peak(tmp_path: Path) -> None:
    # min val_loss at a row that is NOT the IC peak.
    rows = [
        {"step": 500, "val_loss_epoch": 0.60, "val_rank_ic": 0.012},  # min loss
        {"step": 1000, "val_loss_epoch": 0.95, "val_rank_ic": 0.040},  # peak IC
    ]
    path = tmp_path / "m.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    summary = run_ic_summary(path)
    assert summary.peak_ic == 0.040
    assert summary.best_ckpt_ic == 0.012


def test_aggregate_ic_basic() -> None:
    agg = aggregate_ic([0.0139, 0.0109, 0.0171])
    assert agg.n == 3
    assert agg.min == pytest.approx(0.0109)
    assert agg.max == pytest.approx(0.0171)
    assert agg.mean == pytest.approx((0.0139 + 0.0109 + 0.0171) / 3)


def test_aggregate_ic_empty_raises() -> None:
    with pytest.raises(ValueError):
        aggregate_ic([])


def test_mde_matches_formula() -> None:
    reps = [0.0139, 0.0109, 0.0171]
    s = float(pd.Series(reps).std(ddof=1))
    expected = 2.0 * s * math.sqrt(2.0 / 3)
    assert mde_for_group_difference(reps, seeds_per_group=3) == pytest.approx(expected)


def test_mde_needs_two_replicates() -> None:
    with pytest.raises(ValueError):
        mde_for_group_difference([0.01], seeds_per_group=3)


def test_dedupe_rows_keeps_first_per_ticker_date() -> None:
    target = torch.tensor([1.0, 2.0, 3.0])
    ids = torch.tensor([10, 10, 11])
    dates = torch.tensor([1, 1, 1])  # (10,1) duplicated
    t, i, d = dedupe_rows(target, ids, dates)
    assert t.tolist() == [1.0, 3.0]
    assert i.tolist() == [10, 11]
    assert d.tolist() == [1, 1]


def test_lagged_signal_uses_prior_date_per_ticker() -> None:
    # ticker 10 on dates 1,2,3 with targets 0.1,0.2,0.3
    target = torch.tensor([0.1, 0.2, 0.3])
    ids = torch.tensor([10, 10, 10])
    dates = torch.tensor([1, 2, 3])
    signal, valid = lagged_target_signal(target, ids, dates, lag=1)
    assert valid.tolist() == [False, True, True]
    values = signal[valid].tolist()
    assert values[0] == pytest.approx(0.1)
    assert values[1] == pytest.approx(0.2)  # yesterday's target


def test_lagged_signal_handles_interleaved_tickers_and_unsorted_dates() -> None:
    # Two interleaved tickers, rows NOT in date order. Signal value encodes the
    # source: tenths digit = ticker, units digit = the prior date it came from.
    # A naive target[:-1] shift would fail this; only correct per-ticker
    # date-ordering passes.
    #            t1@d2  t2@d3  t1@d1  t2@d1  t1@d3
    target = torch.tensor([0.12, 0.23, 0.11, 0.21, 0.13])
    ids = torch.tensor([1, 2, 1, 2, 1])
    dates = torch.tensor([2, 3, 1, 1, 3])
    signal, valid = lagged_target_signal(target, ids, dates, lag=1)
    # Per ticker, sorted by date, lag-1 prior target:
    #   row0 t1@d2 -> t1@d1 = 0.11
    #   row1 t2@d3 -> t2@d1 = 0.21
    #   row2 t1@d1 -> invalid (no prior)
    #   row3 t2@d1 -> invalid (no prior)
    #   row4 t1@d3 -> t1@d2 = 0.12
    assert valid.tolist() == [True, True, False, False, True]
    assert signal[0].item() == pytest.approx(0.11)
    assert signal[1].item() == pytest.approx(0.21)
    assert signal[4].item() == pytest.approx(0.12)


def test_cross_sectional_ic_perfect_rank_is_one() -> None:
    # two days, signal ranks tickers identically to target each day
    signal = torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    ids = torch.tensor([1, 2, 3, 1, 2, 3])
    dates = torch.tensor([1, 1, 1, 2, 2, 2])
    out = cross_sectional_ic(signal, target, ids, dates)
    assert out["ic_mean"] == pytest.approx(1.0)
    assert out["n_days"] == 2.0


def test_shuffle_within_day_preserves_per_day_multiset() -> None:
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    dates = torch.tensor([1, 1, 2, 2])
    g = torch.Generator().manual_seed(0)
    shuffled = shuffle_within_day(target, dates, generator=g)
    assert sorted(shuffled[dates == 1].tolist()) == [1.0, 2.0]
    assert sorted(shuffled[dates == 2].tolist()) == [3.0, 4.0]
