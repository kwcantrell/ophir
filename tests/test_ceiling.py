import math
from pathlib import Path

import pandas as pd
import pytest
import torch

from ophir.ceiling import (
    ICAggregate,  # noqa: F401
    NullBand,  # noqa: F401
    OffsetRunIC,  # noqa: F401
    RunICSummary,
    aggregate_ic,
    cross_sectional_ic,
    dedupe_rows,
    lagged_target_signal,
    mde_for_group_difference,
    per_offset_shuffle_null,
    pooled_baseline_ceiling,
    run_ic_summary,
    run_offset_ic,
    shuffle_within_day,
    signal_decay_curve,
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


def test_cross_sectional_ic_with_valid_matches_filtered() -> None:
    # The valid= mask from lagged_target_signal must score exactly the finite
    # rows — this is the seam the E1 experiment relies on.
    target = torch.tensor([0.12, 0.23, 0.11, 0.21, 0.13])
    ids = torch.tensor([1, 2, 1, 2, 1])
    dates = torch.tensor([2, 3, 1, 1, 3])
    sig, valid = lagged_target_signal(target, ids, dates, lag=1)
    scored = cross_sectional_ic(sig, target, ids, dates, valid=valid)
    manual = cross_sectional_ic(sig[valid], target[valid], ids[valid], dates[valid])
    assert scored["n_days"] == manual["n_days"] == 1.0
    assert scored["ic_mean"] == pytest.approx(manual["ic_mean"])
    assert scored["ic_mean"] == pytest.approx(1.0)


def test_cross_sectional_ic_excludes_nonfinite_without_valid() -> None:
    # A NaN signal row must never be scored, even when valid is not passed.
    # Day 1 has a NaN row (ticker 4) whose target would break the rank if
    # included; excluding it leaves a perfect cross-sectional rank on both days.
    signal = torch.tensor([1.0, 2.0, 3.0, float("nan"), 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 3.0, 0.5, 2.0, 3.0])
    ids = torch.tensor([1, 2, 3, 4, 2, 3])
    dates = torch.tensor([1, 1, 1, 1, 2, 2])
    out = cross_sectional_ic(signal, target, ids, dates)
    assert out["n_days"] == 2.0
    assert out["ic_mean"] == pytest.approx(1.0)


def test_shuffle_within_day_actually_permutes() -> None:
    # A large single day with a fixed seed must not return the identity,
    # guarding against a regression that drops the permutation.
    target = torch.arange(8, dtype=torch.float32)
    dates = torch.zeros(8, dtype=torch.long)
    g = torch.Generator().manual_seed(0)
    shuffled = shuffle_within_day(target, dates, generator=g)
    assert sorted(shuffled.tolist()) == target.tolist()  # still a permutation
    assert not torch.equal(shuffled, target)  # and not the identity


def test_signal_decay_curve_perfect_reversal() -> None:
    # 3 tickers over 3 days; each day's returns are the rank-reversal of the
    # prior day, so a 1-lead reversal signal perfectly predicts the cross-section.
    #            d1(t1,t2,t3)  d2(reversed) d3(reversed again)
    target = torch.tensor([1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0])
    ids = torch.tensor([1, 2, 3, 1, 2, 3, 1, 2, 3])
    dates = torch.tensor([1, 1, 1, 2, 2, 2, 3, 3, 3])
    curve = signal_decay_curve(target, ids, dates, leads=(1,), kind="reversal")
    assert curve == pytest.approx({1: 1.0})
    mom = signal_decay_curve(target, ids, dates, leads=(1,), kind="momentum")
    assert mom[1] == pytest.approx(-1.0)


def test_signal_decay_curve_rejects_bad_kind() -> None:
    target = torch.tensor([1.0, 2.0])
    ids = torch.tensor([1, 2])
    dates = torch.tensor([1, 1])
    with pytest.raises(ValueError):
        signal_decay_curve(target, ids, dates, leads=(1,), kind="trend")


def test_pooled_baseline_ceiling_means_in_range() -> None:
    decay = {1: 0.05, 5: 0.03, 90: 0.0}
    assert pooled_baseline_ceiling(decay, response_size=10) == pytest.approx(0.04)
    assert pooled_baseline_ceiling(decay, response_size=90) == pytest.approx(
        (0.05 + 0.03 + 0.0) / 3
    )


def test_pooled_baseline_ceiling_empty_is_nan() -> None:
    assert math.isnan(pooled_baseline_ceiling({90: 0.1}, response_size=10))


def test_per_offset_shuffle_null_brackets_zero_and_widens_when_thin() -> None:
    n_days = 20
    wide_n, thin_n = 8, 3
    wide, thin = wide_n * n_days, thin_n * n_days
    tg = torch.Generator().manual_seed(1)
    target = torch.randn(wide + thin, generator=tg)
    ids = torch.cat([torch.arange(wide_n).repeat(n_days), torch.arange(thin_n).repeat(n_days)])
    dates = torch.cat(
        [
            torch.arange(n_days).repeat_interleave(wide_n),
            torch.arange(n_days).repeat_interleave(thin_n),
        ]
    )
    offsets = torch.cat(
        [torch.ones(wide, dtype=torch.long), torch.full((thin,), 2, dtype=torch.long)]
    )
    g = torch.Generator().manual_seed(0)
    bands = per_offset_shuffle_null(target, ids, dates, offsets, [1, 2], n_perms=200, generator=g)
    assert abs(bands["h1"].mean) < bands["h1"].std  # null centered on ~0
    assert bands["h1"].p05 < 0.0 < bands["h1"].p95
    assert bands["h1"].n_rows == wide and bands["h2"].n_rows == thin
    assert bands["h2"].std > bands["h1"].std  # thinner cross-section -> wider null
    assert bands["h1"].n_perms == 200


def test_per_offset_shuffle_null_empty_bucket_is_nan() -> None:
    target = torch.tensor([0.1, 0.2, 0.3, 0.4])
    ids = torch.tensor([1, 2, 1, 2])
    dates = torch.tensor([1, 1, 2, 2])
    offsets = torch.tensor([1, 1, 1, 1])
    g = torch.Generator().manual_seed(0)
    bands = per_offset_shuffle_null(target, ids, dates, offsets, [1, 90], n_perms=50, generator=g)
    assert bands["h90"].n_rows == 0
    assert math.isnan(bands["h90"].mean) and math.isnan(bands["h90"].p95)


def test_run_offset_ic_means_snapshots_and_reports_peak(tmp_path: Path) -> None:
    rows = [
        {"step": 100, "val_rank_ic_h1": 0.02},
        {"step": 200, "val_rank_ic_h1": 0.10},
        {"step": 300, "val_rank_ic_h1": 0.06},
    ]
    path = tmp_path / "m.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    out = run_offset_ic(path, [1], burn_in_steps=0)
    assert out["h1"].n_snapshots == 3
    assert out["h1"].peak == pytest.approx(0.10)
    assert out["h1"].snapshot_mean == pytest.approx(0.06)


def test_run_offset_ic_burn_in_excludes_early_steps(tmp_path: Path) -> None:
    rows = [
        {"step": 50, "val_rank_ic_h1": 0.0},
        {"step": 500, "val_rank_ic_h1": 0.08},
    ]
    path = tmp_path / "m.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    out = run_offset_ic(path, [1], burn_in_steps=100)
    assert out["h1"].n_snapshots == 1
    assert out["h1"].snapshot_mean == pytest.approx(0.08)


def test_run_offset_ic_missing_bucket_is_nan(tmp_path: Path) -> None:
    rows = [{"step": 100, "val_rank_ic_h1": 0.05, "val_rank_ic_h90": float("nan")}]
    path = tmp_path / "m.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    out = run_offset_ic(path, [1, 90])
    assert out["h1"].snapshot_mean == pytest.approx(0.05)
    assert math.isnan(out["h90"].snapshot_mean) and out["h90"].n_snapshots == 0
