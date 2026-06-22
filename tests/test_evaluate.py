"""Unit tests for the CPU-safe metric core in :mod:`ophir.evaluate`.

The accumulation path runs the CUDA forward, but the metric helpers and the
report formatter operate on plain tensors, so they are exercised here without a
model or a GPU.
"""

import math

import pytest
import torch

from ophir.evaluate import (
    _spearman,
    accumulate_targets,
    dedupe_by_ticker_date,
    directional_accuracy,
    format_report,
    prefix_last_observed,
    rank_ic,
    rank_ic_by_offset,
    skill_score,
    skill_score_vs_baseline,
    target_metrics,
)
from ophir.model_data import OHLCMulitClassPredictorInput


class _FakeModel:
    """Returns its batch as a populated forward output, ignoring device moves."""

    def cuda(self) -> "_FakeModel":
        return self

    def eval(self) -> "_FakeModel":
        return self

    def __call__(self, batch: dict[str, object]) -> OHLCMulitClassPredictorInput:
        obj = OHLCMulitClassPredictorInput(
            feature_input=batch["feature_input"],  # type: ignore[arg-type]
            response_size=batch["response_size"],  # type: ignore[arg-type]
            trade_occured=batch["trade_occured"],  # type: ignore[arg-type]
            targets=batch["targets"],  # type: ignore[arg-type]
            stock_id=batch.get("stock_id"),  # type: ignore[arg-type]
            date_ordinal=batch.get("date_ordinal"),  # type: ignore[arg-type]
        )
        # Perfect predictions so error metrics are trivially checkable.
        obj.model_output = obj.targets.clone()
        return obj


def _toy_batch(response_size: int = 2) -> dict[str, object]:
    # B=1, S=4, 3 channels; prefix cols 0..1, response cols 2..3, all traded.
    targets = torch.tensor([[[0.0, 0.1, 0.2], [0.0, 0.3, 0.4], [0.0, 0.5, 0.6], [0.0, 0.7, 0.8]]])
    return {
        "feature_input": torch.zeros(1, 4, 12),
        "targets": targets,
        "trade_occured": torch.ones(1, 4, dtype=torch.bool),
        "response_size": torch.tensor(response_size),
    }


def test_accumulate_targets_reports_persistence_baseline() -> None:
    model = _FakeModel()
    acc = accumulate_targets(model, [_toy_batch()], max_batches=1)  # type: ignore[arg-type]

    assert "upside" in acc.baselines
    # Last traded prefix upside value is col 1 -> 0.3, broadcast over the horizon.
    torch.testing.assert_close(acc.baselines["upside"], torch.tensor([0.3, 0.3]))


def test_dedupe_keeps_first_per_ticker_date() -> None:
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([10.0, 20.0, 30.0])
    ids = torch.tensor([5, 5, 6])
    dates = torch.tensor([100, 100, 100])  # ticker 5 appears twice on day 100

    dp, dt, dd = dedupe_by_ticker_date(pred, target, ids, dates)

    assert dp.tolist() == [1.0, 3.0]  # second (5,100) dropped
    assert dt.tolist() == [10.0, 30.0]
    assert dd == ["100", "100"]


def _toy_identity_batch() -> dict[str, object]:
    # B=2 (two tickers), S=3, response_size=1. Same date so they form one day's
    # cross-section; predictions rank the two names in target order.
    targets = torch.tensor(
        [
            [[0.0, 0.1, 0.1], [0.0, 0.1, 0.1], [0.02, 0.1, 0.1]],
            [[0.0, 0.1, 0.1], [0.0, 0.1, 0.1], [-0.01, 0.1, 0.1]],
        ]
    )
    return {
        "feature_input": torch.zeros(2, 3, 12),
        "targets": targets,
        "trade_occured": torch.ones(2, 3, dtype=torch.bool),
        "response_size": torch.tensor(1),
        "stock_id": torch.tensor([5, 6]),
        "date_ordinal": torch.tensor([[10, 11, 12], [10, 11, 12]]),
    }


def test_evaluate_model_reports_rank_ic() -> None:
    from ophir import evaluate as ev

    model = _FakeModel()  # perfect predictions
    out = ev.evaluate_model(model, [_toy_identity_batch()], max_batches=1)  # type: ignore[arg-type]

    assert "rank_ic_mean" in out["r_close"]
    # One day, two names, perfectly ranked -> IC is 1.0 (two distinct tickers on
    # day 12 survive dedupe). float32 Spearman lands a hair above 1.0.
    assert abs(out["r_close"]["rank_ic_mean"] - 1.0) < 1e-6


def test_target_metrics_perfect_prediction_is_zero() -> None:
    values = torch.tensor([0.1, -0.2, 0.3, 0.0])
    metrics = target_metrics(values, values)

    assert metrics["n"] == 4.0
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["bias"] == 0.0


def test_target_metrics_constant_offset() -> None:
    target = torch.tensor([1.0, 2.0, 3.0])
    pred = target + 2.0  # constant +2 error

    metrics = target_metrics(pred, target)

    assert metrics["n"] == 3.0
    assert math.isclose(metrics["mae"], 2.0)
    assert math.isclose(metrics["rmse"], 2.0)
    assert math.isclose(metrics["bias"], 2.0)


def test_target_metrics_empty_is_nan() -> None:
    empty = torch.empty(0)
    metrics = target_metrics(empty, empty)

    assert metrics["n"] == 0.0
    assert math.isnan(metrics["mae"])
    assert math.isnan(metrics["rmse"])
    assert math.isnan(metrics["bias"])


def test_directional_accuracy_all_correct() -> None:
    pred = torch.tensor([0.5, -0.5, 0.2])
    target = torch.tensor([0.1, -0.9, 0.3])

    assert directional_accuracy(pred, target) == 1.0


def test_directional_accuracy_all_wrong() -> None:
    pred = torch.tensor([0.5, -0.5, 0.2])
    target = torch.tensor([-0.1, 0.9, -0.3])

    assert directional_accuracy(pred, target) == 0.0


def test_skill_score_beats_baseline() -> None:
    target = torch.tensor([1.0, -1.0, 1.0, -1.0])
    pred = target * 0.9  # close to target, far better than predicting zero

    assert skill_score(pred, target) > 0.0


def test_skill_score_ties_zero_baseline() -> None:
    target = torch.tensor([1.0, -1.0, 1.0, -1.0])
    pred = torch.zeros_like(target)  # exactly the zero baseline

    assert math.isclose(skill_score(pred, target), 0.0, abs_tol=1e-7)


def test_skill_score_worse_than_baseline() -> None:
    target = torch.tensor([1.0, -1.0, 1.0, -1.0])
    pred = target * -1.0  # sign-flipped: larger error than predicting zero

    assert skill_score(pred, target) < 0.0


def test_format_report_contains_targets_and_labels() -> None:
    results = {
        "best-val": {
            "r_close": {
                "n": 10.0,
                "mae": 0.1234,
                "rmse": 0.2345,
                "bias": -0.01,
                "directional_accuracy": 0.55,
                "skill_score": 0.12,
            },
            "upside": {"n": 10.0, "mae": 0.05, "rmse": 0.06, "bias": 0.0},
            "downside": {"n": 10.0, "mae": 0.04, "rmse": 0.05, "bias": 0.0},
        }
    }

    report = format_report(results)

    for target in ("r_close", "upside", "downside"):
        assert f"### {target}" in report
    assert "best-val" in report
    assert "directional_accuracy" in report
    assert "0.12340" in report  # mae formatted to 5 decimals


def test_format_report_two_checkpoints_side_by_side() -> None:
    base = {
        "r_close": {"n": 5.0, "mae": 0.1, "rmse": 0.2, "bias": 0.0},
        "upside": {"n": 5.0, "mae": 0.1, "rmse": 0.2, "bias": 0.0},
        "downside": {"n": 5.0, "mae": 0.1, "rmse": 0.2, "bias": 0.0},
    }
    report = format_report({"best-val": base, "time-interval": base})

    assert "| metric | best-val | time-interval |" in report


def test_rank_ic_perfect_daily_ranking() -> None:
    # Two days, three names each; predictions rank names identically to targets.
    dates = ["d1", "d1", "d1", "d2", "d2", "d2"]
    target = torch.tensor([0.03, 0.01, -0.02, -0.01, 0.04, 0.00])
    pred = torch.tensor([3.0, 2.0, 1.0, 1.0, 3.0, 2.0])  # same within-day order

    result = rank_ic(pred, target, dates)

    assert result["n_days"] == 2
    assert abs(result["ic_mean"] - 1.0) < 1e-6
    assert result["ic_std"] < 1e-6


def test_rank_ic_inverted_ranking_is_negative() -> None:
    dates = ["d1", "d1", "d1"]
    target = torch.tensor([0.03, 0.01, -0.02])
    pred = torch.tensor([-3.0, -2.0, -1.0])  # exactly reversed

    result = rank_ic(pred, target, dates)

    assert abs(result["ic_mean"] + 1.0) < 1e-6


def test_skill_vs_baseline_positive_when_model_beats_baseline() -> None:
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    pred = target.clone()  # perfect
    baseline = torch.tensor([0.0, 0.0, 0.0, 0.0])  # naive

    assert abs(skill_score_vs_baseline(pred, target, baseline) - 1.0) < 1e-6


def test_skill_vs_baseline_is_nan_when_baseline_is_perfect() -> None:
    target = torch.tensor([1.0, 2.0, 3.0])
    assert skill_score_vs_baseline(target, target, target) != skill_score_vs_baseline(
        target, target, target
    )  # nan != nan


def test_prefix_last_observed_picks_last_traded_prefix_day() -> None:
    # B=1, S=5, response_size=2 -> prefix is columns 0..2.
    values = torch.tensor([[10.0, 20.0, 30.0, 99.0, 99.0]])
    trade = torch.tensor([[True, True, False, True, True]])  # col 2 is a pad day
    out = prefix_last_observed(values, trade, response_size=2)
    # Last traded prefix column is 1 (col 2 did not trade), so value 20.0.
    torch.testing.assert_close(out, torch.tensor([20.0]))


def test_prefix_last_observed_falls_back_to_position_zero() -> None:
    values = torch.tensor([[5.0, 6.0, 7.0, 8.0]])
    trade = torch.tensor([[False, False, False, True]])  # no traded prefix day (S-rs=2)
    out = prefix_last_observed(values, trade, response_size=2)
    torch.testing.assert_close(out, torch.tensor([5.0]))


def test_spearman_constant_pred_is_nan() -> None:
    pred = torch.full((5,), 0.5)
    target = torch.tensor([0.1, 0.3, -0.2, 0.4, 0.0])
    assert math.isnan(_spearman(pred, target))


def test_spearman_constant_target_is_nan() -> None:
    pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    target = torch.full((5,), 0.0)
    assert math.isnan(_spearman(pred, target))


def test_spearman_perfect_ranking_still_finite() -> None:
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([0.1, 0.5, 0.9])
    assert abs(_spearman(pred, target) - 1.0) < 1e-6


def test_rank_ic_skips_constant_pred_day() -> None:
    # Day "bad" has all-constant predictions (zero variance); day "good" has a
    # clean monotonic relationship.  Only the good day must be counted.
    dates = ["good", "good", "good", "bad", "bad", "bad"]
    target = torch.tensor([0.03, 0.01, -0.02, 0.05, -0.01, 0.02])
    pred = torch.tensor([3.0, 2.0, 1.0, 0.5, 0.5, 0.5])  # bad day is flat

    result = rank_ic(pred, target, dates)

    assert result["n_days"] == 1.0
    assert abs(result["ic_mean"] - 1.0) < 1e-6


def test_rank_ic_by_offset_buckets_independently() -> None:
    # One day, 3 tickers. offset 1 rows rank perfectly with target; offset 2 rows
    # are perfectly anti-ranked; offset 5 has no rows.
    pred = torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 3.0, 3.0, 2.0, 1.0])
    ids = torch.tensor([1, 2, 3, 1, 2, 3])
    dates = torch.tensor([1, 1, 1, 1, 1, 1])
    offsets = torch.tensor([1, 1, 1, 2, 2, 2])
    out = rank_ic_by_offset(pred, target, ids, dates, offsets, buckets=(1, 2, 5))
    assert out["h1"] == pytest.approx(1.0)
    assert out["h2"] == pytest.approx(-1.0)
    assert math.isnan(out["h5"])
