"""Unit tests for the CPU-safe metric core in :mod:`ophir.evaluate`.

The accumulation path runs the CUDA forward, but the metric helpers and the
report formatter operate on plain tensors, so they are exercised here without a
model or a GPU.
"""

import math

import torch

from ophir.evaluate import (
    directional_accuracy,
    format_report,
    rank_ic,
    skill_score,
    target_metrics,
)


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
