"""Tests for the backtest layer (fake model + synthetic data; no GPU/network)."""

import types

import numpy as np
import pandas as pd

from ophir.agent import backtest as bt_mod
from ophir.agent.backtest import (
    _as_of_input,
    compute_metrics,
    purged_kfold,
    signal_cv,
    walk_forward,
)


def test_compute_metrics_known_series():
    m = compute_metrics([0.10, -0.05, 0.10], periods_per_year=12.0)
    assert abs(m["total_return"] - (1.1 * 0.95 * 1.1 - 1.0)) < 1e-9
    assert m["sharpe"] > 0
    assert m["max_drawdown"] < 0  # the -5% period draws down
    assert m["hit_rate"] == 2 / 3


def test_compute_metrics_empty():
    m = compute_metrics([])
    assert m["sharpe"] == 0.0
    assert m["total_return"] == 0.0


def test_purged_kfold_no_overlap_or_embargo():
    n, horizon = 20, 2
    label_end = [min(i + horizon, n - 1) for i in range(n)]
    folds = purged_kfold(n, label_end, n_splits=4, embargo=1)
    assert folds
    for train, test in folds:
        assert set(train).isdisjoint(test)
        test_lo, test_hi = test[0], test[-1]
        for i in train:
            assert not (i <= test_hi and label_end[i] >= test_lo)  # purge respected
            assert not (test_hi < i <= test_hi + 1)  # embargo respected


def test_signal_cv_detects_correlation():
    n = 40
    predicted = [float(i) for i in range(n)]
    realized = [float(i) + (0.1 if i % 2 else -0.1) for i in range(n)]
    out = signal_cv(predicted, realized, list(range(n)), n_splits=4, embargo=0)
    assert out["n_folds"] >= 1
    assert out["mean_ic"] > 0.5


def test_signal_cv_constant_prediction_is_safe():
    n = 20
    out = signal_cv([0.05] * n, [float(i) for i in range(n)], list(range(n)), n_splits=4)
    assert out["mean_ic"] == 0.0  # constant predictions -> nan ICs skipped, no crash


class _Tensorish:
    """Minimal tensor-like stand-in supporting the predict-path call chain."""

    def __init__(self, arr):
        self._arr = arr

    def detach(self):
        return self

    def cpu(self):
        return self

    def reshape(self, *_args):
        return self

    def tolist(self):
        return list(self._arr)


def _fake_model(daily_r=0.0003):
    class _Model:
        def __call__(self, _md):
            base = np.full(90, daily_r, dtype="float32")
            return types.SimpleNamespace(
                predicted_r_close=_Tensorish(base),
                predicted_downside=_Tensorish(np.full(90, 0.01, dtype="float32")),
            )

    return _Model()


def _frame(n=700, start="2023-01-01", seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=n, freq="D")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    df = pd.DataFrame(
        {
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1_000, 1_000_000, n).astype(float),
        },
        index=idx,
    )
    df.index.name = "utc_time"
    return df


def test_as_of_input_no_lookahead():
    frame = _frame()
    asof = pd.Timestamp("2024-01-15")
    full = _as_of_input(frame, asof)
    truncated = _as_of_input(frame[frame.index <= asof], asof)
    # appending later bars must not change the as-of input
    assert np.allclose(full["feature_input"].numpy(), truncated["feature_input"].numpy())


def test_walk_forward_runs(monkeypatch):
    monkeypatch.setattr(bt_mod.audit, "log_event", lambda *a, **k: None)
    frame = _frame()
    monkeypatch.setattr(bt_mod, "load_daily_ohlcv", lambda _s, **_k: frame.copy())
    result = walk_forward(
        ["AAA", "BBB"],
        model=_fake_model(),
        benchmark=frame["close"],
        start="2024-01-01",
        end="2024-06-01",
        rebalance_days=21,
        cv_splits=3,
    )
    assert result.n_rebalances >= 1
    assert len(result.equity_curve) >= 2
    assert "sharpe" in result.metrics
    assert "mean_ic" in result.ic
