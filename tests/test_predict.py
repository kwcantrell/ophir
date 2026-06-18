"""Tests for agent prediction (model + GPU mocked, no real checkpoint/data)."""

import types

import pandas as pd
import torch

from ophir.agent import predict as predict_mod
from ophir.agent.predict import Forecast, predict_ticker, rank


def _fake_out(n=90):
    base = torch.zeros(n, dtype=torch.float32)
    return types.SimpleNamespace(
        predicted_r_close=base + 0.001,
        predicted_upside=base + 1.01,
        predicted_downside=base + 0.99,
    )


class _FakeModel:
    def __call__(self, md):
        return _fake_out()


def test_predict_ticker_builds_forecast(monkeypatch, tmp_path):
    monkeypatch.setattr(predict_mod, "forecast_window_tensors", lambda *a, **k: {})
    monkeypatch.setattr(predict_mod.audit, "log_event", lambda *a, **k: None)
    idx = pd.to_datetime(["2026-06-01", "2026-06-02", "2026-06-03"])
    df = pd.DataFrame(
        {"high": [1, 2, 3], "low": [1, 2, 3], "close": [1, 2, 3], "volume": [1, 2, 3]},
        index=idx,
    )
    df.index.name = "utc_time"
    monkeypatch.setattr(predict_mod, "load_history", lambda *a, **k: df)

    fc = predict_ticker("aapl", model=_FakeModel(), ensure_data=False, stocks_dir=str(tmp_path))

    assert isinstance(fc, Forecast)
    assert fc.symbol == "AAPL"
    assert fc.asof == "2026-06-03"
    assert fc.horizon == 90
    assert len(fc.r_close) == 90
    assert fc.cum_return > 0
    assert fc.score == fc.cum_return


def test_rank_orders_and_truncates():
    fcs = [
        Forecast(
            symbol=s,
            asof="2026-06-03",
            horizon=1,
            r_close=[r],
            upside=[1.0],
            downside=[1.0],
            cum_return=r,
            score=r,
        )
        for s, r in [("A", 0.1), ("B", 0.3), ("C", 0.2), ("D", -0.1)]
    ]
    top = rank(fcs, top_k=2)
    assert [f.symbol for f in top] == ["B", "C"]
