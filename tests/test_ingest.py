"""Tests for ``ophir.agent.data`` ingestion (Yahoo Finance is mocked).

Network is patched out via ``yfinance.Ticker``; the parquet root is redirected
to ``tmp_path`` so nothing touches the package's ``.ophir/`` layout.
"""

import pandas as pd
import pytest

from ophir.agent.feed import latest_window_tensors, load_daily_ohlcv
from ophir.agent.ingest import ingest, ingest_many


def _yahoo_frame(make_ohlcv, n_days=500):
    """A yfinance-like frame: Title-case OHLCV, tz-aware ``Date`` index."""
    base = make_ohlcv(n_days=n_days)
    df = base.rename(columns={c: c.capitalize() for c in base.columns})
    df.insert(0, "Open", df["Close"].shift(1).fillna(df["Close"]))
    df.index = df.index.tz_localize("UTC")
    df.index.name = "Date"
    return df


@pytest.fixture
def fake_yahoo(monkeypatch, make_ohlcv):
    """Patch ``yfinance.Ticker`` to return a deterministic 500-day frame."""
    frame = _yahoo_frame(make_ohlcv, n_days=500)

    class _FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            return frame.copy()

    monkeypatch.setattr("yfinance.Ticker", _FakeTicker)
    return frame


def test_ingest_writes_model_ready_parquet(fake_yahoo, tmp_path):
    dest = ingest("test", days=730, stocks_dir=str(tmp_path))

    assert dest == tmp_path / "symbol=TEST" / "data.parquet"
    assert dest.exists()
    out = pd.read_parquet(dest)
    assert list(out.columns) == ["utc_time", "high", "low", "close", "volume"]
    assert len(out) == 500
    assert out["utc_time"].dt.tz is None  # tz-naive, per ophir convention


def test_load_daily_ohlcv_roundtrip(fake_yahoo, tmp_path):
    ingest("TEST", days=730, stocks_dir=str(tmp_path))

    df = load_daily_ohlcv("TEST", stocks_dir=str(tmp_path))
    assert list(df.columns) == ["high", "low", "close", "volume"]
    assert df.index.name == "utc_time"
    assert df.index.is_monotonic_increasing


def test_latest_window_tensors_shapes(fake_yahoo, tmp_path):
    ingest("TEST", days=730, stocks_dir=str(tmp_path))

    md = latest_window_tensors("TEST", seq_len=30, response_size=10, stocks_dir=str(tmp_path))
    assert tuple(md["feature_input"].shape) == (30, 13)
    assert tuple(md["targets"].shape) == (30, 3)
    assert tuple(md["trade_occured"].shape) == (30,)


def test_ingest_many_skips_failures(monkeypatch, make_ohlcv, tmp_path):
    good = _yahoo_frame(make_ohlcv, n_days=120)

    class _MixedTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            if self.symbol == "BAD":
                return pd.DataFrame()
            return good.copy()

    monkeypatch.setattr("yfinance.Ticker", _MixedTicker)

    paths = ingest_many(["good", "BAD"], days=200, stocks_dir=str(tmp_path))
    assert set(paths) == {"GOOD"}
    assert paths["GOOD"].exists()


def test_ingest_unknown_symbol_raises(monkeypatch, tmp_path):
    class _EmptyTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            return pd.DataFrame()

    monkeypatch.setattr("yfinance.Ticker", _EmptyTicker)

    with pytest.raises(ValueError, match="No Yahoo Finance data"):
        ingest("ZZZZ", days=730, stocks_dir=str(tmp_path))
