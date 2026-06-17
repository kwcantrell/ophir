"""Tests for feed.load_history: calendar-aware as-of trim + stale-feed guard.

``load_daily_ohlcv`` is monkeypatched to a synthetic frame so no parquet/network is
touched; ``as_of`` is passed explicitly so the result does not depend on the clock.
"""

import pandas as pd
import pytest

from ophir.agent import feed


def _frame(dates):
    idx = pd.to_datetime(dates)
    return pd.DataFrame({"high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)


def test_drops_forming_bar(monkeypatch):
    # Bars through Tue 6/16 plus a forming Wed 6/17 row; as_of=Tue drops 6/17.
    df = _frame(["2026-06-12", "2026-06-15", "2026-06-16", "2026-06-17"])
    monkeypatch.setattr(feed, "load_daily_ohlcv", lambda *a, **k: df)
    out = feed.load_history("AAPL", as_of="2026-06-16")
    assert str(out.index.max().date()) == "2026-06-16"
    assert pd.Timestamp("2026-06-17") not in out.index


def test_keeps_session_bar(monkeypatch):
    df = _frame(["2026-06-15", "2026-06-16"])
    monkeypatch.setattr(feed, "load_daily_ohlcv", lambda *a, **k: df)
    out = feed.load_history("AAPL", as_of="2026-06-16")
    assert len(out) == 2


def test_raises_on_stale_feed(monkeypatch):
    # Latest bar is 6/15 but the expected session is 6/16 -> stale.
    df = _frame(["2026-06-12", "2026-06-15"])
    monkeypatch.setattr(feed, "load_daily_ohlcv", lambda *a, **k: df)
    with pytest.raises(ValueError, match="stale"):
        feed.load_history("AAPL", as_of="2026-06-16")


def test_raises_on_empty_after_cutoff(monkeypatch):
    df = _frame(["2026-06-20"])  # only a bar after the cutoff
    monkeypatch.setattr(feed, "load_daily_ohlcv", lambda *a, **k: df)
    with pytest.raises(ValueError, match="No bars"):
        feed.load_history("AAPL", as_of="2026-06-16")
