"""Tests for the NYSE trading-calendar helpers (deterministic via an injected ``now``).

Anchored on 2026-06-19 (Juneteenth, a Friday NYSE holiday) so the holiday / weekend
paths are exercised against a real exchange-calendar entry, not the wall clock.
"""

import pandas as pd

from ophir.agent.market_calendar import is_trading_day, last_closed_session


def _et(stamp):
    return pd.Timestamp(stamp, tz="America/New_York")


def test_mid_session_returns_prior_day():
    # Wed 2026-06-17 13:30 ET: market still open -> last closed session is Tue 6/16.
    assert str(last_closed_session(_et("2026-06-17 13:30")).date()) == "2026-06-16"


def test_after_close_returns_today():
    # Wed 2026-06-17 16:30 ET: after the 16:00 close -> today's session counts.
    assert str(last_closed_session(_et("2026-06-17 16:30")).date()) == "2026-06-17"


def test_weekend_returns_last_session():
    # Sat 2026-06-20: the prior Friday 6/19 is Juneteenth (closed) -> Thu 6/18.
    assert str(last_closed_session(_et("2026-06-20 12:00")).date()) == "2026-06-18"


def test_holiday_returns_prior_session():
    # Juneteenth Fri 2026-06-19 midday -> last real session is Thu 6/18.
    assert str(last_closed_session(_et("2026-06-19 12:00")).date()) == "2026-06-18"


def test_naive_input_treated_as_eastern():
    # A tz-naive timestamp is read as US-Eastern wall-clock.
    assert str(last_closed_session(pd.Timestamp("2026-06-17 13:30")).date()) == "2026-06-16"


def test_is_trading_day():
    assert is_trading_day(pd.Timestamp("2026-06-17"))  # Wednesday session
    assert not is_trading_day(pd.Timestamp("2026-06-19"))  # Juneteenth
    assert not is_trading_day(pd.Timestamp("2026-06-20"))  # Saturday
