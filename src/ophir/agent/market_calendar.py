"""NYSE trading-calendar helpers for the live trading loop.

Answers two questions the daily run needs before it forecasts or trades:

* :func:`last_closed_session` -- the date of the most recent NYSE session that has
  already *closed*, so the model conditions on completed bars (T-1 close at a
  mid-session run) and never on today's still-forming bar.
* :func:`is_trading_day` -- whether a given calendar day is an NYSE session, so the
  unattended task can fail safe (skip the cycle) on weekends and holidays rather
  than forecasting off stale data and queuing orders.

Anchored on UTC / US-Eastern, never the machine clock, so the answer is correct
regardless of the host timezone. Named ``market_calendar`` to avoid shadowing the
stdlib ``calendar`` module. ``pandas_market_calendars`` is imported lazily (it pulls
in several calendar packages) and cached.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

_LOOKBACK_DAYS = 12  # comfortably spans any weekend + holiday cluster
_EASTERN = "America/New_York"


@lru_cache(maxsize=1)
def _nyse() -> Any:
    """Return the cached NYSE market calendar."""
    import pandas_market_calendars as mcal

    return mcal.get_calendar("NYSE")


def _to_utc(now: Any) -> pd.Timestamp:
    """Coerce ``now`` (timestamp-like or ``None``) to a tz-aware UTC timestamp.

    A naive input is read as US-Eastern wall-clock time; ``None`` means right now.
    """
    ts = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if ts.tzinfo is None:
        ts = ts.tz_localize(_EASTERN)
    return ts.tz_convert("UTC")


def last_closed_session(now: Any = None) -> pd.Timestamp:
    """Return the date of the most recent NYSE session already closed by ``now``.

    Parameters
    ----------
    now : timestamp-like, optional
        The reference instant. Defaults to the current time. A tz-naive value is
        interpreted as US-Eastern wall-clock; a tz-aware value is honoured as given.

    Returns
    -------
    pandas.Timestamp
        A tz-naive midnight timestamp (matching the parquet index) for the session
        date. At a mid-session run this is the prior trading day; after the close it
        is today; after a weekend / holiday it is the last real session.

    Raises
    ------
    ValueError
        If no NYSE session has closed within the trailing lookback window.
    """
    now_utc = _to_utc(now)
    sched = _nyse().schedule(
        start_date=(now_utc - pd.Timedelta(days=_LOOKBACK_DAYS)).date(),
        end_date=now_utc.date(),
    )
    closed = sched.index[sched["market_close"] <= now_utc]
    if len(closed) == 0:
        raise ValueError(f"no closed NYSE session in the {_LOOKBACK_DAYS} days before {now_utc}")
    return pd.Timestamp(closed[-1]).normalize()


def is_trading_day(day: Any = None) -> bool:
    """Return ``True`` if ``day`` is an NYSE session.

    Parameters
    ----------
    day : date-like, optional
        The calendar day to test. Defaults to today in US-Eastern. A tz-aware value
        is first converted to US-Eastern so the calendar date is unambiguous.
    """
    if day is None:
        target = pd.Timestamp.now(tz=_EASTERN).date()
    else:
        ts = pd.Timestamp(day)
        target = (ts.tz_convert(_EASTERN) if ts.tzinfo is not None else ts).date()
    return bool(len(_nyse().valid_days(start_date=target, end_date=target)) > 0)
