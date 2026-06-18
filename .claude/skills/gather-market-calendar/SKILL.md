---
name: gather-market-calendar
description: >-
  Use when the live trading loop needs to know whether the NYSE is open today or which
  session's bars to condition on -- gating `ophir trade`, picking the as-of close for
  `load_history`, or skipping weekends/holidays (e.g. Juneteenth). Reuse
  `last_closed_session(now=None)` for the most recent already-closed NYSE session and
  `is_trading_day(day=None)` for an open/closed check; both come from the NYSE
  pandas-market-calendars calendar. Reach for it to fetch, debug, or extend the
  trading-calendar gate rather than re-deriving sessions from raw dates.
---

# Gather NYSE trading calendar

Answer two questions the daily run needs before it forecasts or trades: which NYSE
session has most recently closed, and whether a given day is an NYSE session at all.
See [docs/data-inputs.md](docs/data-inputs.md) for the full data-source catalog.

## Source
- **pandas-market-calendars**, calendar id `"NYSE"` (`mcal.get_calendar("NYSE")`).
- [agent/market_calendar.py:48](src/ophir/agent/market_calendar.py:48) -- `last_closed_session(now=None) -> pd.Timestamp`: the date of the most recent NYSE session already closed by `now`. Returns a tz-naive midnight `Timestamp` matching the parquet index. Raises `ValueError` if no session closed in the trailing 12-day lookback (`_LOOKBACK_DAYS`).
- [agent/market_calendar.py:80](src/ophir/agent/market_calendar.py:80) -- `is_trading_day(day=None) -> bool`: `True` if `day` (default: today in US-Eastern) is an NYSE session.
- [agent/market_calendar.py:30](src/ophir/agent/market_calendar.py:30) -- `_nyse()`: `lru_cache(maxsize=1)`-wrapped calendar accessor; lazy-imports the library on first call.

## Fields
- `last_closed_session` reads the `mcal` `schedule(...)` DataFrame; it filters on the **`market_close`** column (tz-aware **UTC** timestamps) against `now`, then takes the last index entry and `.normalize()`s it to midnight.
- `is_trading_day` uses `valid_days(start_date, end_date)` (a `DatetimeIndex` of sessions); a non-empty result means the day is open.
- Time anchoring: a tz-naive `now`/`day` is read as **US-Eastern** wall-clock (`America/New_York`); tz-aware inputs are honored/converted. The answer never depends on the host machine's timezone.

## Fail-safe & caching
- **No session in lookback -> raises `ValueError`.** Callers fail safe: `ophir trade` no-ops the whole cycle when `is_trading_day()` is `False` ([cli.py:399](src/ophir/cli.py:399)), logging `cycle_skipped` / `not_a_trading_day` and placing no orders -- so weekends, holidays, and full-day closures (e.g. Juneteenth) never queue trades on stale data.
- `_nyse()` is `lru_cache`d, so the calendar (and its lazy import) is built once per process. No network at call time -- the holiday rules ship with the library.
- `last_closed_session` is the **default as-of cutoff** for [agent/feed.py:112](src/ophir/agent/feed.py:112) (`load_history`), which trims each ticker to bars on/before that session (T-1 close at a mid-session run) and refuses a stale feed.

## How to use / extend
- CLI: `ophir trade --broker paper <SYMBOLS...>` self-gates via `is_trading_day()`; `ophir manage <SYMBOLS...>` conditions on `last_closed_session` through `load_history` and does not place orders.
- In code:
  ```python
  from ophir.agent.market_calendar import is_trading_day, last_closed_session

  if not is_trading_day():
      return  # weekend / holiday -- skip the cycle
  asof = last_closed_session()  # tz-naive midnight, matches the parquet index
  ```
- Pass an explicit `now=`/`day=` (timestamp-like, naive = ET) to backtest or reproduce a specific instant deterministically.
- **Add a sibling calendar the same way:** add a helper module (e.g. `agent/market_calendar_<exchange>.py`) that mirrors this one -- an `lru_cache`d accessor calling `mcal.get_calendar("<id>")` (e.g. `"CME"`, `"LSE"`), then `last_closed_session`/`is_trading_day` reading `schedule(...market_close)` and `valid_days(...)`. Keep the UTC/ET anchoring and the 12-day lookback so the fail-safe behavior stays identical.
