"""Shared fixtures for the ``ophir.ticker`` test suite.

All fixtures are deterministic (seeded) and never touch the network or the
package's on-disk ``.ophir/`` layout. Filesystem fixtures use pytest's
built-in ``tmp_path``.
"""

import numpy as np
import pandas as pd
import pytest

from ophir.ticker import StockSplit

# The 12 feature columns produced by ``extract_features`` (in order). The
# non-empty path additionally appends ``trade_occured`` (a bool flag, not a
# feature) before slicing, so its output has 13 columns.
FEATURE_COLS = [
    "r_close",
    "10_norm_returns",
    "10_norm_volume",
    "10_volatility",
    "20_norm_returns",
    "20_norm_volume",
    "20_volatility",
    "60_norm_returns",
    "60_norm_volume",
    "60_volatility",
    "upside",
    "downside",
]


def _make_ohlcv(
    n_days: int = 120,
    start: str = "2020-01-01",
    seed: int = 1234,
    freq: str = "B",
    base_price: float = 100.0,
):
    """Build a deterministic datetime-indexed OHLCV frame.

    ``freq="B"`` (business days) is deliberate: ``extract_features`` and
    ``get_start_dates`` reindex onto a daily calendar, so weekends become
    ``trade_occured == False`` padding rows, exercising the real pad path.
    Prices are strictly positive so the ``np.log`` features never warn.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=n_days, freq=freq)
    rets = rng.normal(0.0, 0.01, size=n_days)
    close = base_price * np.exp(np.cumsum(rets))
    high = close * (1.0 + np.abs(rng.normal(0.0, 0.005, size=n_days)))
    low = close * (1.0 - np.abs(rng.normal(0.0, 0.005, size=n_days)))
    volume = rng.integers(1_000, 1_000_000, size=n_days).astype(float)
    return pd.DataFrame(
        {"high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.fixture
def feature_cols():
    """The ordered 12 feature column names ``extract_features`` produces."""
    return list(FEATURE_COLS)


@pytest.fixture
def make_ohlcv():
    """Factory: ``make_ohlcv(n_days=..., seed=..., freq=...)`` -> OHLCV frame."""
    return _make_ohlcv


@pytest.fixture
def ohlcv_df(make_ohlcv):
    """A default 120-business-day deterministic OHLCV frame."""
    return make_ohlcv()


@pytest.fixture
def empty_ohlcv_df():
    """An empty OHLCV frame with an empty ``DatetimeIndex``."""
    return pd.DataFrame(
        {"high": [], "low": [], "close": [], "volume": []},
        index=pd.DatetimeIndex([]),
    )


@pytest.fixture
def raw_tick_df(make_ohlcv):
    """Raw pre-aggregation frame with a ``utc_time`` column, 2 ticks/day.

    Mirrors what ``StockHandler.stock_df`` expects to read from parquet
    (before it does its own ``groupby("date")`` aggregation).
    """
    base = make_ohlcv(n_days=80, seed=99)
    rows = []
    for ts, row in base.iterrows():
        for hour in (10, 14):
            rows.append(
                {
                    "utc_time": ts + pd.Timedelta(hours=hour),
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"] / 2.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def parquet_dir(tmp_path, raw_tick_df):
    """A Hive-partitioned parquet tree under ``tmp_path``.

    Returns ``(base_path, {symbol: parquet_path})`` with:

    * ``AAA`` -- full history (passes every filter),
    * ``BBB`` -- ~5-day span (fails ``min_ticker_history``),
    * ``CCC`` -- volume == 1.0 (fails ``min_volume``),
    * a decoy ``_logs/`` dir (no ``=``) that must be ignored.
    """
    cutoff = raw_tick_df["utc_time"].min() + pd.Timedelta(days=5)
    bbb = raw_tick_df[raw_tick_df["utc_time"] < cutoff].copy()
    ccc = raw_tick_df.copy()
    ccc["volume"] = 1.0

    symbols = {"AAA": raw_tick_df, "BBB": bbb, "CCC": ccc}
    paths = {}
    for sym, frame in symbols.items():
        part = tmp_path / f"symbol={sym}"
        part.mkdir()
        parquet_path = part / "data.parquet"
        frame.to_parquet(parquet_path)
        paths[sym] = str(parquet_path)

    (tmp_path / "_logs").mkdir()  # decoy: no '=', must be excluded
    return str(tmp_path), paths


@pytest.fixture
def stock_split():
    """A single 2:1 split on 2020-03-01, dates as ``np.datetime64``.

    Mirrors the shape ``get_splits`` produces.
    """
    return StockSplit(
        id="AAA",
        dates=[np.datetime64("2020-03-01")],
        ratios=[2.0],
    )


@pytest.fixture
def seed_np_random():
    """Seed the *global* ``np.random`` state (the code uses the legacy API).

    Not autouse: only RNG-touching tests request it, so accidental RNG
    coupling stays visible elsewhere.
    """
    np.random.seed(0)
    yield
    np.random.seed()
