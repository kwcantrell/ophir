"""Tests for the network functions, with all network fully mocked.

No test here touches the real network or the package's ``.ophir/`` layout:
``pandas.read_html`` and ``yfinance.Ticker`` are patched, and ``get_splits``
is always given an explicit ``cache_path`` under ``tmp_path``.
"""

import pickle

import numpy as np
import pandas as pd
import pytest

from ophir.ticker import StockSplit, get_sp_500_symbols, get_splits


@pytest.fixture(autouse=True)
def _block_real_read_html(mocker):
    """Fail loudly if a test reaches ``pd.read_html`` without overriding it."""
    mocker.patch(
        "ophir.ticker.splits.pd.read_html",
        side_effect=AssertionError("unmocked network call to read_html"),
    )


# --------------------------------------------------------------------------- #
# get_sp_500_symbols
# --------------------------------------------------------------------------- #


def test_get_sp_500_symbols_parses_symbol_column(mocker):
    table = pd.DataFrame({"Symbol": ["AAA", "BBB", "CCC"]})
    read_html = mocker.patch("ophir.ticker.splits.pd.read_html", return_value=[table])

    assert get_sp_500_symbols() == ["AAA", "BBB", "CCC"]

    args, kwargs = read_html.call_args
    assert args[0] == "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    assert kwargs["storage_options"]["User-Agent"].startswith("Mozilla/5.0")


def test_get_sp_500_symbols_swallows_error_then_unbound(mocker, capsys):
    # Pinned latent bug: the bare except only prints, leaving ``dfs`` unbound,
    # so the function then raises UnboundLocalError. Ambiguous fix -> pin it.
    mocker.patch("ophir.ticker.splits.pd.read_html", side_effect=RuntimeError("boom"))

    with pytest.raises(UnboundLocalError):
        get_sp_500_symbols()

    assert "An error occurred: boom" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# get_splits
# --------------------------------------------------------------------------- #


@pytest.fixture
def cache_path(tmp_path):
    return str(tmp_path / "cache.pkl")


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _dump(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def test_get_splits_cache_miss_fetches_and_writes(mocker, cache_path):
    series = pd.Series([2.0], index=pd.DatetimeIndex(["2020-03-01"]))
    ticker = mocker.patch("yfinance.Ticker", return_value=mocker.Mock(splits=series))

    result = get_splits(["AAA"], cache_path=cache_path)

    ticker.assert_called_once_with("AAA")
    assert set(result) == {"AAA"}
    split = result["AAA"]
    assert split.id == "AAA"
    assert split.dates == [np.datetime64("2020-03-01")]
    assert split.ratios == [2.0]
    # the cache file was written and round-trips
    assert _load(cache_path)["AAA"].ratios == [2.0]


def test_get_splits_tz_aware_index_is_made_naive(mocker, cache_path):
    series = pd.Series([3.0], index=pd.DatetimeIndex(["2020-03-01"]).tz_localize("US/Eastern"))
    mocker.patch("yfinance.Ticker", return_value=mocker.Mock(splits=series))

    split = get_splits(["AAA"], cache_path=cache_path)["AAA"]

    assert split.dates == [np.datetime64("2020-03-01T00:00:00")]


def test_get_splits_no_splits_records_sentinel(mocker, cache_path):
    mocker.patch(
        "yfinance.Ticker",
        return_value=mocker.Mock(splits=pd.Series([], dtype=float)),
    )

    result = get_splits(["AAA"], cache_path=cache_path)

    assert result == {}  # sentinel excluded from the returned mapping
    assert _load(cache_path) == {"AAA": None}  # but persisted so it isn't re-queried


def test_get_splits_series_none_records_sentinel(mocker, cache_path):
    mocker.patch("yfinance.Ticker", return_value=mocker.Mock(splits=None))

    result = get_splits(["AAA"], cache_path=cache_path)

    assert result == {}
    assert _load(cache_path) == {"AAA": None}


def test_get_splits_full_cache_hit_skips_network(mocker, cache_path):
    cached = {
        "AAA": StockSplit(id="AAA", dates=[np.datetime64("2020-03-01")], ratios=[2.0]),
        "BBB": None,
    }
    _dump(cache_path, cached)
    ticker = mocker.patch("yfinance.Ticker")

    result = get_splits(["AAA", "BBB"], cache_path=cache_path)

    ticker.assert_not_called()
    # the early-return path does NOT filter sentinels (pinned behavior)
    assert result["AAA"].ratios == [2.0]
    assert result["BBB"] is None


def test_get_splits_partial_miss_fetches_only_missing(mocker, cache_path):
    _dump(
        cache_path,
        {"AAA": StockSplit(id="AAA", dates=[np.datetime64("2020-03-01")], ratios=[2.0])},
    )
    series = pd.Series([4.0], index=pd.DatetimeIndex(["2021-06-01"]))
    ticker = mocker.patch("yfinance.Ticker", return_value=mocker.Mock(splits=series))

    result = get_splits(["AAA", "BBB"], cache_path=cache_path)

    ticker.assert_called_once_with("BBB")
    assert result["AAA"].ratios == [2.0]
    assert result["BBB"].ratios == [4.0]
    assert set(_load(cache_path)) == {"AAA", "BBB"}  # cache rewritten with both


def test_get_splits_fetch_failure_is_skipped(mocker, cache_path, capsys):
    mocker.patch("yfinance.Ticker", side_effect=RuntimeError("boom"))

    result = get_splits(["AAA"], cache_path=cache_path)

    assert result == {}
    assert "[get_splits] AAA failed: boom" in capsys.readouterr().out
    assert _load(cache_path) == {}  # nothing recorded, will retry next time
