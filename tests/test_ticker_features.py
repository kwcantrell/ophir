"""Tests for ``extract_features`` and ``extract_model_data``."""

import numpy as np
import pandas as pd
import pytest
import torch

from ophir.ticker import extract_features, extract_model_data

# --------------------------------------------------------------------------- #
# extract_features
# --------------------------------------------------------------------------- #


def test_extract_features_column_layout(ohlcv_df, feature_cols):
    out = extract_features(ohlcv_df)

    # 13 features + the trailing bool ``trade_occured`` flag (14 total).
    assert list(out.columns) == [*feature_cols, "trade_occured"]
    assert out["trade_occured"].dtype == np.bool_


def test_extract_features_reindexes_to_daily_calendar(ohlcv_df):
    out = extract_features(ohlcv_df)

    expected = pd.date_range(ohlcv_df.index.min(), ohlcv_df.index.max(), freq="D")
    pd.testing.assert_index_equal(out.index, expected)


def test_extract_features_pads_weekends(ohlcv_df, feature_cols):
    out = extract_features(ohlcv_df)

    weekend = out.index[out.index.dayofweek >= 5]
    weekday_with_data = ohlcv_df.index[10]  # a known business day inside the range

    assert (~out.loc[weekend, "trade_occured"]).all()
    assert (out.loc[weekend, feature_cols] == 0.0).all().all()
    assert bool(out.loc[weekday_with_data, "trade_occured"])


def test_extract_features_has_no_nans(ohlcv_df):
    out = extract_features(ohlcv_df)
    assert not out.isna().to_numpy().any()


def test_extract_features_single_row(make_ohlcv, feature_cols):
    out = extract_features(make_ohlcv(n_days=1))

    assert len(out) == 1
    assert list(out.columns) == [*feature_cols, "trade_occured"]
    assert bool(out["trade_occured"].iloc[0])


def test_extract_features_empty_input_early_return(empty_ohlcv_df, feature_cols):
    # Quirk (pinned): the empty-input branch returns *before* padding/slicing,
    # so the result is the un-sliced frame -- original OHLCV columns plus the
    # 13 intermediate feature columns, but WITHOUT ``trade_occured``.
    out = extract_features(empty_ohlcv_df)

    assert out.empty
    assert list(out.columns) == ["high", "low", "close", "volume", *feature_cols]
    assert "trade_occured" not in out.columns


# --------------------------------------------------------------------------- #
# extract_model_data
# --------------------------------------------------------------------------- #


@pytest.fixture
def feature_window(ohlcv_df):
    return extract_features(ohlcv_df).iloc[:30]


def test_extract_model_data_int_response_size(feature_window, feature_cols):
    md = extract_model_data(feature_window, response_size=5)

    assert set(md) == {"feature_input", "targets", "trade_occured", "response_size"}
    assert md["feature_input"].dtype == torch.float32
    # the bool ``trade_occured`` column is excluded from feature_input
    assert md["feature_input"].shape == (len(feature_window), len(feature_cols))
    assert md["targets"].shape == (len(feature_window), 3)
    assert md["trade_occured"].dtype == torch.bool
    assert torch.equal(md["response_size"], torch.tensor([5]))


def test_extract_model_data_return_date_adds_time(feature_window):
    md = extract_model_data(feature_window, response_size=5, return_date=True)

    assert "time" in md
    assert md["time"].dtype == np.dtype("datetime64[ns]")
    assert len(md["time"]) == len(feature_window)


def test_extract_model_data_array_response_size_double_wraps(feature_window):
    # Dataset callers pass ``np.array([rs])`` into a function that wraps again,
    # producing a (1, 1) tensor. ``ui.py`` passes a raw int -> (1,). This
    # inconsistency is pinned, not fixed (changing it risks the model contract).
    md = extract_model_data(feature_window, response_size=np.array([5]))

    assert md["response_size"].shape == (1, 1)
    assert torch.equal(md["response_size"], torch.tensor([[5]]))


def test_np_bool_alias_available():
    # ``extract_model_data`` relies on ``np.bool`` (restored in numpy 2.0).
    # The project pins numpy>=2.2.6; this guard fails loudly on a downgrade.
    assert hasattr(np, "bool")
