"""Tests for ``StockStreamer`` window slicing and reconstruction."""

import numpy as np
import pandas as pd
import pytest

from ophir.ticker import StockStreamer, extract_features, get_starts

# --------------------------------------------------------------------------- #
# construction / sizing
# --------------------------------------------------------------------------- #


def test_streamer_size_matches_get_starts(ohlcv_df):
    streamer = StockStreamer(ohlc_df=ohlcv_df, seq_len=20, offset=5)

    preprocessed = streamer.preprocessed_ohlc_df
    first_valid = int(preprocessed["feature_valid"].to_numpy().argmax())
    expected = get_starts(preprocessed, 20, 5, first_valid_row=first_valid)
    assert streamer.size == len(streamer) == len(expected)


def test_streamer_first_window_starts_at_or_after_first_valid_row(ohlcv_df):
    streamer = StockStreamer(ohlc_df=ohlcv_df, seq_len=20, offset=5)

    preprocessed = streamer.preprocessed_ohlc_df
    first_valid_pos = int(preprocessed["feature_valid"].to_numpy().argmax())
    # The first start must not be before the first valid row.
    assert streamer.starts[0] >= first_valid_pos
    # Every row in the first window must be feature_valid or a padding row.
    first_window = streamer[int(streamer.starts[0])]
    trading_in_window = first_window[first_window["trade_occured"]]
    assert trading_in_window["feature_valid"].all()


def test_streamer_offset_minus_one_is_non_overlapping(ohlcv_df):
    streamer = StockStreamer(ohlc_df=ohlcv_df, seq_len=15, offset=-1)

    # offset == -1 is remapped to seq_len; create_iterator() was built before
    # that remap but is lazy, so next() still uses the corrected starts.
    assert streamer.offset == streamer.seq_len == 15
    first = streamer.next()
    second = streamer.next()
    assert len(first) == 15
    assert len(second) == 15
    # non-overlapping: consecutive starts stride by seq_len
    assert streamer.starts[1] - streamer.starts[0] == 15


def test_streamer_empty_frame(empty_ohlcv_df):
    streamer = StockStreamer(ohlc_df=empty_ohlcv_df, seq_len=10, offset=5)

    assert streamer.starts == []
    assert len(streamer) == 0
    assert not hasattr(streamer, "preprocessed_ohlc_df")
    with pytest.raises(StopIteration):
        streamer.next()


def test_streamer_getitem_returns_seq_len_window(ohlcv_df):
    streamer = StockStreamer(ohlc_df=ohlcv_df, seq_len=20, offset=5)
    assert len(streamer[0]) == 20


# --------------------------------------------------------------------------- #
# iteration
# --------------------------------------------------------------------------- #


def test_create_iterator_yields_all_windows(ohlcv_df):
    streamer = StockStreamer(ohlc_df=ohlcv_df, seq_len=20, offset=5)

    windows = list(streamer.create_iterator())
    assert len(windows) == streamer.size
    assert all(len(w) == 20 for w in windows)


def test_next_raises_stopiteration_when_exhausted(ohlcv_df):
    streamer = StockStreamer(ohlc_df=ohlcv_df, seq_len=20, offset=5)

    collected = [streamer.next() for _ in range(streamer.size)]
    assert len(collected) == streamer.size
    with pytest.raises(StopIteration):
        streamer.next()


def test_shuffle_is_a_permutation(ohlcv_df, seed_np_random):
    ordered = StockStreamer(ohlc_df=ohlcv_df, seq_len=10, offset=5)
    shuffled = StockStreamer(ohlc_df=ohlcv_df, seq_len=10, offset=5, shuffle=True)

    ordered_starts = [w.index[0] for w in ordered.create_iterator()]
    shuffled_starts = [w.index[0] for w in shuffled.create_iterator()]

    assert sorted(shuffled_starts) == sorted(ordered_starts)
    assert shuffled_starts != ordered_starts  # deterministic under seed 0


# --------------------------------------------------------------------------- #
# price reconstruction
# --------------------------------------------------------------------------- #


def test_get_starting_close_uses_last_close_at_or_before_start(ohlcv_df):
    streamer = StockStreamer(ohlc_df=ohlcv_df, seq_len=20, offset=5)
    window = ohlcv_df.iloc[50:60]

    assert streamer.get_starting_close(window) == ohlcv_df["close"].iloc[50]


def test_get_ohlcs_reconstructs_absolute_candles(ohlcv_df):
    streamer = StockStreamer(ohlc_df=ohlcv_df, seq_len=20, offset=5)

    start_date = ohlcv_df.index[30]
    idx = pd.date_range(start_date, periods=5, freq="D")
    df = pd.DataFrame(
        {
            "trade_occured": [True, True, False, True, True],
            "target_r_close": [0.00, 0.01, 0.0, -0.02, 0.03],
            "predicted_r_close": [0.00, 0.02, 0.0, -0.01, 0.04],
            "target_upside": [1.01, 1.02, 1.0, 1.03, 1.04],
            "predicted_upside": [1.05, 1.06, 1.0, 1.07, 1.08],
            "target_downside": [0.99, 0.98, 1.0, 0.97, 0.96],
            "predicted_downside": [0.95, 0.94, 1.0, 0.93, 0.92],
        },
        index=idx,
    )

    out = streamer.get_ohlcs(df)

    start_close = ohlcv_df["close"].iloc[30]
    kept = df[df["trade_occured"].to_numpy()]
    exp_tclose = start_close * np.exp(kept["target_r_close"].cumsum())
    exp_pclose = start_close * np.exp(kept["predicted_r_close"].cumsum())

    assert len(out) == int(df["trade_occured"].sum())
    np.testing.assert_allclose(out["target_close"], exp_tclose)
    np.testing.assert_allclose(out["predicted_close"], exp_pclose)
    np.testing.assert_allclose(out["target_open"], exp_tclose.shift(1))
    np.testing.assert_allclose(out["target_high"], exp_tclose * kept["target_upside"])
    np.testing.assert_allclose(out["target_low"], exp_tclose * kept["target_downside"])
    np.testing.assert_allclose(out["predicted_high"], exp_pclose * kept["predicted_upside"])
    np.testing.assert_allclose(out["predicted_low"], exp_pclose * kept["predicted_downside"])


def test_streamer_applies_split_in_post_init(ohlcv_df, stock_split):
    streamer = StockStreamer(ohlc_df=ohlcv_df.copy(), seq_len=20, offset=5, stock_split=stock_split)

    before = pd.Timestamp("2020-02-03")  # a business day before the 2020-03-01 split
    np.testing.assert_allclose(
        streamer.ohlc_df.loc[before, "close"], ohlcv_df.loc[before, "close"] / 2.0
    )
    np.testing.assert_allclose(
        streamer.ohlc_df.loc[before, "volume"], ohlcv_df.loc[before, "volume"] * 2.0
    )


def test_extract_features_window_feeds_get_starting_close(ohlcv_df):
    # sanity: a real preprocessed window's start date resolves to a raw close
    streamer = StockStreamer(ohlc_df=ohlcv_df, seq_len=20, offset=5)
    window = streamer[0]
    assert isinstance(streamer.get_starting_close(window), float)
    # extract_features round-trips without raising on this frame
    assert not extract_features(ohlcv_df).empty
