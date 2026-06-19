"""Tests for ``StockStreamerDataset`` and ``StockHandlerDataset``."""

import numpy as np
import pytest
import torch

from ophir.ticker import (
    StockHandlerDataset,
    StockHanlder,
    StockStreamer,
    StockStreamerDataset,
)

MODEL_KEYS = {"feature_input", "targets", "trade_occured", "response_size"}


# --------------------------------------------------------------------------- #
# StockStreamerDataset
# --------------------------------------------------------------------------- #


@pytest.fixture
def two_streamers(make_ohlcv):
    a = StockStreamer(ohlc_df=make_ohlcv(n_days=120, seed=1), seq_len=20, offset=20)
    b = StockStreamer(ohlc_df=make_ohlcv(n_days=140, seed=2), seq_len=20, offset=20)
    assert a.size >= 1 and b.size >= 1
    return a, b


def test_streamer_dataset_lengths_and_len(two_streamers):
    a, b = two_streamers
    ds = StockStreamerDataset([a, b], response_size=5)

    np.testing.assert_array_equal(ds.lengths, np.cumsum([a.size, b.size]))
    assert len(ds) == a.size + b.size


def test_streamer_dataset_index_to_streamer_boundary(two_streamers):
    a, b = two_streamers
    ds = StockStreamerDataset([a, b], response_size=5)
    lengths = ds.lengths

    # mirrors the source's index math, pinning the cumulative-length boundary
    assert int(np.argwhere(lengths > (lengths[0] - 1))[0].squeeze()) == 0
    assert int(np.argwhere(lengths > lengths[0])[0].squeeze()) == 1


def test_streamer_dataset_getitem_payload_and_double_wrap(two_streamers):
    a, b = two_streamers
    ds = StockStreamerDataset([a, b], response_size=5)

    item = ds[0]
    assert set(item) == MODEL_KEYS
    # self.response_size is np.array([5]); extract_model_data wraps again -> (1,1)
    assert item["response_size"].shape == (1, 1)


def test_streamer_dataset_restarts_exhausted_iterator(make_ohlcv):
    s = StockStreamer(ohlc_df=make_ohlcv(n_days=120, seed=3), seq_len=20, offset=20)
    ds = StockStreamerDataset([s], response_size=5)

    # access well past the streamer's size: the StopIteration is swallowed and
    # the iterator transparently restarted, so every call yields a payload.
    for _ in range(2 * s.size + 1):
        assert set(ds[0]) == MODEL_KEYS


def test_streamer_dataset_out_of_range_raises_indexerror(two_streamers):
    a, b = two_streamers
    ds = StockStreamerDataset([a, b], response_size=5)

    with pytest.raises(IndexError):
        _ = ds[len(ds)]


def test_streamer_dataset_return_date_includes_time(two_streamers):
    a, _ = two_streamers  # one streamer is enough; seq_len=20
    ds = StockStreamerDataset([a], response_size=5, return_date=True)

    item = ds[0]
    assert set(item) == MODEL_KEYS | {"time"}
    assert item["time"].dtype == np.dtype("datetime64[ns]")
    assert len(item["time"]) == a.seq_len


# --------------------------------------------------------------------------- #
# StockHandlerDataset
# --------------------------------------------------------------------------- #


def _streamer_handler(base_path, **kwargs):
    defaults = {
        "seq_len": 5,
        "base_path": base_path,
        "return_stock_id": False,
        "return_streamer": True,
    }
    defaults.update(kwargs)
    return StockHanlder(**defaults)


def test_handler_dataset_default_cache_size_matches_training(parquet_dir):
    # Direct instantiation must not silently use the autocorrelated cache=1;
    # the default mirrors the training default (run_training cache_size=8) so
    # batches mix across stocks.
    base_path, _ = parquet_dir
    ds = StockHandlerDataset(_streamer_handler(base_path), response_size=5)
    assert ds.cache_size == 8


def test_handler_dataset_init_prints_offset_and_cache(parquet_dir, capsys):
    base_path, _ = parquet_dir
    handler = _streamer_handler(base_path)

    StockHandlerDataset(handler, response_size=5, cache_size=3)

    out = capsys.readouterr().out
    assert f"offset: {handler.offset}" in out
    assert "cache: 3" in out


def test_handler_dataset_single_process_conserves_count(parquet_dir, mocker):
    base_path, _ = parquet_dir
    handler = _streamer_handler(base_path)
    mocker.patch("ophir.ticker.get_worker_info", return_value=None)

    expected_total = sum(handler[i].size for i in range(len(handler)))
    ds = StockHandlerDataset(handler, response_size=5, cache_size=1)

    items = list(ds)
    assert len(items) == expected_total
    assert all(set(it) == MODEL_KEYS for it in items)


def test_handler_dataset_cache_size_does_not_change_count(parquet_dir, mocker):
    base_path, _ = parquet_dir
    handler = _streamer_handler(base_path)
    mocker.patch("ophir.ticker.get_worker_info", return_value=None)

    expected_total = sum(handler[i].size for i in range(len(handler)))
    ds = StockHandlerDataset(handler, response_size=5, cache_size=2)

    assert len(list(ds)) == expected_total


def test_handler_dataset_worker_sharding(parquet_dir, mocker):
    base_path, _ = parquet_dir
    handler = _streamer_handler(base_path)
    mocker.patch(
        "ophir.ticker.get_worker_info",
        return_value=mocker.Mock(id=1, num_workers=2),
    )
    spy = mocker.spy(StockHanlder, "__getitem__")

    ds = StockHandlerDataset(handler, response_size=5, cache_size=1)
    list(ds)

    accessed = {int(c.args[1]) for c in spy.call_args_list}
    assert accessed == set(np.arange(1, len(handler), 2).tolist())


def test_handler_dataset_empty_handler_yields_nothing(parquet_dir, mocker, capsys):
    base_path, _ = parquet_dir
    handler = _streamer_handler(base_path)
    handler.keep_stocks([])
    mocker.patch("ophir.ticker.get_worker_info", return_value=None)

    ds = StockHandlerDataset(handler, response_size=5)
    capsys.readouterr()  # drop init + keep_stocks prints

    assert list(ds) == []


def test_handler_dataset_cache_size_larger_than_handler(parquet_dir, mocker):
    # cache_size exceeds the number of stocks: the loop still terminates and
    # yields every window exactly once (cache just stops growing once all
    # streamers are loaded).
    base_path, _ = parquet_dir
    handler = _streamer_handler(base_path)
    mocker.patch("ophir.ticker.get_worker_info", return_value=None)

    expected_total = sum(handler[i].size for i in range(len(handler)))
    ds = StockHandlerDataset(handler, response_size=5, cache_size=100)

    items = list(ds)
    assert len(items) == expected_total
    assert all(set(it) == MODEL_KEYS for it in items)


def test_handler_dataset_supports_multiple_passes(parquet_dir, mocker):
    # Iterating twice should both produce a full pass (each __iter__ rebuilds
    # local state and asks the handler for fresh streamers).
    base_path, _ = parquet_dir
    handler = _streamer_handler(base_path)
    mocker.patch("ophir.ticker.get_worker_info", return_value=None)

    expected_total = sum(handler[i].size for i in range(len(handler)))
    ds = StockHandlerDataset(handler, response_size=5)

    pass_1 = list(ds)
    pass_2 = list(ds)
    assert len(pass_1) == len(pass_2) == expected_total


def test_handler_dataset_yields_identity_when_enabled(parquet_dir, mocker):
    base_path, _ = parquet_dir
    mocker.patch("numpy.random.randint", return_value=0)
    handler = _streamer_handler(base_path, seq_len=20, offset=20)
    ds = StockHandlerDataset(handler, response_size=5, cache_size=1, return_identity=True)
    payload = next(iter(ds))

    assert "stock_id" in payload and "date_ordinal" in payload
    assert payload["stock_id"].dtype == torch.long
    # stock_id is a valid index into the handler's stock list.
    assert 0 <= int(payload["stock_id"]) < len(handler)


def test_handler_dataset_single_stock_cache_one(parquet_dir, mocker, capsys):
    # Degenerate path: 1 stock, cache_size=1 -- one streamer fills the cache,
    # gets exhausted, pops to empty, loop exits cleanly.
    base_path, _ = parquet_dir
    handler = _streamer_handler(base_path)
    handler.keep_stocks(["AAA"])
    mocker.patch("ophir.ticker.get_worker_info", return_value=None)

    expected = handler[0].size
    ds = StockHandlerDataset(handler, response_size=5, cache_size=1)
    capsys.readouterr()  # drop init + keep_stocks prints

    items = list(ds)
    assert len(items) == expected
    assert all(set(it) == MODEL_KEYS for it in items)
