import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info


def get_stock_parquets(base_path):
    stock_dirs = os.listdir(base_path)

    def parquet(path):
        for p in os.listdir(os.path.join(base_path, path)):
            if p.endswith(".parquet"):
                return os.path.join(path, p)

    stocks = {
        path.split("=")[-1]: os.path.join(base_path, parquet(path))
        for path in stock_dirs
        if "=" in path
    }
    return stocks


def get_starts(df, seq_len, offset):
    num_start = max(0, len(df) - seq_len)
    starts = np.arange(0, num_start, offset)
    return starts


def get_start_dates(df: pd.DataFrame, seq_len, offset):
    dates = df.index.to_series()
    calendar = pd.date_range(dates.min(), dates.max(), freq="D")
    starts = np.arange(0, len(calendar) - seq_len, offset)
    return calendar[starts].to_numpy()


def get_sp_500_symbols():
    # Wikipedia URL for S&P 500 companies
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    try:
        dfs = pd.read_html(url, storage_options={"User-Agent": headers["User-Agent"]})
    except Exception as e:
        print(f"An error occurred: {e}")

    return list(dfs[0]["Symbol"])


def get_splits(tickers: List[str], cache_path: str = None) -> Dict[str, "StockSplit"]:
    import pickle

    import yfinance as yf
    from tqdm import tqdm

    if cache_path is None:
        from .register import DATA_DIR
        cache_path = os.path.join(DATA_DIR, "yf_splits_cache.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached: Dict[str, StockSplit] = pickle.load(f)
        missing = [t for t in tickers if t not in cached]
        if not missing:
            return {t: cached[t] for t in tickers if t in cached}
    else:
        cached = {}
        missing = list(tickers)

    for ticker in tqdm(missing, desc="Fetching splits"):
        try:
            series = yf.Ticker(ticker).splits
        except Exception as e:
            print(f"[get_splits] {ticker} failed: {e}")
            continue
        if series is None or len(series) == 0:
            cached[ticker] = None  # sentinel: queried, no splits
            continue
        naive_index = series.index.tz_localize(None) if series.index.tz is not None else series.index
        cached[ticker] = StockSplit(
            id=ticker,
            dates=list(naive_index.to_numpy()),
            ratios=list(series.to_numpy().astype(float)),
        )

    with open(cache_path, "wb") as f:
        pickle.dump(cached, f)

    return {t: cached[t] for t in tickers if cached.get(t) is not None}


@dataclass
class StockSplit:
    id: str
    dates: List[np.datetime64]
    ratios: List[float]

    def apply_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        df: DataFrame with index = datetime, columns = open/high/low/close/volume
        splits: list of dicts with keys {"date", "ratio"}
        """
        df = df.sort_index()

        # Create cumulative adjustment factor
        adj_factor = pd.Series(1.0, index=df.index)

        for date, ratio in zip(self.dates, self.ratios):
            split_date = pd.to_datetime(date)

            # Apply to all dates BEFORE split date
            adj_factor.loc[adj_factor.index < split_date] /= ratio

        # Apply price adjustments
        price_cols = ["close"]
        df[price_cols] = df[price_cols].mul(adj_factor, axis=0)

        # Volume goes the opposite way
        if "volume" in df.columns:
            df["volume"] = df["volume"] / adj_factor

        return df


def extract_features(df: pd.DataFrame, winsorize_returns: bool = False) -> pd.DataFrame:
    feature_cols = []

    def add_feature(feature_col, feature_val, df: pd.DataFrame) -> pd.DataFrame:
        df = df.assign(**{feature_col: feature_val})
        feature_cols.append(feature_col)
        return df

    # time delta
    delta_days = df.index.to_series().diff().dt.days
    df = add_feature("time_delta", np.log(delta_days), df)

    # returns
    prev_close = df["close"].shift(1)
    df = add_feature("r_close", np.log(df["close"] / prev_close), df)

    # does not add new features just updates r_close columns
    if winsorize_returns:
        df["r_close"] = df["r_close"].clip(
            lower=df["r_close"].quantile(0.001), upper=df["r_close"].quantile(0.999)
        )

    def add_rolling_features(window_size, df):
        # normalized returns
        eps = 1e-8
        r_close = df["r_close"]
        r_close_rolling_std = r_close.rolling(window_size).std()
        df = add_feature(
            f"{window_size}_norm_returns", r_close / (r_close_rolling_std + eps), df
        )

        # volatility normalization
        log_volume: pd.DataFrame = np.log(df["volume"] + eps)
        mu = log_volume.rolling(window_size).mean()
        sigma = log_volume.rolling(window_size).std()
        norm_volume = (log_volume - mu) / (sigma + eps).clip(-5, 5)
        df = add_feature(f"{window_size}_norm_volume", norm_volume, df)
        df = add_feature(f"{window_size}_volatility", r_close_rolling_std, df)
        return df

    df = add_rolling_features(10, df)
    df = add_rolling_features(20, df)
    df = add_rolling_features(60, df)
    df = add_feature("upside", np.log(df["high"] / df["close"]), df)
    df = add_feature("downside", np.log(df["close"] / df["low"]), df)

    if df.empty:
        return df

    calendar = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df_pad = df.reindex(index=calendar)

    df_pad = add_feature("trade_occured", df_pad["close"].notna(), df_pad)

    PAD_VALUE = 0.0
    for col in feature_cols:
        df_pad[col] = df_pad[col].fillna(PAD_VALUE)

    return df_pad[feature_cols]


@dataclass(kw_only=True)
class StockStreamer:
    ohlc_df: pd.DataFrame
    seq_len: int
    offset: int
    stock_split: StockSplit = None
    shuffle: bool = False

    def __post_init__(self):
        if len(self.ohlc_df) < 1:
            self.starts = []
            self.iterator = iter(self.create_iterator())
            return

        # apply stock splits
        if self.stock_split is not None:
            self.ohlc_df = self.stock_split.apply_splits(self.ohlc_df)
        self.preprocessed_ohlc_df = extract_features(self.ohlc_df)
        self.iterator = iter(self.create_iterator())
        if self.offset == -1:
            self.offset = self.seq_len

        self.starts = get_starts(self.preprocessed_ohlc_df, self.seq_len, self.offset)

    @property
    def size(self):
        return len(self.starts)

    def get_starting_close(self, df: pd.DataFrame):
        date = df.index.min()
        date_mask = self.ohlc_df.index <= date
        return self.ohlc_df[date_mask].iloc[-1]["close"]

    def get_ohlcs(self, df: pd.DataFrame):
        """Reconstruct OHLC candle sticks for model output"""
        start_close = self.get_starting_close(df)

        # remove pad days
        trade_mask = df["trade_occured"].to_numpy().reshape(-1)
        df = df.loc[trade_mask]

        df = df.assign(
            **{
                "target_close": start_close * np.exp(df["target_r_close"].cumsum()),
                "predicted_close": start_close
                * np.exp(df["predicted_r_close"].cumsum()),
            }
        )

        df["target_open"] = df["target_close"].shift(1)
        df["predicted_open"] = df["predicted_close"].shift(1)

        df["target_high"] = df["target_close"] * df["target_upside"]
        df["predicted_high"] = df["predicted_close"] * df["predicted_upside"]

        df["target_low"] = df["target_close"] * df["target_downside"]
        df["predicted_low"] = df["predicted_close"] * df["predicted_downside"]

        return df

    def __len__(self):
        return self.size

    def __getitem__(self, i: int):
        return self.preprocessed_ohlc_df.iloc[i : i + self.seq_len]

    def create_iterator(self):
        starts = np.arange(len(self.starts))
        if self.shuffle:
            np.random.shuffle(starts)
        for start in starts:
            start = self.starts[start]
            yield self.preprocessed_ohlc_df.iloc[start : start + self.seq_len]

    def next(self):
        return next(self.iterator)


@dataclass(kw_only=True)
class StockHanlder:
    seq_len: int
    base_path: str
    return_stock_id: bool
    return_streamer: bool
    stock_splits: Dict[str, StockSplit] = None
    shuffle: bool = False
    offset: int = -1
    min_year: int = None
    max_year: int = None
    min_volume: float = None
    min_ticker_history: int = None
    winsorize_returns: bool = True

    def __post_init__(self):
        self.stock_dict = get_stock_parquets(self.base_path)
        self.stocks = list(self.stock_dict.keys())

        if self.offset == -1:
            self.offset = self.seq_len

    def __len__(self):
        return len(self.stocks)

    def __getitem__(self, index):
        if isinstance(index, str):
            index = self.stocks.index(index)
        stock = self.stocks[index]
        return self.stock(stock)

    def stock(self, stock):
        if self.return_streamer:
            return self.stock_streamer(stock)
        else:
            return self.stock_df(stock)

    def keep_stocks(self, stock_list):
        kept_stocks = {
            stock: self.stock_dict[stock]
            for stock in stock_list
            if stock in self.stock_dict
        }
        print(
            f"stocks kept: {len(kept_stocks)}/{len(self.stock_dict)}, stocks not found: {len(stock_list) - len(stock_list)}"
        )
        self.stock_dict = kept_stocks
        self.stocks = list(self.stock_dict.keys())

    def stock_df(self, stock) -> pd.DataFrame:
        path = self.stock_dict[stock]
        df = pd.read_parquet(path)

        if self.min_volume is not None and df["volume"].mean() < self.min_volume:
            return pd.DataFrame()

        if self.min_ticker_history is not None:
            days_spanned = (df["utc_time"].max() - df["utc_time"].min()).days
            if days_spanned < self.min_ticker_history:
                return pd.DataFrame()
        df["date"] = df["utc_time"].dt.normalize()
        df = (
            df.groupby("date")
            .agg({"high": "max", "low": "min", "close": "last", "volume": "sum"})
            .sort_index()
        )
        df = df.dropna(subset=["high", "low", "close"])

        if len(df) < 1:
            return df

        if self.min_year is not None:
            df = df.loc[df.index.to_series().dt.year >= self.min_year]

        if self.max_year is not None:
            df = df.loc[df.index.to_series().dt.year < self.max_year]

        return df

    def stock_streamer(self, stock: str) -> StockStreamer:
        stock_split = None
        if self.stock_splits is not None and stock in self.stock_splits:
            stock_split = self.stock_splits[stock]

        return StockStreamer(
            ohlc_df=self.stock_df(stock),
            seq_len=self.seq_len,
            offset=self.offset,
            shuffle=self.shuffle,
            stock_split=stock_split,
        )


def extract_model_data(df: pd.DataFrame, response_size: int, return_date: bool = False):
    features = [c for c, d in zip(df.columns, df.dtypes) if d != np.bool]
    feature_input = df[features].to_numpy()
    targets = df[["r_close", "upside", "downside"]].to_numpy()
    trade_occured = df["trade_occured"].to_numpy()
    model_data = {
        "feature_input": torch.from_numpy(feature_input).float(),
        "targets": torch.from_numpy(targets).float(),
        "trade_occured": torch.from_numpy(trade_occured),
        "response_size": torch.from_numpy(np.array([response_size])),
    }
    if return_date:
        model_data["time"] = df.index.to_numpy()
    return model_data


class StockStreamerDataset(Dataset):
    def __init__(
        self,
        stock_streamers: List[StockStreamer],
        response_size: int,
        return_date: bool = False,
    ):
        self.stock_streamers = stock_streamers
        self.iterators = [
            iter(streamer.create_iterator()) for streamer in self.stock_streamers
        ]
        self.lengths = np.array(
            np.cumsum([stremer.size for stremer in self.stock_streamers])
        )

        self.response_size = np.array([response_size])
        self.return_date = return_date

    def __len__(self):
        return self.lengths[-1]

    def __getitem__(self, index):
        index = np.argwhere(self.lengths > index)[0].squeeze()
        try:
            df = next(self.iterators[index])
        except StopIteration:
            self.iterators[index] = iter(self.stock_streamers[index].create_iterator())
            df = next(self.iterators[index])
        return extract_model_data(df, self.response_size, self.return_date)


class StockHandlerDataset(IterableDataset):
    def __init__(
        self, stock_hanlder: List[StockHanlder], response_size: int, cache_size: int = 1
    ):
        self.stock_hanlder = stock_hanlder
        self.response_size = np.array([response_size])
        self.cache_size = cache_size

        print(
            f"Creating StockHandlerDataset with offset: {self.stock_hanlder.offset} and cache: {self.cache_size}"
        )

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            # Single-process data loading
            start = 0
            step = 1
        else:
            # Multi-worker data loading
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            start = worker_id
            step = num_workers

        processed_stocks = 0
        cache = []
        cur_stock = 0
        shard_stock_indices = np.arange(start, len(self.stock_hanlder), step)
        while processed_stocks < len(shard_stock_indices):
            if len(cache) < self.cache_size and cur_stock < len(shard_stock_indices):
                stock_ind = shard_stock_indices[cur_stock]
                streamer = self.stock_hanlder[stock_ind]
                cache.append(streamer)
                cur_stock += 1

            cache_index = np.random.randint(len(cache))

            try:
                df = cache[cache_index].next()
                yield extract_model_data(df, self.response_size)
            except StopIteration:
                processed_stocks += 1
                cache.pop(cache_index)
