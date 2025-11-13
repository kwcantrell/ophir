import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import torch
import typer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing_extensions import Annotated

from .coin_datasets import Clip, ModifierPipeline, PercentageChange, Split
from .register import DATA_DIR, get_massive_client


class Ticker:
    def load_tickers(symbol):
        parquets = os.listdir(DATA_DIR)
        tickers = []
        for parquet in parquets:
            if symbol in parquet:
                sic_code = parquet.split("-")[0]
                df = pd.read_parquet(os.path.join(DATA_DIR, parquet))
                tickers.append(Ticker(symbol, sic_code=sic_code, df=df))
        return tickers

    def __init__(self, symbol, sic_code=None, df=None):
        self.symbol = symbol
        self.sic_code = sic_code
        self.df = df
        self._fetch_aggregate_bars()

    def __len__(self):
        return self.df.shape[0]

    def day_periods(start_i, num_days):
        pass

    def _construct_path_name(self):
        self.start_date = self.df["Date"].min()
        self.end_date = self.df["Date"].max()
        self.data_path = os.path.join(DATA_DIR, f"{self.sic_code}-{self.symbol}.parquet")

    def _fetch_aggregate_bars(self):
        client = get_massive_client()
        details = client.get_ticker_details(self.symbol)
        self.sic_code = details.sic_code
        self.sic_description = details.sic_description

        if self.df is None:
            aggs = []
            for a in client.list_aggs(
                self.symbol, 1, "minute", "2020-09-10", "2025-11-11", limit=50000
            ):
                aggs.append(a)

            self.df = pd.DataFrame([agg.__dict__ for agg in aggs])
            self.df["Date"] = pd.to_datetime(
                self.df["timestamp"], utc=True, unit="ms"
            ).dt.tz_convert("America/New_York")
            self.df["market_hour"] = self.df["Date"].apply(self.market_hour)
        self._construct_path_name()

    def save_parquet(self):
        self.df.to_parquet(self.data_path)

    def market_hour(self, date):
        if date.hour >= 4 and (date.hour <= 9 and date.minute < 30):
            return "pre-market"
        elif (date.hour >= 9 and date.minute < 30) and (date.hour <= 4):
            return "regular-market"
        return "after-market"


class DayAgg(Dataset):
    def __init__(
        self,
        day_info: np.ndarray,
        ticker_info: np.ndarray,
        common_tickers: np.ndarray,
        ticker_to_industry,
        elements_per_sample: int,
        enc_mod_pipeline: Optional[ModifierPipeline] = None,
        iterations_per_epoch: int = 10000,
    ):
        self.day_info = day_info
        self.ticker_info = ticker_info
        self.common_tickers = common_tickers
        self.common_ticker_tokens = np.arange(len(self.common_tickers), dtype=np.long)
        self.ticker_to_industry = ticker_to_industry
        self.elements_per_sample = elements_per_sample + 1
        self.enc_mod_pipeline = enc_mod_pipeline
        self.iterations_per_epoch = iterations_per_epoch
        self.last_start_day = self.day_info.shape[0] - elements_per_sample - 1

    def __len__(self):
        return self.iterations_per_epoch

    def __getitem__(self, sample_idx):
        start_day = np.random.randint(self.last_start_day)
        day_info = self.day_info[start_day : start_day + self.elements_per_sample][1:]

        stock_idx = np.random.randint(self.ticker_info.shape[1])
        stock_token = self.common_ticker_tokens[stock_idx]

        ticker = self.common_tickers[stock_idx]
        if ticker in self.ticker_to_industry:
            sector_token, industry_token = self.ticker_to_industry[ticker]
        else:
            sector_token, industry_token = (0, 0)
        day_enc_tickers, day_dec_tickers = self.enc_mod_pipeline(
            self.ticker_info[start_day : start_day + self.elements_per_sample, stock_idx]
        )

        day_enc_tickers = np.expand_dims(day_enc_tickers, axis=-1).astype(np.float32)
        day_dec_tickers = np.expand_dims(day_dec_tickers, axis=-1).astype(np.float32)
        sector_token = np.array([sector_token]).astype(np.long)
        industry_token = np.array([industry_token]).astype(np.long)
        stock_token = np.array([stock_token]).astype(np.long)
        return (
            (
                torch.from_numpy(day_enc_tickers),
                torch.from_numpy(day_dec_tickers),
                torch.from_numpy(day_info[:, 0].astype(np.long) - 2016),
                torch.from_numpy(day_info[:, 1].astype(np.long) - 1),
                torch.from_numpy(day_info[:, 2].astype(np.long) - 1),
                torch.from_numpy(sector_token),
                torch.from_numpy(industry_token),
                torch.from_numpy(stock_token),
            ),
            torch.from_numpy(np.concat([day_enc_tickers, day_dec_tickers], axis=0)),
        )


def construct_day_agg_dataset():
    def year_month_day(fp):
        return tuple(int(i) for i in fp.split(".")[0].split("-"))

    dir_path = "/media/kalen/Seagate Portable Drive/day-aggregates"
    day_aggs = sorted(
        [day_agg for day_agg in os.listdir(dir_path) if int(day_agg.split("-")[0]) > 2016]
    )
    day_info = np.vstack([year_month_day(fp) for fp in day_aggs], dtype=np.long)
    print("loading tickers...")
    ticker_info = [
        pd.read_csv(os.path.join(dir_path, day_agg), compression="gzip", index_col=0)[
            "close"
        ].sort_index()
        for day_agg in day_aggs
    ]

    print("filtering tickers...")
    common_tickers = ticker_info[0].index
    for ticker in ticker_info[1:]:
        common_tickers = common_tickers.intersection(ticker.index)
    ticker_info = np.array(
        [ticker.loc[ticker.index.isin(common_tickers)] for ticker in ticker_info]
    )

    print("loading industry...")
    with open(os.path.join(DATA_DIR, "tick-industry.json"), "r") as f:
        tick_industry = json.load(f)

    industry_token = {}
    sector_token = {}
    tick_ind_sec_token = {}
    for tick, (industry, sector) in tick_industry.items():
        if industry not in industry_token:
            industry_token[industry] = len(industry_token) + 1

        if sector not in sector_token:
            sector_token[sector] = len(sector_token) + 1

        tick_ind_sec_token[tick] = (sector_token[sector], industry_token[industry])

    max_value = 10
    elements_in_encoder = 150
    encoder_pipeline = ModifierPipeline(
        [PercentageChange(), Clip(max_value), Split(elements_in_encoder)]
    )

    day_agg_dataset = DayAgg(
        day_info,
        ticker_info,
        common_tickers,
        tick_ind_sec_token,
        elements_per_sample=180,
        enc_mod_pipeline=encoder_pipeline,
        iterations_per_epoch=100000,
    )
    return DataLoader(
        day_agg_dataset, batch_size=128, num_workers=2, pin_memory=True, persistent_workers=True
    )


app = typer.Typer(no_args_is_help=True, chain=True)


@app.command()
def add_symbol(
    symbol: Annotated[List[str], typer.Argument(help="Ticker symbol to construct parquet.")] = [],
):
    if isinstance(symbol, list):
        for s in symbol:
            Ticker(s).save_parquet()
    else:
        Ticker(symbol).save_parquet()


@app.command()
def update_ticker_to_industry():
    tick_industry_path = os.path.join(DATA_DIR, "tick-industry.json")
    if os.path.exists(tick_industry_path):
        with open(tick_industry_path, "r") as f:
            tick_industry = json.load(f)
    else:
        tick_industry = {}

    fp = os.path.join(DATA_DIR, "industry.json")
    with open(fp, "r") as f:
        industries = json.load(f)

    for industry in tqdm(industries):
        print(f"processing {industry}")
        r = requests.get(
            f"https://financialmodelingprep.com/stable/company-screener?industry={industry}&apikey=REDACTED"
        )

        for ticker_info in r.json():
            tick_industry[ticker_info["symbol"]] = [ticker_info["industry"], ticker_info["sector"]]

    with open(tick_industry_path, "w") as f:
        json.dump(tick_industry, f, indent=4, sort_keys=True)
