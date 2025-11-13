from __future__ import annotations

import json
import os
from abc import abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller
from torch.utils.data import DataLoader, Dataset


class Modifier:
    _is_fit: Optional[bool] = False
    _validate_reverse: bool = False
    _is_dynamic: bool = False

    def __init__(self, validate_reverse: bool = False):
        self._validate_reverse = validate_reverse

    def __call__(self, original_input: np.ndarray) -> np.ndarray:
        if not self._is_fit:
            self.fit(original_input)

        return self._apply(original_input)

    def _maybe_validate_reverse(self, reverse: np.ndarray) -> np.ndarray:
        if not self._validate_reverse:
            return reverse

        orig_part = reverse[: self.orig_size]
        x0_same = np.isclose(orig_part[0], self.x0)
        x1_same = np.isclose(orig_part[1], self.x1)
        xt_same = np.isclose(orig_part[-1], self.xt)
        mean_same = np.isclose(orig_part.mean(), self.mean)
        var_same = np.isclose(orig_part.var(), self.var)

        if x0_same and x1_same and xt_same and mean_same and var_same:
            return reverse

        raise ValueError(
            f"{type(self)}: reverse() returned invalid result. Most likely due to input being split after fit() was called"
        )

    def fit(self, original_input: np.ndarray) -> Modifier:
        self.x0 = original_input[0]
        self.x1 = original_input[1]
        self.xt = original_input[-1]
        self.mean = original_input.mean()
        self.var = original_input.var()
        self.orig_size = len(original_input)
        self._is_fit = True
        return self

    @abstractmethod
    def _apply(self, original_input: np.ndarray) -> np.ndarray:
        pass

    def apply(self, original_input: np.ndarray) -> np.ndarray:
        return self(original_input)

    @abstractmethod
    def _reverse(self, modified_input: np.ndarray):
        pass

    def reverse(self, modified_input: np.ndarray) -> np.ndarray:
        reverse = self._reverse(modified_input)
        return self._maybe_validate_reverse(reverse)


class Diff(Modifier):
    def _apply(self, original_input: np.ndarray) -> np.ndarray:
        return original_input[1:] - original_input[:-1]

    def _reverse(self, modified_input: np.ndarray) -> np.ndarray:
        """Recover original modified_input upto a consant. I.e. reverse(modified_input) == reverse(modified_input + c)"""
        prev_x = modified_input[0]
        for i in range(1, len(modified_input), 1):
            modified_input[i] += prev_x
            prev_x = modified_input[i]
        return modified_input


class PercentageChange(Modifier):
    def _apply(self, original_input: np.ndarray) -> np.ndarray:
        change = (original_input[1:] - original_input[:-1]) / original_input[:-1]
        return change * 100


class Log(Modifier):
    def _apply(self, input: np.ndarray) -> np.ndarray:
        return np.where(~np.isclose(input, 0), np.log(input), 0)

    def _reverse(self, reverse: np.ndarray) -> np.ndarray:
        return np.exp(reverse)


class Exp(Modifier):
    def _apply(self, input: np.ndarray) -> np.ndarray:
        return np.exp(input)

    def _reverse(self, reverse: np.ndarray) -> np.ndarray:
        return np.log(reverse)


class Split(Modifier):
    def __init__(self, split_size):
        super().__init__()
        self._validate_reverse = False
        self.split_size = split_size

    def _apply(self, original_input):
        return (original_input[: self.split_size], original_input[self.split_size :])


class WindowSum(Modifier):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
        self._validate_reverse = False

    def _apply(self, original_input: np.ndarray) -> np.ndarray:
        if original_input.shape[0] % self.window_size > 0:
            raise ValueError("AvgWindow: input must be evenly divisible by window_size!")
        return original_input.reshape(-1, self.window_size).sum(axis=-1)

    def _reverse(self, modified_input):
        output = modified_input
        return output


class Trim(Modifier):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
        self._validate_reverse = False

    def _apply(self, original_input: np.ndarray) -> np.ndarray:
        blocks = int(original_input.shape[0] / self.window_size)
        return original_input[: blocks * self.window_size]

    def _reverse(self, modified_input):
        output = modified_input
        return output


class WindowMean(Modifier):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
        self._validate_reverse = False

    def _apply(self, original_input: np.ndarray) -> np.ndarray:
        if original_input.shape[0] % self.window_size > 0:
            raise ValueError("AvgWindow: input must be evenly divisible by window_size!")
        return original_input.reshape(-1, self.window_size).mean(axis=-1)

    def _reverse(self, modified_input):
        output = modified_input
        return output


class Mean(Modifier):
    global_mean: float

    def _apply(self, original_input):
        return original_input - self.global_mean

    def _reverse(self, modified_input):
        return modified_input + self.global_mean

    def fit(self, input: np.ndarray):
        print("fitting mean")
        self.global_mean = input.mean()
        self._is_fit = True


class MinMax(Modifier):
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max

    def _apply(self, original_input):
        return (original_input - self.min) / self.max

    def _reverse(self, modified_input):
        return modified_input * self.max + self.min


class Floor(Modifier):
    def _apply(self, original_input):
        return np.floor(original_input)


class Scale(Modifier):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def _apply(self, original_input):
        return original_input * self.factor


class Clip(Modifier):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def _apply(self, original_input):
        modified_output = np.abs(original_input)
        modified_output = np.where(modified_output <= self.size, modified_output, self.size)
        return modified_output * np.sign(original_input)


class ZNorm(Modifier):
    global_mean: float
    global_std: float

    def _apply(self, original_input):
        return (original_input - self.global_mean) / self.global_std

    def _reverse(self, modified_input):
        return modified_input * self.global_std + self.global_mean

    def fit(self, input: np.ndarray):
        print("fitting znorm")
        self.global_mean = input.mean()
        self.global_std = input.std()
        self._is_fit = True


class Identity(Modifier):
    def _apply(self, original_input: np.ndarray) -> np.ndarray:
        return original_input

    def _reverse(self, modified_input: np.ndarray) -> np.ndarray:
        return modified_input


class NormPDF(Modifier):
    def __init__(self, min, max, num_buckets, std=1):
        super().__init__()
        self.min = min
        self.max = max
        self.std = std
        self.num_buckets = num_buckets
        self.x_values = np.linspace(self.min, self.max, num=self.num_buckets, endpoint=True)

    def _apply(self, original_input: np.array) -> Tuple[np.ndarray, np.ndarray]:
        mod_input = []
        for i in original_input:
            mod_input.append(norm.pdf(self.x_values, loc=i, scale=self.std))
        probs = np.vstack(mod_input)
        return self.x_values[probs.argmax(-1)]


class Tokenize(Modifier):
    def __init__(self, min, max, num_buckets, std=1):
        super().__init__()
        self.min = min
        self.max = max
        self.std = std
        self.num_buckets = num_buckets
        self.x_values = np.linspace(self.min, self.max, num=self.num_buckets, endpoint=True)

    def _apply(self, original_input: np.array) -> Tuple[np.ndarray, np.ndarray]:
        mod_input = []
        for i in original_input:
            mod_input.append(norm.pdf(self.x_values, loc=i, scale=self.std))
        probs = np.vstack(mod_input)
        return probs.argmax(-1)

    def _reverse(self, modified_input):
        return self.x_values[modified_input.argmax(-1)]


class Lambda(Modifier):
    def __init__(self, apply_func, reverse_func):
        super().__init__()
        self._validate_reverse = False
        self.apply_func = apply_func
        self.reverse_func = reverse_func

    def _apply(self, original_input: np.ndarray) -> np.ndarray:
        return self.apply_func(original_input)

    def _reverse(self, modified_input: np.ndarray) -> np.ndarray:
        return self.reverse_func(modified_input)


class LogTransform(Modifier):
    _global_mean: Optional[float] = None

    def __init__(self):
        super().__init__()
        self._validate_reverse = False

    @property
    def global_mean(self):
        return self._global_mean

    @global_mean.setter
    def global_mean(self, mean):
        if self.global_mean is not None:
            raise ValueError("Attempting to override global mean.")
        self._global_mean = mean

    def _apply(self, original_input):
        original_input = np.log(original_input)

        if self.global_mean is None:
            self.global_mean = original_input.mean()
        else:
            print("Calling Log Transform multiple times.")
        return original_input - self.global_mean

    def _reverse(self, modified_input):
        return np.exp(modified_input + self.global_mean)


class ModifierPipeline:
    def __init__(self, modifiers: Optional[List[Modifier]] = None):
        self.modifiers = modifiers if modifiers is not None else [Identity()]
        self.dynamic_modifiers = []

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        for layer in self.modifiers:
            x = layer(x)
        return x

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self(x)

    def reverse(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        for layer in self.modifiers[::-1]:
            x = layer.reverse(x)
        return x

    def extend_pipeline(self, pipeline: ModifierPipeline) -> ModifierPipeline:
        modifiers = [mod for mod in self.modifiers]
        for mod in pipeline.modifiers:
            modifiers.append(mod)
        return ModifierPipeline(modifiers)


class SingleCoinPriceDataset(Dataset):
    def __init__(
        self,
        preprocess_data: np.ndarray,
        elements_per_sample: int,
        elements_in_encoder: int,
        points_per_day: int,
        preprocess_pipeline: Optional[ModifierPipeline] = None,
        enc_mod_pipeline: Optional[ModifierPipeline] = None,
        dec_mod_pipeline: Optional[ModifierPipeline] = None,
        return_sample_index: bool = False,
    ):
        self.preprocess_data = preprocess_data
        self.preprocess_pipeline = preprocess_pipeline
        self.enc_mod_pipeline = enc_mod_pipeline
        self.dec_mod_pipeline = dec_mod_pipeline
        self.points_per_day = points_per_day
        self.elements_per_sample = (elements_per_sample + 1) * self.points_per_day
        self.elements_in_encoder = elements_in_encoder * self.points_per_day

        self.return_sample_index = return_sample_index
        self.size = self.preprocess_data.shape[0] - self.elements_per_sample + 1

    def get_full_dataset(self, include_encoder_pipeline: bool = True) -> np.ndarray:
        if include_encoder_pipeline:
            return self.enc_mod_pipeline(self.preprocess_data.copy())
        return self.preprocess_data.copy()

    def reverse_preprocess(
        self, x: np.ndarray, include_encoder_pipeline: bool = True
    ) -> np.ndarray:
        if include_encoder_pipeline:
            x = self.enc_mod_pipeline.reverse(x)
        return self.preprocess_pipeline.reverse(x)

    def check_stationary(self, sample_ratio: float, include_encoder_pipeline: bool = True) -> float:
        samples_to_take = int(self.preprocess_data.shape[0] * sample_ratio)
        sample_indices = np.random.choice(
            np.arange(self.preprocess_data.shape[0], dtype=np.int64),
            samples_to_take,
            replace=False,
        )
        sample_indices.sort()

        samples = self.preprocess_data[sample_indices]
        if include_encoder_pipeline:
            samples = self.enc_mod_pipeline(samples)
        return adfuller(samples)[1]

    def __len__(self):
        return self.size

    def __getitem__(self, sample):
        start_index = sample
        end_index = start_index + self.elements_per_sample
        elements = (
            self.preprocess_data[start_index:end_index]
            .copy()
            .reshape(-1, self.points_per_day)
            .mean(-1)
        )
        (encoder_input, decoder_input) = self.enc_mod_pipeline(elements)

        encoder_input = np.expand_dims(encoder_input, axis=-1).astype(np.float32)
        decoder_input = np.expand_dims(decoder_input, axis=-1).astype(np.float32)

        if self.dec_mod_pipeline is not None:
            decoder_input, decoder_output = self.dec_mod_pipeline(decoder_input)
        else:
            decoder_output = decoder_input.copy()
        if not self.return_sample_index:
            return (
                (torch.from_numpy(encoder_input), torch.from_numpy(decoder_input[:-1])),
                torch.from_numpy(np.concat([encoder_input, decoder_output], axis=0)),
            )

        return (encoder_input, decoder_input), (start_index, end_index)


class MultiPriceDataset(Dataset):
    def __init__(
        self,
        preprocess_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        tokens: List[np.ndarray],
        elements_per_sample: int,
        points_per_day: int,
        enc_mod_pipeline: Optional[ModifierPipeline] = None,
    ):
        self.preprocess_data = preprocess_data
        self.tokens = tokens
        self.enc_mod_pipeline = enc_mod_pipeline
        self.points_per_day = points_per_day
        self.elements_per_sample = (elements_per_sample + 1) * self.points_per_day

        self.samples_per_data = np.array(
            [len(price) - self.elements_per_sample + 1 for (_, _, _, price) in self.preprocess_data]
        )
        self.size = self.samples_per_data.sum()

    def __len__(self):
        return self.size

    def __getitem__(self, sample_idx):
        sample = sample_idx + 1
        data_index = 0
        while self.samples_per_data[: data_index + 1].sum() < sample:
            data_index += 1
        sample_offset = sample - self.samples_per_data[:data_index].sum()
        start_index = sample_offset - 1
        end_index = start_index + self.elements_per_sample

        (year, month, day, price) = self.preprocess_data[data_index]
        year, month, day = (
            year[start_index : end_index : self.points_per_day],
            month[start_index : end_index : self.points_per_day],
            day[start_index : end_index : self.points_per_day],
        )
        price = price[start_index:end_index]
        elements = price.reshape(-1, self.points_per_day).mean(-1)
        (encoder_price, decoder_price) = self.enc_mod_pipeline(elements)
        sector_token, stock_token = self.tokens[data_index]

        encoder_price = np.expand_dims(encoder_price, axis=-1).astype(np.float32)
        decoder_price = np.expand_dims(decoder_price, axis=-1).astype(np.float32)
        sector_token = np.array([sector_token]).astype(np.long)
        stock_token = np.array([stock_token]).astype(np.long)

        return (
            (
                torch.from_numpy(encoder_price),
                torch.from_numpy(decoder_price),
                torch.from_numpy(year[: len(elements) - 1].astype(np.long)),
                torch.from_numpy(month[: len(elements) - 1].astype(np.long) - 1),
                torch.from_numpy(day[: len(elements) - 1].astype(np.long) - 1),
                torch.from_numpy(sector_token),
                torch.from_numpy(stock_token),
            ),
            torch.from_numpy(np.concat([encoder_price, decoder_price], axis=0)),
        )


def construct_datasets(
    dir_path: str,
    sub_unit: str,
    min_value: int,
    max_value: int,
    elements_per_sample: int,
    elements_in_encoder: int,
    quicktest: bool = False,
):
    points_per_day = 1  # 24 if sub_unit == "h" else 24 * 60
    sector_token_path = os.path.join(dir_path, "sector_tokens.json")
    stock_token_path = os.path.join(dir_path, "stock_tokens.json")

    sector_tokens = {}
    stock_tokens = {}

    if os.path.exists(sector_token_path):
        print("loading sector tokens...")
        with open(sector_token_path, "r") as f:
            sector_tokens = json.load(f)

    if os.path.exists(stock_token_path):
        print("loading stock tokens...")
        with open(stock_token_path, "r") as f:
            stock_tokens = json.load(f)

    def add_token(tokens, item):
        if item not in tokens:
            token = len(tokens)
            tokens[item] = token

    def tokenize_file(fn):
        sector_stock = fn.split(".parquet")[0].split("-")
        sector = "_".join(sector_stock[:-1])
        stock = sector_stock[-1]
        add_token(sector_tokens, sector), add_token(stock_tokens, stock)
        return sector_tokens[sector], stock_tokens[stock]

    def create_time_blocks(df: pd.DataFrame, time_col: str, val_col: str, unit: str) -> pd.Series:
        out_df = df.copy()
        out_df["time_block"] = df[time_col].dt.floor(unit)
        out_df = out_df.groupby("time_block").mean()
        return out_df[val_col]

    def preprocess_price_data(dir_path, fn):
        fp = os.path.join(dir_path, fn)
        tokens = tokenize_file(fn)

        price_df = pd.read_parquet(fp)  # , columns=["timestamp", "close"])
        # price_df["Date"] = pd.DatetimeIndex(price_df["timestamp"])
        price_df = price_df[["Date", "close"]]
        dataset_series = price_df.set_index("Date", inplace=False).sort_index().reset_index()
        if sub_unit == "h":
            time_block_price = create_time_blocks(dataset_series, "Date", "close", "h")
        else:
            time_block_price = dataset_series[["Date", "close"]].set_index("Date")

        year = time_block_price.index.year.to_numpy() - 2016
        month = time_block_price.index.month.to_numpy()
        day = time_block_price.index.day.to_numpy()

        return tokens, (
            year,
            month,
            day,
            time_block_price.to_numpy().squeeze(),
        )

    files = os.listdir(dir_path)
    if quicktest:
        files = files[:1]
    preprocess_data = [
        preprocess_price_data(dir_path, fp) for i, fp in enumerate(files) if ".parquet" in fp
    ]
    tokens, price_data = (
        [data[0] for data in preprocess_data],
        [data[1] for data in preprocess_data],
    )
    with open(sector_token_path, "w") as f:
        json.dump(sector_tokens, f, indent=4)

    with open(stock_token_path, "w") as f:
        json.dump(stock_tokens, f, indent=4)

    train_ratio = 0.8
    train_size = [int(train_ratio * len(price_data)) for (_, _, _, price_data) in price_data]
    train_data = [
        (year[:size], month[:size], day[:size], price[:size])
        for (year, month, day, price), size in zip(price_data, train_size, strict=False)
    ]
    test_data = [
        (year[size:], month[size:], day[size:], price[size:])
        for (year, month, day, price), size in zip(price_data, train_size, strict=False)
    ]

    encoder_pipeline = ModifierPipeline(
        [PercentageChange(), Clip(max_value), Split(elements_in_encoder)]
    )

    train_dataset = MultiPriceDataset(
        preprocess_data=train_data,
        tokens=tokens,
        elements_per_sample=elements_per_sample,
        enc_mod_pipeline=encoder_pipeline,
        points_per_day=points_per_day,
    )

    test_dataset = MultiPriceDataset(
        preprocess_data=test_data,
        tokens=tokens,
        elements_per_sample=elements_per_sample,
        enc_mod_pipeline=encoder_pipeline,
        points_per_day=points_per_day,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    return train_loader, test_loader, sector_tokens, stock_tokens
