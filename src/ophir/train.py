"""Training entrypoints for the OHLC forecaster.

Wires the data pipeline (:mod:`ophir.ticker`) to the Lightning module
(:class:`~ophir.training_models.LightningOHLCPredictor`) and the trainer
factories in :mod:`ophir.register`, exposing the ``ophir train`` (base
pre-training) and ``ophir finetune`` commands.

The train/val split is **by date** with an embargo gap, per the contract in
``CLAUDE.md``: overlapping ``offset`` windows would otherwise straddle the
boundary, so the splits are disjoint year ranges separated by at least one
skipped year. Each window spans ``seq_len`` calendar days (the feature frame is
reindexed onto a daily calendar), so the embargo must skip
``ceil(seq_len / 365)`` years.

Requires a CUDA GPU and the per-stock parquet tree under
``.ophir/data/days/stocks``.
"""

from __future__ import annotations

import math
import os
import random
from typing import TYPE_CHECKING, Any

import typer

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from ophir.ticker import StockHanlder

#: Calendar days per year; windows are sliced from a daily-calendar frame, so
#: the embargo gap is measured in calendar (not trading) days.
_CALENDAR_DAYS_PER_YEAR = 365

#: Maximum sequence length supported by the model's positional encoding.
_MAX_SEQ_LEN = 512

#: Minimum per-head dimension; PyTorch flex-attention cannot compile a smaller
#: head dim on CUDA ("embedding dimension ... must be at least 16").
_MIN_HEAD_DIM = 16


def _validate_dims(emb_dim: int, num_heads: int, seq_len: int, response_size: int) -> None:
    """Validate model/window dimensions, raising ``typer.BadParameter``."""
    if emb_dim % 4 != 0:
        raise typer.BadParameter(f"emb_dim must be a multiple of 4, got {emb_dim}")
    if emb_dim % num_heads != 0:
        raise typer.BadParameter(
            f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads})"
        )
    if emb_dim // num_heads < _MIN_HEAD_DIM:
        raise typer.BadParameter(
            f"head dim (emb_dim // num_heads) must be >= {_MIN_HEAD_DIM} for "
            f"flex-attention; got {emb_dim // num_heads} "
            f"(emb_dim={emb_dim}, num_heads={num_heads})"
        )
    if not 0 < seq_len <= _MAX_SEQ_LEN:
        raise typer.BadParameter(f"seq_len must be in 1..{_MAX_SEQ_LEN}, got {seq_len}")
    if not 0 < response_size < seq_len:
        raise typer.BadParameter(f"response_size must be in 1..{seq_len - 1}, got {response_size}")


def build_split_handlers(
    *,
    base_path: str,
    seq_len: int,
    offset: int,
    min_volume: float,
    train_min_year: int | None,
    train_max_year: int,
    val_min_year: int,
    val_max_year: int | None,
    use_sp500: bool,
    use_quality_allowlist: bool = False,
    clean_rows: bool = False,
    max_abs_r_close: float = 0.75,
) -> tuple[StockHanlder, StockHanlder]:
    """Build disjoint train/val handlers separated by an embargo gap.

    The handlers share everything but their year ranges. ``max_year`` is
    exclusive and ``min_year`` inclusive, so the gap (in years) is
    ``val_min_year - train_max_year``; it must be at least
    ``ceil(seq_len / 365)`` so no window can span both splits.

    Parameters
    ----------
    base_path : str
        Directory of per-symbol parquet partitions.
    seq_len, offset : int
        Window length and stride passed to each handler.
    min_volume : float
        Minimum average volume filter.
    train_min_year, train_max_year : int or None / int
        Train year range (``max_year`` exclusive).
    val_min_year, val_max_year : int / int or None
        Validation year range (``max_year`` exclusive).
    use_sp500 : bool
        If ``True``, fetch the S&P 500 list (Wikipedia) and split history
        (Yahoo) and restrict both handlers to those symbols.
    use_quality_allowlist : bool, optional
        If ``True``, restrict both handlers to the curated allowlist written by
        ``ophir curate`` (see :func:`ophir.register.fetch_quality_symbols_list`).
        Composes with ``use_sp500`` as an intersection. Defaults to ``False``.
    clean_rows : bool, optional
        If ``True``, drop zero-volume and return-spike rows from each stock via
        :func:`ophir.ticker.clean_daily_ohlcv`. Defaults to ``False``.
    max_abs_r_close : float, optional
        Return-spike threshold forwarded to the handlers when ``clean_rows`` is
        set. Defaults to ``0.75``.

    Returns
    -------
    tuple[StockHanlder, StockHanlder]
        The ``(train, val)`` handlers.

    Raises
    ------
    typer.BadParameter
        If the embargo gap is too small for ``seq_len``.
    """
    from ophir.ticker import StockHanlder, get_sp_500_symbols, get_splits

    required_gap = math.ceil(seq_len / _CALENDAR_DAYS_PER_YEAR)
    gap = val_min_year - train_max_year
    if gap < required_gap:
        raise typer.BadParameter(
            f"val_min_year ({val_min_year}) must be at least {required_gap} year(s) "
            f"after train_max_year ({train_max_year}) to embargo a {seq_len}-day "
            f"window; got a {gap}-year gap."
        )

    symbols: list[str] | None = None
    splits: dict[str, Any] | None = None
    if use_sp500:
        symbols = get_sp_500_symbols()
        splits = get_splits(symbols)

    def _handler(min_year: int | None, max_year: int | None) -> StockHanlder:
        handler = StockHanlder(
            seq_len=seq_len,
            base_path=base_path,
            return_stock_id=False,
            return_streamer=True,
            stock_splits=splits,
            offset=offset,
            min_year=min_year,
            max_year=max_year,
            min_volume=min_volume,
            clean_rows=clean_rows,
            max_abs_r_close=max_abs_r_close,
        )
        if symbols is not None:
            handler.keep_stocks(symbols)
        if use_quality_allowlist:
            from ophir import register

            handler.keep_stocks(register.fetch_quality_symbols_list())
        return handler

    return _handler(train_min_year, train_max_year), _handler(val_min_year, val_max_year)


def count_windows(handler: StockHanlder) -> int:
    """Count the training windows across every stock in ``handler``.

    Builds each stock's :class:`~ophir.ticker.StockStreamer` once and sums its
    window count. This is a full preprocessing pass over the data (one parquet
    read + feature extraction per stock), so it can take a while on large
    datasets; it is the price of sizing the schedule to the data instead of a
    constant.

    Parameters
    ----------
    handler : StockHanlder
        The (already filtered) handler whose windows are counted.

    Returns
    -------
    int
        Total number of ``seq_len`` windows the handler can stream per epoch.
    """
    return sum(handler.stock_streamer(stock).size for stock in handler.stocks)


def estimate_windows(handler: StockHanlder, sample_size: int = 256, seed: int = 0) -> int:
    """Estimate total windows from a random sample of stocks.

    Builds streamers for only ``sample_size`` randomly chosen stocks, averages
    their window counts, and extrapolates to the full handler. This avoids a
    full preprocessing pass on large datasets — the estimate is approximate but
    more than accurate enough to set a cosine-schedule horizon. Falls back to an
    exact count (:func:`count_windows`) when the handler has no more stocks than
    the sample size.

    Parameters
    ----------
    handler : StockHanlder
        The (already filtered) handler to estimate over.
    sample_size : int
        Number of stocks to preprocess for the estimate. Defaults to ``256``.
    seed : int
        Seed for the stock sample, for reproducibility. Defaults to ``0``.

    Returns
    -------
    int
        Estimated total number of windows across the handler.
    """
    stocks = handler.stocks
    total_stocks = len(stocks)
    if total_stocks == 0:
        return 0
    if total_stocks <= sample_size:
        return count_windows(handler)
    sampled = random.Random(seed).sample(stocks, sample_size)
    mean_windows = sum(handler.stock_streamer(stock).size for stock in sampled) / sample_size
    return round(mean_windows * total_stocks)


def steps_for_epochs(num_windows: int, batch_size: int, epochs: int) -> int:
    """Convert a window count into a total optimizer-step budget.

    Parameters
    ----------
    num_windows : int
        Total windows per epoch (see :func:`count_windows`).
    batch_size : int
        Samples per step.
    epochs : int
        Number of passes over the data.

    Returns
    -------
    int
        ``epochs * ceil(num_windows / batch_size)`` (at least ``epochs``).
    """
    return epochs * max(1, math.ceil(num_windows / batch_size))


def build_dataloader(
    handler: StockHanlder,
    response_size: int,
    batch_size: int,
    num_workers: int,
    cache_size: int,
) -> DataLoader[dict[str, Any]]:
    """Wrap a handler in a streaming, worker-sharded :class:`DataLoader`.

    Parameters
    ----------
    handler : StockHanlder
        The handler whose stocks are streamed.
    response_size : int
        Number of trailing days the model predicts.
    batch_size, num_workers : int
        Standard ``DataLoader`` knobs. Each handler uses a single
        ``seq_len`` / ``response_size``, so the default collation receives
        uniformly shaped windows.
    cache_size : int
        Number of streamers kept active concurrently per worker (more = more
        cross-stock mixing).

    Returns
    -------
    DataLoader
        A loader over a :class:`~ophir.ticker.StockHandlerDataset`.
    """
    from torch.utils.data import DataLoader

    from ophir.ticker import StockHandlerDataset

    dataset = StockHandlerDataset(handler, response_size=response_size, cache_size=cache_size)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


def train(
    emb_dim: int = 128,
    num_layers: int = 6,
    num_heads: int = 8,
    seq_len: int = 365,
    offset: int = 90,
    response_size: int = 90,
    batch_size: int = 32,
    num_workers: int = 4,
    cache_size: int = 8,
    min_volume: float = 1000.0,
    train_min_year: int | None = None,
    train_max_year: int = 2023,
    val_min_year: int = 2024,
    val_max_year: int | None = None,
    data_dir: str | None = None,
    use_sp500: bool = False,
    use_quality_allowlist: bool = False,
    clean_rows: bool = False,
    max_abs_r_close: float = 0.75,
    epochs: int = 10,
    max_steps: int | None = None,
    window_sample: int = 256,
    val_every_steps: int = 500,
    val_batches: int = 50,
    lr: float = 2e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.03,
) -> None:
    """Pre-train a base :class:`LightningOHLCPredictor` from scratch.

    Builds date-split train/val dataloaders (see :func:`build_split_handlers`),
    instantiates the model, and fits it with
    :func:`~ophir.register.fetch_base_trainer`. Unless ``--max-steps`` is given,
    the step budget is derived from the dataset:
    ``epochs * ceil(num_windows / batch_size)``, where ``num_windows`` is
    *estimated* from a sample of ``window_sample`` stocks
    (:func:`estimate_windows`) so large datasets are not fully scanned. The
    resolved value drives both the trainer and the cosine schedule.

    Validation runs every ``val_every_steps`` optimizer steps over at most
    ``val_batches`` batches, rather than once per (very long) epoch — so
    ``val_loss`` is reported frequently and cheaply. Requires CUDA.

    Parameters
    ----------
    emb_dim, num_layers, num_heads : int
        Model architecture (``emb_dim`` multiple of 4 and divisible by
        ``num_heads``).
    seq_len, offset, response_size : int
        Window length, stride, and forecast horizon.
    batch_size, num_workers, cache_size : int
        Dataloader configuration.
    min_volume : float
        Minimum average volume filter.
    train_min_year, train_max_year, val_min_year, val_max_year : int or None
        Date-split year ranges (``max_year`` exclusive); the gap must embargo at
        least one window length.
    data_dir : str, optional
        Override the data directory (defaults to the package ``.ophir/data/days``).
    use_sp500 : bool
        Restrict to S&P 500 symbols (network fetch). Defaults to ``False``.
    use_quality_allowlist : bool
        Restrict to the curated allowlist from ``ophir curate``. Defaults to
        ``False``.
    clean_rows : bool
        Drop zero-volume and return-spike rows per stock. Defaults to ``False``.
    max_abs_r_close : float
        Return-spike threshold used when ``clean_rows`` is set. Defaults to
        ``0.75``.
    epochs : int
        Number of passes over the data, used to derive ``max_steps`` when it is
        not given. Defaults to ``10``.
    max_steps : int, optional
        Explicit optimizer-step budget. When ``None`` (default) it is computed
        from the (estimated) dataset size and ``epochs``.
    window_sample : int
        Number of stocks sampled to estimate the dataset size. Set ``<= 0`` to
        count every stock exactly (a full data pass). Defaults to ``256``.
    val_every_steps : int
        Run validation every this many optimizer steps. Defaults to ``500``.
    val_batches : int
        Maximum validation batches per validation pass. Defaults to ``50``.
    lr, weight_decay, warmup_ratio : float
        Optimizer / scheduler hyper-parameters.
    """
    from ophir import register
    from ophir.training_models import LightningOHLCPredictor

    _validate_dims(emb_dim, num_heads, seq_len, response_size)

    base_path = os.path.join(data_dir or register.get_default_data_days_dir(), "stocks")
    train_handler, val_handler = build_split_handlers(
        base_path=base_path,
        seq_len=seq_len,
        offset=offset,
        min_volume=min_volume,
        train_min_year=train_min_year,
        train_max_year=train_max_year,
        val_min_year=val_min_year,
        val_max_year=val_max_year,
        use_sp500=use_sp500,
        use_quality_allowlist=use_quality_allowlist,
        clean_rows=clean_rows,
        max_abs_r_close=max_abs_r_close,
    )

    if max_steps is None:
        if window_sample > 0:
            typer.echo(f"Estimating dataset size from a {window_sample}-stock sample…")
            num_windows = estimate_windows(train_handler, sample_size=window_sample)
            prefix = "~"
        else:
            typer.echo("Counting all training windows (one data pass)…")
            num_windows = count_windows(train_handler)
            prefix = ""
        max_steps = steps_for_epochs(num_windows, batch_size, epochs)
        typer.echo(
            f"{prefix}{num_windows} train windows / batch {batch_size} = "
            f"{max_steps // epochs} steps/epoch x {epochs} epochs = {max_steps} max_steps"
        )

    train_dl = build_dataloader(train_handler, response_size, batch_size, num_workers, cache_size)
    val_dl = build_dataloader(val_handler, response_size, batch_size, num_workers, cache_size)

    model = LightningOHLCPredictor(
        emb_dim=emb_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        lr=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        max_steps=max_steps,
    )
    trainer = register.fetch_base_trainer(
        max_steps=max_steps,
        val_check_interval=val_every_steps,
        limit_val_batches=val_batches,
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)


def finetune(
    seq_len: int = 365,
    offset: int = 90,
    response_size: int = 90,
    batch_size: int = 32,
    num_workers: int = 4,
    cache_size: int = 8,
    min_volume: float = 1000.0,
    train_min_year: int | None = None,
    train_max_year: int = 2023,
    val_min_year: int = 2024,
    val_max_year: int | None = None,
    data_dir: str | None = None,
    use_sp500: bool = False,
    use_quality_allowlist: bool = False,
    clean_rows: bool = False,
    max_abs_r_close: float = 0.75,
    strict: bool = False,
    time_version: bool = True,
) -> None:
    """Resume from the latest base checkpoint and finetune.

    Same data setup as :func:`train`, but the model (and its architecture
    hyper-parameters) is restored via
    :func:`~ophir.register.load_base_model_ckpt` and fit with the epoch-driven
    :func:`~ophir.register.fetch_finetune_trainer`. Requires CUDA.

    Parameters
    ----------
    seq_len, offset, response_size : int
        Window length, stride, and forecast horizon.
    batch_size, num_workers, cache_size : int
        Dataloader configuration.
    min_volume : float
        Minimum average volume filter.
    train_min_year, train_max_year, val_min_year, val_max_year : int or None
        Date-split year ranges (``max_year`` exclusive).
    data_dir : str, optional
        Override the data directory.
    use_sp500 : bool
        Restrict to S&P 500 symbols (network fetch). Defaults to ``False``.
    use_quality_allowlist : bool
        Restrict to the curated allowlist from ``ophir curate``. Defaults to
        ``False``.
    clean_rows : bool
        Drop zero-volume and return-spike rows per stock. Defaults to ``False``.
    max_abs_r_close : float
        Return-spike threshold used when ``clean_rows`` is set. Defaults to
        ``0.75``.
    strict : bool
        Passed to the checkpoint loader. Defaults to ``False`` (older
        checkpoints may predate the ``mask_token``).
    time_version : bool
        Load the time-interval checkpoint rather than the best-epoch one.
    """
    from ophir import register

    if not 0 < seq_len <= _MAX_SEQ_LEN:
        raise typer.BadParameter(f"seq_len must be in 1..{_MAX_SEQ_LEN}, got {seq_len}")
    if not 0 < response_size < seq_len:
        raise typer.BadParameter(f"response_size must be in 1..{seq_len - 1}, got {response_size}")

    base_path = os.path.join(data_dir or register.get_default_data_days_dir(), "stocks")
    train_handler, val_handler = build_split_handlers(
        base_path=base_path,
        seq_len=seq_len,
        offset=offset,
        min_volume=min_volume,
        train_min_year=train_min_year,
        train_max_year=train_max_year,
        val_min_year=val_min_year,
        val_max_year=val_max_year,
        use_sp500=use_sp500,
        use_quality_allowlist=use_quality_allowlist,
        clean_rows=clean_rows,
        max_abs_r_close=max_abs_r_close,
    )
    train_dl = build_dataloader(train_handler, response_size, batch_size, num_workers, cache_size)
    val_dl = build_dataloader(val_handler, response_size, batch_size, num_workers, cache_size)

    model = register.load_base_model_ckpt(strict=strict, time_version=time_version)
    trainer = register.fetch_finetune_trainer()
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
