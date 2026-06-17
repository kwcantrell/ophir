"""Backtest & validation for the quant signal.

An event-driven, **walk-forward** backtest that reuses the live risk gate so the
backtest matches production by construction, plus **purged/embargoed** cross-
validation for the overlapping forecast-horizon labels. Forecasting is strictly
as-of: a decision at date ``t`` uses only data through ``t``.

The LLM layers are not backtested (non-deterministic, knowledge-cutoff-bound);
they are validated by paper track record instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ophir.agent import audit
from ophir.agent.config import get_settings
from ophir.agent.feed import load_daily_ohlcv
from ophir.agent.manage import Position, apply_risk_gate

if TYPE_CHECKING:
    from ophir.agent.config import AgentSettings

_SEQ_LEN = 365
_RESPONSE_SIZE = 90
_MIN_HISTORY = 300  # enough rows for the 60-day rolling features + a usable window


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Walk-forward backtest output."""

    equity_curve: list[tuple[str, float]]
    metrics: dict[str, float]
    benchmark_metrics: dict[str, float]
    ic: dict[str, Any]
    n_rebalances: int
    n_trades: int


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(
    period_returns: list[float], *, periods_per_year: float = 12.0
) -> dict[str, float]:
    """Standard performance metrics from a series of per-period returns."""
    empty = {
        "total_return": 0.0,
        "ann_return": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "max_drawdown": 0.0,
        "calmar": 0.0,
        "hit_rate": 0.0,
    }
    if not period_returns:
        return empty
    returns = np.asarray(period_returns, dtype=float)
    equity = np.cumprod(1.0 + returns)
    total_return = float(equity[-1] - 1.0)
    n = len(returns)
    ann_return = (
        float((1.0 + total_return) ** (periods_per_year / n) - 1.0) if total_return > -1.0 else -1.0
    )

    mean = float(returns.mean())
    std = float(returns.std(ddof=1)) if n > 1 else 0.0
    sharpe = float(mean / std * math.sqrt(periods_per_year)) if std > 0 else 0.0
    downside = returns[returns < 0.0]
    dstd = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = float(mean / dstd * math.sqrt(periods_per_year)) if dstd > 0 else 0.0

    peak = np.maximum.accumulate(equity)
    max_drawdown = float((equity / peak - 1.0).min())
    calmar = float(ann_return / abs(max_drawdown)) if max_drawdown < 0 else 0.0
    hit_rate = float((returns > 0.0).mean())

    return {
        "total_return": total_return,
        "ann_return": ann_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "hit_rate": hit_rate,
    }


# --------------------------------------------------------------------------- #
# Purged / embargoed cross-validation
# --------------------------------------------------------------------------- #
def purged_kfold(
    n_samples: int, label_end_idx: list[int], *, n_splits: int = 5, embargo: int = 0
) -> list[tuple[list[int], list[int]]]:
    """Contiguous time folds with purging + embargo (López de Prado).

    ``label_end_idx[i]`` is the bar index at which sample ``i``'s forward-return
    label ends. For each test fold, a train sample is dropped when its
    feature-to-label span ``[i, label_end_idx[i]]`` overlaps the test span
    (**purge**), or when it falls within ``embargo`` bars after the test fold
    (**embargo**). Returns ``(train_idx, test_idx)`` per fold.
    """
    if n_splits < 2 or n_samples < n_splits:
        return []
    fold = n_samples // n_splits
    folds: list[tuple[list[int], list[int]]] = []
    for k in range(n_splits):
        test_start = k * fold
        test_end = n_samples if k == n_splits - 1 else (k + 1) * fold
        test_idx = list(range(test_start, test_end))
        if not test_idx:
            continue
        test_lo, test_hi = test_idx[0], test_idx[-1]
        train_idx: list[int] = []
        for i in range(n_samples):
            if test_start <= i < test_end:
                continue
            if i <= test_hi and label_end_idx[i] >= test_lo:  # purge: label window overlaps test
                continue
            if test_hi < i <= test_hi + embargo:  # embargo after the test fold
                continue
            train_idx.append(i)
        folds.append((train_idx, test_idx))
    return folds


def _spearman(a: Any, b: Any) -> float:
    """Spearman rank correlation; ``nan`` when either side is constant."""
    if len(a) < 2 or float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return float("nan")
    rank_a = np.argsort(np.argsort(a)).astype(float)
    rank_b = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def signal_cv(
    predicted: list[float],
    realized: list[float],
    label_end_idx: list[int],
    *,
    n_splits: int = 5,
    embargo: int = 0,
) -> dict[str, Any]:
    """Per-fold information coefficient (predicted vs realized) under purged CV.

    A constant prediction (e.g. a degenerate model) yields ``nan`` ICs that are
    skipped, so the mean IC reports honestly as ~0.
    """
    pred = np.asarray(predicted, dtype=float)
    real = np.asarray(realized, dtype=float)
    ics: list[float] = []
    for _train, test in purged_kfold(len(pred), label_end_idx, n_splits=n_splits, embargo=embargo):
        if len(test) < 2:
            continue
        index = np.asarray(test, dtype=int)
        ic = _spearman(pred[index], real[index])
        if not math.isnan(ic):
            ics.append(ic)
    arr = np.asarray(ics, dtype=float)
    return {
        "fold_ics": ics,
        "mean_ic": float(arr.mean()) if len(arr) else 0.0,
        "std_ic": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n_folds": len(ics),
    }


# --------------------------------------------------------------------------- #
# As-of forecasting (no look-ahead)
# --------------------------------------------------------------------------- #
def _as_of_input(
    df: Any, asof: Any, *, seq_len: int = _SEQ_LEN, response_size: int = _RESPONSE_SIZE
) -> dict[str, Any]:
    """Model input built only from rows on or before ``asof`` (+ zeroed future)."""
    from ophir.ticker import extract_features, extract_model_data

    history = df[df.index <= asof]
    feats = extract_features(history)
    window = feats.iloc[-(seq_len - response_size) :].reset_index(drop=True)
    future = pd.DataFrame(0.0, index=range(response_size), columns=window.columns)
    # trade_occured=True so the forecast tokens participate in attention and can
    # see history, matching training; False padding-masks them out and collapses
    # every ticker to one constant. (Same fix as feed.forecast_window_tensors.)
    future["trade_occured"] = True
    future = future.astype(window.dtypes)
    full = pd.concat([window, future], ignore_index=True)
    return extract_model_data(full, response_size)


def _forecast_asof(model: Any, df: Any, asof: Any) -> tuple[float, float]:
    """Return (cumulative return, mean predicted downside) forecast as of ``asof``."""
    import torch

    md = _as_of_input(df, asof)
    with torch.no_grad():
        out = model(md)
    r_close = out.predicted_r_close.detach().cpu().reshape(-1).tolist()
    downside = out.predicted_downside.detach().cpu().reshape(-1).tolist()
    cum_return = math.expm1(float(sum(r_close)))
    mean_downside = float(sum(downside) / len(downside)) if downside else 0.0
    return cum_return, mean_downside


def _close_asof(close: Any, asof: Any) -> float | None:
    """Most recent close on or before ``asof`` (``None`` if no bar exists)."""
    prior = close.loc[close.index <= asof]
    return float(prior.iloc[-1]) if len(prior) else None


def _target_weights(
    scored: list[tuple[str, float, float]], settings: AgentSettings
) -> dict[str, float]:
    """Conviction(score)-weighted BUY book, capped + gross-bounded by the live risk gate.

    Mirrors ``quant_decision``'s BUY rule (``score >= buy_threshold`` with the
    downside penalty) and reuses :func:`ophir.agent.manage.apply_risk_gate` for the
    per-name cap and gross-exposure enforcement.
    """
    provisional: list[Position] = []
    for symbol, cum_return, downside_mean in scored:
        score = cum_return - settings.decision_downside_penalty * downside_mean
        if score >= settings.buy_threshold:
            provisional.append(
                Position(
                    symbol=symbol, weight=max(0.0, score), conviction=score, rationale="quant BUY"
                )
            )
    total = sum(p.weight for p in provisional)
    if total > 0:
        provisional = [
            Position(
                symbol=p.symbol,
                weight=p.weight / total,
                conviction=p.conviction,
                rationale=p.rationale,
            )
            for p in provisional
        ]
    gated, _notes, _halted = apply_risk_gate(provisional, settings=settings)
    return {p.symbol: p.weight for p in gated}


# --------------------------------------------------------------------------- #
# Walk-forward backtest
# --------------------------------------------------------------------------- #
def _label_end_indices(ranks: list[int], horizon: int) -> list[int]:
    """For each date-sorted sample, the index where its label window ends."""
    ends: list[int] = []
    n = len(ranks)
    for i in range(n):
        target = ranks[i] + horizon
        end = i
        for j in range(i, n):
            if ranks[j] <= target:
                end = j
            else:
                break
        ends.append(end)
    return ends


def walk_forward(
    symbols: list[str],
    *,
    model: Any,
    benchmark: Any,
    start: str,
    end: str,
    rebalance_days: int = 21,
    cost_bps: float = 5.0,
    cv_splits: int = 5,
    cv_embargo: int = 1,
    settings: AgentSettings | None = None,
    stocks_dir: str | None = None,
) -> BacktestResult:
    """Sequential walk-forward backtest of the quant signal vs a benchmark close series.

    At each rebalance date the signal is forecast **as-of** that date (no look-ahead),
    turned into a BUY book through the live risk gate, and marked to the next bar's
    realized return minus ``cost_bps`` per unit turnover. ``benchmark`` is a
    datetime-indexed close ``Series`` (e.g. SPY) held long over the same dates. The
    per-name (predicted, realized) pairs gathered along the way feed a purged-CV
    information coefficient, returned in ``ic``.
    """
    settings = settings or get_settings()
    frames = {symbol: load_daily_ohlcv(symbol, stocks_dir=stocks_dir) for symbol in symbols}
    calendar = benchmark.loc[
        (benchmark.index >= pd.Timestamp(start)) & (benchmark.index <= pd.Timestamp(end))
    ]
    dates = list(calendar.index[::rebalance_days])

    period_returns: list[float] = []
    bench_returns: list[float] = []
    turnovers: list[float] = []
    ic_pred: list[float] = []
    ic_real: list[float] = []
    ic_rank: list[int] = []
    equity = 1.0
    n_trades = 0
    prev_weights: dict[str, float] = {}
    equity_curve: list[tuple[str, float]] = [(str(dates[0].date()), 1.0)] if dates else []

    for i in range(len(dates) - 1):
        asof, nxt = dates[i], dates[i + 1]
        scored: list[tuple[str, float, float]] = []
        fwd: dict[str, float] = {}
        for symbol, df in frames.items():
            if len(df[df.index <= asof]) < _MIN_HISTORY:
                continue
            c0 = _close_asof(df["close"], asof)
            c1 = _close_asof(df["close"], nxt)
            if c0 is None or c1 is None:
                continue
            cum_return, downside_mean = _forecast_asof(model, df, asof)
            scored.append((symbol, cum_return, downside_mean))
            fwd[symbol] = c1 / c0 - 1.0
            ic_pred.append(cum_return)
            ic_real.append(fwd[symbol])
            ic_rank.append(i)

        weights = _target_weights(scored, settings)
        port_return = sum(weight * fwd.get(symbol, 0.0) for symbol, weight in weights.items())

        names = set(weights) | set(prev_weights)
        turnover = sum(abs(weights.get(s, 0.0) - prev_weights.get(s, 0.0)) for s in names)
        n_trades += sum(
            1 for s in names if abs(weights.get(s, 0.0) - prev_weights.get(s, 0.0)) > 1e-9
        )
        net = port_return - cost_bps / 1e4 * turnover
        equity *= 1.0 + net
        period_returns.append(net)
        turnovers.append(turnover)
        prev_weights = weights

        b0, b1 = _close_asof(benchmark, asof), _close_asof(benchmark, nxt)
        bench_returns.append((b1 / b0 - 1.0) if b0 and b1 else 0.0)
        equity_curve.append((str(nxt.date()), equity))

    ppy = 252.0 / rebalance_days
    metrics = compute_metrics(period_returns, periods_per_year=ppy)
    metrics["mean_turnover"] = float(np.mean(turnovers)) if turnovers else 0.0
    benchmark_metrics = compute_metrics(bench_returns, periods_per_year=ppy)
    label_end = _label_end_indices(ic_rank, horizon=1)
    ic = signal_cv(ic_pred, ic_real, label_end, n_splits=cv_splits, embargo=cv_embargo)
    audit.log_event(
        "backtest",
        n_rebalances=len(period_returns),
        final_equity=round(equity, 4),
        sharpe=round(metrics["sharpe"], 3),
        mean_ic=round(ic["mean_ic"], 4),
    )
    return BacktestResult(
        equity_curve, metrics, benchmark_metrics, ic, len(period_returns), n_trades
    )
