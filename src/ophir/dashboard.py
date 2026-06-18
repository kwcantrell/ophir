"""Standalone live training dashboard for the OHLC forecaster.

Separate from :mod:`ophir.ui` (which loads a checkpoint and hits the network at
import time): this module is import-safe — ``gradio``, ``plotly``, ``pandas``,
``torch``, and the model are imported lazily inside functions, so importing it
is cheap. It surfaces the two things hardest to see during a training run:

* **Loss per target** — live train/val curves for the combined loss and each
  target (``r_close`` / ``upside`` / ``downside``), read from the ``CSVLogger``
  ``metrics.csv`` written during training and auto-refreshed on a timer.
* **Leakage check** — on demand, perturbs the response-block inputs of the
  latest checkpoint and reports the resulting change in the model output per
  target (see :mod:`ophir.leakage`); ``0`` means the masking contract holds.

Launched by :func:`launch` / the ``ophir dashboard`` command.
"""

from __future__ import annotations

import glob
import os
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import gradio as gr
    import plotly.graph_objects as go
    import torch

#: A leakage score at or below this is treated as "no leakage" (floating-point
#: noise only).
_LEAKAGE_TOLERANCE = 1e-5


def _latest_metrics_csv(model_dir: str) -> str | None:
    """Return the newest ``csv-logger`` ``metrics.csv`` under ``model_dir``.

    Parameters
    ----------
    model_dir : str
        Directory passed to the Lightning ``CSVLogger`` (the model directory).

    Returns
    -------
    str or None
        Path to the most-recently-modified ``metrics.csv``, or ``None`` if none
        exist yet.
    """
    matches = glob.glob(os.path.join(model_dir, "csv-logger", "version_*", "metrics.csv"))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def _placeholder(message: str) -> go.Figure:
    """Build an empty dark-themed figure with a centered ``message``."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        annotations=[
            {
                "text": message,
                "showarrow": False,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "font": {"size": 16},
            }
        ],
        margin={"l": 60, "r": 20, "t": 50, "b": 50},
    )
    return fig


def build_loss_figure(model_dir: str) -> go.Figure:
    """Build the live loss-per-target figure from ``metrics.csv``.

    Plots every logged loss series (any column containing ``loss``) against
    ``step``. ``CSVLogger`` writes sparse rows (a metric is ``NaN`` on rows
    where it was not logged), so each series drops its own missing values.
    Returns a placeholder while metrics are unavailable.

    Parameters
    ----------
    model_dir : str
        The model directory holding the ``csv-logger`` output.

    Returns
    -------
    plotly.graph_objects.Figure
        A dark-themed multi-series line chart, or a placeholder figure.
    """
    import pandas as pd  # type: ignore[import-untyped]
    import plotly.graph_objects as go

    path = _latest_metrics_csv(model_dir)
    if path is None:
        return _placeholder("waiting for metrics.csv …")
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return _placeholder("waiting for metrics.csv …")

    if "step" not in df.columns:
        return _placeholder("metrics.csv has no 'step' column yet …")
    loss_cols = [c for c in df.columns if "loss" in c.lower()]
    if not loss_cols:
        return _placeholder("no loss metrics logged yet …")

    fig = go.Figure()
    for col in sorted(loss_cols):
        series = df[["step", col]].dropna()
        if series.empty:
            continue
        fig.add_trace(go.Scatter(x=series["step"], y=series[col], mode="lines", name=col))
    fig.update_layout(
        title="Loss per target",
        template="plotly_dark",
        xaxis_title="step",
        yaxis_title="loss",
        hovermode="x unified",
        margin={"l": 60, "r": 20, "t": 50, "b": 50},
    )
    return fig


def _leakage_window(seq_len: int) -> torch.Tensor:
    """Return a ``(1, seq_len, 13)`` feature window for the leakage check.

    Uses a real offline window for the first available symbol when the parquet
    tree is present; otherwise falls back to random features (the leakage
    property is architectural, so random inputs are a valid probe).
    """
    import torch

    from ophir.register import get_default_data_days_dir

    base_path = os.path.join(get_default_data_days_dir(), "stocks")
    if os.path.isdir(base_path):
        try:
            from ophir.ticker import StockHanlder, extract_model_data

            handler = StockHanlder(
                seq_len=seq_len,
                base_path=base_path,
                return_stock_id=False,
                return_streamer=True,
                stock_splits=None,
                offset=seq_len,
                min_volume=1000,
            )
            symbols = sorted(handler.stocks)
            if symbols:
                symbol = "AAPL" if "AAPL" in symbols else symbols[0]
                df = handler[symbol].preprocessed_ohlc_df.iloc[-seq_len:]
                if len(df) >= seq_len:
                    feature = extract_model_data(df, response_size=1)["feature_input"]
                    return cast("torch.Tensor", feature.unsqueeze(0))
        except (IndexError, KeyError, ValueError, OSError):
            pass
    return torch.randn(1, seq_len, 13)


def run_leakage_check(seq_len: int = 365, response_size: int = 90) -> str:
    """Run the leakage probe against the latest base checkpoint.

    On CUDA, runs the full forward (:func:`~ophir.leakage.end_to_end_leakage_scores`)
    and reports a per-target score; without CUDA, falls back to the CPU-safe
    masking probe (:func:`~ophir.leakage.response_block_leakage_score`).

    Parameters
    ----------
    seq_len, response_size : int
        Window length and forecast horizon to probe.

    Returns
    -------
    str
        A Markdown summary with a pass/leakage verdict.
    """
    import torch

    from ophir import register
    from ophir.leakage import end_to_end_leakage_scores, response_block_leakage_score

    try:
        lightning_model = register.load_base_model_ckpt(strict=False)
    except (FileNotFoundError, IndexError, OSError):
        return "**No base checkpoint found** — train a base model first."

    predictor = lightning_model.ohlc_predictor.eval()

    if torch.cuda.is_available():
        predictor = predictor.cuda()
        feature = _leakage_window(seq_len).cuda()
        scores = end_to_end_leakage_scores(predictor, feature, response_size)
        worst = max(scores.values())
        verdict = "✅ pass" if worst <= _LEAKAGE_TOLERANCE else "🚨 LEAKAGE"
        lines = "\n".join(f"- `{name}`: `{value:.3e}`" for name, value in scores.items())
        return (
            f"### Leakage check — end-to-end forward — {verdict}\n\n"
            "Max change in each target output when only the response block is "
            "perturbed (0 = no leakage):\n\n"
            f"{lines}"
        )

    feature = _leakage_window(seq_len)
    score = response_block_leakage_score(predictor, feature, response_size)
    verdict = "✅ pass" if score <= _LEAKAGE_TOLERANCE else "🚨 LEAKAGE"
    return (
        f"### Leakage check — CPU masking probe — {verdict}\n\n"
        f"Max change in the masked representation when the response block is "
        f"perturbed: `{score:.3e}` (0 = no leakage).\n\n"
        "*CUDA unavailable — ran the CPU-safe representation probe instead of the "
        "full forward.*"
    )


def build_demo(model_dir: str, seq_len: int = 365, response_size: int = 90) -> gr.Blocks:
    """Assemble the dashboard ``Blocks`` app.

    Parameters
    ----------
    model_dir : str
        Directory holding the ``csv-logger`` metrics.
    seq_len, response_size : int
        Window dimensions used by the leakage probe.

    Returns
    -------
    gradio.Blocks
        The (unlaunched) dashboard app.
    """
    import gradio as gr

    with gr.Blocks(title="Ophir training dashboard") as demo:
        gr.Markdown(
            "# Ophir training dashboard\n"
            "Live training diagnostics for the OHLC forecaster: per-target loss "
            "curves (auto-refreshing from `metrics.csv`) and an on-demand "
            "response-block leakage check."
        )
        with gr.Tab("Loss per target"):
            loss_plot = gr.Plot(value=build_loss_figure(model_dir))
            timer = gr.Timer(value=5.0)
            timer.tick(lambda: build_loss_figure(model_dir), outputs=loss_plot)
        with gr.Tab("Leakage check"):
            leakage_md = gr.Markdown(
                "Press **Run leakage check** to probe the latest base checkpoint."
            )
            run_btn = gr.Button("Run leakage check", variant="primary")
            run_btn.click(lambda: run_leakage_check(seq_len, response_size), outputs=leakage_md)
    return cast("gr.Blocks", demo)


def launch(
    port: int = 7861,
    share: bool = False,
    debug: bool = True,
    model_dir: str | None = None,
    seq_len: int = 365,
    response_size: int = 90,
) -> None:
    """Launch the live training dashboard.

    Parameters
    ----------
    port : int, optional
        Gradio server port. Defaults to ``7861`` (so it runs beside
        ``ophir serve``'s ``7860``).
    share : bool, optional
        Expose a public Gradio share link. Defaults to ``False``.
    debug : bool, optional
        Launch Gradio in debug mode. Defaults to ``True``.
    model_dir : str, optional
        Directory holding the ``csv-logger`` metrics and checkpoints. Defaults
        to the package model directory (:data:`ophir.register.MODEL_DIR`).
    seq_len, response_size : int, optional
        Window dimensions used by the leakage probe.
    """
    from ophir.register import MODEL_DIR

    demo = build_demo(model_dir or MODEL_DIR, seq_len=seq_len, response_size=response_size)
    demo.launch(server_port=port, share=share, debug=debug)
