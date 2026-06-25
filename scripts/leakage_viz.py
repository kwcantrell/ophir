"""Web interface that visualizes the OHLC model's target leakage, pre/post fix.

The model forecasts the last ``RESPONSE_SIZE`` days of a window. Leakage means a
forecast for day *t* depends on day *t*'s own input features (which are the
targets). This app makes that visible with an **input-day attribution heatmap**:

* rows  = forecast horizon day (the response block, 0 = first forecast day),
* cols  = input sequence day (0 .. SEQ_LEN-1),
* color = how much that forecast depends on that input day
          (row-normalized magnitude of d(prediction)/d(input)).

Two heatmaps are shown side by side, computed from the *same* weights, differing
only by whether the response block is masked:

* **Leaky (pre-fix)** -- masking disabled: a bright diagonal appears in the
  response region (input day == forecast day), i.e. the model reads the answer.
* **Fixed (post-fix)** -- ``_apply_response_mask`` active: the response region is
  dark; forecasts depend only on the prefix history.

Requires CUDA, the per-stock parquet tree, and a base checkpoint (same runtime
as ``ophir serve``). Run with::

    uv run python scripts/leakage_viz.py

then open the printed local URL.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import torch

from ophir.model_data import OHLCMulitClassPredictorInput
from ophir.register import DATA_DIR
from ophir.ticker import StockHandler, extract_model_data

if TYPE_CHECKING:
    from ophir.training_models import LightningOHLCPredictor

SEQ_LEN = 365
RESPONSE_SIZE = 90
PREFIX_LEN = SEQ_LEN - RESPONSE_SIZE
STRIDE = 3  # subsample forecast days to keep the per-position backward loop snappy
BASE_PATH = os.path.join(DATA_DIR, "days", "stocks")
CHANNELS: dict[str, int] = {
    "close return": OHLCMulitClassPredictorInput.r_close_index,
    "upside": OHLCMulitClassPredictorInput.upside_index,
    "downside": OHLCMulitClassPredictorInput.downside_index,
}

_model: LightningOHLCPredictor | None = None
_handler: StockHandler | None = None
_window_cache: dict[str, torch.Tensor] = {}


def _get_handler() -> StockHandler:
    global _handler
    if _handler is None:
        _handler = StockHandler(
            seq_len=SEQ_LEN,
            base_path=BASE_PATH,
            return_stock_id=False,
            return_streamer=True,
            stock_splits=None,  # offline: no Yahoo/Wikipedia calls
            offset=RESPONSE_SIZE,
            min_volume=1000,
        )
    return _handler


def _get_model() -> LightningOHLCPredictor:
    global _model
    if _model is None:
        import warnings

        from ophir.register import load_base_model_ckpt

        # strict=False: checkpoints predate the new mask_token. Its random init
        # does not affect the attribution *structure* this app visualizes.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _model = load_base_model_ckpt(strict=False, time_version=True).cuda().eval()
    return _model


def _get_window(symbol: str) -> torch.Tensor:
    """Return the (SEQ_LEN, 13) feature window for ``symbol`` as a CUDA tensor."""
    if symbol not in _window_cache:
        streamer = _get_handler()[symbol]
        df = streamer.preprocessed_ohlc_df.iloc[-SEQ_LEN:]  # type: ignore[union-attr]
        md = extract_model_data(df, response_size=RESPONSE_SIZE)
        _window_cache[symbol] = md["feature_input"].cuda()
    return _window_cache[symbol]


def _compute_attribution(symbol: str, channel_idx: int, apply_mask: bool) -> np.ndarray:
    """Row-normalized |d(prediction)/d(input-day)| for each sampled forecast day.

    Returns an array of shape ``(n_sampled_rows, SEQ_LEN)``.
    """
    model = _get_model()
    predictor = model.ohlc_predictor

    feature = _get_window(symbol)
    inp = OHLCMulitClassPredictorInput(
        feature_input=feature,
        targets=torch.zeros((SEQ_LEN, 3), device=feature.device),
        trade_occured=torch.ones(SEQ_LEN, dtype=torch.bool, device=feature.device),
        response_size=torch.tensor(RESPONSE_SIZE),
    )
    inp.feature_input = inp.feature_input.detach().requires_grad_(True)

    # Disable masking for the "leaky" pass by shadowing the bound method with an
    # identity on the instance; restored in `finally`.
    if not apply_mask:
        predictor._apply_response_mask = lambda x, rs: x  # type: ignore[assignment,method-assign]

    rows = range(0, RESPONSE_SIZE, STRIDE)
    attribution = np.zeros((len(list(rows)), SEQ_LEN), dtype=np.float32)
    try:
        for i, p in enumerate(range(0, RESPONSE_SIZE, STRIDE)):
            inp.feature_input.grad = None
            out = predictor(inp)
            assert out.model_output is not None
            out.model_output[0, p, channel_idx].backward()
            assert inp.feature_input.grad is not None
            grad = inp.feature_input.grad[0].abs().sum(dim=-1)
            attribution[i] = grad.detach().cpu().numpy()
    finally:
        if not apply_mask:
            del predictor._apply_response_mask  # type: ignore[method-assign]

    return attribution / (attribution.sum(axis=1, keepdims=True) + 1e-12)


def _self_attention_score(attribution: np.ndarray) -> float:
    """Mean attribution mass a forecast day places on its *own* input day."""
    sampled = range(0, RESPONSE_SIZE, STRIDE)
    diag = [attribution[i, PREFIX_LEN + p] for i, p in enumerate(sampled)]
    return float(np.mean(diag))


def _heatmap(attribution: np.ndarray, title: str) -> go.Figure:
    y = list(range(0, RESPONSE_SIZE, STRIDE))
    fig = go.Figure(
        data=go.Heatmap(
            z=attribution,
            x=list(range(SEQ_LEN)),
            y=y,
            colorscale="Inferno",
            colorbar={"title": "attribution"},
        )
    )
    # Mark where the forecast (response) block begins on the input axis.
    fig.add_vline(x=PREFIX_LEN, line={"color": "cyan", "width": 1, "dash": "dash"})
    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis_title="input day (0 = oldest)",
        yaxis_title="forecast horizon day",
        margin={"l": 60, "r": 20, "t": 50, "b": 50},
    )
    return fig


def build_figures(symbol: str, channel: str) -> tuple[go.Figure, go.Figure, str]:
    """Build the leaky/fixed heatmaps and a summary caption for ``symbol``."""
    channel_idx = CHANNELS[channel]
    leaky = _compute_attribution(symbol, channel_idx, apply_mask=False)
    fixed = _compute_attribution(symbol, channel_idx, apply_mask=True)

    leaky_score = _self_attention_score(leaky)
    fixed_score = _self_attention_score(fixed)

    caption = (
        f"**Self-day attribution** (mass a forecast for day *t* puts on day *t*'s own "
        f"input):\n\n"
        f"- 🔴 Leaky (no mask): **{leaky_score:.1%}** — bright diagonal in the response "
        f"region means the model is reading the answer.\n"
        f"- 🟢 Fixed (masked): **{fixed_score:.2%}** — the response region is dark; "
        f"forecasts use only the prefix (input days left of the dashed line).\n\n"
        f"Channel: *{channel}*. The dashed cyan line at input day {PREFIX_LEN} marks the "
        f"start of the forecast horizon."
    )
    leaky_fig = _heatmap(leaky, f"Leaky (pre-fix) — {symbol} · {channel}")
    fixed_fig = _heatmap(fixed, f"Fixed (post-fix) — {symbol} · {channel}")
    return leaky_fig, fixed_fig, caption


def _build_app() -> gr.Blocks:
    symbols = sorted(_get_handler().stocks)
    default_symbol = "AAPL" if "AAPL" in symbols else symbols[0]
    with gr.Blocks(title="Ophir leakage viz") as demo:
        gr.Markdown(
            "# Ophir leakage visualizer\n"
            "Input-day attribution for the OHLC forecaster, with the response-block "
            "masking fix **off** (leaky) vs **on** (fixed). A diagonal in the response "
            "region is leakage: forecasting a day from that same day's inputs."
        )
        with gr.Row():
            symbol_dd = gr.Dropdown(symbols, value=default_symbol, label="Ticker")
            channel_dd = gr.Dropdown(list(CHANNELS), value="close return", label="Target channel")
            run_btn = gr.Button("Compute", variant="primary")
        caption = gr.Markdown()
        with gr.Row():
            leaky_plot = gr.Plot(label="Leaky (pre-fix)")
            fixed_plot = gr.Plot(label="Fixed (post-fix)")

        run_btn.click(
            build_figures,
            inputs=[symbol_dd, channel_dd],
            outputs=[leaky_plot, fixed_plot, caption],
        )
    return demo


def serve(port: int = 7861, share: bool = False) -> None:
    """Launch the leakage-visualization Gradio app."""
    _build_app().launch(server_port=port, share=share)


if __name__ == "__main__":
    serve()
