"""The Ophir Gradio dashboard and local-LLM chat.

Building blocks: per-ticker inference (:func:`get_ohlc`), a predicted-vs-actual
candlestick figure (:func:`build_ohlc_figure`), a 3-D PCA stock-embedding cloud
(:func:`build_embedding_figure`), and an Ollama-backed chat (:func:`chat`),
assembled into a Gradio ``Blocks`` app launched by :func:`serve`.

.. warning::

   Importing this module runs live network calls (Wikipedia, Yahoo Finance),
   builds a :class:`~ophir.ticker.StockHanlder`, and loads a base checkpoint
   onto a CUDA device. It therefore cannot be imported by the documentation
   build and requires a GPU, a checkpoint, network access, and a local Ollama
   server at runtime.
"""

import os

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import tqdm
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from ophir.model_data import OHLCMulitClassPredictorInput
from ophir.register import DATA_DIR, load_base_model_ckpt
from ophir.ticker import (
    StockHanlder,
    extract_model_data,
    get_sp_500_symbols,
    get_splits,
)

sp_500 = get_sp_500_symbols()
splits = get_splits(sp_500)

base_path = os.path.join(DATA_DIR, "days", "stocks")
elements_per_sample = 365
response_size = 90
offset = 90

val_stock_handler = StockHanlder(
    seq_len=elements_per_sample,
    base_path=base_path,
    return_stock_id=False,
    return_streamer=True,
    stock_splits=splits,
    offset=offset,
    min_year=2024,
    min_volume=1000,
    winsorize_returns=True,
)
val_stock_handler.keep_stocks(sp_500)
stock_streamers = {}
model, ckpt_path = load_base_model_ckpt(return_ckpt_path=True, time_version=True)
model = model.cuda()

# ------------------------------------------------------------------
# 1️⃣  LLM / Prompt
# ------------------------------------------------------------------
system_prompt = SystemMessage(
    content=(
        "Given the following conversation, relevant context, and a\n",
        "follow up question, reply with an answer to the current question\n",
        "the user is asking. Return only your response to the question\n",
        "given the above information following the users instructions as needed.",
    )
)

llm = ChatOllama(model="gpt-oss:20b")


# ------------------------------------------------------------------
# 2️⃣  Helper functions
# ------------------------------------------------------------------
def get_ohlc(symbol: str) -> tuple[pd.DataFrame, OHLCMulitClassPredictorInput]:
    """Run the model on a ticker and reconstruct its candles.

    Parameters
    ----------
    symbol : str
        The ticker symbol to evaluate.

    Returns
    -------
    tuple[pandas.DataFrame, OHLCMulitClassPredictorInput]
        The reconstructed target/predicted OHLC frame and the populated model
        output object.
    """
    if symbol not in stock_streamers:
        stock_streamers[symbol] = val_stock_handler[symbol]

    streamer = stock_streamers[symbol]
    with torch.no_grad():
        model_input = extract_model_data(
            streamer.preprocessed_ohlc_df.iloc[-365:],
            response_size=response_size,
            return_date=True,
        )
        model_output = model(model_input)
    ohlcs = stock_streamers[symbol].get_ohlcs(model_output.to_pandas()[0])
    return ohlcs, model_output


def build_ohlc_figure(symbol: str) -> go.Figure:
    """Build the predicted-vs-actual candlestick figure for a ticker.

    Parameters
    ----------
    symbol : str
        The ticker symbol to plot.

    Returns
    -------
    plotly.graph_objects.Figure
        A dark-themed figure with target and predicted candlestick traces.
    """
    df, _ = get_ohlc(symbol)
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["target_open"],
                high=df["target_high"],
                low=df["target_low"],
                close=df["target_close"],
            ),
            go.Candlestick(
                x=df.index,
                open=df["predicted_open"],
                high=df["predicted_high"],
                low=df["predicted_low"],
                close=df["predicted_close"],
            ),
        ]
    )
    fig.update_layout(
        title=f"OHLC for {symbol}",
        title_x=0.5,
        template="plotly_dark",
        margin={"l": 20, "r": 20, "t": 40, "b": 50},
        hovermode="x unified",
    )
    return fig


def percent_return(closes: np.ndarray) -> float:
    """Compute the percent return between the first and last close.

    Parameters
    ----------
    closes : numpy.ndarray
        A 1-D series of close prices.

    Returns
    -------
    float
        ``100 · (last - first) / first`` (with a small epsilon guard).
    """
    initial_val = closes[0] + 1e-8
    final_val = closes[-1]
    return 100 * (final_val - initial_val) / initial_val


def build_embedding_figure() -> go.Figure:
    """Build the 3-D PCA stock-embedding point cloud.

    Collects every kept stock's pooled embedding, PCA-projects the
    winsorized/normalized embeddings to 3-D, and renders a scatter colored by
    each stock's predicted percent return (with reference axes).

    Returns
    -------
    plotly.graph_objects.Figure
        The dark-themed 3-D embedding figure.
    """
    # ------------------------------------------------------------------
    # 1️⃣  Gather all embeddings
    # ------------------------------------------------------------------
    empbeddings = []
    r_closes = []  # array
    for stock in tqdm.tqdm(val_stock_handler.stocks):
        ohlcs, model_outputs = get_ohlc(stock)  # embeddings: np.ndarray
        if len(ohlcs) < 250:
            print(f"{stock} only covers {len(ohlcs)}/{elements_per_sample} history.")
        else:
            # r_closes.append((model_outputs.predicted_r_close.mean(dim=1)))
            response_size = (
                model_outputs.response_size.cpu().numpy() + 1
            )  # add one to include last history day
            r_closes.append(
                percent_return(ohlcs.iloc[-response_size:]["predicted_close"].to_numpy())
            )
            empbeddings.append(model_outputs.stock_embeddings)

    # Concatenate along the sample axis and force float32

    r_close = np.array(r_closes)
    embeddings = torch.concatenate(empbeddings, dim=0)

    # ------------------------------------------------------------------
    # 2️⃣  PCA - Project Stock embeddings into a 3d space
    # given N stocks, embeddings is an NxE matrix
    # ------------------------------------------------------------------
    x_centered = embeddings - embeddings.mean(0, keepdim=True)

    # winsorize each dimension to remove extreme values
    x_centered = x_centered.clip(min=x_centered.quantile(0.01), max=x_centered.quantile(0.91))
    x_centered = x_centered / (
        x_centered.std(dim=0, keepdim=True) + 1e-8
    )  # (optional) unit variance

    # create point cloud
    _u, _s, v = torch.pca_lowrank(x_centered, q=5)
    point_cloud = torch.matmul(x_centered, v[:, :3])

    # center point cloud
    centered_cloud = point_cloud - point_cloud.mean(0, keepdim=True)

    # unit variance
    centered_cloud = centered_cloud / (centered_cloud.std(0, keepdim=True) + 1e-8)

    # scale dims to [-1,1]
    max_range = centered_cloud.abs().max(dim=0).values
    centered_cloud /= max_range

    # bring everything to numpy
    centered_cloud = centered_cloud.cpu().numpy()

    # ------------------------------------------------------------------
    # 3️⃣  Build the scatter trace
    # ------------------------------------------------------------------
    trace = go.Scatter3d(
        x=centered_cloud[:, 0],
        y=centered_cloud[:, 1],
        z=centered_cloud[:, 2],
        mode="markers",
        marker={
            "size": 5,
            "opacity": 0.8,
            # If you want a colour-map, uncomment the next line:
            "color": r_close,  # mean close, make sure this is numpy
            "colorscale": "Viridis",
            "showscale": True,  # show a color bar
            "colorbar": {"title": "r_close"},
            "cmin": np.percentile(r_close, 5),
            "cmax": np.percentile(r_close, 95),
        },
    )

    # ------------------------------------------------------------------
    #  Build the axis traces
    # ------------------------------------------------------------------
    axis_traces = [
        # X-axis  (red)
        go.Scatter3d(
            x=[-1, 1],
            y=[0, 0],
            z=[0, 0],
            mode="lines",
            line={"color": "red", "width": 4, "showscale": False},
            name="X-axis",
            showlegend=False,
        ),
        # Y-axis  (green)
        go.Scatter3d(
            x=[0, 0],
            y=[-1, 1],
            z=[0, 0],
            mode="lines",
            line={"color": "green", "width": 4, "showscale": False},
            name="Y-axis",
            showlegend=False,
        ),
        # Z-axis  (blue)
        go.Scatter3d(
            x=[0, 0],
            y=[0, 0],
            z=[-1, 1],
            mode="lines",
            line={"color": "blue", "width": 4, "showscale": False},
            name="Z-axis",
            showlegend=False,
        ),
    ]

    # ------------------------------------------------------------------
    #  Build the axis label traces
    # ------------------------------------------------------------------
    axis_labels = [
        go.Scatter3d(
            x=[1, 0, 0],
            y=[0, 1, 0],
            z=[0, 0, 1],
            mode="markers+text",  # show both markers and text
            marker={"size": 5, "color": "rgba(0,0,0,0)", "showscale": False},
            text=["PC 1", "PC 2", "PC 3"],  # list of strings, same length as data
            textposition=[
                "top center",
                "top center",
                "top center",
            ],  # where the label sits relative to the marker
            textfont={"size": 12, "color": "white"},
            showlegend=False,
        ),
    ]

    # ------------------------------------------------------------------
    #  Create figure
    # ------------------------------------------------------------------
    fig = go.Figure(data=[trace, *axis_traces, *axis_labels])

    # ------------------------------------------------------------------
    #  Set the background axis (these are difference than the trace axes)
    # ------------------------------------------------------------------
    axis_dict = {
        "showticklabels": False,
        "showgrid": False,
        "zeroline": False,
        "showline": False,
        "showspikes": False,
        "ticks": "",
    }

    # ------------------------------------------------------------------
    #  Dark model and camera config
    # ------------------------------------------------------------------
    fig.update_layout(
        template="plotly_dark",
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        dragmode="orbit",
        scene={
            "xaxis": dict(title="", backgroundcolor="rgba(0,0,0,0)", **axis_dict),
            "yaxis": dict(title="", backgroundcolor="rgba(0,0,0,0)", **axis_dict),
            "zaxis": dict(title="", backgroundcolor="rgba(0,0,0,0)", **axis_dict),
            "aspectmode": "cube",
            "camera": {
                "eye": {"x": 0, "y": 0, "z": 1.5},  # <-- further away
                "center": {"x": 0, "y": 0, "z": 0},  # <-- look at data centre
                "up": {"x": 0, "y": 1, "z": 0},
                "projection": {"type": "perspective"},
            },
        },
    )

    return fig


# ------------------------------------------------------------------
# 3️⃣  Chat logic (now compatible with Gradio)
# ------------------------------------------------------------------
def chat(user_message: str, history: list) -> tuple[list, list]:
    """Append a user turn, query the local LLM, and return updated history.

    Parameters
    ----------
    user_message : str
        The string typed by the user.
    history : list
        Conversation history as dicts with ``role`` and ``content`` keys.

    Returns
    -------
    tuple[list, list]
        The updated history, returned for both the chatbot component and the
        Gradio ``State``.
    """
    if not user_message:
        # Nothing to do - just echo the current history back
        return history, history

    # Append the user message to the conversation history
    history.append({"role": "user", "content": user_message})

    # Build LangChain messages for the LLM
    lc_messages = [system_prompt] + [
        HumanMessage(content=m["content"])
        if m["role"] == "user"
        else AIMessage(content=m["content"])
        for m in history
    ]

    # Call the LLM
    response = llm.invoke(lc_messages)
    assistant_reply = response.content

    # Append the assistant's reply to the history
    history.append({"role": "assistant", "content": assistant_reply})

    # Return the updated history for both outputs
    return history, history


# 4️⃣  UI - Chat on the left, plots + scrollable list on the right
with gr.Blocks() as demo:
    with gr.Row(equal_height=True, elem_classes=["max-row"]):
        # ── Left column (ticker list) ──
        with gr.Column(scale=0, min_width=150, elem_classes="thin", variant="compact"):
            tickers = sorted(val_stock_handler.stocks)
            ticker_list = gr.Radio(
                choices=tickers,
                value="AAPL",
                label="Select a ticker",
                interactive=True,
                elem_classes=["ticker-list"],  # CSS hook
            )

        with gr.Column(scale=5, elem_classes=["same-line"]):
            # ── Right column (plots) ──
            # with gr.Column(scale=1, elem_classes=["same-row"]):
            ohlc_plot = gr.Plot(
                label="OHLC", value=build_ohlc_figure("AAPL"), elem_classes=["center"]
            )
            # Assume build_embedding_figure() is defined above
            embedding_plot = gr.Plot(
                label="Embedding Scatter (3-D)", value=build_embedding_figure()
            )
            ticker_list.change(
                lambda symbol: build_ohlc_figure(symbol),
                inputs=ticker_list,
                outputs=ohlc_plot,
            )

    with gr.Row(), gr.Column():
        # ── Left column (chat) ──
        chatbot = gr.Chatbot(label="Chatbot", height=800)
        msg = gr.Textbox(
            elem_id="user_textbox",
            placeholder="Type your message…",
            lines=1,
            interactive=True,
        )
        send_btn = gr.Button("Send", variant="primary")
        state = gr.State(value=[])

        send_btn.click(chat, inputs=[msg, state], outputs=[chatbot, state])
        msg.submit(chat, inputs=[msg, state], outputs=[chatbot, state]).then(
            lambda: gr.update(msg.elem_id, value="")
        )

        # # Global CSS (order doesn't matter - it's applied to the whole page)
        gr.HTML(
            """
            <style>
            /* 1️⃣  The row that holds the two columns */
            .max-row { max-height:95vh; }

            /* 2️⃣  Left column - keep it narrow, scrollable, full-height */
            .thin {
                flex: 0 0 150px;      /* fixed width - adjust if you want more */
                max-height:95vh;
                overflow-y:auto;
            }

            /* 3️⃣  Make the Radio component itself a column flex container */
            .gradio-radio.ticker-list {
                display:flex          !important;
                flex-direction:column !important;
                flex-wrap:nowrap      !important;
                max-height:100%;
                width:100% !important;
                overflow-y:auto;
            }

            /* 4️⃣  Each option takes 100 % of the column's width */
            .gradio-radio.ticker-list .gradio-radio__item {
                flex:0 0 100% !important;
                width:100% !important;
                display:block !important;   /* block-level so it fills the width */
            }

            /* 5️⃣  Optional - make the inner label block-level as well */
            .gradio-radio.ticker-list .gradio-radio__label {
                display:block !important;
                width:100% !important;
            }
            .same-line {
                vertical-align: middle;
            }
            .center {
                vertical-align: middle;
            }
            </style>
            """
        )


# ------------------------------------------------------------------
# 5️⃣  Launch
# ------------------------------------------------------------------
def serve(port: int = 7860, share: bool = False, debug: bool = True) -> None:
    """Launch the Gradio dashboard.

    Parameters
    ----------
    port : int, optional
        Gradio server port. Defaults to ``7860``.
    share : bool, optional
        If ``True``, expose a public Gradio share link. Defaults to ``False``.
    debug : bool, optional
        If ``True``, launch Gradio in debug mode. Defaults to ``True``.
    """
    demo.launch(server_port=port, share=share, debug=debug)


if __name__ == "__main__":
    serve()
