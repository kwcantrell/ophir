Overview
========

Ophir trains a full-encoder (BERT-style) masked transformer over sequences of
daily OHLC (Open / High / Low / Close) candles. For each day in the *response*
window it predicts three targets:

* **r_close** — the log relative close return,
* **upside** — the intraday high relative to the close,
* **downside** — the intraday low relative to the close.

Predictions are reconstructed back into candlesticks and explored in an
interactive Gradio dashboard, which also embeds every S&P 500 stock into the
model's learned representation space and projects it into a 3-D point cloud
colored by predicted return.

What it does
------------

#. **Data pipeline** (:mod:`ophir.ticker`) — loads historical OHLC data from
   per-stock parquet files, applies stock-split adjustments, and extracts a
   13-dimensional technical feature vector per day (log returns, rolling
   normalized returns / volume / volatility over 10/20/60-day windows, plus
   upside and downside). Missing calendar days are padded and flagged with a
   ``trade_occured`` mask.
#. **Model** (:mod:`ophir.models`) — a stack of transformer blocks with ALiBi
   positional bias, ``torch`` flex-attention using a cached causal + prefix
   block mask, and ReZero residual scaling. The model emits per-day predictions
   for the response tokens plus a single pooled stock embedding.
#. **Training** (:mod:`ophir.training_models`) — a PyTorch-Lightning module
   wraps the predictor, computing a weighted smooth-L1 loss
   (``close + 0.5·upside + 0.5·downside``) and optimizing with AdamW + cosine
   warmup, with a separate learning rate for the ReZero parameters.
#. **Visualization** (:mod:`ophir.ui`) — a Gradio app shows predicted-vs-actual
   candlesticks, a 3-D PCA stock-embedding cloud, and a chat panel backed by a
   local Ollama model.

Package modules
---------------

================================  ====================================================
Module                            Purpose
================================  ====================================================
:mod:`ophir`                      Package root.
:mod:`ophir.cli`                  Typer CLI app (the ``ophir`` entry point).
:mod:`ophir.models`               Core transformer architecture.
:mod:`ophir.training_models`      PyTorch-Lightning training wrapper.
:mod:`ophir.model_data`           Structured model input/output dataclass.
:mod:`ophir.ticker`               Stock data loading, splits, features, datasets.
:mod:`ophir.register`             Trainer factories, checkpoint loaders, data dirs.
``ophir.ui``                      Gradio UI and local-LLM chat (see :doc:`architecture`).
================================  ====================================================

.. note::

   ``ophir.ui`` is documented narratively in :doc:`architecture` rather than via
   autodoc: importing it runs live network calls (Wikipedia, Yahoo Finance) and
   loads a model checkpoint at import time, so it cannot be imported by the
   documentation build.
