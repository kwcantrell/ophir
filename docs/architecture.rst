Architecture
============

This page is a guided tour of how data flows from raw parquet files to the
interactive dashboard. The :doc:`API reference <api/index>` documents the
individual symbols; this page explains how they fit together.

Feature pipeline (:mod:`ophir.ticker`)
--------------------------------------

* :func:`~ophir.ticker.get_stock_parquets` discovers one parquet file per
  stock under a base path. :class:`~ophir.ticker.StockHanlder` reads them
  (from parquet or, when ``source="sqlite"``, from a :mod:`ophir.sqlite_store`
  database), aggregates to daily candles, and applies optional ``min_year`` /
  ``max_year`` / ``min_volume`` / history filters.
* :class:`~ophir.ticker.StockSplit` holds split dates and ratios and
  back-adjusts close prices (and inversely, volume) via
  :meth:`~ophir.ticker.StockSplit.apply_splits`. Split history is fetched once
  with :func:`~ophir.ticker.get_splits` (Yahoo Finance) and cached.
* :func:`~ophir.ticker.extract_features` turns an OHLC frame into the 12-feature
  model input: the log close return ``r_close`` (optionally winsorized), rolling
  normalized returns / normalized volume / volatility over 10-, 20-, and 60-day
  windows, and the ``upside`` / ``downside`` log ratios.
  The frame is reindexed onto a daily calendar; padded days are zero-filled and
  flagged by the boolean ``trade_occured`` column.
* :class:`~ophir.ticker.StockStreamer` slices a preprocessed frame into
  fixed-length windows; :class:`~ophir.ticker.StockStreamerDataset` and
  :class:`~ophir.ticker.StockHandlerDataset` adapt these to
  ``torch.utils.data`` loaders. :func:`~ophir.ticker.extract_model_data`
  packages a window into the tensors the model expects.

Structured I/O (:mod:`ophir.model_data`)
-----------------------------------------

:class:`~ophir.model_data.OHLCMulitClassPredictorInput` is the single object
threaded through the model. It carries ``feature_input`` ``(B, S, 12)``, the
``trade_occured`` padding mask, ``targets`` ``(B, S, 3)``, and the integer
``response_size``. The model writes ``model_output`` and ``stock_embeddings``
back onto the same object. Convenience accessors expose the per-target slices
(``target_r_close`` / ``predicted_r_close`` and the upside/downside pairs),
:meth:`~ophir.model_data.OHLCMulitClassPredictorInput.pca_projection` reduces
embeddings to 3-D, and
:meth:`~ophir.model_data.OHLCMulitClassPredictorInput.to_pandas` reconstructs
per-stock frames for plotting.

Model (:mod:`ophir.models`)
---------------------------

:class:`~ophir.models.OHLCMulitClassParameters` is the frozen hyper-parameter
dataclass (``emb_dim``, ``num_layers``, ``num_heads``). The forward path:

#. A linear ``feature_mlp`` projects the 12 features to ``emb_dim`` and adds a
   trainable positional encoding (max length 512).
#. :class:`~ophir.models.CausalPrefixBlockMasks` builds (and caches, keyed on
   sequence/response size) a flex-attention block mask that combines a *prefix*
   mask (history tokens attend only within history) with a *causal* mask, and
   ``and``\ s in a padding mask from
   :func:`~ophir.models.create_padding_mask`.
#. Each :class:`~ophir.models.TransformerBlock` applies
   :class:`~ophir.models.FlexMHA` — multi-head attention whose score is
   modified by ALiBi slopes from :func:`~ophir.models.get_alibi_slopes` — and
   an :class:`~ophir.models.MLP`, each wrapped in a ReZero-scaled residual
   (a single learnable ``_rezero`` scalar initialized to 0).
#. The last ``response_size`` token embeddings are projected to the 3 targets;
   their mean is the pooled ``stock_embeddings`` vector.

Training wrapper (:mod:`ophir.training_models`)
-----------------------------------------------

:class:`~ophir.training_models.LightningOHLCPredictor` wraps the predictor.
:meth:`~ophir.training_models.LightningOHLCPredictor.compute_loss` applies a
masked smooth-L1 loss to each target and returns
``close + 0.5·upside + 0.5·downside``.
:meth:`~ophir.training_models.LightningOHLCPredictor.configure_optimizers`
splits parameters into decay / no-decay / ReZero groups (the ReZero group gets
its own learning rate) and schedules AdamW with a cosine warmup.
:func:`ophir.register.fetch_base_trainer` /
:func:`~ophir.register.fetch_finetune_trainer` build the corresponding
Lightning ``Trainer`` objects and checkpoint callbacks, and
:func:`~ophir.register.load_base_model_ckpt` /
:func:`~ophir.register.load_fintuned_ckpt` restore the latest checkpoint.

The dashboard (``ophir.ui``)
----------------------------

``ophir.ui`` is **not** autodoc'd: at import time it calls
``get_sp_500_symbols()`` (Wikipedia), ``get_splits()`` (Yahoo Finance),
constructs a :class:`~ophir.ticker.StockHanlder`, and loads a base checkpoint
onto CUDA — so importing it requires the network, a checkpoint, and a GPU.

Its public surface is the single function ``ophir.ui.serve(port, share,
debug)``, invoked by ``ophir serve``. Internally:

* ``get_ohlc(symbol)`` streams the last 365 days for a ticker, runs the model,
  and reconstructs predicted vs. actual candles via
  :meth:`~ophir.ticker.StockStreamer.get_ohlcs`.
* ``build_ohlc_figure`` renders the predicted/actual candlestick comparison;
  ``build_embedding_figure`` collects every stock's pooled embedding, PCA-
  projects it to 3-D, and colors the point cloud by predicted percent return.
* ``chat`` relays the conversation to a local Ollama model
  (``ChatOllama(model="gpt-oss:20b")``) via LangChain.

These pieces are assembled into a Gradio ``Blocks`` app (ticker list +
candlestick plot + 3-D embedding cloud + chat) that ``serve`` launches.
