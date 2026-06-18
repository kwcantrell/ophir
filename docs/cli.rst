CLI reference
=============

Installing the package registers the ``ophir`` console script (entry point
``ophir.cli:app``), a `Typer <https://typer.tiangolo.com/>`_ application. The
commands below are documented manually because Typer apps are not introspectable
by the standard Sphinx CLI extensions; the underlying functions are also
available in the :doc:`API reference <api/index>` (:mod:`ophir.cli`).

``ophir serve``
---------------

Launch the Ophir Gradio UI (:func:`ophir.ui.serve`).

.. code-block:: bash

   ophir serve [--port INTEGER] [--share / --no-share] [--debug / --no-debug]

============== ========= ============================================
Option         Default   Description
============== ========= ============================================
``--port``     ``7860``  Gradio server port.
``--share``    ``False`` Expose a public Gradio share link.
``--debug``    ``True``  Launch Gradio in debug mode.
============== ========= ============================================

.. note::

   ``serve`` imports and launches :mod:`ophir.ui`, which at import time fetches
   the S&P 500 list and split history from the network and loads a trained base
   checkpoint onto a CUDA device. A GPU, a checkpoint, network access, and a
   local Ollama server are therefore required for this command to run.

``ophir register massive-key``
------------------------------

Store a `MASSIVE <https://pypi.org/project/massive/>`_ API key for data
fetching. The key is written to ``.massive_key`` inside the package's
``.ophir/`` directory.

.. code-block:: bash

   ophir register massive-key <KEY>

============== ============================================
Argument       Description
============== ============================================
``KEY``        The MASSIVE API key to store.
============== ============================================

The stored key is later read by :func:`ophir.register.get_massive_client` to
construct an authenticated ``massive.RESTClient``.

``ophir train``
---------------

Pre-train a base model from scratch (:func:`ophir.train.train`). Builds a
**by-date** train/validation split with an embargo gap (see
:func:`ophir.train.build_split_handlers`), wires the streaming datasets to a
:class:`~ophir.training_models.LightningOHLCPredictor`, and fits it with
:func:`ophir.register.fetch_base_trainer`. ``--max-steps`` drives both the
trainer and the cosine schedule.

.. code-block:: bash

   ophir train [--emb-dim INT] [--num-layers INT] [--num-heads INT]
               [--seq-len INT] [--offset INT] [--response-size INT]
               [--batch-size INT] [--num-workers INT] [--cache-size INT]
               [--train-max-year INT] [--val-min-year INT]
               [--max-steps INT] [--lr FLOAT] [--use-sp500 / --no-use-sp500]

- ``--emb-dim`` (``128``) — token embedding size (multiple of 4; ``emb_dim // num_heads`` must be ≥ 16 for flex-attention).
- ``--num-layers`` (``6``) — transformer blocks.
- ``--num-heads`` (``8``) — attention heads.
- ``--seq-len`` (``365``) — window length in calendar days (≤ 512).
- ``--offset`` (``90``) — stride between window starts.
- ``--response-size`` (``90``) — forecast horizon (trailing masked days).
- ``--batch-size`` (``32``) — samples per batch.
- ``--train-max-year`` (``2023``) — train upper year bound (exclusive).
- ``--val-min-year`` (``2024``) — validation lower year bound (inclusive).
- ``--epochs`` (``10``) — passes over the data; sizes the step budget when ``--max-steps`` is omitted.
- ``--max-steps`` (auto) — explicit optimizer-step budget; defaults to ``epochs × ceil(num_windows / batch_size)``.
- ``--window-sample`` (``256``) — stocks sampled to *estimate* ``num_windows`` (avoids a full data scan); ``≤ 0`` counts every stock exactly.
- ``--val-every-steps`` (``500``) — run validation every N optimizer steps (decoupled from the epoch).
- ``--val-batches`` (``50``) — max validation batches per validation pass.
- ``--lr`` (``2e-4``) — AdamW learning rate.
- ``--use-sp500`` (``False``) — restrict to S&P 500 symbols (network fetch).
- ``--use-quality-allowlist`` (``False``) — restrict to the curated allowlist written by ``ophir curate`` (intersects with ``--use-sp500`` when both are set).
- ``--clean-rows`` (``False``) — drop zero-volume and return-spike rows per stock via :func:`ophir.ticker.clean_daily_ohlcv`.
- ``--max-abs-r-close`` (``0.75``) — single-day log-return magnitude treated as a glitch when ``--clean-rows`` is set.

``ophir curate``
----------------

Scan the per-stock parquet tree and write a high-quality **symbol allowlist**
(:func:`ophir.curation.curate`). Each symbol is scored on liquidity, history
length & continuity, price sanity, and staleness/flatlines; passing symbols are
written to ``<DATA_DIR>/quality-symbols.txt`` (consumed by ``ophir train
--use-quality-allowlist``) and every symbol's metrics to
``<DATA_DIR>/quality-stats.json``. Row-level cleaning
(:func:`ophir.ticker.clean_daily_ohlcv`) is applied during scoring so the
metrics match what ``--clean-rows`` training sees.

.. code-block:: bash

   ophir curate [--data-dir TEXT] [--min-dollar-volume FLOAT]
                [--min-trading-days INT] [--max-missing-day-fraction FLOAT]
                [--min-median-close FLOAT] [--max-return-spikes INT]
                [--max-abs-r-close FLOAT] [--max-flat-run INT]
                [--max-zero-volume-fraction FLOAT]
                [--use-sp500 / --no-use-sp500]

- ``--min-dollar-volume`` (``1_000_000``) — liquidity floor on median ``close × volume``.
- ``--min-trading-days`` (``252``) — minimum cleaned trading days (~1 year).
- ``--max-missing-day-fraction`` (``0.10``) — max fraction of *business* days with no trade.
- ``--min-median-close`` (``5.0``) — penny-stock floor on the median close.
- ``--max-return-spikes`` (``0``) — max pre-clean ``|r_close| > max_abs_r_close`` days.
- ``--max-abs-r-close`` (``0.75``) — single-day log return treated as a split error / glitch.
- ``--max-flat-run`` (``10``) — max run of identical consecutive closes.
- ``--max-zero-volume-fraction`` (``0.05``) — max fraction of zero-volume days.
- ``--use-sp500`` (``False``) — score only S&P 500 symbols (network fetch).

``ophir finetune``
------------------

Resume from the latest base checkpoint and finetune
(:func:`ophir.train.finetune`). Same date-split data setup as ``train`` but the
model is restored via :func:`ophir.register.load_base_model_ckpt` and fit with
the epoch-driven :func:`ophir.register.fetch_finetune_trainer`.

.. code-block:: bash

   ophir finetune [--seq-len INT] [--response-size INT] [--batch-size INT]
                  [--strict / --no-strict] [--time-version / --no-time-version]

``ophir evaluate``
------------------

Score a checkpoint on the held-out validation set (:func:`ophir.evaluate.evaluate`).
Rebuilds the same **by-date** validation split as ``train`` (only the validation
handler is used), loads the checkpoint(s), runs them over at most
``--val-batches`` batches, and prints a per-target accuracy report (MAE, RMSE,
bias, plus directional accuracy and a zero-baseline skill score for ``r_close``).

.. code-block:: bash

   ophir evaluate [--seq-len INT] [--offset INT] [--response-size INT]
                  [--batch-size INT] [--val-batches INT]
                  [--train-max-year INT] [--val-min-year INT]
                  [--use-sp500 / --no-use-sp500]
                  [--strict / --no-strict] [--finetuned / --no-finetuned]

- ``--seq-len`` (``365``) — window length in calendar days (≤ 512).
- ``--response-size`` (``90``) — forecast horizon (trailing masked days).
- ``--val-batches`` (``50``) — max validation batches to score.
- ``--finetuned`` (``False``) — evaluate the latest finetuned checkpoint; otherwise both base checkpoints (best-``val_loss`` and time-interval) are reported side by side.
- ``--strict`` (``False``) — require an exact ``state_dict`` match when loading.

.. note::

   ``evaluate`` runs the full flex-attention forward, so it requires a CUDA GPU,
   the per-stock parquet tree, and a trained checkpoint.

``ophir dashboard``
-------------------

Launch the live training dashboard (:func:`ophir.dashboard.launch`). Shows
per-target loss curves read live from the ``CSVLogger`` ``metrics.csv``, an
on-demand response-block leakage check, and an on-demand validation evaluation
against the latest checkpoints.

.. code-block:: bash

   ophir dashboard [--port INTEGER] [--share / --no-share]
                   [--debug / --no-debug] [--model-dir TEXT]

- ``--port`` (``7861``) — Gradio server port (so it runs beside ``serve``).
- ``--share`` (``False``) — expose a public Gradio share link.
- ``--debug`` (``True``) — launch Gradio in debug mode.
- ``--model-dir`` (none) — directory holding the metrics and checkpoints.

.. note::

   ``train`` and ``finetune`` require a CUDA GPU and the per-stock parquet tree
   under ``.ophir/data/days/stocks``. Unlike ``serve``, the ``dashboard``
   module is import-safe (no network or checkpoint load at import time); its
   leakage check loads a checkpoint on demand and prefers CUDA when available.

``ophir migrate-sqlite``
------------------------

Convert the per-ticker parquet tree into a single SQLite store
(:func:`ophir.cli.migrate_sqlite`). Builds one table per ticker plus a
``_tickers`` manifest; idempotent (skips tickers already present unless
``--overwrite``).

.. code-block:: bash

   ophir migrate-sqlite [--src PATH] [--dst PATH] [--overwrite / --no-overwrite]

================= =============================== ===================================
Option            Default                          Description
================= =============================== ===================================
``--src``         ``<DATA_DIR>/days/stocks``       Parquet base directory.
``--dst``         ``<DATA_DIR>/days/stocks.db``    Destination SQLite file.
``--overwrite``   ``False``                        Rewrite tables already present.
================= =============================== ===================================
