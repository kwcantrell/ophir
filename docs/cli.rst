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

``ophir ingest``
----------------

Ingest a ticker's daily OHLC from Yahoo Finance into a model-ready dataset
(:func:`ophir.agent.ingest.ingest`).

.. code-block:: bash

   ophir ingest <SYMBOL> [--days INTEGER]

============== ========= ============================================
Option         Default   Description
============== ========= ============================================
``SYMBOL``     --        Ticker symbol, e.g. ``AAPL`` (required).
``--days``     ``730``   Calendar days of history to fetch.
============== ========= ============================================

Writes ``<DATA_DIR>/days/stocks/symbol=<SYMBOL>/data.parquet`` in the layout
:class:`ophir.ticker.StockHanlder` reads, reusing
:func:`ophir.ticker.extract_features`. The default ~2 years covers the model's
365-day window plus rolling-feature warmup. No GPU required.

``ophir predict``
-----------------

Forecast a ticker's next 90 days with the trained model
(:func:`ophir.agent.predict.predict_ticker`); ingests the ticker first if
needed. Requires a CUDA GPU and a trained checkpoint.

.. code-block:: bash

   ophir predict <SYMBOL>

============== ============================================
Argument       Description
============== ============================================
``SYMBOL``     Ticker symbol to forecast (e.g. ``AAPL``).
============== ============================================

``ophir rank``
--------------

Forecast several tickers and print the top picks by predicted cumulative return
(:func:`ophir.agent.predict.rank`). Requires a CUDA GPU and a trained checkpoint.

.. code-block:: bash

   ophir rank <SYMBOL> [<SYMBOL> ...] [--top-k INTEGER]

============== ========= ============================================
Option         Default   Description
============== ========= ============================================
``SYMBOL``     --        One or more ticker symbols to rank.
``--top-k``    ``5``     Number of top picks to print.
============== ========= ============================================

``ophir decide``
----------------

Turn each ticker's forecast into a buy / sell / hold decision via two tracks — a
deterministic quant rule and the local Ollama model — and compare them
(:func:`ophir.agent.decide.compare_decisions`). Requires a CUDA GPU and a trained
checkpoint; the Ollama track additionally needs a local Ollama server serving the
configured model.

.. code-block:: bash

   ophir decide <SYMBOL> [<SYMBOL> ...] [--track both|quant|ollama] [--top-k INTEGER]

============== ========= ============================================
Option         Default   Description
============== ========= ============================================
``SYMBOL``     --        One or more ticker symbols to decide on.
``--track``    ``both``  Which decision track(s) to run.
``--top-k``    ``5``     Decide on at most this many forecasts.
============== ========= ============================================

Decisions are advisory and paper-only: the LLM is grounded in the model's forecast
numbers, its output is validated, and any ambiguity (an unreachable server, a
malformed reply, an unknown action) falls back to ``HOLD``. Every decision is
written to the audit trail.

If the Ollama track is selected and no server answers at the configured URL
(``AGENT_OLLAMA_BASE_URL``, default ``http://localhost:11434``), the command
prints a one-line warning before running and every Ollama decision is ``HOLD``.

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
