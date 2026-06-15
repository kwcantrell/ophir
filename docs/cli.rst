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

``ophir research``
------------------

Build grounded research briefs for the top-ranked tickers
(:func:`ophir.agent.research.research_many`). For each top pick the command
gathers fundamentals (Yahoo Finance ``.info``), recent news (Yahoo Finance
``.news``), and technicals (ophir features plus the model forecast), then asks the
local Ollama model to summarize *only that data* into a cited brief. Requires a
CUDA GPU and a trained checkpoint; the LLM summaries also need a local Ollama
server.

.. code-block:: bash

   ophir research <SYMBOL> [<SYMBOL> ...] [--top-k INTEGER]

============== ========= ============================================
Option         Default   Description
============== ========= ============================================
``SYMBOL``     --        One or more ticker symbols to research.
``--top-k``    ``5``     Research at most this many top-ranked tickers.
============== ========= ============================================

The data is fetched deterministically and the model only summarizes it -- it
never invents figures. If Ollama is unreachable the command warns and returns the
grounded data with a neutral, ``llm_ok=False`` analysis. Briefs are written to the
audit trail.

``ophir debate``
----------------

Argue a bull and a bear thesis for each top-ranked ticker
(:func:`ophir.agent.debate.debate_many`). Each ticker is forecast and researched,
then the local Ollama model argues the strongest bullish and bearish cases *from
that research brief only*. Requires a CUDA GPU and a trained checkpoint; the
theses also need a local Ollama server.

.. code-block:: bash

   ophir debate <SYMBOL> [<SYMBOL> ...] [--top-k INTEGER]

============== ========= ============================================
Option         Default   Description
============== ========= ============================================
``SYMBOL``     --        One or more ticker symbols to debate.
``--top-k``    ``5``     Debate at most this many top-ranked tickers.
============== ========= ============================================

Each side reasons only over the grounded brief and never invents figures; if
Ollama is unreachable the theses fall back to neutral (``llm_ok=False``). Weighing
bull against bear is a later phase; debates are written to the audit trail.

``ophir manage``
----------------

Build a gated target portfolio from the full agent ensemble
(:func:`ophir.agent.manage.manage`). Each top-ranked ticker is run through the
decision, research, and debate layers; a manager LLM then chooses and ranks the
names with a conviction each, deterministic code sizes the convictions
(inverse-volatility, scaled toward the target annual volatility), and a risk gate
enforces the per-name cap, gross-exposure limit, and a drawdown / daily-loss
kill-switch. Requires a CUDA GPU and a trained checkpoint; the manager also needs
a local Ollama server.

.. code-block:: bash

   ophir manage <SYMBOL> [<SYMBOL> ...] [--top-k INTEGER]

============== ========= ============================================
Option         Default   Description
============== ========= ============================================
``SYMBOL``     --        One or more ticker symbols to consider.
``--top-k``    ``5``     Consider at most this many top-ranked tickers.
============== ========= ============================================

The LLM only proposes picks and convictions; weights are derived deterministically
and clamped by the risk gate, so no position can exceed the per-name cap and the
book never exceeds the gross-exposure limit. If Ollama is unreachable the manager
returns an all-cash portfolio. Portfolios and gate actions are audit-logged.

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
