Installation
============

Requirements
------------

* **Python >= 3.10.**
* **A CUDA GPU.** Training and inference paths call ``.cuda()`` and configure
  Lightning with ``accelerator="cuda"``; the UI will not run on CPU.
* **A trained base checkpoint.** ``ophir serve`` loads the latest base
  checkpoint from the package data directory at startup; without one it will
  not start.
* **Ollama running locally**, serving the ``gpt-oss:20b`` model, for the chat
  panel.
* **Network access at UI startup** — the S&P 500 constituent list is fetched
  from Wikipedia and split history from Yahoo Finance (results are cached under
  the package ``.ophir/`` directory).

Installing the package
----------------------

Using `uv <https://docs.astral.sh/uv/>`_ (recommended):

.. code-block:: bash

   uv sync                 # create the environment and install ophir

Using pip:

.. code-block:: bash

   pip install .           # or: pip install -e .   (editable, for development)

The distribution name and the import name are both ``ophir``. Installing the
package registers the ``ophir`` console script (see :doc:`cli`).

Setting up Ollama
-----------------

The chat panel (``ophir serve``) and the Ollama decision track
(``ophir decide --track ollama`` / ``both``) call a local
`Ollama <https://ollama.com/>`_ server running the ``gpt-oss:20b`` model. Without
it those LLM calls fail safe to ``HOLD`` (the decision layer never trades on an
unreachable model).

1. **Install Ollama.**

   .. code-block:: bash

      # Windows
      winget install --id Ollama.Ollama -e
      # macOS / Linux: download from https://ollama.com/download

2. **Start the server** (the desktop app starts it automatically; otherwise):

   .. code-block:: bash

      ollama serve

3. **Pull the model** (~13 GB; needs roughly 16 GB of VRAM):

   .. code-block:: bash

      ollama pull gpt-oss:20b

4. **Verify** it is reachable and the model is present:

   .. code-block:: bash

      ollama list
      curl http://localhost:11434/api/tags

To point ophir at a non-default host or port, set ``AGENT_OLLAMA_BASE_URL``
(e.g. ``AGENT_OLLAMA_BASE_URL=http://192.168.1.10:11434``); the model name is
likewise overridable via ``AGENT_OLLAMA_MODEL``.

Building the documentation
--------------------------

The documentation toolchain is declared as the ``docs`` dependency group:

.. code-block:: bash

   uv sync --group docs
   uv run --group docs sphinx-build -b html docs docs/_build/html
   # open docs/_build/html/index.html

The build mocks all heavy/optional imports (torch, gradio, lightning, …), so it
does not require a GPU or the runtime dependencies to be importable.
