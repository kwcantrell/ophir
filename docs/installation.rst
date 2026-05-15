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

Building the documentation
--------------------------

The documentation toolchain is declared as the ``docs`` dependency group:

.. code-block:: bash

   uv sync --group docs
   uv run --group docs sphinx-build -b html docs docs/_build/html
   # open docs/_build/html/index.html

The build mocks all heavy/optional imports (torch, gradio, lightning, …), so it
does not require a GPU or the runtime dependencies to be importable.
