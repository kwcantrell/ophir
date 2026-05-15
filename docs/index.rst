ophir documentation
====================

**ophir** is a BERT-style masked transformer for stock OHLC prediction, with a
Gradio UI and a local-LLM chat panel. It trains a full-encoder transformer over
sequences of daily OHLC candles and predicts three forward targets per day —
relative close return, intraday upside, and intraday downside — then explores
the predictions and learned stock embeddings in an interactive dashboard.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   overview
   installation
   cli
   architecture
   api/index
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
