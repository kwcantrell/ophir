"""ophir: a BERT-style masked transformer for stock OHLC prediction.

The package provides the model architecture (:mod:`ophir.models`), a
PyTorch-Lightning training wrapper (:mod:`ophir.training_models`), the data
pipeline (:mod:`ophir.ticker`), the structured model I/O dataclass
(:mod:`ophir.model_data`), checkpoint/trainer helpers (:mod:`ophir.register`),
a Gradio dashboard (``ophir.ui``), and the ``ophir`` Typer CLI
(:mod:`ophir.cli`).
"""


def hello() -> str:
    """Return a friendly greeting.

    Returns
    -------
    str
        A constant greeting string. Useful as a trivial import smoke test.
    """
    return "Hello from ophir!"
