"""Single-file SQLite store for per-ticker stock data.

Mirrors the parquet ingest path in :mod:`ophir.ticker`: one table per ticker
in a single database, with a ``_tickers`` manifest mapping each true symbol to
its sanitized table name and the column dtypes needed to restore frames
byte-identically to ``pandas.read_parquet``.
"""

from __future__ import annotations

import re


def sanitize_table_name(ticker: str, used: set[str]) -> str:
    """Return a unique, SQL-safe table name for ``ticker``.

    The name is ``t_`` followed by ``ticker`` with every non-alphanumeric
    character replaced by ``_``. If that name is already in ``used``, a
    numeric suffix (``_2``, ``_3``, …) is appended until it is unique. The
    chosen name is added to ``used``.

    Parameters
    ----------
    ticker : str
        The true ticker symbol (may contain ``.`` and other punctuation).
    used : set[str]
        Names already assigned; mutated in place with the returned name.

    Returns
    -------
    str
        A unique table name safe to use as a SQLite identifier.
    """
    base = "t_" + re.sub(r"[^0-9A-Za-z]", "_", ticker)
    name = base
    suffix = 2
    while name in used:
        name = f"{base}_{suffix}"
        suffix += 1
    used.add(name)
    return name
