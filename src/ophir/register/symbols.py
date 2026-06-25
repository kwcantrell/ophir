"""Persisted ignore / quality symbol-list management."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ophir.register import layout

if TYPE_CHECKING:
    from collections.abc import Iterable


def clear_ignore_symbols() -> None:
    """Delete the ignore-symbols list, re-enabling all symbols.

    Removes ``<DATA_DIR>/ignore-symbols.txt`` if it exists; a no-op otherwise.
    """
    print("Reseting ignore symbols mode...")
    path = os.path.join(layout.DATA_DIR, "ignore-symbols.txt")
    if os.path.exists(path):
        os.remove(path)


def set_ignore_symbols(symbols: Iterable[str]) -> None:
    """Add symbols to the persistent ignore list.

    The new symbols are unioned with the existing list and written back,
    sorted, to ``<DATA_DIR>/ignore-symbols.txt``.

    Parameters
    ----------
    symbols : Iterable[str]
        Ticker symbols to add to the ignore list.
    """
    merged = sorted(set(fetch_ignore_symbols_list()).union(symbols))
    with open(os.path.join(layout.DATA_DIR, "ignore-symbols.txt"), "w") as f:
        for symbol in merged:
            f.write(f"{symbol}\n")
    print(f"Entering Ignore Symbol mode...currently ignoring {len(merged)} symbols")


def fetch_ignore_symbols_list() -> list[str]:
    """Return the persisted ignore-symbols list.

    Returns
    -------
    list[str]
        Symbols read from ``<DATA_DIR>/ignore-symbols.txt``, or an empty list
        if the file does not exist.
    """
    if not os.path.exists(os.path.join(layout.DATA_DIR, "ignore-symbols.txt")):
        return []

    with open(os.path.join(layout.DATA_DIR, "ignore-symbols.txt")) as f:
        symbols = [symbol.strip() for symbol in f.readlines()]
    return symbols


def set_quality_symbols(symbols: Iterable[str]) -> None:
    """Write the curated quality allowlist, replacing any existing list.

    Unlike the ignore list (which unions), each curation run produces a complete
    allowlist, so the file is overwritten with the sorted, de-duplicated symbols.

    Parameters
    ----------
    symbols : Iterable[str]
        Ticker symbols that passed curation.
    """
    merged = sorted(set(symbols))
    with open(os.path.join(layout.DATA_DIR, "quality-symbols.txt"), "w") as f:
        for symbol in merged:
            f.write(f"{symbol}\n")
    print(f"Wrote quality allowlist with {len(merged)} symbols")


def fetch_quality_symbols_list() -> list[str]:
    """Return the persisted quality allowlist.

    Returns
    -------
    list[str]
        Symbols read from ``<DATA_DIR>/quality-symbols.txt``, or an empty list
        if the file does not exist.
    """
    if not os.path.exists(os.path.join(layout.DATA_DIR, "quality-symbols.txt")):
        return []

    with open(os.path.join(layout.DATA_DIR, "quality-symbols.txt")) as f:
        symbols = [symbol.strip() for symbol in f.readlines()]
    return symbols


def clear_quality_symbols() -> None:
    """Delete the quality allowlist if it exists; a no-op otherwise."""
    path = os.path.join(layout.DATA_DIR, "quality-symbols.txt")
    if os.path.exists(path):
        os.remove(path)
