"""MASSIVE API client and the ``massive_key`` Typer command."""

from __future__ import annotations

import os
from typing import Annotated

import typer
from massive import RESTClient

from ophir.register import layout


def get_massive_client() -> RESTClient:
    """Construct an authenticated MASSIVE REST client.

    Reads the API key previously stored by the ``ophir register massive-key``
    command (see :func:`massive_key`).

    Returns
    -------
    massive.RESTClient
        A client authenticated with the stored key.

    Raises
    ------
    AssertionError
        If no key file exists under :data:`OPHIR_DIR`.
    """
    assert os.path.exists(os.path.join(layout.OPHIR_DIR, ".massive_key"))
    with open(os.path.join(layout.OPHIR_DIR, ".massive_key")) as f:
        key = f.readline().strip()
    return RESTClient(key)


app = typer.Typer()


@app.command()
def massive_key(
    key: Annotated[str, typer.Argument(help="MASSIVE API key")],
) -> None:
    """Store a MASSIVE API key for later data fetching.

    Backs the ``ophir register massive-key`` CLI command. The key is written
    to ``.massive_key`` under :data:`OPHIR_DIR` and later read by
    :func:`get_massive_client`.

    Parameters
    ----------
    key : str
        The MASSIVE API key to persist.
    """
    with open(os.path.join(layout.OPHIR_DIR, ".massive_key"), "w") as f:
        f.write(f"{key}\n")
