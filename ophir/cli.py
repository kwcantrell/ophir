import os

import typer
from massive import RESTClient
from typing_extensions import Annotated

from . import register, ticker

# def ticker_data(
#     symbols: Annotated[List[str], typer.Argument(help="list of ticker symbols")] = "",
# ):
#     pass


app = typer.Typer()
app.add_typer(register.app, name="register")
app.add_typer(ticker.app, name="ticker")
