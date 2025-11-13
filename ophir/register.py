import os

import typer
from massive import RESTClient
from typing_extensions import Annotated

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get the directory containing the current file
current_dir = os.path.dirname(current_file_path)
OPHIR_DIR = os.path.join(current_dir, ".ophir")
DATA_DIR = os.path.join(OPHIR_DIR, "data")
MODEL_DIR = os.path.join(OPHIR_DIR, "model")


if not os.path.exists(OPHIR_DIR):
    os.makedirs(OPHIR_DIR)

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


def get_massive_client():
    assert os.path.exists(os.path.join(OPHIR_DIR, ".massive_key"))
    with open(os.path.join(OPHIR_DIR, ".massive_key"), "r") as f:
        key = f.readline().strip()
    return RESTClient(key)


app = typer.Typer()


@app.command()
def massive_key(
    key: Annotated[str, typer.Argument(help="MASSIVE API key")],
):
    with open(os.path.join(OPHIR_DIR, ".massive_key"), "w") as f:
        f.write(f"{key}\n")
