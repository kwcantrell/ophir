import os
from typing import List

import typer
from typing_extensions import Annotated

from . import coin_datasets, models, register, training_models

__all__ = [
    "app",
    "coin_datasets",
    "models",
    "training_models",
    "get_massive_client",
]
