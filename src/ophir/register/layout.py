"""On-disk ``.ophir/`` layout: data/model directory constants and path helpers.

The ``.ophir/`` root is anchored to the ``ophir`` package directory (one level
above this ``register`` subpackage) and its data/model subdirectories are
created on import, preserving the location used before ``register`` became a
package.
"""

from __future__ import annotations

import os

# layout.py lives at src/ophir/register/layout.py; the .ophir/ root has always
# been anchored at the ophir package dir (src/ophir/), so go up TWO levels.
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPHIR_DIR = os.path.join(current_dir, ".ophir")
DATA_DIR = os.path.join(OPHIR_DIR, "data")
MODEL_DIR = os.path.join(OPHIR_DIR, "model")
BASE_NAME = "ophir-ohlc-base"
FINETUNE_NAME = "ophire-ohlc-finetuned"
BASE_MODEL_CKPT = os.path.join(MODEL_DIR, f"{BASE_NAME}.ckpt")
BASE_BEST_CKPT = os.path.join(MODEL_DIR, f"{BASE_NAME}-best.ckpt")
TIME_MODIFIER = "-time-check"
EPOCH_MODIFIER = "best-{epoch:02d}-{val_loss:.5f}"

if not os.path.exists(OPHIR_DIR):
    os.makedirs(OPHIR_DIR)

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


def get_default_data_days_dir() -> str:
    """Return the default directory holding per-day stock data.

    Returns
    -------
    str
        ``<DATA_DIR>/days``.
    """
    return os.path.join(DATA_DIR, "days")


def quality_stats_path() -> str:
    """Return the path of the curation stats JSON.

    Returns
    -------
    str
        ``<DATA_DIR>/quality-stats.json``.
    """
    return os.path.join(DATA_DIR, "quality-stats.json")
