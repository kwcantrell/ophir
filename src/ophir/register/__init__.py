"""Filesystem, checkpoint, and Lightning ``Trainer`` helpers.

This package owns the on-disk ``.ophir/`` layout (:mod:`ophir.register.layout`),
ignore/quality symbol-list management (:mod:`ophir.register.symbols`), the
Lightning ``Trainer`` factories (:mod:`ophir.register.trainers`), checkpoint
resolution and loaders (:mod:`ophir.register.checkpoints`), and the MASSIVE API
client plus its ``massive_key`` Typer command (:mod:`ophir.register.client`).
The full public surface — constants, functions, and the Typer ``app`` — is
re-exported here, so ``from ophir import register`` and ``register.<name>`` work
exactly as before.
"""

from __future__ import annotations

from ophir.register.checkpoints import _feature_dim_mismatch as _feature_dim_mismatch
from ophir.register.checkpoints import _latest_base_ckpt as _latest_base_ckpt
from ophir.register.checkpoints import _latest_finetuned_ckpt as _latest_finetuned_ckpt
from ophir.register.checkpoints import _raise_load_error_with_hint as _raise_load_error_with_hint
from ophir.register.checkpoints import _resolve_base_ckpt_path as _resolve_base_ckpt_path
from ophir.register.checkpoints import load_base_model_ckpt, load_finetuned_ckpt
from ophir.register.layout import (
    BASE_BEST_CKPT,
    BASE_MODEL_CKPT,
    BASE_NAME,
    DATA_DIR,
    EPOCH_MODIFIER,
    FINETUNE_NAME,
    MODEL_DIR,
    OPHIR_DIR,
    TIME_MODIFIER,
    get_default_data_days_dir,
    quality_stats_path,
)
from ophir.register.symbols import (
    clear_ignore_symbols,
    clear_quality_symbols,
    fetch_ignore_symbols_list,
    fetch_quality_symbols_list,
    set_ignore_symbols,
    set_quality_symbols,
)
from ophir.register.trainers import _best_checkpoint_callback as _best_checkpoint_callback
from ophir.register.trainers import fetch_base_trainer, fetch_finetune_trainer, predict_trainer

__all__ = [
    "BASE_BEST_CKPT",
    "BASE_MODEL_CKPT",
    "BASE_NAME",
    "DATA_DIR",
    "EPOCH_MODIFIER",
    "FINETUNE_NAME",
    "MODEL_DIR",
    "OPHIR_DIR",
    "TIME_MODIFIER",
    "app",
    "clear_ignore_symbols",
    "clear_quality_symbols",
    "fetch_base_trainer",
    "fetch_finetune_trainer",
    "fetch_ignore_symbols_list",
    "fetch_quality_symbols_list",
    "get_default_data_days_dir",
    "get_massive_client",
    "load_base_model_ckpt",
    "load_finetuned_ckpt",
    "massive_key",
    "predict_trainer",
    "quality_stats_path",
    "set_ignore_symbols",
    "set_quality_symbols",
]


from ophir.register.client import app, get_massive_client, massive_key
