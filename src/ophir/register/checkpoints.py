"""Checkpoint path resolution, model loaders, and load-error hints."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal, NoReturn, overload

from ophir.register import layout

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from ophir.training_models import LightningOHLCPredictor


def _latest_base_ckpt(filename: str) -> str:
    """Return the filename of the most recent base checkpoint.

    Parameters
    ----------
    filename : str
        Substring that base checkpoint files must contain.

    Returns
    -------
    str
        The highest ``-v<N>`` version among matches, or the sorted-last match
        when none carry a ``-v<N>`` suffix.

    Raises
    ------
    FileNotFoundError
        When no file in :data:`MODEL_DIR` contains ``filename``.
    """
    base_paths = sorted(path for path in os.listdir(layout.MODEL_DIR) if filename in path)
    if not base_paths:
        raise FileNotFoundError(f"no checkpoint matching {filename!r} in {layout.MODEL_DIR}")
    versioned = sorted(
        (int(version.removeprefix(f"{filename}-v").removesuffix(".ckpt")), version)
        for version in base_paths
        if f"{filename}-v" in version
    )
    return versioned[-1][1] if versioned else base_paths[-1]


def _latest_finetuned_ckpt() -> str:
    """Return the filename of the most recent finetuned checkpoint.

    Returns
    -------
    str
        The latest :data:`FINETUNE_NAME` checkpoint filename (highest
        ``-v<N>`` version, or the sole match) within :data:`MODEL_DIR`.
    """
    fintune_paths = sorted(
        [path for path in os.listdir(layout.MODEL_DIR) if layout.FINETUNE_NAME in path]
    )
    if len(fintune_paths) > 1:
        base_versions = sorted(
            [
                (
                    int(version.removeprefix(f"{layout.FINETUNE_NAME}-v").removesuffix(".ckpt")),
                    version,
                )
                for version in fintune_paths
                if f"{layout.FINETUNE_NAME}-v" in version
            ]
        )
        latest_version = base_versions[-1][1]
    else:
        latest_version = fintune_paths[0]

    return latest_version


def _resolve_base_ckpt_path(file_name: str | None = None, time_version: bool = True) -> str:
    """Resolve a base checkpoint path without loading the model.

    ``time_version=True`` selects the latest rolling ``-time-check-v<N>``
    checkpoint; ``time_version=False`` selects the explicit canonical
    best-checkpoint ``{layout.MODEL_DIR}/{name}-best.ckpt``.

    Parameters
    ----------
    file_name : str or None, optional
        Base name. ``None`` uses :data:`BASE_NAME`.
    time_version : bool, optional
        Select the rolling time-check checkpoint (``True``) or the canonical
        best checkpoint (``False``). Defaults to ``True``.

    Returns
    -------
    str
        Absolute path to the resolved checkpoint.

    Raises
    ------
    FileNotFoundError
        When no matching checkpoint exists.
    """
    name = file_name if file_name is not None else layout.BASE_NAME
    if time_version:
        return os.path.join(layout.MODEL_DIR, _latest_base_ckpt(name + layout.TIME_MODIFIER))
    path = os.path.join(layout.MODEL_DIR, f"{name}-best.ckpt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"canonical base checkpoint not found: {path}")
    return path


def _feature_dim_mismatch(state_dict: Mapping[str, Any]) -> tuple[int, int] | None:
    """Return ``(found, expected)`` when a checkpoint disagrees on feature width.

    Inspects the ``feature_mlp`` weight in a checkpoint ``state_dict`` and
    compares its input dimension to the current model's
    :data:`ophir.models.FEATURE_DIM`.

    Parameters
    ----------
    state_dict : mapping of str to Any
        A checkpoint ``state_dict`` (parameter name to tensor).

    Returns
    -------
    tuple[int, int] or None
        ``(found, expected)`` when the checkpoint's ``feature_mlp`` input dim
        differs from the current model; ``None`` when it matches or no
        ``feature_mlp`` weight is present (cannot check — do not block).
    """
    from ophir.models import FEATURE_DIM

    key = next((k for k in state_dict if k.endswith("feature_mlp.weight")), None)
    if key is None:
        return None
    found = int(state_dict[key].shape[1])
    return (found, FEATURE_DIM) if found != FEATURE_DIM else None


def _raise_load_error_with_hint(ckpt_path: str, original: RuntimeError) -> NoReturn:
    """Re-raise ``original``, clarifying it first if it is a feature-dim drift.

    A stale checkpoint (trained before a feature-schema change) fails
    ``load_from_checkpoint`` with an opaque ``size mismatch`` error. This reads
    the checkpoint's ``feature_mlp`` width and, on a mismatch, raises a clear,
    actionable error pointing at the promotion runbook; otherwise it re-raises
    the original error unchanged.
    """
    import torch

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception:
        raise original from None
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else {}
    mismatch = _feature_dim_mismatch(state_dict)
    if mismatch is None:
        raise original
    found, expected = mismatch
    raise RuntimeError(
        f"checkpoint {ckpt_path!r} has feature_mlp input dim {found}, but the current "
        f"model expects {expected}; it predates a feature-schema change and must be "
        f"retrained and re-promoted (see docs/checkpoint-promotion.md)."
    ) from original


@overload
def load_base_model_ckpt(
    strict: bool = ...,
    return_ckpt_path: Literal[False] = ...,
    file_name: str | None = ...,
    time_version: bool = ...,
) -> LightningOHLCPredictor: ...


@overload
def load_base_model_ckpt(
    strict: bool = ...,
    *,
    return_ckpt_path: Literal[True],
    file_name: str | None = ...,
    time_version: bool = ...,
) -> tuple[LightningOHLCPredictor, str]: ...


def load_base_model_ckpt(
    strict: bool = True,
    return_ckpt_path: bool = False,
    file_name: str | None = None,
    time_version: bool = True,
) -> LightningOHLCPredictor | tuple[LightningOHLCPredictor, str]:
    """Load the latest base checkpoint.

    Parameters
    ----------
    strict : bool, optional
        Passed to ``load_from_checkpoint``; require an exact ``state_dict``
        match. Defaults to ``True``.
    return_ckpt_path : bool, optional
        If ``True``, also return the resolved checkpoint path. Defaults to
        ``False``.
    file_name : str, optional
        Base name to load. Defaults to :data:`BASE_NAME`.
    time_version : bool, optional
        If ``True``, load the time-interval checkpoint; otherwise the
        best-epoch checkpoint. Defaults to ``True``.

    Returns
    -------
    LightningOHLCPredictor or tuple[LightningOHLCPredictor, str]
        The restored model, and the checkpoint path when
        ``return_ckpt_path`` is ``True``.
    """
    from ophir.training_models import LightningOHLCPredictor

    last_ckpt = _resolve_base_ckpt_path(file_name, time_version)
    print(f"loading {last_ckpt}")

    try:
        model = LightningOHLCPredictor.load_from_checkpoint(last_ckpt, strict=strict)
    except RuntimeError as exc:
        _raise_load_error_with_hint(last_ckpt, exc)

    if not return_ckpt_path:
        return model
    return model, last_ckpt


@overload
def load_finetuned_ckpt(
    strict: bool = ...,
    return_ckpt_path: Literal[False] = ...,
) -> LightningOHLCPredictor: ...


@overload
def load_finetuned_ckpt(
    strict: bool = ...,
    *,
    return_ckpt_path: Literal[True],
) -> tuple[LightningOHLCPredictor, str]: ...


def load_finetuned_ckpt(
    strict: bool = True,
    return_ckpt_path: bool = False,
) -> LightningOHLCPredictor | tuple[LightningOHLCPredictor, str]:
    """Load the latest finetuned checkpoint.

    Parameters
    ----------
    strict : bool, optional
        Passed to ``load_from_checkpoint``. Defaults to ``True``.
    return_ckpt_path : bool, optional
        If ``True``, also return the resolved checkpoint path. Defaults to
        ``False``.

    Returns
    -------
    LightningOHLCPredictor or tuple[LightningOHLCPredictor, str]
        The restored model, and the checkpoint path when
        ``return_ckpt_path`` is ``True``.
    """
    from ophir.training_models import LightningOHLCPredictor

    latest_version = _latest_finetuned_ckpt()
    print(f"loading {latest_version}")

    last_ckpt = os.path.join(layout.MODEL_DIR, latest_version)
    if not return_ckpt_path:
        return LightningOHLCPredictor.load_from_checkpoint(last_ckpt, strict=strict)
    return LightningOHLCPredictor.load_from_checkpoint(last_ckpt, strict=strict), last_ckpt
