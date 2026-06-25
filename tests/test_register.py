"""CPU-safe tests for the trainer factories in :mod:`ophir.register`.

Each factory builds a CUDA ``lightning.Trainer``, which cannot be constructed
without a GPU, so ``lightning.Trainer`` and the callback/logger classes are
stubbed. This pins the mixed-precision string the factories request without
touching a device or the filesystem.
"""

from typing import Any

import lightning
import lightning.pytorch.callbacks as lp_callbacks
import lightning.pytorch.loggers as lp_loggers
import pytest

from ophir import register
from ophir.register import _best_checkpoint_callback


class _CapturedTrainer:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _stub(*_a: Any, **_k: Any) -> object:
    return object()


@pytest.fixture
def stub_lightning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lightning, "Trainer", _CapturedTrainer)
    monkeypatch.setattr(lp_callbacks, "ModelCheckpoint", _stub)
    monkeypatch.setattr(lp_callbacks, "LearningRateMonitor", _stub)
    monkeypatch.setattr(lp_loggers, "CSVLogger", _stub)
    monkeypatch.setattr(lp_loggers, "TensorBoardLogger", _stub)


@pytest.mark.usefixtures("stub_lightning")
def test_fetch_base_trainer_uses_bf16() -> None:
    trainer = register.fetch_base_trainer(max_steps=10)
    assert trainer.kwargs["precision"] == "bf16-mixed"  # type: ignore[attr-defined]
    assert trainer.kwargs["accelerator"] == "cuda"  # type: ignore[attr-defined]


@pytest.mark.usefixtures("stub_lightning")
def test_fetch_finetune_trainer_uses_bf16() -> None:
    trainer = register.fetch_finetune_trainer()
    assert trainer.kwargs["precision"] == "bf16-mixed"  # type: ignore[attr-defined]
    assert trainer.kwargs["accelerator"] == "cuda"  # type: ignore[attr-defined]


@pytest.mark.usefixtures("stub_lightning")
def test_predict_trainer_uses_bf16() -> None:
    trainer = register.predict_trainer()
    assert trainer.kwargs["precision"] == "bf16-mixed"  # type: ignore[attr-defined]
    assert trainer.kwargs["accelerator"] == "cuda"  # type: ignore[attr-defined]


def test_best_checkpoint_monitors_near_ic_when_flagged() -> None:
    cb = _best_checkpoint_callback("model", monitor_near_ic=True)
    assert cb.monitor == "val_rank_ic_near"
    assert cb.mode == "max"


def test_best_checkpoint_defaults_to_val_loss() -> None:
    cb = _best_checkpoint_callback("model", monitor_near_ic=False)
    assert cb.monitor == "val_loss"
    assert cb.mode == "min"


def test_best_checkpoint_filename_embeds_near_ic_when_flagged() -> None:
    cb = _best_checkpoint_callback("model", monitor_near_ic=True)
    assert "val_rank_ic_near" in cb.filename
    assert "val_loss" not in cb.filename


def test_best_checkpoint_filename_embeds_val_loss_by_default() -> None:
    cb = _best_checkpoint_callback("model", monitor_near_ic=False)
    assert "val_loss" in cb.filename


def test_latest_base_ckpt_picks_highest_version(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(register, "MODEL_DIR", str(tmp_path))
    for v in (1, 2, 10):
        (tmp_path / f"m-time-check-v{v}.ckpt").write_text("")
    assert register._latest_base_ckpt("m-time-check") == "m-time-check-v10.ckpt"


def test_latest_base_ckpt_no_match_raises(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(register, "MODEL_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        register._latest_base_ckpt("nothing")


def test_latest_base_ckpt_unversioned_matches_no_indexerror(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: many matches, none with a `-v<N>` suffix -> sorted-last, no IndexError.
    monkeypatch.setattr(register, "MODEL_DIR", str(tmp_path))
    for name in ("p-a.ckpt", "p-b.ckpt", "p-c.ckpt"):
        (tmp_path / name).write_text("")
    assert register._latest_base_ckpt("p-") == "p-c.ckpt"
