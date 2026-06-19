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
