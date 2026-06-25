from pathlib import Path

import pytest

from ophir.trading.forecast import load_forecasts


def test_no_model_dir_returns_empty() -> None:
    assert load_forecasts(["AAPL", "MSFT"], None) == {}


def test_missing_checkpoint_returns_empty(tmp_path: Path) -> None:
    assert load_forecasts(["AAPL"], tmp_path) == {}


def test_present_checkpoint_does_not_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ophir import register

    # Keep resolution off the real ``.ophir/`` layout (and any CUDA forward):
    # point MODEL_DIR at tmp_path, which has no canonical ``-best.ckpt``, so
    # ``load_base_model_ckpt(time_version=False)`` raises FileNotFoundError and
    # ``load_forecasts`` degrades to {} — deterministically, on any host.
    monkeypatch.setattr(register.layout, "MODEL_DIR", str(tmp_path))
    (tmp_path / "base.ckpt").write_bytes(b"")
    # Inference is wired behind CUDA + data + checkpoint guards; the adapter
    # returns {} without raising when the canonical checkpoint is absent.
    result = load_forecasts(["AAPL"], tmp_path)
    assert isinstance(result, dict)


def test_present_checkpoint_returns_empty_without_cuda(tmp_path: Path) -> None:
    # On a CUDA-less host (CI/test invariant) a present checkpoint must still
    # degrade to {} without attempting the flex-attention forward.
    import torch

    if torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA present: this test asserts the no-CUDA degrade path")
    (tmp_path / "base.ckpt").write_bytes(b"")
    assert load_forecasts(["AAPL"], tmp_path) == {}
