from pathlib import Path

from ophir.trading.forecast import load_forecasts


def test_no_model_dir_returns_empty() -> None:
    assert load_forecasts(["AAPL", "MSFT"], None) == {}


def test_missing_checkpoint_returns_empty(tmp_path: Path) -> None:
    assert load_forecasts(["AAPL"], tmp_path) == {}


def test_present_checkpoint_does_not_raise(tmp_path: Path) -> None:
    (tmp_path / "base.ckpt").write_bytes(b"")
    # Inference is now wired behind CUDA + data + checkpoint guards.  On hosts
    # without CUDA (or missing data), the adapter returns {} without raising.
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
