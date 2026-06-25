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


def test_stale_checkpoint_runtimeerror_degrades_to_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A schema-stale checkpoint raises RuntimeError from the loader (the dim-13
    # feature-drift case). load_forecasts must degrade to {} like the documented
    # "no signals available" fallback, not propagate and crash `trade propose`.
    import torch

    import ophir.ticker as ticker
    from ophir import register

    (tmp_path / "base.ckpt").write_bytes(b"")
    # Force execution past the CUDA + inputs guards to the checkpoint load on any
    # host (no real CUDA forward runs — the load raises first).
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        ticker, "build_latest_inputs", lambda syms: {"AAPL": {"feature_input": torch.zeros(1)}}
    )

    def _raise_stale(**kwargs: object) -> object:
        raise RuntimeError(
            "checkpoint has feature_mlp input dim 13, but the current model expects 12"
        )

    monkeypatch.setattr(register, "load_base_model_ckpt", _raise_stale)

    assert load_forecasts(["AAPL"], tmp_path) == {}


def test_present_checkpoint_returns_empty_without_cuda(tmp_path: Path) -> None:
    # On a CUDA-less host (CI/test invariant) a present checkpoint must still
    # degrade to {} without attempting the flex-attention forward.
    import torch

    if torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA present: this test asserts the no-CUDA degrade path")
    (tmp_path / "base.ckpt").write_bytes(b"")
    assert load_forecasts(["AAPL"], tmp_path) == {}
