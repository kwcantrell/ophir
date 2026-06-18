from pathlib import Path

from ophir.trading.forecast import load_forecasts


def test_no_model_dir_returns_empty() -> None:
    assert load_forecasts(["AAPL", "MSFT"], None) == {}


def test_missing_checkpoint_returns_empty(tmp_path: Path) -> None:
    assert load_forecasts(["AAPL"], tmp_path) == {}


def test_present_checkpoint_does_not_raise(tmp_path: Path) -> None:
    (tmp_path / "base.ckpt").write_bytes(b"")
    # Real inference is not wired yet; the adapter must still return a dict
    # (empty is acceptable) without raising.
    result = load_forecasts(["AAPL"], tmp_path)
    assert isinstance(result, dict)
