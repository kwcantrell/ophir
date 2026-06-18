"""Offline tests for the training command helpers in :mod:`ophir.train`.

Covers the pure parameter/embargo guards. The handler-building and ``fit``
paths require the parquet tree and a CUDA GPU, so they are exercised by the
end-to-end run rather than here.
"""

from typing import Any

import pytest
import typer

from ophir import train
from ophir.training_models import LightningOHLCPredictor


def test_validate_dims_rejects_non_multiple_of_four() -> None:
    with pytest.raises(typer.BadParameter):
        train._validate_dims(emb_dim=7, num_heads=1, seq_len=12, response_size=4)


def test_validate_dims_rejects_indivisible_heads() -> None:
    with pytest.raises(typer.BadParameter):
        train._validate_dims(emb_dim=8, num_heads=3, seq_len=12, response_size=4)


def test_validate_dims_rejects_small_head_dim() -> None:
    # 64 // 8 = 8 < 16 (flex-attention minimum).
    with pytest.raises(typer.BadParameter):
        train._validate_dims(emb_dim=64, num_heads=8, seq_len=12, response_size=4)


def test_validate_dims_rejects_seq_len_over_max() -> None:
    with pytest.raises(typer.BadParameter):
        train._validate_dims(emb_dim=64, num_heads=4, seq_len=1000, response_size=4)


def test_validate_dims_rejects_response_not_smaller_than_seq() -> None:
    with pytest.raises(typer.BadParameter):
        train._validate_dims(emb_dim=64, num_heads=4, seq_len=12, response_size=12)


def test_validate_dims_accepts_valid_config() -> None:
    train._validate_dims(emb_dim=64, num_heads=4, seq_len=12, response_size=4)


class _FakeStreamer:
    def __init__(self, size: int) -> None:
        self.size = size


class _FakeHandler:
    """Duck-typed stand-in for StockHanlder exposing only what the estimators use."""

    def __init__(self, sizes: dict[str, int]) -> None:
        self.stocks = list(sizes)
        self._sizes = sizes

    def stock_streamer(self, stock: str) -> _FakeStreamer:
        return _FakeStreamer(self._sizes[stock])


def test_count_windows_sums_streamer_sizes() -> None:
    handler = _FakeHandler({"A": 2, "B": 3, "C": 5})
    assert train.count_windows(handler) == 10  # type: ignore[arg-type]


def test_estimate_windows_exact_for_small_handler() -> None:
    handler = _FakeHandler({"A": 2, "B": 3, "C": 5})
    assert train.estimate_windows(handler, sample_size=256) == 10  # type: ignore[arg-type]


def test_estimate_windows_extrapolates_from_sample() -> None:
    # Every stock has 4 windows, so any sample averages 4 -> 4 * 1000 = 4000.
    handler = _FakeHandler({f"S{i}": 4 for i in range(1000)})
    assert train.estimate_windows(handler, sample_size=10) == 4000  # type: ignore[arg-type]


def test_steps_for_epochs_rounds_up_partial_batch() -> None:
    # 100 windows / batch 32 -> ceil = 4 steps/epoch * 3 epochs = 12.
    assert train.steps_for_epochs(num_windows=100, batch_size=32, epochs=3) == 12


def test_steps_for_epochs_floors_at_one_step_per_epoch() -> None:
    assert train.steps_for_epochs(num_windows=0, batch_size=32, epochs=5) == 5
    assert train.steps_for_epochs(num_windows=1, batch_size=32, epochs=1) == 1


def test_build_split_handlers_rejects_zero_year_embargo() -> None:
    with pytest.raises(typer.BadParameter):
        train.build_split_handlers(
            base_path="/nonexistent",
            seq_len=365,
            offset=90,
            min_volume=1000.0,
            train_min_year=None,
            train_max_year=2023,
            val_min_year=2023,
            val_max_year=None,
            use_sp500=False,
        )


def test_build_split_handlers_requires_two_year_gap_for_long_window() -> None:
    # seq_len 400 spans > 1 calendar year, so ceil(400/365) = 2 years are needed.
    with pytest.raises(typer.BadParameter):
        train.build_split_handlers(
            base_path="/nonexistent",
            seq_len=400,
            offset=90,
            min_volume=1000.0,
            train_min_year=None,
            train_max_year=2023,
            val_min_year=2024,
            val_max_year=None,
            use_sp500=False,
        )


class _FakeTrainer:
    def __init__(self) -> None:
        self.fitted_model: LightningOHLCPredictor | None = None

    def fit(self, model: LightningOHLCPredictor, **_: Any) -> None:
        self.fitted_model = model


@pytest.fixture
def patched_engine(monkeypatch: pytest.MonkeyPatch) -> _FakeTrainer:
    """Stub out data + trainer so run_training builds a real CPU model only."""
    trainer = _FakeTrainer()
    monkeypatch.setattr(train, "build_split_handlers", lambda **_: ("train_h", "val_h"))
    monkeypatch.setattr(train, "build_dataloader", lambda *a, **k: "loader")
    monkeypatch.setattr(train, "estimate_windows", lambda *a, **k: 1000)
    monkeypatch.setattr(train, "_validate_dims", lambda *a, **k: None)
    from ophir import register

    monkeypatch.setattr(register, "fetch_base_trainer", lambda **_: trainer)
    monkeypatch.setattr(register, "get_default_data_days_dir", lambda: "/tmp")
    return trainer


def test_run_training_forwards_hyperparameters(patched_engine: _FakeTrainer) -> None:
    model = train.run_training(
        emb_dim=16,
        num_layers=1,
        num_heads=2,
        lr=1e-3,
        rezero_lr=5e-4,
        weight_decay=0.05,
        betas=(0.9, 0.98),
        upside_weight=0.4,
        downside_weight=0.7,
        loss_decay=0.5,
    )
    assert model is patched_engine.fitted_model
    assert model.lr == 1e-3
    assert model.rezero_lr == 5e-4
    assert model.betas == (0.9, 0.98)
    assert model.upside_weight == 0.4
    assert model.downside_weight == 0.7


def test_run_training_passes_val_identity(
    monkeypatch: pytest.MonkeyPatch, patched_engine: _FakeTrainer
) -> None:
    calls: list[bool] = []

    def fake_loader(*_a: Any, return_identity: bool = False, **_k: Any) -> str:
        calls.append(return_identity)
        return "loader"

    monkeypatch.setattr(train, "build_dataloader", fake_loader)
    train.run_training(emb_dim=16, num_layers=1, num_heads=2, val_identity=True)
    # Two loaders built (train, val); the val one carries identity.
    assert calls == [False, True]
