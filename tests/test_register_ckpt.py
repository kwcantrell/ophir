"""Tests for register's checkpoint picker (filesystem only, no GPU)."""

import os

import pytest

from ophir import register


def _touch(path, mtime=None):
    path.write_bytes(b"")
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def test_picks_highest_version_suffix(monkeypatch, tmp_path):
    monkeypatch.setattr(register, "MODEL_DIR", str(tmp_path))
    for name in [
        "ophir-ohlc-base-time-check.ckpt",
        "ophir-ohlc-base-time-check-v1.ckpt",
        "ophir-ohlc-base-time-check-v5.ckpt",
        "ophir-ohlc-base-time-check-v4.ckpt",
    ]:
        _touch(tmp_path / name)
    assert (
        register._latest_base_ckpt("ophir-ohlc-base-time-check")
        == "ophir-ohlc-base-time-check-v5.ckpt"
    )


def test_picks_newest_mtime_without_version(monkeypatch, tmp_path):
    monkeypatch.setattr(register, "MODEL_DIR", str(tmp_path))
    _touch(tmp_path / "ophir-ohlc-basebest-epoch=00-val_loss=0.01392.ckpt", mtime=1_000)
    _touch(tmp_path / "ophir-ohlc-basebest-epoch=21-val_loss=0.01498.ckpt", mtime=2_000)
    _touch(tmp_path / "ophir-ohlc-basebest-epoch=22-val_loss=0.00658.ckpt", mtime=3_000)
    assert (
        register._latest_base_ckpt("ophir-ohlc-basebest-")
        == "ophir-ohlc-basebest-epoch=22-val_loss=0.00658.ckpt"
    )


def test_single_match_returned(monkeypatch, tmp_path):
    monkeypatch.setattr(register, "MODEL_DIR", str(tmp_path))
    _touch(tmp_path / "ophir-ohlc-basebest-epoch=07-val_loss=0.50000.ckpt")
    assert register._latest_base_ckpt("ophir-ohlc-basebest-").endswith(
        "epoch=07-val_loss=0.50000.ckpt"
    )


def test_no_match_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(register, "MODEL_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        register._latest_base_ckpt("does-not-exist-")


def test_finetuned_picker_uses_finetune_name(monkeypatch, tmp_path):
    monkeypatch.setattr(register, "MODEL_DIR", str(tmp_path))
    _touch(tmp_path / f"{register.FINETUNE_NAME}-v2.ckpt")
    _touch(tmp_path / f"{register.FINETUNE_NAME}-v9.ckpt")
    assert register._latest_finetuned_ckpt() == f"{register.FINETUNE_NAME}-v9.ckpt"
