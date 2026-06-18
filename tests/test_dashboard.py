"""Offline tests for the dashboard metric helpers in :mod:`ophir.dashboard`.

The Gradio server and the (CUDA/checkpoint) leakage panel are not exercised
here; these cover CSV discovery and the loss-figure builder against synthetic
``metrics.csv`` files.
"""

from pathlib import Path

from ophir import dashboard


def _write_metrics(tmp_path: Path, contents: str) -> str:
    version_dir = tmp_path / "csv-logger" / "version_0"
    version_dir.mkdir(parents=True)
    (version_dir / "metrics.csv").write_text(contents)
    return str(tmp_path)


def test_latest_metrics_csv_returns_none_when_absent(tmp_path: Path) -> None:
    assert dashboard._latest_metrics_csv(str(tmp_path)) is None


def test_latest_metrics_csv_finds_file(tmp_path: Path) -> None:
    model_dir = _write_metrics(tmp_path, "step,train_loss\n0,1.0\n")
    found = dashboard._latest_metrics_csv(model_dir)
    assert found is not None
    assert found.endswith("metrics.csv")


def test_build_loss_figure_placeholder_without_metrics(tmp_path: Path) -> None:
    fig = dashboard.build_loss_figure(str(tmp_path))
    assert len(fig.data) == 0  # placeholder carries an annotation, no traces


def test_build_loss_figure_plots_each_loss_series(tmp_path: Path) -> None:
    model_dir = _write_metrics(
        tmp_path,
        "step,train_loss,train_r_close_loss,val_loss\n0,1.0,0.5,\n1,0.8,0.4,\n2,,,0.7\n",
    )
    fig = dashboard.build_loss_figure(model_dir)
    names = {trace.name for trace in fig.data}
    assert {"train_loss", "train_r_close_loss", "val_loss"} <= names
