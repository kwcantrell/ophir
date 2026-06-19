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
    # Bare loss columns (no _epoch/_step suffix) fall back to plotting all of them.
    model_dir = _write_metrics(
        tmp_path,
        "step,train_loss,train_r_close_loss,val_loss\n0,1.0,0.5,\n1,0.8,0.4,\n2,,,0.7\n",
    )
    fig = dashboard.build_loss_figure(model_dir)
    names = {trace.name for trace in fig.data}
    assert {"train_loss", "train_r_close_loss", "val_loss"} <= names


def test_build_loss_figure_filters_by_granularity(tmp_path: Path) -> None:
    model_dir = _write_metrics(
        tmp_path,
        "step,val_loss_epoch,val_loss_step\n0,0.3,0.31\n1,0.28,0.27\n",
    )
    epoch_names = {t.name for t in dashboard.build_loss_figure(model_dir, "epoch").data}
    step_names = {t.name for t in dashboard.build_loss_figure(model_dir, "step").data}
    assert epoch_names == {"val_loss_epoch"}
    assert step_names == {"val_loss_step"}


def test_summarize_rezero_runs_tabulates_final_values(tmp_path: Path) -> None:
    def _arm(name: str, body: str) -> str:
        d = tmp_path / name / "version_0"
        d.mkdir(parents=True)
        (d / "metrics.csv").write_text(body)
        return str(d)

    versions = {
        "shallow": _arm(
            "shallow",
            "step,val_rank_ic,rezero_mean_abs,rezero_max_abs\n10,0.02,,\n20,0.03,,\n",
        ),
        "deep_open": _arm(
            "deep_open",
            "step,val_rank_ic,rezero_mean_abs,rezero_max_abs\n10,0.05,0.4,0.6\n20,0.07,0.5,0.7\n",
        ),
    }
    df = dashboard.summarize_rezero_runs(versions)
    row = df.set_index("arm")
    assert abs(row.loc["deep_open", "val_rank_ic"] - 0.07) < 1e-9
    assert abs(row.loc["deep_open", "rezero_mean_abs"] - 0.5) < 1e-9
    assert abs(row.loc["shallow", "val_rank_ic"] - 0.03) < 1e-9
    # shallow logged no gate stats -> NaN
    assert row.loc["shallow", "rezero_mean_abs"] != row.loc["shallow", "rezero_mean_abs"]
