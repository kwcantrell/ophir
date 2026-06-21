from pathlib import Path

import pandas as pd

from ophir.ceiling import RunICSummary, run_ic_summary


def _write_metrics(tmp_path: Path) -> Path:
    # Mimic a Lightning CSVLogger metrics.csv: train-step rows have NaN val
    # metrics; epoch rows carry both val_loss_epoch and val_rank_ic.
    rows = [
        {"step": 100, "val_loss_epoch": None, "val_rank_ic": None},  # train step
        {"step": 500, "val_loss_epoch": 0.90, "val_rank_ic": 0.010},  # epoch 1
        {"step": 1000, "val_loss_epoch": 0.70, "val_rank_ic": 0.030},  # peak IC, min loss
        {"step": 1500, "val_loss_epoch": 0.75, "val_rank_ic": 0.020},  # later
        {"step": 2000, "val_loss_epoch": 0.80, "val_rank_ic": 0.014},  # final (annealed)
    ]
    path = tmp_path / "metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_run_ic_summary_extracts_peak_best_final(tmp_path: Path) -> None:
    summary = run_ic_summary(_write_metrics(tmp_path))
    assert summary == RunICSummary(
        peak_ic=0.030, peak_step=1000, best_ckpt_ic=0.030, final_ic=0.014
    )


def test_run_ic_summary_best_ckpt_differs_from_peak(tmp_path: Path) -> None:
    # min val_loss at a row that is NOT the IC peak.
    rows = [
        {"step": 500, "val_loss_epoch": 0.60, "val_rank_ic": 0.012},  # min loss
        {"step": 1000, "val_loss_epoch": 0.95, "val_rank_ic": 0.040},  # peak IC
    ]
    path = tmp_path / "m.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    summary = run_ic_summary(path)
    assert summary.peak_ic == 0.040
    assert summary.best_ckpt_ic == 0.012
