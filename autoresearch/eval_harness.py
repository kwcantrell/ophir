"""Immutable trial scorer for the autoresearch loop.

FROZEN during a session — ``loop.py`` sha256-pins this file (and
``_sealed.py``) and aborts the session on any change. It owns the acceptance
measurement: the sealed split, a fixed seeded ticker panel, the
deterministic batch subset, and the ``rank_ic_near`` math (reused verbatim
from ``ophir.evaluate`` so the loop, the eval report, and training logs
agree).

Splits (from ``_sealed.py``): acceptance = rows 2024-2025; ``--holdout`` =
rows 2026+ - the sealed slice, ONLY for graduating a champion. A 365-row
window needs ~1.5y of rows, so the holdout cannot produce windows until
~mid-2027; until then this command exits 3 with an explanatory message and
graduation relies on multi-seed + full-budget runs on the acceptance split.

Determinism/representativeness: seeds everything, uses ``num_workers=0``,
and restricts the universe to a fixed seeded panel of tickers so the scored
subset is identical across trials and machines (the raw handler order is
filesystem-dependent and front-loaded). The panel is the first
``PANEL_SIZE`` seed-shuffled tickers that actually form windows in the
split (zero-window tickers are skipped), so the measurement base is
~``PANEL_SIZE`` names/day rather than a blind sample. Requires CUDA.

Guardrail scope: ``main()`` binds every trusted symbol (torch, lightning,
``ophir.register``/``evaluate``/``train``, the sealed constants) BEFORE
importing the agent-authored ``train_experiment`` module, so its top-level
code cannot monkeypatch the scorer's dependencies first. This is
defense-in-depth: eval integrity against adversarial experiment code
ultimately rests on human diff review of the kept commits at graduation, not
on this import ordering alone.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

SEQ_LEN = 365
WINDOW_OFFSET = 90
RESPONSE_SIZE = 90
BATCH_SIZE = 32
MIN_VOLUME = 1000.0
VAL_BATCHES = 200  # matches the confirm harness's powered operating point
EVAL_SEED = 0
PANEL_SIZE = 64  # fixed seeded ticker panel for cross-machine determinism
MIN_EVAL_ROWS = 2000  # below this the daily-IC mean is underpowered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--holdout", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    # Bind every trusted symbol BEFORE importing the agent-authored module, so
    # its top-level code cannot monkeypatch ophir.evaluate/register before the
    # scorer captures them. This is defense-in-depth only: integrity against
    # adversarial experiment code ultimately rests on human diff review of kept
    # commits at graduation (see the module docstring).
    import lightning as L  # noqa: I001 - block ordered trusted-first; agent module imported last
    import torch

    from ophir import register
    from ophir.evaluate import evaluate_model
    from ophir.train import build_dataloader, build_split_handlers

    from _sealed import (
        ACCEPT_VAL_MAX_YEAR,
        ACCEPT_VAL_MIN_YEAR,
        HOLDOUT_VAL_MIN_YEAR,
        TRAIN_MAX_YEAR,
    )
    from train_experiment import MODEL_CLASS

    torch.set_float32_matmul_precision("high")
    L.seed_everything(EVAL_SEED, workers=True)

    if args.holdout:
        val_min, val_max = HOLDOUT_VAL_MIN_YEAR, None
    else:
        val_min, val_max = ACCEPT_VAL_MIN_YEAR, ACCEPT_VAL_MAX_YEAR
    base_path = os.path.join(register.get_default_data_days_dir(), "stocks")
    _, val_handler = build_split_handlers(
        base_path=base_path,
        seq_len=SEQ_LEN,
        offset=WINDOW_OFFSET,
        min_volume=MIN_VOLUME,
        train_min_year=None,
        train_max_year=TRAIN_MAX_YEAR,
        val_min_year=val_min,
        val_max_year=val_max,
        use_sp500=False,
    )
    stocks = sorted(val_handler.stocks)
    random.Random(EVAL_SEED).shuffle(stocks)
    panel: list[str] = []
    for symbol in stocks:
        try:
            has_windows = val_handler.stock_streamer(symbol).size > 0
        except Exception:  # unreadable or too-short ticker: skip it
            continue
        if has_windows:
            panel.append(symbol)
        if len(panel) >= PANEL_SIZE:
            break
    val_handler.keep_stocks(panel)
    val_dl = build_dataloader(
        val_handler,
        RESPONSE_SIZE,
        BATCH_SIZE,
        num_workers=0,
        cache_size=8,
        return_identity=True,
    )

    try:
        model = MODEL_CLASS.load_from_checkpoint(args.ckpt)
    except Exception as exc:  # any load failure must be legible, not swallowed
        print(
            f"CHECKPOINT LOAD FAILED: {exc}\n"
            "If ExperimentPredictor gained an __init__, it must call "
            "super().__init__ and self.save_hyperparameters().",
            file=sys.stderr,
        )
        return 4

    results = evaluate_model(model, val_dl, max_batches=VAL_BATCHES)
    r_close = results["r_close"]
    payload = {
        "rank_ic_near": r_close.get("rank_ic_near", float("nan")),
        "rank_ic_mean": r_close.get("rank_ic_mean", float("nan")),
        "n": r_close.get("n", 0.0),
        **{k: v for k, v in r_close.items() if k.startswith("h")},
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(json.dumps(payload))

    if payload["n"] < MIN_EVAL_ROWS:
        which = "holdout" if args.holdout else "acceptance"
        extra = (
            " (expected before ~mid-2027: a 365-row window needs ~1.5y of post-embargo rows)"
            if args.holdout
            else ""
        )
        print(
            f"TOO FEW ROWS for a powered {which} measurement: "
            f"n={payload['n']} < {MIN_EVAL_ROWS}{extra}",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
