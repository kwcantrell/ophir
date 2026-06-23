"""Confirm per-offset cross-sectional skill multi-seed against a per-offset null.

Joins the logged ``val_rank_ic_h*`` from one or more ``--log-offset-ic`` training
runs (averaged across validation snapshots and seeds) to a within-day permutation
null computed from a CPU validation harvest, and prints a per-offset verdict
table (see ``ophir.ceiling.confirm_offset_skill``).

The harvest is a ``torch.save`` dict with tensors ``target``, ``ids``, ``dates``,
``offsets`` (offsets via ``ophir.training_models.trading_day_offsets``). Run with::

    uv run python scripts/confirm_offset_skill.py \
        --harvest harvest.pt \
        --metrics .ophir/.../version_0/metrics.csv .ophir/.../version_1/metrics.csv

CPU-only; no model or CUDA required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ophir.ceiling import confirm_offset_skill, format_verdict_table
from ophir.training_models import _OFFSET_BUCKETS


def load_harvest(
    path: str | Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a saved harvest dict into a ``(target, ids, dates, offsets)`` tuple."""
    blob = torch.load(path, map_location="cpu")
    return blob["target"], blob["ids"], blob["dates"], blob["offsets"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harvest", required=True, type=Path)
    parser.add_argument("--metrics", required=True, nargs="+", type=Path)
    parser.add_argument("--n-perms", type=int, default=500)
    parser.add_argument("--burn-in-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    verdicts = confirm_offset_skill(
        args.metrics,
        load_harvest(args.harvest),
        list(_OFFSET_BUCKETS),
        n_perms=args.n_perms,
        burn_in_steps=args.burn_in_steps,
        seed=args.seed,
    )
    print(format_verdict_table(verdicts))


if __name__ == "__main__":
    main()
