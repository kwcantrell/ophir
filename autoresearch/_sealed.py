"""Sealed split and determinism constants for the autoresearch harness.

Hash-pinned by ``loop.py`` (a change aborts the session) and NEVER
agent-editable: the mutable ``train_experiment.py`` may not contain year
literals and must import these instead. Rationale:

- Train rows end before the embargo year; acceptance-val rows span
  2024-2025 (a 365-row window cannot fit in one calendar year, so a
  single-year split would be EMPTY); rows from 2026 on are the sealed
  holdout, readable only by ``eval_harness.py --holdout`` at graduation.
- ``NUM_WORKERS`` is sealed because loader parallelism changes streaming
  interleave and therefore which checkpoint gets selected — a proposer
  "optimizing throughput" would silently change measurement noise.
"""

TRAIN_MIN_YEAR: int | None = None
TRAIN_MAX_YEAR = 2023  # exclusive bound is the handler's convention: rows < 2024
ACCEPT_VAL_MIN_YEAR = 2024
ACCEPT_VAL_MAX_YEAR = 2026  # exclusive: acceptance never sees a 2026 row
HOLDOUT_VAL_MIN_YEAR = 2026  # graduation only; scoreable ~mid-2027 onward
NUM_WORKERS = 4
