"""Deterministic outer loop for the autoresearch harness.

Torch-free by design: this module imports only the standard library so its
decision, validation, and logging logic is unit-testable offline (the repo
suite is CPU-only). The GPU work happens in subprocesses
(``train_experiment.py`` / ``eval_harness.py``).

The runner — not the proposer agent — owns git, timeouts, the accept/reject
decision, and the trial log. Every non-keep outcome reverts to the
iteration's captured ``base_sha`` (never ``HEAD~1``), and trial commits use
``--no-verify`` because the repo pre-commit hook (mypy/ruff) is intentionally
not the gate for agent-written experiment code. See
``docs/superpowers/specs/2026-07-07-autoresearch-forecaster-harness-design.md``.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable

HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HARNESS_DIR)

#: The single file the proposer agent may modify (repo-relative).
MUTABLE_FILE = "autoresearch/train_experiment.py"

#: Files whose sha256 is pinned for the whole session; a mismatch aborts.
PINNED_FILES = (
    "autoresearch/eval_harness.py",
    "autoresearch/loop.py",
    "autoresearch/_sealed.py",
)

#: This exact import must survive every proposal: split/determinism constants
#: live only in the hash-pinned ``_sealed.py``.
SEALED_IMPORT_LINE = (
    "from _sealed import ACCEPT_VAL_MAX_YEAR, ACCEPT_VAL_MIN_YEAR, NUM_WORKERS, "
    "TRAIN_MAX_YEAR  # SEALED: loop.py enforces this line verbatim"
)

#: Year-like literals (2019-2031) are banned from the mutable file so split
#: years cannot be hardcoded around the sealed module. Chosen to exclude
#: common non-year constants (128, 2048, 10000, ...).
YEAR_LITERAL_RE = re.compile(r"\b20(19|2[0-9]|3[01])\b")

#: Accept a trial only if rank_ic_near > best + EPSILON. Initial value sits
#: above the confirm-harness 3-seed MDE (0.0069); MUST be recalibrated from
#: >=3 baseline seeds (epsilon = 2*SE) before any unattended session.
EPSILON = 0.02

#: Abort the session if the baseline metric falls outside this band — a
#: baseline of -0.3 or +0.5 means the harness, not the model, is broken.
BASELINE_SANITY = (-0.02, 0.20)

TRAIN_TIMEOUT_S = 600
EVAL_TIMEOUT_S = 1800
PROPOSE_TIMEOUT_S = 600
MAX_STEPS = 10000
SEED = 0
PROPOSER_MODEL = "opus"
MAX_CONSECUTIVE_PROPOSER_FAILS = 3

#: A subprocess runner: (cmd, cwd=..., timeout=..., input_text=...) ->
#: (returncode, merged output). Injectable so tests never spawn processes.
Runner = Callable[..., tuple[int, str]]

RESULTS_HEADER = "\t".join(
    ("iter", "utc", "hypothesis", "status", "rank_ic_near", "h1", "h5", "wall_s", "commit")
)

#: Metric keys copied from the harness JSON into the loop's bookkeeping.
_METRIC_KEYS = ("rank_ic_near", "rank_ic_mean", "n", "h1", "h2", "h5", "h10", "h20", "h40", "h90")


def decide(candidate: float | None, best: float | None, epsilon: float = EPSILON) -> bool:
    """Return ``True`` when ``candidate`` should replace ``best``.

    The first finite result becomes the baseline; after that a candidate must
    beat the incumbent by more than ``epsilon``. ``None``/NaN never win.
    """
    if candidate is None or candidate != candidate:
        return False
    if best is None or best != best:
        return True
    return candidate > best + epsilon


def parse_porcelain(text: str) -> tuple[list[str], list[str]]:
    """Split ``git status --porcelain`` output into (modified, untracked) paths.

    Renames report their destination path. Anything not ``??`` counts as a
    tracked modification (staged or not) — the loop treats them identically.
    """
    modified: list[str] = []
    untracked: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        status, path = line[:2], line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if status == "??":
            untracked.append(path)
        else:
            modified.append(path)
    return modified, untracked


def diff_is_valid(modified: list[str], untracked: list[str], session_rel: str) -> bool:
    """A proposal is valid iff it modified exactly the mutable file.

    Untracked files are tolerated only inside the session directory (the
    proposer writes its ``.hypothesis`` note there).
    """
    if modified != [MUTABLE_FILE]:
        return False
    return all(path.startswith(session_rel + "/") for path in untracked)


def validate_experiment_source(text: str) -> str | None:
    """Check post-edit invariants of the mutable file; ``None`` when valid."""
    if SEALED_IMPORT_LINE not in text:
        return "sealed import line missing (_sealed constants were bypassed)"
    stripped = text.replace(SEALED_IMPORT_LINE, "")
    match = YEAR_LITERAL_RE.search(stripped)
    if match:
        return f"year literal {match.group(0)!r} found (split years live in _sealed.py only)"
    return None


def parse_metrics(path: str) -> dict[str, float]:
    """Read the named metric keys from the eval harness's JSON output."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return {key: float(raw[key]) for key in _METRIC_KEYS if key in raw}


def format_result_row(
    *,
    iteration: int,
    utc: str,
    hypothesis: str,
    status: str,
    rank_ic_near: float | None,
    h1: float | None,
    h5: float | None,
    wall_s: float,
    commit: str,
) -> str:
    """Render one tab-separated results.tsv row (tabs/newlines sanitized)."""

    def _num(value: float | None) -> str:
        if value is None or value != value:
            return "nan"
        return f"{value:.5f}"

    clean_hyp = " ".join(hypothesis.split())
    cells = (
        str(iteration),
        utc,
        clean_hyp,
        status,
        _num(rank_ic_near),
        _num(h1),
        _num(h5),
        f"{wall_s:.0f}",
        commit,
    )
    return "\t".join(cells)


def append_result(tsv_path: str, row: str) -> None:
    """Append ``row`` to the trial log, writing the header on first use."""
    new = not os.path.exists(tsv_path)
    with open(tsv_path, "a", encoding="utf-8") as fh:
        if new:
            fh.write(RESULTS_HEADER + "\n")
        fh.write(row + "\n")
