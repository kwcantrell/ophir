# Autoresearch Forecaster Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An autoresearch loop (`autoresearch/`) that lets a headless proposer agent iteratively edit one experiment file, trains each edit for a time-boxed 10k steps, scores `rank_ic_near` with an immutable eval harness, and keeps or reverts via git — logging every trial.

**Architecture:** Spec/script split per the design spec
(`docs/superpowers/specs/2026-07-07-autoresearch-forecaster-harness-design.md`):
`program.md` (human-owned directives), `train_experiment.py` (the ONLY
agent-mutable file; inlines the training assembly with a local trainer),
`_sealed.py` (hash-pinned split/determinism constants), `eval_harness.py`
(frozen scorer; fixed 2024–2025 acceptance split on a seeded 64-ticker
panel, sealed 2026+ holdout), `loop.py` (torch-free deterministic runner;
owns git via a per-iteration `base_sha` anchor, timeouts, the ε-decision,
and `results.tsv`). Incorporates the 2026-07-07 adversarial-panel fixes
(empty-split blocker, commit-rc/`base_sha` git safety, ε recalibration,
representative eval panel, sealed-module guard).

**Tech Stack:** Python 3.10+, PyTorch Lightning (via existing ophir modules), `uv`, `git`, headless `claude -p`, argparse (no new dependencies).

## Global Constraints

- Tests are offline + CPU-only; never touch the network, CUDA, or the package `.ophir/` layout (repo rule). `loop.py` must import **stdlib only** so its tests stay torch-free.
- Trial artifacts must NEVER be written into `.ophir/model` — everything goes under the per-iteration `--out-dir` (that is why `train_experiment.py` builds its own trainer instead of `register.fetch_base_trainer`).
- Splits (all year bounds `max` exclusive, from `_sealed.py` only): train ≤ 2023; **acceptance val = rows in 2024–2025** (`val_min_year=2024, val_max_year=2026` — a 365-row window cannot fit in a single calendar year, so a one-year split is EMPTY); **holdout = 2026+**, `--holdout` only. The 2026 slice cannot form full windows until ~mid-2027 — until then graduation substitutes multi-seed + full-budget runs on the acceptance split, and `eval_harness.py --holdout` must fail with a message saying exactly that.
- The mutable file may contain **no year-like literals** (regex `\b20(19|2[0-9]|3[01])\b`); years come only from the pinned `_sealed.py` import.
- `EPSILON = 0.02` initial (≥ the confirm-harness 3-seed MDE of 0.0069, scaled for single-seed noise). **Before any overnight session** it must be recalibrated: ≥3 baseline seeds through the harness, `ε = 2·SE` of single-seed `rank_ic_near`. `--epsilon` overrides.
- Train time-box: `600` s per trial at `--max-steps 10000 --seed 0` (10k steps ≈ 6 min on the RTX 3090). Eval time-box `1800` s (200 batches, `num_workers=0`).
- Trial commits use `git commit --no-verify` (the repo pre-commit hook runs mypy/ruff and would reject agent-written experiment code — the loop's diff-gate is the governing check); every git/subprocess return code is checked; reverts go to a captured `base_sha`, never `HEAD~1`.
- `autoresearch/` is NOT part of the `ophir` package: outside mypy scope (`files=["src/ophir"]`) and pytest collection (`testpaths=["tests"]`). But ruff (incl. `ANN`) applies: **every function, param, and return in `autoresearch/*.py` must be annotated.** `uv run ruff check autoresearch tests && uv run ruff format --check autoresearch tests` must pass after every task.
- Conventional Commits (scope `autoresearch`); update `CHANGELOG.md` `[Unreleased]` in Task 5. Run `uv run pytest` before each commit that touches `tests/`.

---

### Task 1: `loop.py` pure helpers (decision, validation, parsing, results log)

**Files:**
- Create: `autoresearch/loop.py`
- Create: `tests/test_autoresearch_loop.py`
- Modify: `.gitignore` (append `autoresearch/runs/`)

**Interfaces:**
- Produces (used by Task 2 and its tests):
  - `decide(candidate: float | None, best: float | None, epsilon: float = EPSILON) -> bool`
  - `parse_porcelain(text: str) -> tuple[list[str], list[str]]` — `(modified_tracked, untracked)` paths
  - `diff_is_valid(modified: list[str], untracked: list[str], session_rel: str) -> bool`
  - `validate_experiment_source(text: str) -> str | None` — `None` when valid, else a reason
  - `parse_metrics(path: str) -> dict[str, float]` — reads only the named metric keys
  - `format_result_row(**fields) -> str` and `append_result(tsv_path: str, row: str) -> None`
  - Constants: `EPSILON = 0.02`, `MUTABLE_FILE = "autoresearch/train_experiment.py"`, `SEALED_IMPORT_LINE`, `YEAR_LITERAL_RE`, `PINNED_FILES`, `TRAIN_TIMEOUT_S = 600`, `EVAL_TIMEOUT_S = 1800`, `PROPOSE_TIMEOUT_S = 600`, `MAX_STEPS = 10000`, `SEED = 0`, `BASELINE_SANITY = (-0.02, 0.20)`, `RESULTS_HEADER`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_autoresearch_loop.py`. `loop.py` is not part of the `ophir` package, so load it by file path:

```python
"""Offline unit tests for the torch-free autoresearch loop runner."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "autoresearch_loop", REPO_ROOT / "autoresearch" / "loop.py"
)
assert _SPEC is not None and _SPEC.loader is not None
loop = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(loop)


class TestDecide:
    def test_first_finite_result_becomes_baseline(self) -> None:
        assert loop.decide(0.01, None) is True

    def test_nan_candidate_is_rejected(self) -> None:
        assert loop.decide(float("nan"), None) is False
        assert loop.decide(float("nan"), 0.05) is False

    def test_none_candidate_is_rejected(self) -> None:
        assert loop.decide(None, 0.05) is False

    def test_must_beat_best_by_epsilon(self) -> None:
        # Clearly inside / clearly outside the band; no exact-FP-bound asserts.
        assert loop.decide(0.0649, 0.05, epsilon=0.02) is False
        assert loop.decide(0.0751, 0.05, epsilon=0.02) is True

    def test_default_epsilon_is_applied(self) -> None:
        assert loop.decide(0.055, 0.05) is False


class TestParsePorcelain:
    def test_modified_and_untracked_are_split(self) -> None:
        text = " M autoresearch/train_experiment.py\n?? autoresearch/runs/s1/.hypothesis\n"
        modified, untracked = loop.parse_porcelain(text)
        assert modified == ["autoresearch/train_experiment.py"]
        assert untracked == ["autoresearch/runs/s1/.hypothesis"]

    def test_empty_status_is_clean(self) -> None:
        assert loop.parse_porcelain("") == ([], [])

    def test_staged_and_renamed_count_as_modified(self) -> None:
        text = "M  a.py\nR  old.py -> new.py\n"
        modified, _ = loop.parse_porcelain(text)
        assert "a.py" in modified
        assert "new.py" in modified


class TestDiffIsValid:
    SESSION = "autoresearch/runs/s1"

    def test_exactly_the_mutable_file_is_valid(self) -> None:
        assert loop.diff_is_valid([loop.MUTABLE_FILE], [], self.SESSION) is True

    def test_no_edit_is_invalid(self) -> None:
        assert loop.diff_is_valid([], [], self.SESSION) is False

    def test_touching_other_tracked_files_is_invalid(self) -> None:
        assert (
            loop.diff_is_valid([loop.MUTABLE_FILE, "src/ophir/safety.py"], [], self.SESSION)
            is False
        )

    def test_untracked_inside_session_dir_is_allowed(self) -> None:
        assert (
            loop.diff_is_valid([loop.MUTABLE_FILE], [f"{self.SESSION}/.hypothesis"], self.SESSION)
            is True
        )

    def test_untracked_outside_session_dir_is_invalid(self) -> None:
        assert loop.diff_is_valid([loop.MUTABLE_FILE], ["evil.py"], self.SESSION) is False


class TestValidateExperimentSource:
    def test_valid_source_passes(self) -> None:
        text = f"import os\n{loop.SEALED_IMPORT_LINE}\nx = 128\nhidden = 2048\n"
        assert loop.validate_experiment_source(text) is None

    def test_missing_sealed_import_is_rejected(self) -> None:
        reason = loop.validate_experiment_source("x = 1\n")
        assert reason is not None and "sealed import" in reason

    def test_year_literal_is_rejected(self) -> None:
        text = f"{loop.SEALED_IMPORT_LINE}\nval_max_year = 2025\n"
        reason = loop.validate_experiment_source(text)
        assert reason is not None and "year literal" in reason

    def test_non_year_numbers_are_fine(self) -> None:
        text = f"{loop.SEALED_IMPORT_LINE}\nemb = 2048\nlr = 0.0002\nn = 10000\n"
        assert loop.validate_experiment_source(text) is None


class TestMetricsAndResults:
    def test_parse_metrics_reads_named_keys_only(self, tmp_path: Path) -> None:
        p = tmp_path / "metrics.json"
        p.write_text('{"rank_ic_near": 0.061, "h1": 0.09, "h5": 0.05, "n": 1000, "junk": 9}')
        metrics = loop.parse_metrics(str(p))
        assert metrics["rank_ic_near"] == 0.061
        assert "junk" not in metrics

    def test_parse_metrics_tolerates_nan(self, tmp_path: Path) -> None:
        p = tmp_path / "metrics.json"
        p.write_text('{"rank_ic_near": NaN}')
        assert math.isnan(loop.parse_metrics(str(p))["rank_ic_near"])

    def test_result_row_is_tab_separated_and_sanitized(self) -> None:
        row = loop.format_result_row(
            iteration=3,
            utc="2026-07-07T05:00:00Z",
            hypothesis="try\tranking\nloss",
            status="keep",
            rank_ic_near=0.061,
            h1=0.09,
            h5=0.05,
            wall_s=412.0,
            commit="abc1234",
        )
        cells = row.split("\t")
        assert len(cells) == len(loop.RESULTS_HEADER.split("\t"))
        assert cells[2] == "try ranking loss"
        assert cells[3] == "keep"

    def test_append_result_writes_header_once(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        loop.append_result(str(tsv), "1\trow")
        loop.append_result(str(tsv), "2\trow")
        lines = tsv.read_text().splitlines()
        assert lines[0] == loop.RESULTS_HEADER
        assert lines[1:] == ["1\trow", "2\trow"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_autoresearch_loop.py -v`
Expected: FAIL at module load — `FileNotFoundError` (no `autoresearch/loop.py`).

- [ ] **Step 3: Create `autoresearch/loop.py` with the pure helpers**

```python
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

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone

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
```

(The orchestration half — `run_iteration`, `main` — is Task 2; the file is
extended there.)

- [ ] **Step 4: Append `autoresearch/runs/` to `.gitignore`**

```gitignore
autoresearch/runs/
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_autoresearch_loop.py -v`
Expected: all PASS.

- [ ] **Step 6: Lint, format, full suite**

Run: `uv run ruff check autoresearch tests && uv run ruff format --check autoresearch tests && uv run pytest`
Expected: clean. If `ruff format --check` flags files, run `uv run ruff format autoresearch tests` and re-check.

- [ ] **Step 7: Commit**

```bash
git add autoresearch/loop.py tests/test_autoresearch_loop.py .gitignore
git commit -m "feat(autoresearch): loop runner pure helpers with offline tests"
```

---

### Task 2: `loop.py` orchestration (iterations, git safety, pins, prompt, main)

**Files:**
- Modify: `autoresearch/loop.py` (append)
- Modify: `tests/test_autoresearch_loop.py` (append)

**Interfaces:**
- Consumes: every Task 1 helper and constant.
- Produces:
  - `run(cmd: list[str], *, cwd: str, timeout: float | None = None, input_text: str | None = None) -> tuple[int, str]` — default subprocess runner returning `(returncode, stdout+stderr)`; never raises (timeout → `(-1, "TIMEOUT")`).
  - `run_iteration(iteration: int, session_dir: str, best_ic: float | None, base_sha: str, *, propose: bool, epsilon: float, runner: Runner = run) -> IterationResult`
  - `IterationResult` (plain class, stdlib-only): `status: str` (`keep|discard|crash|invalid|proposer-fail`), `rank_ic_near: float | None`, `h1: float | None`, `h5: float | None`, `hypothesis: str`, `wall_s: float`
  - `build_prompt(program_text: str, results_tail: str, log_text: str, hypothesis_path: str) -> str`
  - `pin_hashes() -> dict[str, str]` and `check_pins(pins: dict[str, str]) -> list[str]`
  - `main(argv: list[str] | None = None) -> int` — flags: `--session NAME` (required), `--max-iters INT` (default 2), `--max-wall-clock-s INT` (default 28800), `--epsilon FLOAT` (default `EPSILON`)
- Task 6 runs `uv run python autoresearch/loop.py --session smoke --max-iters 2`.

Git-safety invariants (from the adversarial review — do not weaken):
1. Capture `base_sha` before anything mutates; every non-keep path ends in `git reset --hard <base_sha>` (idempotent; never `HEAD~1`).
2. Check the return code of `git commit`; a hook- or otherwise-failed commit is `invalid`, not silently "committed".
3. Trial commits pass `--no-verify`.
4. No unscoped `git clean`; stray untracked files (already outside the session dir → `invalid`) are removed individually by path.
5. `main` writes `<session>/.in-flight` (the `base_sha`) before each iteration, removes it after logging; on startup an existing `.in-flight` triggers `git reset --hard <sha>` recovery. `KeyboardInterrupt` mid-iteration also reverts before exiting.

- [ ] **Step 1: Append the failing orchestration tests**

Append to `tests/test_autoresearch_loop.py`:

```python
class FakeRunner:
    """Scripted subprocess stand-in: maps a command marker to (rc, output)."""

    def __init__(self, script: dict[str, tuple[int, str]]) -> None:
        self.script = script
        self.calls: list[list[str]] = []

    def __call__(
        self,
        cmd: list[str],
        *,
        cwd: str,
        timeout: float | None = None,
        input_text: str | None = None,
    ) -> tuple[int, str]:
        self.calls.append(cmd)
        for marker, result in self.script.items():
            if marker in " ".join(cmd):
                return result
        return (0, "")

    def commands(self, marker: str) -> list[list[str]]:
        return [c for c in self.calls if marker in " ".join(c)]


BASE_SHA = "base0000"


def _make_session(tmp_path: Path, iter_name: str, metrics: str) -> str:
    session_dir = tmp_path / "runs" / "s1"
    iter_dir = session_dir / iter_name
    iter_dir.mkdir(parents=True)
    (iter_dir / "metrics.json").write_text(metrics)
    (iter_dir / "best-step=1.ckpt").write_text("stub")
    (session_dir / ".hypothesis").write_text("wider near-band loss weighting")
    return str(session_dir)


def _experiment_file_ok(monkeypatch, tmp_path: Path) -> None:
    exp = tmp_path / "train_experiment.py"
    exp.write_text(f"{loop.SEALED_IMPORT_LINE}\n")
    monkeypatch.setattr(loop, "MUTABLE_PATH", str(exp))


class TestRunIteration:
    def _propose_runner(self, metrics_ok: bool = True) -> FakeRunner:
        return FakeRunner({"status --porcelain": (0, f" M {loop.MUTABLE_FILE}\n")})

    def test_keep_flow_commits_and_never_resets(self, tmp_path, monkeypatch) -> None:
        session_dir = _make_session(tmp_path, "iter-001", '{"rank_ic_near": 0.08}')
        _experiment_file_ok(monkeypatch, tmp_path)
        runner = self._propose_runner()
        result = loop.run_iteration(
            1, session_dir, 0.03, BASE_SHA, propose=True, epsilon=0.02, runner=runner
        )
        assert result.status == "keep"
        assert result.rank_ic_near == 0.08
        commit_cmds = runner.commands("commit")
        assert commit_cmds and "--no-verify" in commit_cmds[0]
        assert not runner.commands("reset --hard")

    def test_discard_flow_resets_to_base_sha(self, tmp_path, monkeypatch) -> None:
        session_dir = _make_session(tmp_path, "iter-001", '{"rank_ic_near": 0.031}')
        _experiment_file_ok(monkeypatch, tmp_path)
        runner = self._propose_runner()
        result = loop.run_iteration(
            1, session_dir, 0.03, BASE_SHA, propose=True, epsilon=0.02, runner=runner
        )
        assert result.status == "discard"
        resets = runner.commands("reset --hard")
        assert resets and resets[0][-1] == BASE_SHA

    def test_invalid_diff_never_trains(self, tmp_path, monkeypatch) -> None:
        session_dir = _make_session(tmp_path, "iter-001", '{"rank_ic_near": 0.08}')
        _experiment_file_ok(monkeypatch, tmp_path)
        runner = FakeRunner({"status --porcelain": (0, " M src/ophir/safety.py\n")})
        result = loop.run_iteration(
            1, session_dir, None, BASE_SHA, propose=True, epsilon=0.02, runner=runner
        )
        assert result.status == "invalid"
        assert not runner.commands("train_experiment.py --max-steps")
        assert runner.commands("reset --hard")

    def test_failed_commit_is_invalid_and_never_trains(self, tmp_path, monkeypatch) -> None:
        session_dir = _make_session(tmp_path, "iter-001", '{"rank_ic_near": 0.08}')
        _experiment_file_ok(monkeypatch, tmp_path)
        runner = FakeRunner(
            {
                "status --porcelain": (0, f" M {loop.MUTABLE_FILE}\n"),
                "commit": (1, "hook rejected"),
            }
        )
        result = loop.run_iteration(
            1, session_dir, None, BASE_SHA, propose=True, epsilon=0.02, runner=runner
        )
        assert result.status == "invalid"
        assert not runner.commands("train_experiment.py --max-steps")

    def test_proposer_failure_is_its_own_status(self, tmp_path, monkeypatch) -> None:
        session_dir = _make_session(tmp_path, "iter-001", '{"rank_ic_near": 0.08}')
        _experiment_file_ok(monkeypatch, tmp_path)
        runner = FakeRunner({"claude": (1, "not logged in")})
        result = loop.run_iteration(
            1, session_dir, None, BASE_SHA, propose=True, epsilon=0.02, runner=runner
        )
        assert result.status == "proposer-fail"
        assert not runner.commands("train_experiment.py --max-steps")

    def test_train_timeout_is_crash_and_resets(self, tmp_path, monkeypatch) -> None:
        session_dir = _make_session(tmp_path, "iter-001", '{"rank_ic_near": 0.08}')
        _experiment_file_ok(monkeypatch, tmp_path)
        runner = FakeRunner(
            {
                "status --porcelain": (0, f" M {loop.MUTABLE_FILE}\n"),
                "train_experiment.py": (-1, "TIMEOUT"),
            }
        )
        result = loop.run_iteration(
            1, session_dir, None, BASE_SHA, propose=True, epsilon=0.02, runner=runner
        )
        assert result.status == "crash"
        assert runner.commands("reset --hard")

    def test_baseline_iteration_skips_proposal_and_never_commits(
        self, tmp_path, monkeypatch
    ) -> None:
        session_dir = _make_session(tmp_path, "iter-000", '{"rank_ic_near": 0.06}')
        _experiment_file_ok(monkeypatch, tmp_path)
        runner = FakeRunner({})
        result = loop.run_iteration(
            0, session_dir, None, BASE_SHA, propose=False, epsilon=0.02, runner=runner
        )
        assert result.status == "keep"
        assert result.hypothesis == "baseline"
        assert not runner.commands("claude")
        assert not runner.commands("commit")
        assert not runner.commands("reset --hard")


class TestPromptAndPins:
    def test_prompt_carries_contract_and_context(self) -> None:
        prompt = loop.build_prompt("PROGRAM", "iter\t...", "abc fix loss", "/tmp/s/.hypothesis")
        assert "PROGRAM" in prompt
        assert loop.MUTABLE_FILE in prompt
        assert ".hypothesis" in prompt
        assert "one" in prompt.lower()

    def test_pins_detect_tampering(self, tmp_path, monkeypatch) -> None:
        target = tmp_path / "eval_harness.py"
        target.write_text("original")
        monkeypatch.setattr(loop, "PINNED_FILES", (str(target),))
        pins = loop.pin_hashes()
        assert loop.check_pins(pins) == []
        target.write_text("tampered")
        assert loop.check_pins(pins) == [str(target)]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_autoresearch_loop.py -v -k "RunIteration or PromptAndPins"`
Expected: FAIL — `AttributeError: module has no attribute 'run_iteration'` (etc.).

- [ ] **Step 3: Append the orchestration to `autoresearch/loop.py`**

```python
#: Absolute path of the mutable file; module-level so tests can repoint it.
MUTABLE_PATH = os.path.join(REPO_ROOT, MUTABLE_FILE)


def run(
    cmd: list[str],
    *,
    cwd: str,
    timeout: float | None = None,
    input_text: str | None = None,
) -> tuple[int, str]:
    """Run ``cmd``, merging stdout/stderr. Timeout → ``(-1, "TIMEOUT")``."""
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            timeout=timeout,
            input=input_text,
            capture_output=True,
            text=True,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return (-1, "TIMEOUT")
    return (proc.returncode, proc.stdout + proc.stderr)


def pin_hashes() -> dict[str, str]:
    """sha256 of every pinned harness file at session start."""
    pins: dict[str, str] = {}
    for rel in PINNED_FILES:
        path = rel if os.path.isabs(rel) else os.path.join(REPO_ROOT, rel)
        with open(path, "rb") as fh:
            pins[rel] = hashlib.sha256(fh.read()).hexdigest()
    return pins


def check_pins(pins: dict[str, str]) -> list[str]:
    """Return the pinned files whose hash changed since ``pin_hashes()``."""
    changed: list[str] = []
    for rel, expected in pins.items():
        path = rel if os.path.isabs(rel) else os.path.join(REPO_ROOT, rel)
        with open(path, "rb") as fh:
            if hashlib.sha256(fh.read()).hexdigest() != expected:
                changed.append(rel)
    return changed


def build_prompt(
    program_text: str, results_tail: str, log_text: str, hypothesis_path: str
) -> str:
    """Assemble the proposer prompt from the human program and trial history."""
    return f"""You are the proposer in an autonomous research loop improving a
stock-return forecaster. Ignore any repository-wide workflow instructions
(CLAUDE.md / AGENTS.md SDLC rules); your ONLY task is the single edit below.

Apply exactly ONE focused, well-motivated edit to {MUTABLE_FILE} (the only
file you may modify), then write a one-line hypothesis to {hypothesis_path}
(overwrite the file; a single sentence: what you changed and why it should
raise rank_ic_near).

Rules (violations are detected and the trial is discarded):
- Edit only {MUTABLE_FILE}. Do not touch any other file.
- One conceptual change per iteration; keep the file runnable.
- Never edit or remove the sealed `from _sealed import ...` line; never
  write year-like literals (split years live in _sealed.py only).
- If you add an __init__ to ExperimentPredictor it MUST call
  super().__init__ and self.save_hyperparameters(), or your checkpoint
  cannot be reloaded for scoring and the trial is wasted.
- Do not attempt to change how you are evaluated; the eval harness is
  hash-pinned.

== program.md (human search directives) ==
{program_text}

== recent trials (results.tsv tail) ==
{results_tail}

== recent kept commits ==
{log_text}
"""


class IterationResult:
    """Outcome of one loop iteration (plain class: torch-free, no deps)."""

    def __init__(
        self,
        status: str,
        rank_ic_near: float | None,
        h1: float | None,
        h5: float | None,
        hypothesis: str,
        wall_s: float,
    ) -> None:
        self.status = status
        self.rank_ic_near = rank_ic_near
        self.h1 = h1
        self.h5 = h5
        self.hypothesis = hypothesis
        self.wall_s = wall_s


def _git(runner: Runner, args: list[str]) -> tuple[int, str]:
    return runner(["git", *args], cwd=REPO_ROOT)


def _revert(runner: Runner, base_sha: str) -> None:
    """Restore the tree to the iteration's anchor (idempotent)."""
    _git(runner, ["reset", "--hard", base_sha])


def _remove_paths(paths: list[str]) -> None:
    """Delete stray untracked files the proposer left outside the session."""
    for rel in paths:
        target = os.path.join(REPO_ROOT, rel)
        if os.path.isfile(target):
            os.remove(target)


def run_iteration(
    iteration: int,
    session_dir: str,
    best_ic: float | None,
    base_sha: str,
    *,
    propose: bool,
    epsilon: float,
    runner: Runner = run,
) -> IterationResult:
    """Run one propose → commit → train → score → decide cycle."""
    start = time.monotonic()
    iter_dir = os.path.join(session_dir, f"iter-{iteration:03d}")
    os.makedirs(iter_dir, exist_ok=True)
    hypothesis_path = os.path.join(session_dir, ".hypothesis")
    hypothesis = "baseline"

    def _done(status: str, ic: float | None, h1: float | None, h5: float | None) -> IterationResult:
        return IterationResult(status, ic, h1, h5, hypothesis, time.monotonic() - start)

    if propose:
        program_text = _read(os.path.join(HARNESS_DIR, "program.md"))
        results_tail = _tail(os.path.join(session_dir, "results.tsv"), 30)
        _, log_text = _git(runner, ["log", "--oneline", "-10"])
        prompt = build_prompt(program_text, results_tail, log_text, hypothesis_path)
        rc, _out = runner(
            [
                "claude",
                "-p",
                "--model",
                PROPOSER_MODEL,
                "--allowedTools",
                "Read,Edit,Write,Grep,Glob",
                "--permission-mode",
                "acceptEdits",
                "--settings",
                os.path.join(session_dir, "settings.json"),
            ],
            cwd=REPO_ROOT,
            timeout=PROPOSE_TIMEOUT_S,
            input_text=prompt,
        )
        if rc != 0:
            _revert(runner, base_sha)
            return _done("proposer-fail", None, None, None)
        hypothesis = _read(hypothesis_path).strip() or "(no hypothesis written)"

        _, porcelain = _git(runner, ["status", "--porcelain"])
        modified, untracked = parse_porcelain(porcelain)
        session_rel = os.path.relpath(session_dir, REPO_ROOT)
        source_problem = validate_experiment_source(_read(MUTABLE_PATH))
        if not diff_is_valid(modified, untracked, session_rel) or source_problem:
            _revert(runner, base_sha)
            _remove_paths([p for p in untracked if not p.startswith(session_rel + "/")])
            return _done("invalid", None, None, None)

        _git(runner, ["add", MUTABLE_FILE])
        rc, _out = _git(runner, ["commit", "--no-verify", "-m", f"autoresearch: {hypothesis}"])
        if rc != 0:
            _revert(runner, base_sha)
            return _done("invalid", None, None, None)

    rc, _out = runner(
        [
            "uv",
            "run",
            "python",
            MUTABLE_FILE,
            "--max-steps",
            str(MAX_STEPS),
            "--seed",
            str(SEED),
            "--out-dir",
            iter_dir,
        ],
        cwd=REPO_ROOT,
        timeout=TRAIN_TIMEOUT_S,
    )
    ckpts = sorted(glob.glob(os.path.join(iter_dir, "best*.ckpt")))
    if rc != 0 or not ckpts:
        if propose:
            _revert(runner, base_sha)
        return _done("crash", None, None, None)

    metrics_path = os.path.join(iter_dir, "metrics.json")
    rc, _out = runner(
        [
            "uv",
            "run",
            "python",
            os.path.join(HARNESS_DIR, "eval_harness.py"),
            "--ckpt",
            ckpts[-1],
            "--out",
            metrics_path,
        ],
        cwd=REPO_ROOT,
        timeout=EVAL_TIMEOUT_S,
    )
    if rc != 0 or not os.path.exists(metrics_path):
        if propose:
            _revert(runner, base_sha)
        return _done("crash", None, None, None)

    metrics = parse_metrics(metrics_path)
    ic = metrics.get("rank_ic_near")
    if decide(ic, best_ic, epsilon):
        return _done("keep", ic, metrics.get("h1"), metrics.get("h5"))
    if propose:
        _revert(runner, base_sha)
    return _done("discard", ic, metrics.get("h1"), metrics.get("h5"))


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return ""


def _tail(path: str, n: int) -> str:
    lines = _read(path).splitlines()
    return "\n".join(lines[-n:])


def _head_sha() -> str:
    rc, out = run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    return out.strip() if rc == 0 else ""


def _recover_in_flight(in_flight_path: str) -> None:
    """If a previous session died mid-iteration, roll back to its anchor."""
    sha = _read(in_flight_path).strip()
    if sha:
        print(f"Recovering interrupted iteration: reset --hard {sha}")
        run(["git", "reset", "--hard", sha], cwd=REPO_ROOT)
    if os.path.exists(in_flight_path):
        os.remove(in_flight_path)


def main(argv: list[str] | None = None) -> int:
    """Session driver: baseline iteration 0, then proposal iterations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, help="session name under autoresearch/runs/")
    parser.add_argument("--max-iters", type=int, default=2)
    parser.add_argument("--max-wall-clock-s", type=int, default=28800)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    args = parser.parse_args(argv)

    session_dir = os.path.join(HARNESS_DIR, "runs", args.session)
    os.makedirs(session_dir, exist_ok=True)
    in_flight = os.path.join(session_dir, ".in-flight")
    if os.path.exists(in_flight):
        _recover_in_flight(in_flight)

    # Minimal settings so the headless proposer does not inherit repo hooks.
    settings_path = os.path.join(session_dir, "settings.json")
    with open(settings_path, "w", encoding="utf-8") as fh:
        json.dump({"hooks": {}}, fh)

    tsv = os.path.join(session_dir, "results.tsv")
    pins = pin_hashes()
    best_ic: float | None = None
    proposer_fails = 0
    started = time.monotonic()

    for i in range(args.max_iters):
        if time.monotonic() - started > args.max_wall_clock_s:
            print("Wall-clock budget exhausted; stopping.")
            break
        tampered = check_pins(pins)
        if tampered:
            print(f"ABORT: pinned harness file(s) changed: {tampered}")
            return 2

        base_sha = _head_sha()
        with open(in_flight, "w", encoding="utf-8") as fh:
            fh.write(base_sha)
        try:
            result = run_iteration(
                i, session_dir, best_ic, base_sha, propose=(i > 0), epsilon=args.epsilon
            )
        except KeyboardInterrupt:
            print("Interrupted; reverting to the iteration anchor.")
            run(["git", "reset", "--hard", base_sha], cwd=REPO_ROOT)
            os.remove(in_flight)
            return 130

        append_result(
            tsv,
            format_result_row(
                iteration=i,
                utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                hypothesis=result.hypothesis,
                status=result.status,
                rank_ic_near=result.rank_ic_near,
                h1=result.h1,
                h5=result.h5,
                wall_s=result.wall_s,
                commit=_head_sha()[:7],
            ),
        )
        os.remove(in_flight)
        print(f"iter {i}: {result.status} rank_ic_near={result.rank_ic_near} best={best_ic}")

        if result.status == "keep":
            best_ic = result.rank_ic_near
            proposer_fails = 0
        elif result.status == "proposer-fail":
            proposer_fails += 1
            if proposer_fails >= MAX_CONSECUTIVE_PROPOSER_FAILS:
                print("ABORT: proposer failed repeatedly (auth/CLI issue?).")
                return 3
        else:
            proposer_fails = 0

        if i == 0:
            ic = result.rank_ic_near
            if result.status != "keep" or ic is None:
                print("ABORT: baseline iteration failed; fix the harness before looping.")
                return 1
            low, high = BASELINE_SANITY
            if not low <= ic <= high:
                print(f"ABORT: baseline rank_ic_near {ic} outside sanity band {BASELINE_SANITY}.")
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_autoresearch_loop.py -v`
Expected: all PASS (Task 1 tests included).

- [ ] **Step 5: Lint, format, full suite**

Run: `uv run ruff check autoresearch tests && uv run ruff format --check autoresearch tests && uv run pytest`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add autoresearch/loop.py tests/test_autoresearch_loop.py
git commit -m "feat(autoresearch): iteration orchestration with base-sha git safety"
```

---

### Task 3: `_sealed.py`, `train_experiment.py` baseline, `run_loop.sh`

**Files:**
- Create: `autoresearch/_sealed.py`
- Create: `autoresearch/train_experiment.py`
- Create: `autoresearch/run_loop.sh` (mode 755)

**Interfaces:**
- Consumes: `ophir.train.build_split_handlers` / `build_dataloader` (`src/ophir/train.py:67,264`), `ophir.training_models.LightningOHLCPredictor`, `ophir.register.get_default_data_days_dir`.
- Produces (contract with `loop.py`/`eval_harness.py` — breaking any of these makes trials `crash` or `invalid`):
  - `_sealed.py` constants: `TRAIN_MIN_YEAR`, `TRAIN_MAX_YEAR`, `ACCEPT_VAL_MIN_YEAR`, `ACCEPT_VAL_MAX_YEAR`, `HOLDOUT_VAL_MIN_YEAR`, `NUM_WORKERS`
  - `train_experiment.py` CLI `--max-steps INT --seed INT --out-dir PATH` (all required)
  - exactly one `best*.ckpt` under `--out-dir` (monitor `val_rank_ic_near`, mode max)
  - module attribute `MODEL_CLASS` (a `LightningModule` subclass loadable via `MODEL_CLASS.load_from_checkpoint`)
  - the verbatim sealed import line (must match `loop.SEALED_IMPORT_LINE` byte-for-byte)

- [ ] **Step 1: Create `autoresearch/_sealed.py`**

```python
"""Sealed split and determinism constants for the autoresearch harness.

Hash-pinned by ``loop.py`` (a change aborts the session) and NEVER
agent-editable: the mutable ``train_experiment.py`` may not contain year
literals and must import these instead. Rationale:

- Train rows end before the embargo year; acceptance-val rows span
  2024–2025 (a 365-row window cannot fit in one calendar year, so a
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
```

- [ ] **Step 2: Create `autoresearch/train_experiment.py`**

```python
"""Agent-mutable training assembly for the autoresearch loop.

THE ONLY FILE THE PROPOSER AGENT MAY EDIT. Everything here — architecture,
loss, optimizer, schedule, data knobs (except the sealed import) — is the
search space. You may inline/override any ophir component (e.g. subclass and
reimplement ``compute_loss``) as long as the harness contract holds:

Contract (loop.py / eval_harness.py depend on it; breaking it wastes the trial):
- CLI: ``--max-steps``, ``--seed``, ``--out-dir``.
- Writes exactly one ``best*.ckpt`` under ``--out-dir``.
- Exposes ``MODEL_CLASS`` for ``eval_harness.py`` to load the checkpoint.
  If you give ``ExperimentPredictor`` its own ``__init__`` it MUST call
  ``super().__init__(...)`` and ``self.save_hyperparameters()`` or the
  checkpoint cannot be reloaded for scoring.
- The sealed import line below stays verbatim; no year literals anywhere.

Requires CUDA (same runtime constraint as ``ophir train``).
"""

from __future__ import annotations

import argparse
import os
from typing import TYPE_CHECKING

from _sealed import ACCEPT_VAL_MAX_YEAR, ACCEPT_VAL_MIN_YEAR, NUM_WORKERS, TRAIN_MAX_YEAR  # SEALED: loop.py enforces this line verbatim

from ophir.training_models import LightningOHLCPredictor

if TYPE_CHECKING:
    import lightning as L


class ExperimentPredictor(LightningOHLCPredictor):
    """The experiment surface: override loss/optimizer/model pieces here."""


MODEL_CLASS = ExperimentPredictor

# ---- experiment configuration (agent-editable) -----------------------------
MODEL_KWARGS: dict[str, object] = {
    "emb_dim": 128,
    "num_layers": 6,
    "num_heads": 8,
    "lr": 2e-4,
    "rezero_lr": 3e-4,
    "weight_decay": 0.01,
    "betas": (0.9, 0.95),
    "warmup_ratio": 0.03,
    "loss_decay": 0.6,
    "close_weight": 1.0,
    "upside_weight": 0.5,
    "downside_weight": 0.5,
    "rezero_init": 0.0,
}
SEQ_LEN = 365
WINDOW_OFFSET = 90
RESPONSE_SIZE = 90
BATCH_SIZE = 32
CACHE_SIZE = 8
MIN_VOLUME = 1000.0
VAL_EVERY_STEPS = 500
VAL_BATCHES = 50
# -----------------------------------------------------------------------------


def build_trainer(out_dir: str, max_steps: int) -> L.Trainer:
    """Local trainer writing all artifacts under ``out_dir``.

    Deliberately NOT ``register.fetch_base_trainer``: trials must never write
    into the real ``.ophir/model`` registry.
    """
    import lightning as L  # noqa: PLC0415 — heavy import deferred past --help
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

    best = ModelCheckpoint(
        monitor="val_rank_ic_near",
        mode="max",
        dirpath=out_dir,
        filename="best-{step}-{val_rank_ic_near:.5f}",
        save_top_k=1,
        save_on_train_epoch_end=False,
    )
    return L.Trainer(
        max_steps=max_steps,
        precision="bf16-mixed",
        default_root_dir=out_dir,
        accelerator="cuda",
        callbacks=[best, LearningRateMonitor("step")],
        logger=CSVLogger(out_dir, name="csv"),
        val_check_interval=VAL_EVERY_STEPS,
        check_val_every_n_epoch=None,
        limit_val_batches=VAL_BATCHES,
        gradient_clip_val=1,
        gradient_clip_algorithm="norm",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    import lightning as L
    import torch

    from ophir import register
    from ophir.train import build_dataloader, build_split_handlers

    torch.set_float32_matmul_precision("high")
    L.seed_everything(args.seed, workers=True)

    base_path = os.path.join(register.get_default_data_days_dir(), "stocks")
    train_handler, val_handler = build_split_handlers(
        base_path=base_path,
        seq_len=SEQ_LEN,
        offset=WINDOW_OFFSET,
        min_volume=MIN_VOLUME,
        train_min_year=None,
        train_max_year=TRAIN_MAX_YEAR,
        val_min_year=ACCEPT_VAL_MIN_YEAR,
        val_max_year=ACCEPT_VAL_MAX_YEAR,
        use_sp500=False,
    )
    train_dl = build_dataloader(train_handler, RESPONSE_SIZE, BATCH_SIZE, NUM_WORKERS, CACHE_SIZE)
    val_dl = build_dataloader(
        val_handler, RESPONSE_SIZE, BATCH_SIZE, NUM_WORKERS, CACHE_SIZE, return_identity=True
    )

    model = MODEL_CLASS(max_steps=args.max_steps, **MODEL_KWARGS)
    trainer = build_trainer(args.out_dir, args.max_steps)
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    main()
```

Note: `_sealed` resolves because the script runs as `python autoresearch/train_experiment.py` (its own directory heads `sys.path`). If ruff objects to the long sealed-import line (`E501`), append `  # noqa: E501` INSIDE the plan's `SEALED_IMPORT_LINE` constant too — the two must stay byte-identical; prefer keeping the line under the limit by shortening the comment consistently in both places.

- [ ] **Step 3: Create `autoresearch/run_loop.sh`**

```bash
#!/usr/bin/env bash
# Thin entry for the autoresearch loop: refuses main/detached-HEAD and a
# dirty tree, then hands off to the deterministic runner. Usage:
# ./autoresearch/run_loop.sh --session smoke --max-iters 2 [--epsilon 0.02]
set -euo pipefail
cd "$(dirname "$0")/.."

branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$branch" == "main" || "$branch" == "HEAD" ]]; then
    echo "refusing to run the loop on main or a detached HEAD" >&2
    exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "working tree not clean; commit or stash first" >&2
    exit 1
fi

exec uv run python autoresearch/loop.py "$@"
```

Then: `chmod +x autoresearch/run_loop.sh`

- [ ] **Step 4: Verify the experiment file parses, its CLI works (CPU-safe), and the sentinel matches byte-for-byte**

Run: `uv run python autoresearch/train_experiment.py --help`
Expected: argparse usage text (imports ophir/torch on CPU; CUDA is lazy, not touched by `--help`).

Run: `uv run python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('l', 'autoresearch/loop.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
text = open('autoresearch/train_experiment.py').read()
problem = m.validate_experiment_source(text)
assert problem is None, problem
print('sealed import + year-literal check OK')"`
Expected: `sealed import + year-literal check OK`.

- [ ] **Step 5: Lint, format, full suite**

Run: `uv run ruff check autoresearch && uv run ruff format --check autoresearch && uv run pytest`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add autoresearch/_sealed.py autoresearch/train_experiment.py autoresearch/run_loop.sh
git commit -m "feat(autoresearch): sealed split module, baseline experiment, loop entry"
```

---

### Task 4: `eval_harness.py` (immutable scorer)

**Files:**
- Create: `autoresearch/eval_harness.py`

**Interfaces:**
- Consumes: `MODEL_CLASS` from `autoresearch/train_experiment.py`; `_sealed` constants; `ophir.evaluate.evaluate_model` (`src/ophir/evaluate.py:465`); `ophir.train.build_split_handlers`/`build_dataloader`.
- Produces: CLI `--ckpt PATH --out PATH [--holdout]`; JSON at `--out` with keys `rank_ic_near`, `rank_ic_mean`, `n`, and the `h*` curve. Exit codes: `0` ok, `3` too-few-rows (incl. not-yet-scoreable holdout), `4` checkpoint-load failure. `loop.py` consumes `rank_ic_near`, `h1`, `h5`.

- [ ] **Step 1: Create `autoresearch/eval_harness.py`**

```python
"""Immutable trial scorer for the autoresearch loop.

FROZEN during a session — ``loop.py`` sha256-pins this file (and
``_sealed.py``) and aborts the session on any change. It owns the acceptance
measurement: the sealed split, a fixed seeded ticker panel, the
deterministic batch subset, and the ``rank_ic_near`` math (reused verbatim
from ``ophir.evaluate`` so the loop, the eval report, and training logs
agree).

Splits (from ``_sealed.py``): acceptance = rows 2024–2025; ``--holdout`` =
rows 2026+ — the sealed slice, ONLY for graduating a champion. A 365-row
window needs ~1.5y of rows, so the holdout cannot produce windows until
~mid-2027; until then this command exits 3 with an explanatory message and
graduation relies on multi-seed + full-budget runs on the acceptance split.

Determinism/representativeness: seeds everything, uses ``num_workers=0``,
and restricts the universe to a fixed seeded panel of tickers so the scored
subset is identical across trials and machines (the raw handler order is
filesystem-dependent and front-loaded). Requires CUDA.
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
    import lightning as L
    import torch
    from _sealed import (
        ACCEPT_VAL_MAX_YEAR,
        ACCEPT_VAL_MIN_YEAR,
        HOLDOUT_VAL_MIN_YEAR,
        TRAIN_MAX_YEAR,
    )
    from train_experiment import MODEL_CLASS

    from ophir import register
    from ophir.evaluate import evaluate_model
    from ophir.train import build_dataloader, build_split_handlers

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
    panel = random.Random(EVAL_SEED).sample(stocks, min(PANEL_SIZE, len(stocks)))
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
    except Exception as exc:  # noqa: BLE001 — any load failure must be legible
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
            " (expected before ~mid-2027: a 365-row window needs ~1.5y of "
            "post-embargo rows)"
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
```

- [ ] **Step 2: Verify it parses and the CLI works (CPU-safe)**

Run: `uv run python autoresearch/eval_harness.py --help`
Expected: argparse usage text.

- [ ] **Step 3: Lint, format**

Run: `uv run ruff check autoresearch && uv run ruff format --check autoresearch`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add autoresearch/eval_harness.py
git commit -m "feat(autoresearch): immutable rank_ic_near eval harness (seeded panel, sealed holdout)"
```

---

### Task 5: `program.md` + CHANGELOG + spec amendments

**Files:**
- Create: `autoresearch/program.md`
- Modify: `CHANGELOG.md` (`[Unreleased]` → `### Added`)
- Modify: `docs/superpowers/specs/2026-07-07-autoresearch-forecaster-harness-design.md` (graduation-bar amendments from the adversarial panel)

**Interfaces:**
- Consumes: nothing (prose). `loop.py.build_prompt` embeds `program.md` verbatim.

- [ ] **Step 1: Create `autoresearch/program.md`**

```markdown
# program.md — human search directives for the autoresearch loop

You (the proposer) improve a BERT-style masked transformer that forecasts
three forward OHLC targets per day. **Objective: maximize `rank_ic_near`** —
pooled daily cross-sectional Spearman rank-IC of `r_close` at trading-day
offsets 1–5, on a fixed held-out split. Current baseline is iteration 0 of
`results.tsv`; the multi-seed reference point is ≈ +0.066.

## Ground rules

- One focused change per iteration. Keep `train_experiment.py` runnable and
  self-consistent; your run is killed at 10 minutes.
- You may inline any ophir component into `train_experiment.py` (e.g. copy a
  method into `ExperimentPredictor` and modify it) — but never edit files
  under `src/ophir`.
- If you give `ExperimentPredictor` its own `__init__`, it MUST call
  `super().__init__(...)` and `self.save_hyperparameters()`, or your
  checkpoint cannot be reloaded for scoring and the trial is wasted.
- Never touch the sealed `from _sealed import ...` line; never write
  year-like literals (the split lives in the pinned `_sealed.py`).
- Simplicity rule: a marginal gain does not justify added complexity. On a
  near-tie, prefer the simpler variant. Reverting a kept-but-marginal
  complexity increase is a valid proposal.
- Every feature must be knowable strictly before the prediction timestamp.
  Never introduce anything that peeks into the response block.

## Known results (do not re-litigate)

- `rezero_lr` dominates hyperparameter importance; `lr`, `loss_decay` matter.
- `rezero_init` tuning does NOT help (multi-seed confirmed). Do not tune it.
- Skill concentrates at offsets 1–5 and dies by offset ~10; the pooled
  90-day objective dilutes it. That is WHY the metric is `rank_ic_near`.
- Plain hyperparameter grid-walking is the Optuna sweep's job, not yours —
  only propose a hyperparameter change with a mechanistic rationale.

## Promising directions (highest leverage first)

1. **Near-horizon loss shaping.** The loss's time-decay (`loss_decay`)
   currently spreads weight across all 90 response days. Concentrate
   training signal on offsets 1–5 (steeper decay, truncated weighting, or a
   dedicated near-band loss term).
2. **Rank the cross-section, don't regress it.** Add a pairwise/listwise
   ranking term on `r_close` within each day's cross-section — the decision
   is "long the top names", so ranking loss aligns training with use.
3. **Response-block framing.** A shorter effective horizon (smaller
   `RESPONSE_SIZE`, keeping eval offsets 1–5 intact) may stop far-horizon
   noise from dominating gradients.
4. **Feature-side ideas** with strict causal lagging (e.g. volatility
   normalization of returns before embedding).
5. Architecture changes last — the evidence says the ceiling is framing,
   not capacity.

## Measurement honesty (why some wins don't count)

- Acceptance needs `rank_ic_near > best + ε` (ε set by the runner; your
  10k-step single-seed measurement is noisy — most true small gains will
  not clear it, and that is intentional).
- A 10k-step win can be a proxy artifact; champions face multi-seed and
  full-budget re-runs at graduation. Prefer changes with a mechanism, not
  a lucky number.

## When stuck

If 3+ consecutive proposals are discarded, switch families (e.g. from loss
shaping to ranking) rather than iterating on the failed idea; consider a
revert-to-simpler proposal if recent kept changes look like noise.
```

- [ ] **Step 2: Update `CHANGELOG.md`**

Read the current `[Unreleased]` section and add under `### Added` (create the subsection if absent):

```markdown
- Autoresearch harness (`autoresearch/`): autonomous edit → time-boxed train →
  `rank_ic_near` eval → keep-or-revert loop with per-trial logging, sealed
  holdout split, and hash-pinned eval harness.
```

- [ ] **Step 3: Amend the design spec's honesty/graduation sections**

In `docs/superpowers/specs/2026-07-07-autoresearch-forecaster-harness-design.md`, apply the adversarial-panel corrections (Edit, keeping the rest intact):

1. In "Deliberate departures" item 2, replace the ε sentence with: ε starts
   at 0.02 (above the confirm-harness 3-seed MDE of 0.0069) and MUST be
   recalibrated from ≥3 baseline seeds (`ε = 2·SE`) before any unattended
   session.
2. In item 3 (sealed holdout), append: the 2026 slice cannot form full
   365-row windows until ~mid-2027; until then it stays sealed and
   graduation substitutes the item-4 bar below.
3. Replace item 4 (graduation bar) with: a champion must (a) survive
   multi-seed (0/1/2) 10k-step re-runs, (b) survive at least one
   full-budget (~100k-step) run clearing the incumbent on the acceptance
   split, (c) be scored on the sealed holdout once scoreable, and (d) report
   a deflated significance whose trial count N includes all loop sessions
   (`results.tsv`, every status) AND the Optuna sweep trials on the same
   objective.
4. In "Layout", add `_sealed.py` (pinned split/determinism constants) to
   the table, and change the sentinel-line mechanism description to the
   sealed-import + no-year-literals check.

- [ ] **Step 4: Commit**

```bash
git add autoresearch/program.md CHANGELOG.md docs/superpowers/specs/2026-07-07-autoresearch-forecaster-harness-design.md
git commit -m "docs(autoresearch): search program, changelog, spec graduation-bar amendments"
```

---

### Task 6: Supervised smoke trial (GPU)

**Files:**
- No new source files. Produces `autoresearch/runs/smoke/results.tsv` (gitignored) and a session report.

**Interfaces:**
- Consumes: everything above, the local RTX 3090, and data under `.ophir/data/days/stocks`.

This task is run inline by the coordinating session (NOT a subagent — it
needs the GPU, ~30 minutes, and human-visible output).

- [ ] **Step 1: Preflight**

Run: `nvidia-smi --query-gpu=name,memory.used --format=csv && git status --porcelain && git branch --show-current`
Expected: 3090 mostly idle; clean tree; branch `feat/autoresearch-harness` (not main).

- [ ] **Step 2: Split preflight (cheap, catches the empty-split class of bug before GPU time)**

Run: `uv run python -c "
import os
from ophir import register
from ophir.train import build_split_handlers
base = os.path.join(register.get_default_data_days_dir(), 'stocks')
_, vh = build_split_handlers(base_path=base, seq_len=365, offset=90, min_volume=1000.0,
    train_min_year=None, train_max_year=2023, val_min_year=2024, val_max_year=2026,
    use_sp500=False)
s = vh.stocks[0]
print('val stocks:', len(vh.stocks), 'first-stock windows:', vh.stock_streamer(s).size)"`
Expected: `val stocks:` in the hundreds+, `first-stock windows:` > 0. Zero windows means the acceptance split is broken — stop and fix before any training.

- [ ] **Step 3: Baseline-only iteration first (fastest failure isolation)**

Run: `./autoresearch/run_loop.sh --session smoke --max-iters 1`
Expected: `iter 0: keep rank_ic_near=<finite float> best=None`; exit 0;
`autoresearch/runs/smoke/results.tsv` has the header + one `keep` row with a
finite `rank_ic_near` inside the sanity band (roughly +0.02 to +0.10 per the
confirm-harness evidence); `autoresearch/runs/smoke/iter-000/` contains one
`best-*.ckpt` and `metrics.json`.

If iteration 0 crashes: debug directly (`uv run python
autoresearch/train_experiment.py --max-steps 200 --seed 0 --out-dir
/tmp/ar-debug` is a ~30 s probe; then run eval_harness on its checkpoint by
hand), fix, commit the fix, re-run. Common suspects: a
`LightningOHLCPredictor` kwarg mismatch, no `best*.ckpt` because
`val_rank_ic_near` was not logged (identity missing), eval exit 3 (panel too
thin — raise `PANEL_SIZE`/`VAL_BATCHES`), or checkpoint-glob mismatch.

- [ ] **Step 4: Full smoke: baseline + one proposal**

Run: `rm -rf autoresearch/runs/smoke && ./autoresearch/run_loop.sh --session smoke --max-iters 2`
Expected: two result rows. Row 0 `keep` (baseline). Row 1 is a real
proposal: verify (a) the hypothesis column is a meaningful sentence, (b) if
`keep` → a new commit `autoresearch: <hypothesis>` exists at HEAD; if
`discard`/`crash`/`invalid`/`proposer-fail` → `git rev-parse HEAD` equals
the pre-session HEAD and `git status --porcelain` is empty.

- [ ] **Step 5: Verify the audit chain and registry isolation**

Run: `git log --oneline -5 && cat autoresearch/runs/smoke/results.tsv && ls -la src/ophir/.ophir/model | head`
Expected: log and TSV tell the same story (every trial has a row; only kept
trials have commits); no new/modified files in the real model registry
(mtimes predate the session).

- [ ] **Step 6: Commit the session record**

```bash
git add -f autoresearch/runs/smoke/results.tsv
git commit -m "chore(autoresearch): smoke-trial session record"
```

(`-f` because `autoresearch/runs/` is gitignored; the smoke record is the
one exception worth keeping — it is the harness's acceptance evidence.)

---

## Verification checklist (end of plan)

- [ ] `uv run pytest` — full suite green (offline, CPU-only).
- [ ] `uv run ruff check . && uv run ruff format --check .` — clean.
- [ ] `uv run mypy src/ophir` — clean (autoresearch/ intentionally out of scope).
- [ ] Smoke trial ran: results.tsv rows match git history; no writes under `src/ophir/.ophir/model`.
- [ ] The keep-or-revert invariant holds: HEAD is either the baseline or a chain of `autoresearch:` commits, each with a matching `keep` row.
- [ ] Follow-up recorded (not this delivery): ε recalibration from ≥3 baseline seeds before the first unattended session; proxy-fidelity check (10k vs 100k rank correlation) before trusting long sessions.
