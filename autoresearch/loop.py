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
import ast
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

#: Environment files pinned by content-or-absence: if any of these appear,
#: disappear, or change mid-session the harness aborts (they can steer the
#: headless proposer or the loop's own behavior). Missing files are pinned to
#: the sentinel ``"ABSENT"`` so an added file also trips the check.
PINNED_ENV_FILES = (
    ".claude/settings.json",
    ".claude/settings.local.json",
    "CLAUDE.md",
    "AGENTS.md",
)

#: Sentinel hash recorded for a pinned file that does not exist at pin time.
ABSENT_PIN = "ABSENT"

#: The sealed constant names ``_sealed.py`` owns. The mutable file may *import*
#: them but must never rebind them (an assignment target) — rebinding is a
#: split/determinism leakage vector — and ``*_year`` call keywords may only be
#: one of these names or ``None``.
SEALED_NAMES = frozenset(
    {"ACCEPT_VAL_MAX_YEAR", "ACCEPT_VAL_MIN_YEAR", "NUM_WORKERS", "TRAIN_MAX_YEAR"}
)

#: Git ``core.hooksPath`` used for every loop git call, set in ``main()`` to an
#: empty per-session dir so no repo/user hook fires during a trial. ``None``
#: (the default) means plain ``git`` — unit tests stay unaffected unless they
#: opt in by setting this.
HOOKS_PATH: str | None = None

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


def _is_sealed_name_or_none(value: ast.expr) -> bool:
    """True when ``value`` is a bare sealed ``Name`` or the constant ``None``."""
    if isinstance(value, ast.Name) and value.id in SEALED_NAMES:
        return True
    return isinstance(value, ast.Constant) and value.value is None


def _sealed_binding_problem(node: ast.AST) -> str | None:
    """Reject any binding form that would rebind a sealed name.

    Covers ``Name`` in a non-Load context (plain/aug/ann assignments, for
    targets, with-as, comprehension targets, walrus, del), function parameters
    (``ast.arg``, incl. lambda/posonly/kwonly), import aliases (``asname`` if
    set, else the imported name), ``except ... as`` names, and
    ``global``/``nonlocal`` declarations. The verbatim
    ``from _sealed import <name>`` bindings (no ``as``) are the one allowed
    form — that IS the sealed import.
    """

    def _rebound(name: str) -> str:
        return f"sealed name {name!r} rebound (split constants live in _sealed.py only)"

    if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
        if node.id in SEALED_NAMES:
            return _rebound(node.id)
    elif isinstance(node, ast.arg):
        if node.arg in SEALED_NAMES:
            return _rebound(node.arg)
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        for alias in node.names:
            bound = alias.asname or alias.name.split(".")[0]
            if bound not in SEALED_NAMES:
                continue
            allowed = (
                isinstance(node, ast.ImportFrom)
                and node.module == "_sealed"
                and alias.asname is None
            )
            if not allowed:
                return _rebound(bound)
    elif isinstance(node, ast.ExceptHandler):
        if node.name is not None and node.name in SEALED_NAMES:
            return _rebound(node.name)
    elif isinstance(node, (ast.Global, ast.Nonlocal)):
        for name in node.names:
            if name in SEALED_NAMES:
                return _rebound(name)
    return None


def _splat_dict_problem(d: ast.Dict) -> str | None:
    """Check a dict literal used as ``**`` kwargs: keys must be verifiable.

    Every key must be a string constant (a non-constant key could smuggle a
    ``*_year`` keyword past the gate), and any ``*_year`` key's value must be
    a sealed ``Name`` or ``None`` — same rule as an explicit keyword.
    """
    for key, value in zip(d.keys, d.values, strict=True):
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            return "unverifiable splat key (dict keys in **kwargs must be string literals)"
        if key.value.endswith("_year") and not _is_sealed_name_or_none(value):
            return (
                f"'{key.value}' keyword must be a sealed name or None "
                "(year values live in _sealed.py only)"
            )
    return None


def _collect_name_bindings(tree: ast.AST) -> dict[str, list[ast.expr]]:
    """Map each statically-resolvable name to the values it is assigned.

    A name is resolvable only when EVERY binding of it in the file is a plain
    ``Assign``/``AnnAssign`` with a bare ``Name`` target. Names also bound any
    other way (for targets, walrus, parameters, import aliases, ``except as``,
    tuple unpacking, ...) are dropped, so a ``**`` splat of them is treated as
    unverifiable rather than trusted.
    """
    resolvable: dict[str, list[ast.expr]] = {}
    resolvable_count: dict[str, int] = {}
    bound_count: dict[str, int] = {}

    def _bump(counts: dict[str, int], name: str) -> None:
        counts[name] = counts.get(name, 0) + 1

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    resolvable.setdefault(target.id, []).append(node.value)
                    _bump(resolvable_count, target.id)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            resolvable.setdefault(node.target.id, []).append(node.value)
            _bump(resolvable_count, node.target.id)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            _bump(bound_count, node.id)
        elif isinstance(node, ast.arg):
            _bump(bound_count, node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                _bump(bound_count, alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name is not None:
            _bump(bound_count, node.name)

    for name in list(resolvable):
        if bound_count.get(name, 0) != resolvable_count.get(name, 0):
            del resolvable[name]
    return resolvable


def _call_kwargs_problem(node: ast.Call, resolvable: dict[str, list[ast.expr]]) -> str | None:
    """Enforce the ``*_year`` keyword rule on a call, including ``**`` splats."""
    for kw in node.keywords:
        if kw.arg is not None:
            if kw.arg.endswith("_year") and not _is_sealed_name_or_none(kw.value):
                return (
                    f"'{kw.arg}' keyword must be a sealed name or None "
                    "(year values live in _sealed.py only)"
                )
            continue
        # ``**`` splat: its contents must be statically verifiable.
        value = kw.value
        if isinstance(value, ast.Dict):
            problem = _splat_dict_problem(value)
            if problem:
                return problem
            continue
        if isinstance(value, ast.Name):
            dicts = resolvable.get(value.id)
            if dicts is None:
                return f"unverifiable splat (**{value.id} is not a plainly-assigned dict literal)"
            for d in dicts:
                if not isinstance(d, ast.Dict):
                    return f"unverifiable splat (**{value.id} has a non-dict-literal binding)"
                problem = _splat_dict_problem(d)
                if problem:
                    return problem
            continue
        return "unverifiable splat (**kwargs must be a dict literal or a plainly-assigned name)"
    return None


def validate_experiment_source(text: str) -> str | None:
    """Check post-edit invariants of the mutable file; ``None`` when valid.

    Layered defense, cheapest first: the sealed import line must survive
    verbatim, no year-like literal may appear outside it, and — via an AST
    parse — no sealed name may be rebound through ANY binding form and every
    ``*_year`` call keyword (explicit or smuggled through a ``**`` splat)
    must be a sealed name or ``None``. An unparseable file is rejected.

    This gate is defense against honest mistakes and cheap gaming, not a
    sandbox: integrity against a determined adversary rests on human diff
    review of the kept commits at graduation.
    """
    if SEALED_IMPORT_LINE not in text:
        return "sealed import line missing (_sealed constants were bypassed)"
    stripped = text.replace(SEALED_IMPORT_LINE, "")
    match = YEAR_LITERAL_RE.search(stripped)
    if match:
        return f"year literal {match.group(0)!r} found (split years live in _sealed.py only)"

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return "unparseable"

    resolvable = _collect_name_bindings(tree)
    for node in ast.walk(tree):
        problem = _sealed_binding_problem(node)
        if problem:
            return problem
        if isinstance(node, ast.Call):
            problem = _call_kwargs_problem(node, resolvable)
            if problem:
                return problem
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


def pin_hashes(
    rels: tuple[str, ...] | None = None, *, allow_absent: bool = False
) -> dict[str, str]:
    """sha256 of each pinned path at session start.

    ``rels`` defaults to :data:`PINNED_FILES` (resolved at call time so tests
    may monkeypatch it). When ``allow_absent`` is set a missing file is
    recorded as the sentinel :data:`ABSENT_PIN` instead of raising, so a later
    *appearance* of the file counts as a change (used for the environment files
    in :data:`PINNED_ENV_FILES`). With ``allow_absent`` false a missing file is
    a programming error and propagates.
    """
    if rels is None:
        rels = PINNED_FILES
    pins: dict[str, str] = {}
    for rel in rels:
        path = rel if os.path.isabs(rel) else os.path.join(REPO_ROOT, rel)
        if allow_absent and not os.path.exists(path):
            pins[rel] = ABSENT_PIN
            continue
        with open(path, "rb") as fh:
            pins[rel] = hashlib.sha256(fh.read()).hexdigest()
    return pins


def check_pins(pins: dict[str, str]) -> list[str]:
    """Return the pinned paths that changed since ``pin_hashes()``.

    Flags content changes and, for :data:`ABSENT_PIN`-pinned paths, the file
    appearing (or, for a content pin, the file disappearing).
    """
    changed: list[str] = []
    for rel, expected in pins.items():
        path = rel if os.path.isabs(rel) else os.path.join(REPO_ROOT, rel)
        if not os.path.exists(path):
            if expected != ABSENT_PIN:
                changed.append(rel)
            continue
        with open(path, "rb") as fh:
            current = hashlib.sha256(fh.read()).hexdigest()
        if current != expected:
            changed.append(rel)
    return changed


def build_prompt(program_text: str, results_tail: str, log_text: str, hypothesis_path: str) -> str:
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
    """Run git through the runner, forcing the session's empty hooksPath.

    When :data:`HOOKS_PATH` is set (in ``main()``) every git invocation carries
    ``-c core.hooksPath=<dir>`` so no repo or user hook can fire mid-trial;
    when ``None`` (the unit-test default) plain ``git`` is used.
    """
    prefix = ["git"]
    if HOOKS_PATH is not None:
        prefix = ["git", "-c", f"core.hooksPath={HOOKS_PATH}"]
    return runner([*prefix, *args], cwd=REPO_ROOT)


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
    rc, out = _git(run, ["rev-parse", "HEAD"])
    return out.strip() if rc == 0 else ""


def _recover_all_in_flight() -> None:
    """Recover every interrupted iteration across all sessions before starting.

    A live iteration writes its anchor sha to ``<session>/.in-flight`` and
    removes it on completion, so any surviving marker means a runner died
    mid-trial (this session or a previous one). For each: reset ``--hard`` to
    the recorded sha, drop the marker, and append a ``runner-died`` row to
    *that* session's ``results.tsv`` (metrics nan; commit = post-reset HEAD)
    so the record reflects the interrupted trial.
    """
    for in_flight_path in sorted(glob.glob(os.path.join(HARNESS_DIR, "runs", "*", ".in-flight"))):
        sha = _read(in_flight_path).strip()
        if sha:
            print(f"Recovering interrupted iteration: reset --hard {sha}")
            _git(run, ["reset", "--hard", sha])
        if os.path.exists(in_flight_path):
            os.remove(in_flight_path)
        session_dir = os.path.dirname(in_flight_path)
        nan = float("nan")
        append_result(
            os.path.join(session_dir, "results.tsv"),
            format_result_row(
                iteration=-1,
                # UP017 suppressed: datetime.UTC needs 3.11+; runtime floor is 3.10.
                utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),  # noqa: UP017
                hypothesis="(recovered interrupted iteration)",
                status="runner-died",
                rank_ic_near=nan,
                h1=nan,
                h5=nan,
                wall_s=nan,
                commit=_head_sha()[:7],
            ),
        )


def main(argv: list[str] | None = None) -> int:
    """Session driver: baseline iteration 0, then proposal iterations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, help="session name under autoresearch/runs/")
    parser.add_argument("--max-iters", type=int, default=2)
    parser.add_argument("--max-wall-clock-s", type=int, default=28800)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    args = parser.parse_args(argv)

    global HOOKS_PATH

    session_dir = os.path.join(HARNESS_DIR, "runs", args.session)
    os.makedirs(session_dir, exist_ok=True)

    # Empty per-session hooks dir forced onto every loop git call, so no repo or
    # user git hook can fire during a trial.
    hooks_path = os.path.join(session_dir, "hooks")
    os.makedirs(hooks_path, exist_ok=True)
    HOOKS_PATH = hooks_path

    tsv = os.path.join(session_dir, "results.tsv")
    # The session's own trial log must not be git-tracked: a tracked results.tsv
    # inside the (gitignored) runs/ tree would make porcelain report it modified
    # on every append (poisoning diff validation) and reset --hard would erase
    # appended rows. Abort before touching anything.
    if _git(run, ["ls-files", "--error-unmatch", tsv])[0] == 0:
        print(f"ABORT: session results.tsv is git-tracked: {tsv}")
        return 4

    # Recover any interrupted iteration across all sessions (this one included)
    # before starting: reset to its anchor and log a runner-died row.
    _recover_all_in_flight()
    in_flight = os.path.join(session_dir, ".in-flight")

    # Minimal settings so the headless proposer does not inherit repo hooks.
    settings_path = os.path.join(session_dir, "settings.json")
    with open(settings_path, "w", encoding="utf-8") as fh:
        json.dump({"hooks": {}}, fh)

    pins = pin_hashes()
    pins.update(pin_hashes(PINNED_ENV_FILES, allow_absent=True))
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
            _git(run, ["reset", "--hard", base_sha])
            os.remove(in_flight)
            return 130

        append_result(
            tsv,
            format_result_row(
                iteration=i,
                # UP017 suppressed: datetime.UTC needs 3.11+; runtime floor is 3.10.
                utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),  # noqa: UP017
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
