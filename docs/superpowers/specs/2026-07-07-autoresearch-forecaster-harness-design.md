# Autoresearch forecaster harness — design

**Date:** 2026-07-07
**Branch:** `feat/autoresearch-harness`
**Status:** approved design; implementation plan to follow.

## Background

Adapts Karpathy's autoresearch pattern (spec/script split, single mutable
file, immutable eval harness, time-boxed keep-or-revert-with-git loop, headless
Claude Code as the agent) to ophir's forecaster, per the
`autoresearch-best-practices` OKF bundle
(github.com/kwcantrell/okf-bundles) and the `quant-trading` skill.

Why this fits now: the blessed near-horizon operating point is shipped —
training logs and the eval report agree on `val_rank_ic_near` /
`rank_ic_near` (trading-day offsets 1–5, `_NEAR_OFFSET_K = 5`) — and short
seeded runs are cheap on the local RTX 3090 (`--max-steps 10000` ≈ 6 min,
validated by the confirm harness). The Optuna sweep already covers pure
hyperparameter search, so the loop's unique value is **code-level
experiments** — loss shaping, ranking objectives, feature and architecture
edits — that a fixed sweep space cannot reach.

The loop produces **candidates with evidence, not changes to `src/ophir`**.
Graduation into the package remains a human-reviewed spec/PR step.

## Layout

New top-level directory `autoresearch/` (peer of `scripts/`; not part of the
`ophir` package, excluded from strict-mypy scope):

| File | Role | Mutability |
| --- | --- | --- |
| `program.md` | Search directives: objective, ideas menu, known dead ends, simplicity rule | Human only |
| `train_experiment.py` | Self-contained training assembly (model + loss + training loop; imports ophir data plumbing). CLI: `--max-steps --seed --out-dir` | **Agent only — the sole mutable file** |
| `eval_harness.py` | Loads the trial checkpoint, computes `rank_ic_near` + per-offset curve (`h1…h90`) on the fixed val split, emits JSON | Frozen during a session (sha256-pinned by the runner) |
| `_sealed.py` | Pinned split/determinism constants (train/accept-val/holdout year bounds, `NUM_WORKERS`) | Frozen during a session (sha256-pinned by the runner); never agent-editable |
| `loop.py` | Outer loop: invoke proposer → commit → timed run → eval → keep/revert → log. Torch-free, unit-testable | Frozen during a session |
| `run_loop.sh` | Thin entry: worktree/branch setup, session dir, invokes `loop.py` | Frozen during a session |
| `runs/<session>/results.tsv` | Append-only trial log | Runner only; gitignored during runs, committed at session end |

`train_experiment.py` v0 (the baseline) reproduces the current blessed
configuration so iteration 0 establishes the baseline metric on the loop's
own eval path before any edits are accepted.

## Loop mechanics (one iteration)

1. **Propose.** `loop.py` invokes `claude -p` (headless; opus/fable tier — the
   proposal is the hard-reasoning step) with `program.md`, `results.tsv`, and
   recent git log in the prompt. Tool allowlist: Read/Edit/Write/Grep — no
   Bash (Write is needed only for the `.hypothesis` note). The agent applies
   **one focused edit** to `train_experiment.py` and writes a one-line
   hypothesis to `runs/<session>/.hypothesis`.
2. **Validate the diff.** Runner checks `git diff --name-only` touched exactly
   `autoresearch/train_experiment.py`, that `eval_harness.py`/`loop.py`/
   `_sealed.py`'s sha256 match the session pin, that the sealed
   `from _sealed import ...` line survives verbatim in
   `train_experiment.py`, and that no year-like literal (2019–2031) was
   introduced outside that import — the split years live only in
   `_sealed.py`. An AST parse then enforces the deeper invariants: the sealed
   names (`ACCEPT_VAL_MAX_YEAR`/`ACCEPT_VAL_MIN_YEAR`/`NUM_WORKERS`/
   `TRAIN_MAX_YEAR`) cannot be rebound by any assignment, and every `*_year`
   call keyword must be a sealed name or `None` — closing the "smuggle a year
   through a variable or expression" route. Any violation → status `invalid`,
   hard reset, next iteration.
3. **Commit** with the hypothesis as the message (commit-first: each
   experiment is an atomic, revertable unit).
4. **Train.** `timeout 600 uv run python autoresearch/train_experiment.py
   --seed 0 --max-steps 10000 --out-dir runs/<session>/iter-<n>/`.
   Timeout or nonzero exit → status `crash`, `git reset --hard HEAD~1`.
5. **Score.** `uv run python autoresearch/eval_harness.py --run-dir …` →
   JSON with `rank_ic_near` and the per-offset curve.
6. **Decide (deterministic runner code — the agent never scores itself).**
   Keep if `rank_ic_near > best + EPSILON`; else `git reset --hard HEAD~1`.
   `EPSILON` is the runner constant defined in item 2 of the departures
   section below (confirm-harness null band; recalibrated from ≥3 baseline
   seeds; not agent-editable).
7. **Log.** Append to `results.tsv`: iteration, timestamp, hypothesis, steps,
   wall seconds, `rank_ic_near`, `h1`, `h5`, status
   (`keep`/`discard`/`crash`/`invalid`). Every trial is logged, including
   failures.

Session ends on `--max-iters` or `--max-wall-clock`, whichever first.

## Deliberate departures from Karpathy (quant-honesty rules)

1. **Every trial is logged.** Karpathy erases failures via reset; quant rule
   B1 says the trial count is what makes the final number meaningful. The
   reset erases the *code*, never the *record*.
2. **ε acceptance threshold.** Single-seed `rank_ic_near` at 10k steps is
   noisy; ε starts at 0.02 (above the confirm-harness 3-seed MDE of 0.0069)
   and MUST be recalibrated from ≥3 baseline seeds (`ε = 2·SE`) before any
   unattended session, so coin-flip "wins" can't advance the branch.
3. **Sealed holdout.** Acceptance uses the 2024–2025 val years. The newest
   data slice (2026) is never read by the loop and is evaluated only at
   graduation — the check on the loop slowly overfitting its val split
   across ~70 trials. The 2026 slice cannot form full 365-row windows until
   ~mid-2027; until then it stays sealed and graduation substitutes the
   item-4 bar below.
4. **Graduation bar.** A champion must (a) survive multi-seed (0/1/2)
   10k-step re-runs, (b) survive at least one full-budget (~100k-step) run
   clearing the incumbent on the acceptance split, (c) be scored on the
   sealed holdout once scoreable, and (d) report a deflated significance
   whose trial count N includes all loop sessions (`results.tsv`, every
   status) AND the Optuna sweep trials on the same objective — before any
   of its ideas are proposed back into `src/ophir` via the normal spec/PR
   flow.

## program.md v1 (seeded search directives)

- **Objective:** maximize `rank_ic_near` (offsets 1–5) on the fixed val split.
- **Known dead ends:** `rezero_init` tuning (multi-seed confirmed no effect);
  pure hyperparameter grid-walking (the Optuna sweep owns that).
- **Known signal:** `rezero_lr` dominates sweep importance.
- **Promising directions (quant skill):** near-horizon loss shaping
  (loss-decay reweighting toward offsets 1–5); a cross-sectional ranking loss
  (rule F3: rank, don't regress); feature/causal-lag experiments — before
  more architecture.
- **Simplicity rule:** marginal metric gains do not justify added complexity;
  prefer the simpler variant on a near-tie.

## Guardrails

- Dedicated worktree + `autoresearch/session-<date>` branch; `main` untouched.
- 10-minute kill per training run; `--max-iters` / `--max-wall-clock` caps.
- Proposer allowlist Read/Edit/Write/Grep; the runner owns all shell and git.
  (Write is scoped to the `.hypothesis` note; the diff validator still rejects
  any tracked change outside `train_experiment.py`.)
- Hook + settings isolation: every loop git call runs with
  `-c core.hooksPath=<session>/hooks` (an empty per-session dir) so no repo or
  user git hook fires mid-trial, and the runner content-or-absence pins
  `.claude/settings.json`, `.claude/settings.local.json`, `CLAUDE.md`, and
  `AGENTS.md` alongside the harness files — an appearance, disappearance, or
  change of any of them aborts the session.
- Eval-integrity ordering: `eval_harness.py` binds every trusted symbol
  (torch/lightning/`ophir.*`/sealed constants) before importing the
  agent-authored `train_experiment` module, so its top-level code cannot
  monkeypatch the scorer's dependencies first. This is defense-in-depth only —
  eval integrity against adversarial experiment code ultimately rests on human
  diff review of the kept commits at graduation.
- No network required (data is local under `.ophir/`).

## Testing

- `loop.py` keeps decision, logging, diff-validation, and results-parsing
  logic pure and torch-free → offline CPU pytest coverage (tmp git repos via
  `tmp_path`), consistent with the suite's offline/CPU-only constraint.
- `train_experiment.py` / `eval_harness.py` are GPU-runtime paths, exercised
  by the smoke trial, not the suite.

## Delivery scope (this iteration)

Harness + **smoke trial**: two supervised loop iterations end-to-end —
iteration 0 establishes the baseline metric; one real accept/reject decision
follows — then stop and report. Unattended overnight runs are a later
"press play" step, not part of this delivery.

## Out of scope

- Two-tier proxy/confirm trial budgets (v2 upgrade if throughput matters).
- Wiring any champion into `src/ophir` or the trading seam.
- OS-level sandboxing of the proposer (single-user local box; the allowlist
  plus worktree isolation is the v1 containment).
