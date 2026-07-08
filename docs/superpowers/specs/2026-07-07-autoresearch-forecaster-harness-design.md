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
| `loop.py` | Outer loop: invoke proposer → commit → timed run → eval → keep/revert → log. Torch-free, unit-testable | Frozen during a session |
| `run_loop.sh` | Thin entry: worktree/branch setup, session dir, invokes `loop.py` | Frozen during a session |
| `runs/<session>/results.tsv` | Append-only trial log | Runner only; gitignored during runs, committed at session end |

`train_experiment.py` v0 (the baseline) reproduces the current blessed
configuration so iteration 0 establishes the baseline metric on the loop's
own eval path before any edits are accepted.

## Loop mechanics (one iteration)

1. **Propose.** `loop.py` invokes `claude -p` (headless; opus/fable tier — the
   proposal is the hard-reasoning step) with `program.md`, `results.tsv`, and
   recent git log in the prompt. Tool allowlist: Read/Edit/Grep only — no
   Bash. The agent applies **one focused edit** to `train_experiment.py` and
   writes a one-line hypothesis to `runs/<session>/.hypothesis`.
2. **Validate the diff.** Runner checks `git diff --name-only` touched exactly
   `autoresearch/train_experiment.py` and that `eval_harness.py`'s sha256
   matches the session pin. Any violation → status `invalid`, hard reset,
   next iteration.
3. **Commit** with the hypothesis as the message (commit-first: each
   experiment is an atomic, revertable unit).
4. **Train.** `timeout 600 uv run python autoresearch/train_experiment.py
   --seed 0 --max-steps 10000 --out-dir runs/<session>/iter-<n>/`.
   Timeout or nonzero exit → status `crash`, `git reset --hard HEAD~1`.
5. **Score.** `uv run python autoresearch/eval_harness.py --run-dir …` →
   JSON with `rank_ic_near` and the per-offset curve.
6. **Decide (deterministic runner code — the agent never scores itself).**
   Keep if `rank_ic_near > best + EPSILON`; else `git reset --hard HEAD~1`.
   `EPSILON` is a runner constant derived from the confirm-harness null band,
   not agent-editable.
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
   noisy; ε (from the confirm-harness null band) keeps coin-flip "wins" from
   advancing the branch.
3. **Sealed holdout.** Acceptance uses the 2024–2025 val years. The newest
   data slice (2026) is never read by the loop and is evaluated only at
   graduation — the check on the loop slowly overfitting its val split
   across ~70 trials.
4. **Graduation bar.** A champion must survive multi-seed (0/1/2) re-runs at
   10k steps — and clear the sealed holdout — before any of its ideas are
   proposed back into `src/ophir` via the normal spec/PR flow.

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
- Proposer allowlist Read/Edit/Grep; the runner owns all shell and git.
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
