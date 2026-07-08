# OKF-Bundle-Guided Repo Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire ophir's existing quality gates into CI + branch protection, make `AGENTS.md` the canonical cross-platform agent guide, and expand pre-commit with ruff and secret scanning — per the approved spec at `docs/superpowers/specs/2026-07-07-okf-bundle-repo-setup-design.md`.

**Architecture:** Pure configuration/documentation work — no runtime code changes. Local file tasks land on a `chore/okf-repo-setup` branch; GitHub-side tasks (push, PR, CI verification, branch protection) come last because branch protection can only reference a status check that has run at least once.

**Tech Stack:** GitHub Actions, `astral-sh/setup-uv@v8`, pre-commit (local `uv run --frozen` hooks + gitleaks `v8.30.1`), `gh` CLI.

## Global Constraints

- The CI job MUST be named exactly `checks` — branch protection references this context string.
- mypy targets Python 3.10, ruff targets 3.12 — do not change either.
- Tests are offline + CPU-only; CI uses a stock Ubuntu runner, no secrets, no GPU.
- ruff version comes from `uv.lock` (0.15.13) via `uv run --frozen` — never from a pre-commit rev.
- Linear history on `main`: fast-forward/rebase only, no merge commits.
- Do NOT touch `src/ophir`, `tests/`, or `pyproject.toml`.
- Update `CHANGELOG.md` `[Unreleased]` (Keep a Changelog format).
- Repo: `kwcantrell/ophir`; local `main` is ~200 commits ahead of `origin/main` — Task 5 publishes it (flagged to the user during brainstorming).

---

### Task 1: AGENTS.md canonical, CLAUDE.md import shim

**Files:**
- Create: `AGENTS.md` (via `git mv` from `CLAUDE.md`, then edit)
- Create: `CLAUDE.md` (new two-line shim)

**Interfaces:**
- Produces: `AGENTS.md` with a `## Git conventions` section and updated Commands section; later tasks (2, 4) reference the hooks and CI it describes.

- [ ] **Step 1: Move the file, preserving history**

```bash
cd /home/kalen/ophir
git checkout -b chore/okf-repo-setup main
git mv CLAUDE.md AGENTS.md
```

- [ ] **Step 2: Edit AGENTS.md**

Apply exactly these four edits to `AGENTS.md`:

1. Replace the first two lines:

```markdown
# CLAUDE.md

Guidance for agents working in this repository. See `README.md` for the full
```

with:

```markdown
# AGENTS.md

Guidance for agents working in this repository. See `README.md` for the full
```

2. In the `## Commands` code block, replace the line:

```
uv run pre-commit install                    # one-time: installs the mypy hook
```

with:

```
uv run pre-commit install                    # one-time: installs the mypy/ruff/gitleaks hooks
```

3. Immediately after the line `Run a single test file with `uv run pytest tests/test_<name>.py`.`, add:

```markdown

CI (`.github/workflows/ci.yml`) runs the same four checks (ruff check, ruff
format, mypy, pytest) on every PR and on pushes to `main`.
```

4. After the `## Conventions` section's last bullet (`- Update the ...CHANGELOG.md... for notable changes.`), insert a new section before `## Dev workflow`:

```markdown

## Git conventions

- **Conventional Commits**: `type(scope): description` — types in use: `feat`,
  `fix`, `docs`, `refactor`, `test`, `chore`, `style`.
- **Branch naming**: `category/short-description` (`feat/`, `fix/`, `docs/`,
  `refactor/`, `chore/`).
- **Merging**: PRs into `main` with linear history (fast-forward or rebase —
  no merge commits). `main` requires the `checks` CI job to pass.
```

- [ ] **Step 3: Write the new CLAUDE.md shim**

Create `CLAUDE.md` with exactly this content:

```markdown
# CLAUDE.md

@AGENTS.md
```

- [ ] **Step 4: Verify**

```bash
head -3 AGENTS.md        # expect "# AGENTS.md" header
cat CLAUDE.md            # expect the 3-line shim above
grep -c "Git conventions" AGENTS.md   # expect 1
git status --short       # expect: R  CLAUDE.md -> AGENTS.md, A/?? CLAUDE.md
```

- [ ] **Step 5: Commit**

```bash
git add AGENTS.md CLAUDE.md
git commit -m "docs: make AGENTS.md canonical; CLAUDE.md imports it"
```

---

### Task 2: Pre-commit expansion (ruff + gitleaks)

**Files:**
- Modify: `.pre-commit-config.yaml`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: hooks named `ruff-check`, `ruff-format`, `gitleaks` alongside the existing `mypy` hook.

- [ ] **Step 1: Replace `.pre-commit-config.yaml` with**

```yaml
# Run `uv run pre-commit install` once after `uv sync`.
# mypy and ruff run via `uv run --frozen` against the project venv
# (torch/lightning/numpy/py.typed all present), NOT an isolated pre-commit
# venv, so the hook, CI, and dev env share the single version pinned in
# pyproject.toml / uv.lock.
repos:
  - repo: local
    hooks:
      - id: mypy
        name: mypy (uv run, project venv)
        entry: uv run --frozen mypy
        language: system
        pass_filenames: false
        always_run: true
        args: ["src/ophir"]
        types: [python]
      - id: ruff-check
        name: ruff check (uv run, project venv)
        entry: uv run --frozen ruff check --force-exclude
        language: system
        types_or: [python, pyi]
        require_serial: true
      - id: ruff-format
        name: ruff format --check (uv run, project venv)
        entry: uv run --frozen ruff format --check --force-exclude
        language: system
        types_or: [python, pyi]
        require_serial: true
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.30.1
    hooks:
      - id: gitleaks
```

Note: `ruff-format` is check-only (non-mutating) — CI behaves identically, and
the Claude Code PostToolUse hook already auto-formats agent edits.

- [ ] **Step 2: Run the full hook suite to verify it fails or passes honestly**

```bash
uv run pre-commit run --all-files
```

Expected: all four hooks run; `mypy`, `ruff check`, `ruff format --check` PASS
(tree is currently clean); `gitleaks` PASSES (first run downloads/builds the
hook env — this is the one hook with a pinned rev, by design: it is not a
Python dependency of the project).

If any hook fails, STOP and report — the tree was green before this task, so
a failure means the hook config is wrong, not the code.

- [ ] **Step 3: Commit**

```bash
git add .pre-commit-config.yaml
git commit -m "chore: add ruff check/format and gitleaks pre-commit hooks"
```

---

### Task 3: CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Produces: a workflow job with context name `checks` — Task 6's branch protection references it verbatim.

- [ ] **Step 1: Create `.github/workflows/ci.yml` with exactly**

```yaml
name: CI

on:
  pull_request:
  push:
    branches: [main]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: astral-sh/setup-uv@v8
        with:
          python-version-file: ".python-version"
          enable-cache: true
      - run: uv sync --frozen --group dev
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run mypy src/ophir
      - run: uv run pytest
```

Do NOT rename the `checks` job — branch protection (Task 6) requires that
exact context string.

- [ ] **Step 2: Validate the YAML parses**

```bash
uv run python -c "import yaml, pathlib; yaml.safe_load(pathlib.Path('.github/workflows/ci.yml').read_text()); print('ok')"
```

Expected: `ok` (PyYAML is available transitively in the dev env; if the import
fails, use `python3 -c` with the system interpreter instead). Real validation
happens when Task 5 pushes and the workflow runs.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add checks workflow (ruff, mypy, pytest via uv)"
```

---

### Task 4: PR template + changelog

**Files:**
- Create: `.github/pull_request_template.md`
- Modify: `CHANGELOG.md` (top of `## [Unreleased]` → `### Added`)

- [ ] **Step 1: Create `.github/pull_request_template.md` with exactly**

```markdown
## Summary

<!-- What changed and why. -->

## Test plan

<!-- How this was verified: commands run, CI results, manual checks. -->

## Checklist

- [ ] `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`,
      and `uv run mypy src/ophir` pass locally
- [ ] `CHANGELOG.md` `[Unreleased]` updated for notable changes
```

- [ ] **Step 2: Add the changelog entry**

In `CHANGELOG.md`, under `## [Unreleased]` / `### Added`, insert as the FIRST
bullet:

```markdown
- Repo setup guided by the OKF best-practice bundles
  ([kwcantrell/okf-bundles](https://github.com/kwcantrell/okf-bundles)):
  GitHub Actions CI (`.github/workflows/ci.yml`, job `checks`) running
  ruff check / ruff format / mypy / pytest on every PR and push to `main`;
  branch protection on `main` (required `checks` run, linear history, no
  force pushes); `AGENTS.md` as the canonical agent guide with `CLAUDE.md`
  importing it via `@AGENTS.md`; pre-commit expanded with ruff (project-venv
  hooks) and gitleaks secret scanning; a PR template. Spec:
  `docs/superpowers/specs/2026-07-07-okf-bundle-repo-setup-design.md`.
```

- [ ] **Step 3: Run pre-commit on the changed files**

```bash
uv run pre-commit run --all-files
```

Expected: all hooks PASS (markdown files only trigger gitleaks).

- [ ] **Step 4: Commit**

```bash
git add .github/pull_request_template.md CHANGELOG.md
git commit -m "docs: add PR template and changelog entry for repo setup"
```

---

### Task 5: Publish and verify CI

**Interfaces:**
- Consumes: the `checks` workflow from Task 3.
- Produces: a green `checks` run on GitHub — the prerequisite for Task 6.

- [ ] **Step 1: Push `main` (publishes ~200 local commits — user-approved during brainstorming; confirm with the user if executing autonomously)**

```bash
git push origin main
```

- [ ] **Step 2: Push the branch and open a PR**

```bash
git push -u origin chore/okf-repo-setup
gh pr create --base main --title "chore: OKF-bundle-guided repo setup" \
  --body "CI workflow, AGENTS.md canonical, pre-commit ruff+gitleaks, PR template. Spec: docs/superpowers/specs/2026-07-07-okf-bundle-repo-setup-design.md. Test plan: checks job on this PR is the test."
```

- [ ] **Step 3: Watch the CI run to completion**

```bash
gh pr checks --watch
```

Expected: `checks` → pass. First run is slow (torch install; the uv cache
warms later runs). If it fails, read the log with `gh run view --log-failed`,
fix on the branch, push, and re-watch. Do not proceed until green.

- [ ] **Step 4: Merge with linear history**

```bash
gh pr merge --rebase --delete-branch
git checkout main && git pull --ff-only origin main
```

Expected: `main` now contains the four task commits, no merge commit.

---

### Task 6: Branch protection on main

**Interfaces:**
- Consumes: the `checks` context, which has now run at least once (Task 5).

- [ ] **Step 1: Apply protection via gh api**

```bash
gh api -X PUT repos/kwcantrell/ophir/branches/main/protection --input - <<'JSON'
{
  "required_status_checks": {"strict": true, "checks": [{"context": "checks"}]},
  "enforce_admins": false,
  "required_pull_request_reviews": null,
  "restrictions": null,
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false
}
JSON
```

Expected: HTTP 200 with a JSON body echoing the settings. If it fails with
403/404 (token scope or plan limits), STOP and report — the settings are
recorded in the spec §2 for manual application in the GitHub UI; nothing else
depends on this call.

- [ ] **Step 2: Verify the applied settings**

```bash
gh api repos/kwcantrell/ophir/branches/main/protection --jq \
  '{checks: .required_status_checks.checks, linear: .required_linear_history.enabled, force: .allow_force_pushes.enabled, del: .allow_deletions.enabled}'
```

Expected output:

```json
{"checks":[{"context":"checks"}],"linear":true,"force":false,"del":false}
```

- [ ] **Step 3: Final verification sweep (spec §Testing)**

```bash
uv run pre-commit run --all-files    # all hooks pass
uv run pytest -q                     # suite untouched, still green
```

Then confirm in a fresh Claude Code session that project instructions load
(the `@AGENTS.md` import resolves — the session context should show the
architecture/constraints content).
