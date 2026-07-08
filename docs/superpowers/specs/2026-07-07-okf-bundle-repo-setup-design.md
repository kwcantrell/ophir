# OKF-Bundle-Guided Repo Setup — Design

**Date:** 2026-07-07
**Source of guidance:** the three OKF knowledge bundles in
[kwcantrell/okf-bundles](https://github.com/kwcantrell/okf-bundles)
(`git-best-practices`, `ai-agent-repo-structure`, `claude-best-practices`).

## Goal

Apply the bundles' concrete recommendations to ophir. A full gap analysis found
ophir already strong internally (commit discipline, strict typing, deterministic
tests, lean CLAUDE.md, local automation); the gaps are external: no CI, no
branch protection, no cross-platform agent discoverability, and incomplete
pre-commit coverage.

## Scope

**In scope (approved):**

1. CI + branch protection
2. Agent discoverability (AGENTS.md canonical)
3. Local git hygiene (pre-commit expansion, PR template, documented conventions)

**Explicitly out of scope (solo-repo ceremony, deferred by decision):**

- CODEOWNERS, required reviewers, GPG-signed commits/tags
- `.claude/rules/` split — deferred until AGENTS.md outgrows ~200 lines
  (currently ~90; the bundles' own threshold is 200)
- Separate CONTRIBUTING.md — conventions live in AGENTS.md instead

## Design

### 1. CI workflow — `.github/workflows/ci.yml`

- Triggers: `pull_request`, and `push` to `main`.
- One Ubuntu job named `checks` (this exact name is the required status check
  in §2), steps:
  1. Checkout.
  2. `astral-sh/setup-uv` with the Python version read from `.python-version`
     (3.10, the lowest supported runtime).
  3. `uv sync --group dev` (frozen — respects `uv.lock`).
  4. Run exactly the four documented checks:
     `uv run ruff check .`, `uv run ruff format --check .`,
     `uv run mypy src/ophir`, `uv run pytest`.
- Concurrency group keyed on workflow + ref with `cancel-in-progress: true`.
- No secrets, no GPU: the test suite is offline + CPU-only by hard constraint,
  so a stock runner suffices with no test changes.

### 2. Branch protection on `main`

Applied once via `gh api` against `kwcantrell/ophir`; settings recorded here
for reproducibility:

- Required status check: the `checks` job from §1 (strict — branch must be up
  to date with `main`).
- Required linear history (matches the existing pre-linearize-main cleanup).
- Force pushes and branch deletion blocked.
- **No** required reviewer count and **no** admin enforcement: the solo owner
  can merge their own PRs once CI is green and retains a direct-push escape
  hatch for emergencies.

### 3. AGENTS.md canonical, CLAUDE.md imports it

- `git mv CLAUDE.md AGENTS.md` (preserves history/blame).
- New `CLAUDE.md` containing only a pointer line and the `@AGENTS.md` import,
  so Claude Code loads identical content while Copilot/Cursor/API agents read
  the cross-platform standard file.
- Content updates made during the move (edits, not a rewrite):
  - New **Git conventions** section documenting existing practice:
    Conventional Commits (`type(scope): description`); branch naming
    `category/short-description` with `feat/`, `fix/`, `docs/`, `refactor/`
    categories; PRs merged into `main` with linear history.
  - **Commands** section notes CI runs the same four checks on every PR.
  - Pre-commit line updated: hooks are now mypy + ruff check + ruff format +
    gitleaks (see §4).

### 4. Pre-commit expansion — `.pre-commit-config.yaml`

- Existing local mypy hook: unchanged.
- **Ruff hooks follow the same local-hook pattern as mypy** — `uv run --frozen
  ruff check` and `uv run --frozen ruff format --check` as `language: system`
  hooks — rather than the separately-versioned `ruff-pre-commit` repo. This
  guarantees the hook, CI, and dev environment all use the single ruff version
  pinned in `uv.lock` (0.15.13 at time of writing) with no rev drift.
  The format hook is check-only (non-mutating), consistent with CI; the
  PostToolUse hook already auto-formats Claude's edits.
- **Secret scanning:** the official `gitleaks` pre-commit hook (pinned rev),
  scanning staged changes. Closes the `.env` / `MASSIVE_API_KEY` exposure risk
  before anything reaches git history.

### 5. PR template — `.github/pull_request_template.md`

Minimal three-part template:

- **Summary** — what and why.
- **Test plan** — how it was verified.
- **Checklist** — `uv run pytest` / ruff / mypy pass; `CHANGELOG.md
  [Unreleased]` updated for notable changes.

### 6. Changelog

Add an `[Unreleased]` entry describing the setup work itself.

## Error handling

- CI failures are the feature: they block merges via §2.
- If `gh api` branch-protection calls fail (permissions/plan limits), record
  the intended settings in this spec (done above) and apply manually in the
  GitHub UI; nothing else depends on the API call succeeding.
- A gitleaks false positive is handled with an inline
  `# gitleaks:allow` comment or a repo `.gitleaks.toml` allowlist — never by
  skipping the hook.

## Testing / verification

1. `uv run pre-commit run --all-files` passes on a clean tree.
2. Push the branch; the CI workflow runs and goes green on GitHub.
3. Open a PR; the template appears and the CI check is listed as required.
4. Verify a direct force-push to `main` is rejected.
5. Start a fresh Claude Code session; confirm project instructions still load
   (the `@AGENTS.md` import resolves).
6. `src/ophir`, the safety gate, and the test suite are untouched — full
   suite still passes.

## Non-goals reaffirmed

Nothing in this work touches runtime code, the non-overridable safety gate
(`src/ophir/trading/safety.py`), `account_mode` validation, or test behavior.
