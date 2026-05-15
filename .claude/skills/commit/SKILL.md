---
name: commit
description: >-
  Stage relevant tracked + untracked changes, update README/docs and
  CHANGELOG, bump the version in pyproject.toml and docs/conf.py, and create
  one or more well-formed commits after showing the user the full plan for
  confirmation. Use when the user asks to "commit", "make a commit", "commit
  my changes", or "commit and bump". Never pushes; never creates tags.
---

# commit

Turn the current working tree into one or more well-formed commits for the
**ophir** repo, keeping release hygiene intact. For **every** commit this
infers a SemVer level, bumps the version in `pyproject.toml` **and**
`docs/conf.py`, cuts a dated "Keep a Changelog" section with correct
compare-link footers, updates hand-written docs when behavior changed, stages
the unit atomically, shows the full plan, and only commits after explicit
confirmation. **It never runs `git push` and never creates git tags.**

## Arguments

Parse `$ARGUMENTS` loosely; all are optional and order-independent:

- **Free text** (anything not matched below) → extra intent/context to fold
  into the generated commit message(s) and changelog wording. It informs the
  message; it is not necessarily used verbatim as the subject.
- **`amend`** (bare keyword) → amend the previous commit instead of creating a
  new one. Forces a single unit (no splitting). See **B-6**.
- **Path or glob that resolves to an existing tracked-or-untracked path**
  (e.g. `src/ophir/cli.py`, `docs/`) → restrict the entire run to changes
  under those paths. Everything else is left completely untouched and reported
  as "deferred".

Disambiguation: a token is a path scope only if it matches an existing path or
glob; otherwise it is message text. `amend` is always the keyword unless
quoted. Tokens that look like version overrides (`--minor`, `v0.3.0`) or
skip-step flags are **not supported** — if you see one, tell the user the
SemVer level is inferred per commit and is correctable at the confirmation
gate, then continue.

---

## Procedure

### Phase 0 — Preconditions

1. Confirm a git work tree: `git rev-parse --is-inside-work-tree`. If not,
   abort with a clear message. Capture the original HEAD SHA:
   `git rev-parse HEAD`.
2. Inspect HEAD state:
   - **Mid-merge / rebase / cherry-pick** (`git rev-parse -q --verify
     MERGE_HEAD`, a `.git/rebase-*` dir, or `CHERRY_PICK_HEAD`) → **abort**.
     Tell the user to conclude that operation first; this skill is not a
     conflict-resolution tool.
   - **Detached HEAD** (`git symbolic-ref -q HEAD` fails) → do not silently
     proceed. Warn loudly; the confirmation gate defaults to **abort**.
3. Read the baseline version from all three sources:
   - `pyproject.toml` → `version = "X.Y.Z"` (line 3, under `[project]`).
   - `docs/conf.py` → `release = "X.Y.Z"` and `version = "X.Y.Z"`
     (≈ lines 23–24).
   - `CHANGELOG.md` → the top released `## [X.Y.Z] - DATE` section.
   If they disagree, do **not** guess (see **B-11**): record the mismatch as a
   gate warning and ask which is authoritative; the first unit resyncs all
   three.
4. Record pre-existing staged files: `git diff --cached --name-only` (for
   **B-7**).

### Phase 1 — Discover changes

5. Build the candidate set from `git status --porcelain=v1`: modified, deleted,
   and renamed tracked files **plus** untracked-not-ignored files (`??`
   entries already exclude `.gitignore`d paths like `oldcode/`, `data/`,
   `saved-models/`, `.claude/settings.local.json`).
6. If a path-scope argument was given, filter the set to matching paths; the
   rest is reported as **deferred** and left untouched.
7. Empty set → **B-1** (report "nothing to commit", make no edits, exit).
8. For each file, gather the diff (`git diff` / `git diff --cached` for
   already-staged), the change type (A/M/D/R), and a brief read of the most
   significant hunks — enough to judge grouping and SemVer impact.

### Phase 2 — Group into ordered commit units

9. Partition the candidate set into ordered **commit units**:
   - `amend` given → exactly **one** unit (all in-scope changes); skip the
     rest of the grouping rules.
   - Otherwise group by a **single coherent concern**: a code change travels
     with its own docs/tests (e.g. `src/ophir/cli.py` + `docs/cli.rst` + its
     tests + the README section describing it). Split apart: unrelated source
     changes; docs-only edits unrelated to code; infra/config (`.github/`,
     `.claude/`, ruff config in `pyproject.toml`, `.devcontainer/`); notebooks
     (`notebooks/`, usually their own exploratory unit). Group dependency
     changes (`pyproject.toml` `dependencies`, `uv.lock`) with the code that
     needs them when traceable, else their own unit.
   - **Never split a single file's hunks across units** (one file → one unit).
   - A pure refactor intermixed with a behavior change in the same file stays
     together; the dominant (behavior) signal drives the message and SemVer.
   - Order units so prerequisites land first (a new dependency or core module
     before the feature using it; config before code relying on it). The
     version trajectory follows this order.
   - Prefer **fewer** units. If you would produce more than ~5, flag it at the
     gate and suggest the user narrow scope or regroup.

### Phase 3 — Infer the full version trajectory (before any disk write)

10. Maintain a running version starting at the baseline. For each unit in
    order, infer its SemVer level from its diff using the **0.x heuristics**
    below, advance the running version, and let the next unit infer relative
    to the advanced value. Produce the whole trajectory, e.g.
    `0.1.0 → 0.2.0 → 0.2.1`.
11. For each unit also draft: the commit message (imperative subject ~50–70
    chars, blank line, then a bullet-point body — **no** conventional-commit
    prefix; attribution is already disabled in `.claude/settings.json`, so add
    no `Co-authored-by`/footer), and the CHANGELOG category + bullet(s).

#### Pre-1.0 SemVer heuristics (`0.MINOR.PATCH`)

The major slot stays `0` while pre-1.0: **breaking changes bump MINOR**,
**backward-compatible changes bump PATCH**.

- **MAJOR (→ `1.0.0`)** — never auto-inferred from ordinary changes. Only if
  the user's intent explicitly declares "first stable / leaving 0.x". If
  there's any doubt, do not pick major; raise it as a gate question.
- **MINOR (`0.M.P` → `0.M+1.0`)** — breaking or substantial while in 0.x:
  removed/renamed a public symbol, CLI command, or flag; changed a
  signature/default/output format/on-disk or config schema incompatibly;
  raised a dependency floor or `requires-python` so existing installs break; a
  sizable new user-facing feature (new CLI subcommand, new public module/API).
- **PATCH (`0.M.P` → `0.M.P+1`)** — backward-compatible: bug fix with no API
  change; small additive non-breaking enhancement (new optional flag with a
  safe default); internal refactor, perf, type annotations, ruff-only cleanup;
  docs-only, tests-only, CI/tooling/config, notebooks, lockfile refresh.
- **Tie-break:** take the **highest** level any file in the unit warrants.
  Deletions of public surface bias MINOR; pure additions bias PATCH;
  modifications judged by compatibility. Free-text intent can raise the level;
  the gate is where the user definitively raises or lowers it.

`amend` mode infers **no new bump** — see **B-6**.

### Phase 4 — Confirmation gate (BEFORE any commit)

12. Present the full plan, then **stop and require an explicit "yes"**. Show:
    - **Version trajectory**: `0.1.0 → 0.2.0 → 0.2.1`, with the inferred level
      and a one-line rationale per step.
    - **Ordered commit list**, per unit: proposed subject + body bullets; the
      exact file grouping with A/M/D/R markers; which docs will be updated; the
      CHANGELOG section + category; the resulting version.
    - **Deferred / out-of-scope / pre-staged** files (B-7), and any
      **warnings**: detached HEAD, version-file mismatch (B-11), > ~5 units,
      non-user-facing unit cutting a version (B-3), suspicious untracked files
      (B-2).
    - **Correction options** (conversational — no special args): adjust any
      unit's SemVer level (then recompute the whole downstream trajectory);
      regroup / merge / split / reorder / drop a unit (a dropped unit's files
      stay unstaged); edit messages or changelog bullets; toggle `amend` for
      the first unit; or abort.
13. Use `AskUserQuestion` (or a plain confirmation prompt) for the decision.
    When any warning is present, the **default selection is abort**. Proceed
    only on explicit confirmation.

### Phase 5 — Materialize + commit each unit sequentially (after approval)

> **Critical ordering.** `CHANGELOG.md`, `pyproject.toml`, and `docs/conf.py`
> are cumulative across units. Materialize **and commit** each unit fully
> *before* materializing the next. Never materialize all units then commit —
> that would put the final cumulative version/changelog state into commit #1.

For each unit `N` (in order) with target version `Vn`:

1. **Docs** — only if user-facing behavior changed. Edit `README.md` and/or
   the relevant `docs/*.rst` (`cli.rst`, `overview.rst`, `architecture.rst`,
   `installation.rst`). **Never** edit `docs/api/generated/` (gitignored,
   autosummary-generated). **Never** run `sphinx-build` — CI builds/deploys.
   Docs-only or non-user-facing units skip this step.
2. **Version bump** — `pyproject.toml` line 3 → `version = "Vn"`;
   `docs/conf.py` → `release = "Vn"` **and** `version = "Vn"`.
3. **CHANGELOG** — insert a new `## [Vn] - 2026-05-14` section (use today's
   actual date) directly **below** `## [Unreleased]` and **above** the
   previous top version, with the appropriate `### Added` / `### Changed` /
   `### Fixed` (also `Removed` / `Deprecated` / `Security` if apt) subsections
   and this unit's bullets. Move any existing relevant `[Unreleased]` bullets
   down into it; leave `## [Unreleased]` present but empty. Rewrite the footer
   compare-link block so it stays correct across multiple new versions:
   - `[Unreleased]` compares the newest version tag → `HEAD`.
   - Each released version compares the **previous** version tag → its own
     tag, **except** the original `0.1.0`, which keeps its existing
     `releases/tag/v0.1.0` form (do not rewrite it).
   - Use `vX.Y.Z` tag names in URLs (forward-declared; this skill does not
     create tags).

   Example footer after a run that added `0.2.0` then `0.2.1` atop `0.1.0`:
   ```
   [Unreleased]: https://github.com/kwcantrell/ophir/compare/v0.2.1...HEAD
   [0.2.1]: https://github.com/kwcantrell/ophir/compare/v0.2.0...v0.2.1
   [0.2.0]: https://github.com/kwcantrell/ophir/compare/v0.1.0...v0.2.0
   [0.1.0]: https://github.com/kwcantrell/ophir/releases/tag/v0.1.0
   ```
   Body section order becomes: `## [Unreleased]` (empty) → `## [0.2.1]` →
   `## [0.2.0]` → `## [0.1.0]`.
4. **Stage atomically** — `git add --` with an **explicit pathspec list**:
   this unit's files plus `pyproject.toml`, `docs/conf.py`, `CHANGELOG.md`,
   and any docs touched in step 1. **Never** use `git add -A` / `git add .` —
   other units' changes and deferred files must stay unstaged.
5. **Re-stage after the ruff hook** — the PostToolUse hook runs
   `ruff check --fix` + `ruff format` on every edited `.py` file, including
   `docs/conf.py`, *after* the Edit returns, which can leave the staged blob
   stale. Re-run `git add --` for the `.py` paths in this unit, then verify
   with `git status --porcelain` that nothing for this unit is left
   unstaged/dirty. If `ruff check` exited non-zero (lint failure) on a source
   file the unit needs, **pause** and surface the ruff output — do not commit
   broken code; let the user decide or apply the fix and re-stage.
6. **Commit** — `git commit -m "<subject>" -m "<body>"` (do not write temp
   message files). Then move on to unit `N+1`.

### Phase 6 — Summary

14. Report: each commit created (short SHA + subject), the final version, and
    the new CHANGELOG sections. State explicitly that **nothing was pushed and
    no git tags were created**, and suggest the manual follow-up (`git push`,
    tagging at release time) **without performing it**.

---

## Edge cases & decisions

- **B-1 No changes** — clean tree and nothing staged: report "nothing to
  commit", make no edits, no version/changelog, exit.
- **B-2 Only untracked** — valid; new files generally read as `Added`. Flag
  any suspicious large/binary/data-looking untracked files in the plan so the
  user can exclude accidental artifacts.
- **B-3 Non-user-facing unit** — every commit must still cut a version (firm
  requirement). Write a truthful minimal entry under the best-fitting category
  (usually a PATCH `### Changed`/`### Fixed`, e.g. "Internal refactor; no
  user-facing changes." / "Update CI workflow." / "Refresh dependency
  lockfile."). Never invent user-facing impact. Surface it at the gate so the
  user may merge it into another unit to avoid trivial version churn.
- **B-6 `amend` (no double-bump)** — the previous commit already carries a
  bump + section `Vp`. Do **not** infer or apply a new bump. Fold the new /
  scoped changes into the existing top `## [Vp]` section and reuse `Vp` in
  `pyproject.toml`/`conf.py`. Only if the enlarged diff escalates the level,
  rewrite `Vp` (its number, section header, and footer) **in place** — still
  exactly one version delta versus the commit before the amended one, never
  two. Update the amended commit message to reflect the combined change. If
  the previous commit has no skill-style changelog section, refuse `amend`
  unless the user explicitly forces it (then amend content only and ask how to
  handle the changelog).
- **B-7 Pre-existing staged files** — include them in discovery and assign
  them to units, flagged "was pre-staged" in the plan. Offer at the gate to
  instead `git restore --staged` and exclude them. Never silently commit
  pre-staged files without surfacing them.
- **B-8 Scope × multi-commit** — a path scope filters the candidate set
  *first*; grouping then operates only within it (multiple units may still
  form). Out-of-scope changes stay entirely untouched (unstaged, working tree
  preserved) and never receive a version/changelog. The trajectory reflects
  only in-scope units.
- **B-9 ruff hook** — see Phase 5.5: always re-`git add` edited `.py` files
  and verify clean before committing the unit.
- **B-10 Partial failure mid-sequence** — if a commit fails after some commits
  were already made: stop immediately (do not proceed to the next unit). Do
  **not** auto-rollback existing commits. Report which units committed (SHAs),
  which failed and why, and what remains staged/materialized. Offer explicit
  recovery options with exact commands — (i) fix the cause and resume from the
  failed unit, (ii) `git reset --soft <original HEAD>` to collapse everything
  back (edits preserved in the tree), (iii) leave as-is — and let the user
  choose. A failure *before any commit* (during Phase 5 materialization of the
  first unit) is safe to fully revert: undo the skill's own edits and unstage
  to restore the pre-run state.
- **B-11 Version-file mismatch** — never guess. Surface at the gate, ask which
  source is authoritative, and have the first unit's version step resync all
  three (`pyproject.toml`, `docs/conf.py`, CHANGELOG baseline).
- **B-12 Malformed CHANGELOG** — if the expected anchors (`## [Unreleased]`,
  the footer link block) are absent, surface it and ask before restructuring.
  Never blind-regex a file that lacks the expected structure.
- **B-13 Deletes / renames** — keep both sides of a rename in the same unit;
  renaming/deleting public surface biases MINOR; deleting dead/internal code
  is PATCH.
