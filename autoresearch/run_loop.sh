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
