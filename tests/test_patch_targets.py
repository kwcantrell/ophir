"""Guard against stale ``mock.patch`` / ``monkeypatch.setattr`` string targets.

When a symbol moves to a different module (e.g. a package split), any test that
patches it by its old dotted path silently rots: the patch either fails to find
the attribute or no longer affects the relocated code. The public-API parity
checks used during such refactors do not catch this, because the broken name is
an *internal* patch target, not part of the package's public surface.

This module AST-scans every test file for string-literal patch targets that
point into the ``ophir`` package and asserts each one still resolves to a real
attribute. A broken target fails here — naming the file, line, and stale path —
instead of surfacing as a cryptic ``AttributeError`` deep inside an unrelated
test.

Only string-literal first arguments to ``patch`` / ``setattr`` calls are
checked; dynamic (computed) targets are reported by
:func:`test_dynamic_targets_are_reported_not_silently_skipped` so the coverage
gap is never silent.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

# Top-level package whose patch targets we guard. A target must start with
# ``PACKAGE + "."`` to be collected.
PACKAGE = "ophir"

TESTS_DIR = Path(__file__).resolve().parent

# Call attribute names that take a dotted-string target as their first
# positional argument: ``mock.patch("a.b.c")`` and
# ``monkeypatch.setattr("a.b.c", value)``.
_PATCH_FUNCS = {"patch", "setattr"}


def _resolve_target(dotted: str) -> None:
    """Resolve a dotted patch target the way ``mock`` does, or raise.

    Walks decreasing-length prefixes of ``dotted`` until one imports as a
    module, then follows the remaining components with ``getattr``. Raises
    :class:`ImportError` if no prefix imports, or :class:`AttributeError` if a
    trailing component is missing — exactly the failure modes a stale target
    produces.

    Parameters
    ----------
    dotted : str
        A dotted path such as ``"ophir.ticker.splits.pd.read_html"``.
    """
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for attr in parts[i:]:
            obj = getattr(obj, attr)
        return
    raise ImportError(f"no importable module prefix in {dotted!r}")


def _iter_patch_calls(tree: ast.AST) -> list[ast.Call]:
    """Return every ``patch``/``setattr`` call node in ``tree``."""
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _PATCH_FUNCS
        ):
            calls.append(node)
    return calls


def _collect() -> tuple[list[tuple[str, str, int]], list[tuple[str, int]]]:
    """Scan the test tree for ophir-targeting patch calls.

    Returns
    -------
    resolved_candidates : list of (target, relpath, lineno)
        String-literal targets starting with ``ophir.`` to be resolved.
    dynamic : list of (relpath, lineno)
        Patch calls whose first argument is a non-literal expression — reported
        so the coverage gap is visible, never silently skipped.
    """
    targets: list[tuple[str, str, int]] = []
    dynamic: list[tuple[str, int]] = []
    for path in sorted(TESTS_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = str(path.relative_to(TESTS_DIR))
        for call in _iter_patch_calls(tree):
            if not call.args:
                continue
            first = call.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                if first.value.split(".", 1)[0] == PACKAGE:
                    targets.append((first.value, rel, call.lineno))
            elif not isinstance(first, ast.Constant):
                dynamic.append((rel, call.lineno))
    return targets, dynamic


_TARGETS, _DYNAMIC = _collect()


def test_collector_finds_patch_targets() -> None:
    """The collector must actually find targets (guards a broken AST walk).

    Without this, a regression in :func:`_collect` would yield zero
    parametrized cases and the guard would pass vacuously.
    """
    assert _TARGETS, "no ophir patch targets collected — AST scan is broken"


@pytest.mark.parametrize(
    ("target", "relpath", "lineno"),
    [pytest.param(t, r, n, id=f"{r}:{n}:{t}") for t, r, n in _TARGETS],
)
def test_patch_target_resolves(target: str, relpath: str, lineno: int) -> None:
    """Every string-literal ophir patch target still resolves to an attribute."""
    try:
        _resolve_target(target)
    except (ImportError, AttributeError) as exc:
        pytest.fail(
            f"{relpath}:{lineno}: patch target {target!r} no longer resolves "
            f"({exc}). If the code moved, update the patch to where the name now "
            f"lives (patch-where-used)."
        )


def test_dynamic_targets_are_reported_not_silently_skipped() -> None:
    """Non-literal patch targets cannot be statically checked — surface them.

    This test never fails; it documents (via captured output) any patch call
    whose target is computed at runtime, so the static guard's coverage gap is
    explicit rather than silent.
    """
    if _DYNAMIC:
        print(
            "Note: "
            + str(len(_DYNAMIC))
            + " patch call(s) use non-literal targets (not statically checked): "
            + ", ".join(f"{r}:{n}" for r, n in _DYNAMIC)
        )


def test_resolver_accepts_a_valid_target() -> None:
    """Self-check: a known-good target resolves without raising."""
    _resolve_target("ophir.ticker.datasets.get_worker_info")


def test_resolver_rejects_a_missing_attribute() -> None:
    """Self-check: a moved/removed name is detected (the bug class this guards)."""
    with pytest.raises((ImportError, AttributeError)):
        _resolve_target("ophir.ticker.datasets.this_name_does_not_exist")
