"""Tests for the SQLite per-ticker stock store."""

from ophir.sqlite_store import sanitize_table_name


def test_sanitize_basic():
    used: set[str] = set()
    assert sanitize_table_name("A", used) == "t_A"
    assert "t_A" in used


def test_sanitize_replaces_non_alphanumerics():
    used: set[str] = set()
    assert sanitize_table_name("A.WD", used) == "t_A_WD"
    assert sanitize_table_name("AAC.U", used) == "t_AAC_U"


def test_sanitize_resolves_collisions():
    used: set[str] = set()
    first = sanitize_table_name("A.WD", used)
    second = sanitize_table_name("A_WD", used)  # sanitizes to the same base
    assert first == "t_A_WD"
    assert second == "t_A_WD_2"
    assert {"t_A_WD", "t_A_WD_2"} <= used
