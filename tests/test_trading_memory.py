from pathlib import Path

from ophir.trading.memory import read_memory, upsert_section, write_memory


def test_upsert_replaces_existing_section() -> None:
    doc = "# AAPL\n\n## Thesis\n\nold thesis\n\n## Notes\n\nkeep me\n"
    out = upsert_section(doc, "Thesis", "new thesis")
    assert "new thesis" in out
    assert "old thesis" not in out
    assert "keep me" in out  # other sections preserved


def test_upsert_appends_new_section() -> None:
    doc = "# AAPL\n\n## Thesis\n\nt\n"
    out = upsert_section(doc, "Risks", "earnings next week")
    assert "## Risks" in out
    assert "earnings next week" in out
    assert "## Thesis" in out


def test_upsert_on_empty_doc() -> None:
    out = upsert_section("", "Thesis", "first")
    assert "## Thesis" in out
    assert "first" in out


def test_read_missing_returns_empty(tmp_path: Path) -> None:
    assert read_memory(tmp_path / "nope.md") == ""


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "tickers" / "AAPL.md"
    write_memory(target, "# AAPL\n")
    assert read_memory(target) == "# AAPL\n"
