"""Edit entity-memory markdown files by section (upsert), plus thin file I/O."""

from pathlib import Path


def upsert_section(markdown: str, heading: str, body: str) -> str:
    """Replace the ``## {heading}`` section's body, or append the section.

    Sections are delimited by lines equal to ``## <name>``. The body of the
    target section is replaced with ``body``; all other sections keep their
    order and content. If the heading is absent, a new section is appended.
    """
    marker = f"## {heading}"
    lines = markdown.splitlines()
    section_block = ["", marker, "", body.rstrip("\n"), ""]

    start: int | None = None
    for i, line in enumerate(lines):
        if line.strip() == marker:
            start = i
            break

    if start is None:
        prefix = markdown.rstrip("\n")
        joined = "\n".join(section_block).strip("\n")
        return (prefix + "\n\n" + joined + "\n") if prefix else joined + "\n"

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break

    new_lines = [*lines[:start], marker, "", body.rstrip("\n"), "", *lines[end:]]
    return "\n".join(new_lines).rstrip("\n") + "\n"


def read_memory(path: str | Path) -> str:
    """Return the file's text, or ``""`` if it does not exist."""
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else ""


def write_memory(path: str | Path, text: str) -> None:
    """Write ``text`` to ``path``, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
