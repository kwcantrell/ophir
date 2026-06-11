"""Structured logging and an append-only audit trail for the trading agent.

Wraps ``structlog`` so each agent decision (forecast, pick, order) is written as
one JSON line to ``<DATA_DIR>/agent-audit.jsonl`` for later replay. Named
``audit`` (not ``logging``) to avoid shadowing the stdlib module.
"""

from __future__ import annotations

import atexit
from pathlib import Path
from typing import Any, TextIO

import structlog

_configured = False
_atexit_registered = False
_log_file: TextIO | None = None


def _default_audit_path() -> Path:
    from ophir.register import DATA_DIR

    return Path(DATA_DIR) / "agent-audit.jsonl"


def _close_log_file() -> None:
    global _log_file
    if _log_file is not None:
        _log_file.close()
        _log_file = None


def configure(audit_file: str | Path | None = None) -> None:
    """Route structlog to an append-only JSON audit file.

    Parameters
    ----------
    audit_file : str or pathlib.Path, optional
        Destination JSONL file. Defaults to ``<DATA_DIR>/agent-audit.jsonl``.
    """
    global _configured, _atexit_registered, _log_file
    path = Path(audit_file) if audit_file is not None else _default_audit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _close_log_file()
    _log_file = path.open("a", encoding="utf-8")
    if not _atexit_registered:
        atexit.register(_close_log_file)
        _atexit_registered = True
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.WriteLoggerFactory(file=_log_file),
        cache_logger_on_first_use=False,
    )
    _configured = True


def get_logger(name: str = "agent") -> Any:
    """Return a structlog logger, configuring the audit sink on first use."""
    if not _configured:
        configure()
    return structlog.get_logger(name)


def log_event(event: str, **fields: Any) -> None:
    """Append one structured event as a JSON line to the audit trail."""
    get_logger().info(event, **fields)
