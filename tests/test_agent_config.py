"""Tests for the agent config (pydantic-settings) and the audit log."""

import json

import pytest
from pydantic import ValidationError

from ophir.agent import audit
from ophir.agent.config import AgentSettings


def test_defaults_are_paper_safe():
    s = AgentSettings()
    assert s.mode == "paper"
    assert s.dry_run is True
    assert s.allow_live is False
    assert s.top_k == 5
    assert s.horizon_days == 90


def test_live_without_allow_live_raises():
    with pytest.raises(ValidationError, match="allow_live"):
        AgentSettings(mode="live")


def test_live_with_allow_live_ok():
    s = AgentSettings(mode="live", allow_live=True)
    assert s.mode == "live"


def test_env_override(monkeypatch):
    monkeypatch.setenv("AGENT_TOP_K", "7")
    assert AgentSettings().top_k == 7


def test_audit_log_writes_json_line(tmp_path):
    audit_file = tmp_path / "audit.jsonl"
    audit.configure(audit_file=str(audit_file))
    audit.log_event("unit_test", symbol="AAA", value=1)

    lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["event"] == "unit_test"
    assert rec["symbol"] == "AAA"
    assert rec["value"] == 1
