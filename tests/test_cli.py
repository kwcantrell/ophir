"""Smoke tests for the Typer CLI wiring (no command execution)."""

import pytest
from typer.testing import CliRunner

from ophir.cli import app

runner = CliRunner()


def test_sweep_command_is_registered() -> None:
    names = {cmd.callback.__name__ for cmd in app.registered_commands if cmd.callback}
    assert "sweep" in names


def test_sweep_help_lists_key_options() -> None:
    result = runner.invoke(app, ["sweep", "--help"])
    assert result.exit_code == 0
    assert "--trials" in result.output
    assert "--confirm-top" in result.output


def test_sweep_handles_all_pruned(monkeypatch: pytest.MonkeyPatch) -> None:
    import optuna

    from ophir import sweep as sweep_mod

    empty_study = optuna.create_study(direction="maximize")
    monkeypatch.setattr(sweep_mod, "run_sweep", lambda **_: empty_study)
    result = runner.invoke(app, ["sweep", "--trials", "0", "--confirm-top", "0"])
    assert result.exit_code == 0
    assert "No trials completed" in result.output
