"""Smoke tests for the Typer CLI wiring (no command execution)."""

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
