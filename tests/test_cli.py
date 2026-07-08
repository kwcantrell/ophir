"""Tests for the ``ophir`` Typer CLI command wiring and behavior."""

import pytest
from click import unstyle
from typer.testing import CliRunner

from ophir.cli import app

runner = CliRunner()


def test_sweep_command_is_registered() -> None:
    names = {cmd.callback.__name__ for cmd in app.registered_commands if cmd.callback}
    assert "sweep" in names


def test_sweep_help_lists_key_options() -> None:
    result = runner.invoke(app, ["sweep", "--help"])
    assert result.exit_code == 0
    # unstyle: Rich force-styles help output when it detects CI (GITHUB_ACTIONS),
    # which would break plain-substring asserts.
    assert "--trials" in unstyle(result.output)
    assert "--confirm-top" in unstyle(result.output)


def test_sweep_handles_all_pruned(monkeypatch: pytest.MonkeyPatch) -> None:
    import optuna

    from ophir import sweep as sweep_mod

    empty_study = optuna.create_study(direction="maximize")
    monkeypatch.setattr(sweep_mod, "run_sweep", lambda **_: empty_study)
    result = runner.invoke(app, ["sweep", "--trials", "0", "--confirm-top", "0"])
    assert result.exit_code == 0
    assert "No trials completed" in result.output


def test_migrate_sqlite_is_registered():
    result = runner.invoke(app, ["migrate-sqlite", "--help"])
    assert result.exit_code == 0
    assert "--src" in unstyle(result.output)
    assert "--dst" in unstyle(result.output)
    assert "--overwrite" in unstyle(result.output)


def test_migrate_sqlite_runs(parquet_dir, tmp_path):
    base_path, paths = parquet_dir
    db_path = str(tmp_path / "stocks.db")
    result = runner.invoke(app, ["migrate-sqlite", "--src", base_path, "--dst", db_path])
    assert result.exit_code == 0
    assert f"{len(paths)} tickers written" in result.output


def test_importances_command_is_registered() -> None:
    result = runner.invoke(app, ["importances", "--help"])
    assert result.exit_code == 0
    assert "study" in unstyle(result.output).lower()
