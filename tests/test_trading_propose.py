import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ophir.trading.cli import app
from ophir.trading.forecast import OphirForecast

runner = CliRunner()

CONFIG = {
    "account_mode": "paper",
    "depth": "lean",
    "shortlist_size": 15,
    "verify_votes": 1,
    "limits": {
        "max_position_pct": 0.05,
        "max_option_premium_pct": 0.02,
        "halt_new_entries_day_loss_pct": 0.02,
        "flatten_tactical_day_loss_pct": 0.04,
        "max_deployed_pct": 0.80,
        "min_cash_pct": 0.20,
        "max_core_pct": 0.50,
        "max_tactical_pct": 0.30,
        "max_sector_pct": 0.25,
        "max_open_positions": 15,
        "max_total_option_premium_pct": 0.10,
    },
}


def _fc(symbol: str, r_close: float) -> OphirForecast:
    return OphirForecast(symbol=symbol, r_close=r_close, upside=0.0, downside=0.0)


def _config_file(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(CONFIG))
    return cfg


def test_propose_emits_orders_with_side_and_notional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config_file(tmp_path)
    monkeypatch.setattr(
        "ophir.trading.forecast.load_forecasts",
        lambda symbols, model_dir: {"AAA": _fc("AAA", 0.05), "BBB": _fc("BBB", -0.05)},
    )
    result = runner.invoke(
        app,
        [
            "propose",
            "--symbols",
            "AAA,BBB",
            "--model-dir",
            str(tmp_path),
            "--base-notional",
            "1000",
            "--config",
            str(cfg),
        ],
    )
    assert result.exit_code == 0, result.output
    orders = {o["symbol"]: o for o in json.loads(result.stdout)}
    assert orders["AAA"]["side"] == "buy"
    assert orders["BBB"]["side"] == "sell"
    # blended = 0.6 * ophir (CORE weight); ophir saturates at +/-1 for two names.
    assert orders["AAA"]["notional"] == pytest.approx(600.0)
    assert orders["AAA"]["asset_class"] == "equity"
    assert orders["AAA"]["sleeve"] == "core"
    assert orders["AAA"]["sector"] is None
    assert orders["AAA"]["is_defined_risk"] is True
    assert orders["AAA"]["is_short_option"] is False


def test_propose_emits_empty_when_forecasts_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config_file(tmp_path)
    monkeypatch.setattr("ophir.trading.forecast.load_forecasts", lambda symbols, model_dir: {})
    result = runner.invoke(
        app,
        [
            "propose",
            "--symbols",
            "AAA,BBB",
            "--model-dir",
            str(tmp_path),
            "--base-notional",
            "1000",
            "--config",
            str(cfg),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == []


def test_propose_skips_neutral_signals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _config_file(tmp_path)
    # Identical forecasts -> zero dispersion -> neutral -> skipped (|0| <= 0.0).
    monkeypatch.setattr(
        "ophir.trading.forecast.load_forecasts",
        lambda symbols, model_dir: {"AAA": _fc("AAA", 0.01), "BBB": _fc("BBB", 0.01)},
    )
    result = runner.invoke(
        app,
        [
            "propose",
            "--symbols",
            "AAA,BBB",
            "--model-dir",
            str(tmp_path),
            "--base-notional",
            "1000",
            "--config",
            str(cfg),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == []


def test_propose_reads_symbols_from_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _config_file(tmp_path)
    sym_file = tmp_path / "syms.txt"
    sym_file.write_text("AAA\nBBB\n")
    monkeypatch.setattr(
        "ophir.trading.forecast.load_forecasts",
        lambda symbols, model_dir: {"AAA": _fc("AAA", 0.05), "BBB": _fc("BBB", -0.05)},
    )
    result = runner.invoke(
        app,
        [
            "propose",
            "--symbols",
            str(sym_file),
            "--model-dir",
            str(tmp_path),
            "--base-notional",
            "1000",
            "--config",
            str(cfg),
        ],
    )
    assert result.exit_code == 0, result.output
    assert {o["symbol"] for o in json.loads(result.stdout)} == {"AAA", "BBB"}
