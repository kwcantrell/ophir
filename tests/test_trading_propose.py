import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ophir.trading.cli import app
from ophir.trading.forecast import OphirForecast

runner = CliRunner()


def _series(n: int, drift: float, base: float = 100.0, noise: float = 0.003) -> list[float]:
    closes = [base]
    for i in range(1, n):
        ret = drift + (noise if i % 2 == 0 else -noise)
        closes.append(closes[-1] * (1.0 + ret))
    return closes


@pytest.fixture(autouse=True)
def _no_momentum_data(monkeypatch: pytest.MonkeyPatch) -> None:
    # Default: no momentum closes (offline + deterministic). The momentum test
    # below overrides this with its own monkeypatch, which takes precedence.
    monkeypatch.setattr("ophir.trading.momentum.load_recent_closes", lambda symbols, base_path: {})


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


def test_propose_deduplicates_repeated_symbols(
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
            "AAA,AAA,BBB",
            "--model-dir",
            str(tmp_path),
            "--base-notional",
            "1000",
            "--config",
            str(cfg),
        ],
    )
    assert result.exit_code == 0, result.output
    orders = json.loads(result.stdout)
    # A repeated symbol must not produce a duplicate order.
    assert [o["symbol"] for o in orders] == ["AAA", "BBB"]


def test_propose_momentum_drives_order_when_ophir_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config_file(tmp_path)
    # No ophir forecasts at all -> ophir component is None.
    monkeypatch.setattr("ophir.trading.forecast.load_forecasts", lambda symbols, model_dir: {})
    monkeypatch.setattr(
        "ophir.trading.momentum.load_recent_closes",
        lambda symbols, base_path: {
            "UP": _series(80, 0.01),
            "DOWN": _series(80, -0.01),
        },
    )
    result = runner.invoke(
        app,
        [
            "propose",
            "--symbols",
            "UP,DOWN",
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
    assert orders["UP"]["side"] == "buy"
    assert orders["DOWN"]["side"] == "sell"
    # ophir=None -> blend over momentum(0.25)+sentiment(0.15); momentum=+/-1 ->
    # blended = 0.25/0.40 = 0.625 -> notional = 1000 * 0.625.
    assert orders["UP"]["notional"] == pytest.approx(625.0)


def test_propose_empty_when_no_signals_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config_file(tmp_path)
    monkeypatch.setattr("ophir.trading.forecast.load_forecasts", lambda symbols, model_dir: {})
    monkeypatch.setattr("ophir.trading.momentum.load_recent_closes", lambda symbols, base_path: {})
    result = runner.invoke(
        app,
        [
            "propose",
            "--symbols",
            "UP,DOWN",
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
