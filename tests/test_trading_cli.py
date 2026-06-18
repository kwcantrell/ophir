import json
from pathlib import Path

from typer.testing import CliRunner

from ophir.trading.cli import app

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
ORDER = {
    "symbol": "AAPL",
    "side": "buy",
    "sleeve": "core",
    "asset_class": "equity",
    "notional": 1000.0,
    "sector": "Technology",
    "is_defined_risk": True,
    "is_short_option": False,
}
SNAPSHOT = {
    "equity": 100000.0,
    "cash": 50000.0,
    "day_pl": 0.0,
    "open_position_count": 0,
    "held_symbols": [],
    "symbol_exposure": {},
    "sector_exposure": {},
    "sleeve_exposure": {},
    "option_premium_at_risk": 0.0,
    "account_mode": "paper",
}


def _files(tmp_path: Path) -> tuple[Path, Path, Path]:
    c = tmp_path / "config.json"
    o = tmp_path / "order.json"
    s = tmp_path / "snap.json"
    c.write_text(json.dumps(CONFIG))
    o.write_text(json.dumps(ORDER))
    s.write_text(json.dumps(SNAPSHOT))
    return c, o, s


def test_gate_approves(tmp_path: Path) -> None:
    c, o, s = _files(tmp_path)
    result = runner.invoke(
        app, ["gate", "--config", str(c), "--order", str(o), "--snapshot", str(s)]
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["action"] == "approve"
    assert payload["approved_notional"] == 1000.0


def test_gate_rejects_nonzero_exit(tmp_path: Path) -> None:
    bad_snap = {**SNAPSHOT, "account_mode": "live"}
    c, o, s = _files(tmp_path)
    s.write_text(json.dumps(bad_snap))
    result = runner.invoke(
        app, ["gate", "--config", str(c), "--order", str(o), "--snapshot", str(s)]
    )
    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["action"] == "reject"
