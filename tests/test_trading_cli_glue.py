import json
from pathlib import Path

from typer.testing import CliRunner

from ophir.trading.cli import app

runner = CliRunner()

DECISION = {
    "date": "2026-06-18",
    "symbol": "AAPL",
    "sleeve": "core",
    "side": "buy",
    "asset_class": "equity",
    "notional": 1000.0,
    "sector": "Technology",
    "thesis": "t",
    "signals": {"ophir": 0.05},
    "entry_price": 100.0,
    "target": 110.0,
    "stop": 90.0,
    "order_id": "oid-1",
    "status": "open",
    "realized_pl": None,
    "scored": False,
}


def test_record_then_close(tmp_path: Path) -> None:
    dpath = tmp_path / "d.json"
    dpath.write_text(json.dumps(DECISION))
    r = runner.invoke(
        app,
        ["record", "--ledger-dir", str(tmp_path), "--month", "2026-06", "--decision", str(dpath)],
    )
    assert r.exit_code == 0
    assert json.loads(r.stdout)["appended"] is True

    r2 = runner.invoke(
        app,
        [
            "close",
            "--ledger-dir",
            str(tmp_path),
            "--month",
            "2026-06",
            "--order-id",
            "oid-1",
            "--status",
            "closed",
            "--realized-pl",
            "150.0",
        ],
    )
    assert r2.exit_code == 0
    assert json.loads(r2.stdout)["updated"] == 1
    lines = (tmp_path / "2026-06.jsonl").read_text().splitlines()
    rec = json.loads(lines[0])
    assert rec["status"] == "closed"
    assert rec["realized_pl"] == 150.0
    assert rec["scored"] is True


def test_performance_writes_markdown(tmp_path: Path) -> None:
    curve = tmp_path / "curve.json"
    curve.write_text(json.dumps([100000.0, 101000.0, 100500.0, 102000.0]))
    out = tmp_path / "performance.md"
    r = runner.invoke(
        app,
        ["performance", "--equity-curve", str(curve), "--out", str(out)],
    )
    assert r.exit_code == 0
    payload = json.loads(r.stdout)
    assert "total_return" in payload
    assert "sharpe" in payload
    assert "max_drawdown" in payload
    assert out.exists()
    assert "Total return" in out.read_text()
