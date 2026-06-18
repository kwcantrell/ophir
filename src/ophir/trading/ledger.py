"""Append-only JSONL decision ledger — the outcome-attribution source of truth."""

import json
from collections.abc import Mapping
from pathlib import Path

from ophir.trading.types import AssetClass, DecisionRecord, Side, Sleeve


def record_to_dict(record: DecisionRecord) -> dict[str, object]:
    """Serialize a record to a JSON-safe dict (enums become their values)."""
    return {
        "date": record.date,
        "symbol": record.symbol,
        "sleeve": record.sleeve.value,
        "side": record.side.value,
        "asset_class": record.asset_class.value,
        "notional": record.notional,
        "sector": record.sector,
        "thesis": record.thesis,
        "signals": dict(record.signals),
        "entry_price": record.entry_price,
        "target": record.target,
        "stop": record.stop,
        "order_id": record.order_id,
        "status": record.status,
        "realized_pl": record.realized_pl,
        "scored": record.scored,
    }


def record_from_dict(data: Mapping[str, object]) -> DecisionRecord:
    """Inverse of :func:`record_to_dict`."""
    signals_raw = data["signals"]
    if not isinstance(signals_raw, dict):
        raise TypeError("ledger record 'signals' must be a JSON object")
    return DecisionRecord(
        date=str(data["date"]),
        symbol=str(data["symbol"]),
        sleeve=Sleeve(str(data["sleeve"])),
        side=Side(str(data["side"])),
        asset_class=AssetClass(str(data["asset_class"])),
        notional=float(data["notional"]),  # type: ignore[arg-type]
        sector=None if data["sector"] is None else str(data["sector"]),
        thesis=str(data["thesis"]),
        signals={str(k): float(v) for k, v in signals_raw.items()},
        entry_price=None if data["entry_price"] is None else float(data["entry_price"]),  # type: ignore[arg-type]
        target=None if data["target"] is None else float(data["target"]),  # type: ignore[arg-type]
        stop=None if data["stop"] is None else float(data["stop"]),  # type: ignore[arg-type]
        order_id=None if data["order_id"] is None else str(data["order_id"]),
        status=str(data["status"]),
        realized_pl=None if data["realized_pl"] is None else float(data["realized_pl"]),  # type: ignore[arg-type]
        scored=bool(data["scored"]),
    )


def ledger_path(ledger_dir: str | Path, month: str) -> Path:
    """Return the JSONL path for a ``YYYY-MM`` month."""
    return Path(ledger_dir) / f"{month}.jsonl"


def append_decision(ledger_dir: str | Path, month: str, record: DecisionRecord) -> None:
    """Append one decision as a JSON line, creating the directory if needed."""
    path = ledger_path(ledger_dir, month)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record_to_dict(record)) + "\n")


def load_decisions(ledger_dir: str | Path, month: str) -> list[DecisionRecord]:
    """Load all decisions for a month, or ``[]`` if the file does not exist."""
    path = ledger_path(ledger_dir, month)
    if not path.exists():
        return []
    records: list[DecisionRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(record_from_dict(json.loads(line)))
    return records
