import json
from pathlib import Path

import pytest

from ophir.trading.config import ConfigError, load_config

VALID = {
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


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data))
    return p


def test_loads_valid_config(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, VALID))
    assert cfg.account_mode == "paper"
    assert cfg.limits.max_position_pct == 0.05
    assert cfg.shortlist_size == 15


def test_rejects_bad_account_mode(tmp_path: Path) -> None:
    bad = {**VALID, "account_mode": "real"}
    with pytest.raises(ConfigError, match="account_mode"):
        load_config(_write(tmp_path, bad))


def test_rejects_missing_limit(tmp_path: Path) -> None:
    bad = {**VALID, "limits": {k: v for k, v in VALID["limits"].items() if k != "min_cash_pct"}}
    with pytest.raises(ConfigError, match="min_cash_pct"):
        load_config(_write(tmp_path, bad))


def test_rejects_non_positive_shortlist(tmp_path: Path) -> None:
    bad = {**VALID, "shortlist_size": 0}
    with pytest.raises(ConfigError, match="shortlist_size"):
        load_config(_write(tmp_path, bad))
