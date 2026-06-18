"""Load and validate the alpaca-trader ``config.json`` into typed objects."""

import json
from dataclasses import fields
from pathlib import Path

from ophir.trading.types import GuardrailLimits, TradingConfig

_ACCOUNT_MODES = {"paper", "live"}
_DEPTHS = {"lean", "balanced", "deep"}


class ConfigError(ValueError):
    """Raised when ``config.json`` is missing keys or has invalid values."""


def _limits_from(data: dict[str, object]) -> GuardrailLimits:
    names = [f.name for f in fields(GuardrailLimits)]
    missing = [n for n in names if n not in data]
    if missing:
        raise ConfigError(f"limits missing keys: {', '.join(sorted(missing))}")
    kwargs: dict[str, object] = {}
    for name in names:
        value = data[name]
        if name == "max_open_positions":
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ConfigError(f"limits.{name} must be a positive int")
            kwargs[name] = value
        else:
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                raise ConfigError(f"limits.{name} must be a non-negative number")
            kwargs[name] = float(value)
    return GuardrailLimits(**kwargs)  # type: ignore[arg-type]


def load_config(path: str | Path) -> TradingConfig:
    """Parse and validate the trading config file.

    Parameters
    ----------
    path : str or Path
        Location of ``config.json``.

    Returns
    -------
    TradingConfig
        The validated configuration.

    Raises
    ------
    ConfigError
        If a key is missing or a value is out of range.
    """
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a JSON object")
    mode = raw.get("account_mode")
    if mode not in _ACCOUNT_MODES:
        raise ConfigError(f"account_mode must be one of {sorted(_ACCOUNT_MODES)}")
    depth = raw.get("depth")
    if depth not in _DEPTHS:
        raise ConfigError(f"depth must be one of {sorted(_DEPTHS)}")
    for key in ("shortlist_size", "verify_votes"):
        value = raw.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConfigError(f"{key} must be a positive int")
    limits_raw = raw.get("limits")
    if not isinstance(limits_raw, dict):
        raise ConfigError("limits must be a JSON object")
    return TradingConfig(
        account_mode=str(mode),
        limits=_limits_from(limits_raw),
        shortlist_size=int(raw["shortlist_size"]),
        verify_votes=int(raw["verify_votes"]),
        depth=str(depth),
    )
