"""Typed configuration for the trading agent (``pydantic-settings``).

A single :class:`AgentSettings` object is the source of truth for run mode and
risk limits, sourced from ``AGENT_*`` environment variables and an optional
``.env`` file. Defaults are deliberately safe: paper mode, dry-run on, and live
trading hard-gated behind ``allow_live``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """Validated trading-agent settings with paper-first, fail-safe defaults."""

    model_config = SettingsConfigDict(env_prefix="AGENT_", env_file=".env", extra="ignore")

    # run mode
    mode: Literal["paper", "live"] = "paper"
    dry_run: bool = True
    allow_live: bool = False

    # signal / selection
    top_k: int = 5
    horizon_days: int = 90

    # risk knobs (consumed by later phases; safe defaults now)
    max_position_weight: float = 0.05
    target_gross_exposure: float = 0.95
    annual_vol_target: float = 0.15
    max_drawdown_kill: float = 0.20
    daily_loss_limit: float = 0.05

    # broker credentials (optional until the execution phase)
    alpaca_key_id: SecretStr | None = None
    alpaca_secret_key: SecretStr | None = None

    @model_validator(mode="after")
    def _enforce_paper_only(self) -> AgentSettings:
        """Refuse live mode unless ``allow_live`` is explicitly set."""
        if self.mode == "live" and not self.allow_live:
            raise ValueError(
                "Refusing live mode without allow_live=True. "
                "Paper trading only until explicitly opted in."
            )
        return self


@lru_cache
def get_settings() -> AgentSettings:
    """Return the cached :class:`AgentSettings` (loaded once per process)."""
    return AgentSettings()
