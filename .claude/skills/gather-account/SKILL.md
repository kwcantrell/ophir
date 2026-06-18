---
name: gather-account
description: >-
  Use when you need the live Alpaca PAPER trading account snapshot -- equity, cash,
  drawdown, daily loss, or current positions (per-symbol market value) -- to size a
  rebalance, feed the risk-gate kill-switch, or debug a `--broker alpaca` run. Reuse
  AlpacaPaperBroker.get_account / get_positions from execute.py rather than calling
  alpaca-py directly. This source is paper-only (paper=True is hardcoded; there is no
  live path), so reach for it whenever account state, buying power, or held positions
  must come from broker truth instead of local memory.
---

# Gather Alpaca paper account

Fetch the live Alpaca **paper** account snapshot (equity, cash, drawdown, daily loss)
and current positions, for sizing and the drawdown kill-switch. Data-input catalog:
see [docs/data-inputs.md](docs/data-inputs.md).

## Source
- **Provider:** Alpaca **paper** trading API (`paper-api.alpaca.markets`) via the
  official `alpaca-py` SDK (`alpaca.trading.client.TradingClient`, lazily imported).
- **Paper-only:** `paper=True` is hardcoded at [agent/execute.py:137](src/ophir/agent/execute.py:137) -- there is no live endpoint in this code.
- Existing functions to reuse (do **not** re-implement against alpaca-py):
  - [agent/execute.py:140](src/ophir/agent/execute.py:140) `AlpacaPaperBroker.get_account() -> Account`
  - [agent/execute.py:152](src/ophir/agent/execute.py:152) `AlpacaPaperBroker.get_positions() -> dict[str, Decimal]`
  - constructor [agent/execute.py:127](src/ophir/agent/execute.py:127) `AlpacaPaperBroker(settings=None)`
- Credentials are `SecretStr | None` on `AgentSettings` -- `alpaca_key_id` /
  `alpaca_secret_key` at [agent/config.py:60](src/ophir/agent/config.py:60), sourced
  from `AGENT_ALPACA_KEY_ID` / `AGENT_ALPACA_SECRET_KEY` (env prefix `AGENT_`).

## Fields
- `Account` ([agent/execute.py:59](src/ophir/agent/execute.py:59)), all `Decimal`/`float`:
  - `equity: Decimal` -- from `acct.equity`.
  - `cash: Decimal` -- from `acct.cash`.
  - `daily_loss: float` -- `(last_equity - equity) / last_equity`, clamped to `>= 0`.
  - `drawdown: float` -- mirrors `daily_loss` (same clamped value; intraday only, no peak tracking).
- `get_positions()` -> `dict[str, Decimal]`: `{pos.symbol: Decimal(pos.market_value)}`
  for every open position (long market value in dollars; empty dict when flat).

## Fail-safe & caching
- **Missing keys -> hard fail:** constructing `AlpacaPaperBroker` with either key
  unset raises `ValueError` ([agent/execute.py:130](src/ophir/agent/execute.py:130)) --
  it never silently falls back to the in-process `PaperBroker`.
- **No caching:** every call hits Alpaca live; settings are cached once per process by
  `get_settings()` ([agent/config.py:81](src/ophir/agent/config.py:81)), the account is not.
- **Per-order isolation:** account/position reads are not retried, but order *submission*
  fails safe -- a single rejected order is logged (`order_failed`) and the rebalance
  continues ([agent/execute.py:255](src/ophir/agent/execute.py:255)).
- Keys are loaded from the gitignored `ophir-bot/.env` into the process by
  `rebalance.ps1` (the ophir repo intentionally ships no `.env`); never commit keys.

## How to use / extend
- **CLI:** `ophir trade <SYMBOLS...> --broker alpaca` -- selects `AlpacaPaperBroker`
  ([cli.py:409](src/ophir/cli.py:409)) and reads the account + positions before
  reconciling. Stays dry-run unless `--execute` is passed. `--broker paper` uses the
  in-process simulator instead (no network, no keys).
- **In code:**
  ```python
  from ophir.agent.execute import AlpacaPaperBroker
  broker = AlpacaPaperBroker()          # raises ValueError if AGENT_ALPACA_* unset
  acct = broker.get_account()           # Account(equity, cash, drawdown, daily_loss)
  positions = broker.get_positions()    # {"AAPL": Decimal("1234.56"), ...}
  ```
- **Add a sibling account source the same way:** add a new class implementing the
  `Broker` Protocol ([agent/execute.py:69](src/ophir/agent/execute.py:69)) --
  `get_account() -> Account`, `get_positions() -> dict[str, Decimal]`, `submit_order(order)` --
  add its `AGENT_*` `SecretStr` field to `AgentSettings`, then branch on a new `--broker`
  value at [cli.py:409](src/ophir/cli.py:409). Keep money math in `Decimal` and stay paper-first.
