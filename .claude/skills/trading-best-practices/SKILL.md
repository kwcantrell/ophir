---
name: trading-best-practices
description: >-
  Industry best practices for building/modifying the ophir stock-trading agent.
  Use whenever writing or changing code that ingests market data, generates
  trading signals, sizes positions, decides buy/sell/hold, executes orders,
  backtests a strategy, or puts an LLM in the trading loop. Enforces paper-first
  / dry-run defaults, deterministic auditable execution, a risk gate + drawdown
  kill-switch, look-ahead/survivorship-bias-free backtests, and LLM-safety rails.
---

# Trading best practices

Apply these whenever you touch the trading agent. They exist to *not lose money and
not fool yourself*. Full rationale: `ophir-bot/trading-agent-blueprint.md`.

## Non-negotiable invariants
- **Paper-first.** `mode="paper"`, `dry_run=True`, `allow_live=False` are the defaults.
  Going live needs an explicit, loud opt-in. Refuse to start against a live endpoint
  unless `allow_live` is set on purpose.
- **Fail safe, not open.** On any ambiguity (stale data, disconnect, breached limit) the
  default action is *do nothing*, never *trade anyway*.
- **Broker is the source of truth.** Never trust local memory for positions/cash;
  reconcile against the broker every cycle.
- **Deterministic, idempotent execution path.** Same inputs → same orders. Deterministic
  `client_order_id` (e.g. `f"{date}:{symbol}:{side}"`) so a retry/double-run can't
  double-trade.
- **Observability.** Every decision + order goes to an append-only audit trail; you must
  be able to answer "why did it buy X on day Y?" months later.

## Data & signal
- **No look-ahead.** The signal for day T may only use data through T-1's close. Lag every
  feature; never `.shift()` the wrong way.
- **Staleness & quality checks before inference.** Verify last bar is recent, no gaps/dupes,
  tz-consistent (ophir is tz-naive). Stale feed → skip the cycle.
- **Split/dividend adjustment must be consistent** across a window (don't mix adjusted and
  raw). Reuse ophir's `extract_features` contract — don't re-derive the 13 features.
- Use **all three** ophir targets (`r_close`, `upside`, `downside`) for a risk-aware score,
  not just `r_close`.

## Position sizing & risk
- **Volatility targeting is the default** sizing method (scale each name inversely to its
  vol; scale the book to a target annual vol, e.g. 15%). Equal-weight is only a baseline.
- **Fractional Kelly only** (¼–½). Full Kelly is too aggressive and very sensitive to
  estimation error — overestimating edge → risk of ruin.
- **Constraints:** per-name cap (≤5%), sector caps, gross/net exposure bounds, liquidity
  cap (% of ADV), turnover budget + no-trade band to cut churn.
- **Drawdown kill-switch is the single most important control** — halt new risk on a
  peak-to-trough breach (e.g. 20%) and alert a human. Add a daily-loss limit.
- **Risk Gate (pre-trade):** a hard checkpoint *after* portfolio construction and *before*
  the OMS that can **veto / scale / halt**. Property to hold: no gate output can push the
  book past any configured limit.

## Execution
- Use **`alpaca-py`**, never the deprecated `alpaca-trade-api`.
- **`Decimal`, never `float`**, for share quantities and prices (float rounding → rejected
  orders).
- **Retries only on idempotent ops** (`tenacity`, backoff + jitter). A submit that times
  out is resolved by *querying order status*, not blind resubmit.
- **Reconcile vs broker truth** each cycle; handle partial fills (re-target residual) and
  rejects (log + alert).
- **Delta-reconcile, never liquidate-and-rebuy.** Sells first (free buying power), then buys.
- Prefer **marketable-limit or MOC/LOC** over naked market orders for a daily strategy.
- **Gate on the market calendar** (`pandas_market_calendars`); respect PDT (<$25k) and T+1
  settlement (size against `buying_power`, not raw cash). Single-run lock prevents
  scheduler double-fire.

## Backtesting & validation
- **Two engines:** `vectorbt` for fast signal/parameter sweeps; a small **event-driven loop
  that reuses your signal/portfolio/risk modules** for execution realism (backtest == live
  by construction).
- **The three biases that destroy backtests:** look-ahead, **survivorship** (don't backtest
  today's S&P 500 over history — use point-in-time membership), and **overfitting /
  data-snooping** (out-of-sample holdout, walk-forward, deflated Sharpe).
- **Model real costs:** commission/fees, half-spread, slippage (`base + k·size/ADV`), market
  impact. A pre-cost edge that dies after costs is the norm.
- **Validation:** walk-forward; **purged & embargoed CV** when labels overlap in time (your
  multi-day horizon leaks under naive CV); touch the out-of-sample lockbox once.
- **Benchmark vs SPY**, report Sharpe/Sortino/MaxDD/Calmar/turnover. Pin a **golden-file
  backtest** as a regression test.

## LLM in the trading loop (critical — this build lets the LLM pick)
LLMs **hallucinate**, are **non-deterministic**, and have a **knowledge cutoff**. Research
shows their trading decisions are unstable and over-sensitive to input noise. Therefore:
- **Ground every claim in tool-fetched real data**, never model memory. Require citations
  for any number or fact. No fabricated prices/figures.
- **The LLM's picks MUST pass the deterministic Risk Gate** and stay **paper / `dry_run`**.
  An LLM never sizes or places an order that bypasses the gate.
- **Log the full rationale** (inputs, research, debate, final picks) to the audit trail.
- **Reduce instability:** sample/seed deliberately, prefer selective consensus across
  multiple runs, keep prompts structured. Treat LLM output as *advisory even when it
  "decides."*
- The **safest** pattern is LLM-advisory + human-gated. If you move toward production,
  migrate decision authority back to the deterministic model + risk rules and keep the LLM
  for research and reporting.

## Anti-patterns (reject these)
Look-ahead/survivorship/overfitting; ignoring costs; trusting local state over the broker;
fire-and-forget orders; no kill-switch; secrets in git or shared paper/live keys; skipping
paper burn-in; float share quantities; naive timezones; scheduler double-fire; trading on
stale data; **letting an LLM place/size orders without a deterministic gate.**

## Recommended libraries (2026)
`alpaca-py` (broker) · `vectorbt` + `backtesting.py`/`nautilus_trader` (backtest) ·
`quantstats`/`pyfolio-reloaded` (analytics) · `cvxpy`/`riskfolio-lib` (sizing) ·
`pydantic-settings` (config) · `structlog` (logging) · `tenacity` (retries) ·
`pandas_market_calendars` (calendar) · `hypothesis` (property tests) ·
`sqlalchemy`+sqlite/postgres (state).

## References
- `ophir-bot/trading-agent-blueprint.md` (primary).
- López de Prado, *Advances in Financial Machine Learning*; Chan, *Quantitative Trading*;
  Clenow, *Trading Evolved*.
- Backtest-bias & LLM-trading-risk web sources (2026) are catalogued in `ophir-bot/roadmap.md`.
