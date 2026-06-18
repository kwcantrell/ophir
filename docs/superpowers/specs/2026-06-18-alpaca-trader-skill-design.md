# Design: `alpaca-trader` skill

**Date:** 2026-06-18
**Status:** Approved (design); pending implementation plan
**Scope:** A project-level Claude Code skill in the ophir repo that runs a
mixed-strategy AI trading agent over the Alpaca MCP, learns from outcomes into an
entity-organized memory base, and is built to graduate from paper trading to live
money.

## 1. Purpose & shape

A project-level skill at `.claude/skills/alpaca-trader/` that:

- Trades a **mixed multi-strategy** book (long-term core + tactical swing) over the
  Alpaca MCP server.
- Runs **two passes per trading day** — a morning pass that *acts* and an evening
  pass that *learns*.
- Reflects every run's results into an **entity-organized memory base**
  (`memories/*.md`) plus a structured decision ledger, so trading patterns and
  company/industry knowledge accumulate over time.
- Is designed to flip from **paper to live money** via a single, safety-interlocked
  config switch.

**Realization:** one skill doc (`SKILL.md`) plus **two committed Workflow scripts**
(`morning.js`, `evening.js`) that fan out across sleeves/candidates with adversarial
verification. **Manually triggered to start** (paper phase), structured so the same
scripts can later be wrapped in `/schedule` cloud cron without rewrite.

### Authority & rollout

- **End goal:** fully autonomous live-money trading within the guardrails in §6.
- **Start:** fully autonomous on an Alpaca **paper** account — real strategy loop,
  fake money — to build and validate the agent before any real capital is at risk.

## 2. File layout

```
.claude/skills/alpaca-trader/
  SKILL.md              # when-to-use, daily SOP, config, safety contract
  config.json           # account mode, sleeve allocations, guardrail limits, depth knobs
  workflows/
    morning.js          # act: screen -> analyze -> verify -> safety-gate -> place
    evening.js          # learn: pull fills/P&L -> score theses -> update memories
  lib/
    safety.md           # the pre-trade gate spec (single chokepoint all orders pass)
    signals.md          # how ophir + technicals + momentum + news blend
memories/
  tickers/<SYM>.md      # per-company distilled knowledge
  sectors/<sector>.md   # per-industry knowledge
  patterns.md           # learned, generalizable trading patterns
  lessons.md            # mistakes + corrections
  ledger/<YYYY-MM>.jsonl # decision ledger (structured, append-only) -- source of truth
  performance.md        # rolling metrics vs SPY, hit-rate, calibration
```

- The **ledger** is structured JSONL — machine-attributable, append-only, the source
  of truth for tying outcomes back to decisions.
- The `.md` files are the **distilled** human/agent-readable knowledge.
- `memories/` lives at the **repo root** (outside `.claude/`) so it is easy to inspect
  and version.

## 3. Two sleeves

- **Core** (position; weeks–months): universe = S&P 500 (matches ophir coverage).
  **ophir forecast is the primary signal**, with news/fundamentals as confirmation.
  Low churn. Capped at **≤50% of equity**.
- **Tactical** (swing; days–weeks): universe = dynamic discovery via most-active /
  market movers / news. **Technicals + momentum + news sentiment are primary**;
  ophir used only when it covers the name. Capped at **≤30% of equity**.
- **≥20% cash floor** always reserved.

## 4. Morning workflow (act)

1. **Pre-flight:** `get_clock` (confirm a trading day), `get_account_info`,
   `get_all_positions`; compute current sleeve exposures and the day's risk budget.
2. **Screen (cheap):** ophir top/bottom forecasts for the core sleeve, plus
   `get_most_active_stocks` / `get_market_movers` / `get_news` for the tactical
   sleeve → shortlist of ~10–20 candidates (`shortlist_size` config knob). Cost is
   bounded by shortlist size, **not** universe size.
3. **Analyze (fan-out):** one analyst agent per shortlisted candidate. Each gathers
   bars / quotes / snapshot / news, blends signals per §8, and returns a structured
   thesis: direction, conviction, entry/stop/target, requested size, and the raw
   signal values used.
4. **Verify:** N-vote devil's-advocate verification per *proposed* trade
   (`verify_votes` knob: 1 = balanced, 3 = deep). Each verifier tries to refute the
   thesis; trades that fail the vote are dropped.
5. **Safety gate:** every surviving order passes through the single pre-trade gate
   (§6), which resizes or rejects as needed.
6. **Place:** `place_stock_order` / `place_option_order` (paper account). Every placed
   order is written to the **ledger** with full thesis + signal snapshot.

## 5. Evening workflow (learn)

1. Pull `get_orders` (fills), `get_all_positions`, `get_portfolio_history`,
   `get_account_activities`.
2. **Attribute (fan-out):** for each open/closed thesis in the ledger, match
   realized/unrealized P&L back to the decision; score the thesis right/wrong and
   score **signal calibration** — did ophir's predicted return/upside/downside
   verify, and did the blend beat ophir alone?
3. **Update memories:** write distilled findings into `tickers/`, `sectors/`,
   `patterns.md`, and `lessons.md`. Dedup against existing content — update in place,
   don't append blindly.
4. **Update `performance.md`:** rolling return vs SPY, Sharpe, max drawdown,
   thesis hit-rate, signal calibration.

## 6. Safety layer (hard, non-overridable)

A **single pre-trade gate** that every order passes through, reading limits from
`config.json`. The agent cannot override these. Default limits:

- **Per-position:** ≤5% of equity at entry; options ≤2% premium-at-risk each.
- **Daily kill-switch:** halt all new entries at −2% day P&L; flatten the tactical
  sleeve at −4%.
- **Deployment:** ≤80% deployed, ≥20% cash floor; core ≤50%, tactical ≤30% of equity.
- **Concentration:** ≤25% exposure per sector, ≤15 open positions.
- **Options:** defined-risk only, **no naked short options**, total option
  premium-at-risk ≤10% of equity.
- **Account-mode assertion (paper↔live interlock):** the workflow asserts that the
  connected Alpaca account matches `config.account_mode` and refuses to run on a
  mismatch. This is the interlock guarding the paper→live switch.

Numeric limits are starting defaults, tunable in `config.json`. The *structure*
(single gate, non-overridable, mode interlock) is fixed.

## 7. Config knobs (`config.json`)

- `account_mode` (`paper` | `live`)
- Sleeve allocation percentages
- All guardrail limits from §6
- `shortlist_size`
- `verify_votes`
- Depth preset (`lean` | `balanced` | `deep`)

**Phased default:** start **lean** for cheap paper iteration while plumbing is
proven → switch to **balanced** as the daily default → reserve **deep** for periodic
deep-dives.

Approximate per-day output-token cost (combined morning + evening):

| Tier | Agents/day | ~Tokens/day |
| --- | --- | --- |
| Lean | ~5–8 | ~150–300k |
| Balanced | ~25–45 | ~600k–1.2M |
| Deep | ~50–90 | ~1.5–3M |

(MCP tool results — bars, news, option chains — are token-heavy, so these skew
higher than typical agent runs.)

## 8. Signal blend (`lib/signals.md`)

Each candidate receives a normalized score combining:

- **ophir forecast** (relative close return, intraday upside, intraday downside) when
  available,
- **trend / momentum** derived from bars, and
- **news sentiment** (soft input).

Core sleeve weights ophir highest; tactical sleeve weights technicals/sentiment
highest. When ophir is unavailable (no CUDA/checkpoint), the blend **degrades
gracefully** to the remaining signals — the workflow never crashes on a missing model.

## 9. Graduation: paper → live

- **Phase 1 (signal maturation):** optimize **thesis hit-rate + signal calibration**.
  P&L is tracked but **not gating** — a one-decision-per-day window is too noisy to
  trust early.
- **Phase 2 (graduation gate → live):** all of the following required —
  ≥60 trading days, beats SPY on a **risk-adjusted (Sharpe)** basis, max drawdown
  **<8%**, positive realized P&L, and a stable hit-rate.

Flipping `account_mode` to `live` is a deliberate manual action, gated by the §6
account-mode interlock.

## 10. Caveats / risks (acknowledged, not solved here)

- Paper-fill fidelity ≠ live fills; treat Phase-1 P&L cautiously.
- ophir inference needs CUDA + a trained checkpoint; absent → the signal degrades, it
  does not crash the run.
- MCP news/sentiment is shallow; sentiment is a soft input, **never** the sole basis
  for a trade.
- Multi-day holds and partial fills make outcome attribution fuzzy — the ledger stores
  full entry context so the evening pass can do best-effort matching.

## 11. Out of scope (this spec)

- Crypto trading (Alpaca supports it; excluded by decision).
- Intraday monitoring loops (morning/evening cadence only).
- Automated `/schedule` cron registration (designed for, deferred until the paper loop
  is proven).
- Live-money trading (gated behind Phase-2 graduation).
