---
name: alpaca-trader
description: >-
  Run the daily paper-trading loop over the Alpaca MCP for the ophir account:
  a morning pass that screens/analyzes/verifies and places gated orders, and an
  evening pass that scores outcomes and updates the memories knowledge base. Use
  when the user says "run the morning trade pass", "run the evening review",
  "trade today", or "do the daily trading routine".
---

# alpaca-trader

Mixed-strategy (core + tactical) AI trading on an Alpaca **paper** account, with
a non-overridable safety gate and an entity-organized memory base. Deterministic
logic lives in `ophir.trading` (called via `ophir trade …`); analysis fans out
through two `Workflow` scripts in `workflows/`.

## Invariants (never violate)

1. **Paper only** for now: `config.json` has `account_mode: "paper"`. The gate
   rejects on any account-mode mismatch — do not edit config to bypass it.
2. **Every order passes `ophir trade gate`** before placement. Honor REJECT
   (skip) and RESIZE (place the smaller `approved_notional`).
3. **No order placement inside Workflow agents.** Workflows return proposals;
   only the main agent (you) places orders via the Alpaca MCP.
4. Read `lib/safety.md` and `lib/signals.md` before acting.
5. **Account-mode interlock uses only the live connection, never config.**
   `account_mode` passed to `build_snapshot` MUST reflect the ACTUAL connected
   Alpaca account as returned by `get_account_info`. Do NOT copy it from
   `config.json`. If you cannot positively confirm the connected account matches
   `config.account_mode`, STOP and ask the operator rather than trading.
6. **SELL orders must not exceed the currently held quantity for that symbol.**
   Phase 1 is long-only; do not open shorts. The gate auto-approves SELLs and
   does not check this — the agent is responsible for the size cap.

## Arguments

Parse `$ARGUMENTS`: `morning` runs the act pass, `evening` runs the learn pass.
If neither is given, ask which pass to run.

## Morning pass (act)

1. **Confirm trading day:** `get_clock`. If the market is closed today, stop.
2. **Account + positions:** `get_account_info` (equity, cash, day P&L),
   `get_all_positions`. Resolve each position's sector (use `get_asset` or a
   cached map). Read the account type from `get_account_info` directly — this
   is the `account_mode` you MUST pass to `build_snapshot` (see Invariant 5).
3. **Tactical flatten guardrail:** compute the day's loss fraction:
   `day_loss_frac = -day_pl / equity` (positive when losing). If
   `day_loss_frac >= config.limits.flatten_tactical_day_loss_pct` (default 4%),
   place SELL orders to close every tactical-sleeve position (SELLs at current
   held quantity are auto-approved by the gate — see `lib/safety.md`), then skip
   all new entries for the day and go directly to step 9 (summarize).
4. **Load ledger sleeve tags:** read this month's ledger via
   `python -m ophir.trading.ledger`-style load, or just pass records to the
   snapshot builder:
   ```bash
   uv run python -c "import json,sys; from ophir.trading.exposure import build_snapshot, PositionInput; from ophir.trading.types import AssetClass; from ophir.trading.ledger import load_decisions; \
   # build PositionInput list from the MCP positions you gathered, then:
   # print(json.dumps(<AccountSnapshot as dict>))"
   ```
   In practice: construct a `snapshot.json` (matching `AccountSnapshot` fields)
   using `build_snapshot` so the gate can consume it.
5. **ophir forecasts:** attempt `load_forecasts(symbols, model_dir)`; if it
   returns `{}` (no checkpoint), proceed without the ophir component.
6. **Screen → seed candidates:** core = top/bottom ophir names if available else
   liquid S&P 500 names of interest; tactical = `get_most_active_stocks` /
   `get_market_movers` / `get_news`. Trim to `shortlist_size`.
7. **Run analysis workflow:** call the `Workflow` tool with
   `scriptPath: ".claude/skills/alpaca-trader/workflows/morning.js"` and `args`
   per its contract (date, depth knobs, sleeves, ophirForecasts, seedCandidates,
   memoryNotes). It returns `{ proposals }`.
8. **Gate + place each proposal:** for every proposal, write `order.json` and the
   current `snapshot.json`, then:
   ```bash
   uv run ophir trade gate --config .claude/skills/alpaca-trader/config.json \
     --order order.json --snapshot snapshot.json
   ```
   - REJECT (non-zero exit) → skip, note the reason.
   - APPROVE/RESIZE → place via `place_stock_order` / `place_option_order` using
     `approved_notional` (convert to qty/contracts via latest price). Update the
     running snapshot's exposures so the next proposal is gated against the new
     state. SELL orders must not exceed the currently held quantity (Invariant 6).
9. **Record:** for each placed order, append a `DecisionRecord` to the ledger:
   ```bash
   uv run ophir trade record --ledger-dir memories/ledger --month <YYYY-MM> \
     --decision decision.json
   ```
10. **Summarize** what was placed, skipped (with gate reasons), and current
    exposure vs. the caps.

## Evening pass (learn)

1. **Pull results:** `get_orders` (fills), `get_all_positions`,
   `get_portfolio_history`, `get_account_activities`.
2. **Score open/closed theses:** for each open ledger record, get a mark price
   (`get_stock_snapshot`) and compute the outcome with `score_record`
   (`ophir.trading.outcomes`). Skip any record whose `entry_price` is None —
   it cannot be scored and `score_record` raises on it. Build the `openTheses`
   array (include `realized_return`, `predicted_ophir`, `correct`).
3. **Run reflection workflow:** call `Workflow` with
   `scriptPath: ".claude/skills/alpaca-trader/workflows/evening.js"` and
   `args.openTheses`. It returns `{ updates }`.
4. **Apply ledger closures:** for positions that closed, update the ledger:
   ```bash
   uv run ophir trade close --ledger-dir memories/ledger --month <YYYY-MM> \
     --order-id <id> --status closed --realized-pl <pnl>
   ```
5. **Apply memory updates:** for each update, upsert the section into the right
   file (`memories/tickers/<SYM>.md`, `memories/sectors/<sector>.md`,
   `memories/patterns.md`, or `memories/lessons.md`):
   ```bash
   uv run python -c "from ophir.trading.memory import read_memory, write_memory, upsert_section; \
   p='memories/tickers/AAPL.md'; write_memory(p, upsert_section(read_memory(p), 'Thesis review', '<body>'))"
   ```
6. **Refresh performance:** build an equity curve from `get_portfolio_history`
   and run:
   ```bash
   uv run ophir trade performance --equity-curve curve.json --out memories/performance.md
   ```
7. **Summarize** hit-rate, calibration, and notable lessons. Track Phase-1
   progress (hit-rate + calibration) toward the paper→live graduation bar.

## Depth tiers

`config.json -> depth` (`lean`/`balanced`/`deep`) and `shortlist_size` /
`verify_votes` scale the morning fan-out. Start `lean`; raise once the loop is
proven. See the design spec for cost ballparks.
