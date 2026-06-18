# Trading memories

Knowledge base maintained by the `alpaca-trader` skill.

- `tickers/<SYM>.md` — per-company distilled knowledge (theses, what worked).
- `sectors/<sector>.md` — per-industry knowledge.
- `patterns.md` — generalizable trading patterns that have repeated.
- `lessons.md` — mistakes and their corrections.
- `ledger/<YYYY-MM>.jsonl` — append-only decision ledger (machine source of
  truth for outcome attribution). Do not hand-edit.
- `performance.md` — rolling return vs SPY, Sharpe, drawdown, hit-rate.

Entity files are edited by section via `ophir.trading.memory.upsert_section`.
