# Safety contract

Every order MUST pass `ophir trade gate` before placement. The gate is
implemented in `src/ophir/trading/safety.py::evaluate_order` — that code, not
this doc, is authoritative.

## Hard limits (from `config.json -> limits`)

| Limit | Default | Meaning |
| --- | --- | --- |
| `max_position_pct` | 5% | Max equity in one symbol at entry. |
| `max_option_premium_pct` | 2% | Max premium-at-risk per option order. |
| `halt_new_entries_day_loss_pct` | 2% | Halt new BUYs once the day is down this much. |
| `flatten_tactical_day_loss_pct` | 4% | Evening/intraday: flatten the tactical sleeve. |
| `max_deployed_pct` | 80% | Max total deployed. |
| `min_cash_pct` | 20% | Cash floor the agent may never spend below. |
| `max_core_pct` / `max_tactical_pct` | 50% / 30% | Per-sleeve exposure caps. |
| `max_sector_pct` | 25% | Per-sector exposure cap. |
| `max_open_positions` | 15 | Position count cap. |
| `max_total_option_premium_pct` | 10% | Aggregate option premium-at-risk cap. |

## Non-negotiables

- Account-mode interlock: the gate REJECTS if the live account's mode does not
  match `config.account_mode`. This guards the paper→live switch.
- No naked short options (must be defined-risk).
- SELLs (risk-reducing) are always allowed at full size.
- The agent cannot override a REJECT or a RESIZE. A RESIZE means place the
  smaller `approved_notional`, not the requested size.

## Calling the gate

```bash
ophir trade gate --config <config.json> --order <order.json> --snapshot <snapshot.json>
```
Exit code is non-zero on REJECT. Parse stdout JSON for `action` /
`approved_notional` / `reasons`.
