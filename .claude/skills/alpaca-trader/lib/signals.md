# Signals & sleeves

Each candidate gets a blended score in [-1, 1] from three components, each first
normalized to [-1, 1] (`src/ophir/trading/signals.py`):

- **ophir** — model forecast (relative close return). May be absent (no
  checkpoint / uncovered name); the blend renormalizes over the remaining
  weights. Never fabricate an ophir value when unavailable.
- **momentum** — from recent bars (e.g. trend / rate-of-change).
- **sentiment** — soft signal from `get_news`; never the sole basis for a trade.

## Sleeve weights (`signals.py`)

- **Core** (S&P 500, weeks–months): `CORE_WEIGHTS` = ophir 0.6 / momentum 0.25 /
  sentiment 0.15. ophir-led.
- **Tactical** (movers/news discovery, days–weeks): `TACTICAL_WEIGHTS` =
  ophir 0.2 / momentum 0.5 / sentiment 0.3. Technicals-led.

## Sleeve allocation

Core ≤ 50% of equity, tactical ≤ 30%, cash ≥ 20% — enforced by the gate, not by
the analyst. Analysts propose; the gate sizes.
