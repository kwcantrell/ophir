# Reference: risk sizing & execution

Deep treatment of turning a signal into positions without blowing up: position
sizing, volatility targeting, portfolio construction, transaction-cost modeling,
risk gating, and paper→live discipline.

Citation keys (`[R1]`…) resolve in **Sources**. Tags: **[verified]** survived
this project's adversarial research check; **[canon]** durable/established.

---

## 1. Position sizing — never bet full Kelly — `[R1]` **[canon]**

The **Kelly criterion** maximizes long-run log-wealth growth: bet a fraction
proportional to edge/variance (`f* ≈ μ/σ²` in the continuous approximation).
Two hard rules from practice:

- **Use fractional Kelly (¼ to ½).** Full Kelly is the *growth-optimal* bet only
  if your edge and variance estimates are exact. They never are. Estimation noise
  makes full Kelly systematically over-levered; the growth penalty for betting
  *half* Kelly is small (~25% of growth rate) while the variance reduction is
  large. Half-Kelly is a standard default. `[R1]`
- **Kelly assumes you can re-bet continuously and survive drawdowns.** With
  fat tails and discrete rebalancing, the realized optimal fraction is *below* the
  theoretical one. Cap per-name and per-sleeve exposure independently of Kelly.

## 2. Volatility targeting & risk parity — size by risk — `[R2]` **[verified/canon]**

Prefer sizing by **risk contribution**, not dollar amount:

- **Volatility targeting:** scale gross exposure so portfolio vol hits a target
  (e.g. 10% annualized). Position weight ∝ 1/σ_i, so quieter names get more
  notional and the book's risk is stable across regimes. Vol targeting also
  improves risk-adjusted returns and tail behavior in most studied assets,
  because volatility is persistent and forecastable while returns are not. `[R2]`
- **Beware the feedback loop.** Mechanical vol targeting forces selling into
  spikes; in a crowded crash this is procyclical and can deepen drawdowns
  ("chasing your own tail"). Use slower vol estimates, floors/caps on leverage,
  and don't de-risk on a single-day spike. `[R2]`
- **Risk parity** generalizes this to multi-asset: equalize each asset's *risk*
  contribution rather than its capital. Sensible default for diversified futures
  books; sensitive to the covariance estimate (shrink it).

## 3. Transaction costs & execution — model them before believing the edge — `[R3]` **[canon]**

Most "edges" die on costs. Always net out, in this order of magnitude:

1. **Half-spread** (you cross it on entry and exit).
2. **Market impact** — temporary (you move the price as you trade) + permanent
   (your trade reveals information). The **Almgren-Chriss** framework models the
   trade-off between *impact* (trade slow) and *timing risk* (trade fast), and
   yields an optimal execution trajectory minimizing expected cost + λ·variance.
   Impact scales roughly with (order size / ADV) and with volatility. `[R3]`
3. **Slippage & fees/borrow** — commissions, exchange fees, short-borrow,
   financing/funding (crucial in crypto perps and leveraged futures).

**Backtest discipline:** never backtest at the mid with zero cost. Use a
conservative per-trade cost (spread + a size-aware impact term) and stress it
2–3×; if the edge survives, it's plausibly real. If a strategy's Sharpe collapses
when costs double, it's a cost-arbitrage illusion, not alpha.

## 4. Risk gating & guardrails — **[canon]**

A deterministic, **non-overridable** pre-trade gate is the structural way to make
sizing rules un-bypassable under pressure:

- **Hard caps:** per-name notional, per-sector/sleeve exposure, gross/net
  leverage, max position count.
- **Loss guardrails:** daily-loss flatten threshold, max drawdown halt.
- **Resize, don't just reject:** if an order breaches a cap, approve the largest
  compliant size rather than killing the trade — preserves the signal while
  honoring the limit.
- **Account-mode interlock:** verify the *live connection's* account mode, never
  trust a config file. (ophir's `safety.py` implements exactly this; honor its
  `reject`/`resize`. Never weaken or route around it.)

## 5. Portfolio construction — **[canon]**

- **Mean-variance optimization is estimation-error-maximizing** — small errors in
  expected returns produce wild, concentrated weights. Use shrinkage
  (Ledoit-Wolf covariance), position constraints, or skip MVO for simpler
  risk-based schemes (equal-risk, vol-targeted) unless you have strong, stable
  return estimates.
- **Diversify the bet, not just the names.** Correlation clustering means 30
  names can be one bet. Constrain by factor/sector exposure, not headcount.
- **Rebalance on a band, not a clock** where costs matter — rebalance when a
  weight drifts past a no-trade band, to avoid churning costs against small drifts.

## 6. Paper→live discipline — graduate on evidence — `[R4]` **[canon]**

Paper trading flatters you: no impact, perfect fills, no emotional slippage.
Define a **written graduation bar** before scaling real capital:

- A minimum track length with **out-of-sample, cost-aware** P&L (not in-sample
  backtest).
- **Calibration**, not just hit-rate: do realized outcomes match predicted
  probabilities/returns? (ophir's evening pass scores this via the ledger.)
- **Live-vs-paper slippage budget:** measure the gap between paper fills and
  modeled real fills; if live underperforms paper by more than the budget, stop
  and diagnose before adding size.
- **Scale capital in tranches** tied to sustained metrics, not on a single good
  month. Decay is the null hypothesis — most edges fade.

## Asset-class notes

- **Crypto:** funding rates dominate carry on perps; 24/7 means no overnight
  flat; exchange/counterparty risk is a real, non-market loss channel — cap
  per-venue exposure.
- **Futures/FX:** financing and roll yield are part of total return; size in
  risk (vol) units across contracts of very different notional/tick value.

---

## Sources

- `[R1]` Kelly criterion & fractional Kelly — J. L. Kelly Jr., "A New
  Interpretation of Information Rate" (1956); standard fractional-Kelly practice.
  **[canon]**
- `[R2]` Volatility targeting / tail behavior and the procyclicality caveat —
  AQR, "Chasing Your Own Tail (Risk), Revisited" —
  https://www.aqr.com/-/media/AQR/Documents/Insights/White-Papers/AQR-Chasing-Your-Own-Tail-Risk-Revisited.pdf
  and Financial Analysts Journal —
  https://www.tandfonline.com/doi/full/10.1080/0015198X.2020.1790853 **[verified]**
- `[R3]` Optimal execution & market impact — Almgren & Chriss, "Optimal Execution
  of Portfolio Transactions" —
  https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf **[canon]**
- `[R4]` Paper→live discipline, calibration, capital tranching — synthesized from
  the backtest-overfitting literature (see `backtesting.md`) + practitioner
  consensus. **[canon]**
