---
name: quant-trading
description: >-
  Best-practices playbook and cited reference for quantitative trading:
  ML return-forecasting, position sizing & execution, data-source selection,
  and backtest rigor. Use when designing or evaluating a trading signal,
  choosing/validating a data feed, judging whether a backtest is trustworthy,
  sizing positions, or modeling transaction costs — across US equities, crypto,
  and futures/FX. Covers the 2023–2026 research frontier and the durable canon.
---

# quant-trading

A layered knowledge skill. **This file is the playbook** — a one-screen
checklist and anti-pattern table you apply directly. The `reference/` docs are
the deep, cited treatment of each area; read one only when you're actually
working in it (progressive disclosure).

This is a *reference* skill, not an operational one. For the live daily
paper-trading loop, use the `alpaca-trader` skill instead. The two are
complementary: `alpaca-trader` runs trades; `quant-trading` tells you whether a
signal, backtest, data source, or sizing rule is sound.

## When to read which reference

| You are… | Read |
| --- | --- |
| Designing/training a return-prediction model, choosing a loss, fighting overfit features | `reference/forecasting.md` |
| Sizing positions, setting vol targets, modeling costs/slippage, planning paper→live | `reference/risk-and-execution.md` |
| Picking or vetting a price/fundamentals/alt-data/news feed | `reference/data-sources.md` |
| Judging whether a backtest result is real or an artifact of selection | `reference/backtesting.md` |

## The playbook (highest-leverage rules)

**Forecasting**
1. **Optimize the decision, not the forecast.** End-to-end / decision-focused
   training (loss = realized net risk-adjusted return) beats predict-then-optimize.
   An MSE-accurate model can still lose money — the objective is misaligned. [F1]
2. **For cross-sectional selection, rank — don't regress.** Pairwise/listwise
   ranking losses beat MSE when the decision is "long the top, short the bottom." [F3]
3. **Enforce strict causal lagging.** Every feature must be knowable strictly
   before the prediction timestamp (causal masking / "directed delay"). The most
   common subtle leak is a feature that peeks one bar ahead. [F4]
4. **Treat headline ML results as hypotheses.** Self-reported arXiv gains are
   mostly un-replicated preprints. LLM-generated "formulaic alphas" did **not**
   robustly improve accuracy under adversarial check — don't bolt on hype. [F5]

**Risk & execution**
5. **Never bet full Kelly.** Use fractional Kelly (¼–½) or volatility targeting;
   full Kelly is wildly over-levered once estimates are noisy. [R1]
6. **Size by risk, not by dollars.** Target a portfolio volatility; let position
   size scale inversely with each name's volatility. [R2]
7. **Model costs before you believe the edge.** Subtract realistic
   spread + impact + slippage (Almgren-Chriss style) — many "edges" are
   smaller than their costs. [R3]
8. **Paper→live needs a written graduation bar**, not a vibe: a minimum live-like
   track (hit-rate + calibration + cost-aware P&L) before scaling capital. [R4]

**Data**
9. **Demand point-in-time, survivorship-free data.** If delisted names are
   missing or fundamentals aren't as-reported-then, your backtest is
   optimistically biased — often by more than the strategy's entire edge. [D1, D2]
10. **Know the feed's provenance and licensing** before depending on it: free
    feeds (yfinance, exchange REST) for prototyping; vetted vendors
    (Norgate, Sharadar/Nasdaq, Polygon) for anything you'll trade. [D3]

**Backtesting**
11. **Count your trials.** A Sharpe is meaningless without knowing how many
    configurations you tried — the expected *maximum* Sharpe of pure noise grows
    with the number of trials. [B1]
12. **Deflate the Sharpe.** Report the Deflated Sharpe Ratio (DSR) / Probabilistic
    Sharpe Ratio; it corrects for trial count, non-normality, and sample length. [B2]
13. **Use purged, embargoed CV** (CPCV) — not a naive train/test split and not
    plain walk-forward; both leak or under-test. [B3]
14. **One backtest is a research log, not evidence.** The result you cherry-picked
    is selection-biased by construction; the honest number is the deflated,
    multiple-testing-adjusted one. [B4]

## Anti-pattern table

| Anti-pattern | Why it bites | Fix |
| --- | --- | --- |
| Training to MSE, trading the top names | Accuracy ≠ P&L; objective mismatch | Decision-focused loss or ranking loss [F1, F3] |
| Feature peeks ≥1 bar ahead | Lookahead leakage inflates everything | Strict causal lag + purge/embargo [F4, B3] |
| Reporting best-of-N Sharpe | Selection bias; max-of-noise grows with N | Track N, report DSR/PSR [B1, B2] |
| Naive k-fold on time series | Adjacent train/test bars leak | CPCV with purge + embargo [B3] |
| Full Kelly / fixed share count | Over-leverage; ignores vol | Fractional Kelly + vol targeting [R1, R2] |
| Backtest with zero/flat costs | "Edge" is below the spread | Spread+impact+slippage model [R3] |
| Survivorship-biased universe | Dead losers silently dropped | PIT, delisting-inclusive data [D1] |
| Believing one arXiv SOTA number | Un-replicated, often overfit | Replicate; adversarially verify [F5] |

## How this maps to ophir

- **Safety gate (`ophir.trading.safety`)** is the non-overridable pre-trade
  guardrail — the structural embodiment of rules 5–8. Never route around it.
- **Append-only ledger (`ophir.trading.ledger`)** is the outcome-attribution
  source of truth — it's what makes an honest paper→live graduation bar (rule 8)
  measurable. Score with `outcomes.py`.
- **Forecast seam (`ophir.trading.forecast` / `load_forecasts`)** is where rules
  1–4 land: ophir's model already shows meaningful *near-horizon* cross-sectional
  IC that gets diluted ~3–5× by the long pooled horizon (see the near-horizon
  finding). A short-horizon operating point and a ranking-oriented read of the
  forecast are the high-leverage moves, not more architecture.
- **Evaluation (`ophir.evaluate`)** already reports cross-sectional rank-IC —
  keep validation purged/embargoed (rule 13) and report deflated metrics when
  comparing many checkpoints/sweeps (rules 11–12), since sweeps are exactly the
  multiple-testing machine the DSR exists to correct.

Citation keys `[F#] [R#] [D#] [B#]` resolve in the **Sources** block of the
matching reference doc. Each reference also marks **[verified]** (passed this
project's adversarial research check) vs. **[canon]** (durable, widely-established
result added for completeness).
