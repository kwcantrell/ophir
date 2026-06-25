# Reference: ML / return forecasting

Deep treatment of machine-learned return prediction for trading. Read this when
designing or training a signal model, choosing a loss, engineering features, or
deciding whether a reported method is worth adopting.

Citation keys (`[F1]`…) resolve in **Sources** at the bottom. Each finding is
tagged **[verified]** (survived this project's 3-vote adversarial research check)
or **[canon]** (durable, widely-established; added for completeness).

---

## 1. Optimize the decision, not the forecast — `[F1]` **[verified]**

The strongest 2023–2026 theme is **decision-focused / end-to-end learning**:
train the model against the objective you actually care about — realized,
cost-aware, risk-adjusted return — rather than a proxy like next-day MSE and then
optimizing a portfolio on top.

- A forecasting model trained to minimize MSE creates an **objective mismatch**:
  it spends capacity getting the easy, large, market-driven moves right (which
  you can't trade) and under-weights the small cross-sectional differences that
  actually drive a long/short book. `[F1]`
- End-to-end deep portfolio managers that train the policy *directly* against net
  risk-adjusted return (e.g. DeePM-style architectures) report better realized
  performance than predict-then-optimize pipelines on the same data. `[F1]`

**Practical translation.** If you can make the loss differentiable through the
sizing/selection step, do it. If you can't, at least choose a *forecast* loss
whose minimizer aligns with the decision (next item).

## 2. Rank, don't regress, for cross-sectional selection — `[F3]` **[verified]**

When the decision is "go long the top names and short the bottom," the right
objective is a **ranking-oriented loss** (pairwise or listwise learning-to-rank),
not pointwise MSE.

- Ranking losses optimize the *ordering* of names within each cross-section,
  which is exactly what a market-neutral or top-decile strategy monetizes. `[F3]`
- Transformer-based stock ranking on S&P 500 data shows that the **choice of
  ranking loss** (pairwise/listwise) materially changes downstream selection
  quality — the loss is a first-class design decision, not a detail. `[F3]`
- Evaluate with **rank-IC / Spearman IC per cross-section**, not pooled R².
  Pooled metrics dilute the per-day signal (a model can have strong daily IC that
  a pooled number hides — this is exactly ophir's near-horizon dilution finding).

## 3. Transformers for sequences — useful, but discipline the inputs — `[F4]` **[verified/canon]**

Transformer/attention models are now the default sequence architecture for
financial time series, and several 2025–2026 variants report out-of-sample gains:

- **SERT** (single-directional representation transformer) and **SIT**
  (Signature-Informed Transformer) report positive out-of-sample R² on equities,
  with attention capturing temporal-sparsity patterns. `[F4]`
- The decisive design choice is a **strict causal / "directed delay" (Causal
  Sieve)** input pipeline: every feature feeding a prediction at time *t* must be
  knowable strictly before *t*. A single one-bar-ahead peek is the most common
  silent leak and inflates every metric. `[F4]`

**Caveat (read this):** these are largely **un-replicated preprints** with
self-reported gains on specific universes/horizons. Treat any single "we beat
SOTA by X" as a hypothesis to replicate, not a result to trust. Architecture is
rarely the binding constraint — data quality, causal lagging, the loss, and the
operating horizon usually dominate.

## 4. Feature engineering & regime awareness — **[canon]**

- **Cross-sectional standardization per timestamp** (rank or z-score within each
  day's universe) removes market-wide drift and is what makes ranking losses and
  IC meaningful.
- **Stationarity vs. memory trade-off:** differencing to enforce stationarity
  destroys predictive memory. Fractional differentiation (López de Prado) keeps
  maximum memory subject to a stationarity constraint — prefer it to integer
  differencing when a feature has long memory. `[F-LdP]`
- **Regime detection** (volatility regimes, trend/mean-revert states) is best used
  as a *conditioning* input or a gate on position size, not as a hard on/off
  switch — regime labels are themselves noisy and prone to lookahead if fit on
  the full sample.
- **Ensembling** across seeds/horizons/architectures reduces variance, but
  ensembling *over a sweep you also selected on* re-introduces selection bias —
  keep the ensemble decision out-of-sample (see `backtesting.md`).

## 5. What did NOT survive verification — anti-hype — `[F5]` **[refuted]**

Two widely-repeated claims **failed** this project's adversarial check (0/3 votes
to confirm) and should be treated skeptically:

- **"Feeding LLM-generated formulaic alpha features into downstream models
  significantly improves predictive accuracy."** Did not hold up robustly across
  the model families tested (Transformer, LSTM, TCN, SVR, RF). LLM-proposed
  formulaic alphas are a search heuristic, not a free accuracy boost; any
  candidate must clear the same multiple-testing bar as a hand-built feature. `[F5]`
- **"Over 90% of academic trading strategies fail in live trading"** — a popular
  rhetorical stat that could not be substantiated to the cited source. The
  *direction* (academic backtests overstate live performance) is well-supported by
  the backtest-overfitting literature; the specific number is not. Cite the
  mechanism (selection bias, costs, decay), not the folk statistic.

## Asset-class notes

- **Equities:** the cross-sectional ranking framing above is the mainstream and
  best-supported setting.
- **Crypto:** add funding-rate, perp-vs-spot basis, and on-chain features;
  beware 24/7 trading (no overnight gap, different bar conventions) and far
  thinner survivorship-clean history. `[X1]`
- **Futures/FX:** model the **continuous-contract roll** explicitly — a naive
  stitched series injects phantom returns at each roll. Trend/time-series momentum
  remains the most durable, widely-replicated futures signal. `[X2]`

---

## Sources

- `[F1]` Decision-focused / end-to-end deep portfolio management; MSE objective
  mismatch — https://arxiv.org/pdf/2510.03129 **[verified]**
- `[F3]` Transformer-based cross-sectional stock ranking; pairwise/listwise
  ranking-loss choice — https://arxiv.org/html/2510.14156v1 and
  https://arxiv.org/pdf/2505.01575 **[verified]**
- `[F4]` SERT / Signature-Informed Transformer; strict causal "directed delay"
  inputs; temporal-sparsity attention — https://arxiv.org/html/2512.16251v1 and
  https://arxiv.org/pdf/2601.05975 **[verified]**
- `[F5]` LLM-generated formulaic alphas did **not** robustly improve accuracy —
  https://arxiv.org/pdf/2508.04975 **[refuted — cite as cautionary]**
- `[X1]` Crypto-specific ML methods and pitfalls —
  https://arxiv.org/pdf/2512.22476 and
  https://www.frontiersin.org/journals/blockchain/articles/10.3389/fbloc.2026.1811716/full
- `[X2]` Futures/FX methods and roll/continuation pitfalls —
  https://arxiv.org/html/2602.00776v1 and https://www.mdpi.com/2227-7072/14/5/103
- `[F-LdP]` Fractional differentiation, cross-sectional features —
  Marcos López de Prado, *Advances in Financial Machine Learning* (Wiley, 2018) **[canon]**
