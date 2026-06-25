# Reference: backtesting rigor

How to tell whether a backtest result is real or an artifact of selection. This
is the area where the research is most settled and most ignored. Read it before
trusting *any* Sharpe ratio, yours or anyone else's.

Citation keys (`[B1]`…) resolve in **Sources**. Tags: **[verified]** survived
this project's adversarial research check; **[canon]** durable/established.

---

## 1. The multiple-testing problem — count your trials — `[B1]` **[verified]**

The central result: **the expected maximum Sharpe ratio of pure-noise strategies
grows with the number of trials.** If you try enough configurations, one will look
great by chance alone.

- The expected maximum Sharpe across *N* independent backtests rises roughly with
  `√(2 ln N)` (times the per-trial Sharpe standard error) — so the more you search,
  the higher the "winner" you should *expect from luck*. `[B1]`
- **Corollary: a backtest is uninterpretable unless you know how many trials
  produced it.** A Sharpe of 2 from 1 trial and a Sharpe of 2 from 10,000 trials
  are completely different evidence. Report the trial count. `[B1]`
- This is why a single, cherry-picked backtest is a **research log entry, not
  evidence**. The number you publish is selection-biased by construction. `[B4]`

## 2. The Deflated Sharpe Ratio — `[B2]` **[verified]**

The **Deflated Sharpe Ratio (DSR)** (Bailey & López de Prado) is the standard
correction. It gives the probability that the observed Sharpe exceeds what you'd
expect from the *best of N* noise trials, correcting for **three** distinct
distortions at once:

1. **Selection bias / multiple testing** — via the number of trials *N* and the
   variance of trial Sharpes.
2. **Non-normality** — skew and kurtosis of returns (fat-tailed strategies need a
   higher bar).
3. **Sample length** — short tracks are noisier.

The **Probabilistic Sharpe Ratio (PSR)** is the single-trial version (probability
the true Sharpe exceeds a benchmark given skew/kurtosis/length); **DSR** is PSR
with the benchmark set to the expected max of N trials. **Always report DSR/PSR,
not the raw Sharpe**, when any search was involved. `[B2]`

> Rule of thumb: if you can't state *N* (how many variants you tried, including
> informal ones), you can't compute DSR — and you should distrust the result.

## 3. Cross-validation that doesn't leak — `[B3]` **[verified]**

Naive k-fold CV is **invalid on time series**: adjacent train/test bars share
information (overlapping label windows, autocorrelation), so the test set leaks
into training. The fixes, in increasing order of rigor:

- **Purging:** remove training observations whose label window overlaps the test
  set. `[B3]`
- **Embargo:** additionally drop a buffer of training samples immediately *after*
  each test block, to kill serial-correlation leakage across the boundary. `[B3]`
- **Combinatorial Purged Cross-Validation (CPCV):** train/test over many
  combinations of purged, embargoed blocks, producing a *distribution* of
  out-of-sample paths rather than one. In controlled synthetic experiments, **CPCV
  recovers the truest performance estimate, while plain walk-forward performs
  worst** among common schemes (it tests on a single path and overfits to it). `[B3]`

**Hierarchy to use:** CPCV (purge + embargo) ≫ purged k-fold ≫ walk-forward ≫
naive k-fold (never). Walk-forward is intuitive and fine for a final realistic
*replay*, but it is a weak *validator* because it yields one path.

## 4. The full lookahead/leakage checklist — **[canon]**

Leakage is any information from the future (or from the test set) reaching the
model. Audit every one of these:

- **Feature lag:** is every feature knowable strictly before the prediction bar?
  (See `forecasting.md` causal masking.)
- **Label overlap:** do label windows of train and test observations overlap?
  Purge them. `[B3]`
- **Survivorship / universe:** is the historical universe PIT, including delisted
  names? (See `data-sources.md`.) `[D1]`
- **Restated data:** are fundamentals as-reported-then, not later-restated? `[D2]`
- **Normalization leakage:** were scalers/PCA/feature-selection fit on the full
  sample (incl. test)? Fit inside each train fold only. (ophir's `leakage.py`
  exists to diagnose response-block target leakage — use it.)
- **Parameter leakage:** were hyperparameters chosen by peeking at the test set?
  That's just more trials — fold it into *N* for the DSR.

## 5. Realistic costs & the "dangers of backtesting" — `[B5]` **[verified/canon]**

Even a leak-free, trial-counted backtest lies if it's frictionless:

- Apply realistic **spread + impact + slippage + fees/borrow** (see
  `risk-and-execution.md` §3). Stress costs 2–3×; if the edge dies, it was
  cost arbitrage. `[B5]`
- Beware **regime-specific overfit**: a strategy tuned on one volatility regime
  (e.g. the post-2009 bull) is not validated until it's tested across regimes.
- Beware **capacity**: an edge that works at \$10k may vanish at \$10M once your
  own impact moves the price.

## 6. A trustworthy-backtest checklist

1. State **N** (every variant tried, formal and informal). `[B1]`
2. Report **DSR / PSR**, not raw Sharpe. `[B2]`
3. Validate with **CPCV** (purge + embargo), not naive CV or lone walk-forward. `[B3]`
4. Pass the **leakage checklist** (feature lag, label overlap, PIT data, fold-local
   normalization). `[B3, D1, D2]`
5. Net out **realistic, stressed costs**. `[B5]`
6. Confirm **out-of-sample across regimes** and within **capacity**.
7. Remember: the honest number is the *deflated, multiple-testing-adjusted* one —
   not the best run you found. `[B4]`

## ophir callout

ophir's **sweep harness (`sweep.py`)** is a multiple-testing machine *by design* —
every Optuna trial is a trial in the DSR sense. When reporting a swept
configuration's edge, deflate by the number of trials. ophir's **`evaluate.py`**
already reports cross-sectional rank-IC; keep its validation purged/embargoed and
report deflated metrics when comparing many checkpoints. **`leakage.py`** is the
built-in response-block leakage diagnostic — run it as the first line of the
leakage checklist.

---

## Sources

- `[B1]` Expected maximum Sharpe grows with trial count; backtest uninterpretable
  without N — Bailey & López de Prado et al., "The Probability of Backtest
  Overfitting" / "Pseudo-Mathematics and Financial Charlatanism"; see also
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 **[verified]**
- `[B2]` Deflated Sharpe Ratio (selection bias + non-normality + sample length) —
  Bailey & López de Prado, "The Deflated Sharpe Ratio" —
  https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf and
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 **[verified]**
- `[B3]` Purging, embargo, CPCV; walk-forward worst / CPCV best in controlled
  tests — https://en.wikipedia.org/wiki/Purged_cross-validation and
  https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110 and
  https://arxiv.org/pdf/2512.12924 **[verified]**
- `[B4]` Selection bias of single backtests; reproducibility/overfitting —
  https://arxiv.org/pdf/2512.12924 **[verified]**
- `[B5]` Dangers of backtesting (costs, regimes, capacity) — Palomar,
  *Portfolio Optimization* book §8.3 —
  https://portfoliooptimizationbook.com/book/8.3-dangers-backtesting.html **[verified]**
- `[canon]` Foundational methods (fractional differentiation, purging/embargo,
  CPCV, DSR) — Marcos López de Prado, *Advances in Financial Machine Learning*
  (Wiley, 2018). **[canon]**
