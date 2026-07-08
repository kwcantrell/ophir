# program.md — human search directives for the autoresearch loop

You (the proposer) improve a BERT-style masked transformer that forecasts
three forward OHLC targets per day. **Objective: maximize `rank_ic_near`** —
pooled daily cross-sectional Spearman rank-IC of `r_close` at trading-day
offsets 1–5, on a fixed held-out split. Current baseline is iteration 0 of
`results.tsv`; the multi-seed reference point is ≈ +0.066.

## Ground rules

- One focused change per iteration. Keep `train_experiment.py` runnable and
  self-consistent; your run is killed at 10 minutes.
- You may inline any ophir component into `train_experiment.py` (e.g. copy a
  method into `ExperimentPredictor` and modify it) — but never edit files
  under `src/ophir`.
- If you give `ExperimentPredictor` its own `__init__`, it MUST call
  `super().__init__(...)` and `self.save_hyperparameters()`, or your
  checkpoint cannot be reloaded for scoring and the trial is wasted.
- Never touch the sealed `from _sealed import ...` line; never write
  year-like literals (the split lives in the pinned `_sealed.py`).
- Never construct `StockHandler` directly; go through `build_split_handlers`
  (the loop rejects direct `StockHandler` references).
- Simplicity rule: a marginal gain does not justify added complexity. On a
  near-tie, prefer the simpler variant. Reverting a kept-but-marginal
  complexity increase is a valid proposal.
- Every feature must be knowable strictly before the prediction timestamp.
  Never introduce anything that peeks into the response block.

## Known results (do not re-litigate)

- `rezero_lr` dominates hyperparameter importance; `lr`, `loss_decay` matter.
- `rezero_init` tuning does NOT help (multi-seed confirmed). Do not tune it.
- Skill concentrates at offsets 1–5 and dies by offset ~10; the pooled
  90-day objective dilutes it. That is WHY the metric is `rank_ic_near`.
- Plain hyperparameter grid-walking is the Optuna sweep's job, not yours —
  only propose a hyperparameter change with a mechanistic rationale.

## Promising directions (highest leverage first)

1. **Near-horizon loss shaping.** The loss's time-decay (`loss_decay`)
   currently spreads weight across all 90 response days. Concentrate
   training signal on offsets 1–5 (steeper decay, truncated weighting, or a
   dedicated near-band loss term).
2. **Rank the cross-section, don't regress it.** Add a pairwise/listwise
   ranking term on `r_close` within each day's cross-section — the decision
   is "long the top names", so ranking loss aligns training with use.
3. **Response-block framing.** A shorter effective horizon (smaller
   `RESPONSE_SIZE`, keeping eval offsets 1–5 intact) may stop far-horizon
   noise from dominating gradients.
4. **Feature-side ideas** with strict causal lagging (e.g. volatility
   normalization of returns before embedding).
5. Architecture changes last — the evidence says the ceiling is framing,
   not capacity.

## Measurement honesty (why some wins don't count)

- Acceptance needs `rank_ic_near > best + ε` (ε set by the runner; your
  10k-step single-seed measurement is noisy — most true small gains will
  not clear it, and that is intentional).
- A 10k-step win can be a proxy artifact; champions face multi-seed and
  full-budget re-runs at graduation. Prefer changes with a mechanism, not
  a lucky number.

## When stuck

If 3+ consecutive proposals are discarded, switch families (e.g. from loss
shaping to ranking) rather than iterating on the failed idea; consider a
revert-to-simpler proposal if recent kept changes look like noise.
