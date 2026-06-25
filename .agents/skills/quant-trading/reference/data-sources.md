# Reference: data sources

A survey of high-quality data feeds for quant trading, plus the quality
properties that decide whether a feed is safe to build on. Read this when picking
or vetting a price/fundamentals/alternative/news source.

Citation keys (`[D1]`…) resolve in **Sources**. Tags: **[verified]** /
**[canon]** / **[vendor]** (vendor/docs — descriptive, not adversarially tested).

---

## 1. The three quality properties that matter most — `[D1]` **[canon]**

Before features, before models, a feed must clear these or the backtest is biased:

- **Survivorship-free.** The universe must include **delisted, merged, and
  bankrupt** names *as they existed historically*. A feed that only contains
  today's live tickers silently deletes the losers — backtests on it overstate
  returns, often by more than the entire strategy edge. This is the single most
  common and most damaging data flaw. `[D1]`
- **Point-in-time (PIT) / as-reported.** Fundamentals and index membership must
  reflect what was *known on that date*, including original (un-restated) values
  and the real reporting/publication lag. Using restated financials or today's
  index constituents is lookahead leakage dressed as data. `[D2]`
- **Corporate-action correctness.** Splits, dividends, spin-offs must be adjusted
  consistently. Mixed adjusted/unadjusted series inject phantom jumps. (ophir's
  `ticker.py` does split adjustment — keep its convention consistent end-to-end.)

## 2. Price / OHLCV feeds

| Tier | Equities | Crypto | Futures/FX |
| --- | --- | --- | --- |
| **Free / prototyping** | yfinance, Alpaca (IEX), Stooq, Tiingo (free tier) | exchange REST/WS (Binance, Coinbase, Kraken), CCXT | Yahoo continuous (rough), broker demo feeds |
| **Paid / production** | Polygon, Databento, Norgate `[D3]`, Nasdaq/Refinitiv | Kaiko, Amberdata, CoinAPI, Tardis (tick/L2) | CME DataMine, Databento, Norgate Futures `[D3]` |

- **yfinance / free feeds:** fine for prototyping and teaching; **not**
  survivorship-clean, adjustment conventions vary, and intraday history is thin.
  Never graduate a strategy on free data alone. `[D-free]`
- **Norgate Data:** retail-priced, **survivorship-bias-free** equities and
  continuous futures with delisted history and clean continuous contracts — a
  common serious-retail/prosumer choice. `[D3]` **[vendor]**
- **Databento / Polygon:** programmatic, well-documented modern APIs for US
  equities/options/futures incl. tick and L2; good for PIT-correct pipelines.

## 3. Fundamentals

- **Sharadar / Nasdaq Data Link SF1 (Core US Fundamentals):** as-reported,
  **point-in-time** fundamental data with a documented reporting lag — designed
  for backtesting without lookahead, retail-affordable. A standard PIT
  fundamentals source. `[D4]` **[vendor]**
- Avoid scraping a fundamentals site that only shows *current/restated* numbers —
  it has no PIT discipline and will leak.

## 4. Alternative data

Alt-data (satellite, card transactions, web traffic, app installs, shipping,
sentiment) can carry orthogonal alpha but comes with severe traps:

- **Short history** (often <5 years) → tiny effective sample, high overfit risk.
- **Survivorship & coverage drift** in the panel itself.
- **Licensing & compliance** — MNPI risk, redistribution limits, exclusivity
  decay (alpha erodes as more buyers license the same feed).
- Treat every alt-data signal as a multiple-testing candidate: it must clear the
  *deflated* bar (see `backtesting.md`), not a raw t-stat.

## 5. News & sentiment

- Machine-readable news (Benzinga, RavenPack/Bigdata, Refinitiv News Analytics)
  and LLM-scored sentiment are increasingly used; **timestamp discipline is
  everything** — use the *publication* timestamp, not the analysis timestamp, and
  enforce the same causal lag as any other feature.
- **Sentiment is best kept human-in-the-loop or as a soft tilt**, not a hard
  signal — it's noisy, regime-dependent, and easily gamed. (This matches ophir's
  intent: sentiment stays a human/contextual input, not an automated trigger.)

## 6. Aggregators / research platforms

- **QuantConnect** and similar platforms bundle survivorship-handled equities,
  options, futures, crypto, and alt-datasets with a backtest engine — convenient
  for research, but verify each dataset's PIT/survivorship properties from its
  docs rather than assuming the platform handled it. `[D5]` **[vendor]**

## Selection checklist

1. Is it survivorship-free (delisted names present)? `[D1]`
2. Is it point-in-time / as-reported, with a real publication lag? `[D2]`
3. Are corporate actions adjusted consistently?
4. What's the license — can you use it for the intended (paper/live) purpose?
5. How long is the clean history vs. the horizon you're modeling?
6. Free feed → prototyping only; vetted vendor before any capital. `[D3, D4]`

---

## Sources

- `[D1]` Survivorship bias in trading datasets — QuantRocket, "Survivorship
  Bias" — https://www.quantrocket.com/blog/survivorship-bias/ **[verified-topic]**
- `[D2]` Point-in-time / as-reported correctness — Sharadar SF1 methodology
  (below) + López de Prado on lookahead. **[canon]**
- `[D3]` Survivorship-bias-free equities & continuous futures — Norgate Data —
  https://norgatedata.com/ **[vendor]**
- `[D4]` Point-in-time US fundamentals — Sharadar Core US Fundamentals (SF1),
  Nasdaq Data Link — https://data.nasdaq.com/databases/SF1 **[vendor]**
- `[D5]` Research-platform dataset coverage & survivorship handling —
  QuantConnect datasets overview —
  https://www.quantconnect.com/docs/v2/writing-algorithms/datasets/overview **[vendor]**
- `[D-free]` Free-feed limitations — general practitioner consensus; yfinance and
  exchange REST are prototyping tools, not production data. **[canon]**
