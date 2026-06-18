---
name: gather-fundamentals
description: >-
  Use when you need a ticker's company fundamentals (sector, industry, marketCap,
  trailingPE/forwardPE, profitMargins, beta, dividendYield, 52-week high/low,
  currentPrice) from Yahoo Finance. Reuse gather_fundamentals(symbol) in
  src/ophir/agent/research.py — it wraps yfinance yf.Ticker(sym).info into the
  curated _FUNDAMENTAL_KEYS subset that feeds ResearchBrief.fundamentals. Reach
  for it to fetch / refresh / debug / extend the fundamentals dimension of a
  research brief.
---

# Gather fundamentals

Fetch a curated subset of Yahoo Finance company fundamentals for one ticker,
deterministically, as the fundamentals dimension of an ophir research brief.

See [docs/data-inputs.md](docs/data-inputs.md) for the full data-source catalog.

## Source
- **Provider/library:** Yahoo Finance company fundamentals via `yfinance` —
  `yf.Ticker(sym).info` (a dict of company/market metadata).
- **Reuse this function — do not write new fetch code:**
  - [agent/research.py:107](src/ophir/agent/research.py:107) — `gather_fundamentals(symbol: str) -> dict[str, Any]`. Imports `yfinance` lazily, reads `.info`, and projects it down to `_FUNDAMENTAL_KEYS`.
  - [agent/research.py:35](src/ophir/agent/research.py:35) — the `_FUNDAMENTAL_KEYS` list (the exact whitelist of returned keys).
  - [agent/research.py:288](src/ophir/agent/research.py:288) — `research_ticker` calls it to populate `ResearchBrief.fundamentals`, which is then JSON-dumped into the grounded LLM prompt.

## Fields
On success, returns a flat dict with exactly these keys (missing keys map to
`None` — `info.get(key)`):
- `sector` — GICS-style sector name (str)
- `industry` — industry name (str)
- `marketCap` — market capitalization (number)
- `trailingPE` — trailing price/earnings (number)
- `forwardPE` — forward price/earnings (number)
- `profitMargins` — net profit margin, fractional (number)
- `beta` — beta vs. market (number)
- `dividendYield` — dividend yield (number)
- `fiftyTwoWeekHigh` — 52-week high price (number)
- `fiftyTwoWeekLow` — 52-week low price (number)
- `currentPrice` — latest quoted price (number)

## Fail-safe & caching
- On any network/parse failure (the `yf.Ticker(...).info` call raises), it
  returns a single-key dict `{"error": "fundamentals unavailable (<ExceptionName>)"}`
  (e.g. `fundamentals unavailable (HTTPError)`), so the rest of the research brief
  (news, technicals) still builds. The caller never crashes on a fundamentals miss.
- An empty/falsy `.info` is coerced to `{}`, so every key returns `None` rather
  than raising.
- **No caching / rate-limit handling here** — it hits Yahoo live on every call.
  `yfinance` does its own session handling; ophir adds no TTL, retry, or staleness
  check. Treat values as point-in-time at fetch.
- Provenance: `research_ticker` records the source string `"yfinance:info"` in
  `ResearchBrief.sources` ([agent/research.py:310](src/ophir/agent/research.py:310)).

## How to use / extend
- **Direct call:**
  ```python
  from ophir.agent.research import gather_fundamentals
  funds = gather_fundamentals("AAPL")   # -> dict keyed by _FUNDAMENTAL_KEYS, or {"error": ...}
  ```
- **Via the brief:** it runs automatically inside `research_ticker(symbol, ...)`
  / `research_many([...])` — the returned `ResearchBrief.fundamentals` is this dict.
- **Add/remove a fundamental field:** edit the `_FUNDAMENTAL_KEYS` list at
  [agent/research.py:35](src/ophir/agent/research.py:35). Any valid `.info` key
  works (e.g. `priceToBook`, `enterpriseValue`); it flows straight into the brief
  and the LLM prompt with no other change.
- **Add a sibling data source the same way:** write a small best-effort
  `gather_<thing>(symbol) -> dict` that wraps the provider call in
  `try/except Exception` and returns `{"error": "<thing> unavailable (<Name>)"}` on
  failure (mirror `gather_news` at [agent/research.py:144](src/ophir/agent/research.py:144)),
  call it in `research_ticker`, thread the result into `_build_messages`, and append
  its provenance string to `sources`.
