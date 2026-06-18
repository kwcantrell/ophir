---
name: gather-news
description: >-
  Use when you need recent news headlines for a ticker (e.g. AAPL, NVDA) in the
  research layer -- fetching, refreshing, debugging, or extending the news
  dimension of a research brief. Pulls Yahoo Finance headlines via yfinance and
  normalizes them through `gather_news` / `_normalize_news_item` in
  `agent/research.py`. Reach for it when a brief's `news` list is empty, when
  yfinance changes its payload shape, or when adding a sibling headline source.
---

# Gather news (Yahoo Finance headlines)

Fetch recent, normalized news headlines for one ticker to feed the grounded
`ResearchBrief.news` dimension. See [docs/data-inputs.md](docs/data-inputs.md).

## Source
- Provider/library: Yahoo Finance recent headlines via **yfinance**, read as
  `yf.Ticker(sym).news` (a list of dicts).
- Reuse these existing functions -- do **not** write a new fetcher:
  - [agent/research.py:144](src/ophir/agent/research.py:144) --
    `gather_news(symbol, limit=8) -> list[dict[str, Any]]`: best-effort fetch +
    per-item normalization. Slices `raw[:limit]` and skips non-dict entries.
  - [agent/research.py:118](src/ophir/agent/research.py:118) --
    `_normalize_news_item(item)`: collapses both yfinance payload shapes into the
    four canonical fields below.
- Called from [agent/research.py:289](src/ophir/agent/research.py:289) inside
  `research_ticker`, with `limit=settings.research_news_limit`.

## Fields
Each normalized item is a dict with exactly these keys:
- `title` -- headline text (`""` if absent).
- `publisher` -- source name.
- `link` -- canonical article URL (`""` if absent).
- `published` -- ISO-ish date string.

`_normalize_news_item` handles two raw shapes:
- **Newer nested:** `content.{title, provider.displayName, canonicalUrl.url
  (or clickThroughUrl.url), pubDate}` -- detected when `item["content"]` is a
  dict; `pubDate` is passed through as a string.
- **Older flat:** `{title, publisher, link, providerPublishTime}` -- the epoch
  `providerPublishTime` is formatted to `%Y-%m-%d` (UTC) for `published`.

Non-empty `link` values become provenance entries in `ResearchBrief.sources`
(see [agent/research.py:311](src/ophir/agent/research.py:311)).

## Fail-safe & caching
- On any fetch/parse failure `gather_news` catches the exception and returns
  `[]` -- the brief stays grounded with no news rather than crashing (matches the
  fundamentals fail-safe at [agent/research.py:113](src/ophir/agent/research.py:113)).
- No local caching or rate-limiting layer -- each call hits Yahoo Finance live;
  `limit` (default `8`) bounds how many items are kept and is the only throttle.
- Staleness: items carry only a `published` date, not a freshness guarantee; an
  empty list is the expected signal that headlines were unavailable.

## How to use / extend
- The default limit comes from `research_news_limit: int = 8`
  ([agent/config.py:46](src/ophir/agent/config.py:46)); a brief built via
  `research_ticker(symbol)` calls `gather_news(symbol, limit=settings.research_news_limit)`
  automatically.
- Standalone fetch:
  ```python
  from ophir.agent.research import gather_news
  gather_news("AAPL", limit=5)  # -> [{"title", "publisher", "link", "published"}, ...]
  ```
- To add a **sibling headline source** the same way: write a `gather_<src>` that
  fetches best-effort, returns `[]` on `except Exception`, and maps each raw
  record through a `_normalize_<src>_item` emitting the same four keys
  (`title`/`publisher`/`link`/`published`). Wire it into `research_ticker`
  alongside the `news =` line, merge into the `news` list, and append any links
  to `sources` so every datum stays cited.
