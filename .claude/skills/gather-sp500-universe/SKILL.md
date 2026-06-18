---
name: gather-sp500-universe
description: >-
  Use when you need the S&P 500 constituent list (the ~500 ticker symbols the
  Gradio UI's ticker picker offers) — to fetch it, refresh it, or debug why the
  picker is empty/stale. Reuse ophir.ticker.get_sp_500_symbols(), which scrapes
  the Wikipedia constituents table via pandas.read_html. This is the UI-only
  universe; live paper trading uses the curated ophir-bot/watchlist.txt instead,
  so reach here for the dashboard symbol list, NOT for the trading universe.
---

# Gather the S&P 500 universe (Wikipedia)

Fetch the current S&P 500 constituent ticker symbols for the Gradio dashboard's
ticker picker. See [docs/data-inputs.md](docs/data-inputs.md) for the full data-input catalog.

## Source
- **Provider:** Wikipedia, page `List of S&P 500 companies`, scraped with
  `pandas.read_html`. URL: `https://en.wikipedia.org/wiki/List_of_S%26P_500_companies`.
- **Reuse this — do not write new scraping code:**
  - [ticker.py:108](src/ophir/ticker.py:108) — `get_sp_500_symbols() -> list[str]`.
    Reads `pd.read_html(url, storage_options={"User-Agent": ...})`, takes the
    first table (`dfs[0]`), and returns `[str(s) for s in dfs[0]["Symbol"]]`.
    A browser-like **User-Agent header is required** (Wikipedia 403s the default).
- **Call site:** imported and called at module import in
  [ui.py:39](src/ophir/ui.py:39) (`sp_500 = get_sp_500_symbols()`), then fed to
  `get_splits(sp_500)` and the dashboard ticker picker. Live trading does **not**
  call this.

## Fields
- Returns a flat `list[str]` of ~500 uppercase ticker strings (e.g. `"AAPL"`,
  `"MSFT"`, `"BRK.B"`). Only the `Symbol` column of `table[0]` is used.
- The scraped table also carries `Security`, `GICS Sector`, `GICS Sub-Industry`,
  `Headquarters Location`, `Date added`, `CIK`, `Founded` — all currently dropped.
- Note: Wikipedia uses dot tickers (`BRK.B`, `BF.B`); some downstream feeds expect
  dash form (`BRK-B`) — normalize if you forward these symbols.

## Fail-safe & caching
- **No caching, no rate limiting** — every call is a fresh live HTTP request, run
  at `ophir.ui` import time. Importing the UI hits the network.
- The `pd.read_html` call is wrapped in `try/except`; on failure it **prints the
  error and continues**, so `dfs` is unbound and the `return` then raises
  `NameError`/`UnboundLocalError`. Treat a fetch failure as a hard import failure,
  not graceful degradation. If you harden this, return an empty list (or a cached
  fallback) inside the `except` so the picker degrades gracefully.
- The list reflects Wikipedia *as of the request* (point-in-time membership is not
  preserved) — fine for the live UI picker, but do **not** backtest over it
  (survivorship bias; see the trading-best-practices skill).

## How to use / extend
- **Refresh / inspect from a REPL:**
  ```python
  from ophir.ticker import get_sp_500_symbols
  symbols = get_sp_500_symbols()   # live HTTP; needs network
  len(symbols), symbols[:5]
  ```
- **Empty/stale picker?** Check network reachability to Wikipedia and that the
  User-Agent header is still being passed via `storage_options`; a bare
  `read_html` will 403.
- **Live trading universe is separate:** the operational watchlist is the
  hand-curated `ophir-bot/watchlist.txt` (~28 liquid large caps, one ticker per
  line; `#` comments and blanks ignored). Edit that file to change what the bot
  scores — not this scraper.
- **Add a sibling source the same way:** write a small `get_<universe>_symbols()
  in `ticker.py` returning `list[str]` (e.g. NASDAQ-100 from its Wikipedia table,
  or a Russell list from a vendor CSV), pass a User-Agent if scraping, wrap the
  fetch in `try/except` returning a safe fallback, and import it in `ui.py`
  alongside `get_sp_500_symbols`. Document it in `docs/data-inputs.md`.
