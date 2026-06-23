"""CoinGecko public API fetchers (cross-exchange aggregates) — no API key needed.
Pure I/O. Rate-limited (~5-15 req/min); the shared Session retries 429s with backoff."""
from .base import SESSION, TIMEOUT

BASE = "https://api.coingecko.com/api/v3"


def fetch_aggregate_volume(base: str, quote: str = "USD") -> float:
    """Cross-exchange 24h volume for `base`, denominated in `quote`.

    Resolves the ticker via /coins/markets (CoinGecko collapses symbol collisions
    to the primary asset; request market_cap_desc and take the top match
    defensively) and returns its `total_volume`."""
    r = SESSION.get(
        f"{BASE}/coins/markets",
        params={"vs_currency": quote.lower(), "symbols": base.lower(), "order": "market_cap_desc"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    if not data:
        raise ValueError(f"coingecko: no market data for {base}/{quote}")
    vol = data[0].get("total_volume")
    if vol is None:
        raise ValueError(f"coingecko: no 24h volume for {base}/{quote}")
    return float(vol)


def fetch_global_metrics() -> dict:
    """Global metrics: total market cap (USD), its 24h % change, and the market-cap
    dominance % per asset (btc, eth, usdt, usdc, ...). Returns the `data` object."""
    r = SESSION.get(f"{BASE}/global", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["data"]
