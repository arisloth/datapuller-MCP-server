"""Coinbase Exchange public API fetcher (spot) — no API key needed. Pure I/O."""
from .base import SESSION, TIMEOUT

BASE = "https://api.exchange.coinbase.com"


def fetch_24h(base: str) -> dict:
    """24h stats for the {base}-USD product. Returns volume (base asset) and last
    price; multiply for USD volume. Raises HTTPError on 404 (not listed)."""
    r = SESSION.get(f"{BASE}/products/{base}-USD/stats", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()
