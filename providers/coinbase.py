"""Coinbase Exchange public API fetchers (spot) — no API key needed. Pure I/O."""
from .base import SESSION, TIMEOUT

BASE = "https://api.exchange.coinbase.com"

# Project interval -> Coinbase candle granularity (seconds). Coinbase only
# supports these six buckets and caps a request at 300 candles.
GRANULARITY = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "6h": 21600, "1d": 86400}
MAX_CANDLES = 300


def _base_asset(symbol: str) -> str:
    """'BTCUSDT'/'BTCUSD' -> 'BTC' (Coinbase products are {base}-USD)."""
    for quote in ("USDT", "USD"):
        if symbol.endswith(quote):
            return symbol[: -len(quote)]
    return symbol


def fetch_24h(base: str) -> dict:
    """24h stats for the {base}-USD product. Returns volume (base asset) and last
    price; multiply for USD volume. Raises HTTPError on 404 (not listed)."""
    r = SESSION.get(f"{BASE}/products/{base}-USD/stats", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_klines_spot(symbol: str, interval: str, limit: int):
    """{base}-USD spot candles normalized to Binance-style [openTime_ms, o, h, l, c, v]
    rows (oldest first). Capped at 300 candles; no taker-buy field — CVD/taker
    metrics degrade to n/a. Raises ValueError on unsupported interval."""
    g = GRANULARITY.get(interval)
    if g is None:
        raise ValueError(f"coinbase: unsupported interval '{interval}' (supported: {', '.join(GRANULARITY)})")
    product = f"{_base_asset(symbol)}-USD"
    r = SESSION.get(f"{BASE}/products/{product}/candles", params={"granularity": g}, timeout=TIMEOUT)
    r.raise_for_status()
    rows = r.json()  # newest first: [time_s, low, high, open, close, volume]
    if not rows:
        raise ValueError(f"coinbase: no candles for {product}")
    return [[int(k[0]) * 1000, k[3], k[2], k[1], k[4], k[5]]
            for k in rows[: min(limit, MAX_CANDLES)][::-1]]


def fetch_orderbook(symbol: str) -> dict:
    """Top-20 L2 order book for the {base}-USD product."""
    product = f"{_base_asset(symbol)}-USD"
    r = SESSION.get(f"{BASE}/products/{product}/book", params={"level": 2}, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return {
        "bids": [[b[0], b[1]] for b in data["bids"][:20]],
        "asks": [[a[0], a[1]] for a in data["asks"][:20]],
    }
