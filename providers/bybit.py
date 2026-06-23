"""Bybit v5 public API fetchers (linear perps) — no API key needed. Pure I/O."""
from .base import SESSION, TIMEOUT

ORDERBOOK = "https://api.bybit.com/v5/market/orderbook"
TICKERS   = "https://api.bybit.com/v5/market/tickers"


def fetch_orderbook(symbol: str) -> dict:
    r = SESSION.get(ORDERBOOK, params={"category": "linear", "symbol": symbol, "limit": 20}, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if data["retCode"] != 0:
        raise ValueError(data["retMsg"])
    res = data["result"]
    return {
        "bids": [[b[0], b[1]] for b in res["b"]],
        "asks": [[a[0], a[1]] for a in res["a"]],
    }


def fetch_ticker(symbol: str) -> dict:
    """Linear ticker: funding rate, next funding, mark/index price, 24h turnover.
    Bybit funds every 8h for majors (some alts differ)."""
    r = SESSION.get(TICKERS, params={"category": "linear", "symbol": symbol}, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if data["retCode"] != 0:
        raise ValueError(data["retMsg"])
    lst = data["result"]["list"]
    if not lst:
        raise ValueError(f"bybit: no ticker for {symbol}")
    return lst[0]
