"""Bybit v5 public API fetchers (linear perps + spot) — no API key needed. Pure I/O."""
from .base import SESSION, TIMEOUT

ORDERBOOK = "https://api.bybit.com/v5/market/orderbook"
TICKERS   = "https://api.bybit.com/v5/market/tickers"
KLINE     = "https://api.bybit.com/v5/market/kline"

# Project interval -> Bybit kline interval. No 8h/3d on Bybit.
SPOT_INTERVALS = {
    "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
    "1h": "60", "2h": "120", "4h": "240", "6h": "360", "12h": "720",
    "1d": "D", "1w": "W", "1M": "M",
}


def _ok(r) -> dict:
    """raise_for_status + Bybit's own retCode envelope; returns `result`."""
    r.raise_for_status()
    data = r.json()
    if data["retCode"] != 0:
        raise ValueError(data["retMsg"])
    return data["result"]


def fetch_klines_spot(symbol: str, interval: str, limit: int):
    """Spot klines normalized to Binance-style [openTime_ms, o, h, l, c, v] rows
    (oldest first). No taker-buy field — CVD/taker metrics degrade to n/a."""
    iv = SPOT_INTERVALS.get(interval)
    if iv is None:
        raise ValueError(f"bybit: unsupported interval '{interval}' (supported: {', '.join(SPOT_INTERVALS)})")
    r = SESSION.get(
        KLINE,
        params={"category": "spot", "symbol": symbol, "interval": iv, "limit": min(limit, 1000)},
        timeout=TIMEOUT,
    )
    rows = _ok(r)["list"]  # newest first: [startTime_ms, o, h, l, c, volume, turnover]
    if not rows:
        raise ValueError(f"bybit: no spot klines for {symbol}")
    return [[int(k[0]), k[1], k[2], k[3], k[4], k[5]] for k in reversed(rows)]


def fetch_orderbook(symbol: str) -> dict:
    r = SESSION.get(ORDERBOOK, params={"category": "linear", "symbol": symbol, "limit": 20}, timeout=TIMEOUT)
    res = _ok(r)
    return {
        "bids": [[b[0], b[1]] for b in res["b"]],
        "asks": [[a[0], a[1]] for a in res["a"]],
    }


def _ticker(category: str, symbol: str) -> dict:
    r = SESSION.get(TICKERS, params={"category": category, "symbol": symbol}, timeout=TIMEOUT)
    lst = _ok(r)["list"]
    if not lst:
        raise ValueError(f"bybit: no {category} ticker for {symbol}")
    return lst[0]


def fetch_ticker(symbol: str) -> dict:
    """Linear ticker: funding rate, next funding, mark/index price, 24h turnover.
    Bybit funds every 8h for majors (some alts differ)."""
    return _ticker("linear", symbol)


def fetch_24h_spot(symbol: str) -> dict:
    """Spot ticker: turnover24h (quote volume, USDT) for cross-venue spot comparison."""
    return _ticker("spot", symbol)
