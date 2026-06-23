"""
Binance public API fetchers (spot + USDⓈ-M futures) — no API key needed.
Pure I/O: every function returns parsed JSON and does no formatting or math.
"""
from .base import SESSION, TIMEOUT

SPOT_KLINES = "https://api.binance.com/api/v3/klines"
SPOT_DEPTH  = "https://api.binance.com/api/v3/depth"
SPOT_24H    = "https://api.binance.com/api/v3/ticker/24hr"
FAPI        = "https://fapi.binance.com"


def fetch_klines(symbol: str, interval: str, limit: int):
    """Spot klines, falling back to USDⓈ-M futures on 400 (symbol not on spot)."""
    r = SESSION.get(SPOT_KLINES, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=TIMEOUT)
    if r.status_code == 400:
        r = SESSION.get(f"{FAPI}/fapi/v1/klines", params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_klines_spot(symbol: str, interval: str, limit: int):
    """Spot klines only (no futures fallback) — for spot-vs-perp comparison."""
    r = SESSION.get(SPOT_KLINES, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_klines_futures(symbol: str, interval: str, limit: int):
    """USDⓈ-M perpetual klines only — for spot-vs-perp comparison."""
    r = SESSION.get(f"{FAPI}/fapi/v1/klines", params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_orderbook(symbol: str) -> dict:
    """Top-20 order book, spot with futures fallback on 400."""
    r = SESSION.get(SPOT_DEPTH, params={"symbol": symbol, "limit": 20}, timeout=TIMEOUT)
    if r.status_code == 400:
        r = SESSION.get(f"{FAPI}/fapi/v1/depth", params={"symbol": symbol, "limit": 20}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_premium_index(symbol: str) -> dict:
    r = SESSION.get(f"{FAPI}/fapi/v1/premiumIndex", params={"symbol": symbol}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_funding_history(symbol: str, limit: int = 100) -> list:
    """Settled funding-rate history (newest last) — for extremes/percentile.
    Each row: {symbol, fundingTime (ms), fundingRate, markPrice}."""
    r = SESSION.get(f"{FAPI}/fapi/v1/fundingRate", params={"symbol": symbol, "limit": limit}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_24h(symbol: str) -> dict:
    """Spot 24h ticker stats: price change, volume (base), quoteVolume (USDT)."""
    r = SESSION.get(SPOT_24H, params={"symbol": symbol}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_24h_futures(symbol: str) -> dict:
    """USDⓈ-M perp 24h ticker: quoteVolume (USDT) for spot-vs-perp comparison."""
    r = SESSION.get(f"{FAPI}/fapi/v1/ticker/24hr", params={"symbol": symbol}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_open_interest(symbol: str) -> float:
    r = SESSION.get(f"{FAPI}/fapi/v1/openInterest", params={"symbol": symbol}, timeout=TIMEOUT)
    r.raise_for_status()
    return float(r.json()["openInterest"])


def fetch_open_interest_hist(symbol: str, period: str = "1h", limit: int = 6) -> list:
    r = SESSION.get(
        f"{FAPI}/futures/data/openInterestHist",
        params={"symbol": symbol, "period": period, "limit": limit},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def fetch_long_short_ratio(symbol: str, period: str = "1h", limit: int = 1) -> list:
    r = SESSION.get(
        f"{FAPI}/futures/data/globalLongShortAccountRatio",
        params={"symbol": symbol, "period": period, "limit": limit},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()
