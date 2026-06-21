"""
Raw HTTP fetchers for Binance / Bybit / Hyperliquid public APIs (no key needed).
Every function returns parsed JSON (dicts/lists) and does no formatting.
"""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE             = "https://api.binance.com/api/v3/klines"
DEPTH_BASE       = "https://api.binance.com/api/v3/depth"
TICKER_24H_BASE  = "https://api.binance.com/api/v3/ticker/24hr"
FAPI_BASE        = "https://fapi.binance.com"
BYBIT_DEPTH_BASE = "https://api.bybit.com/v5/market/orderbook"
HYPERLIQUID_API  = "https://api.hyperliquid.xyz/info"
COINBASE_BASE    = "https://api.exchange.coinbase.com"
COINGECKO_BASE   = "https://api.coingecko.com/api/v3"

TIMEOUT = 10

# Shared session: reuses connections and retries transient errors (429 / 5xx)
# with exponential backoff. status=400 is NOT retried — it's our spot→futures
# fallback signal in fetch_klines / fetch_orderbook_binance.
SESSION = requests.Session()
_retry = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset({"GET", "POST"}),
    raise_on_status=False,
)
_adapter = HTTPAdapter(max_retries=_retry)
SESSION.mount("https://", _adapter)
SESSION.mount("http://", _adapter)


def fetch_klines(symbol: str, interval: str, limit: int):
    r = SESSION.get(BASE, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=TIMEOUT)
    if r.status_code == 400:
        r = SESSION.get(f"{FAPI_BASE}/fapi/v1/klines", params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_orderbook_binance(symbol: str) -> dict:
    r = SESSION.get(DEPTH_BASE, params={"symbol": symbol, "limit": 20}, timeout=TIMEOUT)
    if r.status_code == 400:
        r = SESSION.get(f"{FAPI_BASE}/fapi/v1/depth", params={"symbol": symbol, "limit": 20}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_orderbook_bybit(symbol: str) -> dict:
    r = SESSION.get(
        BYBIT_DEPTH_BASE,
        params={"category": "linear", "symbol": symbol, "limit": 20},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    if data["retCode"] != 0:
        raise ValueError(data["retMsg"])
    res = data["result"]
    return {
        "bids": [[b[0], b[1]] for b in res["b"]],
        "asks": [[a[0], a[1]] for a in res["a"]],
    }


def fetch_orderbook_hyperliquid(symbol: str, limit: int = 20) -> dict:
    coin = symbol[:-4] if symbol.endswith("USDT") else symbol
    r = SESSION.post(
        HYPERLIQUID_API,
        json={"type": "l2Book", "coin": coin},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    bids_raw, asks_raw = data["levels"]
    return {
        "bids": [[lvl["px"], lvl["sz"]] for lvl in bids_raw[:limit]],
        "asks": [[lvl["px"], lvl["sz"]] for lvl in asks_raw[:limit]],
    }


def fetch_premium_index(symbol: str) -> dict:
    r = SESSION.get(f"{FAPI_BASE}/fapi/v1/premiumIndex", params={"symbol": symbol}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_open_interest(symbol: str) -> float:
    r = SESSION.get(f"{FAPI_BASE}/fapi/v1/openInterest", params={"symbol": symbol}, timeout=TIMEOUT)
    r.raise_for_status()
    return float(r.json()["openInterest"])


def fetch_open_interest_hist(symbol: str, period: str = "1h", limit: int = 6) -> list:
    r = SESSION.get(
        f"{FAPI_BASE}/futures/data/openInterestHist",
        params={"symbol": symbol, "period": period, "limit": limit},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def fetch_24h_binance(symbol: str) -> dict:
    """24h ticker stats: price change, volume (base), quoteVolume (USDT)."""
    r = SESSION.get(TICKER_24H_BASE, params={"symbol": symbol}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_24h_coinbase(base: str) -> dict:
    """24h stats for the {base}-USD Coinbase product. Returns volume (base
    asset) and last price; multiply for USD volume. Raises HTTPError on 404."""
    r = SESSION.get(f"{COINBASE_BASE}/products/{base}-USD/stats", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_aggregate_volume(base: str, quote: str = "USD") -> float:
    """CoinGecko cross-exchange 24h volume for `base`, denominated in `quote`.

    Resolves the ticker via /coins/markets (CoinGecko collapses symbol
    collisions to the primary asset; we request market_cap_desc and take the
    top match defensively) and returns its `total_volume`. Keyless public API —
    rate-limited (~5-15 req/min); the shared Session retries 429s with backoff.
    """
    r = SESSION.get(
        f"{COINGECKO_BASE}/coins/markets",
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


def fetch_long_short_ratio(symbol: str, period: str = "1h", limit: int = 1) -> list:
    r = SESSION.get(
        f"{FAPI_BASE}/futures/data/globalLongShortAccountRatio",
        params={"symbol": symbol, "period": period, "limit": limit},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def compute_futures_context(symbol: str) -> dict:
    """Fetch Binance USD-M perpetual futures context for `symbol`, once.

    Performs all the futures round-trips (premium index, open interest + 5h
    history, global long/short ratio) and returns a single normalized dict.
    Each field is None when its endpoint is unavailable (e.g. no perp for the
    symbol), so callers never re-fetch and formatters do no I/O.
    """
    ctx = {
        "symbol": symbol,
        "funding_rate_pct": None,
        "next_funding": None,
        "mark_price": None,
        "index_price": None,
        "open_interest": None,
        "oi_change_pct_5h": None,
        "price_change_pct_5h": None,
        "long_short_ratio": None,
        "long_pct": None,
        "short_pct": None,
    }

    try:
        pm = fetch_premium_index(symbol)
        ctx["funding_rate_pct"] = float(pm["lastFundingRate"]) * 100
        ctx["next_funding"] = int(pm["nextFundingTime"])
        ctx["mark_price"] = float(pm["markPrice"])
        ctx["index_price"] = float(pm["indexPrice"])
    except Exception:
        pass

    try:
        ctx["open_interest"] = fetch_open_interest(symbol)
        hist = fetch_open_interest_hist(symbol, period="1h", limit=6)
        if len(hist) >= 2:
            oi_old = float(hist[0]["sumOpenInterest"])
            oi_new = float(hist[-1]["sumOpenInterest"])
            ctx["oi_change_pct_5h"] = (oi_new - oi_old) / oi_old * 100 if oi_old else 0.0
            # Implied mark price per snapshot = notional value / contracts. Lets us
            # measure price change over the *same* window as OI (for the quadrant)
            # without a second request. Guard against missing/zero fields.
            val_old = float(hist[0].get("sumOpenInterestValue", 0) or 0)
            val_new = float(hist[-1].get("sumOpenInterestValue", 0) or 0)
            if oi_old and oi_new and val_old and val_new:
                px_old = val_old / oi_old
                px_new = val_new / oi_new
                ctx["price_change_pct_5h"] = (px_new - px_old) / px_old * 100 if px_old else 0.0
    except Exception:
        pass

    try:
        data = fetch_long_short_ratio(symbol, period="1h", limit=1)
        if data:
            ls = data[0]
            ctx["long_short_ratio"] = float(ls["longShortRatio"])
            ctx["long_pct"] = float(ls["longAccount"]) * 100
            ctx["short_pct"] = float(ls["shortAccount"]) * 100
    except Exception:
        pass

    return ctx
