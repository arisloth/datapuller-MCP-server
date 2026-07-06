"""Hyperliquid public info API fetchers (perps + spot) — no API key needed. Pure I/O."""
import time

from .base import SESSION, TIMEOUT

API = "https://api.hyperliquid.xyz/info"

# Kline intervals Hyperliquid's candleSnapshot accepts, in minutes.
INTERVAL_MIN = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "8h": 480, "12h": 720,
    "1d": 1440, "3d": 4320, "1w": 10080, "1M": 43200,
}


def _coin(symbol: str) -> str:
    return symbol[:-4] if symbol.endswith("USDT") else symbol


def fetch_orderbook(symbol: str, limit: int = 20) -> dict:
    r = SESSION.post(API, json={"type": "l2Book", "coin": _coin(symbol)}, timeout=TIMEOUT)
    r.raise_for_status()
    bids_raw, asks_raw = r.json()["levels"]
    return {
        "bids": [[lvl["px"], lvl["sz"]] for lvl in bids_raw[:limit]],
        "asks": [[lvl["px"], lvl["sz"]] for lvl in asks_raw[:limit]],
    }


def fetch_ctx(symbol: str) -> dict:
    """Per-asset context: funding (HOURLY), mark/oracle price, OI, 24h notional vol.
    Raises if the coin isn't listed."""
    coin = _coin(symbol)
    r = SESSION.post(API, json={"type": "metaAndAssetCtxs"}, timeout=TIMEOUT)
    r.raise_for_status()
    meta, ctxs = r.json()
    for i, asset in enumerate(meta["universe"]):
        if asset["name"] == coin:
            return ctxs[i]
    raise ValueError(f"hyperliquid: no perp for {coin}")


def _spot_pair_name(meta: dict, coin: str) -> str:
    """Canonical id of the coin's USDC spot pair — the universe `name`
    (e.g. '@107' or 'PURR/USDC'). Spot pairs are USDC-quoted."""
    token = next((t["index"] for t in meta["tokens"] if t["name"] == coin), None)
    if token is None:
        raise ValueError(f"hyperliquid: no spot token {coin}")
    usdc = next((t["index"] for t in meta["tokens"] if t["name"] == "USDC"), 0)
    for pair in meta["universe"]:
        if pair["tokens"] == [token, usdc]:
            return pair["name"]
    raise ValueError(f"hyperliquid: no {coin}/USDC spot pair")


def fetch_spot_ctx(symbol: str) -> dict:
    """Spot-pair context: dayNtlVlm (24h USDC notional), markPx, midPx.
    Ctxs are matched by their `coin` field — they do NOT align positionally
    with the universe list. Raises if the coin has no USDC spot pair."""
    r = SESSION.post(API, json={"type": "spotMetaAndAssetCtxs"}, timeout=TIMEOUT)
    r.raise_for_status()
    meta, ctxs = r.json()
    pair = _spot_pair_name(meta, _coin(symbol))
    ctx = next((c for c in ctxs if c.get("coin") == pair), None)
    if ctx is None:
        raise ValueError(f"hyperliquid: no ctx for spot pair {pair}")
    return ctx


def fetch_klines_spot(symbol: str, interval: str, limit: int):
    """USDC spot-pair candles normalized to Binance-style [openTime_ms, o, h, l, c, v]
    rows (oldest first). No taker-buy field — CVD/taker metrics degrade to n/a."""
    minutes = INTERVAL_MIN.get(interval)
    if minutes is None:
        raise ValueError(f"hyperliquid: unsupported interval '{interval}' (supported: {', '.join(INTERVAL_MIN)})")
    r = SESSION.post(API, json={"type": "spotMeta"}, timeout=TIMEOUT)
    r.raise_for_status()
    pair = _spot_pair_name(r.json(), _coin(symbol))

    end_ms = int(time.time() * 1000)
    req = {"coin": pair, "interval": interval, "startTime": end_ms - minutes * 60_000 * limit, "endTime": end_ms}
    r = SESSION.post(API, json={"type": "candleSnapshot", "req": req}, timeout=TIMEOUT)
    r.raise_for_status()
    rows = r.json()  # oldest first: {t: openTime_ms, o, h, l, c, v, ...}
    if not rows:
        raise ValueError(f"hyperliquid: no spot candles for {pair}")
    return [[int(c["t"]), c["o"], c["h"], c["l"], c["c"], c["v"]] for c in rows[-limit:]]
