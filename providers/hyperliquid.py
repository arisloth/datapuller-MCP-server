"""Hyperliquid public info API fetchers (perps) — no API key needed. Pure I/O."""
from .base import SESSION, TIMEOUT

API = "https://api.hyperliquid.xyz/info"


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
