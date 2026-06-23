"""
Alpaca market-data fetcher for stocks & ETFs (incl. commodity ETFs). Pure I/O.

Unlike the crypto providers this one needs credentials — read from the environment
(never hard-coded, never committed):
    APCA_API_KEY_ID, APCA_API_SECRET_KEY
Optional ALPACA_FEED selects the data feed: 'iex' (free, partial volume) or 'sip'
(paid, full consolidated tape). Default 'iex'.

Bars are normalized to the Binance-style row [openTime_ms, o, h, l, c, v] so every
downstream indicator works unchanged. Note: equity bars carry no taker-buy volume,
so CVD/taker-ratio degrade to N/A automatically (indicators._has_taker).
"""
import os
from datetime import datetime, timezone

from .base import SESSION, TIMEOUT

DATA = "https://data.alpaca.markets"

# Project interval -> Alpaca timeframe. Unsupported intervals (e.g. 3d) raise.
INTERVALS = {
    "1m": "1Min", "3m": "3Min", "5m": "5Min", "15m": "15Min", "30m": "30Min",
    "1h": "1Hour", "2h": "2Hour", "4h": "4Hour", "6h": "6Hour", "8h": "8Hour", "12h": "12Hour",
    "1d": "1Day", "1w": "1Week", "1M": "1Month",
}


def feed() -> str:
    """Configured data feed ('iex' default, or 'sip' if subscribed)."""
    return os.getenv("ALPACA_FEED", "iex").lower()


def _creds() -> tuple[str, str]:
    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise ValueError(
            "alpaca: set APCA_API_KEY_ID and APCA_API_SECRET_KEY env vars to fetch equity data"
        )
    return key, secret


def _to_ms(ts: str) -> int:
    return int(datetime.fromisoformat(ts.replace("Z", "+00:00"))
               .astimezone(timezone.utc).timestamp() * 1000)


def fetch_ohlcv(symbol: str, interval: str, limit: int):
    """Stock/ETF bars normalized to [openTime_ms, o, h, l, c, v] (oldest first).
    Splits/dividends adjusted. Raises ValueError on unsupported interval or missing
    creds; returns [] when the symbol has no bars."""
    tf = INTERVALS.get(interval)
    if tf is None:
        raise ValueError(f"alpaca: unsupported interval '{interval}' (supported: {', '.join(INTERVALS)})")
    key, secret = _creds()
    r = SESSION.get(
        f"{DATA}/v2/stocks/bars",
        params={"symbols": symbol, "timeframe": tf, "limit": limit,
                "adjustment": "all", "feed": feed(), "sort": "asc"},
        headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    bars = r.json().get("bars", {}).get(symbol, []) or []
    return [[_to_ms(b["t"]), b["o"], b["h"], b["l"], b["c"], b["v"]] for b in bars]
