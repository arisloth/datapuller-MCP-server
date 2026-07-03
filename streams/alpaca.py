"""
Alpaca equity trade + quote (NBBO) stream → tape store.

Equity prints carry no taker flag, so each trade is classified via Lee–Ready
against the prevailing NBBO already on the tape (store.ingest_equity_trade);
that's why quotes are always subscribed alongside trades. Non-regular-way
prints are filtered by condition code inside the store.

Feed comes from ALPACA_FEED ('iex' free tier — partial tape, 1 connection,
~30 symbols — or 'sip' full tape); credentials from APCA_API_KEY_ID /
APCA_API_SECRET_KEY, both reused from providers/alpaca.py. Tapes are keyed
"EQ:<SYMBOL>".
"""
import re
from datetime import datetime

from providers import alpaca as alpaca_rest
from store import STORE

from .base import StreamClient

WS_BASE = "wss://stream.data.alpaca.markets/v2"


def _ts_ms(iso: str) -> int:
    """RFC-3339 with up to nanosecond precision → epoch ms (fromisoformat
    only accepts microseconds, so trim the fractional part to 6 digits)."""
    m = re.match(r"(.+?)\.(\d+)(Z|[+-].+)$", iso)
    if m:
        head, frac, tz = m.groups()
        iso = f"{head}.{frac[:6]}{tz}"
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * 1000)


class AlpacaStream(StreamClient):
    """One connection for all equity symbols (Alpaca allows exactly one)."""

    def __init__(self, feed: str | None = None):
        self.feed = feed or alpaca_rest.feed()
        super().__init__(f"{WS_BASE}/{self.feed}")
        self.symbols: set[str] = set()

    async def on_open(self, ws) -> None:
        key, secret = alpaca_rest._creds()
        await self.send({"action": "auth", "key": key, "secret": secret})
        if self.symbols:
            await self._request("subscribe", sorted(self.symbols))

    async def _request(self, action: str, symbols) -> None:
        await self.send({"action": action, "trades": list(symbols), "quotes": list(symbols)})

    async def subscribe(self, symbol: str) -> None:
        symbol = symbol.upper()
        if symbol in self.symbols:
            return
        self.symbols.add(symbol)
        if self.connected:
            await self._request("subscribe", [symbol])

    async def unsubscribe(self, symbol: str) -> None:
        symbol = symbol.upper()
        if symbol not in self.symbols:
            return
        self.symbols.discard(symbol)
        if self.connected:
            await self._request("unsubscribe", [symbol])

    def handle(self, msg) -> None:
        # Alpaca delivers a JSON array of messages; "T" is the message type:
        #   {"T":"q","S":"AAPL","bp":..,"bs":..,"ap":..,"as":..,"t":"..."}
        #   {"T":"t","S":"AAPL","p":..,"s":..,"c":["@"],"t":"..."}
        #   {"T":"error","code":406,...} / {"T":"success",...} / {"T":"subscription",...}
        # Quotes must be applied before trades with equal timestamps, which
        # matches Alpaca's in-array ordering.
        if not isinstance(msg, list):
            return
        for m in msg:
            kind = m.get("T")
            if kind == "q":
                STORE.ingest_quote(
                    f"EQ:{m['S']}", _ts_ms(m["t"]),
                    bid=float(m["bp"]), ask=float(m["ap"]),
                    bid_size=float(m["bs"]), ask_size=float(m["as"]),
                )
            elif kind == "t":
                STORE.ingest_equity_trade(
                    f"EQ:{m['S']}", _ts_ms(m["t"]),
                    float(m["p"]), float(m["s"]),
                    conditions=m.get("c"),
                )
            # "error" frames (406 connection limit, 405 symbol limit, 402 auth)
            # are terminal for this attempt; the base client's reconnect loop
            # handles recovery, and missing creds are caught before connect.
