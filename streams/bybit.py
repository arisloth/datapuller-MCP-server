"""
Bybit v5 public linear (USDT perp) trade stream → tape store.

publicTrade carries the true taker side per print (`S`: "Buy"/"Sell").
The manager sources the PERP tape from Bybit rather than Binance futures:
Binance's fstream data plane is silently filtered on some networks (handshake
succeeds, no frames ever arrive), while Bybit's stream is reliable and the
perp aggressor-flow signal is equivalent.
"""
from store import STORE

from .base import StreamClient

LINEAR_WS = "wss://stream.bybit.com/v5/public/linear"


class BybitTradeStream(StreamClient):
    app_ping = {"op": "ping"}   # Bybit expects a JSON ping every ~20s

    def __init__(self, prefix: str = "PERP"):
        super().__init__(LINEAR_WS)
        self.prefix = prefix
        self.symbols: set[str] = set()

    async def on_open(self, ws) -> None:
        if self.symbols:
            await self._request("subscribe", sorted(self.symbols))

    async def _request(self, op: str, symbols) -> None:
        await self.send({"op": op, "args": [f"publicTrade.{s}" for s in symbols]})

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
        # {"topic":"publicTrade.BTCUSDT","type":"snapshot","ts":...,
        #  "data":[{"T":<ms>,"s":"BTCUSDT","S":"Buy"|"Sell","v":"<qty>","p":"<price>",...}]}
        # Acks ({"success":true,"op":"subscribe"}) and pongs are ignored.
        if not isinstance(msg, dict) or not str(msg.get("topic", "")).startswith("publicTrade."):
            return
        for t in msg.get("data", []):
            STORE.ingest_trade(
                f"{self.prefix}:{t['s']}",
                int(t["T"]),
                float(t["p"]),
                float(t["v"]),
                "buy" if t["S"] == "Buy" else "sell",
            )
