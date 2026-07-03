"""
Binance aggTrade streams (spot + USDⓈ-M perp) → tape store.

aggTrade carries the true aggressor side: `m` is "buyer is the market maker",
so m=true → the taker SOLD, m=false → the taker BOUGHT. No classification
needed (contrast with equities, where the side must be inferred via Lee–Ready).

Tapes are keyed "PERP:<SYMBOL>" / "SPOT:<SYMBOL>" so the spot-vs-perp split
survives into the store.
"""
from store import STORE

from .base import StreamClient

SPOT_WS = "wss://stream.binance.com:9443/ws"
PERP_WS = "wss://fstream.binance.com/ws"


class BinanceTradeStream(StreamClient):
    """One connection per market (spot or perp), dynamic SUBSCRIBE/UNSUBSCRIBE."""

    def __init__(self, url: str, prefix: str):
        super().__init__(url)
        self.prefix = prefix        # "SPOT" | "PERP"
        self.symbols: set[str] = set()
        self._req_id = 0

    async def on_open(self, ws) -> None:
        if self.symbols:
            await self._request("SUBSCRIBE", sorted(self.symbols))

    async def _request(self, method: str, symbols) -> None:
        self._req_id += 1
        await self.send({
            "method": method,
            "params": [f"{s.lower()}@aggTrade" for s in symbols],
            "id": self._req_id,
        })

    async def subscribe(self, symbol: str) -> None:
        symbol = symbol.upper()
        if symbol in self.symbols:
            return
        self.symbols.add(symbol)
        if self.connected:
            await self._request("SUBSCRIBE", [symbol])

    async def unsubscribe(self, symbol: str) -> None:
        symbol = symbol.upper()
        if symbol not in self.symbols:
            return
        self.symbols.discard(symbol)
        if self.connected:
            await self._request("UNSUBSCRIBE", [symbol])

    def handle(self, msg) -> None:
        # {"e":"aggTrade","s":"BTCUSDT","p":"<price>","q":"<qty>","T":<ms>,"m":<bool>}
        # Non-trade frames (subscribe acks {"result":null,"id":n}) are ignored.
        if not isinstance(msg, dict) or msg.get("e") != "aggTrade":
            return
        STORE.ingest_trade(
            f"{self.prefix}:{msg['s']}",
            int(msg["T"]),
            float(msg["p"]),
            float(msg["q"]),
            "sell" if msg["m"] else "buy",
        )
