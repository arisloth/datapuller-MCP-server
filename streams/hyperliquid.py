"""
Hyperliquid trades stream (USDC spot pairs) → tape store.

The `trades` channel carries the true taker side per print (`side`: "B" buy /
"A" sell). Pairs are subscribed by their canonical id ('@107', 'PURR/USDC'),
which the manager resolves over REST (providers.hyperliquid.spot_pair_name)
before subscribing — the stream layer itself does no HTTP. Tapes are keyed
"SPOT:<SYMBOL>" so the spot ladder reads identically regardless of venue.
"""
from store import STORE

from .base import StreamClient

WS = "wss://api.hyperliquid.xyz/ws"


class HyperliquidTradeStream(StreamClient):
    app_ping = {"method": "ping"}   # HL closes connections idle for ~60s

    def __init__(self, prefix: str = "SPOT"):
        super().__init__(WS)
        self.prefix = prefix
        self.coins: dict[str, str] = {}     # pair id ('@107') -> symbol ('HYPEUSDT')
        self.symbols: dict[str, str] = {}   # symbol -> pair id

    async def on_open(self, ws) -> None:
        for coin in sorted(self.coins):
            await self._request("subscribe", coin)

    async def _request(self, method: str, coin: str) -> None:
        await self.send({"method": method, "subscription": {"type": "trades", "coin": coin}})

    async def subscribe(self, symbol: str, coin: str) -> None:
        symbol = symbol.upper()
        if symbol in self.symbols:
            return
        self.symbols[symbol] = coin
        self.coins[coin] = symbol
        if self.connected:
            await self._request("subscribe", coin)

    async def unsubscribe(self, symbol: str) -> None:
        symbol = symbol.upper()
        coin = self.symbols.pop(symbol, None)
        if coin is None:
            return
        self.coins.pop(coin, None)
        if self.connected:
            await self._request("unsubscribe", coin)

    def handle(self, msg) -> None:
        # {"channel":"trades","data":[{"coin":"@107","side":"B"|"A","px":"71.5",
        #  "sz":"12.4","time":<ms>,...}]}
        # Subscription acks ({"channel":"subscriptionResponse",...}) and pongs
        # ({"channel":"pong"}) are ignored.
        if not isinstance(msg, dict) or msg.get("channel") != "trades":
            return
        for t in msg.get("data", []):
            symbol = self.coins.get(t["coin"])
            if symbol is None:
                continue
            STORE.ingest_trade(
                f"{self.prefix}:{symbol}",
                int(t["time"]),
                float(t["px"]),
                float(t["sz"]),
                "buy" if t["side"] == "B" else "sell",
            )
