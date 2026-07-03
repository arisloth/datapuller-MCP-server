"""
Stream lifecycle manager: owns the background event loop and venue clients.

Everything is lazy — nothing connects until the first ensure_subscribed_*()
call from a tool. The event loop runs on a daemon thread so the (sync) FastMCP
tools never block on stream work; subscriptions are handed across threads with
run_coroutine_threadsafe and are fire-and-forget (the tool returns immediately
and reports "warming" until trades accumulate in the store).

Venue choice: SPOT tape from Binance, PERP tape from Bybit — Binance's futures
WS data plane is silently filtered on some networks (see streams/bybit.py).
"""
import asyncio
import threading

from providers import alpaca as alpaca_rest
from store import SubscriptionBudget

from . import alpaca, binance, bybit

PERP_VENUE = "bybit"
SPOT_VENUE = "binance"

_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_clients: dict[str, object] = {}
_budget_crypto = SubscriptionBudget()
_budget_equity = SubscriptionBudget()  # 30 = Alpaca IEX symbol limit


def _ensure_loop() -> asyncio.AbstractEventLoop:
    global _loop
    with _lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()
            threading.Thread(target=_loop.run_forever, daemon=True,
                             name="stream-loop").start()
        return _loop


def _submit(coro) -> None:
    asyncio.run_coroutine_threadsafe(coro, _ensure_loop())


def ensure_subscribed_crypto(symbol: str) -> None:
    """Idempotent, non-blocking: subscribe the perp (Bybit) and spot (Binance)
    trade streams for `symbol`, starting the loop and clients on first use.
    LRU-evicts the oldest symbol past the budget."""
    symbol = symbol.upper()
    with _lock:
        if "perp" not in _clients:
            _clients["perp"] = bybit.BybitTradeStream("PERP")
            _clients["spot"] = binance.BinanceTradeStream(binance.SPOT_WS, "SPOT")
            started = list(_clients.values())
        else:
            started = []
    for client in started:
        _submit(client.run())

    evicted = _budget_crypto.touch(symbol)
    for client in (_clients["perp"], _clients["spot"]):
        _submit(client.subscribe(symbol))
        for old in evicted:
            _submit(client.unsubscribe(old))


def ensure_subscribed_equity(symbol: str) -> None:
    """Idempotent, non-blocking: subscribe trades+quotes for an equity symbol
    on the single Alpaca connection. Raises ValueError when Alpaca credentials
    are missing (checked up front so the tool can report it, instead of the
    stream thread reconnect-looping on a doomed auth)."""
    alpaca_rest._creds()
    symbol = symbol.upper()
    with _lock:
        if "equity" not in _clients:
            _clients["equity"] = alpaca.AlpacaStream()
            started = [_clients["equity"]]
        else:
            started = []
    for client in started:
        _submit(client.run())

    evicted = _budget_equity.touch(symbol)
    _submit(_clients["equity"].subscribe(symbol))
    for old in evicted:
        _submit(_clients["equity"].unsubscribe(old))
