"""
Stream lifecycle manager: owns the background event loop and venue clients.

Everything is lazy — nothing connects until the first ensure_subscribed_*()
call from a tool. The event loop runs on a daemon thread so the (sync) FastMCP
tools never block on stream work; subscriptions are handed across threads with
run_coroutine_threadsafe and are fire-and-forget (the tool returns immediately
and reports "warming" until trades accumulate in the store).

Venue choice: PERP tape from Bybit — Binance's futures WS data plane is
silently filtered on some networks (see streams/bybit.py). SPOT tape is routed
per symbol: Binance, falling back to Bybit spot, then Hyperliquid spot for
coins Binance doesn't list (probed once over REST, cached). All spot tapes are
keyed "SPOT:<SYMBOL>" — exactly one venue streams a given symbol, so the
ladder never double-counts.
"""
import asyncio
import threading

from providers import alpaca as alpaca_rest
from providers import binance as binance_rest
from providers import bybit as bybit_rest
from providers import hyperliquid as hyperliquid_rest
from store import SubscriptionBudget

from . import alpaca, binance, bybit, hyperliquid

PERP_VENUE = "bybit"
SPOT_VENUE = "binance"   # default spot venue; per-symbol routing via spot_venue()

_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_clients: dict[str, object] = {}
_budget_crypto = SubscriptionBudget()
_budget_equity = SubscriptionBudget()  # 30 = Alpaca IEX symbol limit
_spot_route: dict[str, tuple[str, str]] = {}  # symbol -> (venue, venue coin id)


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


def _resolve_spot_route(symbol: str) -> tuple[str, str]:
    """Which venue serves `symbol`'s spot tape, probed once over REST and cached:
    Binance, else Bybit spot, else Hyperliquid spot (whose pair id doubles as the
    WS subscription coin). Coins listed nowhere default to Binance — the tape
    just stays empty, same behavior as before routing existed."""
    if symbol in _spot_route:
        return _spot_route[symbol]

    def probe_binance():
        binance_rest.fetch_24h(symbol)
        return "binance", symbol

    def probe_bybit():
        bybit_rest.fetch_24h_spot(symbol)
        return "bybit", symbol

    def probe_hyperliquid():
        return "hyperliquid", hyperliquid_rest.spot_pair_name(symbol)

    for probe in (probe_binance, probe_bybit, probe_hyperliquid):
        try:
            _spot_route[symbol] = probe()
            break
        except Exception:
            continue
    else:
        _spot_route[symbol] = ("binance", symbol)
    return _spot_route[symbol]


def spot_venue(symbol: str) -> str:
    """Venue serving the symbol's spot tape ('binance'/'bybit'/'hyperliquid').
    Resolved by ensure_subscribed_crypto; defaults to 'binance' before that."""
    return _spot_route.get(symbol.upper(), (SPOT_VENUE, ""))[0]


def _spot_client(venue: str):
    """Get-or-create the spot stream client for a venue (call under _lock).
    Returns (client, started) where started means run() still needs submitting."""
    key = f"spot:{venue}"
    if key in _clients:
        return _clients[key], False
    if venue == "bybit":
        _clients[key] = bybit.BybitTradeStream("SPOT", url=bybit.SPOT_WS)
    elif venue == "hyperliquid":
        _clients[key] = hyperliquid.HyperliquidTradeStream("SPOT")
    else:
        _clients[key] = binance.BinanceTradeStream(binance.SPOT_WS, "SPOT")
    return _clients[key], True


def ensure_subscribed_crypto(symbol: str) -> None:
    """Idempotent, non-blocking: subscribe the perp (Bybit) and spot (routed
    venue) trade streams for `symbol`, starting the loop and clients on first
    use. LRU-evicts the oldest symbol past the budget."""
    symbol = symbol.upper()
    venue, coin = _resolve_spot_route(symbol)
    with _lock:
        started = []
        if "perp" not in _clients:
            _clients["perp"] = bybit.BybitTradeStream("PERP")
            started.append(_clients["perp"])
        spot, spot_started = _spot_client(venue)
        if spot_started:
            started.append(spot)
    for client in started:
        _submit(client.run())

    evicted = _budget_crypto.touch(symbol)
    _submit(_clients["perp"].subscribe(symbol))
    if venue == "hyperliquid":
        _submit(spot.subscribe(symbol, coin))
    else:
        _submit(spot.subscribe(symbol))
    for old in evicted:
        # the evicted symbol may live on a different spot venue — sweep them all
        for key, client in list(_clients.items()):
            if key == "perp" or key.startswith("spot:"):
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
