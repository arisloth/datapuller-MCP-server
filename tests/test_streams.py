"""
Tests for the WebSocket ingestion layer: Binance frame parsing into the tape
store, subscription bookkeeping, manager lazy start, and the get_cvd live
wiring. No network — frames are canned dicts, the manager gets fake clients.
Run with `pytest`.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from store import TapeStore  # noqa: E402
from streams import binance as bstream  # noqa: E402
from streams import bybit as bybit_stream  # noqa: E402
from streams import hyperliquid as hl_stream  # noqa: E402
from streams import manager  # noqa: E402


@pytest.fixture
def fresh_store(monkeypatch):
    s = TapeStore()
    monkeypatch.setattr(bstream, "STORE", s)
    monkeypatch.setattr(bybit_stream, "STORE", s)
    monkeypatch.setattr(hl_stream, "STORE", s)
    return s


# --- BinanceTradeStream.handle → store ---------------------------------------

def test_aggtrade_frame_lands_in_store(fresh_store):
    c = bstream.BinanceTradeStream(bstream.PERP_WS, "PERP")
    # m=False → taker bought; m=True → taker sold
    c.handle({"e": "aggTrade", "s": "BTCUSDT", "p": "50000.5", "q": "0.2", "T": 1_700_000_000_000, "m": False})
    c.handle({"e": "aggTrade", "s": "BTCUSDT", "p": "50000.0", "q": "0.5", "T": 1_700_000_001_000, "m": True})

    f = fresh_store.flow("PERP:BTCUSDT")
    assert f["n_trades"] == 2
    assert f["buy_vol"] == pytest.approx(0.2)
    assert f["sell_vol"] == pytest.approx(0.5)
    assert f["cvd"] == pytest.approx(-0.3)


def test_non_trade_frames_ignored(fresh_store):
    c = bstream.BinanceTradeStream(bstream.SPOT_WS, "SPOT")
    c.handle({"result": None, "id": 1})           # subscribe ack
    c.handle(["unexpected", "list"])              # junk
    c.handle({"e": "kline", "s": "BTCUSDT"})      # other event type
    assert fresh_store.flow("SPOT:BTCUSDT") is None


def test_prefix_separates_markets(fresh_store):
    perp = bstream.BinanceTradeStream(bstream.PERP_WS, "PERP")
    spot = bstream.BinanceTradeStream(bstream.SPOT_WS, "SPOT")
    frame = {"e": "aggTrade", "s": "ETHUSDT", "p": "3000", "q": "1.0", "T": 1_700_000_000_000, "m": False}
    perp.handle(frame)
    spot.handle(frame)
    assert fresh_store.flow("PERP:ETHUSDT")["n_trades"] == 1
    assert fresh_store.flow("SPOT:ETHUSDT")["n_trades"] == 1


# --- BybitTradeStream.handle → store ------------------------------------------

def test_bybit_publictrade_frame_lands_in_store(fresh_store):
    c = bybit_stream.BybitTradeStream("PERP")
    c.handle({"topic": "publicTrade.BTCUSDT", "type": "snapshot", "ts": 1_700_000_000_100,
              "data": [
                  {"T": 1_700_000_000_000, "s": "BTCUSDT", "S": "Buy", "v": "0.4", "p": "50000"},
                  {"T": 1_700_000_000_050, "s": "BTCUSDT", "S": "Sell", "v": "0.1", "p": "49999.5"},
              ]})
    f = fresh_store.flow("PERP:BTCUSDT")
    assert f["n_trades"] == 2
    assert f["cvd"] == pytest.approx(0.3)


def test_bybit_acks_and_pongs_ignored(fresh_store):
    c = bybit_stream.BybitTradeStream("PERP")
    c.handle({"success": True, "ret_msg": "", "op": "subscribe"})
    c.handle({"op": "pong"})
    assert fresh_store.flow("PERP:BTCUSDT") is None


def test_bybit_spot_variant_keys_spot_tape(fresh_store):
    c = bybit_stream.BybitTradeStream("SPOT", url=bybit_stream.SPOT_WS)
    assert c.url == bybit_stream.SPOT_WS
    c.handle({"topic": "publicTrade.HYPEUSDT", "type": "snapshot",
              "data": [{"T": 1_700_000_000_000, "s": "HYPEUSDT", "S": "Buy", "v": "3.0", "p": "70"}]})
    assert fresh_store.flow("SPOT:HYPEUSDT")["cvd"] == pytest.approx(3.0)


# --- HyperliquidTradeStream.handle → store -------------------------------------

def test_hyperliquid_trades_frame_lands_in_store(fresh_store):
    c = hl_stream.HyperliquidTradeStream("SPOT")
    asyncio.run(c.subscribe("HYPEUSDT", "@107"))
    # B = taker bought, A = taker sold; coin id maps back to the symbol
    c.handle({"channel": "trades", "data": [
        {"coin": "@107", "side": "B", "px": "71.5", "sz": "2.0", "time": 1_700_000_000_000},
        {"coin": "@107", "side": "A", "px": "71.4", "sz": "0.5", "time": 1_700_000_001_000},
        {"coin": "@999", "side": "B", "px": "1.0", "sz": "9.9", "time": 1_700_000_002_000},  # unknown coin
    ]})
    f = fresh_store.flow("SPOT:HYPEUSDT")
    assert f["n_trades"] == 2
    assert f["cvd"] == pytest.approx(1.5)


def test_hyperliquid_acks_and_pongs_ignored(fresh_store):
    c = hl_stream.HyperliquidTradeStream("SPOT")
    asyncio.run(c.subscribe("HYPEUSDT", "@107"))
    c.handle({"channel": "subscriptionResponse", "data": {}})
    c.handle({"channel": "pong"})
    assert fresh_store.flow("SPOT:HYPEUSDT") is None


def test_hyperliquid_subscribe_bookkeeping_offline():
    c = hl_stream.HyperliquidTradeStream("SPOT")
    asyncio.run(c.subscribe("hypeusdt", "@107"))
    asyncio.run(c.subscribe("HYPEUSDT", "@107"))   # idempotent, case-normalized
    assert c.symbols == {"HYPEUSDT": "@107"}
    assert c.coins == {"@107": "HYPEUSDT"}
    asyncio.run(c.unsubscribe("HYPEUSDT"))
    assert c.symbols == {} and c.coins == {}
    asyncio.run(c.unsubscribe("HYPEUSDT"))         # no-op, no raise


# --- subscribe/unsubscribe bookkeeping (no connection) ------------------------

def test_subscribe_bookkeeping_offline():
    c = bstream.BinanceTradeStream(bstream.PERP_WS, "PERP")
    asyncio.run(c.subscribe("btcusdt"))
    asyncio.run(c.subscribe("BTCUSDT"))       # idempotent, case-normalized
    assert c.symbols == {"BTCUSDT"}
    asyncio.run(c.unsubscribe("BTCUSDT"))
    assert c.symbols == set()
    asyncio.run(c.unsubscribe("BTCUSDT"))     # no-op, no raise


def test_resubscribe_params_on_open():
    c = bstream.BinanceTradeStream(bstream.PERP_WS, "PERP")
    c.symbols = {"BTCUSDT", "ETHUSDT"}
    sent = []

    async def fake_send(payload):
        sent.append(payload)

    c.send = fake_send
    c._ws = object()   # pretend connected so _request actually sends

    asyncio.run(c.on_open(None))
    assert len(sent) == 1
    assert sent[0]["method"] == "SUBSCRIBE"
    assert sorted(sent[0]["params"]) == ["btcusdt@aggTrade", "ethusdt@aggTrade"]


# --- manager -------------------------------------------------------------------

class _FakeClient:
    instances = []

    def __init__(self, *args, **kwargs):
        self.subscribed = []
        self.unsubscribed = []
        _FakeClient.instances.append(self)

    async def run(self):
        return

    async def subscribe(self, symbol, coin=None):
        self.subscribed.append(symbol)

    async def unsubscribe(self, symbol):
        self.unsubscribed.append(symbol)


@pytest.fixture
def fake_manager(monkeypatch):
    _FakeClient.instances = []
    monkeypatch.setattr(manager.binance, "BinanceTradeStream", _FakeClient)
    monkeypatch.setattr(manager.bybit, "BybitTradeStream", _FakeClient)
    monkeypatch.setattr(manager.hyperliquid, "HyperliquidTradeStream", _FakeClient)
    monkeypatch.setattr(manager, "_clients", {})
    monkeypatch.setattr(manager, "_budget_crypto", manager.SubscriptionBudget(budget=2))
    monkeypatch.setattr(manager, "_spot_route", {})
    # no network in tests: everything routes to Binance unless a test re-patches
    monkeypatch.setattr(manager, "_resolve_spot_route",
                        lambda s: manager._spot_route.setdefault(s, ("binance", s)))
    return manager


def _drain(mgr):
    """Wait for fire-and-forget coroutines submitted to the loop thread."""
    fut = asyncio.run_coroutine_threadsafe(asyncio.sleep(0), mgr._ensure_loop())
    fut.result(timeout=5)


def test_manager_lazy_start_and_subscribe(fake_manager):
    fake_manager.ensure_subscribed_crypto("btcusdt")
    _drain(fake_manager)
    assert len(_FakeClient.instances) == 2            # perp + spot, created once
    for c in _FakeClient.instances:
        assert c.subscribed == ["BTCUSDT"]

    fake_manager.ensure_subscribed_crypto("BTCUSDT")  # no new clients
    _drain(fake_manager)
    assert len(_FakeClient.instances) == 2


def test_manager_lru_eviction_unsubscribes(fake_manager):
    for sym in ("AUSDT", "BUSDT", "CUSDT"):          # budget=2 → A evicted
        fake_manager.ensure_subscribed_crypto(sym)
    _drain(fake_manager)
    for c in _FakeClient.instances:
        assert c.subscribed == ["AUSDT", "BUSDT", "CUSDT"]
        assert c.unsubscribed == ["AUSDT"]


def test_manager_routes_spot_by_venue(fake_manager, monkeypatch):
    routes = {"BTCUSDT": ("binance", "BTCUSDT"),
              "HYPEUSDT": ("bybit", "HYPEUSDT"),
              "PURRUSDT": ("hyperliquid", "PURR/USDC")}
    monkeypatch.setattr(fake_manager, "_resolve_spot_route",
                        lambda s: fake_manager._spot_route.setdefault(s, routes[s]))

    for sym in routes:
        fake_manager.ensure_subscribed_crypto(sym)
    _drain(fake_manager)

    # perp + one spot client per venue, each symbol only on its own venue
    assert set(fake_manager._clients) == {"perp", "spot:binance", "spot:bybit", "spot:hyperliquid"}
    assert fake_manager._clients["perp"].subscribed == ["BTCUSDT", "HYPEUSDT", "PURRUSDT"]
    assert fake_manager._clients["spot:binance"].subscribed == ["BTCUSDT"]
    assert fake_manager._clients["spot:bybit"].subscribed == ["HYPEUSDT"]
    assert fake_manager._clients["spot:hyperliquid"].subscribed == ["PURRUSDT"]

    # venue label for get_cvd
    assert fake_manager.spot_venue("PURRUSDT") == "hyperliquid"
    assert fake_manager.spot_venue("UNRESOLVED") == "binance"

    # budget=2 → BTCUSDT evicted; the sweep reaches every spot client
    for c in _FakeClient.instances:
        assert c.unsubscribed == ["BTCUSDT"]


# --- get_cvd live wiring ---------------------------------------------------------

def test_get_cvd_live_fields(monkeypatch):
    import mcp_server

    # kline REST path: minimal futures/spot klines with taker-buy volume (idx 9)
    def fake_klines(symbol, interval, limit):
        return [
            [1_700_000_000_000, "100", "101", "99", "100.5", "10", 0, 0, 0, "6"],
            [1_700_000_060_000, "100.5", "102", "100", "101.0", "8", 0, 0, 0, "5"],
        ]

    monkeypatch.setattr(mcp_server.binance, "fetch_klines_futures", fake_klines)
    monkeypatch.setattr(mcp_server.binance, "fetch_klines_spot", fake_klines)
    monkeypatch.setattr(mcp_server.stream_manager, "ensure_subscribed_crypto", lambda s: None)

    # cold store → warming note
    monkeypatch.setattr(mcp_server, "STORE", TapeStore())
    r = mcp_server.get_cvd("BTCUSDT")
    assert r["live"]["perp"] is None and r["live"]["spot"] is None
    assert "warming" in r["summary"]

    # warm store → live ladder present (all trades within the last 5s, so every
    # rung holds them all; 15m coverage too short for a shape read yet)
    warm = TapeStore()
    now_ms = __import__("store").now_ms()
    for i in range(5):
        warm.ingest_trade("PERP:BTCUSDT", now_ms - i * 1000, 100.0, 2.0, "buy")
        warm.ingest_trade("SPOT:BTCUSDT", now_ms - i * 1000, 100.0, 1.0, "sell")
    monkeypatch.setattr(mcp_server, "STORE", warm)
    r = mcp_server.get_cvd("BTCUSDT")
    for rung in ("1m", "5m", "15m"):
        assert r["live"]["perp"]["windows"][rung]["cvd"] == pytest.approx(10.0)
        assert r["live"]["spot"]["windows"][rung]["cvd"] == pytest.approx(-5.0)
    assert r["live"]["perp"]["windows"]["1m"]["n_trades"] == 5
    assert r["live"]["perp"]["shape"] is None      # coverage gate: tape too young
    assert "live tape" in r["summary"]


def test_get_cvd_live_shape_when_covered(monkeypatch):
    import mcp_server
    from store import now_ms

    def fake_klines(symbol, interval, limit):
        return [[1_700_000_000_000, "100", "101", "99", "100.5", "10", 0, 0, 0, "6"]]

    monkeypatch.setattr(mcp_server.binance, "fetch_klines_futures", fake_klines)
    monkeypatch.setattr(mcp_server.binance, "fetch_klines_spot", fake_klines)
    monkeypatch.setattr(mcp_server.stream_manager, "ensure_subscribed_crypto", lambda s: None)

    warm = TapeStore()
    t0 = now_ms()
    # chronological ingest, like a live stream: steady buying across the full
    # 15m window, then a heavy burst in the last minute → accelerating
    for i in range(29, -1, -1):
        warm.ingest_trade("PERP:BTCUSDT", t0 - i * 30_000, 100.0, 1.0, "buy")
    for i in range(9, -1, -1):
        warm.ingest_trade("PERP:BTCUSDT", t0 - i * 1000, 100.0, 5.0, "buy")
    monkeypatch.setattr(mcp_server, "STORE", warm)

    r = mcp_server.get_cvd("BTCUSDT")
    assert r["live"]["perp"]["shape"] == "accelerating"
    assert "(accelerating)" in r["summary"]
