"""
Tests for the equity streaming path: Lee–Ready classification, condition-code
filtering, Alpaca frame parsing, and the get_cvd/get_orderbook equity branches.
All offline — canned frames, monkeypatched manager/store, no creds.
Run with `pytest`.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from store import (  # noqa: E402
    TapeStore, classify_trade, is_flow_eligible,
)
from streams import alpaca as astream  # noqa: E402


# --- classify_trade (Lee–Ready) ----------------------------------------------

def test_lee_ready_quote_rule():
    assert classify_trade(100.09, bid=100.00, ask=100.10) == "buy"    # above mid
    assert classify_trade(100.01, bid=100.00, ask=100.10) == "sell"   # below mid
    assert classify_trade(100.10, bid=100.00, ask=100.10) == "buy"    # at ask
    assert classify_trade(100.00, bid=100.00, ask=100.10) == "sell"   # at bid


def test_lee_ready_midpoint_falls_back_to_tick_test():
    assert classify_trade(100.05, bid=100.00, ask=100.10, prev_price=100.00) == "buy"
    assert classify_trade(100.05, bid=100.00, ask=100.10, prev_price=100.20) == "sell"


def test_lee_ready_zero_tick_inherits_last_tick():
    assert classify_trade(100.05, bid=100.00, ask=100.10,
                          prev_price=100.05, last_tick="up") == "buy"
    assert classify_trade(100.05, bid=100.00, ask=100.10,
                          prev_price=100.05, last_tick="down") == "sell"


def test_lee_ready_no_quote_uses_tick_test():
    assert classify_trade(100.05, bid=None, ask=None, prev_price=100.00) == "buy"


def test_lee_ready_unclassifiable():
    assert classify_trade(100.05, bid=None, ask=None) is None
    assert classify_trade(100.05, bid=None, ask=None, prev_price=100.05) is None


# --- condition-code filtering ---------------------------------------------------

def test_flow_eligibility():
    assert is_flow_eligible(None) is True
    assert is_flow_eligible([]) is True
    assert is_flow_eligible(["@"]) is True           # regular sale
    assert is_flow_eligible(["@", "I"]) is True      # odd lot still counts
    assert is_flow_eligible(["W"]) is False          # average price
    assert is_flow_eligible(["@", "4"]) is False     # derivatively priced


def test_ingest_equity_trade_classifies_and_filters():
    s = TapeStore()
    key = "EQ:AAPL"
    s.ingest_quote(key, 1000, bid=100.00, ask=100.10)
    s.ingest_equity_trade(key, 1001, 100.09, 50, conditions=["@"])       # above mid → buy
    s.ingest_equity_trade(key, 1002, 100.01, 30, conditions=["@"])       # below mid → sell
    s.ingest_equity_trade(key, 1003, 100.05, 999, conditions=["W"])      # excluded print
    f = s.flow(key)
    assert f["n_trades"] == 2
    assert f["buy_vol"] == 50
    assert f["sell_vol"] == 30
    assert s.age_s(key, _now_ms=1004) == pytest.approx(0.001)  # excluded print refreshed clock


def test_ingest_equity_trade_tick_state_across_prints():
    s = TapeStore()
    key = "EQ:AAPL"
    s.ingest_quote(key, 1000, bid=100.00, ask=100.10)
    s.ingest_equity_trade(key, 1001, 100.04, 10)   # below mid → sell; seeds tick state
    s.ingest_equity_trade(key, 1002, 100.05, 10)   # at mid, uptick vs 100.04 → buy
    s.ingest_equity_trade(key, 1003, 100.05, 10)   # at mid, zero-tick, last_tick=up → buy
    f = s.flow(key)
    assert f["buy_vol"] == 20
    assert f["sell_vol"] == 10


# --- Alpaca frame parsing --------------------------------------------------------

@pytest.fixture
def fresh_store(monkeypatch):
    s = TapeStore()
    monkeypatch.setattr(astream, "STORE", s)
    return s


def _client():
    return astream.AlpacaStream(feed="iex")


def test_alpaca_quote_then_trade_classified(fresh_store):
    c = _client()
    c.handle([
        {"T": "q", "S": "AAPL", "bp": 100.00, "bs": 3, "ap": 100.10, "as": 2,
         "t": "2026-07-02T14:30:00.123456789Z"},
        {"T": "t", "S": "AAPL", "p": 100.09, "s": 50, "c": ["@"],
         "t": "2026-07-02T14:30:00.223456789Z"},
    ])
    f = fresh_store.flow("EQ:AAPL")
    assert f["n_trades"] == 1
    assert f["buy_vol"] == 50          # above mid → buy
    q = fresh_store.quote("EQ:AAPL")
    assert q["bid"] == 100.00 and q["ask"] == 100.10


def test_alpaca_control_frames_ignored(fresh_store):
    c = _client()
    c.handle([{"T": "success", "msg": "authenticated"}])
    c.handle([{"T": "subscription", "trades": ["AAPL"], "quotes": ["AAPL"]}])
    c.handle([{"T": "error", "code": 405, "msg": "symbol limit exceeded"}])
    c.handle({"not": "a list"})
    assert fresh_store.flow("EQ:AAPL") is None


def test_alpaca_ts_parsing_nanoseconds():
    from datetime import datetime, timezone
    base_ms = int(datetime(2026, 7, 2, 14, 30, tzinfo=timezone.utc).timestamp() * 1000)
    # nanosecond precision is truncated to µs and floors to the right ms
    assert astream._ts_ms("2026-07-02T14:30:00.123456789Z") == base_ms + 123
    assert astream._ts_ms("2026-07-02T14:30:00Z") == base_ms


# --- tool equity branches ----------------------------------------------------------

def _wire_mcp(monkeypatch, store):
    import mcp_server
    monkeypatch.setattr(mcp_server, "STORE", store)
    monkeypatch.setattr(mcp_server.stream_manager, "ensure_subscribed_equity", lambda s: None)
    return mcp_server


def test_get_cvd_equity_warming_then_live(monkeypatch):
    from store import now_ms
    mcp_server = _wire_mcp(monkeypatch, TapeStore())

    r = mcp_server.get_cvd("AAPL")
    assert r["asset_class"] == "equity"
    assert r["live"] is None
    assert "warming" in r["summary"]

    warm = TapeStore()
    t0 = now_ms()
    warm.ingest_quote("EQ:AAPL", t0 - 5000, bid=100.00, ask=100.10)
    for i in range(10):
        warm.ingest_equity_trade("EQ:AAPL", t0 - i * 100, 100.09, 10, conditions=["@"])
    mcp_server = _wire_mcp(monkeypatch, warm)
    r = mcp_server.get_cvd("AAPL")
    assert r["live"]["windows"]["5m"]["cvd"] == pytest.approx(100.0)
    assert r["live"]["classification"] == "lee-ready"
    assert "Lee–Ready" in r["summary"]
    assert "IEX" in r["summary"]      # partial-tape caveat on the default feed


def test_get_cvd_equity_missing_creds(monkeypatch):
    import mcp_server
    monkeypatch.setattr(
        mcp_server.stream_manager, "ensure_subscribed_equity",
        lambda s: (_ for _ in ()).throw(ValueError("alpaca: set APCA_API_KEY_ID ...")),
    )
    r = mcp_server.get_cvd("AAPL")
    assert "error" in r


def test_get_orderbook_equity_nbbo(monkeypatch):
    warm = TapeStore()
    warm.ingest_quote("EQ:SPY", 1_700_000_000_000, bid=500.00, ask=500.02,
                      bid_size=10, ask_size=7)
    mcp_server = _wire_mcp(monkeypatch, warm)

    r = mcp_server.get_orderbook("SPY")
    assert r["asset_class"] == "equity"
    assert r["best_bid"] == 500.00 and r["best_ask"] == 500.02
    assert r["spread"] == pytest.approx(0.02)
    assert "NBBO" in r["summary"] and "not L2" in r["summary"]

    cold = _wire_mcp(monkeypatch, TapeStore())
    r = cold.get_orderbook("SPY")
    assert "warming" in r["summary"]
