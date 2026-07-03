"""
Tests for the in-memory tape store: flow aggregation, quote snapshots,
staleness/warmth semantics, and the LRU subscription budget. All offline.
Run with `pytest`.
"""
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from store import (  # noqa: E402
    TapeStore, SubscriptionBudget, WARM_MIN_TRADES,
)

T0 = 1_700_000_000_000  # arbitrary base timestamp (ms)


def _mk_store_with_trades(trades, key="PERP:BTCUSDT"):
    s = TapeStore()
    for ts, price, size, side in trades:
        s.ingest_trade(key, ts, price, size, side)
    return s


# --- flow / cvd / taker_ratio ----------------------------------------------

def test_flow_signed_aggregation():
    s = _mk_store_with_trades([
        (T0 + 1000, 100.0, 2.0, "buy"),
        (T0 + 2000, 100.1, 1.0, "sell"),
        (T0 + 3000, 100.2, 3.0, "buy"),
    ])
    f = s.flow("PERP:BTCUSDT")
    assert f["cvd"] == 4.0            # 5 buy - 1 sell
    assert f["buy_vol"] == 5.0
    assert f["sell_vol"] == 1.0
    assert f["taker_ratio"] == 5.0
    assert f["n_trades"] == 3
    assert f["from_ms"] == T0 + 1000 and f["to_ms"] == T0 + 3000


def test_flow_window_cuts_old_trades():
    s = _mk_store_with_trades([
        (T0 + 1000, 100.0, 10.0, "buy"),     # outside window
        (T0 + 60_000, 100.0, 2.0, "sell"),   # inside
        (T0 + 61_000, 100.0, 1.0, "buy"),    # inside
    ])
    f = s.flow("PERP:BTCUSDT", window_s=30, _now_ms=T0 + 62_000)
    assert f["n_trades"] == 2
    assert f["cvd"] == -1.0


def test_flow_excludes_unsided_trades():
    s = _mk_store_with_trades([
        (T0 + 1000, 100.0, 5.0, None),
        (T0 + 2000, 100.0, 2.0, "buy"),
    ])
    f = s.flow("PERP:BTCUSDT")
    assert f["n_trades"] == 1
    assert f["cvd"] == 2.0


def test_flow_none_when_empty_or_all_unsided():
    s = TapeStore()
    assert s.flow("PERP:BTCUSDT") is None
    s.ingest_trade("PERP:BTCUSDT", T0, 100.0, 1.0, None)
    assert s.flow("PERP:BTCUSDT") is None
    assert s.cvd("PERP:BTCUSDT") is None
    assert s.taker_ratio("PERP:BTCUSDT") is None


def test_taker_ratio_none_when_no_sells():
    s = _mk_store_with_trades([(T0, 100.0, 1.0, "buy")])
    assert s.taker_ratio("PERP:BTCUSDT") is None
    assert s.cvd("PERP:BTCUSDT") == 1.0


def test_tapes_are_independent_per_key():
    s = TapeStore()
    s.ingest_trade("PERP:BTCUSDT", T0, 100.0, 1.0, "buy")
    s.ingest_trade("SPOT:BTCUSDT", T0, 100.0, 3.0, "sell")
    assert s.cvd("PERP:BTCUSDT") == 1.0
    assert s.cvd("SPOT:BTCUSDT") == -3.0


# --- quotes ------------------------------------------------------------------

def test_quote_snapshot_and_spread():
    s = TapeStore()
    assert s.quote("IEX:AAPL") is None
    s.ingest_quote("IEX:AAPL", T0, bid=100.0, ask=100.2, bid_size=300, ask_size=200)
    q = s.quote("IEX:AAPL")
    assert q["bid"] == 100.0 and q["ask"] == 100.2
    assert q["spread"] == pytest.approx(0.2)
    assert q["mid"] == pytest.approx(100.1)
    assert q["spread_pct"] == pytest.approx(0.2 / 100.1 * 100)
    assert q["ts_ms"] == T0


# --- staleness / warmth ------------------------------------------------------

def test_age_and_warmth():
    s = TapeStore()
    assert s.age_s("PERP:BTCUSDT") is None
    assert s.is_warm("PERP:BTCUSDT") is False

    for i in range(WARM_MIN_TRADES):
        s.ingest_trade("PERP:BTCUSDT", T0 + i * 1000, 100.0, 1.0, "buy")
    assert s.is_warm("PERP:BTCUSDT") is True
    last = T0 + (WARM_MIN_TRADES - 1) * 1000
    assert s.age_s("PERP:BTCUSDT", _now_ms=last + 5_000) == 5.0


def test_quote_updates_freshness_but_not_warmth():
    s = TapeStore()
    s.ingest_quote("IEX:AAPL", T0, 100.0, 100.1)
    assert s.age_s("IEX:AAPL", _now_ms=T0 + 1_000) == 1.0
    assert s.is_warm("IEX:AAPL") is False   # warmth needs trades


# --- subscription budget (LRU) ----------------------------------------------

def test_budget_lru_eviction():
    b = SubscriptionBudget(budget=2)
    assert b.touch("A") == []
    assert b.touch("B") == []
    assert b.touch("A") == []          # refresh A → B is now oldest
    assert b.touch("C") == ["B"]       # evicts LRU
    assert sorted(b.active()) == ["A", "C"]


def test_budget_touch_existing_no_eviction():
    b = SubscriptionBudget(budget=1)
    assert b.touch("A") == []
    assert b.touch("A") == []
    assert b.active() == ["A"]


def test_budget_drop():
    b = SubscriptionBudget(budget=2)
    b.touch("A")
    b.drop("A")
    assert b.active() == []


# --- thread-safety smoke ------------------------------------------------------

def test_concurrent_ingest_and_read():
    s = TapeStore()
    key = "PERP:BTCUSDT"
    n = 2000

    def writer():
        for i in range(n):
            s.ingest_trade(key, T0 + i, 100.0, 1.0, "buy" if i % 2 else "sell")

    t = threading.Thread(target=writer)
    t.start()
    for _ in range(200):
        s.flow(key)      # must never raise mid-ingest
        s.age_s(key)
    t.join()

    f = s.flow(key)
    assert f["n_trades"] == n
    assert f["cvd"] == 0.0   # equal buys and sells
