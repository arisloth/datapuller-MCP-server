"""
Tests for the data-provider layer: asset-class routing and the Alpaca adapter.
HTTP is mocked (monkeypatched session) — no network and no API key required.
Run with `pytest`.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from providers import router, alpaca  # noqa: E402
from indicators import calc_ema, calc_atr  # noqa: E402


# --- router.resolve_asset_class -------------------------------------------

def test_resolve_auto_crypto_vs_equity():
    assert router.resolve_asset_class("BTCUSDT") == "crypto"
    assert router.resolve_asset_class("ethusdt") == "crypto"     # case-insensitive
    assert router.resolve_asset_class("AAPL") == "equity"
    assert router.resolve_asset_class("SPY") == "equity"
    assert router.resolve_asset_class("GLD") == "equity"


def test_resolve_override_wins():
    assert router.resolve_asset_class("AAPL", "crypto") == "crypto"
    assert router.resolve_asset_class("BTCUSDT", "equity") == "equity"


def test_resolve_bad_override():
    with pytest.raises(ValueError):
        router.resolve_asset_class("AAPL", "stonks")


# --- alpaca interval mapping ----------------------------------------------

def test_alpaca_interval_map_known():
    assert alpaca.INTERVALS["1h"] == "1Hour"
    assert alpaca.INTERVALS["4h"] == "4Hour"
    assert alpaca.INTERVALS["1d"] == "1Day"


def test_alpaca_unsupported_interval_raises(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    with pytest.raises(ValueError):
        alpaca.fetch_ohlcv("AAPL", "3d", 100)   # Alpaca has no 3-day timeframe


def test_alpaca_missing_creds_raises(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(ValueError):
        alpaca.fetch_ohlcv("AAPL", "1h", 100)


# --- alpaca bar parsing → normalized Binance-style rows --------------------

class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_alpaca_parses_into_normalized_rows(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    sample = {"bars": {"AAPL": [
        {"t": "2026-06-22T13:30:00Z", "o": 100.0, "h": 101.0, "l": 99.5, "c": 100.5, "v": 1000, "n": 50, "vw": 100.2},
        {"t": "2026-06-22T14:30:00Z", "o": 100.5, "h": 102.0, "l": 100.0, "c": 101.5, "v": 1200, "n": 60, "vw": 101.0},
    ]}}
    monkeypatch.setattr(alpaca.SESSION, "get", lambda *a, **k: _FakeResp(sample))

    rows = alpaca.fetch_ohlcv("AAPL", "1h", 2)
    assert len(rows) == 2
    # normalized shape [openTime_ms, o, h, l, c, v]; ms timestamp is an int
    assert isinstance(rows[0][0], int) and rows[0][0] > 0
    assert rows[0][1:] == [100.0, 101.0, 99.5, 100.5, 1000]
    # downstream indicator math consumes the rows unchanged
    closes = [r[4] for r in rows]
    assert calc_ema(closes, 2) is not None
    assert calc_atr(rows, period=1) is not None


def test_alpaca_empty_symbol_returns_empty(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    monkeypatch.setattr(alpaca.SESSION, "get", lambda *a, **k: _FakeResp({"bars": {}}))
    assert alpaca.fetch_ohlcv("ZZZZ", "1h", 5) == []


# --- router dispatches to the right provider ------------------------------

def test_router_dispatches_equity_to_alpaca(monkeypatch):
    monkeypatch.setattr(alpaca, "fetch_ohlcv", lambda s, i, l: [["EQUITY"]])
    monkeypatch.setattr(router.binance, "fetch_klines", lambda s, i, l: [["CRYPTO"]])
    assert router.fetch_ohlcv("AAPL", "1h", 5) == [["EQUITY"]]
    assert router.fetch_ohlcv("BTCUSDT", "1h", 5) == [["CRYPTO"]]
