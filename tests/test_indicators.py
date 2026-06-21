"""
Unit tests for the pure math in indicators.py — hand-checkable fixtures with
known expected values. No network. Run with `pytest`.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators import (  # noqa: E402
    calc_ema, calc_atr, calc_vwap, calc_adx, calc_obv, calc_volume_ratio,
    calc_volume_profile, analyze_orderbook, calc_fibs, compute_indicators,
    classify_regime, classify_oi_price, classify_long_short, position_size,
    LS_EXTREME_LONG, LS_EXTREME_SHORT,
)


def candle(t, o, h, l, c, v):
    """Binance kline shape: [openTime, open, high, low, close, volume]."""
    return [t, o, h, l, c, v]


# --- calc_ema -------------------------------------------------------------

def test_calc_ema_sma_seed_and_recurrence():
    # seed = mean(1,2,3) = 2; k = 0.5; then 4->3.0, 5->4.0
    assert calc_ema([1, 2, 3, 4, 5], 3) == pytest.approx(4.0)


def test_calc_ema_none_when_too_few():
    assert calc_ema([1, 2], 3) is None


# --- calc_atr -------------------------------------------------------------

def test_calc_atr_constant_true_range():
    # every bar has TR = 2, so Wilder ATR settles at exactly 2
    candles = [
        candle(0, 9, 10, 8, 9, 1),
        candle(1, 10, 11, 9, 10, 1),
        candle(2, 11, 12, 10, 11, 1),
        candle(3, 12, 13, 11, 12, 1),
    ]
    assert calc_atr(candles, period=2) == pytest.approx(2.0)


def test_calc_atr_none_when_too_few():
    candles = [candle(0, 1, 2, 0, 1, 1), candle(1, 1, 2, 0, 1, 1)]
    assert calc_atr(candles, period=14) is None


# --- calc_vwap ------------------------------------------------------------

def test_calc_vwap_and_sigma():
    candles = [
        candle(1000, 2, 2, 2, 2, 10),   # tp = 2
        candle(2000, 4, 4, 4, 4, 10),   # tp = 4
    ]
    vwap, sigma, bars = calc_vwap(candles)
    assert vwap == pytest.approx(3.0)
    assert sigma == pytest.approx(1.0)
    assert bars == 2


def test_calc_vwap_session_filter():
    candles = [
        candle(1000, 2, 2, 2, 2, 10),
        candle(2000, 4, 4, 4, 4, 10),
    ]
    vwap, sigma, bars = calc_vwap(candles, session_start_ts=1500)
    assert vwap == pytest.approx(4.0)
    assert sigma == pytest.approx(0.0)
    assert bars == 1


def test_calc_vwap_zero_volume():
    candles = [candle(0, 1, 1, 1, 1, 0), candle(1, 2, 2, 2, 2, 0)]
    assert calc_vwap(candles) == (None, None, 0)


# --- calc_adx -------------------------------------------------------------

def test_calc_adx_clean_uptrend():
    candles = [candle(i, i, i + 1, i, i + 0.5, 1) for i in range(40)]
    adx, pdi, ndi = calc_adx(candles)
    assert pdi > ndi          # rising market => +DI dominates
    assert adx > 0
    assert 0 <= adx <= 100


def test_calc_adx_too_short_returns_zeros():
    candles = [candle(i, i, i + 1, i, i + 0.5, 1) for i in range(5)]
    assert calc_adx(candles) == (0.0, 0.0, 0.0)


# --- calc_obv -------------------------------------------------------------

def test_calc_obv_rising():
    candles = [
        candle(0, 10, 10, 10, 10, 5),
        candle(1, 11, 11, 11, 11, 5),
        candle(2, 12, 12, 12, 12, 5),
        candle(3, 13, 13, 13, 13, 5),
    ]
    obv, trend = calc_obv(candles)
    assert obv == pytest.approx(15.0)
    assert trend == "rising"


def test_calc_obv_small_input_is_flat_not_crash():
    obv, trend = calc_obv([candle(0, 1, 1, 1, 1, 5)])
    assert obv == 0.0
    assert trend == "flat"


# --- calc_volume_ratio ----------------------------------------------------

def test_calc_volume_ratio_basic():
    # prior 20 bars avg 10, last bar 20 => ratio 2.0
    candles = [candle(i, 1, 1, 1, 1, 10) for i in range(20)]
    candles.append(candle(20, 1, 1, 1, 1, 20))
    assert calc_volume_ratio(candles) == pytest.approx(2.0)


def test_calc_volume_ratio_small_input_no_crash():
    candles = [candle(0, 1, 1, 1, 1, 10), candle(1, 1, 1, 1, 1, 20)]
    assert calc_volume_ratio(candles) == pytest.approx(2.0)


# --- calc_volume_profile --------------------------------------------------

def test_calc_volume_profile_poc_and_value_area():
    candles = [candle(i, 100, 100, 100, 100, 100) for i in range(3)]
    candles.append(candle(3, 110, 110, 110, 110, 1))
    vp = calc_volume_profile(candles, bins=10)
    assert vp is not None
    assert vp["poc"] < 105                       # POC sits at the 100 cluster
    assert vp["val"] <= vp["poc"] <= vp["vah"]
    assert vp["p_min"] == 100 and vp["p_max"] == 110


def test_calc_volume_profile_degenerate_range():
    candles = [candle(i, 100, 100, 100, 100, 5) for i in range(5)]
    assert calc_volume_profile(candles) is None


# --- analyze_orderbook ----------------------------------------------------

def test_analyze_orderbook_buy_pressure():
    depth = {"bids": [[100, 10], [99, 10]], "asks": [[101, 5], [102, 5]]}
    a = analyze_orderbook(depth)
    assert a["best_bid"] == 100
    assert a["best_ask"] == 101
    assert a["spread"] == pytest.approx(1.0)
    assert a["spread_pct"] == pytest.approx(1.0)
    for lvl in a["levels"]:
        assert lvl["ratio"] == pytest.approx(2.0)
        assert lvl["pressure"] == "buy"


def test_analyze_orderbook_infinite_ratio_when_no_asks():
    depth = {"bids": [[100, 10]], "asks": []}
    a = analyze_orderbook(depth)
    assert a["best_ask"] == 0
    assert all(math.isinf(lvl["ratio"]) for lvl in a["levels"])


# --- calc_fibs ------------------------------------------------------------

def test_calc_fibs_levels_and_closest():
    candles = [
        candle(0, 8, 10, 5, 8, 1),
        candle(1, 8, 8, 0, 5, 1),
    ]
    fibs = calc_fibs(candles)
    assert fibs["swing_high"] == 10
    assert fibs["swing_low"] == 0
    # 50% retracement of [0,10] from the high = 5, which equals current close
    assert fibs["levels"][3]["price"] == pytest.approx(5.0)
    assert fibs["closest_index"] == 3


def test_calc_fibs_zero_range():
    candles = [candle(0, 5, 5, 5, 5, 1), candle(1, 5, 5, 5, 5, 1)]
    assert calc_fibs(candles) is None


# --- compute_indicators ---------------------------------------------------

def test_compute_indicators_shape():
    candles = [candle(i, i, i + 1, i, i + 0.5, 1) for i in range(40)]
    out = compute_indicators(candles)
    assert set(out) == {"obv", "obv_trend", "volume_ratio", "adx", "plus_di", "minus_di"}


# --- classify_regime ------------------------------------------------------

def test_classify_regime_trend_up():
    # strong ADX, +DI dominant, price above 200-EMA → uptrend, trend-following
    r = classify_regime(adx=30, plus_di=30, minus_di=10, close=110, ema_200=100)
    assert r["regime"] == "trend_up"
    assert r["mode"] == "trend-following"
    assert r["adx_state"] == "trending"
    assert r["di_direction"] == "bullish"
    assert r["above_200ema"] is True


def test_classify_regime_trend_down():
    r = classify_regime(adx=30, plus_di=10, minus_di=30, close=90, ema_200=100)
    assert r["regime"] == "trend_down"
    assert r["mode"] == "trend-following"
    assert r["di_direction"] == "bearish"


def test_classify_regime_range_suppresses_di():
    # weak ADX → ranging, DI not actionable even though +DI > -DI
    r = classify_regime(adx=12, plus_di=25, minus_di=10, close=110, ema_200=100)
    assert r["regime"] == "range"
    assert r["mode"] == "mean-reversion"
    assert r["adx_state"] == "ranging"
    assert r["di_direction"] is None


def test_classify_regime_developing_is_transitional():
    r = classify_regime(adx=22, plus_di=20, minus_di=10, close=110, ema_200=100)
    assert r["adx_state"] == "developing"
    assert r["regime"] == "transitional"
    assert r["mode"] == "stand-aside"


def test_classify_regime_conflicted_when_di_disagrees_with_200ema():
    # strong ADX, +DI dominant, but price BELOW the 200-EMA → conflicted
    r = classify_regime(adx=30, plus_di=30, minus_di=10, close=90, ema_200=100)
    assert r["regime"] == "transitional"
    assert r["di_direction"] == "bullish"
    assert r["above_200ema"] is False


def test_classify_regime_no_200ema_degrades_gracefully():
    r = classify_regime(adx=30, plus_di=30, minus_di=10, close=110, ema_200=None)
    assert r["above_200ema"] is None
    assert r["regime"] == "trend_up"          # falls back to ADX + DI only


# --- classify_oi_price ----------------------------------------------------

def test_classify_oi_price_four_quadrants():
    assert classify_oi_price(2.0, 2.0)["quadrant"] == "long_buildup"
    assert classify_oi_price(2.0, -2.0)["quadrant"] == "short_buildup"
    assert classify_oi_price(-2.0, 2.0)["quadrant"] == "short_covering"
    assert classify_oi_price(-2.0, -2.0)["quadrant"] == "long_liquidation"


def test_classify_oi_price_flat_is_neutral():
    # both axes inside the ±flat band → neutral
    assert classify_oi_price(0.01, 0.01)["quadrant"] == "neutral"
    # one axis flat → neutral
    assert classify_oi_price(2.0, 0.0)["quadrant"] == "neutral"


def test_classify_oi_price_none_inputs():
    assert classify_oi_price(None, 2.0)["quadrant"] == "neutral"


# --- classify_long_short --------------------------------------------------

def test_classify_long_short_extreme_long_is_contrarian_bearish():
    r = classify_long_short(LS_EXTREME_LONG + 0.5)
    assert r["reading"] == "extreme_long_crowding"
    assert r["contrarian"] == "bearish"


def test_classify_long_short_extreme_short_is_contrarian_bullish():
    r = classify_long_short(LS_EXTREME_SHORT - 0.1)
    assert r["reading"] == "extreme_short_crowding"
    assert r["contrarian"] == "bullish"


def test_classify_long_short_midrange_is_noise():
    # typical live majors (~1.5-2.6) carry no directional edge
    r = classify_long_short(2.0)
    assert r["reading"] == "neutral"
    assert r["contrarian"] is None


def test_classify_long_short_none():
    assert classify_long_short(None)["reading"] == "neutral"


# --- position_size --------------------------------------------------------

def test_position_size_known_values():
    # 1% of 10000 = 100 risk; stop = 50*2 = 100; qty = 100/100 = 1; notional = 1*2000
    ps = position_size(equity=10000, risk_pct=1.0, entry=2000, atr=50, atr_mult=2.0)
    assert ps["risk_amount"] == pytest.approx(100.0)
    assert ps["stop_distance"] == pytest.approx(100.0)
    assert ps["qty"] == pytest.approx(1.0)
    assert ps["notional"] == pytest.approx(2000.0)


def test_position_size_guards_bad_input():
    assert position_size(0, 1.0, 2000, 50, 2.0) is None
    assert position_size(10000, 1.0, 2000, 0, 2.0) is None
    assert position_size(10000, 0, 2000, 50, 2.0) is None
