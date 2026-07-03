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
    calc_cvd, calc_taker_ratio, cvd_divergence, calc_bollinger, detect_squeeze,
    annualize_funding, infer_funding_interval_hours, percentile_rank,
    classify_funding, classify_basis,
    pct_returns, correlation, beta, classify_correlation, classify_rotation,
    detect_candle_patterns, classify_pattern_confirmation,
    classify_confluence,
    LS_EXTREME_LONG, LS_EXTREME_SHORT, FUNDING_EXTREME_APR,
)


def candle(t, o, h, l, c, v):
    """Binance kline shape (first 6 fields): [openTime, open, high, low, close, volume]."""
    return [t, o, h, l, c, v]


def fcandle(t, c, vol, taker_buy, h=None, l=None):
    """Full 12-field kline with taker-buy base volume at index 9 (open=close=c)."""
    h = c if h is None else h
    l = c if l is None else l
    return [t, c, h, l, c, vol, t + 1, vol * c, 1, taker_buy, taker_buy * c, "0"]


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
    assert set(out) == {
        "obv", "obv_trend", "volume_ratio", "adx", "plus_di", "minus_di",
        "cvd", "cvd_trend", "cvd_divergence", "taker_ratio",
        "squeeze_on", "bbw", "bbw_state", "patterns",
    }
    assert isinstance(out["patterns"], list)
    # 6-field candles carry no taker volume → CVD/taker degrade gracefully
    assert out["cvd"] is None and out["cvd_trend"] == "n/a"
    assert out["taker_ratio"] is None
    # squeeze only needs high/low/close, so it still computes
    assert isinstance(out["squeeze_on"], bool)


# --- calc_cvd -------------------------------------------------------------

def test_calc_cvd_rising():
    # each candle: total=10, taker_buy=8 → delta = 2*8-10 = +6; CVD climbs
    candles = [fcandle(i, 100, 10, 8) for i in range(6)]
    cvd, trend = calc_cvd(candles)
    assert cvd == pytest.approx(36.0)
    assert trend == "rising"


def test_calc_cvd_falling():
    candles = [fcandle(i, 100, 10, 2) for i in range(6)]   # delta = 2*2-10 = -6
    cvd, trend = calc_cvd(candles)
    assert cvd == pytest.approx(-36.0)
    assert trend == "falling"


def test_calc_cvd_none_on_truncated_rows():
    candles = [candle(i, 1, 1, 1, 1, 10) for i in range(5)]   # 6-field, no taker field
    assert calc_cvd(candles) == (None, "n/a")


# --- calc_taker_ratio -----------------------------------------------------

def test_calc_taker_ratio_basic():
    # total=10, taker_buy=6 → taker_sell=4 → ratio 1.5
    candles = [fcandle(i, 100, 10, 6) for i in range(4)]
    assert calc_taker_ratio(candles) == pytest.approx(1.5)


def test_calc_taker_ratio_none_without_taker():
    candles = [candle(i, 1, 1, 1, 1, 10) for i in range(4)]
    assert calc_taker_ratio(candles) is None


# --- cvd_divergence -------------------------------------------------------

def test_cvd_divergence_bearish():
    # price rising (closes 100..105) but CVD falling (taker_buy < half of total)
    candles = [fcandle(i, 100 + i, 10, 2) for i in range(6)]
    d = cvd_divergence(candles)
    assert d["price_trend"] == "rising"
    assert d["cvd_trend"] == "falling"
    assert d["divergence"] == "bearish"


def test_cvd_divergence_bullish():
    # price falling but CVD rising → absorption/accumulation
    candles = [fcandle(i, 105 - i, 10, 8) for i in range(6)]
    d = cvd_divergence(candles)
    assert d["price_trend"] == "falling"
    assert d["cvd_trend"] == "rising"
    assert d["divergence"] == "bullish"


def test_cvd_divergence_none_when_aligned():
    candles = [fcandle(i, 100 + i, 10, 8) for i in range(6)]   # price up, CVD up
    assert cvd_divergence(candles)["divergence"] == "none"


# --- calc_bollinger -------------------------------------------------------

def test_calc_bollinger_constant_series_zero_width():
    bb = calc_bollinger([100.0] * 20, period=20)
    assert bb["mid"] == pytest.approx(100.0)
    assert bb["upper"] == pytest.approx(100.0)
    assert bb["lower"] == pytest.approx(100.0)
    assert bb["bbw"] == pytest.approx(0.0)


def test_calc_bollinger_known_values():
    # last 20 = ten 1s + ten 3s → mid 2, population sd 1, ±2σ → [0,4], bbw 2
    bb = calc_bollinger([1.0] * 10 + [3.0] * 10, period=20, mult=2.0)
    assert bb["mid"] == pytest.approx(2.0)
    assert bb["upper"] == pytest.approx(4.0)
    assert bb["lower"] == pytest.approx(0.0)
    assert bb["bbw"] == pytest.approx(2.0)


def test_calc_bollinger_none_too_few():
    assert calc_bollinger([1, 2, 3], period=20) is None


# --- detect_squeeze -------------------------------------------------------

def test_detect_squeeze_on_when_compressed():
    # flat closes (BB collapses) but real high/low range (ATR>0) → BB inside KC
    candles = [fcandle(i, 100, 10, 5, h=101, l=99) for i in range(25)]
    sq = detect_squeeze(candles, period=20)
    assert sq is not None
    assert sq["squeeze_on"] is True


def test_detect_squeeze_off_when_expanded():
    # wide close dispersion (big BB) with tight per-bar range (small ATR) → BB outside KC
    candles = [fcandle(i, 100 + i, 10, 5, h=100 + i + 0.5, l=100 + i - 0.5) for i in range(25)]
    sq = detect_squeeze(candles, period=20)
    assert sq is not None
    assert sq["squeeze_on"] is False


def test_detect_squeeze_none_too_few():
    candles = [fcandle(i, 100, 10, 5) for i in range(10)]
    assert detect_squeeze(candles, period=20) is None


# --- annualize_funding ----------------------------------------------------

def test_annualize_funding_8h_vs_1h():
    # 0.01%/8h → 0.0001 * 3 * 365 * 100 = 10.95% APR
    assert annualize_funding(0.0001, 8.0) == pytest.approx(10.95)
    # same rate hourly → 8x more events → 87.6% APR
    assert annualize_funding(0.0001, 1.0) == pytest.approx(87.6)


def test_annualize_funding_none_inputs():
    assert annualize_funding(None, 8.0) is None
    assert annualize_funding(0.0001, 0) is None


# --- infer_funding_interval_hours -----------------------------------------

def test_infer_funding_interval_8h():
    h = 3_600_000
    hist = [{"fundingTime": i * 8 * h} for i in range(5)]
    assert infer_funding_interval_hours(hist) == pytest.approx(8.0)


def test_infer_funding_interval_default_when_empty():
    assert infer_funding_interval_hours([]) == 8.0


# --- percentile_rank ------------------------------------------------------

def test_percentile_rank():
    series = [1, 2, 3, 4]
    assert percentile_rank(4, series) == pytest.approx(100.0)
    assert percentile_rank(2, series) == pytest.approx(50.0)
    assert percentile_rank(0, series) == pytest.approx(0.0)


def test_percentile_rank_empty():
    assert percentile_rank(5, []) is None


# --- classify_funding -----------------------------------------------------

def test_classify_funding_extreme_long():
    r = classify_funding(FUNDING_EXTREME_APR + 10)
    assert r["reading"] == "extreme_long_crowding"
    assert r["contrarian"] == "bearish"


def test_classify_funding_extreme_short():
    r = classify_funding(-(FUNDING_EXTREME_APR + 10))
    assert r["reading"] == "extreme_short_crowding"
    assert r["contrarian"] == "bullish"


def test_classify_funding_neutral():
    r = classify_funding(10.0)
    assert r["reading"] == "neutral"
    assert r["contrarian"] is None


# --- classify_basis -------------------------------------------------------

def test_classify_basis_states():
    assert classify_basis(0.5)["state"] == "contango"
    assert classify_basis(-0.5)["state"] == "backwardation"
    assert classify_basis(0.0)["state"] == "flat"
    assert classify_basis(None)["state"] == "n/a"


# --- pct_returns / correlation / beta -------------------------------------

def test_pct_returns():
    assert pct_returns([100, 110, 99]) == pytest.approx([0.1, -0.1])


def test_correlation_perfect_positive():
    xs = [1, 2, 3, 4, 5]
    ys = [2, 4, 6, 8, 10]   # ys = 2*xs → r = 1
    assert correlation(xs, ys) == pytest.approx(1.0)


def test_correlation_perfect_negative():
    xs = [1, 2, 3, 4, 5]
    ys = [10, 8, 6, 4, 2]
    assert correlation(xs, ys) == pytest.approx(-1.0)


def test_correlation_zero_variance_is_none():
    assert correlation([1, 1, 1], [1, 2, 3]) is None
    assert correlation([5], [5]) is None


def test_beta_known_slope():
    # alt moves 2x BTC → beta 2
    btc = [0.01, -0.02, 0.03, -0.01]
    alt = [2 * x for x in btc]
    assert beta(alt, btc) == pytest.approx(2.0)


def test_beta_zero_btc_variance_is_none():
    assert beta([0.01, 0.02], [0.0, 0.0]) is None


# --- classify_correlation -------------------------------------------------

def test_classify_correlation_high_vs_low():
    assert classify_correlation(0.9)["level"] == "high"
    assert classify_correlation(0.3)["level"] == "low"
    assert classify_correlation(0.65)["level"] == "moderate"
    assert classify_correlation(None)["level"] == "n/a"


# --- classify_rotation ----------------------------------------------------

def test_classify_rotation_btc_dominant():
    assert classify_rotation(True, 1.0)["read"] == "btc_dominant"


def test_classify_rotation_alt_rotation():
    # BTC.D falling + total cap rising → rotation into alts
    assert classify_rotation(False, 2.0)["read"] == "alt_rotation"


def test_classify_rotation_risk_off():
    # BTC.D falling + total cap falling → risk-off
    assert classify_rotation(False, -2.0)["read"] == "risk_off"


# --- detect_candle_patterns -----------------------------------------------

def _names(candles):
    return [p["pattern"] for p in detect_candle_patterns(candles)]


def test_detect_hammer():
    # small body up top, long lower wick, tiny upper wick
    assert "hammer" in _names([candle(0, 10, 10.3, 9, 10.2, 1)])


def test_detect_shooting_star():
    assert "shooting_star" in _names([candle(0, 10, 11, 9.7, 9.8, 1)])


def test_detect_doji():
    assert "doji" in _names([candle(0, 10, 10.5, 9.5, 10.01, 1)])


def test_detect_marubozu():
    pats = detect_candle_patterns([candle(0, 10, 11, 10, 11, 1)])
    assert {"pattern": "marubozu", "direction": "bullish"} in pats


def test_detect_bullish_engulfing():
    candles = [candle(0, 10, 10, 9, 9, 1),          # prior: bearish
               candle(1, 8.9, 10.2, 8.8, 10.1, 1)]  # current: bullish body engulfs prior
    assert "bullish_engulfing" in _names(candles)


def test_detect_bearish_engulfing():
    candles = [candle(0, 9, 10, 9, 10, 1),
               candle(1, 10.1, 10.2, 8.8, 8.9, 1)]
    assert "bearish_engulfing" in _names(candles)


def test_detect_inside_bar():
    candles = [candle(0, 10, 12, 8, 10, 1),         # wide prior bar
               candle(1, 10, 10.8, 9.5, 10.2, 1)]   # range inside prior
    assert "inside_bar" in _names(candles)


def test_detect_no_pattern():
    assert detect_candle_patterns([candle(0, 10, 10.7, 9.8, 10.5, 1)]) == []


def test_detect_patterns_empty_input():
    assert detect_candle_patterns([]) == []


# --- classify_pattern_confirmation ----------------------------------------

def test_pattern_confirmation_confirmed():
    r = classify_pattern_confirmation("bullish", "rising", 1.2, at_level=True)
    assert r["verdict"] == "confirmed"


def test_pattern_confirmation_weak_without_level():
    r = classify_pattern_confirmation("bullish", "rising", 1.2, at_level=False)
    assert r["verdict"] == "weak"


def test_pattern_confirmation_conflicting():
    r = classify_pattern_confirmation("bullish", "falling", 0.8, at_level=True)
    assert r["verdict"] == "conflicting"


def test_pattern_confirmation_mixed():
    # CVD conflicts (falling) but taker agrees (>1) for a bullish pattern → mixed
    r = classify_pattern_confirmation("bullish", "falling", 1.2, at_level=True)
    assert r["verdict"] == "mixed"


def test_pattern_confirmation_unconfirmed():
    r = classify_pattern_confirmation("bullish", "flat", None, at_level=False)
    assert r["verdict"] == "unconfirmed"


def test_pattern_confirmation_neutral():
    r = classify_pattern_confirmation("neutral", "rising", 1.2, at_level=True)
    assert r["verdict"] == "neutral"


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


def test_classify_regime_developing_aligned_is_early_trend():
    # ADX 20-25 with DI lean and 200-EMA side agreeing → tradable at reduced size
    r = classify_regime(adx=22, plus_di=20, minus_di=10, close=110, ema_200=100)
    assert r["adx_state"] == "developing"
    assert r["regime"] == "transitional"
    assert r["mode"] == "trend-following"
    assert r["conviction"] == "reduced"


def test_classify_regime_developing_misaligned_is_reduced_range():
    # ADX 20-25 but DI lean (bullish) disagrees with 200-EMA side (below) →
    # reduced-size range tactics, not a blanket stand-aside
    r = classify_regime(adx=22, plus_di=20, minus_di=10, close=90, ema_200=100)
    assert r["regime"] == "transitional"
    assert r["mode"] == "mean-reversion"
    assert r["conviction"] == "reduced"


def test_classify_regime_conflicted_when_di_disagrees_with_200ema():
    # strong ADX, +DI dominant, but price BELOW the 200-EMA → conflicted
    r = classify_regime(adx=30, plus_di=30, minus_di=10, close=90, ema_200=100)
    assert r["regime"] == "transitional"
    assert r["mode"] == "stand-aside"
    assert r["conviction"] == "none"
    assert r["di_direction"] == "bullish"
    assert r["above_200ema"] is False


def test_classify_regime_full_conviction_in_trend_and_range():
    assert classify_regime(adx=30, plus_di=30, minus_di=10, close=110, ema_200=100)["conviction"] == "full"
    assert classify_regime(adx=12, plus_di=25, minus_di=10, close=110, ema_200=100)["conviction"] == "full"


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


# --- classify_confluence ----------------------------------------------------

def test_confluence_aligned():
    r = classify_confluence({"regime": "bullish", "cvd": "bullish", "taker": "bullish",
                             "vwap": "neutral", "funding_extreme": None})
    assert r["verdict"] == "aligned"
    assert r["direction"] == "long"
    assert r["agreeing"] == ["cvd", "regime", "taker"]
    assert r["opposing"] == []
    assert "actionable" in r["note"]


def test_confluence_aligned_needs_three_votes():
    # two unopposed agreeing reads → leaning, not a full green light
    r = classify_confluence({"regime": "bearish", "cvd": "bearish", "vwap": "neutral"})
    assert r["verdict"] == "leaning"
    assert r["direction"] == "short"


def test_confluence_leaning_with_minority_opposition():
    r = classify_confluence({"regime": "bullish", "ema_stack": "bullish",
                             "cvd": "bullish", "taker": "bullish", "vwap": "bearish"})
    assert r["verdict"] == "leaning"
    assert r["direction"] == "long"
    assert r["opposing"] == ["vwap"]


def test_confluence_mixed():
    r = classify_confluence({"regime": "bullish", "cvd": "bearish",
                             "taker": "bearish", "ema_stack": "bullish"})
    assert r["verdict"] == "mixed"


def test_confluence_no_signal():
    r = classify_confluence({"regime": "neutral", "cvd": "neutral", "taker": None})
    assert r["verdict"] == "no_signal"
    assert r["direction"] is None
    assert r["neutral"] == ["cvd", "regime"]
