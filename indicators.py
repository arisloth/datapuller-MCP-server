"""
Pure computations over candle / order-book data. No network, no formatting.
Each function returns plain numbers or dicts that formatting.py renders to text.
"""

FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

# --- Stage 1 regime / interpretation thresholds ---------------------------
# Crude, asset- and timeframe-specific defaults (see Plan.md "Regime calibration").
# Tune per instrument; majors and alts behave very differently.
ADX_TREND = 25.0   # ADX >= this => trending regime (DI direction is actionable)
ADX_RANGE = 20.0   # ADX <  this => ranging regime; ADX in [20, 25) is transitional
LS_EXTREME_LONG  = 3.0   # global account L/S >= this => overcrowded longs  (contrarian bearish)
LS_EXTREME_SHORT = 0.7   # global account L/S <= this => overcrowded shorts (contrarian bullish)
OI_FLAT_PCT = 0.1        # |change| below this (either axis) => treated as flat in the OI quadrant
FUNDING_EXTREME_APR = 50.0  # |annualized funding| >= this % => overcrowded (~0.046%/8h). Asset-specific.
CORR_HIGH = 0.8   # alt-vs-BTC corr >= this => "just BTC beta" (trade BTC's regime)
CORR_LOW  = 0.5   # corr <= this => decoupled enough that alt-specific setups carry edge


def calc_obv(candles):
    obv = 0.0
    obv_series = []
    for i, k in enumerate(candles):
        close = float(k[4])
        v = float(k[5])
        if i > 0:
            prev_c = float(candles[i - 1][4])
            if close > prev_c:
                obv += v
            elif close < prev_c:
                obv -= v
        obv_series.append(obv)

    # Need at least 2 windows of 1 bar each to compare a trend.
    n = min(5, len(obv_series) // 2)
    if n == 0:
        return obv, "flat"
    recent_avg = sum(obv_series[-n:]) / n
    prior_avg  = sum(obv_series[-2 * n:-n]) / n
    trend = "rising" if recent_avg > prior_avg else "falling"
    return obv, trend


def calc_ema(closes, period: int):
    """Exponential moving average. SMA-seeded over the first `period` closes,
    then EMA(i) = close*k + EMA(i-1)*(1-k) with k = 2/(period+1).
    Returns None if there aren't enough closes to seed."""
    if len(closes) < period:
        return None
    k = 2 / (period + 1)
    ema = sum(closes[:period]) / period
    for c in closes[period:]:
        ema = c * k + ema * (1 - k)
    return ema


def calc_vwap(candles, session_start_ts: int | None = None):
    """Volume-weighted average price + volume-weighted std deviation.

    If session_start_ts (in ms) is given, only candles with openTime >= it are
    included — for daily session VWAP. Without it, computes over the whole list.
    Returns (vwap, sigma, bars_used). All None if no qualifying volume."""
    pv_sum = 0.0
    v_sum = 0.0
    selected = []
    for k in candles:
        if session_start_ts is not None and k[0] < session_start_ts:
            continue
        h, l, c, v = float(k[2]), float(k[3]), float(k[4]), float(k[5])
        tp = (h + l + c) / 3
        pv_sum += tp * v
        v_sum += v
        selected.append((tp, v))
    if v_sum == 0 or not selected:
        return None, None, 0
    vwap = pv_sum / v_sum
    var = sum((tp - vwap) ** 2 * v for tp, v in selected) / v_sum
    return vwap, var ** 0.5, len(selected)


def calc_atr(candles, period: int = 14):
    """Wilder-smoothed Average True Range. None if not enough candles."""
    if len(candles) < period + 1:
        return None
    trs = []
    for i in range(1, len(candles)):
        h  = float(candles[i][2])
        l  = float(candles[i][3])
        pc = float(candles[i - 1][4])
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def calc_volume_profile(candles, bins: int = 24, value_area_pct: float = 0.70):
    """Price-binned volume histogram → POC + value area (default 70%).
    Each candle's volume is distributed evenly across the bins its high–low covers.
    Returns dict with poc / vah / val / top_nodes, or None if range is degenerate."""
    if not candles:
        return None
    highs = [float(k[2]) for k in candles]
    lows  = [float(k[3]) for k in candles]
    vols  = [float(k[5]) for k in candles]
    p_min, p_max = min(lows), max(highs)
    if p_max <= p_min:
        return None
    bin_width = (p_max - p_min) / bins
    bin_vols = [0.0] * bins
    for h, l, v in zip(highs, lows, vols):
        lo_idx = max(0, min(bins - 1, int((min(l, h) - p_min) / bin_width)))
        hi_idx = max(0, min(bins - 1, int((max(l, h) - p_min) / bin_width)))
        n = hi_idx - lo_idx + 1
        share = v / n
        for i in range(lo_idx, hi_idx + 1):
            bin_vols[i] += share

    poc_idx = max(range(bins), key=lambda i: bin_vols[i])
    total = sum(bin_vols)
    target = total * value_area_pct
    lo, hi = poc_idx, poc_idx
    acc = bin_vols[poc_idx]
    # Expand outward from POC, picking the side with more volume in its next 2 bins.
    while acc < target and (lo > 0 or hi < bins - 1):
        above = sum(bin_vols[hi + 1:min(hi + 3, bins)])
        below = sum(bin_vols[max(0, lo - 2):lo])
        if above >= below and hi < bins - 1:
            hi += 1
            acc += bin_vols[hi]
        elif lo > 0:
            lo -= 1
            acc += bin_vols[lo]
        else:
            break

    return {
        "poc":  p_min + (poc_idx + 0.5) * bin_width,
        "vah":  p_min + (hi + 1) * bin_width,
        "val":  p_min + lo * bin_width,
        "p_min": p_min,
        "p_max": p_max,
        "bin_width": bin_width,
        "top_nodes": sorted(
            ({"price": p_min + (i + 0.5) * bin_width, "volume": bin_vols[i]} for i in range(bins)),
            key=lambda x: -x["volume"],
        )[:5],
    }


def calc_volume_ratio(candles, n=20):
    volumes = [float(k[5]) for k in candles]
    if len(volumes) < 2:
        return 0.0
    # exclude last bar (may be incomplete); average over whatever history exists
    prior = volumes[-n - 1:-1]
    avg = sum(prior) / len(prior)
    if avg == 0:
        return 0.0
    return volumes[-1] / avg


def calc_adx(candles, period=14):
    closes = [float(k[4]) for k in candles]
    highs  = [float(k[2]) for k in candles]
    lows   = [float(k[3]) for k in candles]

    trs, pdms, ndms = [], [], []
    for i in range(1, len(candles)):
        h, l = highs[i], lows[i]
        ph, pl, pc = highs[i - 1], lows[i - 1], closes[i - 1]
        tr  = max(h - l, abs(h - pc), abs(l - pc))
        pdm = max(h - ph, 0.0) if (h - ph) > (pl - l) else 0.0
        ndm = max(pl - l, 0.0) if (pl - l) > (h - ph) else 0.0
        trs.append(tr)
        pdms.append(pdm)
        ndms.append(ndm)

    # Wilder's smoothing seed
    s_tr  = sum(trs[:period])
    s_pdm = sum(pdms[:period])
    s_ndm = sum(ndms[:period])

    dx_series = []
    for i in range(period, len(trs)):
        s_tr  = s_tr  - s_tr  / period + trs[i]
        s_pdm = s_pdm - s_pdm / period + pdms[i]
        s_ndm = s_ndm - s_ndm / period + ndms[i]
        if s_tr == 0:
            continue
        pdi = 100 * s_pdm / s_tr
        ndi = 100 * s_ndm / s_tr
        denom = pdi + ndi
        dx = 100 * abs(pdi - ndi) / denom if denom != 0 else 0.0
        dx_series.append((dx, pdi, ndi))

    if not dx_series:
        return 0.0, 0.0, 0.0

    adx = sum(d[0] for d in dx_series[:period]) / min(period, len(dx_series))
    for dx, pdi, ndi in dx_series[period:]:
        adx = (adx * (period - 1) + dx) / period

    _, last_pdi, last_ndi = dx_series[-1]
    return adx, last_pdi, last_ndi


def compute_indicators(candles) -> dict:
    """Bundle the per-timeframe indicators in one pass so callers compute the math
    once and formatters render from the result (no recompute). Includes the Stage-2
    order-flow context (CVD, taker ratio, squeeze) derived from the same candles;
    those fields are None / "n/a" when the rows lack taker volume or are too short."""
    obv, obv_trend = calc_obv(candles)
    vol_ratio      = calc_volume_ratio(candles)
    adx, pdi, ndi  = calc_adx(candles)
    cvd, cvd_trend = calc_cvd(candles)
    div            = cvd_divergence(candles)
    taker_ratio    = calc_taker_ratio(candles)
    sq             = detect_squeeze(candles)
    return {
        "obv": obv,
        "obv_trend": obv_trend,
        "volume_ratio": vol_ratio,
        "adx": adx,
        "plus_di": pdi,
        "minus_di": ndi,
        "cvd": cvd,
        "cvd_trend": cvd_trend,
        "cvd_divergence": div["divergence"],
        "taker_ratio": taker_ratio,
        "squeeze_on": None if sq is None else sq["squeeze_on"],
        "bbw": None if sq is None else sq["bbw"],
        "bbw_state": None if sq is None else sq["state"],
    }


def analyze_orderbook(depth) -> dict:
    bids = [(float(p), float(q)) for p, q in depth["bids"]]
    asks = [(float(p), float(q)) for p, q in depth["asks"]]

    best_bid = bids[0][0] if bids else 0
    best_ask = asks[0][0] if asks else 0
    spread = best_ask - best_bid
    spread_pct = spread / best_bid * 100 if best_bid else 0

    levels = []
    for n in (5, 10, 20):
        bid_vol = sum(q for _, q in bids[:n])
        ask_vol = sum(q for _, q in asks[:n])
        ratio   = bid_vol / ask_vol if ask_vol else float("inf")
        pressure = "buy" if ratio > 1.05 else "sell" if ratio < 0.95 else "neutral"
        levels.append({"depth": n, "bid_vol": bid_vol, "ask_vol": ask_vol, "ratio": ratio, "pressure": pressure})

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "spread_pct": spread_pct,
        "levels": levels,
    }


def calc_fibs(candles) -> dict | None:
    highs = [float(k[2]) for k in candles]
    lows  = [float(k[3]) for k in candles]
    swing_high = max(highs)
    swing_low  = min(lows)
    rng = swing_high - swing_low
    if rng == 0:
        return None

    current_close = float(candles[-1][4])
    levels = []
    closest_index = None
    closest_dist = float("inf")
    for i, lvl in enumerate(FIB_LEVELS):
        # retracement from high down toward low
        price = swing_high - lvl * rng
        dist = abs(current_close - price)
        if dist < closest_dist:
            closest_dist = dist
            closest_index = i
        levels.append({"level": lvl, "price": price})

    return {
        "swing_low": swing_low,
        "swing_high": swing_high,
        "current_close": current_close,
        "levels": levels,
        "closest_index": closest_index,
    }


# --- Stage 1 interpretation layer -----------------------------------------
# These functions add no new data; they reinterpret values the stack already
# computes (ADX/DI, 200-EMA, OI change, L/S ratio, ATR) per Plan.md Stage 1.


def classify_regime(adx, plus_di, minus_di, close, ema_200) -> dict:
    """The meta-filter: classify the environment so the caller knows which
    playbook applies (trend-following vs mean-reversion vs stand-aside).

    Combines ADX strength with price position vs the 200-EMA. DI direction is
    only treated as actionable in a trending regime — below ADX_TREND it whipsaws.
    `ema_200` may be None (too few candles); regime then degrades to ADX + DI only.
    """
    if adx >= ADX_TREND:
        adx_state = "trending"
    elif adx >= ADX_RANGE:
        adx_state = "developing"
    else:
        adx_state = "ranging"

    above_200 = None if ema_200 is None else close > ema_200
    di_direction = None
    if adx >= ADX_TREND:
        di_direction = "bullish" if plus_di > minus_di else "bearish"

    # Default: not enough strength to trust a trend → range/transitional.
    if adx_state == "ranging":
        regime, mode = "range", "mean-reversion"
        playbook = ("Ranging: fade value-area / band edges; suppress breakout & "
                    "EMA/DI crossover signals. DI direction is unreliable here.")
    elif adx_state == "developing":
        regime, mode = "transitional", "stand-aside"
        playbook = ("Developing (ADX 20-25): trend not confirmed. Wait for ADX>25 "
                    "to commit to trend-following, or treat as range for now.")
    else:  # trending — require ADX, DI and 200-EMA side to agree
        bull = di_direction == "bullish" and (above_200 is None or above_200)
        bear = di_direction == "bearish" and (above_200 is None or not above_200)
        if bull:
            regime, mode = "trend_up", "trend-following"
            playbook = ("Uptrend: trade pullback continuations long; suppress "
                        "mean-reversion shorts. Don't fade strength.")
        elif bear:
            regime, mode = "trend_down", "trend-following"
            playbook = ("Downtrend: trade pullback continuations short; suppress "
                        "mean-reversion longs. Don't fade weakness.")
        else:
            # Strong ADX but DI direction disagrees with the 200-EMA side.
            regime, mode = "transitional", "stand-aside"
            playbook = ("Conflicted: ADX is strong but DI direction and the 200-EMA "
                        "side disagree (possible reversal/transition). Reduce size.")

    return {
        "regime": regime,
        "mode": mode,
        "adx_state": adx_state,
        "di_direction": di_direction,
        "above_200ema": above_200,
        "playbook": playbook,
    }


def classify_oi_price(oi_change_pct, price_change_pct, flat: float = OI_FLAT_PCT) -> dict:
    """Open-interest + price quadrant (Plan.md Tier-1 #3). OI alone has no
    direction; pairing its change with price change classifies the flow. Either
    axis within ±`flat`% is treated as flat → neutral (no clean quadrant)."""
    if oi_change_pct is None or price_change_pct is None:
        return {"quadrant": "neutral", "label": "n/a",
                "interpretation": "insufficient open-interest/price history"}

    oi_up = oi_change_pct > flat
    oi_dn = oi_change_pct < -flat
    px_up = price_change_pct > flat
    px_dn = price_change_pct < -flat

    if px_up and oi_up:
        q, label = "long_buildup", "Long build-up"
        interp = "new money confirming the up-move — healthy, genuine demand."
    elif px_dn and oi_up:
        q, label = "short_buildup", "Short build-up"
        interp = "new shorts confirming the down-move — avoid longs."
    elif px_up and oi_dn:
        q, label = "short_covering", "Short covering / squeeze"
        interp = "rally on closing shorts, not new buyers — weaker, less sustainable."
    elif px_dn and oi_dn:
        q, label = "long_liquidation", "Long liquidation / covering"
        interp = "longs closing or forced out — often capitulation; wait for stabilization."
    else:
        q, label = "neutral", "Neutral / flat"
        interp = "OI and/or price ~flat — no clear positioning signal."

    return {"quadrant": q, "label": label, "interpretation": interp}


def classify_long_short(ratio) -> dict:
    """Global account long/short ratio reframed per Plan.md: the absolute level
    is noise; only genuine extremes carry (contrarian) content. Most retail
    accounts sit structurally long, so 'longs dominant' is NOT directional."""
    if ratio is None:
        return {"reading": "neutral", "contrarian": None, "note": "n/a"}
    if ratio >= LS_EXTREME_LONG:
        return {"reading": "extreme_long_crowding", "contrarian": "bearish",
                "note": f"overcrowded longs (>= {LS_EXTREME_LONG:g}) — contrarian bearish / long-squeeze fuel."}
    if ratio <= LS_EXTREME_SHORT:
        return {"reading": "extreme_short_crowding", "contrarian": "bullish",
                "note": f"overcrowded shorts (<= {LS_EXTREME_SHORT:g}) — contrarian bullish / short-squeeze fuel."}
    return {"reading": "neutral", "contrarian": None,
            "note": "mid-range — noise, not directional (absolute level carries no edge)."}


def position_size(equity, risk_pct, entry, atr, atr_mult) -> dict | None:
    """ATR-normalized position sizing (Plan.md Stage 1 #5). Risk a fixed % of
    equity across an ATR-multiple stop so size scales inversely with volatility.
    Returns None on non-positive inputs (can't size)."""
    if not (equity and equity > 0 and risk_pct and risk_pct > 0
            and entry and entry > 0 and atr and atr > 0 and atr_mult and atr_mult > 0):
        return None
    risk_amount = equity * risk_pct / 100
    stop_distance = atr * atr_mult
    qty = risk_amount / stop_distance
    return {
        "risk_amount": risk_amount,
        "stop_distance": stop_distance,
        "qty": qty,
        "notional": qty * entry,
    }


# --- Stage 2 order-flow layer ---------------------------------------------
# CVD and the taker ratio are built from the kline's taker-buy base volume
# (field index 9), available on both spot and futures klines. No trade tape needed.
#   per-candle delta = takerBuy - takerSell = 2*takerBuyBase - totalVolume
# Functions degrade to None / "n/a" when rows lack field 9 (e.g. truncated input).

def _has_taker(candles) -> bool:
    return bool(candles) and len(candles[0]) > 9


def calc_cvd(candles):
    """Cumulative Volume Delta — running net of taker-buy minus taker-sell volume
    (aggressor intent). Returns (cvd, trend). Unlike OBV (close-to-close) this
    signs executed flow; CVD strictly dominates OBV on perps. The absolute value
    is meaningless (start-point dependent) — read the trend/slope. Returns
    (None, "n/a") when the kline rows carry no taker-buy field."""
    if not _has_taker(candles):
        return None, "n/a"
    cvd = 0.0
    series = []
    for k in candles:
        total = float(k[5])
        taker_buy = float(k[9])
        cvd += 2 * taker_buy - total
        series.append(cvd)

    n = min(5, len(series) // 2)
    if n == 0:
        return cvd, "flat"
    recent_avg = sum(series[-n:]) / n
    prior_avg  = sum(series[-2 * n:-n]) / n
    if recent_avg > prior_avg:
        trend = "rising"
    elif recent_avg < prior_avg:
        trend = "falling"
    else:
        trend = "flat"
    return cvd, trend


def calc_taker_ratio(candles, n: int | None = None):
    """Aggressive order flow: taker-buy ÷ taker-sell volume over the last `n`
    candles (whole window if None). >1 = buy-side aggression dominating right now.
    Returns None when taker data is absent or taker-sell volume is zero."""
    if not _has_taker(candles):
        return None
    rows = candles if n is None else candles[-n:]
    taker_buy = sum(float(k[9]) for k in rows)
    total     = sum(float(k[5]) for k in rows)
    taker_sell = total - taker_buy
    if taker_sell <= 0:
        return None
    return taker_buy / taker_sell


def cvd_divergence(candles) -> dict:
    """Coarse slope-based CVD/price divergence (NOT swing-based — read with
    structure). price up + CVD down => bearish (distribution/exhaustion);
    price down + CVD up => bullish (absorption/accumulation)."""
    _, cvd_trend = calc_cvd(candles)
    if cvd_trend == "n/a":
        return {"price_trend": "n/a", "cvd_trend": "n/a", "divergence": "none"}
    closes = [float(k[4]) for k in candles]
    n = min(5, len(closes) // 2)
    if n == 0:
        return {"price_trend": "flat", "cvd_trend": cvd_trend, "divergence": "none"}
    recent = sum(closes[-n:]) / n
    prior  = sum(closes[-2 * n:-n]) / n
    if recent > prior:
        price_trend = "rising"
    elif recent < prior:
        price_trend = "falling"
    else:
        price_trend = "flat"

    divergence = "none"
    if price_trend == "rising" and cvd_trend == "falling":
        divergence = "bearish"
    elif price_trend == "falling" and cvd_trend == "rising":
        divergence = "bullish"
    return {"price_trend": price_trend, "cvd_trend": cvd_trend, "divergence": divergence}


def calc_bollinger(closes, period: int = 20, mult: float = 2.0):
    """Bollinger Bands over the last `period` closes (SMA mid, population stdev).
    Returns {mid, upper, lower, bbw} where bbw=(upper-lower)/mid is the normalized
    band width (volatility/compression gauge). None if too few closes."""
    if len(closes) < period:
        return None
    window = closes[-period:]
    mid = sum(window) / period
    sd = (sum((c - mid) ** 2 for c in window) / period) ** 0.5
    upper = mid + mult * sd
    lower = mid - mult * sd
    bbw = (upper - lower) / mid if mid else 0.0
    return {"mid": mid, "upper": upper, "lower": lower, "bbw": bbw}


def detect_squeeze(candles, period: int = 20, bb_mult: float = 2.0, kc_mult: float = 1.5):
    """TTM-style volatility squeeze: Bollinger Bands inside Keltner Channels =
    a low-volatility coil that often precedes expansion. KC = SMA(period) ±
    kc_mult·ATR(period) (reuses calc_atr). Also reports current BBW and its
    percentile vs the window's rolling BBW history (compressed/normal/expanded).
    None if too few candles."""
    closes = [float(k[4]) for k in candles]
    if len(closes) < period or len(candles) < period + 1:
        return None
    bb = calc_bollinger(closes, period, bb_mult)
    atr = calc_atr(candles, period)
    if bb is None or atr is None:
        return None

    mid = bb["mid"]
    kc_upper = mid + kc_mult * atr
    kc_lower = mid - kc_mult * atr
    squeeze_on = bb["upper"] < kc_upper and bb["lower"] > kc_lower

    # Percentile rank of the current BBW vs its own rolling history.
    bbws = []
    for i in range(period, len(closes) + 1):
        b = calc_bollinger(closes[:i], period, bb_mult)
        if b is not None:
            bbws.append(b["bbw"])
    cur = bb["bbw"]
    pctile = (sum(1 for x in bbws if x <= cur) / len(bbws) * 100) if len(bbws) > 1 else None

    if pctile is None:
        state = "n/a"
    elif pctile <= 25:
        state = "compressed"
    elif pctile >= 75:
        state = "expanded"
    else:
        state = "normal"

    return {
        "squeeze_on": squeeze_on,
        "bbw": cur,
        "bbw_pctile": pctile,
        "bb_upper": bb["upper"],
        "bb_mid": mid,
        "bb_lower": bb["lower"],
        "kc_upper": kc_upper,
        "kc_lower": kc_lower,
        "state": state,
    }


# --- Stage 3 positioning / funding / basis --------------------------------
# Funding has ~zero single-asset predictive power and can stay extreme for weeks
# in a trend — treat APR extremes as contrarian *context*, never a timing trigger.


def percentile_rank(value, series) -> float | None:
    """Percentile (0-100) of `value` within `series` = share of points <= value.
    None if the series is empty."""
    if not series:
        return None
    return sum(1 for x in series if x <= value) / len(series) * 100


def annualize_funding(rate, interval_hours: float) -> float | None:
    """Funding rate (per interval, as a fraction e.g. 0.0001) → annualized APR %.
    Normalizes venues with different intervals (Binance/Bybit 8h, Hyperliquid 1h)."""
    if rate is None or not interval_hours:
        return None
    return rate * (24.0 / interval_hours) * 365 * 100


def infer_funding_interval_hours(history) -> float:
    """Median gap (hours) between consecutive Binance fundingTime entries.
    Defaults to 8.0 when it can't be determined."""
    times = sorted(int(h["fundingTime"]) for h in history if "fundingTime" in h)
    gaps = [(b - a) / 3_600_000 for a, b in zip(times, times[1:]) if b > a]
    if not gaps:
        return 8.0
    gaps.sort()
    mid = len(gaps) // 2
    median = gaps[mid] if len(gaps) % 2 else (gaps[mid - 1] + gaps[mid]) / 2
    return round(median) or 8.0


def classify_funding(apr, threshold: float = FUNDING_EXTREME_APR) -> dict:
    """Annualized funding → contrarian extreme flag. Mid-range is context, not a signal."""
    if apr is None:
        return {"reading": "neutral", "contrarian": None, "note": "n/a"}
    if apr >= threshold:
        return {"reading": "extreme_long_crowding", "contrarian": "bearish",
                "note": f"funding >= +{threshold:g}% APR — overcrowded longs (contrarian bearish / cascade fuel)."}
    if apr <= -threshold:
        return {"reading": "extreme_short_crowding", "contrarian": "bullish",
                "note": f"funding <= -{threshold:g}% APR — overcrowded shorts (contrarian bullish / squeeze setup)."}
    return {"reading": "neutral", "contrarian": None,
            "note": "funding mid-range — context only, not a timing trigger."}


def classify_basis(basis_pct, flat: float = 0.02) -> dict:
    """Perp basis = (mark - index)/index. Positive = contango (leveraged longs paying
    premium); negative = backwardation (fear). Within ±`flat`% reads as ~flat."""
    if basis_pct is None:
        return {"state": "n/a", "note": "n/a"}
    if basis_pct > flat:
        return {"state": "contango", "note": "perp above spot — leveraged long demand."}
    if basis_pct < -flat:
        return {"state": "backwardation", "note": "perp below spot — fear / weak futures sponsorship."}
    return {"state": "flat", "note": "perp ~ spot."}


# --- Stage 4 context layers (correlation, breadth) ------------------------
# Tells you *when* alt-specific analysis is worth doing, and the higher-timeframe
# rotation backdrop. All keyless: correlation from klines, breadth from CoinGecko.


def pct_returns(closes):
    """Bar-to-bar percentage returns from a close series."""
    return [(closes[i] - closes[i - 1]) / closes[i - 1]
            for i in range(1, len(closes)) if closes[i - 1]]


def correlation(xs, ys):
    """Pearson correlation of two equal-length series. None if <2 points or either
    series has zero variance."""
    n = min(len(xs), len(ys))
    if n < 2:
        return None
    xs, ys = xs[-n:], ys[-n:]
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy) ** 0.5


def beta(alt_returns, btc_returns):
    """Beta of alt vs BTC = cov(alt,btc)/var(btc). None if BTC variance is 0."""
    n = min(len(alt_returns), len(btc_returns))
    if n < 2:
        return None
    a, b = alt_returns[-n:], btc_returns[-n:]
    ma = sum(a) / n
    mb = sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    vb = sum((y - mb) ** 2 for y in b)
    if vb == 0:
        return None
    return cov / vb


def classify_correlation(r) -> dict:
    """Alt-vs-BTC correlation → whether alt-specific analysis is worth doing."""
    if r is None:
        return {"level": "n/a", "note": "n/a"}
    if r >= CORR_HIGH:
        return {"level": "high", "note": "moves as leveraged BTC beta — trade BTC's regime, not alt specifics."}
    if r <= -CORR_HIGH:
        return {"level": "inverse", "note": "strongly inverse to BTC — unusual; treat with caution."}
    if r <= CORR_LOW:
        return {"level": "low", "note": "decoupled — alt-specific setups carry independent edge."}
    return {"level": "moderate", "note": "partly BTC-driven — weight alt-specific signals accordingly."}


def classify_rotation(btc_dom_rising, total_cap_change_pct, flat: float = 0.2) -> dict:
    """Capital-rotation read from BTC.D direction + total-cap change. NOTE: raw BTC.D
    includes stablecoins — cross-check USDT.D before calling 'altseason'."""
    up = total_cap_change_pct is not None and total_cap_change_pct > flat
    down = total_cap_change_pct is not None and total_cap_change_pct < -flat
    if btc_dom_rising:
        return {"read": "btc_dominant",
                "note": "BTC.D rising — capital concentrating in BTC; alts likely bleed (check USDT.D)."}
    # BTC.D falling:
    if up:
        return {"read": "alt_rotation",
                "note": "BTC.D falling + total cap rising — rotation into alts (early-altseason tilt; confirm USDT.D)."}
    if down:
        return {"read": "risk_off",
                "note": "BTC.D falling + total cap falling — broad risk-off / 'stablecoin season', not altseason."}
    return {"read": "neutral", "note": "BTC.D easing with flat total cap — no clear rotation."}
