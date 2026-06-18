"""
Pure computations over candle / order-book data. No network, no formatting.
Each function returns plain numbers or dicts that formatting.py renders to text.
"""

FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]


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
    """Bundle OBV(+trend), volume ratio, and ADX/+DI/-DI in one pass so callers
    compute the math once and formatters render from the result (no recompute)."""
    obv, obv_trend = calc_obv(candles)
    vol_ratio      = calc_volume_ratio(candles)
    adx, pdi, ndi  = calc_adx(candles)
    return {
        "obv": obv,
        "obv_trend": obv_trend,
        "volume_ratio": vol_ratio,
        "adx": adx,
        "plus_di": pdi,
        "minus_di": ndi,
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
