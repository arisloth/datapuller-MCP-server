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
DOJI_BODY_PCT = 0.1      # candle body <= this fraction of range => doji (indecision)
PIN_WICK_RATIO = 2.0     # dominant wick >= this x body => pin bar (hammer / shooting star)
MARUBOZU_WICK_PCT = 0.05 # each wick <= this fraction of range => marubozu (full-body conviction)

# --- Stretch / exhaustion thresholds (the counter-trend layer) --------------
# Crypto trends run hot, so the RSI bands sit wider than the equity-classic 70/30.
RSI_OVERBOUGHT = 75.0
RSI_OVERSOLD   = 25.0
VWAP_SIGMA_EXTREME = 2.0    # |close - VWAP| >= this many σ => stretched
# Percentile gates for the contrarian positioning reads. The absolute floors stop
# "90th percentile of a flat regime" from reading as an extreme.
EXTREME_PCTL_HIGH = 90.0
EXTREME_PCTL_LOW  = 10.0
FUNDING_MIN_EXTREME_APR = 20.0  # percentile gate needs at least this |APR|
LS_PCTL_FLOOR_LONG  = 1.5       # percentile gate needs L/S at least this rich...
LS_PCTL_FLOOR_SHORT = 1.0       # ...or at most this poor


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


def calc_rsi(closes, period: int = 14):
    """Wilder-smoothed RSI over the close series. Returns None if there aren't
    at least period+1 closes; 100.0 when the window has no losses at all."""
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for g, l in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
    if avg_loss == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def find_swing_points(values, left: int = 2, right: int = 2):
    """Fractal pivots: index i is a swing high when values[i] beats the `left`
    values before it strictly and the `right` after it non-strictly (mirrored
    for lows — the asymmetry keeps flat-topped double prints from vanishing).
    Returns (high_idxs, low_idxs), both ascending. The last `right` bars can
    never confirm a pivot yet, so fresh extremes appear only after `right` bars."""
    highs, lows = [], []
    for i in range(left, len(values) - right):
        v = values[i]
        before = values[i - left:i]
        after = values[i + 1:i + 1 + right]
        if all(v > x for x in before) and all(v >= x for x in after):
            highs.append(i)
        if all(v < x for x in before) and all(v <= x for x in after):
            lows.append(i)
    return highs, lows


def classify_stretch(close, rsi, vwap, vwap_sigma) -> dict:
    """Stretch/exhaustion read — the counterweight to the trend cluster. Combines
    RSI(14) bands with the close's σ-distance from (session) VWAP. `contrarian`
    fires only when EVERY available measure agrees the price is extended (a single
    available measure may fire alone); a split read stays 'normal'. Degrades to
    'n/a' when neither measure is available."""
    vwap_sigmas = None
    if vwap is not None and vwap_sigma:
        vwap_sigmas = (close - vwap) / vwap_sigma

    def side(value, hi, lo):
        if value is None:
            return None
        return "up" if value >= hi else "down" if value <= lo else "normal"

    reads = [s for s in (side(rsi, RSI_OVERBOUGHT, RSI_OVERSOLD),
                         side(vwap_sigmas, VWAP_SIGMA_EXTREME, -VWAP_SIGMA_EXTREME))
             if s is not None]
    if not reads:
        state, contrarian = "n/a", None
    elif all(s == "up" for s in reads):
        state, contrarian = "extended_up", "bearish"
    elif all(s == "down" for s in reads):
        state, contrarian = "extended_down", "bullish"
    else:
        state, contrarian = "normal", None
    return {"state": state, "rsi": rsi, "vwap_sigmas": vwap_sigmas, "contrarian": contrarian}


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
    """Volume of the last CLOSED bar vs the average of the n bars before it.
    The newest bar is excluded entirely — it's usually still forming, and an
    in-progress numerator systematically understates the ratio."""
    volumes = [float(k[5]) for k in candles]
    if len(volumes) < 3:
        return 0.0
    prior = volumes[-n - 2:-2]
    avg = sum(prior) / len(prior)
    if avg == 0:
        return 0.0
    return volumes[-2] / avg


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
    patterns       = detect_candle_patterns(candles)
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
        "patterns": patterns,
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

    # Conviction grades the environment instead of a binary go/no-go:
    # 'full' = the playbook applies at normal size, 'reduced' = tradable at
    # smaller size with the stated condition, 'none' = genuinely no edge.
    if adx_state == "ranging":
        regime, mode, conviction = "range", "mean-reversion", "full"
        playbook = ("Ranging: fade value-area / band edges; suppress breakout & "
                    "EMA/DI crossover signals. DI direction is unreliable here.")
    elif adx_state == "developing":
        # Trend not confirmed, but 20-25 is tradable at reduced size — grade the
        # lean by whether tentative DI direction agrees with the 200-EMA side.
        lean = "bullish" if plus_di > minus_di else "bearish"
        lean_aligned = above_200 is None or (above_200 if lean == "bullish" else not above_200)
        regime = "transitional"
        if lean_aligned:
            mode, conviction = "trend-following", "reduced"
            playbook = (f"Developing (ADX 20-25), DI lean and 200-EMA side agree ({lean}): "
                        "early-trend continuations are valid at reduced size; "
                        "go full size once ADX clears 25.")
        else:
            mode, conviction = "mean-reversion", "reduced"
            playbook = ("Developing (ADX 20-25), direction unresolved: range tactics "
                        "at reduced size; commit to trend-following when ADX>25 with "
                        "DI and the 200-EMA side agreeing.")
    else:  # trending — require ADX, DI and 200-EMA side to agree
        bull = di_direction == "bullish" and (above_200 is None or above_200)
        bear = di_direction == "bearish" and (above_200 is None or not above_200)
        if bull:
            regime, mode, conviction = "trend_up", "trend-following", "full"
            playbook = ("Uptrend: trade pullback continuations long; suppress "
                        "mean-reversion shorts. Don't fade strength.")
        elif bear:
            regime, mode, conviction = "trend_down", "trend-following", "full"
            playbook = ("Downtrend: trade pullback continuations short; suppress "
                        "mean-reversion longs. Don't fade weakness.")
        else:
            # Strong ADX but DI direction disagrees with the 200-EMA side.
            regime, mode, conviction = "transitional", "stand-aside", "none"
            playbook = ("Conflicted: ADX is strong but DI direction and the 200-EMA "
                        "side disagree (possible reversal/transition). Stand aside "
                        "or size minimal; resolves when price closes back on the DI "
                        "side of the 200-EMA.")

    return {
        "regime": regime,
        "mode": mode,
        "conviction": conviction,
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


def classify_long_short(ratio, percentile=None) -> dict:
    """Global account long/short ratio reframed per Plan.md: the absolute level
    is noise; only genuine extremes carry (contrarian) content. Most retail
    accounts sit structurally long, so 'longs dominant' is NOT directional.
    Extremes fire on the absolute gates OR on the ratio's percentile vs its own
    recent history (>=90th / <=10th, with a mild absolute floor) — the absolute
    gates alone sit outside the ratio's realistic range on majors."""
    if ratio is None:
        return {"reading": "neutral", "contrarian": None, "note": "n/a"}
    pctl_long = (percentile is not None and percentile >= EXTREME_PCTL_HIGH
                 and ratio >= LS_PCTL_FLOOR_LONG)
    pctl_short = (percentile is not None and percentile <= EXTREME_PCTL_LOW
                  and ratio <= LS_PCTL_FLOOR_SHORT)
    if ratio >= LS_EXTREME_LONG or pctl_long:
        gate = (f">= {LS_EXTREME_LONG:g}" if ratio >= LS_EXTREME_LONG
                else f"{percentile:.0f}th pct of its own history")
        return {"reading": "extreme_long_crowding", "contrarian": "bearish",
                "note": f"overcrowded longs ({gate}) — contrarian bearish / long-squeeze fuel."}
    if ratio <= LS_EXTREME_SHORT or pctl_short:
        gate = (f"<= {LS_EXTREME_SHORT:g}" if ratio <= LS_EXTREME_SHORT
                else f"{percentile:.0f}th pct of its own history")
        return {"reading": "extreme_short_crowding", "contrarian": "bullish",
                "note": f"overcrowded shorts ({gate}) — contrarian bullish / short-squeeze fuel."}
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


def _cvd_series(candles):
    """Running CVD value per candle (taker-buy minus taker-sell, cumulative).
    None when the rows carry no taker-buy field."""
    if not _has_taker(candles):
        return None
    cvd = 0.0
    series = []
    for k in candles:
        cvd += 2 * float(k[9]) - float(k[5])
        series.append(cvd)
    return series


def calc_cvd(candles):
    """Cumulative Volume Delta — running net of taker-buy minus taker-sell volume
    (aggressor intent). Returns (cvd, trend). Unlike OBV (close-to-close) this
    signs executed flow; CVD strictly dominates OBV on perps. The absolute value
    is meaningless (start-point dependent) — read the trend/slope. Returns
    (None, "n/a") when the kline rows carry no taker-buy field."""
    series = _cvd_series(candles)
    if series is None:
        return None, "n/a"
    cvd = series[-1]

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
    """Swing-anchored CVD/price divergence: compares the last two price swing lows
    (and highs) against CVD at the same bars. Price lower-low + CVD higher-low =>
    bullish (absorption/accumulation); price higher-high + CVD lower-high =>
    bearish (distribution/exhaustion). This is the read that CAN fire at a local
    extreme — a slope comparison never does, because both slopes agree there.
    Falls back to the coarse slope comparison when the window has no two swings;
    `method` reports which path produced the verdict."""
    series = _cvd_series(candles)
    _, cvd_trend = calc_cvd(candles)
    if cvd_trend == "n/a":
        return {"price_trend": "n/a", "cvd_trend": "n/a", "divergence": "none", "method": "slope"}
    closes = [float(k[4]) for k in candles]
    n = min(5, len(closes) // 2)
    if n == 0:
        return {"price_trend": "flat", "cvd_trend": cvd_trend, "divergence": "none", "method": "slope"}
    recent = sum(closes[-n:]) / n
    prior  = sum(closes[-2 * n:-n]) / n
    if recent > prior:
        price_trend = "rising"
    elif recent < prior:
        price_trend = "falling"
    else:
        price_trend = "flat"

    high_idxs, low_idxs = find_swing_points(closes)
    if len(high_idxs) >= 2 or len(low_idxs) >= 2:
        # Swing path: check both sides, keep whichever divergence is more recent.
        candidates = []
        if len(low_idxs) >= 2:
            a, b = low_idxs[-2], low_idxs[-1]
            if closes[b] < closes[a] and series[b] > series[a]:
                candidates.append((b, "bullish"))
        if len(high_idxs) >= 2:
            a, b = high_idxs[-2], high_idxs[-1]
            if closes[b] > closes[a] and series[b] < series[a]:
                candidates.append((b, "bearish"))
        divergence = max(candidates)[1] if candidates else "none"
        return {"price_trend": price_trend, "cvd_trend": cvd_trend,
                "divergence": divergence, "method": "swing"}

    divergence = "none"
    if price_trend == "rising" and cvd_trend == "falling":
        divergence = "bearish"
    elif price_trend == "falling" and cvd_trend == "rising":
        divergence = "bullish"
    return {"price_trend": price_trend, "cvd_trend": cvd_trend,
            "divergence": divergence, "method": "slope"}


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


def classify_funding(apr, percentile=None, threshold: float = FUNDING_EXTREME_APR) -> dict:
    """Annualized funding → contrarian extreme flag. Mid-range is context, not a
    signal. Extremes fire on the absolute APR threshold OR on funding's percentile
    vs its own settled history (>=90th / <=10th, floored at
    ±FUNDING_MIN_EXTREME_APR so a quiet regime's top decile doesn't read as hot)."""
    if apr is None:
        return {"reading": "neutral", "contrarian": None, "note": "n/a"}
    pctl_long = (percentile is not None and percentile >= EXTREME_PCTL_HIGH
                 and apr >= FUNDING_MIN_EXTREME_APR)
    pctl_short = (percentile is not None and percentile <= EXTREME_PCTL_LOW
                  and apr <= -FUNDING_MIN_EXTREME_APR)
    if apr >= threshold or pctl_long:
        gate = (f">= +{threshold:g}% APR" if apr >= threshold
                else f"{percentile:.0f}th pct of its own history at {apr:+.0f}% APR")
        return {"reading": "extreme_long_crowding", "contrarian": "bearish",
                "note": f"funding {gate} — overcrowded longs (contrarian bearish / cascade fuel)."}
    if apr <= -threshold or pctl_short:
        gate = (f"<= -{threshold:g}% APR" if apr <= -threshold
                else f"{percentile:.0f}th pct of its own history at {apr:+.0f}% APR")
        return {"reading": "extreme_short_crowding", "contrarian": "bullish",
                "note": f"funding {gate} — overcrowded shorts (contrarian bullish / squeeze setup)."}
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
        return {"level": "high", "note": "moves as leveraged BTC beta — let BTC's regime set direction; use alt specifics for entries/levels."}
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


# --- Candlestick patterns (CONFIRMATION-ONLY layer) -----------------------
# Standalone candle patterns are low-edge (like Fib). Detect them, but only treat
# them as confirmation at a level + with order flow agreeing. Pure OHLC math.


def detect_candle_patterns(candles) -> list:
    """Patterns present on the LATEST bar (uses the last 1-2 candles). Returns a
    list of {pattern, direction} — direction is bullish/bearish/neutral. Empty list
    when nothing is found or there's too little/degenerate data. NOTE: the latest
    candle may be in progress; patterns truly confirm on close."""
    if not candles:
        return []
    o = float(candles[-1][1]); h = float(candles[-1][2])
    l = float(candles[-1][3]); c = float(candles[-1][4])
    rng = h - l
    if rng <= 0:
        return []
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    found = []

    # Doji — tiny body relative to range (indecision). Takes precedence over pin/marubozu.
    if body <= DOJI_BODY_PCT * rng:
        found.append({"pattern": "doji", "direction": "neutral"})
    else:
        # Marubozu — negligible wicks both ends (strong conviction).
        if upper <= MARUBOZU_WICK_PCT * rng and lower <= MARUBOZU_WICK_PCT * rng:
            found.append({"pattern": "marubozu", "direction": "bullish" if c > o else "bearish"})
        # Hammer — long lower wick, small upper (rejection of lows → bullish).
        if lower >= PIN_WICK_RATIO * body and upper <= body:
            found.append({"pattern": "hammer", "direction": "bullish"})
        # Shooting star — long upper wick, small lower (rejection of highs → bearish).
        if upper >= PIN_WICK_RATIO * body and lower <= body:
            found.append({"pattern": "shooting_star", "direction": "bearish"})

    # Two-candle engulfing — current opposite-color body engulfs the prior body.
    if len(candles) >= 2:
        po = float(candles[-2][1]); ph = float(candles[-2][2])
        pl = float(candles[-2][3]); pc = float(candles[-2][4])
        if c > o and pc < po and c >= po and o <= pc:
            found.append({"pattern": "bullish_engulfing", "direction": "bullish"})
        elif c < o and pc > po and o >= pc and c <= po:
            found.append({"pattern": "bearish_engulfing", "direction": "bearish"})
        # Inside bar — current range inside the prior range (compression).
        if h < ph and l > pl:
            found.append({"pattern": "inside_bar", "direction": "neutral"})

    return found


def classify_pattern_confirmation(direction, cvd_trend, taker_ratio, at_level,
                                  flow_shape: str | None = None) -> dict:
    """Whether a directional candle pattern is CONFIRMED by order flow + location.
    Patterns are confirmation, never standalone triggers.

    `flow_shape` (optional) is the live tape's ladder state (see
    classify_flow_ladder). Window CVD/taker lag reversals by construction, so a
    reversal pattern against the prior flow would always score 'conflicting' —
    the live shape breaks that: 'flipping' (tape running against the prior flow)
    turns pure conflict into `reversal_watch`, and 'fading' (prior push going
    stale) softens it to 'mixed'."""
    if direction == "neutral":
        return {"verdict": "neutral",
                "note": "indecision / compression — becomes tradable on the breakout side once CVD/taker pick a direction."}

    # Evaluate CVD (primary flow) and taker ratio (per-interval companion) separately,
    # so genuinely mixed flow isn't mislabeled as confirmation.
    if direction == "bullish":
        cvd_agree, cvd_conflict = cvd_trend == "rising", cvd_trend == "falling"
        taker_agree = taker_ratio is not None and taker_ratio > 1
        taker_conflict = taker_ratio is not None and taker_ratio < 1
    else:  # bearish
        cvd_agree, cvd_conflict = cvd_trend == "falling", cvd_trend == "rising"
        taker_agree = taker_ratio is not None and taker_ratio < 1
        taker_conflict = taker_ratio is not None and taker_ratio > 1

    agree = cvd_agree or taker_agree
    conflict = cvd_conflict or taker_conflict

    if agree and conflict:
        return {"verdict": "mixed",
                "note": f"order flow is mixed on the {direction} pattern (CVD and taker disagree) — wait for them to realign before acting."}
    if conflict:
        if flow_shape == "flipping":
            return {"verdict": "reversal_watch",
                    "note": f"{direction} pattern against the prior flow, but the live tape is FLIPPING "
                            f"the same way — early-reversal candidate, not a fakeout read. Confirm on "
                            f"close; if it holds, this is a genuine counter-trend entry at reduced size."}
        if flow_shape == "fading":
            return {"verdict": "mixed",
                    "note": f"order flow disagrees with the {direction} pattern but the live tape shows the "
                            f"prior push FADING — the conflict is stale; watch for the tape to flip before acting."}
        return {"verdict": "conflicting",
                "note": f"order flow disagrees with the {direction} pattern — fakeout risk; the opposite-side read may be the real signal."}
    if agree and at_level:
        return {"verdict": "confirmed",
                "note": f"{direction} pattern at a level with order flow agreeing — full confirmation. This is the setup the framework exists to catch: act per plan, ATR-sized."}
    if agree:
        return {"verdict": "weak",
                "note": f"{direction} pattern with agreeing order flow but mid-range — tradable at reduced size if the regime agrees; upgrades to confirmed at a POC/VA edge."}
    return {"verdict": "unconfirmed",
            "note": f"{direction} pattern without order-flow or level support — needs CVD/taker agreement to become actionable."}


# --- Live-flow ladder shape -------------------------------------------------
# Compares the per-second signed flow rate of a short window against a long
# one (e.g. 1m vs 15m of tape CVD). The single-window total can't distinguish
# a stale burst from building aggression; the rate ratio can.

FLOW_ACCEL_RATIO = 1.5   # short-window rate >= this × long rate → accelerating
FLOW_FADE_RATIO = 0.5    # short-window rate <= this × long rate → fading


def classify_flow_ladder(cvd_short, cvd_long, short_s: float, long_s: float) -> dict | None:
    """Shape of recent order flow from two ladder rungs. Returns
    {"state": accelerating|steady|fading|flipping|quiet, "note": ...},
    or None when either rung is unavailable."""
    if cvd_short is None or cvd_long is None or not short_s or not long_s:
        return None
    r_short = cvd_short / short_s
    r_long = cvd_long / long_s
    if r_short == 0 and r_long == 0:
        return {"state": "quiet", "note": "no net flow on either window"}
    if r_short * r_long < 0:
        return {"state": "flipping",
                "note": "the most recent flow runs against the broader window — absorption or early reversal"}
    if abs(r_short) >= FLOW_ACCEL_RATIO * abs(r_long):
        return {"state": "accelerating",
                "note": "recent flow is stronger than the broader window — aggression building now"}
    if abs(r_short) <= FLOW_FADE_RATIO * abs(r_long):
        return {"state": "fading",
                "note": "recent flow is weaker than the broader window — the push is stale/fading"}
    return {"state": "steady", "note": "flow rate consistent across windows"}


# --- Confluence (the counterweight to per-signal caveats) ------------------
# Every signal above is framed "not alone"; this is where "together" gets a
# voice. When independent reads agree, say so as plainly as a risk flag.

CONFLUENCE_ALIGNED_MIN = 3  # unopposed agreeing CLUSTERS needed for a full 'aligned' verdict


def _cluster_count(names, clusters):
    """Distinct clusters among `names`; a name without a cluster is its own."""
    if not clusters:
        return len(names)
    return len({clusters.get(k, k) for k in names})


def classify_confluence(votes: dict, clusters: dict | None = None,
                        extension=()) -> dict:
    """Aggregate named directional reads into one explicit alignment verdict.

    `votes` maps signal name → 'bullish' | 'bearish' | 'neutral' | None
    (None = unavailable, excluded).

    `clusters` maps vote name → cluster id; correlated reads (e.g. every function
    of the same recent price path) share a cluster and count ONCE toward
    agreement, so a pure trend-following sweep can't reach 'aligned' by itself —
    it needs an independent dimension (flow, positioning) to agree. Votes absent
    from the map are their own cluster. Without `clusters`, every vote counts.

    `extension` names votes that flag overextension (stretch). When the ONLY
    opposition comes from extension votes, alignment isn't vetoed — the verdict
    is suffixed '_extended': the setup is real but late; enter on a pullback,
    don't chase.

    Verdicts:
      aligned            — >= CONFLUENCE_ALIGNED_MIN clusters agree, none oppose:
                           the methodology's green light. Remaining caveats are
                           sizing inputs, not reasons to stand aside.
      aligned_extended   — same agreement, but price is stretched against the
                           direction: continuation entry on pullback, not chase.
      leaning            — clear cluster majority one way: reduced-size setup.
      leaning_extended   — leaning + stretched: wait for the pullback.
      mixed              — reads genuinely disagree: no edge (standing aside IS
                           the signal).
      no_signal          — every read neutral: nothing loaded, not caution.
    """
    cast = {k: v for k, v in votes.items() if v is not None}
    bull = sorted(k for k, v in cast.items() if v == "bullish")
    bear = sorted(k for k, v in cast.items() if v == "bearish")
    neutral = sorted(k for k, v in cast.items() if v == "neutral")

    if not bull and not bear:
        return {"direction": None, "verdict": "no_signal",
                "agreeing": [], "opposing": [], "neutral": neutral,
                "agreeing_clusters": 0,
                "note": "every read is neutral — nothing is loaded either way. "
                        "That is absence of signal, not a reason for extra caution."}

    if _cluster_count(bull, clusters) >= _cluster_count(bear, clusters):
        direction, agreeing, opposing = "long", bull, bear
    else:
        direction, agreeing, opposing = "short", bear, bull

    agree_c = _cluster_count(agreeing, clusters)
    hard_opposing = [k for k in opposing if k not in extension]
    ext_opposing = [k for k in opposing if k in extension]
    hard_oppose_c = _cluster_count(hard_opposing, clusters)

    ext_note = ""
    if ext_opposing:
        ext_note = (f" ⚠ price is stretched against the {direction} "
                    f"({', '.join(ext_opposing)}) — continuation entry on a pullback, don't chase.")

    if not hard_opposing and agree_c >= CONFLUENCE_ALIGNED_MIN:
        note = (f"{len(agreeing)} reads across {agree_c} independent dimensions agree "
                f"({direction}: {', '.join(agreeing)}) with none opposing — by this stack's own "
                f"confluence standard this IS an actionable window. Treat remaining caveats as "
                f"sizing inputs, not vetoes.")
        verdict = "aligned"
    elif not hard_opposing or agree_c >= 2 * max(hard_oppose_c, 1):
        note = (f"majority of reads lean {direction} ({agree_c} vs {hard_oppose_c} dimensions"
                f"{'; opposing: ' + ', '.join(opposing) if opposing else ''}) — "
                f"a reduced-size setup if regime and levels cooperate.")
        verdict = "leaning"
    else:
        note = (f"reads genuinely disagree ({direction} {agree_c} vs {hard_oppose_c} dimensions: "
                f"{', '.join(opposing)} oppose) — no edge either way. This is the one case "
                f"where standing aside is the signal, not the default.")
        verdict = "mixed"

    if verdict in ("aligned", "leaning") and ext_opposing:
        verdict += "_extended"
        note += ext_note

    return {"direction": direction, "verdict": verdict,
            "agreeing": agreeing, "opposing": opposing, "neutral": neutral,
            "agreeing_clusters": agree_c,
            "note": note}
