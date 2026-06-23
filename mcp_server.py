"""
MCP server exposing live Binance / Bybit / Hyperliquid market data as tools.

Run directly over stdio (for Claude Desktop / Claude Code):
    python mcp_server.py
Or interactively with the MCP Inspector:
    mcp dev mcp_server.py

Each tool returns structured fields PLUS a human-readable `summary` string, and
returns {"error": "..."} on failure instead of raising, so the model can react.
"""
import math
from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP

import services
from providers import binance, bybit, hyperliquid, coinbase, coingecko, alpaca, router
from indicators import (
    analyze_orderbook, compute_indicators, calc_fibs, calc_ema,
    calc_vwap, calc_atr, calc_volume_profile, calc_adx,
    classify_regime, classify_oi_price, classify_long_short, position_size,
    calc_cvd, calc_taker_ratio, cvd_divergence, detect_squeeze,
    annualize_funding, infer_funding_interval_hours, percentile_rank,
    classify_funding, classify_basis,
    pct_returns, correlation, beta, classify_correlation, classify_rotation,
    CORR_HIGH,
    detect_candle_patterns, classify_pattern_confirmation,
)
from formatting import fmt_orderbook, fmt_indicators, fmt_fibs, fmt_futures_context, fmt_market_breadth

mcp = FastMCP("binance-data")

ORDERBOOK_FETCHERS = {
    "binance":     binance.fetch_orderbook,
    "bybit":       bybit.fetch_orderbook,
    "hyperliquid": hyperliquid.fetch_orderbook,
}

# Crypto-derivatives signals have no equity equivalent — tools reject equity symbols cleanly.
def _equity_na(symbol: str, signal: str) -> dict | None:
    if router.resolve_asset_class(symbol) == "equity":
        return {"error": f"N/A for equities ({symbol}) — {signal} is a crypto-derivatives signal"}
    return None


def _equity_volume_caveat(symbol: str, asset_class) -> str:
    """Note appended to volume-bearing summaries when equity data is on the partial IEX feed."""
    if router.resolve_asset_class(symbol, asset_class) == "equity" and alpaca.feed() != "sip":
        return "\n  ⚠ volume low-confidence (Alpaca IEX feed is partial volume; SIP gives full tape)"
    return ""

MAX_LIMIT = 1000


def _clamp_limit(limit: int, lo: int = 1, hi: int = MAX_LIMIT) -> int:
    """Coerce a user-supplied candle count into [lo, hi]. Non-ints fall back to lo."""
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return lo
    return max(lo, min(hi, limit))


@mcp.tool()
def get_klines(symbol: str, interval: str = "1h", limit: int = 50, asset_class: str | None = None) -> dict:
    """Fetch recent OHLCV candles for a crypto pair or a stock/ETF.

    symbol:   crypto pair like 'BTCUSDT' (quote in USDT) OR a stock/ETF ticker like
              'AAPL', 'SPY', 'GLD' (via Alpaca — needs APCA_API_KEY_ID/SECRET env vars).
    interval: one of 1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,1M (Alpaca lacks 3d).
    limit:    number of candles (1-1000).
    asset_class: 'crypto' | 'equity' | None (auto: USDT-suffix → crypto, else equity).

    Returns `columns` (the row field order) and `candles` as compact
    [time, open, high, low, close, volume] rows, the percent change over the
    window, and a one-line summary. Prices are full precision; volume is rounded.
    """
    symbol = symbol.upper()
    limit = _clamp_limit(limit)
    try:
        raw = router.fetch_ohlcv(symbol, interval, limit, asset_class)
    except Exception as e:
        return {"error": f"failed to fetch klines for {symbol} {interval}: {e}"}

    if not raw:
        return {"error": f"no candles returned for {symbol} {interval}"}

    candles = [
        [
            datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
            float(k[1]), float(k[2]), float(k[3]), float(k[4]), round(float(k[5]), 2),
        ]
        for k in raw
    ]
    first_close = float(raw[0][4])
    last_close = float(raw[-1][4])
    change_pct = (last_close - first_close) / first_close * 100 if first_close else 0.0

    return {
        "symbol": symbol,
        "interval": interval,
        "change_pct": round(change_pct, 2),
        "columns": "time,o,h,l,c,v",
        "candles": candles,
        "summary": f"{symbol} {interval}: {len(raw)} candles, change over window {change_pct:+.2f}%, latest close {last_close}",
    }


@mcp.tool()
def get_orderbook(symbol: str, exchange: str = "binance") -> dict:
    """Fetch the top-20 order book and its bid/ask imbalance.

    symbol:   pair like 'BTCUSDT' (Bybit/Binance) — quote in USDT.
    exchange: 'binance', 'bybit', or 'hyperliquid'.

    Returns best bid/ask, spread, and per-depth (5/10/20 level) bid vs ask volume,
    ratio, and a buy/sell/neutral pressure label, plus a text summary.
    """
    symbol = symbol.upper()
    na = _equity_na(symbol, "order book depth")
    if na:
        return na
    exchange = exchange.lower()
    fetcher = ORDERBOOK_FETCHERS.get(exchange)
    if fetcher is None:
        return {"error": f"unknown exchange '{exchange}'; choose from {sorted(ORDERBOOK_FETCHERS)}"}

    try:
        depth = fetcher(symbol)
    except Exception as e:
        return {"error": f"failed to fetch {exchange} order book for {symbol}: {e}"}

    analysis = analyze_orderbook(depth)
    levels = [
        {
            "depth": lvl["depth"],
            "bid_vol": round(lvl["bid_vol"], 4),
            "ask_vol": round(lvl["ask_vol"], 4),
            "ratio": None if math.isinf(lvl["ratio"]) else round(lvl["ratio"], 4),
            "pressure": lvl["pressure"],
        }
        for lvl in analysis["levels"]
    ]

    return {
        "symbol": symbol,
        "exchange": exchange,
        "best_bid": analysis["best_bid"],
        "best_ask": analysis["best_ask"],
        "spread": round(analysis["spread"], 8),
        "spread_pct": round(analysis["spread_pct"], 6),
        "levels": levels,
        "summary": "\n".join(fmt_orderbook(analysis, label=exchange)),
    }


@mcp.tool()
def get_futures_context(symbol: str) -> dict:
    """Fetch Binance USD-M perpetual futures context for a symbol.

    symbol: perpetual like 'BTCUSDT', 'ETHUSDT'.

    Returns funding rate, next funding time, mark/index price, open interest and its
    5h trend, and the global long/short account ratio, plus a text summary. Fields
    are null when the symbol has no perp or an endpoint is unavailable.
    """
    symbol = symbol.upper()
    na = _equity_na(symbol, "perp funding/OI/basis context")
    if na:
        return na
    # Fetch all futures endpoints exactly once, then format from the same data.
    ctx = services.compute_futures_context(symbol)
    summary = "\n".join(fmt_futures_context(ctx))

    next_funding = ctx["next_funding"]
    if next_funding is not None:
        next_funding = datetime.fromtimestamp(next_funding / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def r(x, ndigits):
        return None if x is None else round(x, ndigits)

    # OI alone has no direction — classify the OI+price quadrant, and demote the
    # raw L/S ratio to an extreme-only contrarian flag (Plan.md Stage 1 #2, #4).
    quad = classify_oi_price(ctx["oi_change_pct_5h"], ctx["price_change_pct_5h"])
    ls = classify_long_short(ctx["long_short_ratio"])

    return {
        "symbol": symbol,
        "funding_rate_pct": r(ctx["funding_rate_pct"], 6),
        "funding_apr": r(ctx["funding_apr"], 2),
        "funding_percentile": r(ctx["funding_percentile"], 1),
        "funding_reading": ctx["funding_reading"],
        "next_funding": next_funding,
        "mark_price": ctx["mark_price"],
        "index_price": ctx["index_price"],
        "basis_pct": r(ctx["basis_pct"], 6),
        "basis_state": ctx["basis_state"],
        "open_interest": ctx["open_interest"],
        "oi_change_pct_5h": r(ctx["oi_change_pct_5h"], 4),
        "price_change_pct_5h": r(ctx["price_change_pct_5h"], 4),
        "oi_quadrant": quad["quadrant"],
        "oi_quadrant_label": quad["label"],
        "long_short_ratio": r(ctx["long_short_ratio"], 4),
        "long_pct": r(ctx["long_pct"], 2),
        "short_pct": r(ctx["short_pct"], 2),
        "long_short_reading": ls["reading"],
        "long_short_contrarian": ls["contrarian"],
        "summary": summary,
    }


@mcp.tool()
def get_funding(symbol: str) -> dict:
    """Cross-exchange funding (Binance / Bybit / Hyperliquid), normalized to APR.

    Funding is a positioning/sentiment read, NOT a timing trigger — it has ~zero
    single-asset predictive power and can stay extreme for weeks in a strong trend.
    Use APR extremes as a contrarian flag and cross-exchange divergence as a
    positioning / arbitrage signal.

    symbol: perpetual like 'BTCUSDT'.

    Returns each venue's current funding as annualized APR (Binance interval inferred
    from its history, Bybit assumed 8h, Hyperliquid 1h), an extreme/percentile read for
    Binance vs its own history, and a cross-venue divergence (max−min APR). Per-venue
    null when the symbol isn't listed there.
    """
    symbol = symbol.upper()
    na = _equity_na(symbol, "perp funding")
    if na:
        return na

    bnb = None
    try:
        pm = binance.fetch_premium_index(symbol)
        rate = float(pm["lastFundingRate"])
        hist = binance.fetch_funding_history(symbol, limit=100)
        interval_h = infer_funding_interval_hours(hist)
        apr = annualize_funding(rate, interval_h)
        apr_hist = [annualize_funding(float(h["fundingRate"]), interval_h) for h in hist]
        pct = percentile_rank(apr, apr_hist)
        cls = classify_funding(apr)
        bnb = {
            "rate_pct": round(rate * 100, 6), "interval_hours": interval_h,
            "apr": round(apr, 2), "percentile": None if pct is None else round(pct, 1),
            "reading": cls["reading"], "contrarian": cls["contrarian"], "note": cls["note"],
        }
    except Exception:
        pass

    def simple_venue(rate, interval_h):
        apr = annualize_funding(rate, interval_h)
        return {"rate_pct": round(rate * 100, 6), "interval_hours": interval_h, "apr": round(apr, 2)}

    byb = None
    try:
        byb = simple_venue(float(bybit.fetch_ticker(symbol)["fundingRate"]), 8.0)
    except Exception:
        pass

    hyper = None
    try:
        hyper = simple_venue(float(hyperliquid.fetch_ctx(symbol)["funding"]), 1.0)
    except Exception:
        pass

    if bnb is None and byb is None and hyper is None:
        return {"error": f"no funding data for {symbol} on any venue"}

    aprs = {name: v["apr"] for name, v in (("Binance", bnb), ("Bybit", byb), ("Hyperliquid", hyper)) if v}
    spread = round(max(aprs.values()) - min(aprs.values()), 2) if len(aprs) >= 2 else None

    def vline(name, v):
        if v is None:
            return f"  {name}: n/a (not listed)"
        extra = ""
        if "percentile" in v:
            pc = f", {v['percentile']:.0f}th pct" if v["percentile"] is not None else ""
            extra = f"  [{v['reading']}{pc}]"
        return f"  {name}: {v['apr']:+.1f}% APR ({v['rate_pct']:+.4f}%/{v['interval_hours']:g}h){extra}"

    summary_lines = [f"{symbol} cross-exchange funding (annualized):",
                     vline("Binance", bnb), vline("Bybit", byb), vline("Hyperliquid", hyper)]
    if spread is not None:
        flag = " — wide divergence (positioning/arb)" if spread >= 20 else " — aligned"
        summary_lines.append(f"  Spread: {spread:.1f}% APR{flag}")
    if bnb and bnb["note"]:
        summary_lines.append(f"  → {bnb['note']}")

    return {
        "symbol": symbol,
        "binance": bnb,
        "bybit": byb,
        "hyperliquid": hyper,
        "apr_spread": spread,
        "summary": "\n".join(summary_lines),
    }


@mcp.tool()
def get_indicators(symbol: str, interval: str = "1h", limit: int = 60, asset_class: str | None = None) -> dict:
    """Compute technical indicators for a crypto pair or stock/ETF over a timeframe.

    symbol:   crypto pair like 'BTCUSDT', or a stock/ETF ticker like 'AAPL', 'SPY'.
    interval: candle interval (e.g. 15m, 1h, 4h, 1d). limit: candles to analyze.
    asset_class: 'crypto' | 'equity' | None (auto).

    Returns OBV (+trend), CVD (+trend & price divergence), taker buy/sell ratio,
    volume ratio vs 20-bar average, ADX(14) with +DI/-DI (regime-gated), a TTM
    squeeze flag with Bollinger band width, candlestick patterns, and Fibonacci
    retracement levels, plus a text summary. CVD/taker are null for equities (bars
    carry no taker side); for spot-vs-perp CVD divergence use get_cvd (crypto only).
    Volume-based fields are low-confidence for equities on Alpaca's free IEX feed.
    """
    symbol = symbol.upper()
    limit = _clamp_limit(limit)
    try:
        candles = router.fetch_ohlcv(symbol, interval, limit, asset_class)
    except Exception as e:
        return {"error": f"failed to fetch klines for {symbol} {interval}: {e}"}

    ind  = compute_indicators(candles)
    fibs = calc_fibs(candles)

    summary_lines = fmt_indicators(ind)
    summary_lines.append("Fibonacci retracements (swing high → low):")
    summary_lines += fmt_fibs(fibs)
    summary = "\n".join(summary_lines) + _equity_volume_caveat(symbol, asset_class)

    def r(x, ndigits):
        return None if x is None else round(x, ndigits)

    return {
        "symbol": symbol,
        "interval": interval,
        "obv": round(ind["obv"], 2),
        "obv_trend": ind["obv_trend"],
        "cvd": r(ind["cvd"], 2),
        "cvd_trend": ind["cvd_trend"],
        "cvd_divergence": ind["cvd_divergence"],
        "taker_ratio": r(ind["taker_ratio"], 4),
        "volume_ratio": round(ind["volume_ratio"], 4),
        "adx": round(ind["adx"], 2),
        "plus_di": round(ind["plus_di"], 2),
        "minus_di": round(ind["minus_di"], 2),
        "squeeze_on": ind["squeeze_on"],
        "bbw": r(ind["bbw"], 6),
        "bbw_state": ind["bbw_state"],
        "patterns": ind["patterns"],
        "fibs": fibs,
        "summary": summary,
    }


@mcp.tool()
def get_cvd(symbol: str, interval: str = "15m", limit: int = 96) -> dict:
    """Cumulative Volume Delta on perp + spot, with spot-vs-perp divergence.

    CVD is the running net of taker-buy minus taker-sell volume (aggressor intent),
    built from kline taker-buy volume — impossible to spoof, unlike order-book depth.
    The absolute value is meaningless (start-point dependent): read the trend, the
    CVD-vs-price divergence, and especially the SPOT-vs-PERP split.

    symbol:   pair like 'BTCUSDT' — quote in USDT.
    interval: candle interval (default 15m; keep >=1m — sub-minute CVD is HFT/wash noise).
    limit:    candles to analyze (default 96 = 24h of 15m bars).

    Returns per-market (perp & spot) CVD, trend, taker buy/sell ratio, price change,
    and a coarse CVD-vs-price divergence flag, plus a spot-vs-perp read: a rally on
    strong perp CVD but weak spot CVD is leverage-led and fragile; spot-led CVD is
    higher-conviction accumulation. Degrades to a single market when one is unlisted.
    """
    symbol = symbol.upper()
    na = _equity_na(symbol, "CVD / taker order flow")
    if na:
        return na
    limit = _clamp_limit(limit)

    def market_cvd(fetcher):
        try:
            candles = fetcher(symbol, interval, limit)
        except Exception:
            return None
        if not candles:
            return None
        cvd, trend = calc_cvd(candles)
        if cvd is None:
            return None
        div = cvd_divergence(candles)
        taker = calc_taker_ratio(candles)
        first_close, last_close = float(candles[0][4]), float(candles[-1][4])
        price_chg = (last_close - first_close) / first_close * 100 if first_close else 0.0
        return {
            "cvd": round(cvd, 2),
            "cvd_trend": trend,
            "divergence": div["divergence"],
            "price_trend": div["price_trend"],
            "taker_ratio": None if taker is None else round(taker, 4),
            "price_change_pct": round(price_chg, 2),
        }

    perp = market_cvd(binance.fetch_klines_futures)
    spot = market_cvd(binance.fetch_klines_spot)
    if perp is None and spot is None:
        return {"error": f"no spot or perp kline/taker data for {symbol} {interval}"}

    # Spot-vs-perp conviction read (the highest-value CVD application for a perps trader).
    spot_perp = None
    if perp and spot:
        pr, sp = perp["cvd_trend"], spot["cvd_trend"]
        if pr == "rising" and sp == "rising":
            spot_perp = "broad-based — spot AND perp CVD rising (durable)"
        elif pr == "rising" and sp != "rising":
            spot_perp = "perp-led / leverage-driven — weak spot CVD (fragile, prone to reversal)"
        elif sp == "rising" and pr != "rising":
            spot_perp = "spot-led accumulation — higher conviction (often institutional)"
        elif pr == "falling" and sp == "falling":
            spot_perp = "broad-based selling — spot AND perp CVD falling"
        else:
            spot_perp = f"mixed (perp {pr} / spot {sp})"

    def line(label, m):
        if m is None:
            return f"  {label}: n/a (not listed / no taker data)"
        div_note = "" if m["divergence"] == "none" else f", ⚠ {m['divergence']} divergence"
        taker = f"{m['taker_ratio']:.2f}x" if m["taker_ratio"] is not None else "n/a"
        return (f"  {label}: CVD {m['cvd']:+.0f} ({m['cvd_trend']}{div_note}) "
                f"| taker {taker} | price {m['price_change_pct']:+.2f}%")

    summary_lines = [f"{symbol} {interval} order flow ({limit} bars):", line("Perp", perp), line("Spot", spot)]
    if spot_perp:
        summary_lines.append(f"  → spot-vs-perp: {spot_perp}")

    return {
        "symbol": symbol,
        "interval": interval,
        "perp": perp,
        "spot": spot,
        "spot_vs_perp": spot_perp,
        "summary": "\n".join(summary_lines),
    }


@mcp.tool()
def get_patterns(symbol: str, interval: str = "1h", limit: int = 192, asset_class: str | None = None) -> dict:
    """Candlestick patterns on the latest bar — CONFIRMATION-ONLY, never standalone signals.

    Candle patterns are low-edge alone (like Fibonacci). This tool detects them AND scores
    whether they're confirmed by order flow (CVD slope + taker ratio) and location (at a
    volume-profile node / value-area edge). A pattern with agreeing flow at a level is a valid
    confirmation; the same pattern mid-range with conflicting flow is a fakeout risk.

    symbol:   pair like 'BTCUSDT' — quote in USDT.
    interval: candle interval (default 1h). limit: candles for context (default 192).

    Detects: hammer, shooting_star, doji, marubozu, bullish/bearish_engulfing, inside_bar.
    Returns each with a verdict (confirmed / weak / mixed / conflicting / unconfirmed / neutral)
    plus the CVD/taker/level context. The latest candle may be in progress — patterns confirm on
    close. Pair with get_regime (don't take counter-trend patterns in a strong trend).
    """
    symbol = symbol.upper()
    limit = _clamp_limit(limit)
    try:
        candles = router.fetch_ohlcv(symbol, interval, limit, asset_class)
    except Exception as e:
        return {"error": f"failed to fetch klines for {symbol} {interval}: {e}"}
    if not candles:
        return {"error": "no candles returned"}

    patterns = detect_candle_patterns(candles)
    _, cvd_trend = calc_cvd(candles)
    div = cvd_divergence(candles)
    taker = calc_taker_ratio(candles)
    close = float(candles[-1][4])

    # at_level: latest close near a volume-profile node / value-area edge (within 0.3%).
    at_level = False
    level_note = "no profile"
    vp = calc_volume_profile(candles)
    if vp is not None:
        level_note = f"not at POC/VA (close {close:.6g})"
        for name, lvl in (("POC", vp["poc"]), ("VAH", vp["vah"]), ("VAL", vp["val"])):
            if lvl and abs(close - lvl) / lvl <= 0.003:
                at_level, level_note = True, f"at {name} {lvl:.6g}"
                break

    enriched = []
    for p in patterns:
        conf = classify_pattern_confirmation(p["direction"], cvd_trend, taker, at_level)
        enriched.append({**p, "verdict": conf["verdict"], "note": conf["note"]})

    taker_str = f"{taker:.2f}x" if taker is not None else "n/a"
    if enriched:
        lines = [f"{symbol} {interval} patterns (latest bar; confirms on close):"]
        lines += [f"  {p['pattern']} ({p['direction']}) → {p['verdict']}: {p['note']}" for p in enriched]
        lines.append(f"  context: CVD {cvd_trend}, taker {taker_str}, {level_note}")
    else:
        lines = [f"{symbol} {interval}: no candlestick pattern on the latest bar."]

    return {
        "symbol": symbol,
        "interval": interval,
        "patterns": enriched,
        "cvd_trend": cvd_trend,
        "cvd_divergence": div["divergence"],
        "taker_ratio": None if taker is None else round(taker, 4),
        "at_level": at_level,
        "current_close": close,
        "summary": "\n".join(lines),
    }


@mcp.tool()
def get_emas(symbol: str, interval: str = "1h", limit: int = 500, asset_class: str | None = None) -> dict:
    """Compute 20/50/200 EMAs and the trend stack for a pair.

    symbol:   pair like 'BTCUSDT' — quote in USDT.
    interval: candle interval (e.g. 15m, 1h, 4h, 1d).
    limit:    candles to fetch. Default 500 gives the 200-EMA ~300 bars of
              warmup after the SMA seed (well-converged). Reduce to save API
              time; minimum 200 to compute the 200-EMA.

    Returns ema_20, ema_50, ema_200, current_close, and a 'stack' label
    (bullish | bearish | mixed) plus a one-line summary.
    """
    symbol = symbol.upper()
    limit = _clamp_limit(limit)
    try:
        candles = router.fetch_ohlcv(symbol, interval, limit, asset_class)
    except Exception as e:
        return {"error": f"failed to fetch klines for {symbol} {interval}: {e}"}

    closes = [float(k[4]) for k in candles]
    current_close = closes[-1]
    ema_20  = calc_ema(closes, 20)
    ema_50  = calc_ema(closes, 50)
    ema_200 = calc_ema(closes, 200)

    if None in (ema_20, ema_50, ema_200):
        stack = "n/a"
        stack_text = "insufficient candles for full 20/50/200 stack"
    elif current_close > ema_20 > ema_50 > ema_200:
        stack = "bullish"
        stack_text = "price > 20 > 50 > 200 (bullish stack)"
    elif current_close < ema_20 < ema_50 < ema_200:
        stack = "bearish"
        stack_text = "price < 20 < 50 < 200 (bearish stack)"
    else:
        stack = "mixed"
        stack_text = "EMAs not in a clean trend stack"

    def s(x):
        return "n/a" if x is None else f"{x:.6g}"

    return {
        "symbol": symbol,
        "interval": interval,
        "current_close": current_close,
        "ema_20": ema_20,
        "ema_50": ema_50,
        "ema_200": ema_200,
        "stack": stack,
        "summary": f"{symbol} {interval} EMAs: 20={s(ema_20)} | 50={s(ema_50)} | 200={s(ema_200)} | close={s(current_close)} → {stack_text}",
    }


@mcp.tool()
def get_regime(symbol: str, interval: str = "4h", limit: int = 300, asset_class: str | None = None) -> dict:
    """Classify the trend regime — the meta-filter that should gate every other signal.

    Combines ADX(14) strength with price position vs the 200-EMA to decide which
    playbook applies, so you don't run mean-reversion in a strong trend (or
    trend-following in chop) — the single highest-leverage rule in the stack.

    symbol:   pair like 'BTCUSDT' — quote in USDT.
    interval: candle interval (default 4h; 1h/4h are the documented templates).
    limit:    candles to fetch. Default 300 seeds the 200-EMA with ~100 bars of
              warmup; minimum ~200 to get a 200-EMA at all (else regime degrades
              to ADX+DI only).

    Returns:
      regime  — trend_up | trend_down | range | transitional
      mode    — trend-following | mean-reversion | stand-aside
      adx_state (trending/developing/ranging), di_direction (only in a trend),
      above_200ema, atr(14) for stop sizing, and a one-line playbook + summary.
    """
    symbol = symbol.upper()
    limit = _clamp_limit(limit)
    try:
        candles = router.fetch_ohlcv(symbol, interval, limit, asset_class)
    except Exception as e:
        return {"error": f"failed to fetch klines for {symbol} {interval}: {e}"}

    if not candles:
        return {"error": "no candles returned"}

    closes = [float(k[4]) for k in candles]
    close = closes[-1]
    adx, pdi, ndi = calc_adx(candles)
    ema_200 = calc_ema(closes, 200)
    atr = calc_atr(candles)

    reg = classify_regime(adx, pdi, ndi, close, ema_200)

    ema_note = "" if ema_200 is not None else " (200-EMA n/a — too few candles; ADX+DI only)"
    summary = (
        f"{symbol} {interval} regime: {reg['regime'].upper()} → {reg['mode']} | "
        f"ADX {adx:.1f} ({reg['adx_state']}), +DI {pdi:.1f}/-DI {ndi:.1f}"
        f"{f', close {close:.6g} vs 200-EMA {ema_200:.6g}' if ema_200 is not None else ''}"
        f"{ema_note}\n  {reg['playbook']}"
    )

    return {
        "symbol": symbol,
        "interval": interval,
        "regime": reg["regime"],
        "mode": reg["mode"],
        "adx": round(adx, 2),
        "adx_state": reg["adx_state"],
        "plus_di": round(pdi, 2),
        "minus_di": round(ndi, 2),
        "di_direction": reg["di_direction"],
        "current_close": close,
        "ema_200": ema_200,
        "above_200ema": reg["above_200ema"],
        "atr": atr,
        "playbook": reg["playbook"],
        "summary": summary,
    }


@mcp.tool()
def get_correlation(symbol: str, interval: str = "1h", limit: int = 200,
                    reference: str | None = None, asset_class: str | None = None) -> dict:
    """Rolling correlation + beta of a symbol vs a reference — tells you *when*
    symbol-specific analysis is even worth doing.

    High correlation (>=0.8) means the symbol is just leveraged beta of the reference:
    trade the reference's regime, not the specifics. A genuine drop in correlation =
    decoupling on idiosyncratic flow, and only then do symbol-specific setups carry edge.

    symbol:   pair like 'SOLUSDT' or a stock/ETF like 'AAPL'.
    interval: candle interval (default 1h). limit: candles to correlate (default 200).
    reference: benchmark to correlate against. Defaults to BTCUSDT for crypto, SPY for equities.
    asset_class: 'crypto' | 'equity' | None (auto) — applied to both legs.

    Returns Pearson correlation + beta over the window, a recent-half correlation and a
    `decoupling` flag (recent ≪ full), and a gating read. corr=beta=1.0 vs itself.
    """
    symbol = symbol.upper()
    cls_asset = router.resolve_asset_class(symbol, asset_class)
    if reference is None:
        reference = "BTCUSDT" if cls_asset == "crypto" else "SPY"
    reference = reference.upper()
    limit = _clamp_limit(limit)
    if symbol == reference:
        return {"symbol": symbol, "interval": interval, "reference": reference, "correlation": 1.0,
                "beta": 1.0, "recent_correlation": 1.0, "decoupling": False, "level": "self",
                "summary": f"{symbol} is the reference — correlation 1.0 by definition."}

    try:
        alt = router.fetch_ohlcv(symbol, interval, limit, asset_class)
        ref = router.fetch_ohlcv(reference, interval, limit, asset_class)
    except Exception as e:
        return {"error": f"failed to fetch klines: {e}"}

    n = min(len(alt), len(ref))
    if n < 3:
        return {"error": f"not enough overlapping candles to correlate (got {n})"}
    a_ret = pct_returns([float(k[4]) for k in alt][-n:])
    b_ret = pct_returns([float(k[4]) for k in ref][-n:])

    r = correlation(a_ret, b_ret)
    bta = beta(a_ret, b_ret)
    half = max(2, len(a_ret) // 2)
    r_recent = correlation(a_ret[-half:], b_ret[-half:])
    decoupling = bool(r is not None and r_recent is not None and r_recent < r - 0.2 and r_recent < CORR_HIGH)
    cls = classify_correlation(r)

    def rnd(x, d):
        return None if x is None else round(x, d)

    rstr = "n/a" if r is None else f"{r:.2f}"
    bstr = "n/a" if bta is None else f"{bta:.2f}"
    summary = (f"{symbol} vs {reference} ({interval}, {n} bars): corr {rstr} [{cls['level']}], beta {bstr}"
               f"{' | ⚠ decoupling (recent corr ' + format(r_recent, '.2f') + ')' if decoupling else ''}"
               f"\n  → {cls['note']}")

    return {
        "symbol": symbol,
        "interval": interval,
        "reference": reference,
        "correlation": rnd(r, 4),
        "recent_correlation": rnd(r_recent, 4),
        "beta": rnd(bta, 4),
        "decoupling": decoupling,
        "level": cls["level"],
        "summary": summary,
    }


@mcp.tool()
def get_market_breadth() -> dict:
    """Market-wide breadth for higher-timeframe / alt bias (no symbol).

    Combines CoinGecko global metrics (total market cap + 24h change, BTC/ETH/stablecoin
    dominance) with Binance 24h moves to derive BTC dominance *direction* and the ETH/BTC
    alt-bellwether, then reads the capital-rotation backdrop.

    Returns total market cap, TOTAL2 (ex-BTC), dominance %s, BTC.D direction, ETH/BTC 24h,
    and a rotation read (btc_dominant / alt_rotation / risk_off / neutral).

    Caveat: raw BTC.D includes ~$300B+ of stablecoins — a falling BTC.D in a sell-off can be
    'stablecoin season', not altseason. Read total-cap direction + stablecoin dominance together.
    """
    m = {
        "total_mcap_usd": None, "total_cap_change_24h_pct": None, "total2_usd": None,
        "btc_dominance": None, "eth_dominance": None, "stablecoin_dominance": None,
        "btc_24h_pct": None, "btc_dom_rising": None, "ethbtc_24h_pct": None,
        "rotation_read": None, "rotation_note": None,
    }

    try:
        g = coingecko.fetch_global_metrics()
        m["total_mcap_usd"] = round(g["total_market_cap"]["usd"])
        m["total_cap_change_24h_pct"] = round(g["market_cap_change_percentage_24h_usd"], 2)
        mc = g["market_cap_percentage"]
        m["btc_dominance"] = round(mc["btc"], 2) if "btc" in mc else None
        m["eth_dominance"] = round(mc["eth"], 2) if "eth" in mc else None
        if "usdt" in mc or "usdc" in mc:
            m["stablecoin_dominance"] = round((mc.get("usdt") or 0) + (mc.get("usdc") or 0), 2)
        if m["btc_dominance"] is not None and m["total_mcap_usd"]:
            m["total2_usd"] = round(m["total_mcap_usd"] * (1 - m["btc_dominance"] / 100))
    except Exception:
        pass

    if m["total_mcap_usd"] is None:
        return {"error": "failed to fetch global market metrics"}

    try:
        m["btc_24h_pct"] = round(float(binance.fetch_24h("BTCUSDT")["priceChangePercent"]), 2)
        if m["total_cap_change_24h_pct"] is not None:
            m["btc_dom_rising"] = m["btc_24h_pct"] > m["total_cap_change_24h_pct"]
    except Exception:
        pass

    try:
        m["ethbtc_24h_pct"] = round(float(binance.fetch_24h("ETHBTC")["priceChangePercent"]), 2)
    except Exception:
        pass

    if m["btc_dom_rising"] is not None:
        rot = classify_rotation(m["btc_dom_rising"], m["total_cap_change_24h_pct"])
        m["rotation_read"] = rot["read"]
        m["rotation_note"] = rot["note"]

    m["summary"] = "\n".join(fmt_market_breadth(m))
    return m


THIN_VOLUME_USD = 1_000_000  # aggregate 24h USD below this → thin/illiquid alt


def _fmt_usd(v):
    if v is None:
        return "n/a"
    if v >= 1e9: return f"${v / 1e9:.2f}B"
    if v >= 1e6: return f"${v / 1e6:.2f}M"
    if v >= 1e3: return f"${v / 1e3:.2f}K"
    return f"${v:.2f}"


@mcp.tool()
def get_volume_breakdown(symbol: str) -> dict:
    """Compare 24h volume across Binance, Coinbase, and the cross-exchange aggregate.

    symbol: USDT-quoted pair like 'BTCUSDT'. The base asset (BTC) is used for the
            Coinbase BTC-USD product and the CoinGecko cross-exchange aggregate.

    Returns USD-denominated 24h volumes — Binance spot, Coinbase spot, the CoinGecko
    cross-exchange spot aggregate, and explicit cross-venue PERP volume (Binance /
    Bybit / Hyperliquid) — plus each spot venue's share of the aggregate, a `thin`
    flag when aggregate volume is below $1M, and a spot-vs-perp read (Binance spot vs
    Binance perp). Each field is null if its source is unavailable (e.g. a venue that
    doesn't list the symbol). Note: spot shares are vs the spot aggregate; the perp
    figures are a separate cross-venue comparison (not mixed into the spot shares).

    Use this for:
      - Coinbase share > usual vs Binance → US / institutional skew on majors.
      - Binance + Coinbase << aggregate → flow concentrated on other venues.
      - perp_spot_ratio >> 1 → leverage-led (more fragile); ~1 or <1 → spot-led (more durable).
      - Bybit/Hyperliquid perp vs Binance perp → where leveraged flow concentrates.
      - `thin=true` → treat orderbook/funding signals with caution.
    """
    symbol = symbol.upper()
    na = _equity_na(symbol, "cross-venue crypto volume breakdown")
    if na:
        return na
    if symbol.endswith("USDT"):
        base = symbol[:-4]
    elif symbol.endswith("USD"):
        base = symbol[:-3]
    else:
        base = symbol

    result = {
        "symbol": symbol,
        "base_asset": base,
        "binance_volume_usd": None,
        "binance_perp_volume_usd": None,
        "bybit_perp_volume_usd": None,
        "hyperliquid_perp_volume_usd": None,
        "coinbase_volume_usd": None,
        "aggregate_volume_usd": None,
        "binance_share_pct": None,
        "coinbase_share_pct": None,
        "other_share_pct": None,
        "perp_spot_ratio": None,
        "leverage_read": None,
        "thin": None,
    }

    try:
        bnb = binance.fetch_24h(symbol)
        result["binance_volume_usd"] = round(float(bnb["quoteVolume"]), 2)
    except Exception:
        pass

    try:
        fut = binance.fetch_24h_futures(symbol)
        result["binance_perp_volume_usd"] = round(float(fut["quoteVolume"]), 2)
    except Exception:
        pass

    try:
        result["bybit_perp_volume_usd"] = round(float(bybit.fetch_ticker(symbol)["turnover24h"]), 2)
    except Exception:
        pass

    try:
        result["hyperliquid_perp_volume_usd"] = round(float(hyperliquid.fetch_ctx(symbol)["dayNtlVlm"]), 2)
    except Exception:
        pass

    try:
        cb = coinbase.fetch_24h(base)
        cb_vol = float(cb["volume"])
        cb_last = float(cb["last"]) if cb.get("last") else None
        if cb_last is not None:
            result["coinbase_volume_usd"] = round(cb_vol * cb_last, 2)
    except Exception:
        pass

    try:
        result["aggregate_volume_usd"] = round(coingecko.fetch_aggregate_volume(base, quote="USD"), 2)
    except Exception:
        pass

    agg_usd = result["aggregate_volume_usd"]
    if agg_usd and agg_usd > 0:
        if result["binance_volume_usd"] is not None:
            result["binance_share_pct"] = round(result["binance_volume_usd"] / agg_usd * 100, 2)
        if result["coinbase_volume_usd"] is not None:
            result["coinbase_share_pct"] = round(result["coinbase_volume_usd"] / agg_usd * 100, 2)
        b = result["binance_share_pct"] or 0
        c = result["coinbase_share_pct"] or 0
        result["other_share_pct"] = round(max(0.0, 100 - b - c), 2)
        result["thin"] = agg_usd < THIN_VOLUME_USD

    # Spot-vs-perp (same venue) — leverage-led vs spot-led participation.
    spot_v, perp_v = result["binance_volume_usd"], result["binance_perp_volume_usd"]
    if spot_v and perp_v and spot_v > 0:
        ratio = perp_v / spot_v
        result["perp_spot_ratio"] = round(ratio, 2)
        result["leverage_read"] = "perp/leverage-led (more fragile)" if ratio > 1.5 else \
                                  "balanced" if ratio >= 0.8 else "spot-led (more durable)"

    parts = [f"{base} 24h vol:", f"agg={_fmt_usd(agg_usd)}"]
    if result["binance_volume_usd"] is not None:
        share = f" ({result['binance_share_pct']}%)" if result["binance_share_pct"] is not None else ""
        parts.append(f"Binance spot={_fmt_usd(result['binance_volume_usd'])}{share}")
    if result["coinbase_volume_usd"] is not None:
        share = f" ({result['coinbase_share_pct']}%)" if result["coinbase_share_pct"] is not None else ""
        parts.append(f"Coinbase={_fmt_usd(result['coinbase_volume_usd'])}{share}")
    if result["other_share_pct"] is not None:
        parts.append(f"other={result['other_share_pct']}%")
    # Cross-venue perp volume (Binance/Bybit/Hyperliquid) — separate from the spot aggregate.
    perp_bits = [
        (name, vol) for name, vol in (
            ("Binance", result["binance_perp_volume_usd"]),
            ("Bybit", result["bybit_perp_volume_usd"]),
            ("HL", result["hyperliquid_perp_volume_usd"]),
        ) if vol is not None
    ]
    if perp_bits:
        parts.append("perp: " + " ".join(f"{name}={_fmt_usd(vol)}" for name, vol in perp_bits))
    summary = " | ".join(parts)
    if result["perp_spot_ratio"] is not None:
        summary += f"  | perp/spot {result['perp_spot_ratio']}x → {result['leverage_read']}"
    if result["thin"]:
        summary += "  ⚠ THIN (aggregate < $1M)"

    result["summary"] = summary
    return result


@mcp.tool()
def get_vwap(symbol: str, interval: str = "5m", limit: int = 288, asset_class: str | None = None) -> dict:
    """Session VWAP (resets at 00:00 UTC) + window VWAP, with 1σ/2σ bands.

    symbol:   pair like 'BTCUSDT'.
    interval: candle interval — 5m / 15m default for scalping; 1m for very tight.
    limit:    candles to fetch (default 288 = 24h of 5m bars, enough for a full
              UTC session plus rolling window).

    Returns session_vwap (today, UTC) and window_vwap (whole fetched range), each
    with 1σ and 2σ bands and a bias label vs current close:
      'long'    → close > session_vwap (above value, longs favoured)
      'short'   → close < session_vwap (below value, shorts favoured)
      'neutral' → within 0.1% of session_vwap
    """
    symbol = symbol.upper()
    limit = _clamp_limit(limit)
    try:
        candles = router.fetch_ohlcv(symbol, interval, limit, asset_class)
    except Exception as e:
        return {"error": f"failed to fetch klines for {symbol} {interval}: {e}"}

    if not candles:
        return {"error": "no candles returned"}

    session_start = int(datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    s_vwap, s_sigma, s_bars = calc_vwap(candles, session_start_ts=session_start)
    w_vwap, w_sigma, w_bars = calc_vwap(candles)
    current_close = float(candles[-1][4])

    def bands(vwap, sigma):
        if vwap is None:
            return None, None, None, None
        return vwap + sigma, vwap - sigma, vwap + 2 * sigma, vwap - 2 * sigma

    s_up1, s_dn1, s_up2, s_dn2 = bands(s_vwap, s_sigma)
    w_up1, w_dn1, w_up2, w_dn2 = bands(w_vwap, w_sigma)

    if s_vwap is None:
        bias = "n/a"
        dist_pct = None
    else:
        dist_pct = (current_close - s_vwap) / s_vwap * 100
        if abs(dist_pct) < 0.1:
            bias = "neutral"
        elif current_close > s_vwap:
            bias = "long"
        else:
            bias = "short"

    summary = (
        f"{symbol} {interval} | session VWAP: "
        f"{s_vwap:.6g} ({bias}, {dist_pct:+.2f}% from close {current_close:.6g})"
        if s_vwap is not None else
        f"{symbol} {interval} | session VWAP: n/a | window VWAP: {w_vwap:.6g}" if w_vwap else
        f"{symbol} {interval} | VWAP: n/a"
    )
    if s_vwap is not None:
        summary += f" | 1σ [{s_dn1:.6g}, {s_up1:.6g}] | 2σ [{s_dn2:.6g}, {s_up2:.6g}]"

    return {
        "symbol": symbol,
        "interval": interval,
        "current_close": current_close,
        "session_vwap": s_vwap,
        "session_sigma": s_sigma,
        "session_upper_1sigma": s_up1,
        "session_lower_1sigma": s_dn1,
        "session_upper_2sigma": s_up2,
        "session_lower_2sigma": s_dn2,
        "session_bars": s_bars,
        "window_vwap": w_vwap,
        "window_sigma": w_sigma,
        "window_upper_1sigma": w_up1,
        "window_lower_1sigma": w_dn1,
        "window_upper_2sigma": w_up2,
        "window_lower_2sigma": w_dn2,
        "window_bars": w_bars,
        "bias": bias,
        "distance_pct_from_session_vwap": None if dist_pct is None else round(dist_pct, 4),
        "summary": summary,
    }


@mcp.tool()
def get_atr(symbol: str, interval: str = "30m", limit: int = 100, period: int = 14,
            account_equity: float | None = None, risk_pct: float = 1.0,
            stop_atr_mult: float = 1.5, asset_class: str | None = None) -> dict:
    """ATR(period) for stop placement and ATR-normalized position sizing — NOT a
    directional signal.

    symbol:   pair like 'BTCUSDT'.
    interval: candle interval (default 30m for scalp stops).
    limit:    candles to fetch (100 is plenty for a 14-period ATR to settle).
    period:   ATR period (default 14, Wilder's smoothing).
    account_equity: if given, also return a suggested position size that risks
                    `risk_pct`% of equity across a `stop_atr_mult`×ATR stop, so
                    size scales inversely with volatility. Omit to skip sizing.
    risk_pct:       % of equity to risk per trade (default 1.0).
    stop_atr_mult:  ATR multiple for the stop distance used in sizing (default 1.5).

    Returns the raw ATR plus example stops at 1× and 1.5× ATR from current close,
    for both long and short, and (when account_equity is set) the risk amount,
    stop distance, position size (qty) and notional.
    """
    symbol = symbol.upper()
    limit = _clamp_limit(limit)
    try:
        candles = router.fetch_ohlcv(symbol, interval, limit, asset_class)
    except Exception as e:
        return {"error": f"failed to fetch klines for {symbol} {interval}: {e}"}

    atr = calc_atr(candles, period=period)
    if atr is None:
        return {"error": f"need at least {period + 1} candles, got {len(candles)}"}

    current_close = float(candles[-1][4])
    atr_pct = atr / current_close * 100 if current_close else 0.0

    summary = (
        f"{symbol} {interval} | ATR({period}): {atr:.6g} ({atr_pct:.2f}% of close {current_close:.6g}) "
        f"| 1.5× stop: long below {current_close - 1.5 * atr:.6g}, short above {current_close + 1.5 * atr:.6g}"
    )

    result = {
        "symbol": symbol,
        "interval": interval,
        "period": period,
        "atr": atr,
        "atr_pct": round(atr_pct, 4),
        "current_close": current_close,
        "stop_long_1x_atr":   current_close - atr,
        "stop_long_1_5x_atr": current_close - 1.5 * atr,
        "stop_short_1x_atr":  current_close + atr,
        "stop_short_1_5x_atr": current_close + 1.5 * atr,
        "position_size": None,
        "risk_amount": None,
        "stop_distance": None,
        "notional": None,
    }

    if account_equity is not None:
        ps = position_size(account_equity, risk_pct, current_close, atr, stop_atr_mult)
        if ps is not None:
            result["position_size"] = ps["qty"]
            result["risk_amount"] = ps["risk_amount"]
            result["stop_distance"] = ps["stop_distance"]
            result["notional"] = ps["notional"]
            summary += (
                f" | size: risk {risk_pct:g}% of {account_equity:g} = {ps['risk_amount']:.6g} "
                f"over {stop_atr_mult:g}×ATR ({ps['stop_distance']:.6g}) → "
                f"{ps['qty']:.6g} units (~{ps['notional']:.6g} notional)"
            )

    result["summary"] = summary
    return result


@mcp.tool()
def get_squeeze(symbol: str, interval: str = "1h", limit: int = 100, period: int = 20, asset_class: str | None = None) -> dict:
    """Volatility-regime / breakout-timing filter: TTM-style squeeze + Bollinger width.

    A squeeze is ON when the Bollinger Bands sit inside the Keltner Channels — a
    low-volatility coil that often precedes an expansion. Bollinger Band Width (BBW)
    and its percentile vs recent history gauge compression. NON-directional: it tells
    you a move is loading, not which way — pair with get_cvd / get_regime for direction.

    symbol:   pair like 'BTCUSDT' — quote in USDT.
    interval: candle interval (default 1h). limit: candles to fetch (default 100).
    period:   Bollinger/Keltner lookback (default 20).

    Returns squeeze_on, BBW + percentile + state (compressed/normal/expanded), the
    Bollinger and Keltner band levels, and a summary. ~half of squeezes fail or give
    small moves — require a confirmation trigger before acting.
    """
    symbol = symbol.upper()
    limit = _clamp_limit(limit)
    try:
        candles = router.fetch_ohlcv(symbol, interval, limit, asset_class)
    except Exception as e:
        return {"error": f"failed to fetch klines for {symbol} {interval}: {e}"}

    sq = detect_squeeze(candles, period=period)
    if sq is None:
        return {"error": f"need at least {period + 1} candles, got {len(candles)}"}

    current_close = float(candles[-1][4])
    pctile = sq["bbw_pctile"]
    pctile_str = "n/a" if pctile is None else f"{pctile:.0f}th pct"
    summary = (
        f"{symbol} {interval} | squeeze: {'ON (coiled — expansion likely)' if sq['squeeze_on'] else 'off'} "
        f"| BBW {sq['bbw']:.4f} ({sq['state']}, {pctile_str}) "
        f"| BB [{sq['bb_lower']:.6g}, {sq['bb_upper']:.6g}] | KC [{sq['kc_lower']:.6g}, {sq['kc_upper']:.6g}]"
    )

    def r(x, ndigits):
        return None if x is None else round(x, ndigits)

    return {
        "symbol": symbol,
        "interval": interval,
        "period": period,
        "current_close": current_close,
        "squeeze_on": sq["squeeze_on"],
        "bbw": r(sq["bbw"], 6),
        "bbw_pctile": r(pctile, 2),
        "state": sq["state"],
        "bb_upper": sq["bb_upper"],
        "bb_mid": sq["bb_mid"],
        "bb_lower": sq["bb_lower"],
        "kc_upper": sq["kc_upper"],
        "kc_lower": sq["kc_lower"],
        "summary": summary,
    }


@mcp.tool()
def get_volume_profile(symbol: str, interval: str = "15m", limit: int = 192, bins: int = 24, asset_class: str | None = None) -> dict:
    """Price-binned volume profile → POC (point of control), value area (70%),
    and top high-volume nodes. Use to put entries at HVNs (support) not LVN air.

    symbol:   pair like 'BTCUSDT'.
    interval: candle interval (default 15m for intraday scalping context).
    limit:    candles to fetch (192 = ~48h at 15m).
    bins:     number of price bins (default 24).

    Returns poc, vah (value-area high), val (value-area low), in_value_area flag
    for the current close, the top 5 nodes by volume, and a text summary.
    """
    symbol = symbol.upper()
    limit = _clamp_limit(limit)
    try:
        candles = router.fetch_ohlcv(symbol, interval, limit, asset_class)
    except Exception as e:
        return {"error": f"failed to fetch klines for {symbol} {interval}: {e}"}

    vp = calc_volume_profile(candles, bins=bins)
    if vp is None:
        return {"error": "insufficient or degenerate price range for volume profile"}

    current_close = float(candles[-1][4])
    in_va = vp["val"] <= current_close <= vp["vah"]
    if current_close > vp["vah"]:
        loc = "above value area"
    elif current_close < vp["val"]:
        loc = "below value area"
    else:
        loc = "in value area"

    summary = (
        f"{symbol} {interval} ({len(candles)} bars) | POC: {vp['poc']:.6g} "
        f"| VA: [{vp['val']:.6g}, {vp['vah']:.6g}] | close {current_close:.6g} → {loc}"
    ) + _equity_volume_caveat(symbol, asset_class)

    return {
        "symbol": symbol,
        "interval": interval,
        "bins": bins,
        "p_min": vp["p_min"],
        "p_max": vp["p_max"],
        "poc": vp["poc"],
        "vah": vp["vah"],
        "val": vp["val"],
        "current_close": current_close,
        "in_value_area": in_va,
        "top_nodes": vp["top_nodes"],
        "summary": summary,
    }


if __name__ == "__main__":
    mcp.run()
