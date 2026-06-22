"""
Text renderers. Turn raw candles / structured analysis into the human-readable
lines used by both the CLI snapshot and the MCP tool `summary` fields.

These functions do NO network or heavy computation: callers pass in already
fetched/computed data (see sources.compute_futures_context, indicators.compute_indicators).
"""
from datetime import datetime, timezone

from indicators import (
    compute_indicators, classify_oi_price, classify_long_short,
    ADX_TREND, ADX_RANGE,
)


def fmt_candle(k):
    # Binance kline: [openTime, open, high, low, close, volume, closeTime, quoteVol, trades, ...]
    ts = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    o, h, l, c, v = float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])
    return f"{ts} | O:{o:.4g} H:{h:.4g} L:{l:.4g} C:{c:.4g} V:{v:.4g}"


def fmt_orderbook(analysis, label: str = ""):
    lines = []
    if label:
        lines.append(f"  [{label}]")
    lines.append(
        f"  Best bid: {analysis['best_bid']:.4g}  |  Best ask: {analysis['best_ask']:.4g}  "
        f"|  Spread: {analysis['spread']:.4g} ({analysis['spread_pct']:.4f}%)"
    )
    lines.append(f"  {'Depth':<8} {'Bid vol':>14} {'Ask vol':>14} {'Ratio (b/a)':>12} {'Pressure':>10}")

    for lvl in analysis["levels"]:
        n = lvl["depth"]
        lines.append(
            f"  {f'{n} lvl':<8} {lvl['bid_vol']:>14.4g} {lvl['ask_vol']:>14.4g} "
            f"{lvl['ratio']:>11.2f}x {lvl['pressure']:>10}"
        )

    return lines


def fmt_fibs(fibs):
    if fibs is None:
        return []
    lines = [f"  Fib range: {fibs['swing_low']:.4g} – {fibs['swing_high']:.4g}"]
    for i, lvl in enumerate(fibs["levels"]):
        label = f"  {lvl['level'] * 100:.1f}%  {lvl['price']:.4g}"
        marker = "  <-- current" if i == fibs["closest_index"] else ""
        lines.append(label + marker)
    return lines


def fmt_indicators(values):
    """Render indicator lines from a compute_indicators() dict (or raw candles
    for backwards compatibility — they're computed once if a list is passed)."""
    if not isinstance(values, dict):
        values = compute_indicators(values)
    obv       = values["obv"]
    obv_trend = values["obv_trend"]
    vol_ratio = values["volume_ratio"]
    adx       = values["adx"]
    pdi       = values["plus_di"]
    ndi       = values["minus_di"]
    cvd          = values.get("cvd")
    cvd_trend    = values.get("cvd_trend", "n/a")
    cvd_div      = values.get("cvd_divergence", "none")
    taker_ratio  = values.get("taker_ratio")
    squeeze_on   = values.get("squeeze_on")
    bbw          = values.get("bbw")
    bbw_state    = values.get("bbw_state")

    # ADX as a regime GATE, not a momentum trigger (Plan.md Stage 1 #3):
    # DI direction is only actionable in a trending regime — below ADX_TREND it
    # whipsaws, so we explicitly tell the reader to ignore DI crossovers there.
    if adx < ADX_RANGE:
        adx_label = "ranging — ignore DI crossovers (mean-reversion regime)"
        bias = ""
    elif adx < ADX_TREND:
        adx_label = "developing — DI not yet reliable, wait for ADX>25"
        bias = ""
    else:
        direction = "bullish" if pdi > ndi else "bearish"
        adx_label = "trending — DI direction actionable"
        bias = f" ({direction})"

    # CVD signs every executed trade by aggressor side — it dominates OBV on perps.
    if cvd is None:
        cvd_line = "  CVD: n/a (no taker data) | OBV: " f"{obv:+.0f} ({obv_trend})"
    else:
        div_note = "" if cvd_div == "none" else f"  ⚠ {cvd_div} divergence vs price"
        cvd_line = f"  CVD: {cvd:+.0f} ({cvd_trend}){div_note} | OBV: {obv:+.0f} ({obv_trend})"

    taker_line = (
        f"  Taker buy/sell: {taker_ratio:.2f}x ({'buy' if taker_ratio > 1 else 'sell'}-side aggression)"
        if taker_ratio is not None else "  Taker buy/sell: n/a"
    )

    if squeeze_on is None:
        squeeze_line = "  Squeeze: n/a"
    else:
        bbw_str = f"{bbw:.4f}" if bbw is not None else "n/a"
        squeeze_line = (
            f"  Squeeze: {'ON (coiled — expansion likely)' if squeeze_on else 'off'} "
            f"| BBW {bbw_str} ({bbw_state})"
        )

    return [
        "Indicators:",
        cvd_line,
        taker_line,
        f"  Volume ratio (current / 20-bar avg): {vol_ratio:.2f}x",
        f"  ADX(14): {adx:.1f} [{adx_label}] | +DI: {pdi:.1f} | -DI: {ndi:.1f}{bias}",
        squeeze_line,
        "  (full regime needs the 200-EMA → use get_regime)",
    ]


def fmt_futures_context(ctx):
    """Render the futures-context lines from a sources.compute_futures_context()
    dict (passing a symbol string is still accepted and triggers one fetch)."""
    if not isinstance(ctx, dict):
        import sources
        ctx = sources.compute_futures_context(ctx)

    symbol = ctx["symbol"]
    lines = ["=== Futures Market Context ==="]

    # --- funding rate + mark/index price ---
    if ctx["funding_rate_pct"] is not None:
        rate = ctx["funding_rate_pct"]
        next_ts = datetime.fromtimestamp(ctx["next_funding"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        direction = "longs pay shorts (bullish overheat)" if rate >= 0 else "shorts pay longs (bearish overheat)"
        lines.append(f"  Funding rate:  {rate:+.4f}% per 8h  →  {direction}")
        lines.append(f"  Next funding:  {next_ts}")
        lines.append(f"  Mark price:    {ctx['mark_price']:.4g}  |  Index: {ctx['index_price']:.4g}")
    else:
        lines.append("  Funding rate:  N/A (no perp for this symbol)")

    # --- open interest (current + 5h trend) + OI/price quadrant ---
    if ctx["open_interest"] is not None:
        oi_chg = ctx["oi_change_pct_5h"]
        if oi_chg is None:
            oi_trend = "n/a"
        else:
            oi_trend = f"{'rising' if oi_chg > 0 else 'falling'} ({oi_chg:+.2f}% over 5h)"
        lines.append(f"  Open Interest: {ctx['open_interest']:.4g} {symbol.replace('USDT', '')}  |  Trend: {oi_trend}")
        # OI alone has no direction — pair its change with price change (Plan.md Tier-1 #3).
        q = classify_oi_price(oi_chg, ctx.get("price_change_pct_5h"))
        if q["quadrant"] != "neutral" or oi_chg is not None:
            px_chg = ctx.get("price_change_pct_5h")
            px_str = f"price {px_chg:+.2f}%" if px_chg is not None else "price n/a"
            lines.append(f"  OI quadrant:   {q['label']} ({px_str}, OI {oi_chg:+.2f}% / 5h) — {q['interpretation']}")
    else:
        lines.append("  Open Interest: N/A")

    # --- global long/short account ratio (demoted: extreme-flag only) ---
    if ctx["long_short_ratio"] is not None:
        ratio = ctx["long_short_ratio"]
        ls = classify_long_short(ratio)
        lines.append(
            f"  L/S accounts:  {ratio:.3f}  (Long {ctx['long_pct']:.1f}% / Short {ctx['short_pct']:.1f}%) "
            f"— {ls['note']}"
        )
    else:
        lines.append("  L/S accounts:  N/A")

    return lines
