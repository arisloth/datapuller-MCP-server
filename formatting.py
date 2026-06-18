"""
Text renderers. Turn raw candles / structured analysis into the human-readable
lines used by both the CLI snapshot and the MCP tool `summary` fields.

These functions do NO network or heavy computation: callers pass in already
fetched/computed data (see sources.compute_futures_context, indicators.compute_indicators).
"""
from datetime import datetime, timezone

from indicators import compute_indicators


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

    if adx < 20:
        adx_label = "weak/ranging"
    elif adx < 25:
        adx_label = "developing"
    else:
        adx_label = "trending"

    bias = ""
    if adx >= 20:
        bias = " (bullish bias)" if pdi > ndi else " (bearish bias)"

    return [
        "Indicators:",
        f"  OBV: {obv:+.0f} ({obv_trend})",
        f"  Volume ratio (current / 20-bar avg): {vol_ratio:.2f}x",
        f"  ADX(14): {adx:.1f} ({adx_label}) | +DI: {pdi:.1f} | -DI: {ndi:.1f}{bias}",
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

    # --- open interest (current + 5h trend) ---
    if ctx["open_interest"] is not None:
        oi_chg = ctx["oi_change_pct_5h"]
        if oi_chg is None:
            oi_trend = "n/a"
        else:
            oi_trend = f"{'rising' if oi_chg > 0 else 'falling'} ({oi_chg:+.2f}% over 5h)"
        lines.append(f"  Open Interest: {ctx['open_interest']:.4g} {symbol.replace('USDT', '')}  |  Trend: {oi_trend}")
    else:
        lines.append("  Open Interest: N/A")

    # --- global long/short account ratio ---
    if ctx["long_short_ratio"] is not None:
        ratio = ctx["long_short_ratio"]
        bias = "longs dominant" if ratio > 1 else "shorts dominant"
        lines.append(f"  L/S ratio:     {ratio:.3f} ({bias})  |  Long: {ctx['long_pct']:.1f}%  Short: {ctx['short_pct']:.1f}%")
    else:
        lines.append("  L/S ratio:     N/A")

    return lines
