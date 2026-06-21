"""
Pull multi-timeframe candles from Binance public API (no key needed).
Writes a compact text snapshot to a file alongside the script.

Usage:
    python FetchKlines.py ZECUSDT
    python FetchKlines.py BTCUSDT
    python FetchKlines.py --clean          # delete all snapshot txt files
    python FetchKlines.py --clean BTCUSDT  # delete only BTCUSDT snapshots

Data fetching, indicator math, and text rendering live in sources.py /
indicators.py / formatting.py so the MCP server (mcp_server.py) reuses them.
"""
import sys
import requests
from datetime import datetime, timezone
from pathlib import Path

import sources
from indicators import analyze_orderbook, calc_fibs, calc_adx, calc_ema, calc_atr, classify_regime
from formatting import fmt_candle, fmt_orderbook, fmt_fibs, fmt_indicators, fmt_futures_context

# How many candles per timeframe. Enough lookback for trend/SR, not so much it spams context.
TIMEFRAMES = [
    ("1w",  30),   # ~7 months
    ("1d",  60),   # 2 months
    ("4h",  60),   # 10 days
    ("1h",  48),   # 2 days
    ("15m", 32),   # 8 hours
]

# Dedicated series for the trend-regime meta-filter — needs ~200+ closes for the 200-EMA.
REGIME_INTERVAL, REGIME_LIMIT = "4h", 300


def regime_lines(symbol: str) -> list:
    """Top-level trend-regime classification (Plan.md Stage 1 meta-filter).
    One dedicated 4h/300 fetch so the 200-EMA can seed; gates how to read the
    rest of the snapshot (trend-following vs mean-reversion vs stand-aside)."""
    lines = ["=== Market Regime (meta-filter) ==="]
    try:
        candles = sources.fetch_klines(symbol, REGIME_INTERVAL, REGIME_LIMIT)
    except Exception as e:
        lines.append(f"  N/A ({e})")
        return lines
    closes = [float(k[4]) for k in candles]
    close = closes[-1]
    adx, pdi, ndi = calc_adx(candles)
    ema_200 = calc_ema(closes, 200)
    atr = calc_atr(candles)
    reg = classify_regime(adx, pdi, ndi, close, ema_200)
    lines.append(
        f"  {REGIME_INTERVAL}: {reg['regime'].upper()} → {reg['mode']} | "
        f"ADX {adx:.1f} ({reg['adx_state']}), +DI {pdi:.1f}/-DI {ndi:.1f}"
        + (f", close {close:.6g} vs 200-EMA {ema_200:.6g}" if ema_200 is not None else " (200-EMA n/a)")
    )
    if atr is not None:
        lines.append(f"  ATR(14): {atr:.6g} ({atr / close * 100:.2f}% of close) — stop-distance unit")
    lines.append(f"  → {reg['playbook']}")
    return lines


def clean_snapshots(symbol_filter: str | None = None):
    out_dir = Path(__file__).parent
    pattern = f"{symbol_filter}_????-??-??_????.txt" if symbol_filter else "*_????-??-??_????.txt"
    files = sorted(out_dir.glob(pattern))
    if not files:
        print("No snapshot files found.")
        return
    for f in files:
        f.unlink()
        print(f"Deleted {f.name}")
    print(f"Removed {len(files)} file(s).")


def main():
    args = sys.argv[1:]

    if "--clean" in args:
        args.remove("--clean")
        symbol_filter = args[0].upper() if args else None
        clean_snapshots(symbol_filter)
        return

    symbol = args[0].upper() if args else "BTCUSDT"

    lines = []
    lines.append(f"=== {symbol} multi-timeframe snapshot ===")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")

    lines.extend(regime_lines(symbol))
    lines.append("")

    lines.append("=== Order Book Snapshot (top 20 levels) ===")
    for exch_name, fetcher in [
        ("Binance",      sources.fetch_orderbook_binance),
        ("Bybit",        sources.fetch_orderbook_bybit),
        ("Hyperliquid",  sources.fetch_orderbook_hyperliquid),
    ]:
        try:
            depth = fetcher(symbol)
            lines.extend(fmt_orderbook(analyze_orderbook(depth), label=exch_name))
        except Exception as e:
            lines.append(f"  [{exch_name}] N/A ({e})")
    lines.append("")

    lines.extend(fmt_futures_context(sources.compute_futures_context(symbol)))
    lines.append("")

    for interval, limit in TIMEFRAMES:
        try:
            candles = sources.fetch_klines(symbol, interval, limit)
        except requests.HTTPError as e:
            lines.append(f"[{interval}] ERROR: {e}")
            lines.append("")
            continue

        last_close = float(candles[-1][4])
        first_close = float(candles[0][4])
        change_pct = (last_close - first_close) / first_close * 100

        lines.append(f"--- {interval} ({limit} candles, change over window: {change_pct:+.2f}%) ---")
        for k in candles:
            lines.append(fmt_candle(k))
        lines.append("Fibonacci retracements (swing high → low):")
        lines.extend(fmt_fibs(calc_fibs(candles)))
        lines.extend(fmt_indicators(candles))
        lines.append("")

    # Write next to the script, named like ZECUSDT_2026-05-20_1430.txt
    out_dir = Path(__file__).parent
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    out_path = out_dir / f"{symbol}_{stamp}.txt"
    out_path.write_text("\n".join(lines))

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
