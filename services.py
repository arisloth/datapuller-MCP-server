"""
Orchestration layer: compose provider I/O with domain math into ready-to-render
results. Sits above providers/ and indicators/; below formatting/ and the tools.
Keeps providers pure I/O (no indicators dependency) and formatters I/O-free.
"""
import indicators
from providers import binance


def compute_futures_context(symbol: str) -> dict:
    """Fetch Binance USDⓈ-M perpetual futures context for `symbol`, once.

    Performs all the futures round-trips (premium index + funding history, open
    interest + 5h history, global long/short ratio) and folds in the derived
    measures (basis, funding APR/percentile/extreme). Each field is None when its
    endpoint is unavailable (e.g. no perp for the symbol), so callers never re-fetch
    and formatters do no I/O.
    """
    ctx = {
        "symbol": symbol,
        "funding_rate_pct": None,
        "funding_apr": None,
        "funding_percentile": None,
        "funding_reading": None,
        "next_funding": None,
        "mark_price": None,
        "index_price": None,
        "basis_pct": None,
        "basis_state": None,
        "open_interest": None,
        "oi_change_pct_5h": None,
        "price_change_pct_5h": None,
        "long_short_ratio": None,
        "long_pct": None,
        "short_pct": None,
    }

    try:
        pm = binance.fetch_premium_index(symbol)
        ctx["funding_rate_pct"] = float(pm["lastFundingRate"]) * 100
        ctx["next_funding"] = int(pm["nextFundingTime"])
        ctx["mark_price"] = float(pm["markPrice"])
        ctx["index_price"] = float(pm["indexPrice"])
        if ctx["index_price"]:
            ctx["basis_pct"] = (ctx["mark_price"] - ctx["index_price"]) / ctx["index_price"] * 100
            ctx["basis_state"] = indicators.classify_basis(ctx["basis_pct"])["state"]
    except Exception:
        pass

    # Funding depth: annualize current funding and rank it vs settled history.
    try:
        hist = binance.fetch_funding_history(symbol, limit=100)
        if hist and ctx["funding_rate_pct"] is not None:
            interval_h = indicators.infer_funding_interval_hours(hist)
            ctx["funding_apr"] = indicators.annualize_funding(ctx["funding_rate_pct"] / 100, interval_h)
            apr_hist = [indicators.annualize_funding(float(h["fundingRate"]), interval_h) for h in hist]
            ctx["funding_percentile"] = indicators.percentile_rank(ctx["funding_apr"], apr_hist)
            ctx["funding_reading"] = indicators.classify_funding(ctx["funding_apr"])["reading"]
    except Exception:
        pass

    try:
        ctx["open_interest"] = binance.fetch_open_interest(symbol)
        hist = binance.fetch_open_interest_hist(symbol, period="1h", limit=6)
        if len(hist) >= 2:
            oi_old = float(hist[0]["sumOpenInterest"])
            oi_new = float(hist[-1]["sumOpenInterest"])
            ctx["oi_change_pct_5h"] = (oi_new - oi_old) / oi_old * 100 if oi_old else 0.0
            # Implied mark price per snapshot = notional value / contracts. Lets us
            # measure price change over the *same* window as OI (for the quadrant)
            # without a second request. Guard against missing/zero fields.
            val_old = float(hist[0].get("sumOpenInterestValue", 0) or 0)
            val_new = float(hist[-1].get("sumOpenInterestValue", 0) or 0)
            if oi_old and oi_new and val_old and val_new:
                px_old = val_old / oi_old
                px_new = val_new / oi_new
                ctx["price_change_pct_5h"] = (px_new - px_old) / px_old * 100 if px_old else 0.0
    except Exception:
        pass

    try:
        data = binance.fetch_long_short_ratio(symbol, period="1h", limit=1)
        if data:
            ls = data[0]
            ctx["long_short_ratio"] = float(ls["longShortRatio"])
            ctx["long_pct"] = float(ls["longAccount"]) * 100
            ctx["short_pct"] = float(ls["shortAccount"]) * 100
    except Exception:
        pass

    return ctx
