"""
Asset-class routing for OHLCV: pick the right provider for a symbol.

Crypto symbols are USDT-quoted pairs (BTCUSDT); everything else is treated as an
equity/ETF served by Alpaca. An explicit `asset_class` override always wins.

Crypto OHLCV is multi-venue: Binance first (spot, then USDⓈ-M futures), then
Bybit / Coinbase / Hyperliquid spot for symbols Binance doesn't list.
"""
from . import binance, bybit, coinbase, hyperliquid, alpaca

CRYPTO_SUFFIXES = ("USDT",)


def resolve_asset_class(symbol: str, override: str | None = None) -> str:
    """Return 'crypto' or 'equity' for a symbol. `override` ('crypto'/'equity') wins."""
    if override:
        ov = override.lower()
        if ov in ("crypto", "equity"):
            return ov
        raise ValueError(f"asset_class must be 'crypto' or 'equity', got {override!r}")
    return "crypto" if symbol.upper().endswith(CRYPTO_SUFFIXES) else "equity"


def fetch_ohlcv(symbol: str, interval: str, limit: int, asset_class: str | None = None):
    """Fetch normalized OHLCV rows for any asset class via the right provider.

    Crypto walks the venue chain (Binance spot→futures, then Bybit/Coinbase/
    Hyperliquid spot) and returns the first venue with candles; raises ValueError
    with every venue's failure when none serve the symbol."""
    if resolve_asset_class(symbol, asset_class) != "crypto":
        return alpaca.fetch_ohlcv(symbol, interval, limit)
    # Fallback rows carry no taker-buy field, so CVD/taker metrics degrade to n/a.
    venues = (
        ("binance", binance.fetch_klines),
        ("bybit", bybit.fetch_klines_spot),
        ("coinbase", coinbase.fetch_klines_spot),
        ("hyperliquid", hyperliquid.fetch_klines_spot),
    )
    errors = []
    for venue, fetcher in venues:
        try:
            rows = fetcher(symbol, interval, limit)
            if rows:
                return rows
            errors.append(f"{venue}: no candles")
        except Exception as e:
            errors.append(f"{venue}: {e}")
    raise ValueError(f"no OHLCV for {symbol} {interval} on any venue ({'; '.join(errors)})")
