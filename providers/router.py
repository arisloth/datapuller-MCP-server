"""
Asset-class routing for OHLCV: pick the right provider for a symbol.

Crypto symbols are USDT-quoted pairs (BTCUSDT); everything else is treated as an
equity/ETF served by Alpaca. An explicit `asset_class` override always wins.
"""
from . import binance, alpaca

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
    """Fetch normalized OHLCV rows for any asset class via the right provider."""
    if resolve_asset_class(symbol, asset_class) == "crypto":
        return binance.fetch_klines(symbol, interval, limit)
    return alpaca.fetch_ohlcv(symbol, interval, limit)
