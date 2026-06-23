"""
Data-provider package: pure I/O, one module per source.

Each provider module returns parsed JSON / normalized rows and imports nothing from
the domain (indicators), presentation (formatting), or orchestration (services)
layers. `router` selects the right provider per asset class.

    from providers import binance, bybit, hyperliquid, coinbase, coingecko, alpaca, router
"""
from . import base, binance, bybit, hyperliquid, coinbase, coingecko, alpaca, router

__all__ = ["base", "binance", "bybit", "hyperliquid", "coinbase", "coingecko", "alpaca", "router"]
