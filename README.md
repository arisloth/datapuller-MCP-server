# Binance Data Puller

Pulls live crypto market data from public exchange APIs (Binance, Bybit, Hyperliquid —
no API key needed) and serves it two ways:

1. **CLI** — `FetchKlines.py` writes a compact multi-timeframe text snapshot to disk.
2. **MCP server** — `mcp_server.py` exposes the same data as tools an LLM (Claude Desktop /
   Claude Code) can call on demand, so every answer reflects current market state.

## Layout

| File | Role |
|------|------|
| `sources.py` | Raw HTTP fetchers, return parsed JSON |
| `indicators.py` | Pure math — OBV, volume ratio, ADX, order-book imbalance, Fibonacci |
| `formatting.py` | Text renderers shared by the CLI and the tool summaries |
| `FetchKlines.py` | CLI snapshot tool |
| `mcp_server.py` | MCP (stdio) server exposing the tools |
| `requirements.txt` | `requests`, `mcp[cli]` |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## CLI usage

```bash
.venv/bin/python FetchKlines.py BTCUSDT      # write a snapshot for BTCUSDT
.venv/bin/python FetchKlines.py ZECUSDT
.venv/bin/python FetchKlines.py --clean      # delete all snapshot .txt files
.venv/bin/python FetchKlines.py --clean BTCUSDT  # delete only BTCUSDT snapshots
```

Writes a file like `BTCUSDT_2026-05-29_1430.txt` next to the script.

## MCP tools

Each tool returns structured fields **plus** a `summary` text block, and returns
`{"error": "..."}` on failure instead of crashing.

| Tool | Arguments | Returns |
|------|-----------|---------|
| `get_klines` | `symbol`, `interval="1h"`, `limit=50` | OHLCV candles + % change over the window |
| `get_orderbook` | `symbol`, `exchange="binance"` (`binance`/`bybit`/`hyperliquid`) | best bid/ask, spread, 5/10/20-level bid-vs-ask imbalance + pressure |
| `get_futures_context` | `symbol` | funding rate, next funding, mark/index price, open interest + 5h trend, long/short ratio |
| `get_indicators` | `symbol`, `interval="1h"`, `limit=60` | OBV, volume ratio, ADX(14) with +DI/-DI, Fibonacci retracements |
| `get_emas` | `symbol`, `interval="1h"`, `limit=500` | 20/50/200 EMAs + trend-stack label (bullish/bearish/mixed/n/a) |
| `get_volume_breakdown` | `symbol` | 24h USD volume on Binance + Coinbase + cross-exchange aggregate (via CoinGecko), shares of each, `thin` flag — US/institutional divergence on majors, true total on thin alts |
| `get_vwap` | `symbol`, `interval="5m"`, `limit=288` | Session VWAP (resets at 00:00 UTC) + window VWAP, each with 1σ/2σ bands, plus a long/short/neutral bias vs current close |
| `get_atr` | `symbol`, `interval="30m"`, `limit=100`, `period=14` | ATR(14) and example 1×/1.5× ATR stop distances for long and short — for stop placement, not direction |
| `get_volume_profile` | `symbol`, `interval="15m"`, `limit=192`, `bins=24` | POC, value area (70%), top 5 high-volume nodes, and whether the current close sits inside the value area |

`symbol` is a pair like `BTCUSDT`, `ETHUSDT`, `ZECUSDT` (quote in USDT).

### Try it interactively

```bash
.venv/bin/mcp dev mcp_server.py
```

Opens the MCP Inspector in your browser; call each tool and inspect the responses.

## Use with Claude Desktop

1. Open Claude Desktop → **Settings → Developer → Edit Config**. This opens
   `claude_desktop_config.json` (on macOS it lives at
   `~/Library/Application Support/Claude/claude_desktop_config.json`).

2. Add this server under `mcpServers` (use absolute paths):

   ```json
   {
     "mcpServers": {
       "binance-data": {
         "command": "/Users/arian/Documents/Uni/PP/binancedatapuller/.venv/bin/python",
         "args": ["/Users/arian/Documents/Uni/PP/binancedatapuller/mcp_server.py"]
       }
     }
   }
   ```

   If the file already has other servers, add `"binance-data"` as another key inside the
   existing `mcpServers` object rather than replacing it.

3. **Fully quit and reopen Claude Desktop** (the config is read on startup).

4. The tools appear under the tools/plug icon in the chat input. Then just ask, e.g.:
   - "Check the funding rate and order-book imbalance for ETHUSDT."
   - "Pull 4h candles and the ADX for BTCUSDT and tell me the trend."

   Claude calls the relevant tools and reasons over the fresh data it gets back.

### Use with Claude Code (alternative)

```bash
claude mcp add binance-data -- \
  /Users/arian/Documents/Uni/PP/binancedatapuller/.venv/bin/python \
  /Users/arian/Documents/Uni/PP/binancedatapuller/mcp_server.py
```

## Notes

- Data is fetched on demand per tool call — no streaming. Fresh enough for request/response
  reasoning, and avoids hammering rate limits.
- The most recent candle may be in progress (incomplete), same as the raw exchange data.
- Binance public endpoints are rate-limited; on-demand tool calls stay well within limits.
