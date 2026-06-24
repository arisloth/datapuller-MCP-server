#  Market Data Puller

Pulls live crypto market data from public exchange APIs (Binance, Bybit, Hyperliquid —
no API key needed) and serves it two ways:

1. **CLI** — `FetchKlines.py` writes a compact multi-timeframe text snapshot to disk.
2. **MCP server** — `mcp_server.py` exposes the data as tools an LLM (Claude Desktop /
   Claude Code) can call on demand, so every answer reflects current market state.

**Crypto is keyless** (Binance, Bybit, Hyperliquid, Coinbase, CoinGecko — public APIs).
**Stocks & ETFs** (incl. commodity ETFs like GLD/USO) come from **Alpaca** and need an API
key in your environment — see [Multi-asset](#multi-asset-stocks--etfs). Commodity *futures*
are not covered (use ETF proxies).

## Layout

Layered so dependencies point downward only — I/O ↘ orchestration ↘ math/presentation ↘ tools:

| Path | Layer | Role |
|------|-------|------|
| `providers/` | I/O | One module per source — `binance`, `bybit`, `hyperliquid`, `coinbase`, `coingecko`, `alpaca` (equities), plus `base` (shared pooled HTTP session) and `router` (asset-class dispatch). Pure fetch → parsed JSON / normalized rows; imports nothing from the layers below. |
| `services.py` | Orchestration | Composes providers + indicators into ready results (`compute_futures_context`). |
| `indicators.py` | Domain math | Pure functions — OBV, volume ratio, ADX, EMA/ATR/VWAP, volume profile, Fibonacci, trend regime, OI/price quadrant, L/S & funding extremes, perp basis, CVD + divergence, taker ratio, Bollinger/TTM squeeze, BTC correlation/beta + rotation, candlestick patterns. No I/O. |
| `formatting.py` | Presentation | Text renderers shared by the CLI and tool summaries. No I/O. |
| `FetchKlines.py` | Delivery | CLI snapshot tool. |
| `mcp_server.py` | Delivery | MCP (stdio) server exposing the tools. |
| `requirements.txt` | — | `requests`, `mcp[cli]`, `pytest` |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Multi-asset (stocks & ETFs)

Crypto works out of the box (keyless). To pull **stocks/ETFs** (e.g. `AAPL`, `SPY`, `GLD`),
set Alpaca credentials in your environment (never commit them):

```bash
export APCA_API_KEY_ID=your_key
export APCA_API_SECRET_KEY=your_secret
export ALPACA_FEED=iex      # default; set 'sip' if you have the paid full-tape subscription
```

- **Routing is automatic:** a `USDT`-suffixed symbol (`BTCUSDT`) → crypto; anything else
  (`AAPL`) → equity via Alpaca. Every price tool also takes `asset_class="crypto"|"equity"`
  to force it.
- **Equity-capable tools:** `get_klines`, `get_indicators`, `get_emas`, `get_vwap`, `get_atr`,
  `get_volume_profile`, `get_squeeze`, `get_regime`, `get_patterns`, `get_correlation`
  (reference defaults to `SPY` for equities).
- **Crypto-only tools** (return a clean N/A for equities): `get_orderbook`,
  `get_futures_context`, `get_funding`, `get_cvd`, `get_volume_breakdown`, `get_market_breadth`.
- **IEX feed caveat:** the free Alpaca feed reports only IEX volume (~2–3% of the tape), so
  volume-based fields (volume profile/ratio/OBV) are flagged low-confidence for equities; price
  tools are unaffected. Set `ALPACA_FEED=sip` for full-tape volume.
- **Not available for equities:** CVD/taker (bars carry no trade side) and commodity *futures*
  (use ETF proxies like GLD/USO).

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
| `get_futures_context` | `symbol` | funding rate **+ annualized APR, percentile & extreme flag**, next funding, mark/index price, **perp basis** (contango/backwardation), open interest + 5h trend, **OI/price quadrant** (long build-up / short build-up / short-covering / long-liquidation), and the long/short account ratio **demoted to an extreme-only contrarian flag** (mid-range is labeled noise) |
| `get_funding` | `symbol` | Cross-exchange funding (Binance / Bybit / Hyperliquid) normalized to **APR**, Binance extreme/percentile vs its own history, and the cross-venue APR spread (positioning/arb divergence). Funding is contrarian context, not a timing trigger |
| `get_regime` | `symbol`, `interval="4h"`, `limit=300` | **Trend-regime meta-filter:** ADX(14) + price vs 200-EMA → regime (trend_up/trend_down/range/transitional) and mode (trend-following/mean-reversion/stand-aside), with ATR(14) and a one-line playbook. Gates how to read every other signal |
| `get_indicators` | `symbol`, `interval="1h"`, `limit=60` | OBV, **CVD (+trend & price divergence)**, **taker buy/sell ratio**, volume ratio, ADX(14) with +DI/-DI framed as a **regime gate** (DI actionable only when ADX≥25), **TTM squeeze + Bollinger width**, **candlestick patterns** (confirmation-only), Fibonacci retracements |
| `get_patterns` | `symbol`, `interval="1h"`, `limit=192` | Candlestick patterns on the latest bar (hammer, shooting star, doji, marubozu, engulfing, inside bar) **scored as confirmation only** — each gets a verdict (confirmed/weak/mixed/conflicting/unconfirmed/neutral) from CVD + taker flow + whether it sits at a volume-profile level. Never a standalone signal |
| `get_cvd` | `symbol`, `interval="15m"`, `limit=96` | Cumulative Volume Delta on **perp + spot** with trend, taker ratio, CVD-vs-price divergence, and the **spot-vs-perp** conviction read (perp-led = fragile/leverage; spot-led = higher-conviction). Degrades to one market if the other isn't listed |
| `get_squeeze` | `symbol`, `interval="1h"`, `limit=100`, `period=20` | TTM-style volatility squeeze (Bollinger inside Keltner) + Bollinger band width with percentile and compressed/normal/expanded state — breakout-timing filter, non-directional |
| `get_emas` | `symbol`, `interval="1h"`, `limit=500` | 20/50/200 EMAs + trend-stack label (bullish/bearish/mixed/n/a) |
| `get_volume_breakdown` | `symbol` | 24h USD volume: Binance spot + Coinbase + cross-exchange aggregate (via CoinGecko) with shares, **explicit cross-venue perp volume (Binance / Bybit / Hyperliquid)**, **perp/spot ratio + leverage-led vs spot-led read**, `thin` flag — US/institutional divergence on majors, true total on thin alts |
| `get_vwap` | `symbol`, `interval="5m"`, `limit=288` | Session VWAP (resets at 00:00 UTC) + window VWAP, each with 1σ/2σ bands, plus a long/short/neutral bias vs current close |
| `get_atr` | `symbol`, `interval="30m"`, `limit=100`, `period=14`, `account_equity=None`, `risk_pct=1.0`, `stop_atr_mult=1.5` | ATR(14) and example 1×/1.5× ATR stop distances for long and short. Pass `account_equity` to also get an ATR-normalized **position size** (risk `risk_pct`% across a `stop_atr_mult`×ATR stop) — for stop placement and sizing, not direction |
| `get_volume_profile` | `symbol`, `interval="15m"`, `limit=192`, `bins=24` | POC, value area (70%), top 5 high-volume nodes, and whether the current close sits inside the value area |
| `get_correlation` | `symbol`, `interval="1h"`, `limit=200`, `btc="BTCUSDT"` | Rolling **correlation + beta** of an alt vs BTC, a recent-half correlation with a **decoupling** flag, and a gating read (high corr → trade BTC's regime; low → alt-specific edge valid) |
| `get_market_breadth` | _(none)_ | Total market cap + 24h change, TOTAL2 (ex-BTC), BTC/ETH/stablecoin **dominance** with BTC.D direction, ETH/BTC bellwether, and a **rotation read** (btc-dominant / alt-rotation / risk-off) for higher-timeframe alt bias |

`symbol` is a crypto pair like `BTCUSDT`, `ETHUSDT` (quote in USDT) or — for the equity-capable
tools — a stock/ETF ticker like `AAPL`, `SPY`, `GLD` (needs Alpaca creds; see
[Multi-asset](#multi-asset-stocks--etfs)).

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
         "command": "/absolute/path/to/binancedatapuller/.venv/bin/python",
         "args": ["/absolute/path/to/binancedatapuller/mcp_server.py"]
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
  /absolute/path/to/binancedatapuller/.venv/bin/python \
  /absolute/path/to/binancedatapuller/mcp_server.py
```

## Notes

- Data is fetched on demand per tool call — no streaming. Fresh enough for request/response
  reasoning, and avoids hammering rate limits.
- The most recent candle may be in progress (incomplete), same as the raw exchange data.
- Binance public endpoints are rate-limited; on-demand tool calls stay well within limits.
- All data is from **keyless public APIs**. Two signal classes are intentionally out of scope
  because they aren't available keyless:
  - **Liquidations** — not exposed by keyless public REST (Binance `allForceOrders` returns 400;
    Bybit is WebSocket-only). Cascade *risk* is instead readable from the OI quadrant, funding
    extremes, and ATR.
  - **On-chain flows** (exchange netflows, stablecoin/whale activity) — require a paid/keyed
    provider (Glassnode/CryptoQuant/Nansen) with no price-derivable proxy.
  Both would need a paid/keyed source to add later.
