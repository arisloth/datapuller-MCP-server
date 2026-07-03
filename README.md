# Market Data MCP

A **multi-asset, real-time market-data platform** served over MCP: stocks & ETFs
(`AAPL`, `SPY`, `GLD`) and crypto through one set of analysis tools, combining REST
snapshots with **live WebSocket ingestion** — streaming trades and NBBO quotes feed an
in-memory tape store, and each equity print is classified **buyer- vs seller-initiated
via Lee–Ready (1991)** against the prevailing NBBO, producing order-flow analytics
(CVD, taker ratio) that bar data alone cannot provide.

1. **MCP server** — `mcp_server.py` exposes ~18 analysis tools an LLM (Claude Desktop /
   Claude Code) calls on demand: OHLCV, indicators, trend regime, order flow, VWAP/ATR,
   volume profile, signal confluence. Tool responses are always aggregates — raw ticks
   never cross the tool boundary, so streaming adds zero token overhead.
2. **CLI** — `FetchKlines.py` writes a compact multi-timeframe text snapshot to disk.

**Equities** stream from **Alpaca** (trades + NBBO quotes over WebSocket; bars over REST)
and need a free API key — see [Multi-asset](#multi-asset-stocks--etfs). Additional venue
adapters cover crypto **keylessly** (Binance, Bybit, Hyperliquid, Coinbase, CoinGecko),
including perp derivatives context (funding, open interest, basis). Commodity *futures*
are not covered (use ETF proxies like GLD/USO).

## Architecture

Layered so dependencies point downward only — I/O ↘ orchestration ↘ math/presentation ↘ tools:

| Path | Layer | Role |
|------|-------|------|
| `streams/` | I/O (push) | WebSocket adapters — `alpaca` (equity trades + NBBO quotes), `binance` (spot aggTrade), `bybit` (perp publicTrade) over a shared `base` client (auto-reconnect with exponential backoff, resubscribe-on-reconnect, idle watchdog, venue keepalive). `manager` owns a daemon-thread event loop; subscriptions are lazy (first tool call), budgeted, and LRU-evicted (Alpaca IEX allows 1 connection / ~30 symbols). |
| `store.py` | State | Thread-safe in-memory tape store: per-symbol trade ring buffers + latest quote. **Lee–Ready trade classification** (quote rule → tick test) with **condition-code filtering** (non-regular-way prints excluded from flow). Aggregates out only: CVD, taker ratio, NBBO snapshot, staleness. |
| `providers/` | I/O (pull) | One module per REST source — `binance`, `bybit`, `hyperliquid`, `coinbase`, `coingecko`, `alpaca` (equities), plus `base` (pooled HTTP session with retry) and `router` (asset-class dispatch). Pure fetch → parsed JSON / normalized rows. |
| `services.py` | Orchestration | Composes providers + indicators into ready results (`compute_futures_context`). |
| `indicators.py` | Domain math | Pure functions — OBV, volume ratio, ADX, EMA/ATR/VWAP, volume profile, Fibonacci, trend regime, OI/price quadrant, L/S & funding extremes, perp basis, CVD + divergence, taker ratio, Bollinger/TTM squeeze, correlation/beta + rotation, candlestick patterns, signal confluence. No I/O. |
| `formatting.py` | Presentation | Text renderers shared by the CLI and tool summaries. No I/O. |
| `FetchKlines.py` | Delivery | CLI snapshot tool. |
| `mcp_server.py` | Delivery | MCP (stdio) server exposing the tools; reads REST cold paths and stream hot paths. |
| `requirements.txt` | — | `requests`, `mcp[cli]`, `websockets`, `pytest` |

Stream data flow: `streams/* → store.py (classify + buffer) → mcp_server.py (aggregate out)`.
The REST path stays as the cold path — every tool answers on first call, and the stream
upgrades data quality (trade-level flow, live quotes) once warm.

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
  (reference defaults to `SPY` for equities), `get_confluence` (equities skip the
  derivatives votes), and — **via the WebSocket stream** — `get_cvd` (live Lee–Ready-classified
  order flow) and `get_orderbook` (live NBBO top-of-book). Stream tools subscribe lazily on
  first call ("warming") and need an open US market to tick.
- **Crypto-only tools** (return a clean N/A for equities): `get_futures_context`,
  `get_funding`, `get_volume_breakdown`, `get_market_breadth`.
- **IEX feed caveat:** the free Alpaca feed reports only IEX volume (~2–3% of the tape), so
  volume-based fields (volume profile/ratio/OBV, streamed CVD magnitude) are flagged
  low-confidence for equities; price tools are unaffected. Set `ALPACA_FEED=sip` for the
  full consolidated tape.
- **Not available for equities:** L2 order-book depth (NBBO top-of-book only) and commodity
  *futures* (use ETF proxies like GLD/USO).

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
| `get_orderbook` | `symbol`, `exchange="binance"` (`binance`/`bybit`/`hyperliquid`; ignored for equities) | Crypto: best bid/ask, spread, 5/10/20-level bid-vs-ask imbalance + pressure. Equities: **live NBBO top-of-book** (bid/ask + sizes, spread, quote age) from the Alpaca stream |
| `get_futures_context` | `symbol` | funding rate **+ annualized APR, percentile & extreme flag**, next funding, mark/index price, **perp basis** (contango/backwardation), open interest + 5h trend, **OI/price quadrant** (long build-up / short build-up / short-covering / long-liquidation), and the long/short account ratio **demoted to an extreme-only contrarian flag** (mid-range is labeled noise) |
| `get_funding` | `symbol` | Cross-exchange funding (Binance / Bybit / Hyperliquid) normalized to **APR**, Binance extreme/percentile vs its own history, and the cross-venue APR spread (positioning/arb divergence). Funding is contrarian context, not a timing trigger |
| `get_regime` | `symbol`, `interval="4h"`, `limit=300` | **Trend-regime meta-filter:** ADX(14) + price vs 200-EMA → regime (trend_up/trend_down/range/transitional), mode (trend-following/mean-reversion/stand-aside) and a graded **conviction** (full/reduced/none — `reduced` means smaller size, not no-trade), with ATR(14) and a one-line playbook. Per-timeframe: gates how to read the other signals on that timeframe |
| `get_confluence` | `symbol`, `interval="1h"`, `limit=300` | **Signal-alignment scorecard** — the counterweight to per-tool caveats. Collects the stack's independent directional reads (regime, EMA stack, CVD, taker, VWAP, OI quadrant, funding & L/S contrarian extremes) into one verdict: **aligned** (≥3 agree, none oppose — the methodology's green light), **leaning** (majority — reduced size), **mixed** (genuine disagreement — no edge), **no_signal**. Names which reads agree/oppose; squeeze + ATR as context |
| `get_indicators` | `symbol`, `interval="1h"`, `limit=60` | OBV, **CVD (+trend & price divergence)**, **taker buy/sell ratio**, volume ratio, ADX(14) with +DI/-DI framed as a **regime gate** (DI actionable only when ADX≥25), **TTM squeeze + Bollinger width**, **candlestick patterns** (confirmation-only), Fibonacci retracements |
| `get_patterns` | `symbol`, `interval="1h"`, `limit=192` | Candlestick patterns on the latest bar (hammer, shooting star, doji, marubozu, engulfing, inside bar) **scored as confirmation only** — each gets a verdict (confirmed/weak/mixed/conflicting/unconfirmed/neutral) from CVD + taker flow + whether it sits at a volume-profile level. Never a standalone signal |
| `get_cvd` | `symbol`, `interval="15m"`, `limit=96` | Crypto: Cumulative Volume Delta on **perp + spot** with trend, taker ratio, CVD-vs-price divergence, and the **spot-vs-perp** conviction read (perp-led = fragile/leverage; spot-led = higher-conviction), plus `live` **trade-level flow from the WebSocket tape** (true aggressor side per print; perp via Bybit, spot via Binance) as a **1m/5m/15m window ladder with a shape label** (accelerating / steady / fading / flipping — building aggression vs stale burst vs absorption; shape gated until the tape spans the 15m window). Equities: **the same live ladder from the Alpaca stream, each print classified via Lee–Ready against the prevailing NBBO** (bars carry no taker side — stream-only). First call subscribes and warms in seconds |
| `get_squeeze` | `symbol`, `interval="1h"`, `limit=100`, `period=20` | TTM-style volatility squeeze (Bollinger inside Keltner) + Bollinger band width with percentile and compressed/normal/expanded state — breakout-timing filter, non-directional |
| `get_emas` | `symbol`, `interval="1h"`, `limit=500` | 20/50/200 EMAs + trend-stack label (bullish/bearish/mixed/n/a) |
| `get_volume_breakdown` | `symbol` | 24h USD volume: Binance spot + Coinbase + cross-exchange aggregate (via CoinGecko) with shares, **explicit cross-venue perp volume (Binance / Bybit / Hyperliquid)**, **perp/spot ratio + leverage-led vs spot-led read**, `thin` flag — US/institutional divergence on majors, true total on thin alts |
| `get_vwap` | `symbol`, `interval="5m"`, `limit=288` | Session VWAP (resets at 00:00 UTC) + window VWAP, each with 1σ/2σ bands, plus a long/short/neutral bias vs current close |
| `get_atr` | `symbol`, `interval="30m"`, `limit=100`, `period=14`, `account_equity=None`, `risk_pct=1.0`, `stop_atr_mult=1.5` | ATR(14) and example 1×/1.5× ATR stop distances for long and short. Pass `account_equity` to also get an ATR-normalized **position size** (risk `risk_pct`% across a `stop_atr_mult`×ATR stop) — for stop placement and sizing, not direction |
| `get_volume_profile` | `symbol`, `interval="15m"`, `limit=192`, `bins=24` | POC, value area (70%), top 5 high-volume nodes, and whether the current close sits inside the value area |
| `get_correlation` | `symbol`, `interval="1h"`, `limit=200`, `reference=None` (auto: BTCUSDT/SPY) | Rolling **correlation + beta** of a symbol vs its reference, a recent-half correlation with a **decoupling** flag, and a gating read (high corr → the reference's regime sets direction, symbol specifics pick entries/levels; low → symbol-specific edge valid) |
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
       "market-data-mcp": {
         "command": "/absolute/path/to/binancedatapuller/.venv/bin/python",
         "args": ["/absolute/path/to/binancedatapuller/mcp_server.py"]
       }
     }
   }
   ```

   If the file already has other servers, add `"market-data-mcp"` as another key inside the
   existing `mcpServers` object rather than replacing it.

3. **Fully quit and reopen Claude Desktop** (the config is read on startup).

4. The tools appear under the tools/plug icon in the chat input. Then just ask, e.g.:
   - "Check the funding rate and order-book imbalance for ETHUSDT."
   - "Pull 4h candles and the ADX for BTCUSDT and tell me the trend."

   Claude calls the relevant tools and reasons over the fresh data it gets back.

### Use with Claude Code (alternative)

```bash
claude mcp add market-data-mcp -- \
  /absolute/path/to/binancedatapuller/.venv/bin/python \
  /absolute/path/to/binancedatapuller/mcp_server.py
```

## Notes

- **Two data paths:** REST is fetched on demand per tool call (the cold path — every tool
  answers immediately); the WebSocket tape is a hot path that warms on first use and upgrades
  flow tools to trade-level data. Raw ticks stay inside the process — tools return aggregates
  only, so streaming adds no LLM token overhead.
- Stream connections auto-reconnect with exponential backoff and resubscribe; a symbol's
  subscription is LRU-evicted past the per-venue budget.
- The perp trade tape streams from **Bybit** rather than Binance futures: Binance's fstream
  data plane is silently filtered on some networks (handshake succeeds, no frames), and the
  aggressor-flow signal is equivalent.
- The most recent candle may be in progress (incomplete), same as the raw exchange data.
- Binance public endpoints are rate-limited; on-demand tool calls stay well within limits.
- Crypto data is from **keyless public APIs**. Two signal classes remain out of scope:
  - **Liquidations** — not exposed by keyless public REST; now feasible via the stream layer
    (Bybit publishes liquidations over WebSocket) — see roadmap. Cascade *risk* is meanwhile
    readable from the OI quadrant, funding extremes, and ATR.
  - **On-chain flows** (exchange netflows, stablecoin/whale activity) — require a paid/keyed
    provider (Glassnode/CryptoQuant/Nansen) with no price-derivable proxy.

## Roadmap

- **L2 order-book replica** for crypto via the exchange diff-depth protocol (snapshot +
  buffered-delta sync) — true depth dynamics instead of REST snapshots.
- **Liquidation stream** (Bybit WS `allLiquidation`) as volatility/cascade context.
- **Market-calendar awareness** — distinguish "market closed" from "stream stale" for
  equities (currently reported via `data_age_s`).
- **Signal-calibration harness** — persist tool verdicts (confluence, regime, pattern
  confirmations) and score them against forward returns to tune thresholds on data.
