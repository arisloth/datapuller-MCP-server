"""
In-memory tape store for streamed market data.

Stream adapters (streams/) ingest normalized events here; MCP tools read
AGGREGATES out (CVD, taker ratio, quote snapshot) — raw trades never cross a
tool boundary, so streaming adds no token cost to the model.

Tapes are keyed by venue-qualified symbol, e.g. "PERP:BTCUSDT", "SPOT:BTCUSDT",
"IEX:AAPL", so one base symbol can carry multiple tapes (spot-vs-perp split).

Thread model: adapters write from the stream thread's event loop; MCP tools
read from the server thread. Every tape mutation/read holds that tape's lock.
"""
import threading
import time
from collections import OrderedDict, deque

TRADE_BUFFER = 50_000   # trades kept per tape (deque ring buffer)
WARM_MIN_TRADES = 20    # tape reports warm once it has this many trades
DEFAULT_BUDGET = 30     # max live subscriptions per venue (Alpaca IEX hard limit)


def now_ms() -> int:
    return int(time.time() * 1000)


# --- Equity trade classification (Lee & Ready 1991) -------------------------

# Condition codes whose prints carry no *current* aggressor intent — late or
# out-of-sequence reports, official open/close reprints, derivatively priced
# and average-price trades. Excluded from flow aggregation (they still refresh
# the tape's staleness clock). Regular prints (e.g. "@", "I" odd lot) count.
EXCLUDED_CONDITIONS = {
    "C",  # cash sale (non-standard settlement)
    "G",  # bunched sold (late report)
    "H",  # price variation trade
    "M",  # market center official close (reprint)
    "N",  # next-day settlement
    "P",  # prior reference price
    "Q",  # market center official open (reprint)
    "W",  # average price trade
    "Z",  # sold (out of sequence)
    "4",  # derivatively priced
    "7",  # qualified contingent trade
    "9",  # corrected consolidated close
}


def is_flow_eligible(conditions) -> bool:
    """Whether a print (list of condition codes, may be None) counts for flow."""
    return not conditions or not (set(conditions) & EXCLUDED_CONDITIONS)


def classify_trade(price, bid, ask, prev_price=None, last_tick=None) -> str | None:
    """Lee–Ready (1991) aggressor-side inference for venues whose prints carry
    no taker flag: above the quote midpoint → buyer-initiated, below → seller-
    initiated; at the midpoint (or with no valid quote) fall back to the tick
    test — uptick buy, downtick sell, zero-tick inherits the last tick
    direction. None when unclassifiable."""
    if bid is not None and ask is not None and 0 < bid <= ask:
        mid = (bid + ask) / 2
        if price > mid:
            return "buy"
        if price < mid:
            return "sell"
    if prev_price is not None:
        if price > prev_price:
            return "buy"
        if price < prev_price:
            return "sell"
    if last_tick == "up":
        return "buy"
    if last_tick == "down":
        return "sell"
    return None


class SymbolTape:
    """One venue-qualified symbol's rolling tape: trades + latest top-of-book."""

    def __init__(self, maxlen: int = TRADE_BUFFER):
        self.lock = threading.Lock()
        # (ts_ms, price, size, side) — side is "buy"/"sell" (aggressor) or None.
        self.trades: deque = deque(maxlen=maxlen)
        self.bid = self.ask = None
        self.bid_size = self.ask_size = None
        self.quote_ts_ms = None
        self.last_event_ms = None
        # tick-test state for Lee–Ready (equities)
        self.last_trade_price = None
        self.last_tick = None


class TapeStore:
    """Get-or-create tapes by key; all public reads return plain aggregates."""

    def __init__(self):
        self._tapes: dict[str, SymbolTape] = {}
        self._lock = threading.Lock()

    def tape(self, key: str) -> SymbolTape:
        with self._lock:
            t = self._tapes.get(key)
            if t is None:
                t = self._tapes[key] = SymbolTape()
            return t

    # --- ingestion (called from stream adapters) ---------------------------

    def ingest_trade(self, key: str, ts_ms: int, price: float, size: float,
                     side: str | None) -> None:
        t = self.tape(key)
        with t.lock:
            t.trades.append((ts_ms, price, size, side))
            t.last_event_ms = ts_ms

    def ingest_equity_trade(self, key: str, ts_ms: int, price: float, size: float,
                            conditions=None) -> None:
        """Classify the print via Lee–Ready against the tape's prevailing NBBO,
        then ingest. Ineligible prints (EXCLUDED_CONDITIONS) refresh the
        staleness clock but don't enter the flow tape or tick state."""
        t = self.tape(key)
        with t.lock:
            if not is_flow_eligible(conditions):
                t.last_event_ms = ts_ms
                return
            side = classify_trade(price, t.bid, t.ask, t.last_trade_price, t.last_tick)
            if t.last_trade_price is not None:
                if price > t.last_trade_price:
                    t.last_tick = "up"
                elif price < t.last_trade_price:
                    t.last_tick = "down"
            t.last_trade_price = price
            t.trades.append((ts_ms, price, size, side))
            t.last_event_ms = ts_ms

    def ingest_quote(self, key: str, ts_ms: int, bid: float, ask: float,
                     bid_size: float | None = None, ask_size: float | None = None) -> None:
        t = self.tape(key)
        with t.lock:
            t.bid, t.ask = bid, ask
            t.bid_size, t.ask_size = bid_size, ask_size
            t.quote_ts_ms = ts_ms
            t.last_event_ms = ts_ms

    # --- aggregates (called from MCP tools) --------------------------------

    def flow(self, key: str, window_s: float | None = None,
             _now_ms: int | None = None) -> dict | None:
        """Signed order flow over the trailing window (whole tape if None):
        {cvd, taker_ratio, buy_vol, sell_vol, n_trades, from_ms, to_ms}.
        Unsided trades are excluded. None when no sided trades are in window.
        Assumes trades were ingested in chronological order (live-stream tape) —
        the window scan walks backward and stops at the first out-of-window ts."""
        t = self.tape(key)
        cutoff = None
        if window_s is not None:
            cutoff = (_now_ms if _now_ms is not None else now_ms()) - window_s * 1000
        buy = sell = 0.0
        n = 0
        first_ts = last_ts = None
        with t.lock:
            for ts, _price, size, side in reversed(t.trades):
                if cutoff is not None and ts < cutoff:
                    break
                if side == "buy":
                    buy += size
                elif side == "sell":
                    sell += size
                else:
                    continue
                n += 1
                first_ts = ts
                last_ts = last_ts or ts
        if n == 0:
            return None
        return {
            "cvd": buy - sell,
            "taker_ratio": (buy / sell) if sell > 0 else None,
            "buy_vol": buy,
            "sell_vol": sell,
            "n_trades": n,
            "from_ms": first_ts,
            "to_ms": last_ts,
        }

    def cvd(self, key: str, window_s: float | None = None) -> float | None:
        f = self.flow(key, window_s)
        return None if f is None else f["cvd"]

    def taker_ratio(self, key: str, window_s: float | None = None) -> float | None:
        f = self.flow(key, window_s)
        return None if f is None else f["taker_ratio"]

    def quote(self, key: str) -> dict | None:
        """Latest top-of-book: {bid, ask, bid_size, ask_size, spread, spread_pct,
        mid, ts_ms}. None until a quote has arrived."""
        t = self.tape(key)
        with t.lock:
            if t.bid is None or t.ask is None:
                return None
            mid = (t.bid + t.ask) / 2
            spread = t.ask - t.bid
            return {
                "bid": t.bid, "ask": t.ask,
                "bid_size": t.bid_size, "ask_size": t.ask_size,
                "spread": spread,
                "spread_pct": (spread / mid * 100) if mid else None,
                "mid": mid,
                "ts_ms": t.quote_ts_ms,
            }

    def age_s(self, key: str, _now_ms: int | None = None) -> float | None:
        """Seconds since the last event on this tape. None if never touched.
        Clamped at 0 — venue timestamps can run slightly ahead of local time."""
        t = self.tape(key)
        with t.lock:
            if t.last_event_ms is None:
                return None
            age = ((_now_ms if _now_ms is not None else now_ms()) - t.last_event_ms) / 1000
            return max(0.0, age)

    def is_warm(self, key: str) -> bool:
        """Enough trades to aggregate meaningfully. Freshness is reported
        separately via age_s — a closed market makes a tape old, not invalid."""
        t = self.tape(key)
        with t.lock:
            return len(t.trades) >= WARM_MIN_TRADES


class SubscriptionBudget:
    """LRU registry of live subscriptions, capped at `budget` per instance
    (one instance per venue). touch() marks use and returns any evicted keys
    so the caller can unsubscribe them upstream."""

    def __init__(self, budget: int = DEFAULT_BUDGET):
        self.budget = budget
        self._lock = threading.Lock()
        self._order: OrderedDict[str, None] = OrderedDict()

    def touch(self, key: str) -> list[str]:
        with self._lock:
            if key in self._order:
                self._order.move_to_end(key)
                return []
            self._order[key] = None
            evicted = []
            while len(self._order) > self.budget:
                evicted.append(self._order.popitem(last=False)[0])
            return evicted

    def drop(self, key: str) -> None:
        with self._lock:
            self._order.pop(key, None)

    def active(self) -> list[str]:
        with self._lock:
            return list(self._order)


# Module-level singleton shared by stream adapters and MCP tools.
STORE = TapeStore()
