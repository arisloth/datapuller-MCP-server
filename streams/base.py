"""
Shared async WebSocket plumbing for all stream adapters. No business logic.

StreamClient owns one connection and its lifecycle: connect → on_open (auth +
resubscribe) → read loop dispatching each decoded frame to handle(). Any drop
or idle timeout reconnects with exponential backoff, and on_open re-runs so
subscriptions survive reconnects. handle() is a plain sync method taking a
decoded frame, so adapters are testable without a socket.
"""
import asyncio
import json

import websockets

IDLE_TIMEOUT_S = 90       # no frame for this long → force reconnect
BACKOFF_BASE_S = 1.0
BACKOFF_MAX_S = 60.0
APP_PING_INTERVAL_S = 20  # cadence for venue-level (JSON) keepalive pings


class StreamClient:
    # Venues like Bybit require an application-level ping payload on top of
    # protocol pings; set this in a subclass to enable it.
    app_ping: dict | None = None

    def __init__(self, url: str):
        self.url = url
        self._ws = None
        self._stopped = False

    # --- adapter hooks ------------------------------------------------------

    async def on_open(self, ws) -> None:
        """Auth and (re)subscribe. Called after every (re)connect."""

    def handle(self, msg) -> None:
        """Normalize one decoded frame into the store."""

    # --- plumbing -------------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._ws is not None

    async def send(self, payload: dict) -> None:
        if self._ws is not None:
            await self._ws.send(json.dumps(payload))

    async def _app_ping_loop(self) -> None:
        while True:
            await asyncio.sleep(APP_PING_INTERVAL_S)
            await self.send(self.app_ping)

    async def run(self) -> None:
        """Connect-and-read forever, reconnecting with exponential backoff."""
        backoff = BACKOFF_BASE_S
        while not self._stopped:
            ping_task = None
            try:
                async with websockets.connect(self.url, ping_interval=20, ping_timeout=20) as ws:
                    self._ws = ws
                    await self.on_open(ws)
                    if self.app_ping is not None:
                        ping_task = asyncio.create_task(self._app_ping_loop())
                    backoff = BACKOFF_BASE_S
                    while True:
                        raw = await asyncio.wait_for(ws.recv(), timeout=IDLE_TIMEOUT_S)
                        self.handle(json.loads(raw))
            except asyncio.CancelledError:
                raise
            except Exception:
                pass  # any drop (network, idle timeout, bad frame) → reconnect
            finally:
                if ping_task is not None:
                    ping_task.cancel()
                self._ws = None
            if not self._stopped:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, BACKOFF_MAX_S)

    def stop(self) -> None:
        self._stopped = True
