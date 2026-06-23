"""
Shared HTTP plumbing for all data providers. No business logic, no formatting.

A single pooled `SESSION` is reused across every provider so connections are kept
alive and transient errors (429 / 5xx) retry with exponential backoff. status=400
is intentionally NOT retried — Binance fetchers use it as a spot→futures fallback
signal.
"""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

TIMEOUT = 10

SESSION = requests.Session()
_retry = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset({"GET", "POST"}),
    raise_on_status=False,
)
_adapter = HTTPAdapter(max_retries=_retry)
SESSION.mount("https://", _adapter)
SESSION.mount("http://", _adapter)
