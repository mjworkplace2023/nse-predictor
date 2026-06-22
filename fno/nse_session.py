"""Shared NSE session for option-chain API (v3)."""

from __future__ import annotations

import logging
import threading
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_NSE_HEADERS = {
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
    ),
    "accept": "application/json, text/plain, */*",
    "accept-language": "en-US,en;q=0.9",
    "referer": "https://www.nseindia.com/option-chain",
}

_lock = threading.Lock()
_session: Optional[requests.Session] = None


def get_nse_session() -> requests.Session:
    """Return a warmed-up requests session (cookies from option-chain page)."""
    global _session
    with _lock:
        if _session is None:
            s = requests.Session()
            s.headers.update(_NSE_HEADERS)
            try:
                s.get("https://www.nseindia.com/option-chain", timeout=12)
            except Exception as exc:
                logger.warning("NSE session warmup failed: %s", exc)
            _session = s
        return _session


def reset_nse_session() -> None:
    global _session
    with _lock:
        _session = None
