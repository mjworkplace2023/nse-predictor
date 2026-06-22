"""Display formatting for F&O index values."""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Tuple

import pytz

IST = pytz.timezone("Asia/Kolkata")


def round_index(value: Optional[float]) -> Optional[int]:
    """Round index price/level to whole number (no decimals)."""
    if value is None:
        return None
    return int(round(value))


def format_index(value: Optional[float]) -> str:
    """Format index value for display — no decimals."""
    if value is None:
        return "—"
    return f"{int(round(value)):,}"


def parse_expiry(expiry_str: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Parse NSE expiry string (e.g. 23-Jun-2026).

    Returns (expiry_date, weekday_name, days_to_expiry).
    """
    if not expiry_str:
        return None, None, None
    try:
        expiry_dt = datetime.strptime(expiry_str.strip(), "%d-%b-%Y").date()
        today = datetime.now(IST).date()
        return expiry_str, expiry_dt.strftime("%A"), (expiry_dt - today).days
    except ValueError:
        return expiry_str, None, None
