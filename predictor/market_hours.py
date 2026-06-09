"""NSE market hours helpers (Asia/Kolkata)."""

import datetime

import pytz

IST = pytz.timezone("Asia/Kolkata")
MARKET_OPEN = datetime.time(9, 15)
MARKET_CLOSE = datetime.time(15, 30)


def now_ist() -> datetime.datetime:
    return datetime.datetime.now(IST)


def is_market_open(at: datetime.datetime | None = None) -> bool:
    """Return True during NSE regular session on weekdays."""
    current = at.astimezone(IST) if at is not None else now_ist()
    if current.weekday() >= 5:
        return False
    return MARKET_OPEN <= current.time() <= MARKET_CLOSE


def market_status_message() -> str:
    if is_market_open():
        return "🟢 Market open (09:15–15:30 IST)"
    current = now_ist()
    if current.weekday() >= 5:
        return "🔴 Market closed (weekend)"
    if current.time() < MARKET_OPEN:
        return f"🟡 Pre-market — opens at 09:15 IST"
    return "🔴 Market closed for today"
