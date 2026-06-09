"""
Fetches historical OHLCV data for NSE stocks using yfinance.
All data is in-memory — no database or disk cache.
"""

import logging
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_stock_data(
    symbol: str,
    period: str = "3mo",
    interval: str = "1d",
) -> Optional[pd.DataFrame]:
    """
    Download OHLCV data for a single NSE symbol.

    Args:
        symbol:   yfinance ticker, e.g. "RELIANCE.NS"
        period:   lookback window ("1mo", "3mo", "6mo", "1y")
        interval: bar size ("1d", "1wk")

    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume]
        or None if the download fails / returns empty data.
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)

        if df.empty:
            logger.warning("No data returned for %s", symbol)
            return None

        # Keep only the columns we care about
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df.dropna(subset=["Close"], inplace=True)

        logger.debug("Fetched %d rows for %s", len(df), symbol)
        return df

    except Exception as exc:
        logger.error("Error fetching %s: %s", symbol, exc)
        return None


def fetch_all_stocks(
    symbols: list[str],
    period: str = "3mo",
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """
    Download data for a list of symbols.

    Returns:
        dict mapping symbol -> DataFrame (only successful downloads).
    """
    results: Dict[str, pd.DataFrame] = {}
    total = len(symbols)

    for idx, symbol in enumerate(symbols, start=1):
        logger.info("Fetching [%d/%d] %s", idx, total, symbol)
        df = fetch_stock_data(symbol, period=period, interval=interval)
        if df is not None and not df.empty:
            results[symbol] = df

    logger.info("Successfully fetched data for %d / %d symbols", len(results), total)
    return results


def get_latest_price(df: pd.DataFrame) -> float:
    """Return the most recent closing price from a stock DataFrame."""
    return float(df["Close"].iloc[-1])


def get_price_change_pct(df: pd.DataFrame, days: int = 1) -> float:
    """
    Compute percentage price change over the last `days` trading sessions.

    Returns value as a percentage, e.g. 2.5 means +2.5 %.
    """
    if len(df) < days + 1:
        return 0.0
    close = df["Close"]
    pct = (close.iloc[-1] - close.iloc[-(days + 1)]) / close.iloc[-(days + 1)] * 100
    return round(float(pct), 4)
