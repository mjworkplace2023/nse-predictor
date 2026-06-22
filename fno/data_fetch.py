"""Fetch OHLCV and NSE options chain data."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_ohlcv(
    symbol: str,
    *,
    period: str = "60d",
    interval: str = "15m",
) -> Optional[pd.DataFrame]:
    """Download OHLCV via yfinance."""
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
        if df is None or df.empty:
            return None
        out = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        out.index = pd.to_datetime(out.index)
        out.sort_index(inplace=True)
        out.dropna(subset=["Close"], inplace=True)
        return out
    except Exception as exc:
        logger.warning("OHLCV fetch failed for %s (%s): %s", symbol, interval, exc)
        return None


def fetch_multi_interval(symbol: str) -> Dict[str, pd.DataFrame]:
    """Fetch 15m, 1h, and daily bars for a symbol."""
    return {
        "15m": fetch_ohlcv(symbol, period="60d", interval="15m"),
        "1h": fetch_ohlcv(symbol, period="730d", interval="1h"),
        "1d": fetch_ohlcv(symbol, period="2y", interval="1d"),
    }


def fetch_options_chain(symbol: str) -> Optional[dict]:
    """Fetch live NSE option chain via nsepython (NIFTY, BANKNIFTY, SENSEX)."""
    try:
        from nsepython import nse_optionchain_scrapper

        payload = nse_optionchain_scrapper(symbol)
        if not payload or "data" not in payload:
            return None
        return payload
    except Exception as exc:
        logger.warning("nsepython option chain failed for %s: %s", symbol, exc)
        return None


def option_chain_to_dataframe(payload: dict) -> pd.DataFrame:
    """Flatten NSE option chain payload to a DataFrame."""
    rows = []
    for entry in payload.get("data", []):
        strike = entry.get("strikePrice")
        ce = entry.get("CE") or {}
        pe = entry.get("PE") or {}
        rows.append(
            {
                "strike": strike,
                "ce_oi": ce.get("openInterest", 0) or 0,
                "pe_oi": pe.get("openInterest", 0) or 0,
                "ce_iv": ce.get("impliedVolatility", 0) or 0,
                "pe_iv": pe.get("impliedVolatility", 0) or 0,
                "ce_ltp": ce.get("lastPrice", 0) or 0,
                "pe_ltp": pe.get("lastPrice", 0) or 0,
                "expiry": ce.get("expiryDate") or pe.get("expiryDate"),
            }
        )
    return pd.DataFrame(rows)
