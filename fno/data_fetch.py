"""Fetch OHLCV and NSE options chain data."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from fno.config import INTRADAY_OHLCV_PERIOD
from fno.nse_session import get_nse_session, reset_nse_session

logger = logging.getLogger(__name__)

# NSE index symbols supported by option-chain-v3
_NSE_INDEX_OPTIONS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}


def fetch_ohlcv(
    symbol: str,
    *,
    period: str = INTRADAY_OHLCV_PERIOD,
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


def fetch_intraday_bars(symbol: str) -> Optional[pd.DataFrame]:
    """Fetch 15m bars only (intraday F&O path — avoids slow 1h/1d downloads)."""
    return fetch_ohlcv(symbol, period=INTRADAY_OHLCV_PERIOD, interval="15m")


def fetch_multi_interval(symbol: str) -> Dict[str, pd.DataFrame]:
    """Fetch 15m, 1h, and daily bars for a symbol."""
    return {
        "15m": fetch_ohlcv(symbol, period=INTRADAY_OHLCV_PERIOD, interval="15m"),
        "1h": fetch_ohlcv(symbol, period="60d", interval="1h"),
        "1d": fetch_ohlcv(symbol, period="1y", interval="1d"),
    }


def _fetch_option_chain_v3(symbol: str) -> Optional[dict]:
    """Fetch option chain via NSE v3 API (reliable vs legacy option-chain-indices)."""
    sym = symbol.upper()
    if sym not in _NSE_INDEX_OPTIONS:
        return None

    for attempt in range(2):
        try:
            session = get_nse_session()
            info_resp = session.get(
                f"https://www.nseindia.com/api/option-chain-contract-info?symbol={sym}",
                timeout=12,
            )
            if not info_resp.ok:
                logger.warning("NSE contract-info %s: HTTP %s", sym, info_resp.status_code)
                if attempt == 0:
                    reset_nse_session()
                    continue
                return None

            expiries: List[str] = info_resp.json().get("expiryDates") or []
            if not expiries:
                logger.warning("NSE contract-info %s: no expiries", sym)
                return None

            expiry = expiries[0]
            chain_resp = session.get(
                "https://www.nseindia.com/api/option-chain-v3",
                params={"type": "Indices", "symbol": sym, "expiry": expiry},
                timeout=15,
            )
            if not chain_resp.ok:
                logger.warning("NSE option-chain-v3 %s: HTTP %s", sym, chain_resp.status_code)
                if attempt == 0:
                    reset_nse_session()
                    continue
                return None

            payload = chain_resp.json()
            records = payload.get("records") or {}
            if not records.get("data"):
                return None

            return {
                "data": records["data"],
                "underlyingValue": records.get("underlyingValue"),
                "records": records,
                "expiry": expiry,
            }
        except Exception as exc:
            logger.warning("NSE v3 option chain failed for %s (attempt %d): %s", sym, attempt + 1, exc)
            if attempt == 0:
                reset_nse_session()

    return None


def fetch_options_chain(symbol: str) -> Optional[dict]:
    """Fetch live NSE option chain for NIFTY / BANKNIFTY (Sensex: BSE — not on NSE v3)."""
    payload = _fetch_option_chain_v3(symbol)
    if payload:
        return payload

    # Fallback: nsepython legacy scraper
    try:
        from nsepython import nse_optionchain_scrapper

        legacy = nse_optionchain_scrapper(symbol)
        if legacy and legacy.get("records", {}).get("data"):
            records = legacy["records"]
            return {
                "data": records["data"],
                "underlyingValue": records.get("underlyingValue"),
                "records": records,
            }
    except Exception as exc:
        logger.warning("nsepython fallback failed for %s: %s", symbol, exc)

    return None


def option_chain_to_dataframe(payload: dict) -> pd.DataFrame:
    """Flatten NSE option chain payload to a DataFrame (nearest expiry rows)."""
    if not payload:
        return pd.DataFrame()

    entries = payload.get("data")
    if not entries and payload.get("records"):
        entries = payload["records"].get("data")

    if not entries:
        return pd.DataFrame()

    # If multiple expiries in payload, keep nearest (first listed) expiry only
    expiry_filter = payload.get("expiry")
    if not expiry_filter and entries:
        expiry_filter = entries[0].get("expiryDate")

    rows = []
    for entry in entries:
        if expiry_filter and entry.get("expiryDate") not in (None, expiry_filter):
            continue
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
                "expiry": entry.get("expiryDate") or expiry_filter,
            }
        )
    return pd.DataFrame(rows)
