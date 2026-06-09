"""
Technical indicator calculations using 'ta' library (works on Python 3.14).

Indicators:
  - RSI (14)
  - MACD (12, 26, 9)
  - EMA 20 and EMA 50
  - Bollinger Bands (20, 2)
  - Volume ratio (5-day vs 20-day)

Each function returns a score contribution in [-1, +1].
`compute_technical_score` aggregates them into [-40, +40].
"""

import logging
import numpy as np
import pandas as pd
import ta as ta_lib

logger = logging.getLogger(__name__)


def score_rsi(df: pd.DataFrame) -> tuple:
    """RSI score + raw value. Returns (score[-1,+1], rsi_value)."""
    try:
        rsi_series = ta_lib.momentum.RSIIndicator(df["Close"], window=14).rsi()
        rsi_series = rsi_series.dropna()
        if rsi_series.empty:
            return 0.0, None
        rsi = float(rsi_series.iloc[-1])
        if rsi <= 30:
            score = 1.0
        elif rsi >= 70:
            score = -1.0
        else:
            score = round((50 - rsi) / 20, 4)
        return score, rsi
    except Exception as exc:
        logger.debug("RSI error: %s", exc)
        return 0.0, None


def score_macd(df: pd.DataFrame) -> float:
    """MACD histogram score in [-1, +1]."""
    try:
        macd_ind = ta_lib.trend.MACD(df["Close"], window_fast=12, window_slow=26, window_sign=9)
        hist = macd_ind.macd_diff().dropna()
        if hist.empty:
            return 0.0
        latest = float(hist.iloc[-1])
        prev   = float(hist.iloc[-2]) if len(hist) > 1 else 0.0
        sign_score = 1.0 if latest > 0 else -1.0
        momentum   = 0.5 if (latest > prev and latest > 0) or (latest < prev and latest < 0) else 0.0
        return round(sign_score * 0.5 + momentum, 4)
    except Exception as exc:
        logger.debug("MACD error: %s", exc)
        return 0.0


def score_ema(df: pd.DataFrame) -> float:
    """EMA 20/50 crossover score in [-1, +1]."""
    try:
        ema20 = ta_lib.trend.EMAIndicator(df["Close"], window=20).ema_indicator().dropna()
        ema50 = ta_lib.trend.EMAIndicator(df["Close"], window=50).ema_indicator().dropna()
        if ema20.empty or ema50.empty:
            return 0.0
        price = float(df["Close"].iloc[-1])
        e20   = float(ema20.iloc[-1])
        e50   = float(ema50.iloc[-1])
        score = 0.0
        score += 0.5 if price > e20 else -0.5
        score += 0.5 if e20 > e50   else -0.5
        return round(score, 4)
    except Exception as exc:
        logger.debug("EMA error: %s", exc)
        return 0.0


def score_bollinger(df: pd.DataFrame) -> float:
    """Bollinger Band position score in [-1, +1]."""
    try:
        bb = ta_lib.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
        upper = bb.bollinger_hband().dropna()
        lower = bb.bollinger_lband().dropna()
        if upper.empty or lower.empty:
            return 0.0
        price      = float(df["Close"].iloc[-1])
        band_width = float(upper.iloc[-1]) - float(lower.iloc[-1])
        if band_width == 0:
            return 0.0
        position = (price - float(lower.iloc[-1])) / band_width
        return round(max(-1.0, min(1.0, 1.0 - 2.0 * position)), 4)
    except Exception as exc:
        logger.debug("BB error: %s", exc)
        return 0.0


def score_volume(df: pd.DataFrame) -> float:
    """Volume ratio (5-day avg / 20-day avg) score in [-1, +1]."""
    try:
        vol  = df["Volume"]
        avg5  = float(vol.rolling(5).mean().iloc[-1])
        avg20 = float(vol.rolling(20).mean().iloc[-1])
        if avg20 == 0:
            return 0.0
        ratio = avg5 / avg20
        if ratio >= 1.5:
            return 1.0
        elif ratio <= 0.5:
            return -1.0
        else:
            return round((ratio - 1.0) * 2.0, 4)
    except Exception as exc:
        logger.debug("Volume error: %s", exc)
        return 0.0


# Weights — total max = 40
WEIGHTS = {"rsi": 10, "macd": 10, "ema": 10, "bollinger": 5, "volume": 5}


def compute_technical_score(df: pd.DataFrame) -> dict:
    """
    Aggregate all indicator scores into a single dict.

    Returns:
        { "total": float[-40,+40], "rsi", "macd", "ema",
          "bollinger", "volume", "rsi_value" }
    """
    rsi_raw, rsi_value = score_rsi(df)
    macd_raw  = score_macd(df)
    ema_raw   = score_ema(df)
    bb_raw    = score_bollinger(df)
    vol_raw   = score_volume(df)

    total = (
        rsi_raw  * WEIGHTS["rsi"]
        + macd_raw  * WEIGHTS["macd"]
        + ema_raw   * WEIGHTS["ema"]
        + bb_raw    * WEIGHTS["bollinger"]
        + vol_raw   * WEIGHTS["volume"]
    )

    return {
        "total":     round(total, 2),
        "rsi":       rsi_raw,
        "macd":      macd_raw,
        "ema":       ema_raw,
        "bollinger": bb_raw,
        "volume":    vol_raw,
        "rsi_value": rsi_value,
    }
