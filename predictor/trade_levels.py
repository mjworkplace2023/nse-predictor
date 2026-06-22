"""
Rule-based intraday entry range, target, and stop-loss levels.

Derived from session VWAP, session high/low, and ATR on intraday candles.
These are estimated technical levels — NOT guaranteed trade outcomes.
"""

import logging
from typing import Optional

import pandas as pd
import ta as ta_lib

logger = logging.getLogger(__name__)

ATR_PERIOD = 14
ATR_TARGET_MULT = 2.0
ATR_STOP_MULT = 1.0

# Swing (daily) uses wider ATR multiples — multi-day holds
SWING_LOOKBACK_DAYS = 20
SWING_TARGET_MULT = 2.5
SWING_STOP_MULT = 1.25


def _session_vwap(session_df: pd.DataFrame) -> float:
    if session_df.empty or session_df["Volume"].sum() == 0:
        return float(session_df["Close"].iloc[-1])
    typical = (session_df["High"] + session_df["Low"] + session_df["Close"]) / 3
    return float((typical * session_df["Volume"]).sum() / session_df["Volume"].sum())


def _compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    try:
        atr_series = ta_lib.volatility.AverageTrueRange(
            df["High"], df["Low"], df["Close"], window=period
        ).average_true_range()
        atr_series = atr_series.dropna()
        if atr_series.empty:
            return 0.0
        return float(atr_series.iloc[-1])
    except Exception as exc:
        logger.debug("ATR error: %s", exc)
        return 0.0


def _round2(val: float) -> float:
    return round(val, 2)


def compute_intraday_trade_levels(
    session_df: pd.DataFrame,
    full_df: pd.DataFrame,
    price: float,
    signal: str,
) -> dict:
    """
    Compute intraday trade plan from session structure.

    Returns:
        {
          "trade_action": "LONG" | "SHORT" | "WAIT",
          "entry_low": float | None,
          "entry_high": float | None,
          "target_price": float | None,
          "stop_loss": float | None,
          "risk_reward": float | None,
        }
    """
    empty = {
        "trade_action": "WAIT",
        "entry_low": None,
        "entry_high": None,
        "target_price": None,
        "stop_loss": None,
        "risk_reward": None,
    }

    if session_df.empty or price <= 0:
        return empty

    session_high = float(session_df["High"].max())
    session_low = float(session_df["Low"].min())
    vwap = _session_vwap(session_df)
    atr = _compute_atr(full_df if len(full_df) >= ATR_PERIOD else session_df)

    if atr <= 0:
        atr = price * 0.005  # fallback: 0.5% of price

    is_long = signal in ("BUY", "STRONG BUY")
    is_short = signal in ("SELL", "STRONG SELL")

    if not is_long and not is_short:
        return {
            **empty,
            "entry_low": _round2(session_low),
            "entry_high": _round2(session_high),
        }

    if is_long:
        # Buy zone: pullback toward VWAP, up to current price
        entry_low = _round2(min(vwap, price - 0.5 * atr))
        entry_high = _round2(price)
        if entry_low >= entry_high:
            entry_low = _round2(price - 0.5 * atr)

        target = _round2(max(price + ATR_TARGET_MULT * atr, session_high))
        stop_loss = _round2(min(price - ATR_STOP_MULT * atr, session_low - 0.05 * atr))
        if stop_loss >= entry_low:
            stop_loss = _round2(entry_low - 0.5 * atr)

        risk = price - stop_loss
        reward = target - price
        rr = _round2(reward / risk) if risk > 0 else None

        return {
            "trade_action": "LONG",
            "entry_low": entry_low,
            "entry_high": entry_high,
            "target_price": target,
            "stop_loss": stop_loss,
            "risk_reward": rr,
        }

    # SHORT
    entry_low = _round2(price)
    entry_high = _round2(max(vwap, price + 0.5 * atr))
    if entry_high <= entry_low:
        entry_high = _round2(price + 0.5 * atr)

    target = _round2(min(price - ATR_TARGET_MULT * atr, session_low))
    stop_loss = _round2(max(price + ATR_STOP_MULT * atr, session_high + 0.05 * atr))
    if stop_loss <= entry_high:
        stop_loss = _round2(entry_high + 0.5 * atr)

    risk = stop_loss - price
    reward = price - target
    rr = _round2(reward / risk) if risk > 0 else None

    return {
        "trade_action": "SHORT",
        "entry_low": entry_low,
        "entry_high": entry_high,
        "target_price": target,
        "stop_loss": stop_loss,
        "risk_reward": rr,
    }


def format_entry_range(entry_low: Optional[float], entry_high: Optional[float]) -> str:
    if entry_low is None or entry_high is None:
        return "—"
    if abs(entry_low - entry_high) < 0.01:
        return f"₹{entry_low:,.2f}"
    return f"₹{entry_low:,.2f} – ₹{entry_high:,.2f}"


def compute_daily_trade_levels(
    df: pd.DataFrame,
    price: float,
    signal: str,
) -> dict:
    """
    Compute swing trade plan from daily OHLCV (multi-day hold).

    Uses 20-day high/low structure and daily ATR — wider targets than intraday.
    """
    empty = {
        "trade_action": "WAIT",
        "entry_low": None,
        "entry_high": None,
        "target_price": None,
        "stop_loss": None,
        "risk_reward": None,
    }

    if df.empty or price <= 0:
        return empty

    lookback = min(SWING_LOOKBACK_DAYS, len(df))
    recent = df.tail(lookback)
    swing_high = float(recent["High"].max())
    swing_low = float(recent["Low"].min())
    atr = _compute_atr(df)
    if atr <= 0:
        atr = price * 0.02

    is_long = signal in ("BUY", "STRONG BUY")
    is_short = signal in ("SELL", "STRONG SELL")

    if not is_long and not is_short:
        return {
            **empty,
            "entry_low": _round2(swing_low),
            "entry_high": _round2(swing_high),
        }

    if is_long:
        entry_low = _round2(min(price - 0.75 * atr, swing_low + 0.25 * atr))
        entry_high = _round2(price)
        if entry_low >= entry_high:
            entry_low = _round2(price - 0.5 * atr)

        target = _round2(max(price + SWING_TARGET_MULT * atr, swing_high))
        stop_loss = _round2(min(price - SWING_STOP_MULT * atr, swing_low - 0.1 * atr))
        if stop_loss >= entry_low:
            stop_loss = _round2(entry_low - 0.5 * atr)

        risk = price - stop_loss
        reward = target - price
        rr = _round2(reward / risk) if risk > 0 else None

        return {
            "trade_action": "LONG",
            "entry_low": entry_low,
            "entry_high": entry_high,
            "target_price": target,
            "stop_loss": stop_loss,
            "risk_reward": rr,
        }

    entry_low = _round2(price)
    entry_high = _round2(max(price + 0.75 * atr, swing_high - 0.25 * atr))
    if entry_high <= entry_low:
        entry_high = _round2(price + 0.5 * atr)

    target = _round2(min(price - SWING_TARGET_MULT * atr, swing_low))
    stop_loss = _round2(max(price + SWING_STOP_MULT * atr, swing_high + 0.1 * atr))
    if stop_loss <= entry_high:
        stop_loss = _round2(entry_high + 0.5 * atr)

    risk = stop_loss - price
    reward = price - target
    rr = _round2(reward / risk) if risk > 0 else None

    return {
        "trade_action": "SHORT",
        "entry_low": entry_low,
        "entry_high": entry_high,
        "target_price": target,
        "stop_loss": stop_loss,
        "risk_reward": rr,
    }
