"""Technical and price-action feature engineering for F&O."""

from __future__ import annotations

import numpy as np
import pandas as pd
import ta as ta_lib


def _safe_last(series: pd.Series, default: float = 0.0) -> float:
    if series is None or series.empty:
        return default
    val = series.iloc[-1]
    if pd.isna(val):
        return default
    return float(val)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build indicator features for every row (no lookahead)."""
    if df is None or df.empty or len(df) < 30:
        return pd.DataFrame()

    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]
    volume = out["Volume"].astype(float).copy()

    # Indices (Nifty/Bank Nifty/Sensex) often have zero volume on yfinance
    if volume.fillna(0).sum() <= 0:
        volume = (high - low).abs() * close
    volume = volume.replace(0, np.nan).fillna(1.0)

    out["ema_9"] = ta_lib.trend.EMAIndicator(close, window=9).ema_indicator()
    out["ema_21"] = ta_lib.trend.EMAIndicator(close, window=21).ema_indicator()
    out["ema_50"] = ta_lib.trend.EMAIndicator(close, window=50).ema_indicator()
    out["sma_20"] = ta_lib.trend.SMAIndicator(close, window=20).sma_indicator()

    macd = ta_lib.trend.MACD(close)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_hist"] = macd.macd_diff()

    out["rsi"] = ta_lib.momentum.RSIIndicator(close, window=14).rsi()
    stoch = ta_lib.momentum.StochasticOscillator(high, low, close)
    out["stoch_k"] = stoch.stoch()
    out["stoch_d"] = stoch.stoch_signal()

    bb = ta_lib.volatility.BollingerBands(close)
    out["bb_high"] = bb.bollinger_hband()
    out["bb_low"] = bb.bollinger_lband()
    out["bb_pct"] = bb.bollinger_pband()

    out["atr"] = ta_lib.volatility.AverageTrueRange(high, low, close).average_true_range()
    out["obv"] = ta_lib.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()

    # VWAP (session cumulative for intraday)
    typical = (high + low + close) / 3
    cum_vol = volume.cumsum().replace(0, np.nan)
    out["vwap"] = (typical * volume).cumsum() / cum_vol
    vol_ma = volume.rolling(20).mean().replace(0, np.nan)
    out["vol_ratio"] = (volume / vol_ma).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    out["daily_return"] = close.pct_change()
    out["hl_range"] = (high - low) / close.replace(0, np.nan)
    out["gap"] = (out["Open"] - close.shift(1)) / close.shift(1).replace(0, np.nan)

    body = (close - out["Open"]).abs()
    candle_range = (high - low).replace(0, np.nan)
    out["body_ratio"] = body / candle_range

    out["ema_bull"] = (out["ema_9"] > out["ema_21"]).astype(int)
    out["price_above_vwap"] = (close > out["vwap"]).astype(int)

    return out.dropna()


FEATURE_COLUMNS = [
    "ema_9", "ema_21", "ema_50", "sma_20",
    "macd", "macd_signal", "macd_hist",
    "rsi", "stoch_k", "stoch_d",
    "bb_pct", "atr", "obv", "vwap", "vol_ratio",
    "daily_return", "hl_range", "gap", "body_ratio",
    "ema_bull", "price_above_vwap",
]


def latest_feature_row(feature_df: pd.DataFrame) -> pd.Series:
    """Return the latest feature vector for prediction."""
    cols = [c for c in FEATURE_COLUMNS if c in feature_df.columns]
    return feature_df[cols].iloc[-1]
