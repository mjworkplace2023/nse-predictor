"""
Intraday scorer using 5-minute or 15-minute candles for same-day trading signals.

Score breakdown (same -100 to +100 scale):
  Technical (RSI, MACD, EMA, BB, VWAP, Volume) : [-45, +45]
  Sentiment (reduced weight for intraday)       : [-15, +15]
  Intraday momentum (open, 30m, 1h)             : [-40, +40]

Momentum fields mapped to StockScore:
  change_1d  → since market open %
  change_5d  → last 30 minutes %
  change_20d → last 1 hour %
"""

import datetime
import logging
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import ta as ta_lib

from data.nifty50_symbols import NIFTY50_SYMBOLS, SYMBOL_NAMES, get_display_name
from predictor.fetch_data import (
    fetch_all_stocks,
    get_change_since_open,
    get_intraday_change_pct,
    get_latest_price,
    get_latest_session,
)
from predictor.market_hours import IST
from predictor.scorer import PredictionResult, StockScore, label_signal
from predictor.sentiment import compute_sentiment_scores, fetch_all_headlines

logger = logging.getLogger(__name__)

INTRADAY_WEIGHTS = {
    "rsi": 8,
    "macd": 8,
    "ema": 8,
    "bollinger": 5,
    "vwap": 10,
    "volume": 6,
}


@dataclass(frozen=True)
class IntradayConfig:
    interval: str
    period: str
    rsi_window: int
    macd_fast: int
    macd_slow: int
    macd_sign: int
    ema_fast: int
    ema_slow: int
    min_session_bars: int
    session_bars_per_day: int
    bars_30m: int
    bars_1h: int


INTRADAY_CONFIGS = {
    "5m": IntradayConfig(
        interval="5m",
        period="5d",
        rsi_window=9,
        macd_fast=6,
        macd_slow=13,
        macd_sign=5,
        ema_fast=9,
        ema_slow=21,
        min_session_bars=15,
        session_bars_per_day=75,
        bars_30m=6,
        bars_1h=12,
    ),
    "15m": IntradayConfig(
        interval="15m",
        period="5d",
        rsi_window=7,
        macd_fast=5,
        macd_slow=10,
        macd_sign=4,
        ema_fast=7,
        ema_slow=14,
        min_session_bars=8,
        session_bars_per_day=25,
        bars_30m=2,
        bars_1h=4,
    ),
}


def get_intraday_config(interval: str = "5m") -> IntradayConfig:
    if interval not in INTRADAY_CONFIGS:
        raise ValueError(f"Unsupported interval '{interval}'. Use '5m' or '15m'.")
    return INTRADAY_CONFIGS[interval]


def _score_rsi_intraday(df: pd.DataFrame, cfg: IntradayConfig) -> tuple:
    try:
        rsi_series = ta_lib.momentum.RSIIndicator(
            df["Close"], window=cfg.rsi_window
        ).rsi().dropna()
        if rsi_series.empty:
            return 0.0, None
        rsi = float(rsi_series.iloc[-1])
        if rsi <= 25:
            score = 1.0
        elif rsi >= 75:
            score = -1.0
        else:
            score = round((50 - rsi) / 25, 4)
        return score, rsi
    except Exception as exc:
        logger.debug("Intraday RSI error: %s", exc)
        return 0.0, None


def _score_macd_intraday(df: pd.DataFrame, cfg: IntradayConfig) -> float:
    try:
        macd_ind = ta_lib.trend.MACD(
            df["Close"],
            window_fast=cfg.macd_fast,
            window_slow=cfg.macd_slow,
            window_sign=cfg.macd_sign,
        )
        hist = macd_ind.macd_diff().dropna()
        if hist.empty:
            return 0.0
        latest = float(hist.iloc[-1])
        prev = float(hist.iloc[-2]) if len(hist) > 1 else 0.0
        sign_score = 1.0 if latest > 0 else -1.0
        momentum = 0.5 if (latest > prev and latest > 0) or (latest < prev and latest < 0) else 0.0
        return round(sign_score * 0.5 + momentum, 4)
    except Exception as exc:
        logger.debug("Intraday MACD error: %s", exc)
        return 0.0


def _score_ema_intraday(df: pd.DataFrame, cfg: IntradayConfig) -> float:
    try:
        ema_fast = ta_lib.trend.EMAIndicator(
            df["Close"], window=cfg.ema_fast
        ).ema_indicator().dropna()
        ema_slow = ta_lib.trend.EMAIndicator(
            df["Close"], window=cfg.ema_slow
        ).ema_indicator().dropna()
        if ema_fast.empty or ema_slow.empty:
            return 0.0
        price = float(df["Close"].iloc[-1])
        e_fast = float(ema_fast.iloc[-1])
        e_slow = float(ema_slow.iloc[-1])
        score = 0.0
        score += 0.5 if price > e_fast else -0.5
        score += 0.5 if e_fast > e_slow else -0.5
        return round(score, 4)
    except Exception as exc:
        logger.debug("Intraday EMA error: %s", exc)
        return 0.0


def _score_bollinger_intraday(df: pd.DataFrame) -> float:
    try:
        bb = ta_lib.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
        upper = bb.bollinger_hband().dropna()
        lower = bb.bollinger_lband().dropna()
        if upper.empty or lower.empty:
            return 0.0
        price = float(df["Close"].iloc[-1])
        band_width = float(upper.iloc[-1]) - float(lower.iloc[-1])
        if band_width == 0:
            return 0.0
        position = (price - float(lower.iloc[-1])) / band_width
        return round(max(-1.0, min(1.0, 1.0 - 2.0 * position)), 4)
    except Exception as exc:
        logger.debug("Intraday BB error: %s", exc)
        return 0.0


def _score_vwap(session_df: pd.DataFrame) -> float:
    try:
        if session_df.empty or session_df["Volume"].sum() == 0:
            return 0.0
        typical = (session_df["High"] + session_df["Low"] + session_df["Close"]) / 3
        cum_vol = session_df["Volume"].cumsum()
        vwap = (typical * session_df["Volume"]).cumsum() / cum_vol
        price = float(session_df["Close"].iloc[-1])
        vwap_now = float(vwap.iloc[-1])
        if vwap_now == 0:
            return 0.0
        deviation_pct = (price - vwap_now) / vwap_now * 100
        return round(max(-1.0, min(1.0, deviation_pct / 1.5)), 4)
    except Exception as exc:
        logger.debug("VWAP error: %s", exc)
        return 0.0


def _score_volume_intraday(
    session_df: pd.DataFrame, full_df: pd.DataFrame, cfg: IntradayConfig
) -> float:
    try:
        if session_df.empty:
            return 0.0
        session_vol = float(session_df["Volume"].sum())
        bars_elapsed = len(session_df)
        if bars_elapsed == 0:
            return 0.0

        full_ist = full_df.copy()
        full_ist.index = pd.to_datetime(full_ist.index)
        if full_ist.index.tz is None:
            full_ist.index = full_ist.index.tz_localize("UTC").tz_convert(IST)
        else:
            full_ist.index = full_ist.index.tz_convert(IST)

        session_vols = []
        min_bars = max(5, cfg.min_session_bars // 2)
        for _, day_df in full_ist.groupby(full_ist.index.date):
            if len(day_df) >= min_bars:
                session_vols.append(float(day_df["Volume"].sum()))

        if not session_vols:
            return 0.0

        avg_session_vol = sum(session_vols) / len(session_vols)
        if avg_session_vol == 0:
            return 0.0

        projected_vol = session_vol * (cfg.session_bars_per_day / bars_elapsed)
        ratio = projected_vol / avg_session_vol

        if ratio >= 1.5:
            return 1.0
        if ratio <= 0.6:
            return -1.0
        return round((ratio - 1.0) * 2.0, 4)
    except Exception as exc:
        logger.debug("Intraday volume error: %s", exc)
        return 0.0


def compute_intraday_technical_score(
    full_df: pd.DataFrame, session_df: pd.DataFrame, cfg: IntradayConfig
) -> dict:
    rsi_raw, rsi_value = _score_rsi_intraday(full_df, cfg)
    macd_raw = _score_macd_intraday(full_df, cfg)
    ema_raw = _score_ema_intraday(full_df, cfg)
    bb_raw = _score_bollinger_intraday(full_df)
    vwap_raw = _score_vwap(session_df)
    vol_raw = _score_volume_intraday(session_df, full_df, cfg)

    total = (
        rsi_raw * INTRADAY_WEIGHTS["rsi"]
        + macd_raw * INTRADAY_WEIGHTS["macd"]
        + ema_raw * INTRADAY_WEIGHTS["ema"]
        + bb_raw * INTRADAY_WEIGHTS["bollinger"]
        + vwap_raw * INTRADAY_WEIGHTS["vwap"]
        + vol_raw * INTRADAY_WEIGHTS["volume"]
    )

    return {"total": round(total, 2), "rsi_value": rsi_value}


def compute_intraday_momentum_score(session_df: pd.DataFrame, cfg: IntradayConfig) -> dict:
    since_open = get_change_since_open(session_df)
    change_30m = get_intraday_change_pct(session_df, bars_back=cfg.bars_30m)
    change_1h = get_intraday_change_pct(session_df, bars_back=cfg.bars_1h)

    def normalize(val: float, cap: float) -> float:
        return max(-1.0, min(1.0, val / cap))

    n_open = normalize(since_open, 2.0)
    n_30m = normalize(change_30m, 1.0)
    n_1h = normalize(change_1h, 1.5)

    total = (n_open * 20.0) + (n_30m * 12.0) + (n_1h * 8.0)

    return {
        "total": round(total, 2),
        "change_1d": since_open,
        "change_5d": change_30m,
        "change_20d": change_1h,
    }


def scale_sentiment_for_intraday(sentiment_score: float) -> float:
    return round(max(-15.0, min(15.0, sentiment_score * 0.5)), 2)


def run_intraday_prediction(
    symbols: Optional[List[str]] = None,
    top_n: int = 5,
    interval: str = "5m",
) -> PredictionResult:
    """
    Intraday pipeline: fetch candle data → indicators → sentiment → score → rank.

    Args:
        interval: "5m" or "15m"
    """
    cfg = get_intraday_config(interval)
    now = datetime.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")

    if symbols is None:
        symbols = NIFTY50_SYMBOLS

    logger.info(
        "Starting INTRADAY (%s) prediction for %d symbols at %s",
        cfg.interval, len(symbols), now,
    )

    logger.info("Step 1/3: Fetching %s intraday data...", cfg.interval)
    stock_data = fetch_all_stocks(symbols, period=cfg.period, interval=cfg.interval)

    successful_symbols = list(stock_data.keys())
    failed_count = len(symbols) - len(successful_symbols)

    logger.info("Step 2/3: Fetching news sentiment...")
    try:
        headlines = fetch_all_headlines()
    except Exception as exc:
        logger.warning("Could not fetch headlines: %s", exc)
        headlines = []

    sentiment_data = compute_sentiment_scores(
        successful_symbols, SYMBOL_NAMES, headlines=headlines
    )

    logger.info("Step 3/3: Computing intraday scores...")
    all_scores: List[StockScore] = []

    for symbol in successful_symbols:
        df = stock_data[symbol]
        session_df = get_latest_session(df)

        if session_df.empty or len(session_df) < cfg.min_session_bars:
            logger.warning("Insufficient intraday bars for %s — skipping", symbol)
            failed_count += 1
            continue

        try:
            tech = compute_intraday_technical_score(df, session_df, cfg)
            mom = compute_intraday_momentum_score(session_df, cfg)
            sent = sentiment_data.get(symbol, {"score": 0.0, "raw": None, "count": 0})
            sent_scaled = scale_sentiment_for_intraday(sent["score"])

            final_score = tech["total"] + sent_scaled + mom["total"]
            final_score = round(max(-100.0, min(100.0, final_score)), 2)

            stock = StockScore(
                symbol=symbol,
                name=get_display_name(symbol),
                price=round(get_latest_price(session_df), 2),
                score=final_score,
                technical_score=tech["total"],
                sentiment_score=sent_scaled,
                momentum_score=mom["total"],
                change_1d=mom["change_1d"],
                change_5d=mom["change_5d"],
                change_20d=mom["change_20d"],
                rsi=tech.get("rsi_value"),
                sentiment_raw=sent.get("raw"),
                sentiment_headline_count=sent.get("count", 0),
                signal=label_signal(final_score),
            )
            all_scores.append(stock)

        except Exception as exc:
            logger.error("Intraday scoring failed for %s: %s", symbol, exc)
            failed_count += 1

    all_scores.sort(key=lambda s: s.score, reverse=True)

    result = PredictionResult(
        run_timestamp=now,
        top_gainers=all_scores[:top_n],
        top_losers=all_scores[-top_n:][::-1],
        all_scores=all_scores,
        symbols_processed=len(all_scores),
        symbols_failed=failed_count,
        mode="intraday",
        interval=cfg.interval,
    )

    logger.info(
        "Intraday (%s) complete. Processed: %d, Failed: %d",
        cfg.interval, result.symbols_processed, result.symbols_failed,
    )

    return result
