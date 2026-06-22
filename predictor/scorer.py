"""
Master scorer that combines technical indicators + sentiment
into a final stock score in the range [-100, +100].

Score breakdown:
  Technical indicators : [-40, +40]  (RSI, MACD, EMA, BB, Volume)
  Sentiment            : [-30, +30]  (VADER on news headlines)
  Price momentum       : [-30, +30]  (1-day, 5-day, 20-day returns)

  Total                : [-100, +100]
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from data.nifty50_symbols import NIFTY50_SYMBOLS, SYMBOL_NAMES, get_display_name
from predictor.fetch_data import (
    fetch_all_stocks,
    get_latest_price,
    get_price_change_pct,
)
from predictor.trade_levels import compute_daily_trade_levels, format_entry_range
from predictor.indicators import compute_technical_score
from predictor.sentiment import compute_sentiment_scores, fetch_all_headlines

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StockScore:
    symbol: str
    name: str
    price: float
    score: float                     # final [-100, +100]
    technical_score: float           # [-40, +40]
    sentiment_score: float           # [-30, +30]
    momentum_score: float            # [-30, +30]
    change_1d: float                 # % change 1-day
    change_5d: float                 # % change 5-day
    change_20d: float                # % change 20-day
    rsi: Optional[float] = None
    sentiment_raw: Optional[float] = None
    sentiment_headline_count: int = 0
    signal: str = "NEUTRAL"         # "STRONG BUY", "BUY", "NEUTRAL", "SELL", "STRONG SELL"
    trade_action: Optional[str] = None   # "LONG", "SHORT", "WAIT" (intraday only)
    entry_low: Optional[float] = None
    entry_high: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    risk_reward: Optional[float] = None


@dataclass
class PredictionResult:
    run_timestamp: str
    top_gainers: List[StockScore] = field(default_factory=list)
    top_losers: List[StockScore] = field(default_factory=list)
    all_scores: List[StockScore] = field(default_factory=list)
    symbols_processed: int = 0
    symbols_failed: int = 0
    mode: str = "daily"       # "daily" or "intraday"
    interval: str = "1d"      # "1d", "5m", or "15m"
    intraday_session_date: Optional[str] = None   # YYYY-MM-DD bars used for scoring
    intraday_session_fallback: bool = False       # True when not using today's session


# ---------------------------------------------------------------------------
# Momentum scorer
# ---------------------------------------------------------------------------

def compute_momentum_score(df: pd.DataFrame) -> dict:
    """
    Price momentum across multiple horizons → score in [-30, +30].

    Returns:
        {
          "total":   float,   # [-30, +30]
          "change_1d": float,
          "change_5d": float,
          "change_20d": float,
        }
    """
    c1 = get_price_change_pct(df, days=1)
    c5 = get_price_change_pct(df, days=5)
    c20 = get_price_change_pct(df, days=20)

    # Normalize: cap at ±5 % for 1d, ±10 % for 5d, ±20 % for 20d
    def normalize(val: float, cap: float) -> float:
        return max(-1.0, min(1.0, val / cap))

    n1 = normalize(c1, 5.0)
    n5 = normalize(c5, 10.0)
    n20 = normalize(c20, 20.0)

    # Weights: recent momentum matters more
    total = (n1 * 15.0) + (n5 * 10.0) + (n20 * 5.0)

    return {
        "total": round(total, 2),
        "change_1d": c1,
        "change_5d": c5,
        "change_20d": c20,
    }


# ---------------------------------------------------------------------------
# Signal labelling
# ---------------------------------------------------------------------------

def label_signal(score: float) -> str:
    if score >= 50:
        return "STRONG BUY"
    elif score >= 20:
        return "BUY"
    elif score <= -50:
        return "STRONG SELL"
    elif score <= -20:
        return "SELL"
    else:
        return "NEUTRAL"


# ---------------------------------------------------------------------------
# Master run function
# ---------------------------------------------------------------------------

def run_prediction(
    symbols: Optional[List[str]] = None,
    top_n: int = 5,
) -> PredictionResult:
    """
    Full pipeline: fetch → indicators → sentiment → score → rank.

    Args:
        symbols: list of NSE tickers (defaults to full Nifty 50)
        top_n:   number of top gainers / losers to return

    Returns:
        PredictionResult with ranked stocks.
    """
    import datetime
    import pytz

    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")

    if symbols is None:
        symbols = NIFTY50_SYMBOLS

    logger.info("Starting prediction run for %d symbols at %s", len(symbols), now)

    # Step 1: Fetch OHLCV data
    logger.info("Step 1/3: Fetching market data...")
    stock_data = fetch_all_stocks(symbols, period="3mo", interval="1d")

    successful_symbols = list(stock_data.keys())
    failed_count = len(symbols) - len(successful_symbols)

    # Step 2: Fetch news headlines once (shared across all stocks)
    logger.info("Step 2/3: Fetching news sentiment...")
    try:
        headlines = fetch_all_headlines()
    except Exception as exc:
        logger.warning("Could not fetch headlines: %s", exc)
        headlines = []

    sentiment_data = compute_sentiment_scores(
        successful_symbols, SYMBOL_NAMES, headlines=headlines
    )

    # Step 3: Score each stock
    logger.info("Step 3/3: Computing scores...")
    all_scores: List[StockScore] = []

    for symbol in successful_symbols:
        df = stock_data[symbol]

        try:
            tech = compute_technical_score(df)
            mom = compute_momentum_score(df)
            sent = sentiment_data.get(symbol, {"score": 0.0, "raw": None, "count": 0})

            final_score = tech["total"] + sent["score"] + mom["total"]
            final_score = round(max(-100.0, min(100.0, final_score)), 2)

            signal = label_signal(final_score)
            price = round(get_latest_price(df), 2)
            levels = compute_daily_trade_levels(df, price, signal)

            stock = StockScore(
                symbol=symbol,
                name=get_display_name(symbol),
                price=price,
                score=final_score,
                technical_score=tech["total"],
                sentiment_score=sent["score"],
                momentum_score=mom["total"],
                change_1d=mom["change_1d"],
                change_5d=mom["change_5d"],
                change_20d=mom["change_20d"],
                rsi=tech.get("rsi_value"),
                sentiment_raw=sent.get("raw"),
                sentiment_headline_count=sent.get("count", 0),
                signal=signal,
                trade_action=levels["trade_action"],
                entry_low=levels["entry_low"],
                entry_high=levels["entry_high"],
                target_price=levels["target_price"],
                stop_loss=levels["stop_loss"],
                risk_reward=levels["risk_reward"],
            )
            all_scores.append(stock)

        except Exception as exc:
            logger.error("Scoring failed for %s: %s", symbol, exc)

    # Sort by score descending
    all_scores.sort(key=lambda s: s.score, reverse=True)

    top_gainers = all_scores[:top_n]
    top_losers = all_scores[-top_n:][::-1]  # worst first

    result = PredictionResult(
        run_timestamp=now,
        top_gainers=top_gainers,
        top_losers=top_losers,
        all_scores=all_scores,
        symbols_processed=len(all_scores),
        symbols_failed=failed_count,
    )

    logger.info(
        "Prediction complete. Processed: %d, Failed: %d",
        result.symbols_processed,
        result.symbols_failed,
    )
    logger.info(
        "Top gainer: %s (%.1f)  |  Top loser: %s (%.1f)",
        top_gainers[0].symbol if top_gainers else "N/A",
        top_gainers[0].score if top_gainers else 0,
        top_losers[0].symbol if top_losers else "N/A",
        top_losers[0].score if top_losers else 0,
    )

    return result


def _is_intraday_result(result: PredictionResult) -> bool:
    """True when stored results came from an intraday run."""
    if result is None:
        return False
    if result.mode == "intraday":
        return True
    return getattr(result, "interval", "1d") in ("5m", "15m")


def result_to_dataframe(
    result: PredictionResult,
    *,
    intraday_view: bool | None = None,
    include_trade_levels: bool | None = None,
) -> pd.DataFrame:
    """Convert PredictionResult.all_scores to a pandas DataFrame for display."""
    intraday = (
        intraday_view
        if intraday_view is not None
        else _is_intraday_result(result)
    )
    trade_levels = (
        include_trade_levels
        if include_trade_levels is not None
        else True
    )
    if intraday:
        mom_cols = ("Open %", "30m %", "1h %")
    else:
        mom_cols = ("1D %", "5D %", "20D %")

    def _resolve_action(stock) -> str:
        action = getattr(stock, "trade_action", None)
        if action:
            return action
        if stock.signal in ("BUY", "STRONG BUY"):
            return "LONG"
        if stock.signal in ("SELL", "STRONG SELL"):
            return "SHORT"
        return "WAIT"

    rows = []
    for s in result.all_scores:
        row = {
            "Symbol": s.symbol.replace(".NS", ""),
            "Company": s.name,
            "Price (INR)": s.price,
            "Score": s.score,
            "Signal": s.signal,
            "Technical": s.technical_score,
            "Sentiment": s.sentiment_score,
            "Momentum": s.momentum_score,
            mom_cols[0]: s.change_1d,
            mom_cols[1]: s.change_5d,
            mom_cols[2]: s.change_20d,
            "RSI": round(s.rsi, 1) if s.rsi is not None else None,
            "News Count": s.sentiment_headline_count,
        }
        if intraday:
            row["Action"] = _resolve_action(s)
            if trade_levels:
                row.update({
                    "Entry Range": format_entry_range(
                        getattr(s, "entry_low", None), getattr(s, "entry_high", None)
                    ),
                    "Intraday Target": getattr(s, "target_price", None),
                    "Stop Loss": getattr(s, "stop_loss", None),
                    "R:R": getattr(s, "risk_reward", None),
                })
        elif trade_levels:
            row["Action"] = _resolve_action(s)
            row.update({
                "Entry Range": format_entry_range(
                    getattr(s, "entry_low", None), getattr(s, "entry_high", None)
                ),
                "Target": getattr(s, "target_price", None),
                "Stop Loss": getattr(s, "stop_loss", None),
                "R:R": getattr(s, "risk_reward", None),
            })
        rows.append(row)

    df = pd.DataFrame(rows)
    if intraday and not df.empty:
        col_order = [
            "Symbol", "Company", "Price (INR)", "Score", "Signal", "Action",
            "Entry Range", "Intraday Target", "Stop Loss", "R:R",
            "Technical", "Sentiment", "Momentum",
            "Open %", "30m %", "1h %",
            "RSI", "News Count",
        ]
        df = df[[c for c in col_order if c in df.columns]]
    elif not intraday and not df.empty:
        col_order = [
            "Symbol", "Company", "Price (INR)", "Score", "Signal", "Action",
            "Entry Range", "Target", "Stop Loss", "R:R",
            "Technical", "Sentiment", "Momentum",
            "1D %", "5D %", "20D %",
            "RSI", "News Count",
        ]
        df = df[[c for c in col_order if c in df.columns]]
    return df
