"""
Telegram alert sender for NSE stock predictions.

Requires:
  TELEGRAM_BOT_TOKEN  — from @BotFather
  TELEGRAM_CHAT_ID    — your personal/group chat ID

Usage:
  from alerts.telegram_alert import send_prediction_alert
  send_prediction_alert(result)
"""

import logging
import os
from typing import Optional

import requests
from dotenv import load_dotenv

from predictor.scorer import PredictionResult, StockScore

load_dotenv()

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}/sendMessage"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _signal_emoji(signal: str) -> str:
    mapping = {
        "STRONG BUY": "🚀",
        "BUY": "📈",
        "NEUTRAL": "➡️",
        "SELL": "📉",
        "STRONG SELL": "🔴",
    }
    return mapping.get(signal, "❓")


def _format_stock_line(stock: StockScore, rank: int) -> str:
    emoji = _signal_emoji(stock.signal)
    rsi_str = f"RSI:{stock.rsi:.0f}" if stock.rsi is not None else ""
    return (
        f"{rank}. {emoji} *{stock.symbol.replace('.NS', '')}* — {stock.name}\n"
        f"   Price: ₹{stock.price:,.2f}  |  Score: {stock.score:+.1f}\n"
        f"   1D: {stock.change_1d:+.2f}%  5D: {stock.change_5d:+.2f}%  {rsi_str}\n"
        f"   Signal: {stock.signal}"
    )


def build_message(result: PredictionResult) -> str:
    """Build the full Telegram message string (Markdown)."""
    lines = [
        "🇮🇳 *NSE Nifty50 Daily Prediction*",
        f"📅 {result.run_timestamp}",
        f"📊 Stocks analysed: {result.symbols_processed}",
        "",
        "━━━━━━━━━━━━━━━━━━━━",
        "🏆 *TOP 5 POTENTIAL GAINERS*",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    for i, stock in enumerate(result.top_gainers, start=1):
        lines.append(_format_stock_line(stock, i))
        lines.append("")

    lines += [
        "━━━━━━━━━━━━━━━━━━━━",
        "⚠️ *TOP 5 POTENTIAL LOSERS*",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    for i, stock in enumerate(result.top_losers, start=1):
        lines.append(_format_stock_line(stock, i))
        lines.append("")

    lines += [
        "━━━━━━━━━━━━━━━━━━━━",
        "⚡ _Score = Technical (±40) + Sentiment (±30) + Momentum (±30)_",
        "_This is NOT financial advice. Trade at your own risk._",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sender
# ---------------------------------------------------------------------------

def send_telegram_message(
    message: str,
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
    parse_mode: str = "Markdown",
) -> bool:
    """
    Send a plain text message to a Telegram chat.

    Returns True on success, False on failure.
    """
    token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
    cid = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")

    if not token or not cid:
        logger.warning(
            "Telegram credentials not configured. "
            "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env"
        )
        return False

    url = TELEGRAM_API_BASE.format(token=token)
    payload = {
        "chat_id": cid,
        "text": message,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }

    try:
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        logger.info("Telegram alert sent successfully.")
        return True
    except requests.exceptions.HTTPError as exc:
        logger.error("Telegram HTTP error: %s — %s", exc, response.text)
        return False
    except requests.exceptions.RequestException as exc:
        logger.error("Telegram request error: %s", exc)
        return False


def send_prediction_alert(result: PredictionResult) -> bool:
    """
    Build and send the prediction result as a Telegram message.

    Respects ENABLE_TELEGRAM env var (default: true).
    """
    enabled = os.getenv("ENABLE_TELEGRAM", "true").lower()
    if enabled == "false":
        logger.info("Telegram alerts disabled via ENABLE_TELEGRAM=false")
        return False

    message = build_message(result)
    logger.debug("Telegram message:\n%s", message)
    return send_telegram_message(message)
