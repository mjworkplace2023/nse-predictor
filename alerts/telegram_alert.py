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
from predictor.trade_levels import format_entry_range

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


def _format_daily_stock_line(stock: StockScore, rank: int) -> str:
    emoji = _signal_emoji(stock.signal)
    rsi_str = f"RSI:{stock.rsi:.0f}" if stock.rsi is not None else ""
    return (
        f"{rank}. {emoji} *{stock.symbol.replace('.NS', '')}* — {stock.name}\n"
        f"   Price: ₹{stock.price:,.2f}  |  Score: {stock.score:+.1f}\n"
        f"   1D: {stock.change_1d:+.2f}%  5D: {stock.change_5d:+.2f}%  {rsi_str}\n"
        f"   Signal: {stock.signal}"
    )


def _format_intraday_stock_line(stock: StockScore, rank: int) -> str:
    emoji = _signal_emoji(stock.signal)
    rsi_str = f"RSI:{stock.rsi:.0f}" if stock.rsi is not None else "—"
    lines = [
        f"{rank}. {emoji} *{stock.symbol.replace('.NS', '')}* — {stock.name}",
        f"   ₹{stock.price:,.2f}  |  Score: {stock.score:+.1f}  |  {stock.signal}",
        f"   Open: {stock.change_1d:+.2f}%  |  30m: {stock.change_5d:+.2f}%  |  "
        f"1h: {stock.change_20d:+.2f}%  |  {rsi_str}",
    ]
    action = getattr(stock, "trade_action", None)
    if action in ("LONG", "SHORT"):
        entry = format_entry_range(
            getattr(stock, "entry_low", None), getattr(stock, "entry_high", None)
        )
        target_price = getattr(stock, "target_price", None)
        stop_loss = getattr(stock, "stop_loss", None)
        risk_reward = getattr(stock, "risk_reward", None)
        target = f"₹{target_price:,.2f}" if target_price else "—"
        sl = f"₹{stop_loss:,.2f}" if stop_loss else "—"
        rr = f"{risk_reward:.1f}" if risk_reward else "—"
        lines.append(
            f"   📍 {action}  Entry: {entry}  Tgt: {target}  SL: {sl}  R:R {rr}"
        )
    return "\n".join(lines)


def build_daily_message(result: PredictionResult) -> str:
    """Build Telegram message for daily swing predictions."""
    n = len(result.top_gainers)
    lines = [
        "🇮🇳 *NSE Nifty50 Daily Prediction*",
        f"📅 {result.run_timestamp}",
        f"📊 Stocks analysed: {result.symbols_processed}",
        "",
        "━━━━━━━━━━━━━━━━━━━━",
        f"🏆 *TOP {n} POTENTIAL GAINERS*",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    for i, stock in enumerate(result.top_gainers, start=1):
        lines.append(_format_daily_stock_line(stock, i))
        lines.append("")

    n_losers = len(result.top_losers)
    lines += [
        "━━━━━━━━━━━━━━━━━━━━",
        f"⚠️ *TOP {n_losers} POTENTIAL LOSERS*",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    for i, stock in enumerate(result.top_losers, start=1):
        lines.append(_format_daily_stock_line(stock, i))
        lines.append("")

    lines += [
        "━━━━━━━━━━━━━━━━━━━━",
        "⚡ _Score = Technical (±40) + Sentiment (±30) + Momentum (±30)_",
        "_1D/5D = price % change over 1 and 5 trading days_",
        "_This is NOT financial advice. Trade at your own risk._",
    ]

    return "\n".join(lines)


def build_intraday_message(result: PredictionResult) -> str:
    """Build Telegram message for intraday predictions."""
    interval = getattr(result, "interval", "5m")
    n = len(result.top_gainers)
    lines = [
        f"🇮🇳 *NSE Intraday Alert ({interval})*",
        f"📅 {result.run_timestamp}",
        f"📊 Stocks analysed: {result.symbols_processed}",
        "",
        "━━━━━━━━━━━━━━━━━━━━",
        f"🏆 *TOP {n} GAINERS*",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    for i, stock in enumerate(result.top_gainers, start=1):
        lines.append(_format_intraday_stock_line(stock, i))
        lines.append("")

    n_losers = len(result.top_losers)
    lines += [
        "━━━━━━━━━━━━━━━━━━━━",
        f"⚠️ *TOP {n_losers} LOSERS*",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    for i, stock in enumerate(result.top_losers, start=1):
        lines.append(_format_intraday_stock_line(stock, i))
        lines.append("")

    lines += [
        "━━━━━━━━━━━━━━━━━━━━",
        "⚡ _Technical ±45 · Sentiment ±15 · Momentum ±40_",
        "_Open/30m/1h = price % change (not volume)_",
        "_This is NOT financial advice. Trade at your own risk._",
    ]

    return "\n".join(lines)


def build_message(result: PredictionResult) -> str:
    """Build the full Telegram message string (Markdown)."""
    if result.mode == "intraday":
        return build_intraday_message(result)
    return build_daily_message(result)


# ---------------------------------------------------------------------------
# Sender
# ---------------------------------------------------------------------------

def _config_value(key: str) -> str:
    """Read config from env or Streamlit Cloud secrets."""
    value = os.getenv(key, "").strip()
    if value:
        return value
    try:
        import streamlit as st

        if key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    return ""


def _parse_chat_ids(chat_id: Optional[str] = None) -> list[str]:
    """Return one or more chat IDs (comma-separated in TELEGRAM_CHAT_ID)."""
    raw = chat_id or _config_value("TELEGRAM_CHAT_ID")
    return [part.strip() for part in raw.split(",") if part.strip()]


def send_telegram_message(
    message: str,
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
    parse_mode: str = "Markdown",
) -> tuple[bool, str]:
    """
    Send a plain text message to one or more Telegram chats.

    TELEGRAM_CHAT_ID may be a single ID or comma-separated list, e.g.
    ``-1001234567890,5966698118`` (group + personal).

    Returns (success, detail_message).
    """
    token = (bot_token or _config_value("TELEGRAM_BOT_TOKEN")).strip()
    chat_ids = _parse_chat_ids(chat_id)

    if not token:
        return False, "TELEGRAM_BOT_TOKEN is missing (.env or Streamlit secrets)."
    if not chat_ids:
        return False, "TELEGRAM_CHAT_ID is missing (.env or Streamlit secrets)."

    url = TELEGRAM_API_BASE.format(token=token)
    sent_any = False
    errors: list[str] = []

    for cid in chat_ids:
        payload = {
            "chat_id": cid,
            "text": message,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }
        try:
            response = requests.post(url, json=payload, timeout=15)
            if not response.ok and parse_mode:
                err_text = response.text
                if "can't parse" in err_text.lower() or "parse entities" in err_text.lower():
                    plain_payload = {k: v for k, v in payload.items() if k != "parse_mode"}
                    response = requests.post(url, json=plain_payload, timeout=15)
            response.raise_for_status()
            logger.info("Telegram alert sent to chat_id=%s", cid)
            sent_any = True
        except requests.exceptions.HTTPError:
            try:
                detail = response.json().get("description", response.text)
            except Exception:
                detail = response.text
            msg = f"chat_id {cid}: {detail}"
            logger.error("Telegram HTTP error — %s", msg)
            errors.append(msg)
        except requests.exceptions.RequestException as exc:
            msg = f"chat_id {cid}: {exc}"
            logger.error("Telegram request error — %s", msg)
            errors.append(msg)

    if sent_any:
        return True, "Telegram alert sent."
    if errors:
        return False, errors[0]
    return False, "Telegram send failed for all chat IDs."


def send_prediction_alert(result: PredictionResult) -> tuple[bool, str]:
    """
    Build and send the prediction result as a Telegram message.

    Respects ENABLE_TELEGRAM env var (default: true).
    """
    enabled = _config_value("ENABLE_TELEGRAM") or "true"
    if enabled.lower() == "false":
        logger.info("Telegram alerts disabled via ENABLE_TELEGRAM=false")
        return False, "Telegram alerts are disabled (ENABLE_TELEGRAM=false)."

    message = build_message(result)
    logger.debug("Telegram message:\n%s", message)
    return send_telegram_message(message)
