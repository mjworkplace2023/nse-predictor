"""
WhatsApp alert sender for intraday NSE predictions.

Uses Meta WhatsApp Cloud API to post to a group or individual number.

Required .env variables:
  WHATSAPP_ACCESS_TOKEN      — from Meta Developer Console
  WHATSAPP_PHONE_NUMBER_ID   — your WhatsApp Business phone number ID
  WHATSAPP_GROUP_ID          — group JID, e.g. 120363012345678912@g.us
                               (or set WHATSAPP_TO for a single number)

Optional:
  ENABLE_WHATSAPP=true       — set to "false" to disable
  WHATSAPP_API_VERSION=v21.0 — API version (default v21.0)

Setup guide:
  1. Create a Meta app at https://developers.facebook.com
  2. Add WhatsApp product → get a test/production number
  3. Add your business number to the target WhatsApp group
  4. Use the group invite link / API tools to obtain the group JID
"""

import logging
import os
from typing import Optional

import requests
from dotenv import load_dotenv

from predictor.scorer import PredictionResult, StockScore

load_dotenv()

logger = logging.getLogger(__name__)

WHATSAPP_MAX_CHARS = 4000


def _signal_emoji(signal: str) -> str:
    return {
        "STRONG BUY": "🚀",
        "BUY": "📈",
        "NEUTRAL": "➡️",
        "SELL": "📉",
        "STRONG SELL": "🔴",
    }.get(signal, "❓")


def _format_stock_line(stock: StockScore, rank: int) -> str:
    emoji = _signal_emoji(stock.signal)
    rsi_str = f"{stock.rsi:.0f}" if stock.rsi is not None else "—"
    return (
        f"{rank}. {emoji} *{stock.symbol.replace('.NS', '')}* — {stock.name}\n"
        f"   ₹{stock.price:,.2f}  |  Score: {stock.score:+.1f}  |  {stock.signal}\n"
        f"   Open: {stock.change_1d:+.2f}%  |  30m: {stock.change_5d:+.2f}%  |  RSI: {rsi_str}"
    )


def build_intraday_message(result: PredictionResult) -> str:
    """Build WhatsApp message for an intraday prediction result."""
    interval = getattr(result, "interval", "5m")
    lines = [
        f"🇮🇳 *NSE Intraday Alert ({interval})*",
        f"📅 {result.run_timestamp}",
        f"📊 Stocks analysed: {result.symbols_processed}",
        "",
        "━━━━━━━━━━━━━━━━━━━━",
        "🏆 *TOP GAINERS*",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    for i, stock in enumerate(result.top_gainers, start=1):
        lines.append(_format_stock_line(stock, i))
        lines.append("")

    lines += [
        "━━━━━━━━━━━━━━━━━━━━",
        "⚠️ *TOP LOSERS*",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    for i, stock in enumerate(result.top_losers, start=1):
        lines.append(_format_stock_line(stock, i))
        lines.append("")

    lines += [
        "━━━━━━━━━━━━━━━━━━━━",
        "_Technical ±45 · Sentiment ±15 · Momentum ±40_",
        "_NOT financial advice. Trade at your own risk._",
    ]

    message = "\n".join(lines)
    if len(message) > WHATSAPP_MAX_CHARS:
        message = message[: WHATSAPP_MAX_CHARS - 20] + "\n\n…(truncated)"
    return message


def send_whatsapp_message(
    message: str,
    access_token: Optional[str] = None,
    phone_number_id: Optional[str] = None,
    recipient: Optional[str] = None,
) -> bool:
    """
    Send a text message via Meta WhatsApp Cloud API.

    recipient: group JID (120363…@g.us) or phone number (91XXXXXXXXXX).
    """
    token = access_token or os.getenv("WHATSAPP_ACCESS_TOKEN", "")
    phone_id = phone_number_id or os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
    to = recipient or os.getenv("WHATSAPP_GROUP_ID") or os.getenv("WHATSAPP_TO", "")

    if not token or not phone_id or not to:
        logger.warning(
            "WhatsApp not configured. Set WHATSAPP_ACCESS_TOKEN, "
            "WHATSAPP_PHONE_NUMBER_ID, and WHATSAPP_GROUP_ID (or WHATSAPP_TO) in .env"
        )
        return False

    api_version = os.getenv("WHATSAPP_API_VERSION", "v21.0")
    url = f"https://graph.facebook.com/{api_version}/{phone_id}/messages"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"preview_url": False, "body": message},
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=20)
        response.raise_for_status()
        logger.info("WhatsApp alert sent successfully to %s", to[:20] + "…")
        return True
    except requests.exceptions.HTTPError as exc:
        logger.error("WhatsApp HTTP error: %s — %s", exc, response.text)
        return False
    except requests.exceptions.RequestException as exc:
        logger.error("WhatsApp request error: %s", exc)
        return False


def send_intraday_whatsapp_alert(result: PredictionResult) -> bool:
    """
    Build and send intraday prediction to WhatsApp group/number.

    Respects ENABLE_WHATSAPP env var (default: true).
    Only sends for intraday results.
    """
    enabled = os.getenv("ENABLE_WHATSAPP", "true").lower()
    if enabled == "false":
        logger.info("WhatsApp alerts disabled via ENABLE_WHATSAPP=false")
        return False

    if result.mode != "intraday":
        logger.info("WhatsApp intraday alert skipped — result is not intraday mode")
        return False

    message = build_intraday_message(result)
    return send_whatsapp_message(message)
