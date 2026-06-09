"""
run_intraday.py — Intraday prediction runner with WhatsApp group alerts.

Runs every 5 minutes during NSE market hours (09:15–15:30 IST, weekdays).

Usage:
    python run_intraday.py                          # Run once now
    python run_intraday.py --schedule               # Auto every 5 min (market hours)
    python run_intraday.py --interval 15m           # Use 15-minute candles
    python run_intraday.py --schedule --no-whatsapp # Skip WhatsApp

Environment variables (via .env):
    WHATSAPP_ACCESS_TOKEN
    WHATSAPP_PHONE_NUMBER_ID
    WHATSAPP_GROUP_ID
    ENABLE_WHATSAPP=true
    TOP_N_STOCKS=5
    INTRADAY_INTERVAL=5m
"""

import argparse
import logging
import os
import sys
import time

import schedule
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from alerts.whatsapp_alert import send_intraday_whatsapp_alert
from predictor.intraday_scorer import run_intraday_prediction
from predictor.market_hours import is_market_open, market_status_message, now_ist
from predictor.scorer import result_to_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_intraday")


def run_intraday_job(
    top_n: int = 5,
    interval: str = "5m",
    send_whatsapp: bool = True,
    force: bool = False,
) -> None:
    """Execute one intraday prediction cycle."""
    if not force and not is_market_open():
        logger.info("Market closed — %s", market_status_message())
        return

    logger.info("=" * 60)
    logger.info(
        "Intraday (%s) run started at %s",
        interval,
        now_ist().strftime("%Y-%m-%d %H:%M:%S IST"),
    )
    logger.info("=" * 60)

    result = run_intraday_prediction(top_n=top_n, interval=interval)

    print("\n" + "=" * 60)
    print(f"NSE INTRADAY ({interval}) — {result.run_timestamp}")
    print(f"Stocks processed: {result.symbols_processed}  |  Failed: {result.symbols_failed}")
    print("=" * 60)

    print(f"\nTOP {top_n} INTRADAY GAINERS")
    print("-" * 40)
    for i, s in enumerate(result.top_gainers, 1):
        rsi_str = f"RSI={s.rsi:.0f}" if s.rsi else ""
        print(
            f"{i:2}. {s.symbol.replace('.NS',''):12} {s.name[:25]:25} "
            f"Score={s.score:+6.1f}  ₹{s.price:>9,.2f}  "
            f"Open={s.change_1d:+5.2f}%  {s.signal:11}  {rsi_str}"
        )

    print(f"\nTOP {top_n} INTRADAY LOSERS")
    print("-" * 40)
    for i, s in enumerate(result.top_losers, 1):
        rsi_str = f"RSI={s.rsi:.0f}" if s.rsi else ""
        print(
            f"{i:2}. {s.symbol.replace('.NS',''):12} {s.name[:25]:25} "
            f"Score={s.score:+6.1f}  ₹{s.price:>9,.2f}  "
            f"Open={s.change_1d:+5.2f}%  {s.signal:11}  {rsi_str}"
        )

    df = result_to_dataframe(result)
    print("\nFULL LEADERBOARD:")
    print(df.to_string(index=False))

    if send_whatsapp:
        logger.info("Sending WhatsApp intraday alert...")
        ok = send_intraday_whatsapp_alert(result)
        if ok:
            logger.info("WhatsApp alert delivered.")
        else:
            logger.warning("WhatsApp alert not delivered (check .env credentials).")
    else:
        logger.info("WhatsApp alert skipped.")


def schedule_intraday_job(top_n: int, interval: str, send_whatsapp: bool) -> None:
    """Run intraday prediction every 5 minutes during market hours."""

    def job():
        run_intraday_job(top_n=top_n, interval=interval, send_whatsapp=send_whatsapp)

    schedule.every(5).minutes.do(job)
    logger.info(
        "Intraday scheduler started (%s candles, every 5 min, market hours only).",
        interval,
    )
    logger.info("Press Ctrl+C to stop.")

    # Run immediately if market is open
    if is_market_open():
        run_intraday_job(top_n=top_n, interval=interval, send_whatsapp=send_whatsapp)

    while True:
        schedule.run_pending()
        time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NSE Nifty50 Intraday Predictor with WhatsApp alerts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_intraday.py                         # Run once (market hours only)
  python run_intraday.py --force                 # Run even when market closed
  python run_intraday.py --schedule              # Every 5 min + WhatsApp
  python run_intraday.py --interval 15m          # 15-minute candles
  python run_intraday.py --schedule --no-whatsapp
        """,
    )
    parser.add_argument(
        "--top",
        type=int,
        default=int(os.getenv("TOP_N_STOCKS", "5")),
        help="Number of top/bottom stocks (default: 5)",
    )
    parser.add_argument(
        "--interval",
        choices=["5m", "15m"],
        default=os.getenv("INTRADAY_INTERVAL", "5m"),
        help="Candle interval (default: 5m)",
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Run every 5 minutes during market hours",
    )
    parser.add_argument(
        "--no-whatsapp",
        action="store_true",
        help="Skip WhatsApp group alert",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even when market is closed",
    )

    args = parser.parse_args()
    send_whatsapp = not args.no_whatsapp

    if args.schedule:
        schedule_intraday_job(
            top_n=args.top, interval=args.interval, send_whatsapp=send_whatsapp
        )
    else:
        run_intraday_job(
            top_n=args.top,
            interval=args.interval,
            send_whatsapp=send_whatsapp,
            force=args.force,
        )


if __name__ == "__main__":
    main()
