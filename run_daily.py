"""
run_daily.py — CLI entry point for the NSE Stock Predictor.

Usage:
    python run_daily.py                # Run once immediately
    python run_daily.py --schedule     # Run at 9:30 AM IST every weekday

Environment variables (via .env):
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID
    ENABLE_TELEGRAM     (default: true)
    TOP_N_STOCKS        (default: 5)
"""

import argparse
import logging
import os
import sys
import time
import datetime

import pytz
import schedule
from dotenv import load_dotenv

# Allow imports from project root when run as script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from predictor.scorer import run_prediction, result_to_dataframe
from predictor.intraday_scorer import run_intraday_prediction
from alerts.telegram_alert import send_prediction_alert

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_daily")

IST = pytz.timezone("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Core job
# ---------------------------------------------------------------------------

def run_prediction_job(top_n: int = 5, send_alert: bool = True, intraday: bool = False) -> None:
    """Execute one full prediction cycle."""
    now_ist = datetime.datetime.now(IST)
    mode_label = "INTRADAY" if intraday else "DAILY"
    logger.info("=" * 60)
    logger.info("NSE Predictor %s run started at %s", mode_label, now_ist.strftime("%Y-%m-%d %H:%M:%S IST"))
    logger.info("=" * 60)

    if intraday:
        result = run_intraday_prediction(top_n=top_n)
    else:
        result = run_prediction(top_n=top_n)

    # Print summary to console
    print("\n" + "=" * 60)
    print(f"NSE NIFTY50 {mode_label} PREDICTION — {result.run_timestamp}")
    print(f"Stocks processed: {result.symbols_processed}  |  Failed: {result.symbols_failed}")
    print("=" * 60)

    print(f"\n{'TOP ' + str(top_n) + ' POTENTIAL GAINERS':}")
    print("-" * 40)
    for i, s in enumerate(result.top_gainers, 1):
        rsi_str = f"RSI={s.rsi:.0f}" if s.rsi else ""
        mom_label = "Open" if intraday else "1D"
        print(
            f"{i:2}. {s.symbol.replace('.NS',''):12} {s.name[:25]:25} "
            f"Score={s.score:+6.1f}  ₹{s.price:>9,.2f}  "
            f"{mom_label}={s.change_1d:+5.2f}%  {s.signal:11}  {rsi_str}"
        )

    print(f"\n{'TOP ' + str(top_n) + ' POTENTIAL LOSERS':}")
    print("-" * 40)
    for i, s in enumerate(result.top_losers, 1):
        rsi_str = f"RSI={s.rsi:.0f}" if s.rsi else ""
        mom_label = "Open" if intraday else "1D"
        print(
            f"{i:2}. {s.symbol.replace('.NS',''):12} {s.name[:25]:25} "
            f"Score={s.score:+6.1f}  ₹{s.price:>9,.2f}  "
            f"{mom_label}={s.change_1d:+5.2f}%  {s.signal:11}  {rsi_str}"
        )

    print("\n" + "=" * 60)

    # Full leaderboard
    df = result_to_dataframe(result)
    print("\nFULL LEADERBOARD (sorted by score):")
    print(df.to_string(index=False))

    # Send Telegram alert
    if send_alert:
        logger.info("Sending Telegram alert...")
        ok = send_prediction_alert(result)
        if ok:
            logger.info("Telegram alert delivered.")
        else:
            logger.warning("Telegram alert not delivered (check credentials).")
    else:
        logger.info("Telegram alert skipped (--no-alert flag).")


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def schedule_job(top_n: int, send_alert: bool) -> None:
    """Run prediction at 9:30 AM IST on weekdays."""

    def ist_9_30_job():
        now = datetime.datetime.now(IST)
        # Skip weekends (Saturday=5, Sunday=6)
        if now.weekday() >= 5:
            logger.info("Weekend — skipping prediction run.")
            return
        run_prediction_job(top_n=top_n, send_alert=send_alert)

    schedule.every().day.at("04:00").do(ist_9_30_job)  # 04:00 UTC = 09:30 IST
    logger.info("Scheduler started. Will run at 09:30 IST on weekdays.")
    logger.info("Press Ctrl+C to stop.")

    while True:
        schedule.run_pending()
        time.sleep(30)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NSE Nifty50 Stock Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_daily.py                    # Run once immediately
  python run_daily.py --top 10           # Show top 10
  python run_daily.py --no-alert         # Skip Telegram
  python run_daily.py --schedule         # Run daily at 9:30 AM IST
        """,
    )
    parser.add_argument(
        "--top",
        type=int,
        default=int(os.getenv("TOP_N_STOCKS", "5")),
        help="Number of top/bottom stocks to show (default: 5)",
    )
    parser.add_argument(
        "--no-alert",
        action="store_true",
        help="Skip sending Telegram alert",
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Run on a daily schedule at 9:30 AM IST (weekdays only)",
    )
    parser.add_argument(
        "--intraday",
        action="store_true",
        help="Use intraday scoring (best during market hours)",
    )
    parser.add_argument(
        "--interval",
        choices=["5m", "15m"],
        default=os.getenv("INTRADAY_INTERVAL", "5m"),
        help="Intraday candle interval when --intraday is set (default: 5m)",
    )

    args = parser.parse_args()
    send_alert = not args.no_alert

    if args.schedule:
        schedule_job(top_n=args.top, send_alert=send_alert)
    elif args.intraday:
        from run_intraday import run_intraday_job
        run_intraday_job(
            top_n=args.top,
            interval=args.interval,
            send_whatsapp=send_alert,
            force=True,
        )
    else:
        run_prediction_job(top_n=args.top, send_alert=send_alert, intraday=False)


if __name__ == "__main__":
    main()
