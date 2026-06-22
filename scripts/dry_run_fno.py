#!/usr/bin/env python3
"""Dry-run F&O intraday prediction for Nifty 50, Bank Nifty, Sensex."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fno.config import FNO_INDICES
from fno.option_trades import option_trades_to_dataframe
from fno.predictor import run_fno_intraday_prediction, results_to_dataframe

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> int:
    names = ", ".join(i.name for i in FNO_INDICES)
    print("=" * 60)
    print("F&O Intraday — Dry Run (Indices)")
    print("=" * 60)
    print(f"Indices: {names}")
    print()

    results, option_trades = run_fno_intraday_prediction(include_options=True)
    if not results:
        print("ERROR: No results — check network / yfinance / nsepython.")
        return 1

    df = results_to_dataframe(results)
    print(df.to_string(index=False))
    print()
    print("--- Intraday Option Strikes ---")
    opt_df = option_trades_to_dataframe(option_trades)
    print(opt_df.to_string(index=False))
    print()

    for r in results:
        print(f"--- {r.symbol} ---")
        print(f"  Price: {r.price:,}")
        if r.expiry_date and r.expiry_day:
            print(f"  Expiry: {r.expiry_date} ({r.expiry_day}), {r.days_to_expiry} day(s) to expiry")
        print(f"  ML: {r.ml_signal} ({r.ml_confidence:.0%}) | Options: {r.options_signal}")
        print(f"  Combined: {r.combined_signal} | Action: {r.trade_action}")
        if r.target_price:
            print(f"  Target: {r.target_price:,} | SL: {r.stop_loss:,} | R:R {r.risk_reward}")
        print(f"  Model accuracy (hold-out): {r.model_accuracy:.0%}")
        print(f"  Notes: {r.notes}")
        print()

    print("Dry run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
