"""F&O intraday prediction pipeline — Nifty, Bank Nifty, Sensex."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import pandas as pd
import pytz

from fno.config import FNO_INDICES, FnoIndex
from fno.data_fetch import fetch_multi_interval, fetch_options_chain
from fno.features import build_features, latest_feature_row
from fno.labels import add_intraday_labels
from fno.models import FnoModelBundle, lstm_available, train_models
from fno.options_analytics import OptionsSnapshot, analyze_options
from predictor.trade_levels import compute_intraday_trade_levels

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")


@dataclass
class FnoPredictionResult:
    symbol: str
    price: float
    ml_signal: str
    ml_confidence: float
    options_signal: str
    combined_signal: str
    trade_action: str
    entry_low: Optional[float] = None
    entry_high: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    risk_reward: Optional[float] = None
    pcr: Optional[float] = None
    max_pain: Optional[float] = None
    iv_skew: Optional[float] = None
    model_accuracy: float = 0.0
    lstm_available: bool = False
    notes: str = ""
    as_of: str = ""


def _today_session(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC").tz_convert(IST)
    else:
        idx = idx.tz_convert(IST)
    today = datetime.now(IST).date()
    mask = idx.date == today
    return df.loc[mask]


def _combine_signals(ml_signal: str, ml_conf: float, opt_signal: str) -> str:
    score = {"SELL": -1, "HOLD": 0, "BUY": 1}
    ml_w = min(ml_conf, 0.95)
    opt_w = 0.35
    combined = score.get(ml_signal, 0) * ml_w + score.get(opt_signal, 0) * opt_w
    if combined >= 0.35:
        return "BUY"
    if combined <= -0.35:
        return "SELL"
    return "HOLD"


def _ml_to_trade_signal(ml_signal: str) -> str:
    return {"BUY": "BUY", "SELL": "SELL", "HOLD": "NEUTRAL"}.get(ml_signal, "NEUTRAL")


def predict_index(
    index: FnoIndex,
    options_snap: Optional[OptionsSnapshot] = None,
) -> Optional[FnoPredictionResult]:
    """Run full F&O intraday pipeline for one index."""
    bars = fetch_multi_interval(index.yf_symbol)
    df_15m = bars.get("15m")
    if df_15m is None or len(df_15m) < 80:
        logger.warning("Insufficient 15m data for %s", index.name)
        return None

    feature_df = build_features(df_15m)
    if feature_df.empty or len(feature_df) < 80:
        return None

    labeled = add_intraday_labels(feature_df)
    bundle: FnoModelBundle = train_models(labeled)

    latest = latest_feature_row(feature_df).to_frame().T
    ml_signal, ml_conf = bundle.predict_label(latest)
    price = float(feature_df["Close"].iloc[-1])

    opt_signal = options_snap.signal if options_snap else "NEUTRAL"
    combined = _combine_signals(ml_signal, ml_conf, opt_signal)

    session_df = _today_session(df_15m)
    levels = compute_intraday_trade_levels(
        session_df if not session_df.empty else df_15m.tail(26),
        df_15m,
        price,
        _ml_to_trade_signal(combined if combined != "HOLD" else ml_signal),
    )

    notes = []
    if options_snap:
        notes.append(options_snap.notes)
    else:
        notes.append("Options chain unavailable — ML signal only")
    notes.append(f"ML {ml_signal} ({ml_conf:.0%} conf), ensemble acc {bundle.train_accuracy:.0%}")

    return FnoPredictionResult(
        symbol=index.name,
        price=round(price, 2),
        ml_signal=ml_signal,
        ml_confidence=round(ml_conf, 3),
        options_signal=opt_signal,
        combined_signal=combined,
        trade_action=levels.get("trade_action", "WAIT"),
        entry_low=levels.get("entry_low"),
        entry_high=levels.get("entry_high"),
        target_price=levels.get("target_price"),
        stop_loss=levels.get("stop_loss"),
        risk_reward=levels.get("risk_reward"),
        pcr=options_snap.pcr if options_snap else None,
        max_pain=options_snap.max_pain if options_snap else None,
        iv_skew=options_snap.iv_skew if options_snap else None,
        model_accuracy=round(bundle.train_accuracy, 3),
        lstm_available=lstm_available(),
        notes=" | ".join(notes),
        as_of=datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
    )


def run_fno_intraday_prediction(
    indices: Optional[List[FnoIndex]] = None,
    *,
    include_options: bool = True,
) -> List[FnoPredictionResult]:
    """Run F&O intraday predictions for Nifty 50, Bank Nifty, and Sensex."""
    targets = indices or FNO_INDICES
    results: List[FnoPredictionResult] = []

    for index in targets:
        options_snap: Optional[OptionsSnapshot] = None
        if include_options:
            payload = fetch_options_chain(index.option_symbol)
            if payload:
                options_snap = analyze_options(payload, index.option_symbol)

        try:
            row = predict_index(index, options_snap)
            if row:
                results.append(row)
        except Exception as exc:
            logger.exception("F&O prediction failed for %s: %s", index.name, exc)

    results.sort(
        key=lambda r: (
            {"BUY": 2, "HOLD": 1, "SELL": 0}.get(r.combined_signal, 0),
            r.ml_confidence,
        ),
        reverse=True,
    )
    return results


def results_to_dataframe(results: List[FnoPredictionResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    rows = []
    for r in results:
        rows.append(
            {
                "Index": r.symbol,
                "Price": r.price,
                "ML Signal": r.ml_signal,
                "ML Conf": f"{r.ml_confidence:.0%}",
                "Options": r.options_signal,
                "Signal": r.combined_signal,
                "Action": r.trade_action,
                "Entry Low": r.entry_low,
                "Entry High": r.entry_high,
                "Target": r.target_price,
                "Stop Loss": r.stop_loss,
                "R:R": r.risk_reward,
                "PCR": r.pcr,
                "Max Pain": r.max_pain,
                "IV Skew": r.iv_skew,
                "Model Acc": f"{r.model_accuracy:.0%}",
            }
        )
    return pd.DataFrame(rows)
