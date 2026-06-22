"""F&O intraday prediction pipeline — Nifty, Bank Nifty, Sensex."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytz

from fno.config import FNO_INDICES, FnoIndex
from fno.data_fetch import fetch_intraday_bars, fetch_options_chain
from fno.features import build_features, latest_feature_row
from fno.labels import add_intraday_labels
from fno.models import FnoModelBundle, lstm_available, train_models
from fno.formatting import round_index
from fno.option_trades import OptionTradeRecommendation, build_option_trades_from_results
from fno.options_analytics import OptionsSnapshot, analyze_options
from predictor.trade_levels import compute_intraday_trade_levels

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")


@dataclass
class FnoPredictionResult:
    symbol: str
    price: int
    ml_signal: str
    ml_confidence: float
    options_signal: str
    combined_signal: str
    trade_action: str
    entry_low: Optional[int] = None
    entry_high: Optional[int] = None
    target_price: Optional[int] = None
    stop_loss: Optional[int] = None
    risk_reward: Optional[float] = None
    pcr: Optional[float] = None
    max_pain: Optional[int] = None
    iv_skew: Optional[float] = None
    expiry_date: Optional[str] = None
    expiry_day: Optional[str] = None
    days_to_expiry: Optional[int] = None
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


def _fetch_options_snapshots(
    indices: List[FnoIndex],
    include_options: bool,
) -> Tuple[Dict[str, Optional[OptionsSnapshot]], Dict[str, Optional[dict]]]:
    if not include_options:
        return {}, {}

    to_fetch = [i for i in indices if i.nse_options]
    snapshots: Dict[str, Optional[OptionsSnapshot]] = {}
    payloads: Dict[str, Optional[dict]] = {}

    for index in indices:
        if not index.nse_options:
            snapshots[index.name] = None
            payloads[index.name] = None

    def _load(index: FnoIndex) -> tuple[str, Optional[OptionsSnapshot], Optional[dict]]:
        payload = fetch_options_chain(index.option_symbol)
        if not payload:
            return index.name, None, None
        return index.name, analyze_options(payload, index.option_symbol), payload

    with ThreadPoolExecutor(max_workers=len(to_fetch) or 1) as pool:
        futures = {pool.submit(_load, idx): idx for idx in to_fetch}
        for fut in as_completed(futures):
            name, snap, payload = fut.result()
            snapshots[name] = snap
            payloads[name] = payload

    return snapshots, payloads


def predict_index(
    index: FnoIndex,
    options_snap: Optional[OptionsSnapshot] = None,
    df_15m: Optional[pd.DataFrame] = None,
) -> Optional[FnoPredictionResult]:
    """Run full F&O intraday pipeline for one index."""
    if df_15m is None:
        df_15m = fetch_intraday_bars(index.yf_symbol)
    if df_15m is None or len(df_15m) < 80:
        logger.warning("Insufficient 15m data for %s", index.name)
        return None

    feature_df = build_features(df_15m)
    if feature_df.empty or len(feature_df) < 80:
        return None

    labeled = add_intraday_labels(feature_df)
    bundle: FnoModelBundle = train_models(labeled, fast=True)

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
    elif not index.nse_options:
        notes.append("Sensex options on BSE — NSE chain not used; ML signal only")
    else:
        notes.append("Options chain unavailable — ML signal only")
    notes.append(f"ML {ml_signal} ({ml_conf:.0%} conf), model acc {bundle.train_accuracy:.0%}")

    return FnoPredictionResult(
        symbol=index.name,
        price=round_index(price) or 0,
        ml_signal=ml_signal,
        ml_confidence=round(ml_conf, 3),
        options_signal=opt_signal,
        combined_signal=combined,
        trade_action=levels.get("trade_action", "WAIT"),
        entry_low=round_index(levels.get("entry_low")),
        entry_high=round_index(levels.get("entry_high")),
        target_price=round_index(levels.get("target_price")),
        stop_loss=round_index(levels.get("stop_loss")),
        risk_reward=levels.get("risk_reward"),
        pcr=options_snap.pcr if options_snap else None,
        max_pain=round_index(options_snap.max_pain) if options_snap else None,
        iv_skew=options_snap.iv_skew if options_snap else None,
        expiry_date=options_snap.expiry_date if options_snap else None,
        expiry_day=options_snap.expiry_day if options_snap else None,
        days_to_expiry=options_snap.days_to_expiry if options_snap else None,
        model_accuracy=round(bundle.train_accuracy, 3),
        lstm_available=lstm_available(),
        notes=" | ".join(notes),
        as_of=datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
    )


def run_fno_intraday_prediction(
    indices: Optional[List[FnoIndex]] = None,
    *,
    include_options: bool = True,
) -> Tuple[List[FnoPredictionResult], List[OptionTradeRecommendation]]:
    """Run F&O intraday predictions for Nifty 50, Bank Nifty, and Sensex."""
    targets = indices or FNO_INDICES
    options_by_name, chain_payloads = _fetch_options_snapshots(targets, include_options)

    bars_by_symbol: Dict[str, Optional[pd.DataFrame]] = {}

    def _load_bars(index: FnoIndex) -> tuple[str, Optional[pd.DataFrame]]:
        return index.name, fetch_intraday_bars(index.yf_symbol)

    with ThreadPoolExecutor(max_workers=len(targets)) as pool:
        bar_futures = [pool.submit(_load_bars, idx) for idx in targets]
        for fut in as_completed(bar_futures):
            name, df = fut.result()
            bars_by_symbol[name] = df

    results: List[FnoPredictionResult] = []
    for index in targets:
        try:
            row = predict_index(
                index,
                options_by_name.get(index.name),
                bars_by_symbol.get(index.name),
            )
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
    option_trades = build_option_trades_from_results(results, chain_payloads)
    return results, option_trades


def results_to_dataframe(results: List[FnoPredictionResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    rows = []
    for r in results:
        expiry_label = "—"
        if r.expiry_date and r.expiry_day:
            dte = f"{r.days_to_expiry}d" if r.days_to_expiry is not None else "—"
            expiry_label = f"{r.expiry_date} ({r.expiry_day}, {dte})"
        rows.append(
            {
                "Index": r.symbol,
                "Price": r.price,
                "Expiry": expiry_label,
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
