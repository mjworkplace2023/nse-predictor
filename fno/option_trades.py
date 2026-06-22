"""Intraday option strike recommendations — CALL/PUT with target & SL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from fno.config import FNO_INDICES, FnoIndex, OPTION_SL_MULT, OPTION_TARGET_MULT
from fno.data_fetch import option_chain_to_dataframe
from fno.formatting import parse_expiry


@dataclass
class OptionTradeRecommendation:
    index: str
    strike: int
    action: str  # CALL | PUT | WAIT
    entry_premium: Optional[float] = None
    target: Optional[float] = None
    stop_loss: Optional[float] = None
    risk_reward: Optional[float] = None
    expiry_date: Optional[str] = None
    expiry_day: Optional[str] = None
    days_to_expiry: Optional[int] = None
    notes: str = ""


def _nearest_strike(spot: float, step: int) -> int:
    return int(round(spot / step) * step)


def _pick_strike(spot: float, step: int, action: str, chain_df: pd.DataFrame) -> int:
    atm = _nearest_strike(spot, step)
    if chain_df.empty:
        return atm
    if action == "CALL":
        otm = chain_df.loc[chain_df["strike"] >= spot, "strike"]
        if not otm.empty:
            return int(otm.min())
    elif action == "PUT":
        otm = chain_df.loc[chain_df["strike"] <= spot, "strike"]
        if not otm.empty:
            return int(otm.max())
    return atm


def _ltp_at_strike(chain_df: pd.DataFrame, strike: int, action: str) -> Optional[float]:
    if chain_df.empty:
        return None
    row = chain_df.loc[chain_df["strike"] == strike]
    if row.empty:
        row = chain_df.iloc[(chain_df["strike"] - strike).abs().argsort()[:1]]
    if row.empty:
        return None
    col = "ce_ltp" if action == "CALL" else "pe_ltp"
    ltp = float(row.iloc[0][col])
    return ltp if ltp > 0 else None


def _premium_levels(entry: float) -> tuple[float, float, float]:
    target = round(entry * OPTION_TARGET_MULT, 1)
    sl = round(entry * OPTION_SL_MULT, 1)
    risk = entry - sl
    reward = target - entry
    rr = round(reward / risk, 2) if risk > 0 else 0.0
    return target, sl, rr


def _signal_to_action(combined: str, trade_action: str) -> str:
    if combined == "BUY" or trade_action == "LONG":
        return "CALL"
    if combined == "SELL" or trade_action == "SHORT":
        return "PUT"
    return "WAIT"


def build_option_trade(
    *,
    symbol: str,
    price: int,
    combined_signal: str,
    trade_action: str,
    expiry_date: Optional[str],
    expiry_day: Optional[str],
    days_to_expiry: Optional[int],
    index: FnoIndex,
    chain_payload: Optional[dict],
) -> OptionTradeRecommendation:
    """Build one intraday option trade row from index signal + option chain."""
    action = _signal_to_action(combined_signal, trade_action)
    chain_df = option_chain_to_dataframe(chain_payload) if chain_payload else pd.DataFrame()
    spot = float(price)
    strike = _pick_strike(spot, index.strike_step, action, chain_df)

    exp_date, exp_day, dte = expiry_date, expiry_day, days_to_expiry
    if chain_payload and not exp_date:
        exp_date, exp_day, dte = parse_expiry(chain_payload.get("expiry"))

    entry = target = sl = rr = None
    notes = ""

    if action == "WAIT":
        notes = "No clear direction — wait for CALL/PUT setup"
        if not chain_df.empty:
            entry = _ltp_at_strike(chain_df, strike, "CALL")
    else:
        entry = _ltp_at_strike(chain_df, strike, action)
        if entry:
            target, sl, rr = _premium_levels(entry)
            notes = f"Intraday {action} @ strike {strike:,} (premium-based SL/target)"
        elif not index.nse_options:
            notes = "Sensex options on BSE — strike from spot; premium unavailable"
        else:
            notes = "Premium unavailable — check live chain during market hours"
            action = "WAIT"

    return OptionTradeRecommendation(
        index=symbol,
        strike=strike,
        action=action,
        entry_premium=round(entry, 1) if entry else None,
        target=target,
        stop_loss=sl,
        risk_reward=rr,
        expiry_date=exp_date,
        expiry_day=exp_day,
        days_to_expiry=dte,
        notes=notes,
    )


def build_option_trades_from_results(
    results,
    chain_payloads: Dict[str, Optional[dict]],
) -> List[OptionTradeRecommendation]:
    """Build option trade table for all index prediction results."""
    index_by_name = {i.name: i for i in FNO_INDICES}
    trades: List[OptionTradeRecommendation] = []
    for result in results:
        idx = index_by_name.get(result.symbol)
        if not idx:
            continue
        payload = chain_payloads.get(result.symbol)
        trades.append(
            build_option_trade(
                symbol=result.symbol,
                price=result.price,
                combined_signal=result.combined_signal,
                trade_action=result.trade_action,
                expiry_date=result.expiry_date,
                expiry_day=result.expiry_day,
                days_to_expiry=result.days_to_expiry,
                index=idx,
                chain_payload=payload,
            )
        )
    return trades


def option_trades_to_dataframe(trades: List[OptionTradeRecommendation]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    rows = []
    for t in trades:
        expiry_label = "—"
        if t.expiry_date and t.expiry_day:
            dte = f"{t.days_to_expiry}d" if t.days_to_expiry is not None else "—"
            expiry_label = f"{t.expiry_date} ({t.expiry_day}, {dte})"
        rows.append(
            {
                "Index": t.index,
                "Strike": t.strike,
                "Expiry": expiry_label,
                "Action": t.action,
                "Entry (₹)": t.entry_premium,
                "Target (₹)": t.target,
                "Stop Loss (₹)": t.stop_loss,
                "R:R": t.risk_reward,
            }
        )
    return pd.DataFrame(rows)
