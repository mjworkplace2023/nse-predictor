"""Options-specific analytics: PCR, max pain, IV skew, OI buildup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from fno.config import PCR_BEARISH, PCR_BULLISH
from fno.formatting import parse_expiry, round_index


@dataclass
class OptionsSnapshot:
    symbol: str
    spot: float
    pcr: float
    max_pain: float
    iv_skew: float
    ce_oi_top_strike: float
    pe_oi_top_strike: float
    signal: str
    notes: str
    expiry_date: Optional[str] = None
    expiry_day: Optional[str] = None
    days_to_expiry: Optional[int] = None


def compute_pcr(chain_df: pd.DataFrame) -> float:
    ce_oi = chain_df["ce_oi"].sum()
    pe_oi = chain_df["pe_oi"].sum()
    if ce_oi <= 0:
        return 0.0
    return float(pe_oi / ce_oi)


def compute_max_pain(chain_df: pd.DataFrame, spot: float) -> float:
    """Strike where total option seller loss is minimized (classic max pain)."""
    if chain_df.empty:
        return spot
    strikes = chain_df["strike"].astype(float).values
    ce_oi = chain_df["ce_oi"].astype(float).values
    pe_oi = chain_df["pe_oi"].astype(float).values
    pains = []
    for s in strikes:
        ce_loss = np.maximum(0, strikes - s) * ce_oi
        pe_loss = np.maximum(0, s - strikes) * pe_oi
        pains.append(ce_loss.sum() + pe_loss.sum())
    return float(strikes[int(np.argmin(pains))])


def compute_iv_skew(chain_df: pd.DataFrame, spot: float) -> float:
    """PE IV minus CE IV near ATM — positive = fear/bearish skew."""
    if chain_df.empty:
        return 0.0
    atm_idx = (chain_df["strike"] - spot).abs().idxmin()
    atm_row = chain_df.loc[atm_idx]
    return float(atm_row.get("pe_iv", 0) - atm_row.get("ce_iv", 0))


def oi_buildup_levels(chain_df: pd.DataFrame) -> tuple[float, float]:
    """Top CE/PE OI strikes as resistance/support proxies."""
    if chain_df.empty:
        return 0.0, 0.0
    ce_strike = float(chain_df.loc[chain_df["ce_oi"].idxmax(), "strike"])
    pe_strike = float(chain_df.loc[chain_df["pe_oi"].idxmax(), "strike"])
    return ce_strike, pe_strike


def analyze_options(payload: dict, symbol: str = "NIFTY") -> Optional[OptionsSnapshot]:
    from fno.data_fetch import option_chain_to_dataframe

    if not payload:
        return None
    chain_df = option_chain_to_dataframe(payload)
    if chain_df.empty:
        return None

    records = payload.get("records") or {}
    spot = float(
        payload.get("underlyingValue")
        or records.get("underlyingValue")
        or 0
    )
    pcr = compute_pcr(chain_df)
    max_pain = compute_max_pain(chain_df, spot)
    iv_skew = compute_iv_skew(chain_df, spot)
    ce_strike, pe_strike = oi_buildup_levels(chain_df)
    expiry_raw = payload.get("expiry")
    expiry_date, expiry_day, days_to_expiry = parse_expiry(expiry_raw)

    notes = []
    signal = "NEUTRAL"
    if pcr > PCR_BULLISH:
        signal = "BULLISH"
        notes.append(f"PCR {pcr:.2f} > {PCR_BULLISH} (bullish)")
    elif pcr < PCR_BEARISH:
        signal = "BEARISH"
        notes.append(f"PCR {pcr:.2f} < {PCR_BEARISH} (bearish)")

    if spot and max_pain:
        if spot < max_pain * 0.995:
            notes.append(f"Spot below max pain {max_pain:.0f} — upward pull possible")
        elif spot > max_pain * 1.005:
            notes.append(f"Spot above max pain {max_pain:.0f} — downward pull possible")

    if iv_skew > 2:
        notes.append(f"IV skew +{iv_skew:.1f} — put-side fear")
    elif iv_skew < -2:
        notes.append(f"IV skew {iv_skew:.1f} — call-side greed")

    if expiry_date and expiry_day:
        dte = f"{days_to_expiry} day(s)" if days_to_expiry is not None else "—"
        notes.insert(0, f"Expiry {expiry_date} ({expiry_day}), {dte} to expiry")

    notes.append(f"OI resistance CE @ {int(round(ce_strike))}, support PE @ {int(round(pe_strike))}")

    return OptionsSnapshot(
        symbol=symbol,
        spot=spot,
        pcr=round(pcr, 2),
        max_pain=round_index(max_pain) or 0,
        iv_skew=round(iv_skew, 2),
        ce_oi_top_strike=ce_strike,
        pe_oi_top_strike=pe_strike,
        signal=signal,
        notes="; ".join(notes),
        expiry_date=expiry_date,
        expiry_day=expiry_day,
        days_to_expiry=days_to_expiry,
    )
