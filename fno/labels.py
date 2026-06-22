"""Forward-return labels for 3-class BUY / HOLD / SELL."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fno.config import (
    INTRADAY_BUY_THRESHOLD,
    INTRADAY_FORWARD_BARS,
    INTRADAY_SELL_THRESHOLD,
)


LABEL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}
LABEL_TO_INT = {"SELL": 0, "HOLD": 1, "BUY": 2}


def forward_return(close: pd.Series, periods: int) -> pd.Series:
    return close.shift(-periods) / close - 1.0


def label_from_return(ret: float, buy_th: float, sell_th: float) -> int:
    if ret >= buy_th:
        return LABEL_TO_INT["BUY"]
    if ret <= sell_th:
        return LABEL_TO_INT["SELL"]
    return LABEL_TO_INT["HOLD"]


def add_intraday_labels(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Attach 3-class labels based on forward 15m returns."""
    out = feature_df.copy()
    fwd = forward_return(out["Close"], INTRADAY_FORWARD_BARS)
    out["forward_return"] = fwd
    out["label"] = [
        label_from_return(r, INTRADAY_BUY_THRESHOLD, INTRADAY_SELL_THRESHOLD)
        if not np.isnan(r)
        else np.nan
        for r in fwd
    ]
    return out.dropna(subset=["label"])
