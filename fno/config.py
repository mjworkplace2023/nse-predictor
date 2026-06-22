"""F&O dashboard configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class FnoIndex:
    """One F&O index: display name, yfinance ticker, NSE option-chain symbol."""

    name: str
    yf_symbol: str
    option_symbol: str


# Nifty 50, Bank Nifty, Sensex — only indices traded for F&O intraday
FNO_INDICES: List[FnoIndex] = [
    FnoIndex(name="Nifty 50", yf_symbol="^NSEI", option_symbol="NIFTY"),
    FnoIndex(name="Bank Nifty", yf_symbol="^NSEBANK", option_symbol="BANKNIFTY"),
    FnoIndex(name="Sensex", yf_symbol="^BSESN", option_symbol="SENSEX"),
]

# Label thresholds (forward return)
INTRADAY_FORWARD_BARS = 4          # ~1 hour on 15m chart
INTRADAY_BUY_THRESHOLD = 0.005     # +0.5%
INTRADAY_SELL_THRESHOLD = -0.005   # -0.5%

SWING_FORWARD_DAYS = 5
SWING_BUY_THRESHOLD = 0.01         # +1%
SWING_SELL_THRESHOLD = -0.01

# Options signal thresholds
PCR_BULLISH = 1.2
PCR_BEARISH = 0.7

# Model
TRAIN_TEST_SPLIT = 0.8
RANDOM_STATE = 42
