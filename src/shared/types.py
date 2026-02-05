from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    import pandas as pd

class TradingAction(str, Enum):
    # Strings are easy to print and save (and match the project spec).
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True)
class MarketIndicators:
    """
    Minimal indicator bundle used by Module 1 (and shared by later modules).

    Notes:
    - Keep this lightweight and serializable (dataclass, simple fields).
    - Add fields as needed by later modules (e.g., ATR, returns, etc.).
    """

    rsi: float
    macd: float
    ma20: float
    ma50: float
    volume: float
    volatility: Optional[float] = None


# -----------------------------------------------------------------------------
# Module 2: Strategy Parameter Search
# -----------------------------------------------------------------------------

# Param name -> (min, max) for continuous params. Module 2 searches this space.
ParamRanges = Dict[str, tuple[float, float]]


@dataclass(frozen=True)
class CandidateStrategy:
    """
    A candidate parameter configuration with its backtest performance.

    Handoff from Module 2 to Module 3.
    """

    params: Dict[str, float]
    sharpe: float
    total_return: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    num_trades: int = 0
    explanation: str = ""


# Type alias: pandas DataFrame with columns Open, High, Low, Close, Volume.
# Use for OHLCV history in backtesting.
OHLCVDataFrame = Any  # pandas.DataFrame in practice

