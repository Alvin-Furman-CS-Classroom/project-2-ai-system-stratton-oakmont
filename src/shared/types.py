from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TradingAction(str, Enum):
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

