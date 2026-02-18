"""Backtest engine: OHLCV → indicators → Module 1 rules → returns and Sharpe.

Converts price/volume bars to RSI, MACD, MAs; evaluates rules per bar; computes
position returns and annualized Sharpe ratio.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.shared import MarketIndicators, TradingAction

if TYPE_CHECKING:
    from src.module_1_knowledge_base import HornRule
    from src.module_1_knowledge_base.facts import Params


# Minimum bars needed for indicator computation (MA50 is slowest)
WARMUP_BARS = 50


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI. Returns NaN for first `period` bars."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_macd_line(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """Compute MACD line (fast EMA - slow EMA)."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def indicators_from_ohlcv(ohlcv: pd.DataFrame) -> List[Optional[MarketIndicators]]:
    """
    Compute MarketIndicators for each row of OHLCV history.

    Uses warmup for the first WARMUP_BARS rows (returns None for those).
    Volatility is rolling 20-day std of returns.

    Returns:
        List of MarketIndicators, same length as ohlcv. First WARMUP_BARS are None.
    """
    close = ohlcv["Close"]
    volume = ohlcv["Volume"]

    rsi = _compute_rsi(close)
    macd = _compute_macd_line(close)
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    returns = close.pct_change()
    volatility = returns.rolling(20).std()

    result: List[Optional[MarketIndicators]] = []
    for idx in range(len(ohlcv)):
        if idx < WARMUP_BARS:
            result.append(None)
            continue
        rsi_val = rsi.iloc[idx]
        macd_val = macd.iloc[idx]
        ma20_val = ma20.iloc[idx]
        ma50_val = ma50.iloc[idx]
        vol_val = volatility.iloc[idx]
        if pd.isna(rsi_val) or pd.isna(macd_val) or pd.isna(ma20_val) or pd.isna(ma50_val):
            result.append(None)
            continue
        result.append(
            MarketIndicators(
                rsi=float(rsi_val),
                macd=float(macd_val),
                ma20=float(ma20_val),
                ma50=float(ma50_val),
                volume=float(volume.iloc[idx]),
                volatility=float(vol_val) if not pd.isna(vol_val) else None,
            )
        )
    return result


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Compute annualized Sharpe ratio.

    Assumes daily returns. Annualization factor sqrt(252).
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    excess = returns - (risk_free_rate / 252)
    return float(np.sqrt(252) * excess.mean() / excess.std())


def backtest(
    ohlcv: pd.DataFrame,
    params: Dict[str, float],
    rules: Optional[Sequence[HornRule]] = None,
) -> tuple[np.ndarray, List[TradingAction]]:
    """
    Run a backtest: evaluate rules on each bar, simulate positions, compute returns.

    Assumes we hold a position from close of bar N to close of bar N+1 based on
    the action from bar N. Return = position * (close[t+1]/close[t] - 1).

    Args:
        ohlcv: OHLCV DataFrame.
        params: Params dict for Module 1 fact generation.
        rules: HornRules (defaults to default_trading_rules).

    Returns:
        (returns array, list of actions). Length = len(ohlcv) - 1 - WARMUP_BARS.
    """
    from src.module_1_knowledge_base import evaluate_rules_on_indicators

    if rules is None:
        from src.module_1_knowledge_base import default_trading_rules
        rules = default_trading_rules()

    indicators_list = indicators_from_ohlcv(ohlcv)
    close = ohlcv["Close"].values

    returns_list: List[float] = []
    actions_list: List[TradingAction] = []

    position = 0  # -1 sell, 0 hold, 1 buy
    for idx in range(WARMUP_BARS, len(ohlcv) - 1):
        indicators = indicators_list[idx]
        if indicators is None:
            returns_list.append(0.0)
            actions_list.append(TradingAction.HOLD)
            continue

        result = evaluate_rules_on_indicators(indicators, params=params, rules=rules)
        action = result.action
        actions_list.append(action)

        # Update position
        if action == TradingAction.BUY:
            position = 1
        elif action == TradingAction.SELL:
            position = -1
        # HOLD keeps current position

        # Return = position * (price change)
        ret = (close[idx + 1] / close[idx]) - 1.0
        returns_list.append(float(position * ret))

    return np.array(returns_list), actions_list
