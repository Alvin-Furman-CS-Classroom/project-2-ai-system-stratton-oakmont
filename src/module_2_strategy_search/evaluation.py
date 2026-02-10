"""Single-candidate evaluation: params + OHLCV → CandidateStrategy with Sharpe, metrics, explanation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.shared import CandidateStrategy

if TYPE_CHECKING:
    from src.module_1_knowledge_base import HornRule

from .backtest import backtest, sharpe_ratio


def _compute_max_drawdown(returns: np.ndarray) -> float:
    """Max drawdown from cumulative returns."""
    if len(returns) == 0:
        return 0.0
    cum = np.cumprod(1 + returns) - 1
    running_max = np.maximum.accumulate(cum)
    drawdown = (cum - running_max) / (1 + running_max)
    return float(np.min(drawdown))


def _compute_win_rate(returns: np.ndarray) -> float:
    """Fraction of trading periods with positive return."""
    if len(returns) == 0:
        return 0.0
    # Count only periods where we had a position (non-zero return from our side)
    nonzero = returns[returns != 0]
    if len(nonzero) == 0:
        return 0.0
    return float(np.mean(nonzero > 0))


def evaluate_candidate(
    params: Dict[str, float],
    ohlcv: pd.DataFrame,
    rules: Optional[Sequence[HornRule]] = None,
) -> CandidateStrategy:
    """
    Evaluate a parameter configuration by backtesting and return a CandidateStrategy.

    Args:
        params: Params dict for Module 1.
        ohlcv: OHLCV history.
        rules: HornRules (defaults to default_trading_rules).

    Returns:
        CandidateStrategy with params, sharpe, metrics, and a short explanation.
    """
    returns, _actions = backtest(ohlcv, params, rules)

    total_return = float(np.prod(1 + returns) - 1) if len(returns) > 0 else 0.0
    sharpe = sharpe_ratio(returns)
    max_dd = _compute_max_drawdown(returns)
    win_rate = _compute_win_rate(returns)
    num_trades = sum(1 for a in _actions if a.value != "HOLD")

    explanation = (
        f"Sharpe={sharpe:.3f}, Return={total_return:.2%}, "
        f"MaxDD={max_dd:.2%}, WinRate={win_rate:.1%}, Trades={num_trades}"
    )

    return CandidateStrategy(
        params=dict(params),
        sharpe=sharpe,
        total_return=total_return,
        win_rate=win_rate,
        max_drawdown=max_dd,
        num_trades=num_trades,
        explanation=explanation,
    )
