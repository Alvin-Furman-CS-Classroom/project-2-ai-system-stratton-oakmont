"""Strategy selection logic for Module 3.

This module operates purely on CandidateStrategy objects produced by Module 2.
It does not re-run backtests; instead, it:
- filters out obviously unsuitable strategies (e.g., extreme drawdowns)
- scores remaining strategies according to user risk preferences
- returns the single best-scoring strategy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from src.shared import CandidateStrategy


@dataclass(frozen=True)
class StrategyConstraints:
    """Hard accept/reject thresholds for strategies.

    All thresholds are inclusive on the "good" side. For example,
    max_drawdown_min = -0.15 means we only accept strategies whose
    max_drawdown is >= -0.15 (no worse than a -15% drawdown).
    """

    min_sharpe: float = 0.0
    min_total_return: float = 0.0
    min_win_rate: float = 0.0
    max_drawdown_min: float = -1.0  # e.g., -0.2 means at most 20% drawdown
    min_trades: int = 5


@dataclass(frozen=True)
class SelectionPreferences:
    """Soft preferences that shape the composite score.

    Weights are positive numbers; higher means "more important". The score
    is:

        score = w_sharpe * sharpe
              + w_return * total_return
              + w_win_rate * win_rate
              + w_trades * trades_term
              + w_drawdown * drawdown_term

    where trades_term and drawdown_term are shaped so that:
    - strategies near the desired trade count are rewarded
    - shallower (less negative) drawdowns are rewarded
    """

    w_sharpe: float = 1.0
    w_return: float = 0.5
    w_win_rate: float = 0.25
    w_drawdown: float = 0.75
    w_trades: float = 0.25

    target_trades: int = 50
    trade_tolerance: int = 30  # how far from target before strong penalty


def _passes_constraints(
    strategy: CandidateStrategy, constraints: StrategyConstraints
) -> bool:
    """Check whether a strategy meets all hard constraints."""
    if strategy.sharpe < constraints.min_sharpe:
        return False
    if strategy.total_return < constraints.min_total_return:
        return False
    if strategy.win_rate < constraints.min_win_rate:
        return False
    if strategy.max_drawdown < constraints.max_drawdown_min:
        return False
    if strategy.num_trades < constraints.min_trades:
        return False
    return True


def filter_candidates(
    candidates: Iterable[CandidateStrategy],
    constraints: Optional[StrategyConstraints] = None,
) -> List[CandidateStrategy]:
    """Return only those strategies that satisfy the given constraints."""
    if constraints is None:
        constraints = StrategyConstraints()
    return [c for c in candidates if _passes_constraints(c, constraints)]


def _drawdown_term(max_drawdown: float) -> float:
    """Convert max drawdown into a reward term.

    max_drawdown is typically negative (e.g., -0.15 for -15%).
    We want shallower drawdowns (closer to 0) to be better.
    """
    # Clamp to [-1, 0] then flip sign so that smaller drawdowns → larger term.
    clamped = max(min(max_drawdown, 0.0), -1.0)
    return 1.0 + clamped  # -1 -> 0, 0 -> 1


def _trades_term(num_trades: int, target: int, tolerance: int) -> float:
    """Reward trade counts near the target and penalize extremes."""
    if target <= 0:
        return 0.0
    # Distance from target scaled by tolerance; beyond ~tolerance we down-weight.
    diff = abs(num_trades - target)
    if diff >= tolerance:
        return 0.0
    # Linear decay from 1 at diff=0 to 0 at diff=tolerance.
    return 1.0 - (diff / max(tolerance, 1))


def score_strategy(
    strategy: CandidateStrategy,
    preferences: Optional[SelectionPreferences] = None,
) -> float:
    """Compute a composite score for a strategy.

    Higher is better. This does not enforce any hard thresholds; use
    StrategyConstraints via filter_candidates for that.
    """
    if preferences is None:
        preferences = SelectionPreferences()

    dd_term = _drawdown_term(strategy.max_drawdown)
    trades_term = _trades_term(
        strategy.num_trades, preferences.target_trades, preferences.trade_tolerance
    )

    score = (
        preferences.w_sharpe * strategy.sharpe
        + preferences.w_return * strategy.total_return
        + preferences.w_win_rate * strategy.win_rate
        + preferences.w_drawdown * dd_term
        + preferences.w_trades * trades_term
    )
    return float(score)


def select_strategy(
    candidates: Iterable[CandidateStrategy],
    constraints: Optional[StrategyConstraints] = None,
    preferences: Optional[SelectionPreferences] = None,
) -> Optional[CandidateStrategy]:
    """Select the single best strategy according to constraints and preferences.

    Returns None if no candidates satisfy the constraints.
    """
    filtered = filter_candidates(candidates, constraints)
    if not filtered:
        return None

    if preferences is None:
        preferences = SelectionPreferences()

    best = max(filtered, key=lambda c: score_strategy(c, preferences))
    return best

