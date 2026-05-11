"""MDP definition for adaptive position sizing (Module 5).

State  = (regime_bucket, confidence_bucket, sharpe_bucket, volatility_bucket)
Action = discrete position size  {1%, 5%, 10%, 15%}
Reward = risk-adjusted return proxy for the chosen allocation.
"""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

class PositionAction(IntEnum):
    """Discrete position-size choices (percentage of capital)."""

    PCT_1 = 0
    PCT_5 = 1
    PCT_10 = 2
    PCT_15 = 3


POSITION_PCTS: dict[PositionAction, float] = {
    PositionAction.PCT_1: 0.01,
    PositionAction.PCT_5: 0.05,
    PositionAction.PCT_10: 0.10,
    PositionAction.PCT_15: 0.15,
}

NUM_ACTIONS = len(PositionAction)


# ---------------------------------------------------------------------------
# State buckets
# ---------------------------------------------------------------------------

class RegimeBucket(IntEnum):
    BEARISH = 0
    NEUTRAL = 1
    BULLISH = 2


class ConfidenceBucket(IntEnum):
    LOW = 0       # [0, 0.4)
    MEDIUM = 1    # [0.4, 0.7)
    HIGH = 2      # [0.7, 1.0]


class SharpeBucket(IntEnum):
    NEGATIVE = 0  # < 0
    LOW = 1       # [0, 0.5)
    MEDIUM = 2    # [0.5, 1.0)
    HIGH = 3      # >= 1.0


class VolatilityBucket(IntEnum):
    LOW = 0       # [0, 0.15)
    MEDIUM = 1    # [0.15, 0.30)
    HIGH = 2      # >= 0.30


class State(NamedTuple):
    """Discretized MDP state."""

    regime: RegimeBucket
    confidence: ConfidenceBucket
    sharpe: SharpeBucket
    volatility: VolatilityBucket


NUM_REGIME = len(RegimeBucket)
NUM_CONFIDENCE = len(ConfidenceBucket)
NUM_SHARPE = len(SharpeBucket)
NUM_VOLATILITY = len(VolatilityBucket)
NUM_STATES = NUM_REGIME * NUM_CONFIDENCE * NUM_SHARPE * NUM_VOLATILITY


# ---------------------------------------------------------------------------
# Discretization helpers
# ---------------------------------------------------------------------------

def discretize_regime(regime_str: str) -> RegimeBucket:
    """Map a MarketRegime value string to its bucket."""
    r = regime_str.upper()
    if "BEAR" in r:
        return RegimeBucket.BEARISH
    if "BULL" in r:
        return RegimeBucket.BULLISH
    return RegimeBucket.NEUTRAL


def discretize_confidence(confidence: float) -> ConfidenceBucket:
    if confidence < 0.4:
        return ConfidenceBucket.LOW
    if confidence < 0.7:
        return ConfidenceBucket.MEDIUM
    return ConfidenceBucket.HIGH


def discretize_sharpe(sharpe: float) -> SharpeBucket:
    if sharpe < 0.0:
        return SharpeBucket.NEGATIVE
    if sharpe < 0.5:
        return SharpeBucket.LOW
    if sharpe < 1.0:
        return SharpeBucket.MEDIUM
    return SharpeBucket.HIGH


def discretize_volatility(volatility: float) -> VolatilityBucket:
    if volatility < 0.15:
        return VolatilityBucket.LOW
    if volatility < 0.30:
        return VolatilityBucket.MEDIUM
    return VolatilityBucket.HIGH


def build_state(
    regime: str,
    confidence: float,
    sharpe: float,
    volatility: float,
) -> State:
    """Construct a discretized State from raw continuous inputs."""
    return State(
        regime=discretize_regime(regime),
        confidence=discretize_confidence(confidence),
        sharpe=discretize_sharpe(sharpe),
        volatility=discretize_volatility(volatility),
    )


def state_index(state: State) -> int:
    """Flatten a State tuple into a unique integer for Q-table indexing."""
    idx = state.regime
    idx = idx * NUM_CONFIDENCE + state.confidence
    idx = idx * NUM_SHARPE + state.sharpe
    idx = idx * NUM_VOLATILITY + state.volatility
    return int(idx)


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def compute_reward(
    position_pct: float,
    strategy_return: float,
    max_drawdown: float,
    volatility: float,
) -> float:
    """Risk-adjusted reward for a single sizing decision.

    Balances expected gain against downside risk so the agent learns
    conservative sizing when volatility or drawdown is high.
    """
    gain = position_pct * strategy_return
    risk_penalty = position_pct * abs(max_drawdown) * (1.0 + volatility)
    return gain - 0.5 * risk_penalty
