"""Public API for Module 5: position sizing from upstream module outputs.

Consumes Module 4 (regime + confidence) and Module 3 (CandidateStrategy)
plus volatility and capital to produce a recommended position size with
Q-value explanation and risk assessment.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.module_4_sentiment.pipeline import SentimentAnalysisResult
from src.shared.types import CandidateStrategy

from .agent import PositionSizingAgent, QAgentConfig
from .mdp import (
    POSITION_PCTS,
    PositionAction,
    State,
    build_state,
)


@dataclass(frozen=True)
class PositionSizingResult:
    """Module 5 output — the final system recommendation."""

    position_pct: float
    action: PositionAction
    q_values: dict[PositionAction, float]
    state: State
    capital: float
    dollar_amount: float
    reasoning: str
    risk_assessment: str


def _risk_assessment(state: State, position_pct: float) -> str:
    """Short narrative about why this size is appropriate for the risk level."""
    vol_label = state.volatility.name.lower()
    regime_label = state.regime.name.lower()
    if position_pct <= 0.01:
        return (f"Minimal 1% allocation under {vol_label} volatility / "
                f"{regime_label} regime — preserves capital.")
    if position_pct >= 0.15:
        return (f"Maximum 15% allocation: {regime_label} regime with "
                f"{state.confidence.name.lower()} confidence supports aggressive sizing.")
    return (f"{position_pct:.0%} allocation balances opportunity in a {regime_label} "
            f"regime against {vol_label} volatility risk.")


def _reasoning(action: PositionAction, q_values: dict[PositionAction, float]) -> str:
    """Human-readable explanation of the agent's choice."""
    ranked = sorted(q_values.items(), key=lambda kv: kv[1], reverse=True)
    best = ranked[0]
    lines = [f"Selected {POSITION_PCTS[best[0]]:.0%} (Q = {best[1]:+.4f})."]
    if len(ranked) > 1:
        alts = ", ".join(f"{POSITION_PCTS[a]:.0%}={q:+.4f}" for a, q in ranked[1:])
        lines.append(f"Alternatives: {alts}.")
    return " ".join(lines)


def recommend_position_size(
    sentiment: SentimentAnalysisResult,
    strategy: CandidateStrategy,
    *,
    volatility: float,
    capital: float = 10_000.0,
    agent: PositionSizingAgent | None = None,
) -> PositionSizingResult:
    """Produce a position-sizing recommendation from upstream module outputs.

    Args:
        sentiment: Module 4 result (regime + confidence).
        strategy: Selected strategy from Module 3.
        volatility: Current market volatility estimate (e.g. annualized std).
        capital: Available capital in dollars.
        agent: Pre-trained Q-learning agent.  A default (untrained) agent is
               used when ``None`` — useful for integration tests and demos
               before training is wired up.
    """
    if agent is None:
        agent = PositionSizingAgent(QAgentConfig())

    state = build_state(
        regime=sentiment.regime.value,
        confidence=sentiment.confidence,
        sharpe=strategy.sharpe,
        volatility=volatility,
    )

    action = agent.select_action(state, greedy=True)
    pct = POSITION_PCTS[action]
    q_vals = agent.q_values_for(state)

    return PositionSizingResult(
        position_pct=pct,
        action=action,
        q_values=q_vals,
        state=state,
        capital=capital,
        dollar_amount=round(capital * pct, 2),
        reasoning=_reasoning(action, q_vals),
        risk_assessment=_risk_assessment(state, pct),
    )
