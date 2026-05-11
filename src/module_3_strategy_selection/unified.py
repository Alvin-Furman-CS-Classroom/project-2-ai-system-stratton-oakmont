"""Unified selection: chooses between Module 2 and GA-from-scratch strategies.

This module provides the core logic that:
- fetches candidate strategies from Module 2 (beam search)
- fetches candidate strategies from Module 3 GA (evolved from random)
- combines both pools and selects the single best strategy
- produces a human-readable reason explaining why that strategy was chosen
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from src.module_2_strategy_search.search import search_top_strategies
from src.module_3_evolution import GAConfig, evolve_randomly
from src.shared import CandidateStrategy

from .selection import (
    SelectionPreferences,
    StrategyConstraints,
    score_strategy,
    select_strategy,
)


@dataclass
class SelectionResult:
    """Result of the unified selection process."""

    strategy: Optional[CandidateStrategy]
    """The selected strategy, or None if no candidate passed constraints."""

    reason: str
    """Human-readable explanation of why this strategy was chosen (or why none)."""

    origin: Optional[str]
    """Where the winning strategy came from: 'module_2_search' or 'ga_from_scratch'."""

    m2_count: int
    """Number of candidates from Module 2 search."""

    ga_count: int
    """Number of candidates from GA-from-scratch."""

    m2_best_sharpe: float
    """Best Sharpe among Module 2 candidates."""

    ga_best_sharpe: float
    """Best Sharpe among GA-from-scratch candidates."""

    score: float
    """Composite score of the selected strategy (0 if none selected)."""

    summary: str
    """Full selection summary (metrics, constraints, weights)."""


def _build_selection_reason(
    selected: Optional[CandidateStrategy],
    m2_strategies: List[CandidateStrategy],
    ga_strategies: List[CandidateStrategy],
    preferences: SelectionPreferences,
) -> str:
    """Build a human-readable reason explaining why the strategy was chosen."""
    if selected is None:
        return (
            "No strategy was chosen because all candidates failed the hard constraints. "
            "Consider relaxing StrategyConstraints (e.g., min_sharpe, max_drawdown_min) "
            "or expanding the candidate pools (top_k, GA generations)."
        )

    score = score_strategy(selected, preferences)
    origin = "module_2_search" if selected in m2_strategies else "ga_from_scratch"
    origin_label = "Module 2 beam search" if origin == "module_2_search" else "GA-from-scratch evolution"

    # Key strengths: what made this strategy score highest
    strengths: List[str] = []
    if selected.sharpe > 0:
        strengths.append(f"Sharpe ratio {selected.sharpe:+.3f}")
    if selected.total_return > 0:
        strengths.append(f"total return {selected.total_return:+.1%}")
    if selected.max_drawdown > -0.3:
        strengths.append(f"controlled drawdown ({selected.max_drawdown:+.1%})")
    if selected.win_rate >= 0.55:
        strengths.append(f"win rate {selected.win_rate:.0%}")

    strengths_str = ", ".join(strengths) if strengths else "balanced metrics across constraints"

    reason = (
        f"Chosen from {origin_label} because it achieved the highest composite score ({score:.3f}) "
        f"among all {len(m2_strategies) + len(ga_strategies)} candidates. "
        f"Key strengths: {strengths_str}. "
        f"It was selected over "
    )

    if origin == "module_2_search":
        ga_best = max((s.sharpe for s in ga_strategies), default=float("nan"))
        reason += (
            f"the GA-from-scratch pool (best Sharpe there: {ga_best:+.3f})."
        )
    else:
        m2_best = max((s.sharpe for s in m2_strategies), default=float("nan"))
        reason += (
            f"the Module 2 search pool (best Sharpe there: {m2_best:+.3f})."
        )

    return reason


def gather_unified_candidate_pools(
    ohlcv: Any,
    *,
    top_k: int = 5,
    ga_config: Optional[GAConfig] = None,
) -> tuple[List[CandidateStrategy], List[CandidateStrategy], List[CandidateStrategy]]:
    """Fetch Module 2 beam strategies, GA-from-scratch strategies, and their union.

    Use this when you need the combined candidate list more than once (e.g. verbose
    trace + final summary) without running search/GA twice.

    Returns:
        (m2_strategies, ga_strategies, combined) where combined is m2 + ga.
    """
    if ga_config is None:
        ga_config = GAConfig()

    m2_strategies = search_top_strategies(ohlcv, top_k=top_k, method="beam")
    ga_strategies, _ = evolve_randomly(
        ohlcv,
        param_ranges=ga_config.param_ranges,
        config=ga_config,
        top_k=top_k,
    )
    combined = list(m2_strategies) + list(ga_strategies)
    return m2_strategies, ga_strategies, combined


def finalize_unified_selection(
    m2_strategies: List[CandidateStrategy],
    ga_strategies: List[CandidateStrategy],
    combined: List[CandidateStrategy],
    *,
    constraints: StrategyConstraints,
    preferences: SelectionPreferences,
) -> SelectionResult:
    """Select best strategy from pre-built pools and build reason + summary."""
    m2_best_sharpe = max(s.sharpe for s in m2_strategies) if m2_strategies else 0.0
    ga_best_sharpe = max(s.sharpe for s in ga_strategies) if ga_strategies else 0.0

    best = select_strategy(combined, constraints=constraints, preferences=preferences)
    reason = _build_selection_reason(best, m2_strategies, ga_strategies, preferences)
    summary = _summarize_selection(best, combined, constraints, preferences)

    origin: Optional[str] = None
    score = 0.0
    if best is not None:
        origin = "module_2_search" if best in m2_strategies else "ga_from_scratch"
        score = score_strategy(best, preferences)

    return SelectionResult(
        strategy=best,
        reason=reason,
        origin=origin,
        m2_count=len(m2_strategies),
        ga_count=len(ga_strategies),
        m2_best_sharpe=m2_best_sharpe,
        ga_best_sharpe=ga_best_sharpe,
        score=score,
        summary=summary,
    )


def select_best_from_all_sources(
    ohlcv: Any,
    *,
    top_k: int = 5,
    ga_config: Optional[GAConfig] = None,
    constraints: Optional[StrategyConstraints] = None,
    preferences: Optional[SelectionPreferences] = None,
) -> SelectionResult:
    """Select the single best strategy from Module 2 and GA-from-scratch pools.

    This is the core logic that:
    - fetches top_k strategies from Module 2 beam search
    - fetches top_k strategies from GA evolved from random initialization
    - combines both pools and selects the best by composite score
    - produces a reason explaining why that strategy was chosen

    Args:
        ohlcv: Historical market data for backtesting and evaluation.
        top_k: How many strategies to take from each source.
        ga_config: GA hyperparameters. Defaults to GAConfig() if None.
        constraints: Hard filters for strategies. Defaults to StrategyConstraints() if None.
        preferences: Scoring weights and targets. Defaults to SelectionPreferences() if None.

    Returns:
        SelectionResult with strategy, reason, origin, counts, and full summary.
    """
    if ga_config is None:
        ga_config = GAConfig()
    if constraints is None:
        constraints = StrategyConstraints()
    if preferences is None:
        preferences = SelectionPreferences()

    m2_strategies, ga_strategies, combined = gather_unified_candidate_pools(
        ohlcv, top_k=top_k, ga_config=ga_config
    )
    return finalize_unified_selection(
        m2_strategies,
        ga_strategies,
        combined,
        constraints=constraints,
        preferences=preferences,
    )


def _summarize_selection(
    selected: Optional[CandidateStrategy],
    candidates: List[CandidateStrategy],
    constraints: StrategyConstraints,
    preferences: SelectionPreferences,
) -> str:
    """Produce the standard selection summary (avoids circular import from reporting)."""
    from .reporting import summarize_selection

    return summarize_selection(
        selected, candidates, constraints=constraints, preferences=preferences
    )
