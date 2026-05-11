"""Module 3: Strategy Selection and Explanation.

Takes CandidateStrategy objects produced by Module 2 and:
- filters them using risk/quality constraints
- scores them with a configurable multi-objective function
- selects a single recommended strategy
- generates a short human-readable summary of the choice

Unified selection (`select_best_from_all_sources`) chooses between Module 2
and GA-from-scratch strategies and returns a reason for the choice.
"""

from .selection import (
    SelectionPreferences,
    StrategyConstraints,
    filter_candidates,
    score_strategy,
    select_strategy,
)
from .reporting import print_selection_steps, summarize_selection
from .unified import (
    SelectionResult,
    finalize_unified_selection,
    gather_unified_candidate_pools,
    select_best_from_all_sources,
)

__all__ = [
    "SelectionPreferences",
    "StrategyConstraints",
    "filter_candidates",
    "score_strategy",
    "select_strategy",
    "summarize_selection",
    "print_selection_steps",
    "SelectionResult",
    "gather_unified_candidate_pools",
    "finalize_unified_selection",
    "select_best_from_all_sources",
]

