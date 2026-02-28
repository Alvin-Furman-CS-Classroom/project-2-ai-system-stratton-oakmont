"""Module 3: Strategy Selection and Explanation.

Takes CandidateStrategy objects produced by Module 2 and:
- filters them using risk/quality constraints
- scores them with a configurable multi-objective function
- selects a single recommended strategy
- generates a short human-readable summary of the choice
"""

from .selection import (
    SelectionPreferences,
    StrategyConstraints,
    filter_candidates,
    score_strategy,
    select_strategy,
)
from .reporting import print_selection_steps, summarize_selection

__all__ = [
    "SelectionPreferences",
    "StrategyConstraints",
    "filter_candidates",
    "score_strategy",
    "select_strategy",
    "summarize_selection",
    "print_selection_steps",
]

