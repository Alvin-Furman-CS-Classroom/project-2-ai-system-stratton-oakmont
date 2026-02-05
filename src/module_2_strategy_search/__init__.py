"""Module 2: Strategy Parameter Search (A*, Beam Search).

Searches over parameter ranges to find top configurations by Sharpe ratio.
"""

from .backtest import backtest, indicators_from_ohlcv, sharpe_ratio
from .evaluation import evaluate_candidate
from .search import (
    DEFAULT_PARAM_RANGES,
    beam_search,
    search_top_strategies,
)

__all__ = [
    "DEFAULT_PARAM_RANGES",
    "beam_search",
    "backtest",
    "evaluate_candidate",
    "indicators_from_ohlcv",
    "search_top_strategies",
    "sharpe_ratio",
]
