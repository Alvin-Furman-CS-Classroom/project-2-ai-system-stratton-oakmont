"""Unit tests for Module 2: Strategy Parameter Search."""

import numpy as np
import pandas as pd
import pytest

from src.module_2_strategy_search import (
    DEFAULT_PARAM_RANGES,
    astar_search,
    backtest,
    beam_search,
    evaluate_candidate,
    indicators_from_ohlcv,
    search_top_strategies,
    sharpe_ratio,
)
from src.module_2_strategy_search.backtest import WARMUP_BARS
from src.module_2_strategy_search.search import _heuristic, _param_key
from src.shared import CandidateStrategy
from src.shared.market_data import generate_synthetic_ohlcv


# -----------------------------------------------------------------------------
# Sharpe ratio
# -----------------------------------------------------------------------------


def test_sharpe_ratio_empty_returns_zero():
    """Empty returns array returns 0."""
    assert sharpe_ratio(np.array([])) == 0.0


def test_sharpe_ratio_constant_returns_zero():
    """Constant returns (zero std) returns 0."""
    assert sharpe_ratio(np.array([0.01, 0.01, 0.01])) == 0.0


def test_sharpe_ratio_positive_returns_positive():
    """Positive mean returns with variance give positive Sharpe."""
    returns = np.array([0.02, -0.01, 0.03, 0.01, -0.005])
    assert sharpe_ratio(returns) > 0


# -----------------------------------------------------------------------------
# Indicators from OHLCV
# -----------------------------------------------------------------------------


def test_indicators_from_ohlcv_warmup_is_none():
    """First WARMUP_BARS indicators are None."""
    ohlcv = generate_synthetic_ohlcv(days=100, seed=1)
    indicators = indicators_from_ohlcv(ohlcv)
    assert all(ind is None for ind in indicators[:WARMUP_BARS])
    assert indicators[WARMUP_BARS] is not None


def test_indicators_from_ohlcv_has_required_fields():
    """Indicators have rsi, macd, ma20, ma50, volume, volatility."""
    ohlcv = generate_synthetic_ohlcv(days=100, seed=1)
    indicators = indicators_from_ohlcv(ohlcv)
    valid = [i for i in indicators if i is not None]
    assert len(valid) > 0
    ind = valid[0]
    assert hasattr(ind, "rsi") and hasattr(ind, "macd")
    assert hasattr(ind, "ma20") and hasattr(ind, "ma50")
    assert hasattr(ind, "volume")


# -----------------------------------------------------------------------------
# Backtest
# -----------------------------------------------------------------------------


def test_backtest_returns_same_length_as_tradable_bars():
    """Backtest returns have length = bars - 1 - WARMUP_BARS."""
    ohlcv = generate_synthetic_ohlcv(days=100, seed=1)
    params = {"rsi_oversold": 30.0, "rsi_overbought": 70.0}
    returns, actions = backtest(ohlcv, params)
    expected_len = len(ohlcv) - 1 - WARMUP_BARS
    assert len(returns) == expected_len
    assert len(actions) == expected_len


# -----------------------------------------------------------------------------
# Evaluate candidate
# -----------------------------------------------------------------------------


def test_evaluate_candidate_returns_candidate_strategy():
    """evaluate_candidate returns CandidateStrategy with expected fields."""
    ohlcv = generate_synthetic_ohlcv(days=100, seed=1)
    params = {"rsi_oversold": 30.0, "rsi_overbought": 70.0}
    result = evaluate_candidate(params, ohlcv)
    assert isinstance(result, CandidateStrategy)
    assert result.params == params
    assert hasattr(result, "sharpe")
    assert hasattr(result, "explanation")
    assert "Sharpe=" in result.explanation


# -----------------------------------------------------------------------------
# Search
# -----------------------------------------------------------------------------


def test_beam_search_returns_top_k():
    """Beam search returns at most top_k strategies."""
    ohlcv = generate_synthetic_ohlcv(days=120, seed=2)
    top = beam_search(ohlcv, DEFAULT_PARAM_RANGES, top_k=3, num_iterations=2)
    assert len(top) <= 3


def test_search_top_strategies_returns_list():
    """search_top_strategies returns list of CandidateStrategy."""
    ohlcv = generate_synthetic_ohlcv(days=120, seed=3)
    top = search_top_strategies(ohlcv, top_k=3)
    assert isinstance(top, list)
    assert all(isinstance(s, CandidateStrategy) for s in top)
    assert len(top) <= 3


def test_search_top_strategies_ordered_by_sharpe():
    """Results are ordered by Sharpe descending."""
    ohlcv = generate_synthetic_ohlcv(days=120, seed=4)
    top = search_top_strategies(ohlcv, top_k=5)
    sharpes = [s.sharpe for s in top]
    assert sharpes == sorted(sharpes, reverse=True)


# -----------------------------------------------------------------------------
# A* helpers
# -----------------------------------------------------------------------------


def test_param_key_is_hashable_and_deterministic():
    """_param_key produces a consistent hashable tuple regardless of dict order."""
    params_a = {"rsi_oversold": 30.0, "rsi_overbought": 70.0}
    params_b = {"rsi_overbought": 70.0, "rsi_oversold": 30.0}
    assert _param_key(params_a) == _param_key(params_b)
    # Can be used in a set
    s = {_param_key(params_a)}
    assert _param_key(params_b) in s


def test_heuristic_nonnegative():
    """Heuristic must always be >= 0 (admissible lower bound on improvement)."""
    ranges = {"a": (0.0, 10.0), "b": (0.0, 1.0)}
    # Center of ranges → maximum room
    assert _heuristic({"a": 5.0, "b": 0.5}, 1.0, ranges) >= 0.0
    # At boundary → less room
    assert _heuristic({"a": 0.0, "b": 0.0}, 1.0, ranges) >= 0.0
    # Negative Sharpe
    assert _heuristic({"a": 5.0, "b": 0.5}, -2.0, ranges) >= 0.0


def test_heuristic_higher_at_center_than_boundary():
    """Center of ranges has more room, so heuristic should be larger."""
    ranges = {"x": (0.0, 10.0)}
    h_center = _heuristic({"x": 5.0}, 1.0, ranges)
    h_edge = _heuristic({"x": 0.0}, 1.0, ranges)
    assert h_center > h_edge


def test_heuristic_empty_ranges_returns_zero():
    """Empty param_ranges should return 0."""
    assert _heuristic({"x": 5.0}, 1.0, {}) == 0.0


# -----------------------------------------------------------------------------
# A* Search
# -----------------------------------------------------------------------------


def test_astar_search_returns_top_k():
    """A* search returns at most top_k strategies."""
    ohlcv = generate_synthetic_ohlcv(days=120, seed=5)
    top = astar_search(ohlcv, DEFAULT_PARAM_RANGES, top_k=3, max_expansions=10)
    assert len(top) <= 3
    assert all(isinstance(s, CandidateStrategy) for s in top)


def test_astar_search_ordered_by_sharpe():
    """A* results are sorted by Sharpe ratio descending."""
    ohlcv = generate_synthetic_ohlcv(days=120, seed=6)
    top = astar_search(ohlcv, DEFAULT_PARAM_RANGES, top_k=5, max_expansions=10)
    sharpes = [s.sharpe for s in top]
    assert sharpes == sorted(sharpes, reverse=True)


def test_astar_search_has_valid_params():
    """Every returned strategy should have params within the defined ranges."""
    ohlcv = generate_synthetic_ohlcv(days=120, seed=7)
    top = astar_search(ohlcv, DEFAULT_PARAM_RANGES, top_k=5, max_expansions=10)
    for strategy in top:
        for key, (lo, hi) in DEFAULT_PARAM_RANGES.items():
            if key in strategy.params:
                assert lo <= strategy.params[key] <= hi, (
                    f"{key}={strategy.params[key]} outside [{lo}, {hi}]"
                )


def test_astar_search_single_expansion():
    """A* with max_expansions=1 expands only the start node."""
    ohlcv = generate_synthetic_ohlcv(days=120, seed=8)
    top = astar_search(ohlcv, DEFAULT_PARAM_RANGES, top_k=10, max_expansions=1)
    # Should still return results (at least the start + its neighbors)
    assert len(top) >= 1


def test_search_top_strategies_astar_method():
    """search_top_strategies with method='astar' routes to A* search."""
    ohlcv = generate_synthetic_ohlcv(days=120, seed=9)
    top = search_top_strategies(ohlcv, top_k=3, method="astar")
    assert isinstance(top, list)
    assert all(isinstance(s, CandidateStrategy) for s in top)
    assert len(top) <= 3
