"""Unit tests for Module 2: Strategy Parameter Search."""

import numpy as np
import pandas as pd
import pytest

from src.module_2_strategy_search import (
    DEFAULT_PARAM_RANGES,
    backtest,
    beam_search,
    evaluate_candidate,
    indicators_from_ohlcv,
    search_top_strategies,
    sharpe_ratio,
)
from src.module_2_strategy_search.backtest import WARMUP_BARS
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
