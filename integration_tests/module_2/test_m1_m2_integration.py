"""Integration tests: Module 1 (Knowledge Base) + Module 2 (Strategy Search)."""

import numpy as np

from src.module_2_strategy_search import backtest, evaluate_candidate, search_top_strategies
from src.module_2_strategy_search.backtest import sharpe_ratio
from src.shared import CandidateStrategy
from src.shared.market_data import generate_synthetic_ohlcv


def test_m1_m2_backtest_produces_actions():
    """M1 evaluate_rules_on_indicators + M2 backtest produce valid actions."""
    ohlcv = generate_synthetic_ohlcv(days=100, seed=42)
    params = {"rsi_oversold": 30.0, "rsi_overbought": 70.0}

    returns, actions = backtest(ohlcv, params)

    assert len(returns) > 0
    assert len(actions) == len(returns)
    valid_actions = {"BUY", "SELL", "HOLD"}
    assert all(a.value in valid_actions for a in actions)


def test_m1_m2_search_returns_top_candidates():
    """M2 search returns top strategies usable by M3."""
    ohlcv = generate_synthetic_ohlcv(days=120, seed=123)
    top = search_top_strategies(ohlcv, top_k=5)

    assert len(top) <= 5
    for strategy in top:
        assert strategy.params
        assert hasattr(strategy, "sharpe")
        assert hasattr(strategy, "explanation")


def test_m1_m2_candidate_strategy_has_m3_handoff_fields():
    """CandidateStrategy includes all fields Module 3 needs for GA seeding."""
    ohlcv = generate_synthetic_ohlcv(days=120, seed=200)
    top = search_top_strategies(ohlcv, top_k=3, method="beam")
    for strategy in top:
        assert isinstance(strategy, CandidateStrategy)
        assert isinstance(strategy.params, dict)
        assert isinstance(strategy.sharpe, float)
        assert isinstance(strategy.total_return, float)
        assert isinstance(strategy.win_rate, float)
        assert isinstance(strategy.max_drawdown, float)
        assert isinstance(strategy.num_trades, int)
        assert isinstance(strategy.explanation, str)
        assert len(strategy.explanation) > 0


def test_m1_m2_backtest_returns_finite_values():
    """All backtest returns are finite (no NaN or Inf from indicator computation)."""
    ohlcv = generate_synthetic_ohlcv(days=150, seed=300)
    params = {"rsi_oversold": 25.0, "rsi_overbought": 75.0}
    returns, _ = backtest(ohlcv, params)
    assert np.all(np.isfinite(returns)), "Backtest produced non-finite returns"


def test_m1_m2_search_strategies_have_valid_params():
    """All returned strategies have params within DEFAULT_PARAM_RANGES."""
    from src.module_2_strategy_search import DEFAULT_PARAM_RANGES
    ohlcv = generate_synthetic_ohlcv(days=120, seed=400)
    top = search_top_strategies(ohlcv, top_k=5, method="astar")
    for strategy in top:
        for key, (lo, hi) in DEFAULT_PARAM_RANGES.items():
            if key in strategy.params:
                assert lo <= strategy.params[key] <= hi, (
                    f"{key}={strategy.params[key]} outside [{lo}, {hi}]"
                )
