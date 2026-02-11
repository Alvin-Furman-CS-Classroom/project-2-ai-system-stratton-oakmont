"""Integration tests: Module 1 (Knowledge Base) + Module 2 (Strategy Search)."""

from src.module_2_strategy_search import backtest, search_top_strategies
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
