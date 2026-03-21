"""Unit tests for unified Module 2 + GA-from-scratch selection."""

from __future__ import annotations

from src.module_3_evolution import GAConfig
from src.module_3_strategy_selection import (
    SelectionPreferences,
    StrategyConstraints,
    select_best_from_all_sources,
)
from src.shared.market_data import generate_synthetic_ohlcv


def test_select_best_from_all_sources_returns_result_with_reason():
    ohlcv = generate_synthetic_ohlcv(days=80, seed=7)
    ga_config = GAConfig(population_size=8, generations=3, seed=99)
    constraints = StrategyConstraints(
        min_sharpe=-10.0,
        min_total_return=-1.0,
        min_win_rate=0.0,
        max_drawdown_min=-1.0,
        min_trades=0,
    )

    result = select_best_from_all_sources(
        ohlcv,
        top_k=3,
        ga_config=ga_config,
        constraints=constraints,
        preferences=SelectionPreferences(),
    )

    assert result.strategy is not None
    assert result.origin in ("module_2_search", "ga_from_scratch")
    assert result.m2_count == 3
    assert result.ga_count == 3
    assert "composite score" in result.reason.lower()
    assert "Selected final strategy" in result.summary
