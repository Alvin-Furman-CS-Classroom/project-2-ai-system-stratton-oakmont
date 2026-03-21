"""Unit tests for unified Module 2 + GA-from-scratch selection."""

from __future__ import annotations

from src.module_3_evolution import GAConfig
from src.module_3_strategy_selection import (
    SelectionPreferences,
    StrategyConstraints,
    finalize_unified_selection,
    gather_unified_candidate_pools,
    select_best_from_all_sources,
)
from src.shared import CandidateStrategy
from src.shared.market_data import generate_synthetic_ohlcv


def test_select_best_from_all_sources_returns_result_with_reason():
    ohlcv = generate_synthetic_ohlcv(days=120, seed=7)
    top_k = 3
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
        top_k=top_k,
        ga_config=ga_config,
        constraints=constraints,
        preferences=SelectionPreferences(),
    )

    assert result.strategy is not None
    assert result.origin in ("module_2_search", "ga_from_scratch")
    assert 1 <= result.m2_count <= top_k
    assert result.ga_count == top_k
    assert "composite score" in result.reason.lower()
    assert "Selected final strategy" in result.summary


def test_gather_unified_candidate_pools_sizes_match_top_k():
    ohlcv = generate_synthetic_ohlcv(days=120, seed=11)
    ga_config = GAConfig(population_size=6, generations=2, seed=11)
    top_k = 4
    m2, ga, combined = gather_unified_candidate_pools(
        ohlcv, top_k=top_k, ga_config=ga_config
    )
    # Beam search may return fewer than top_k after diversity filtering.
    assert 1 <= len(m2) <= top_k
    assert len(ga) == top_k
    assert len(combined) == len(m2) + len(ga)


def test_finalize_matches_select_best_from_all_sources():
    """Same pools → same SelectionResult fields as one-shot API."""
    ohlcv = generate_synthetic_ohlcv(days=120, seed=13)
    ga_config = GAConfig(population_size=7, generations=2, seed=13)
    top_k = 3
    constraints = StrategyConstraints(
        min_sharpe=-10.0,
        min_total_return=-1.0,
        min_win_rate=0.0,
        max_drawdown_min=-1.0,
        min_trades=0,
    )
    prefs = SelectionPreferences()

    m2, ga, combined = gather_unified_candidate_pools(
        ohlcv, top_k=top_k, ga_config=ga_config
    )
    a = finalize_unified_selection(
        m2, ga, combined, constraints=constraints, preferences=prefs
    )
    b = select_best_from_all_sources(
        ohlcv,
        top_k=top_k,
        ga_config=ga_config,
        constraints=constraints,
        preferences=prefs,
    )

    assert a.m2_count == b.m2_count
    assert a.ga_count == b.ga_count == top_k
    if a.strategy is not None and b.strategy is not None:
        assert a.strategy.params == b.strategy.params
        assert abs(a.score - b.score) < 1e-9
    assert a.origin == b.origin


def test_select_best_from_all_sources_none_when_constraints_impossible():
    # Backtest requires strictly more than 50 rows (see backtest._validate_ohlcv).
    ohlcv = generate_synthetic_ohlcv(days=80, seed=17)
    ga_config = GAConfig(population_size=6, generations=2, seed=17)
    result = select_best_from_all_sources(
        ohlcv,
        top_k=2,
        ga_config=ga_config,
        constraints=StrategyConstraints(
            min_sharpe=1e9,
            min_total_return=0.0,
            min_win_rate=0.0,
            max_drawdown_min=-1.0,
            min_trades=0,
        ),
        preferences=SelectionPreferences(),
    )
    assert result.strategy is None
    assert result.origin is None
    assert result.score == 0.0
    assert "No strategy was chosen" in result.reason or "failed" in result.reason.lower()
    assert "No strategy satisfied" in result.summary


def test_finalize_unified_selection_explicit_m2_winner():
    m2 = [
        CandidateStrategy(
            params={"x": 1.0},
            sharpe=2.0,
            total_return=0.1,
            win_rate=0.6,
            max_drawdown=-0.1,
            num_trades=50,
        )
    ]
    ga = [
        CandidateStrategy(
            params={"x": 2.0},
            sharpe=0.5,
            total_return=0.05,
            win_rate=0.55,
            max_drawdown=-0.2,
            num_trades=50,
        )
    ]
    combined = m2 + ga
    constraints = StrategyConstraints(
        min_sharpe=0.0,
        min_total_return=0.0,
        min_win_rate=0.5,
        max_drawdown_min=-0.5,
        min_trades=5,
    )
    r = finalize_unified_selection(
        m2, ga, combined, constraints=constraints, preferences=SelectionPreferences()
    )
    assert r.strategy is m2[0]
    assert r.origin == "module_2_search"


def test_finalize_unified_selection_explicit_ga_winner():
    m2 = [
        CandidateStrategy(
            params={"x": 1.0},
            sharpe=0.1,
            total_return=0.01,
            win_rate=0.51,
            max_drawdown=-0.1,
            num_trades=50,
        )
    ]
    ga = [
        CandidateStrategy(
            params={"x": 2.0},
            sharpe=3.0,
            total_return=0.2,
            win_rate=0.6,
            max_drawdown=-0.05,
            num_trades=50,
        )
    ]
    combined = m2 + ga
    constraints = StrategyConstraints(
        min_sharpe=0.0,
        min_total_return=0.0,
        min_win_rate=0.5,
        max_drawdown_min=-0.5,
        min_trades=5,
    )
    r = finalize_unified_selection(
        m2, ga, combined, constraints=constraints, preferences=SelectionPreferences()
    )
    assert r.strategy is ga[0]
    assert r.origin == "ga_from_scratch"
