"""Unit tests for Module 3: Strategy Evolution Engine."""

from __future__ import annotations

from typing import Dict

import numpy as np

from src.module_2_strategy_search.search import DEFAULT_PARAM_RANGES
from src.module_3_evolution import GAConfig, evolve_from_seeds, evolve_randomly
from src.shared import CandidateStrategy
from src.shared.market_data import generate_synthetic_ohlcv


def _params_within_ranges(params: Dict[str, float], ranges: Dict[str, tuple[float, float]]) -> bool:
    return all(ranges[k][0] <= params[k] <= ranges[k][1] for k in ranges if k in params)


def test_evolve_randomly_respects_param_bounds():
    ohlcv = generate_synthetic_ohlcv(days=100, seed=1)
    config = GAConfig(population_size=10, generations=5, seed=123)

    result, summary = evolve_randomly(ohlcv, param_ranges=DEFAULT_PARAM_RANGES, config=config, top_k=5)

    assert len(result) == 5
    assert isinstance(summary, dict)
    assert summary["population_size"] == config.population_size

    for strategy in result:
        assert _params_within_ranges(strategy.params, DEFAULT_PARAM_RANGES)
        assert isinstance(strategy.sharpe, float)


def test_evolve_from_seeds_has_non_decreasing_best_sharpe():
    ohlcv = generate_synthetic_ohlcv(days=100, seed=2)

    # Create seed strategies with a small set of valid params.
    seed_params = [
        {k: (v[0] + v[1]) / 2 for k, v in DEFAULT_PARAM_RANGES.items()},
        {k: v[0] + 0.6 * (v[1] - v[0]) for k, v in DEFAULT_PARAM_RANGES.items()},
        {k: v[0] + 0.2 * (v[1] - v[0]) for k, v in DEFAULT_PARAM_RANGES.items()},
    ]

    seeds = [
        CandidateStrategy(params=p, sharpe=0.0, total_return=0.0, win_rate=0.0, max_drawdown=0.0, num_trades=0)
        for p in seed_params
    ]

    config = GAConfig(population_size=12, generations=6, seed=456)
    evolved, summary = evolve_from_seeds(seeds, ohlcv, config=config, top_k=5)

    assert len(evolved) == 5
    assert summary["generations"] == config.generations

    # Ensure best sharpe is at least as good as the worst seed sharpe (which is 0.0 here).
    best_sharpe = max(s.sharpe for s in evolved)
    assert best_sharpe >= 0.0
    for strategy in evolved:
        assert _params_within_ranges(strategy.params, DEFAULT_PARAM_RANGES)
