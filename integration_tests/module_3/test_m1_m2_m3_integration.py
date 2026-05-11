"""Integration tests: Module 1 + 2 + 3 (Evolution)."""

from src.module_2_strategy_search.search import search_top_strategies
from src.module_3_evolution import evolve_from_seeds
from src.shared.market_data import generate_synthetic_ohlcv


def test_m1_m2_m3_integration_runs_end_to_end():
    # Generate synthetic market data for a fast, reproducible test.
    ohlcv = generate_synthetic_ohlcv(days=120, seed=123)

    # Module 2: find candidate strategies.
    seeds = search_top_strategies(ohlcv, top_k=5, method="beam")
    assert len(seeds) == 5

    # Module 3: evolve from the best candidates.
    evolved, summary = evolve_from_seeds(seeds, ohlcv, top_k=5)

    assert len(evolved) == 5
    assert "best_sharpe_per_generation" in summary
    assert summary["generations"] > 0

    # Check that the output strategies have valid parameter keys
    for s in evolved:
        assert isinstance(s.params, dict)
        assert s.params
