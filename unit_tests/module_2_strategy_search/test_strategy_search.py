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
from src.module_2_strategy_search.search import (
    _clamp_params,
    _diverse_starting_points,
    _diversity_filter,
    _get_successors,
    _heuristic,
    _param_key,
)
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


# -----------------------------------------------------------------------------
# _clamp_params
# -----------------------------------------------------------------------------


def test_clamp_params_within_range_unchanged():
    """Params already within range are returned unchanged."""
    ranges = {"a": (0.0, 10.0), "b": (1.0, 5.0)}
    params = {"a": 5.0, "b": 3.0}
    result = _clamp_params(params, ranges)
    assert result == params


def test_clamp_params_clips_to_bounds():
    """Params outside range are clamped to the nearest bound."""
    ranges = {"a": (0.0, 10.0), "b": (1.0, 5.0)}
    params = {"a": -5.0, "b": 99.0}
    result = _clamp_params(params, ranges)
    assert result["a"] == 0.0
    assert result["b"] == 5.0


def test_clamp_params_ignores_extra_keys():
    """Params not in ranges pass through untouched."""
    ranges = {"a": (0.0, 10.0)}
    params = {"a": 15.0, "extra": 42.0}
    result = _clamp_params(params, ranges)
    assert result["a"] == 10.0
    assert result["extra"] == 42.0


# -----------------------------------------------------------------------------
# _get_successors
# -----------------------------------------------------------------------------


def test_get_successors_returns_neighbors():
    """Successors are generated for each param (±step)."""
    ranges = {"x": (0.0, 10.0), "y": (0.0, 1.0)}
    params = {"x": 5.0, "y": 0.5}
    neighbors = _get_successors(params, ranges, step_fraction=0.25)
    # At least 2 single-param perturbations per param → 4 minimum
    assert len(neighbors) >= 4


def test_get_successors_stay_within_range():
    """All successors have params clamped within valid ranges."""
    ranges = {"x": (0.0, 10.0), "y": (0.0, 1.0)}
    params = {"x": 0.0, "y": 1.0}  # at boundaries
    neighbors = _get_successors(params, ranges, step_fraction=0.5)
    for n in neighbors:
        assert 0.0 <= n["x"] <= 10.0
        assert 0.0 <= n["y"] <= 1.0


def test_get_successors_two_param_perturbations():
    """When key trading params are present, two-param perturbations are generated."""
    ranges = {
        "rsi_oversold": (0.0, 30.0),
        "rsi_overbought": (70.0, 100.0),
        "macd_epsilon": (0.0, 0.1),
    }
    params = {k: (lo + hi) / 2 for k, (lo, hi) in ranges.items()}
    neighbors = _get_successors(params, ranges)
    # 3 params × 2 single = 6, plus C(3,2)=3 pairs × 4 combos = 12 → 18 total
    assert len(neighbors) == 18


# -----------------------------------------------------------------------------
# _diverse_starting_points
# -----------------------------------------------------------------------------


def test_diverse_starting_points_returns_requested_count():
    """Returns exactly num_points starting configurations."""
    ranges = {"a": (0.0, 10.0), "b": (1.0, 5.0)}
    points = _diverse_starting_points(ranges, num_points=5)
    assert len(points) == 5


def test_diverse_starting_points_first_is_center():
    """The first point is the center of each range."""
    ranges = {"a": (0.0, 10.0), "b": (2.0, 8.0)}
    points = _diverse_starting_points(ranges, num_points=1)
    assert points[0]["a"] == pytest.approx(5.0)
    assert points[0]["b"] == pytest.approx(5.0)


def test_diverse_starting_points_within_bounds():
    """All starting points have values within their parameter ranges."""
    ranges = {"x": (0.0, 10.0), "y": (5.0, 15.0), "z": (-1.0, 1.0)}
    points = _diverse_starting_points(ranges, num_points=10)
    for p in points:
        for k, (lo, hi) in ranges.items():
            assert lo <= p[k] <= hi, f"{k}={p[k]} outside [{lo}, {hi}]"


def test_diverse_starting_points_deterministic_with_seed():
    """Same seed produces identical starting points."""
    ranges = {"a": (0.0, 10.0), "b": (1.0, 5.0)}
    p1 = _diverse_starting_points(ranges, num_points=5, seed=99)
    p2 = _diverse_starting_points(ranges, num_points=5, seed=99)
    assert p1 == p2


# -----------------------------------------------------------------------------
# _diversity_filter
# -----------------------------------------------------------------------------


def test_diversity_filter_empty_returns_empty():
    """Empty input returns empty list."""
    assert _diversity_filter([], max_keep=5) == []


def test_diversity_filter_respects_max_keep():
    """Never returns more than max_keep strategies."""
    strategies = [
        CandidateStrategy(params={"a": float(i)}, sharpe=float(i) * 0.1)
        for i in range(20)
    ]
    result = _diversity_filter(strategies, max_keep=5)
    assert len(result) <= 5


def test_diversity_filter_limits_same_sharpe_bucket():
    """Strategies with identical Sharpe are capped per bucket."""
    # All have the same Sharpe → should not fill all slots
    strategies = [
        CandidateStrategy(params={"a": float(i)}, sharpe=1.0)
        for i in range(10)
    ]
    result = _diversity_filter(strategies, max_keep=6)
    assert len(result) <= 6
    # max_per_bucket = max(2, 6//3) = 2, so at most 2 from the one bucket
    assert len(result) <= 2


def test_diversity_filter_keeps_diverse_sharpes():
    """Strategies with different Sharpe values all survive."""
    strategies = [
        CandidateStrategy(params={"a": 1.0}, sharpe=1.0),
        CandidateStrategy(params={"a": 2.0}, sharpe=0.5),
        CandidateStrategy(params={"a": 3.0}, sharpe=0.0),
    ]
    result = _diversity_filter(strategies, max_keep=3)
    assert len(result) == 3


def test_diversity_filter_sorted_descending():
    """Output is sorted by Sharpe ratio descending."""
    strategies = [
        CandidateStrategy(params={"a": 1.0}, sharpe=0.2),
        CandidateStrategy(params={"a": 2.0}, sharpe=0.8),
        CandidateStrategy(params={"a": 3.0}, sharpe=0.5),
    ]
    result = _diversity_filter(strategies, max_keep=3)
    sharpes = [s.sharpe for s in result]
    assert sharpes == sorted(sharpes, reverse=True)


def test_diversity_filter_deduplicates_identical_params():
    """Strategies with identical params are deduplicated regardless of Sharpe bucket."""
    strategies = [
        CandidateStrategy(params={"a": 1.0, "b": 2.0}, sharpe=1.0),
        CandidateStrategy(params={"a": 1.0, "b": 2.0}, sharpe=1.0),
        CandidateStrategy(params={"a": 3.0, "b": 4.0}, sharpe=0.5),
    ]
    result = _diversity_filter(strategies, max_keep=5)
    # Only one copy of the duplicate params should survive
    param_keys = [tuple(sorted(s.params.items())) for s in result]
    assert len(param_keys) == len(set(param_keys))


# -----------------------------------------------------------------------------
# Backtest input validation
# -----------------------------------------------------------------------------


def test_backtest_rejects_non_dataframe():
    """Passing a non-DataFrame raises TypeError with a clear message."""
    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        backtest({"Close": [1, 2, 3]}, {"rsi_oversold": 30.0})


def test_backtest_rejects_missing_columns():
    """DataFrame missing required OHLCV columns raises ValueError."""
    df = pd.DataFrame({"Close": [100.0] * 60, "Volume": [1000] * 60})
    with pytest.raises(ValueError, match="missing required columns"):
        backtest(df, {"rsi_oversold": 30.0})


def test_backtest_rejects_too_few_rows():
    """DataFrame with <= WARMUP_BARS rows raises ValueError."""
    ohlcv = generate_synthetic_ohlcv(days=WARMUP_BARS, seed=10)
    with pytest.raises(ValueError, match="more than"):
        backtest(ohlcv, {"rsi_oversold": 30.0})


def test_backtest_minimal_valid_ohlcv():
    """WARMUP_BARS + 2 rows should produce at least 1 return."""
    ohlcv = generate_synthetic_ohlcv(days=WARMUP_BARS + 2, seed=11)
    returns, actions = backtest(ohlcv, {"rsi_oversold": 30.0})
    assert len(returns) >= 1
    assert len(actions) == len(returns)


# -----------------------------------------------------------------------------
# Evaluate candidate edge cases
# -----------------------------------------------------------------------------


def test_evaluate_candidate_all_hold_produces_zero_trades():
    """Params that produce all HOLD actions result in num_trades=0."""
    ohlcv = generate_synthetic_ohlcv(days=100, seed=12)
    # RSI thresholds set impossibly: oversold at 0, overbought at 100 → nothing triggers
    extreme_params = {
        "rsi_oversold": 0.0,
        "rsi_overbought": 100.0,
        "macd_epsilon": 100.0,
        "macd_strong_threshold": 200.0,
        "ma_crossover_margin": 1.0,
        "volume_high": 1e12,
        "volume_surge_multiplier": 1e6,
        "volume_average": 1e12,
        "volatility_high": 0.0,
        "volatility_low": 0.0,
    }
    result = evaluate_candidate(extreme_params, ohlcv)
    assert isinstance(result, CandidateStrategy)
    assert result.num_trades == 0
    assert result.sharpe == 0.0


def test_evaluate_candidate_metrics_are_finite():
    """All returned metrics should be finite numbers, not NaN or Inf."""
    ohlcv = generate_synthetic_ohlcv(days=120, seed=13)
    result = evaluate_candidate({"rsi_oversold": 30.0, "rsi_overbought": 70.0}, ohlcv)
    assert np.isfinite(result.sharpe)
    assert np.isfinite(result.total_return)
    assert np.isfinite(result.win_rate)
    assert np.isfinite(result.max_drawdown)


# -----------------------------------------------------------------------------
# search_top_strategies error handling
# -----------------------------------------------------------------------------


def test_search_top_strategies_invalid_method_raises():
    """Unknown method raises ValueError with a clear message."""
    ohlcv = generate_synthetic_ohlcv(days=80, seed=14)
    with pytest.raises(ValueError, match="Unknown search method"):
        search_top_strategies(ohlcv, top_k=3, method="invalid")


# -----------------------------------------------------------------------------
# Beam vs A* comparison
# -----------------------------------------------------------------------------


def test_beam_and_astar_both_find_strategies():
    """Both search methods should return non-empty results on the same data."""
    ohlcv = generate_synthetic_ohlcv(days=120, seed=15)
    beam_results = search_top_strategies(ohlcv, top_k=3, method="beam")
    astar_results = search_top_strategies(ohlcv, top_k=3, method="astar")
    assert len(beam_results) >= 1
    assert len(astar_results) >= 1

