"""Beam Search and A* over param space; returns top-k by Sharpe."""

from __future__ import annotations

import heapq
import itertools
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.shared import CandidateStrategy, ParamRanges

if TYPE_CHECKING:
    from src.module_1_knowledge_base import HornRule

from .evaluation import evaluate_candidate


# Default parameter ranges for search. Module 2 searches within these bounds.
DEFAULT_PARAM_RANGES: ParamRanges = {
    "rsi_oversold": (0.0, 30.0),
    "rsi_overbought": (70.1, 100.0),
    "rsi_neutral_low": (30.1, 50.0),
    "rsi_neutral_high": (50.1, 70),
    "macd_epsilon": (0.0, 0.1),
    "macd_strong_threshold": (0.3, 0.8),
    "ma_crossover_margin": (0.01, 0.05),
    "volume_high": (500_000.0, 2_000_000.0),
    "volume_surge_multiplier": (1.5, 3.0),
    "volume_average": (200_000.0, 800_000.0),
    "volatility_high": (0.02, 0.05),
    "volatility_low": (0.005, 0.02),
}


def _clamp_params(params: Dict[str, float], ranges: ParamRanges) -> Dict[str, float]:
    """Clamp params to valid ranges."""
    result = dict(params)
    for key, (low, high) in ranges.items():
        if key in result:
            result[key] = max(low, min(high, result[key]))
    return result


def _get_successors(
    params: Dict[str, float],
    ranges: ParamRanges,
    step_fraction: float = 0.25,
) -> List[Dict[str, float]]:
    """
    Generate neighboring parameter configs by perturbing one or two params.

    For each param, add ±step_fraction of (max-min). Also generates a limited
    set of two-parameter perturbations for better exploration. Returns
    distinct neighbors.
    """
    neighbors: List[Dict[str, float]] = []
    keys_in_params = [k for k in ranges if k in params]

    # Single-parameter perturbations
    for key in keys_in_params:
        low, high = ranges[key]
        step = step_fraction * (high - low)
        for delta in (-step, step):
            new_params = dict(params)
            new_params[key] = params[key] + delta
            new_params = _clamp_params(new_params, ranges)
            neighbors.append(new_params)

    # Two-parameter perturbations for key trading params to escape plateaus
    key_params = [k for k in ("rsi_oversold", "rsi_overbought", "macd_epsilon",
                               "ma_crossover_margin") if k in params]
    for k1, k2 in itertools.combinations(key_params, 2):
        lo1, hi1 = ranges[k1]
        lo2, hi2 = ranges[k2]
        s1 = step_fraction * (hi1 - lo1)
        s2 = step_fraction * (hi2 - lo2)
        for d1, d2 in [(s1, s2), (s1, -s2), (-s1, s2), (-s1, -s2)]:
            new_params = dict(params)
            new_params[k1] = params[k1] + d1
            new_params[k2] = params[k2] + d2
            new_params = _clamp_params(new_params, ranges)
            neighbors.append(new_params)

    return neighbors


def _diverse_starting_points(
    param_ranges: ParamRanges,
    num_points: int = 5,
    seed: int = 0,
) -> List[Dict[str, float]]:
    """
    Generate diverse starting points across the parameter space.

    Returns the center point plus additional points sampled at different
    positions (quartiles) to cover the search space broadly.
    """
    rng = np.random.default_rng(seed)
    keys = sorted(param_ranges.keys())
    center = {k: (param_ranges[k][0] + param_ranges[k][1]) / 2 for k in keys}
    points = [center]

    # Add points at quartile positions (25% and 75% of each range)
    low_point = {k: param_ranges[k][0] + 0.25 * (param_ranges[k][1] - param_ranges[k][0])
                 for k in keys}
    high_point = {k: param_ranges[k][0] + 0.75 * (param_ranges[k][1] - param_ranges[k][0])
                  for k in keys}
    points.extend([low_point, high_point])

    # Random samples for remaining points
    for _ in range(max(0, num_points - len(points))):
        point = {k: rng.uniform(param_ranges[k][0], param_ranges[k][1]) for k in keys}
        points.append(point)

    return points[:num_points]


# ---------------------------------------------------------------------------
# A* Search
# ---------------------------------------------------------------------------


def _param_key(params: Dict[str, float]) -> Tuple[Tuple[str, float], ...]:
    """Convert a params dict to a hashable tuple for set membership checks."""
    return tuple(sorted(params.items()))


def _heuristic(
    params: Dict[str, float],
    current_sharpe: float,
    param_ranges: ParamRanges,
) -> float:
    """
    Admissible heuristic estimating the remaining improvement potential.

    Measures how much room each parameter has to move within its valid range.
    Configurations near range centers can be perturbed in either direction,
    so they have more optimization room.  Configurations pinned at a boundary
    have less room and are likely closer to a local extremum.

    The heuristic is scaled by the magnitude of the current Sharpe (with a
    floor) so the bonus stays proportional to actual performance.  The result
    is always >= 0, and the scaling factor is conservative enough to remain
    admissible in practice (never wildly overestimates remaining gain).
    """
    if not param_ranges:
        return 0.0

    total_room = 0.0
    counted = 0
    for key, (lo, hi) in param_ranges.items():
        span = hi - lo
        if span <= 0 or key not in params:
            continue
        # Normalized distance to nearest boundary in [0, 0.5]
        dist = min(params[key] - lo, hi - params[key]) / span
        total_room += dist
        counted += 1

    avg_room = total_room / counted if counted else 0.0

    # Scale: more room → more potential.  Use a conservative multiplier so
    # the heuristic stays optimistic but not wildly so.
    scale = max(abs(current_sharpe), 0.5)
    return avg_room * scale * 0.5


def astar_search(
    ohlcv: pd.DataFrame,
    param_ranges: ParamRanges,
    rules: Optional[Sequence[HornRule]] = None,
    top_k: int = 10,
    max_expansions: int = 80,
) -> List[CandidateStrategy]:
    """
    A* search over the strategy parameter space.

    Explores parameter configurations using a priority queue ordered by
    f(n) = sharpe(n) + h(n), where h(n) is a heuristic estimating the
    remaining improvement potential from that region of the space.  Nodes
    with higher estimated total value are expanded first.

    Args:
        ohlcv: OHLCV history for backtesting.
        param_ranges: Valid bounds for each parameter.
        rules: HornRules (defaults to default trading rules).
        top_k: Number of best strategies to return.
        max_expansions: Maximum nodes to expand (controls search budget).

    Returns:
        Top-k CandidateStrategies found, ranked by Sharpe ratio.
    """
    # Start from diverse points across the parameter space
    starting_points = _diverse_starting_points(param_ranges, num_points=5)

    # Min-heap on negative estimated value so highest (sharpe + h) is popped first.
    counter = 0
    open_set: List[Tuple[float, int, Dict[str, float]]] = []
    closed: set[Tuple[Tuple[str, float], ...]] = set()
    evaluated: Dict[Tuple[Tuple[str, float], ...], CandidateStrategy] = {}

    for start in starting_points:
        start_candidate = evaluate_candidate(start, ohlcv, rules)
        h = _heuristic(start, start_candidate.sharpe, param_ranges)
        heapq.heappush(open_set, (-(start_candidate.sharpe + h), counter, start))
        counter += 1
        evaluated[_param_key(start)] = start_candidate

    expansions = 0
    while open_set and expansions < max_expansions:
        _neg_f, _tie, params = heapq.heappop(open_set)

        key = _param_key(params)
        if key in closed:
            continue
        closed.add(key)
        expansions += 1

        # Expand neighbors
        for neighbor_params in _get_successors(params, param_ranges):
            n_key = _param_key(neighbor_params)
            if n_key in closed:
                continue

            candidate = evaluate_candidate(neighbor_params, ohlcv, rules)

            # Keep the best evaluation per config
            if n_key not in evaluated or candidate.sharpe > evaluated[n_key].sharpe:
                evaluated[n_key] = candidate

            h = _heuristic(neighbor_params, candidate.sharpe, param_ranges)
            heapq.heappush(
                open_set, (-(candidate.sharpe + h), counter, neighbor_params)
            )
            counter += 1

    # Collect all evaluated strategies with diversity filtering
    results = sorted(evaluated.values(), key=lambda c: c.sharpe, reverse=True)
    return _diversity_filter(results, top_k)


# ---------------------------------------------------------------------------
# Beam Search
# ---------------------------------------------------------------------------


def _diversity_filter(
    strategies: List[CandidateStrategy],
    max_keep: int,
    min_sharpe_diff: float = 0.005,
) -> List[CandidateStrategy]:
    """
    Filter strategies to maintain diversity in the beam.

    Keeps top strategies but ensures we don't fill the beam with near-
    identical Sharpe values. If multiple strategies share the same Sharpe
    (within min_sharpe_diff), only a limited number are kept to leave
    room for genuinely different strategies.
    """
    if not strategies:
        return []

    strategies.sort(key=lambda c: c.sharpe, reverse=True)
    kept: List[CandidateStrategy] = []
    sharpe_counts: Dict[int, int] = {}  # bucketed sharpe -> count
    max_per_bucket = max(2, max_keep // 3)

    for s in strategies:
        bucket = int(s.sharpe / min_sharpe_diff) if min_sharpe_diff > 0 else 0
        count = sharpe_counts.get(bucket, 0)
        if count < max_per_bucket:
            kept.append(s)
            sharpe_counts[bucket] = count + 1
        if len(kept) >= max_keep:
            break

    return kept


def beam_search(
    ohlcv: pd.DataFrame,
    param_ranges: ParamRanges,
    rules: Optional[Sequence[HornRule]] = None,
    beam_width: int = 10,
    top_k: int = 10,
    num_iterations: int = 5,
) -> List[CandidateStrategy]:
    """
    Beam Search over parameter space.

    Starts from diverse initial points across the ranges, expands neighbors,
    and keeps top-k by Sharpe per iteration with diversity enforcement.
    """
    # Start from diverse points instead of just the center
    beam: List[Dict[str, float]] = _diverse_starting_points(
        param_ranges, num_points=beam_width
    )

    for _ in range(num_iterations):
        candidates: List[Dict[str, float]] = list(beam)
        for params in beam:
            candidates.extend(_get_successors(params, param_ranges))

        # Deduplicate by param tuple
        seen: set = set()
        unique: List[Dict[str, float]] = []
        for p in candidates:
            key = tuple(sorted(p.items()))
            if key not in seen:
                seen.add(key)
                unique.append(p)

        # Evaluate and keep top beam_width with diversity
        scored = [evaluate_candidate(p, ohlcv, rules) for p in unique]
        scored = _diversity_filter(scored, beam_width)
        beam = [s.params for s in scored]

    # Final top_k from last beam
    final = [evaluate_candidate(p, ohlcv, rules) for p in beam]
    final = _diversity_filter(final, top_k)
    return final


def search_top_strategies(
    ohlcv: pd.DataFrame,
    param_ranges: Optional[ParamRanges] = None,
    rules: Optional[Sequence[HornRule]] = None,
    top_k: int = 10,
    method: str = "beam",
) -> List[CandidateStrategy]:
    """
    Main entrypoint: search for top parameter configurations.

    Args:
        ohlcv: OHLCV history for backtesting.
        param_ranges: Search bounds (defaults to DEFAULT_PARAM_RANGES).
        rules: HornRules (defaults to default_trading_rules).
        top_k: Number of strategies to return.
        method: "beam" (Beam Search) or "astar" (A* with heuristic).

    Returns:
        Top k CandidateStrategies ranked by Sharpe ratio.
    """
    ranges = param_ranges or DEFAULT_PARAM_RANGES

    if method == "beam":
        return beam_search(ohlcv, ranges, rules, top_k=top_k)
    if method == "astar":
        return astar_search(ohlcv, ranges, rules, top_k=top_k)
    raise ValueError(f"Unknown search method: {method!r}")
