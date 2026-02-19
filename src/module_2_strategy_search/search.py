"""Beam Search (and A* stub) over param space; returns top-k by Sharpe."""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.shared import CandidateStrategy, ParamRanges

if TYPE_CHECKING:
    from src.module_1_knowledge_base import HornRule

from .evaluation import evaluate_candidate


# Default parameter ranges for search. Module 2 searches within these bounds.
DEFAULT_PARAM_RANGES: ParamRanges = {
    "rsi_oversold": (20.0, 40.0),
    "rsi_overbought": (60.0, 80.0),
    "rsi_neutral_low": (35.0, 45.0),
    "rsi_neutral_high": (55.0, 65.0),
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
    step_fraction: float = 0.05,
) -> List[Dict[str, float]]:
    """
    Generate neighboring parameter configs by perturbing one param at a time.

    For each param, add ±step_fraction of (max-min). Returns distinct neighbors.
    """
    neighbors: List[Dict[str, float]] = []
    for key, (low, high) in ranges.items():
        if key not in params:
            continue
        step = step_fraction * (high - low)
        for delta in (-step, step):
            new_params = dict(params)
            new_params[key] = params[key] + delta
            new_params = _clamp_params(new_params, ranges)
            neighbors.append(new_params)
    return neighbors


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
    max_expansions: int = 50,
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
    # Start from center of parameter ranges
    start = {k: (lo + hi) / 2 for k, (lo, hi) in param_ranges.items()}
    start_candidate = evaluate_candidate(start, ohlcv, rules)

    h = _heuristic(start, start_candidate.sharpe, param_ranges)

    # Min-heap on negative estimated value so highest (sharpe + h) is popped first.
    counter = 0
    open_set: List[Tuple[float, int, Dict[str, float]]] = []
    heapq.heappush(open_set, (-(start_candidate.sharpe + h), counter, start))
    counter += 1

    closed: set[Tuple[Tuple[str, float], ...]] = set()
    # Map from param_key → best CandidateStrategy seen for that config.
    evaluated: Dict[Tuple[Tuple[str, float], ...], CandidateStrategy] = {
        _param_key(start): start_candidate
    }

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

    # Collect all evaluated strategies, sort by Sharpe, return top-k
    results = sorted(evaluated.values(), key=lambda c: c.sharpe, reverse=True)
    return results[:top_k]


# ---------------------------------------------------------------------------
# Beam Search
# ---------------------------------------------------------------------------


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

    Start from center of ranges, expand neighbors, keep top-k by Sharpe per iteration.
    """
    # Center of ranges as initial state
    center = {k: (lo + hi) / 2 for k, (lo, hi) in param_ranges.items()}
    beam: List[Dict[str, float]] = [center]

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

        # Evaluate and keep top beam_width
        scored = [evaluate_candidate(p, ohlcv, rules) for p in unique]
        scored.sort(key=lambda c: c.sharpe, reverse=True)
        beam_params = [s.params for s in scored[:beam_width]]
        beam = beam_params

    # Final top_k from last beam
    final = [evaluate_candidate(p, ohlcv, rules) for p in beam]
    final.sort(key=lambda c: c.sharpe, reverse=True)
    return final[:top_k]


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
