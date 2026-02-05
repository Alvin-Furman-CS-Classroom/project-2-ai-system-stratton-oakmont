"""A* and Beam Search for strategy parameter optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

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
    step_fraction: float = 0.1,
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
        method: "beam" (Beam Search) or "astar" (A* - stub for now).

    Returns:
        Top k CandidateStrategies ranked by Sharpe ratio.
    """
    ranges = param_ranges or DEFAULT_PARAM_RANGES

    if method == "beam":
        return beam_search(ohlcv, ranges, rules, top_k=top_k)
    if method == "astar":
        # A* stub: use beam search for now
        return beam_search(ohlcv, ranges, rules, top_k=top_k)
    raise ValueError(f"Unknown search method: {method!r}")
