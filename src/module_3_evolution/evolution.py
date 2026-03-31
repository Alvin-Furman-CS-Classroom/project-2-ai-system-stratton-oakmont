"""Genetic algorithm engine for evolving trading strategy parameters.

This module provides two main entry points:

- evolve_from_seeds: starts from Module 2 candidates and evolves them.
- evolve_randomly: starts from random parameters and evolves them.

Both use Module 2's evaluation pipeline (`evaluate_candidate`) as the fitness function.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.module_2_strategy_search.evaluation import evaluate_candidate
from src.module_2_strategy_search.search import DEFAULT_PARAM_RANGES
from src.shared import CandidateStrategy, ParamRanges


@dataclass(frozen=True)
class GAConfig:
    """Configuration for the genetic algorithm.
    
       - Mutation rate favors exploration (0.2) to help escape local optima
       - Crossover rate (0.8) promotes combining good traits
    """

    population_size: int = 20
    generations: int = 200
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    elitism: int = 2
    tournament_size: int = 3
    immigrant_fraction: float = 0.0
    seed: Optional[int] = None
    restarts: int = 1
    param_ranges: ParamRanges = field(default_factory=lambda: DEFAULT_PARAM_RANGES.copy())


def _clamp_params(params: Dict[str, float], ranges: ParamRanges) -> Dict[str, float]:
    """Clamp params to their allowed ranges."""
    result = dict(params)
    for key, (low, high) in ranges.items():
        if key in result:
            result[key] = max(low, min(high, result[key]))
    return result


def _random_params(rng: np.random.Generator, ranges: ParamRanges) -> Dict[str, float]:
    """Sample a random point in the parameter space."""
    return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in ranges.items()}


def _uniform_crossover(
    parent_a: Dict[str, float],
    parent_b: Dict[str, float],
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Uniform crossover: each gene is chosen randomly from one parent."""
    child: Dict[str, float] = {}
    keys = set(parent_a.keys()) | set(parent_b.keys())
    for k in keys:
        if rng.random() < 0.5 and k in parent_a:
            child[k] = parent_a[k]
        elif k in parent_b:
            child[k] = parent_b[k]
        else:
            # Fallback in case one parent is missing a key (shouldn't happen)
            child[k] = parent_a.get(k, parent_b.get(k, 0.0))
    return child


def _mutate(
    params: Dict[str, float],
    ranges: ParamRanges,
    mutation_rate: float,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Mutate parameters by perturbing some values within their range."""
    mutated = dict(params)
    for k, (low, high) in ranges.items():
        if rng.random() < mutation_rate:
            # Perturb by up to ±10% of the range
            span = high - low
            delta = rng.uniform(-0.1 * span, 0.1 * span)
            mutated[k] = mutated.get(k, (low + high) / 2) + delta
    return _clamp_params(mutated, ranges)


def _tournament_select(
    population: List[CandidateStrategy],
    tournament_size: int,
    rng: np.random.Generator,
) -> CandidateStrategy:
    """Tournament selection (higher sharpe wins)."""
    if tournament_size <= 1 or tournament_size >= len(population):
        # If tournament size is invalid, fall back to best strategy.
        return max(population, key=lambda s: s.sharpe)

    competitors = rng.choice(population, size=tournament_size, replace=False)
    return max(competitors, key=lambda s: s.sharpe)


def _evaluate_population(
    params_list: List[Dict[str, float]],
    ohlcv: Any,
    rules: Optional[Any],
) -> List[CandidateStrategy]:
    return [evaluate_candidate(p, ohlcv, rules) for p in params_list]


def _build_next_generation(
    evaluated: List[CandidateStrategy],
    config: GAConfig,
    rng: np.random.Generator,
    ranges: ParamRanges,
) -> List[Dict[str, float]]:
    """Build the next generation of parameter configs."""
    # Elitism: keep top N
    evaluated_sorted = sorted(evaluated, key=lambda s: s.sharpe, reverse=True)
    effective_elitism = max(
        0,
        min(config.elitism, config.population_size, len(evaluated_sorted)),
    )
    next_gen: List[Dict[str, float]] = [
        dict(s.params) for s in evaluated_sorted[:effective_elitism]
    ]

    # Optional diversity injection: disabled by default (immigrant_fraction=0.0).
    remaining_slots = config.population_size - len(next_gen)
    immigrant_count = int(round(config.immigrant_fraction * config.population_size))
    immigrant_count = max(0, min(immigrant_count, remaining_slots))
    for _ in range(immigrant_count):
        next_gen.append(_random_params(rng, ranges))

    # Fill the rest of the population
    while len(next_gen) < config.population_size:
        # Parent selection
        parent_a = _tournament_select(evaluated_sorted, config.tournament_size, rng)
        parent_b = _tournament_select(evaluated_sorted, config.tournament_size, rng)

        # Crossover
        if rng.random() < config.crossover_rate:
            child = _uniform_crossover(parent_a.params, parent_b.params, rng)
        else:
            child = dict(parent_a.params)

        # Mutation
        child = _mutate(child, ranges, config.mutation_rate, rng)

        next_gen.append(child)

    return next_gen


def _run_ga(
    initial_params: List[Dict[str, float]],
    ohlcv: Any,
    rules: Optional[Any],
    config: GAConfig,
) -> Tuple[List[CandidateStrategy], Dict[str, Any]]:
    """Run GA for config.generations and return top strategies + summary."""
    rng = np.random.default_rng(config.seed)
    ranges = config.param_ranges

    population = [ _clamp_params(p, ranges) for p in initial_params ]

    best_per_gen: List[float] = []

    for _gen in range(config.generations):
        evaluated = _evaluate_population(population, ohlcv, rules)
        best_per_gen.append(max(s.sharpe for s in evaluated))
        population = _build_next_generation(evaluated, config, rng, ranges)

    final_evaluated = _evaluate_population(population, ohlcv, rules)
    final_sorted = sorted(final_evaluated, key=lambda s: s.sharpe, reverse=True)

    summary = {
        "best_sharpe_per_generation": best_per_gen,
        "best_final_sharpe": final_sorted[0].sharpe if final_sorted else 0.0,
        "generations": config.generations,
        "population_size": config.population_size,
    }

    return final_sorted, summary


def _seed_for_restart(base_seed: Optional[int], restart_idx: int) -> Optional[int]:
    """Derive deterministic per-restart seeds when a base seed is provided."""
    if base_seed is None:
        return None
    return int(base_seed + restart_idx)


def evolve_from_seeds(
    seeds: List[CandidateStrategy],
    ohlcv: Any,
    rules: Optional[Any] = None,
    config: Optional[GAConfig] = None,
    top_k: int = 5,
) -> Tuple[List[CandidateStrategy], Dict[str, Any]]:
    """Evolve strategies starting from Module 2 candidates.

    Args:
        seeds: Seed candidate strategies (e.g., from Module 2).
        ohlcv: Historical market data (OHLCV) for backtesting.
        rules: Optional trading rules. Defaults to Module 1 rules.
        config: Genetic algorithm hyperparameters.
        top_k: How many final strategies to return.

    Returns:
        Tuple of (top_strategies, summary).
    """
    cfg = config or GAConfig()
    restart_count = max(1, int(cfg.restarts))

    # Start population with seed params; if no seeds, start from one random genome.
    seed_params = [dict(s.params) for s in seeds]
    if not seed_params:
        fallback_seed = _seed_for_restart(cfg.seed, 0)
        seed_params = [_random_params(np.random.default_rng(fallback_seed), cfg.param_ranges)]

    best_sorted: List[CandidateStrategy] = []
    best_summary: Dict[str, Any] = {}
    best_sharpe = float("-inf")
    selected_restart_idx = 0

    for restart_idx in range(restart_count):
        restart_seed = _seed_for_restart(cfg.seed, restart_idx)
        restart_cfg = replace(cfg, seed=restart_seed)

        population: List[Dict[str, float]] = []
        rng = np.random.default_rng(restart_seed)
        while len(population) < restart_cfg.population_size:
            for p in seed_params:
                if len(population) >= restart_cfg.population_size:
                    break
                mutated = _mutate(p, restart_cfg.param_ranges, restart_cfg.mutation_rate, rng)
                population.append(mutated)

        final_sorted, summary = _run_ga(population, ohlcv, rules, restart_cfg)
        run_best = float(summary.get("best_final_sharpe", float("-inf")))
        if run_best > best_sharpe:
            best_sharpe = run_best
            best_sorted = final_sorted
            best_summary = summary
            selected_restart_idx = restart_idx

    best_summary["restarts"] = restart_count
    best_summary["selected_restart"] = selected_restart_idx
    return best_sorted[:top_k], best_summary


def evolve_randomly(
    ohlcv: Any,
    param_ranges: Optional[ParamRanges] = None,
    rules: Optional[Any] = None,
    config: Optional[GAConfig] = None,
    top_k: int = 5,
) -> Tuple[List[CandidateStrategy], Dict[str, Any]]:
    """Evolve strategies starting from a random population."""
    cfg = config or GAConfig()
    ranges = param_ranges or cfg.param_ranges
    restart_count = max(1, int(cfg.restarts))

    best_sorted: List[CandidateStrategy] = []
    best_summary: Dict[str, Any] = {}
    best_sharpe = float("-inf")
    selected_restart_idx = 0

    for restart_idx in range(restart_count):
        restart_seed = _seed_for_restart(cfg.seed, restart_idx)
        restart_cfg = replace(cfg, seed=restart_seed, param_ranges=ranges)

        rng = np.random.default_rng(restart_seed)
        population = [_random_params(rng, ranges) for _ in range(restart_cfg.population_size)]

        final_sorted, summary = _run_ga(population, ohlcv, rules, restart_cfg)
        run_best = float(summary.get("best_final_sharpe", float("-inf")))
        if run_best > best_sharpe:
            best_sharpe = run_best
            best_sorted = final_sorted
            best_summary = summary
            selected_restart_idx = restart_idx

    best_summary["restarts"] = restart_count
    best_summary["selected_restart"] = selected_restart_idx
    return best_sorted[:top_k], best_summary
