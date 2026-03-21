"""Demo for Module 3: Genetic Algorithm evolution tracing.

This demo runs a small GA instance and prints how the best strategy evolves
across generations, including which parameter values changed.

Run:
    python -m src.module_3_evolution.demo
"""

from __future__ import annotations

import pathlib
import sys
from typing import Dict, List, Optional

# Repo root must be on sys.path BEFORE any `from src...` imports.
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.module_2_strategy_search.search import search_top_strategies
from src.module_3_evolution.evolution import (
    GAConfig,
    _build_next_generation,
    _evaluate_population,
    _mutate,
    _tournament_select,
    _uniform_crossover,
)
from src.shared import CandidateStrategy
from src.shared.market_data import generate_synthetic_ohlcv

# Treat params as "same" if all keys match within this tolerance.
_PARAM_MATCH_TOL = 1e-5


def _params_match_seed(
    params: Dict[str, float], seeds: List[CandidateStrategy]
) -> Optional[CandidateStrategy]:
    """Return the seed strategy whose params match ``params``, or None."""
    for seed in seeds:
        sp = seed.params
        if set(params.keys()) != set(sp.keys()):
            continue
        if all(abs(params[k] - sp[k]) <= _PARAM_MATCH_TOL for k in params):
            return seed
    return None


def _describe_final_origin(
    best: CandidateStrategy, seeds: List[CandidateStrategy]
) -> str:
    """Human-readable line: Module 2 vs GA-evolved (relative to this demo's seeds)."""
    matched = _params_match_seed(best.params, seeds)
    if matched is not None:
        return (
            "Origin: MODULE 2 — final parameters match a beam-search candidate from "
            "Module 2 (same genome as one of the seeds)."
        )
    return (
        "Origin: GA-GENERATED — final parameters are not identical to any Module 2 "
        "seed; they were produced by the genetic algorithm (crossover/mutation), "
        "starting from Module 2 seeds in this demo."
    )


def _param_diff(prev: Dict[str, float], curr: Dict[str, float]) -> List[str]:
    """Return a list of human-readable diffs between two param dicts."""
    diffs: List[str] = []
    keys = sorted(set(prev) | set(curr))
    for k in keys:
        pv = prev.get(k)
        cv = curr.get(k)
        if pv is None:
            diffs.append(f"{k}: +{cv:.4f}")
        elif cv is None:
            diffs.append(f"{k}: -{pv:.4f}")
        elif abs(cv - pv) > 1e-6:
            diffs.append(f"{k}: {pv:.4f} → {cv:.4f}")
    return diffs


def _build_next_generation_with_trace(
    evaluated, config: GAConfig, rng, ranges
):
    """Build the next generation while tracing some of the crossover/mutation events."""
    next_gen: List[Dict[str, float]] = []
    traces: List[str] = []

    evaluated_sorted = sorted(evaluated, key=lambda s: s.sharpe, reverse=True)
    # Keep elites
    next_gen.extend([dict(s.params) for s in evaluated_sorted[: config.elitism]])

    # Generate the rest, recording a few mutation/crossover samples
    sample_limit = 3
    samples = 0

    while len(next_gen) < config.population_size:
        parent_a = _tournament_select(evaluated_sorted, config.tournament_size, rng)
        parent_b = _tournament_select(evaluated_sorted, config.tournament_size, rng)

        if rng.random() < config.crossover_rate:
            child = _uniform_crossover(parent_a.params, parent_b.params, rng)
            op = "crossover"
        else:
            child = dict(parent_a.params)
            op = "copy"

        mutated_child = _mutate(child, ranges, config.mutation_rate, rng)

        if samples < sample_limit:
            diffs = _param_diff(child, mutated_child)
            parent_info = f"A(sharpe={parent_a.sharpe:.3f}) B(sharpe={parent_b.sharpe:.3f})"
            traces.append(
                f"{op} -> mutate: {len(diffs)} changes; {parent_info}; diffs={diffs[:5]}"
            )
            samples += 1

        next_gen.append(mutated_child)

    return next_gen, traces


def main() -> None:
    # Synthetic data keeps the demo deterministic and fast.
    ohlcv = generate_synthetic_ohlcv(days=200, seed=0)

    # Use Module 2 to get seed candidates.
    seeds = search_top_strategies(ohlcv, top_k=5, method="beam")

    config = GAConfig(population_size=16, generations=10, seed=42)

    # Initialize population by mutating seeds (similar to evolve_from_seeds).
    import numpy as np

    rng = np.random.default_rng(config.seed)
    population = []
    while len(population) < config.population_size:
        for s in seeds:
            if len(population) >= config.population_size:
                break
            mutated = _mutate(s.params, config.param_ranges, config.mutation_rate, rng)
            population.append(mutated)

    prev_best_params: Dict[str, float] | None = None

    print("=== GA Demo: Evolution Trace ===")
    print(
        "Population is seeded from Module 2 beam search; the GA refines parameters.\n"
        "At the end we report whether the winner still matches a Module 2 candidate "
        "or is GA-evolved.\n"
    )
    best: CandidateStrategy | None = None
    for gen in range(config.generations):
        evaluated = _evaluate_population(population, ohlcv, rules=None)
        best = max(evaluated, key=lambda s: s.sharpe)

        print(f"\nGeneration {gen:02d}: best Sharpe = {best.sharpe:.4f}")

        if prev_best_params is not None:
            diffs = _param_diff(prev_best_params, best.params)
            if diffs:
                print("  Changes (best strategy compared to previous best):")
                for d in diffs[:6]:
                    print(f"    - {d}")
            else:
                print("  (No changes in best params)")

        prev_best_params = best.params

        population, traces = _build_next_generation_with_trace(
            evaluated, config, rng, config.param_ranges
        )
        print("  Sample genetic events:")
        for t in traces:
            print(f"    - {t}")

    print("\n=== Final best strategy ===")
    assert best is not None
    print(_describe_final_origin(best, seeds))
    print(f"Sharpe: {best.sharpe:.4f}")
    print("Params:")
    for k, v in sorted(best.params.items()):
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
