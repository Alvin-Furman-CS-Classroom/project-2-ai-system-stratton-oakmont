"""Module 3 full demo: unified selection + optional verbose trace.

This demo does **both**:

1. **Unified selection** — builds candidates from **Module 2** (beam search) and
   **GA-from-scratch** (``evolve_randomly``), merges them, picks one winner with
   constraints/preferences, and prints **summary + why it was chosen**.
2. **Verbose trace** (``--verbose``) — runs ``print_selection_steps`` on the **same**
   combined pool so you see receive → filter → score → rank → select.

Run:
    python -m src.module_3_strategy_selection.demo
    python -m src.module_3_strategy_selection.demo --verbose
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# Repo root must be on sys.path before any `from src...` imports.
_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.module_3_evolution import GAConfig
from src.module_3_strategy_selection import (
    SelectionPreferences,
    StrategyConstraints,
    finalize_unified_selection,
    gather_unified_candidate_pools,
    print_selection_steps,
)
from src.shared.market_data import generate_synthetic_ohlcv


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Module 3: Module 2 + GA-from-scratch selection and explanation."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print step-by-step selection trace on the combined candidate pool.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=200,
        help="Number of synthetic trading days to generate.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many strategies to take from Module 2 and from GA each.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=20,
        help="GA generations for the from-scratch run.",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=20,
        help="GA population size.",
    )
    args = parser.parse_args(argv)

    ohlcv = generate_synthetic_ohlcv(days=args.days, seed=0)

    ga_config = GAConfig(
        population_size=args.population_size,
        generations=args.generations,
        seed=42,
    )
    constraints = StrategyConstraints(
        min_sharpe=0.0,
        min_total_return=0.0,
        min_win_rate=0.50,
        max_drawdown_min=-0.30,
        min_trades=5,
    )
    preferences = SelectionPreferences()

    print("=== Module 3 Full Demo ===")
    print(
        f"Synthetic data: {args.days} days | top_k={args.top_k} per source | "
        f"GA pop={args.population_size}, generations={args.generations}\n"
    )

    m2_list, ga_list, combined = gather_unified_candidate_pools(
        ohlcv, top_k=args.top_k, ga_config=ga_config
    )

    m2_best = max(s.sharpe for s in m2_list) if m2_list else 0.0
    ga_best = max(s.sharpe for s in ga_list) if ga_list else 0.0
    print(
        f"Candidates: {len(m2_list)} from Module 2 (best Sharpe {m2_best:+.3f}), "
        f"{len(ga_list)} from GA-from-scratch (best Sharpe {ga_best:+.3f}), "
        f"{len(combined)} total.\n"
    )

    if args.verbose:
        print("=== Verbose selection trace (combined pool) ===\n")
        print_selection_steps(
            combined, constraints=constraints, preferences=preferences
        )
        print()

    result = finalize_unified_selection(
        m2_list,
        ga_list,
        combined,
        constraints=constraints,
        preferences=preferences,
    )

    print("=== Selection result (summary) ===")
    print(result.summary)
    print()
    print("=== Why this strategy was chosen ===")
    print(result.reason)
    print()
    if result.origin:
        print(f"Winner source: {result.origin}")
    print(
        f"Best Sharpe by pool — Module 2: {result.m2_best_sharpe:+.3f}, "
        f"GA-from-scratch: {result.ga_best_sharpe:+.3f}"
    )


if __name__ == "__main__":
    main()
