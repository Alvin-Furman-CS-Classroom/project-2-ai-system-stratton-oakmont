"""Demo for Module 3: strategy selection and explanation.

This demo:
- uses Module 2 to generate candidate strategies
- applies Module 3 constraints and preferences
- selects a final strategy and prints a concise summary

Run:
    python -m src.module_3_strategy_selection.demo
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from src.module_2_strategy_search.search import search_top_strategies
from src.module_3_strategy_selection import (
    SelectionPreferences,
    StrategyConstraints,
    print_selection_steps,
    select_strategy,
    summarize_selection,
)
from src.shared.market_data import generate_synthetic_ohlcv


def _ensure_repo_root_on_path() -> None:
    """Make sure the repository root is on sys.path for direct execution."""
    root = pathlib.Path(__file__).resolve().parents[2]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def main(argv: list[str] | None = None) -> None:
    _ensure_repo_root_on_path()

    parser = argparse.ArgumentParser(description="Module 3 selection demo")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed step-by-step selection trace.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=200,
        help="Number of synthetic trading days to generate.",
    )
    args = parser.parse_args(argv)

    # Generate synthetic OHLCV data (Module 1 style).
    ohlcv = generate_synthetic_ohlcv(days=args.days, seed=0)

    # Use Module 2 to search for candidate strategies.
    print("=== Module 3 Selection Demo ===")
    print(f"Generating candidates from Module 2 over {args.days} days of data...")
    candidates = search_top_strategies(ohlcv, top_k=10, method="beam")
    print(f"Received {len(candidates)} candidate strategies.\n")

    # Example constraints and preferences (these can be tuned by the user).
    constraints = StrategyConstraints(
        min_sharpe=0.0,
        min_total_return=0.0,
        min_win_rate=0.50,
        max_drawdown_min=-0.25,
        min_trades=5,
    )
    preferences = SelectionPreferences(
        w_sharpe=1.0,
        w_return=0.5,
        w_win_rate=0.25,
        w_drawdown=0.75,
        w_trades=0.25,
        target_trades=50,
        trade_tolerance=30,
    )

    if args.verbose:
        print("Running verbose selection trace...\n")
        selected = print_selection_steps(
            candidates, constraints=constraints, preferences=preferences
        )
    else:
        selected = select_strategy(
            candidates, constraints=constraints, preferences=preferences
        )

    print("\n=== Summary ===")
    summary = summarize_selection(
        selected, candidates, constraints=constraints, preferences=preferences
    )
    print(summary)


if __name__ == "__main__":
    main()

