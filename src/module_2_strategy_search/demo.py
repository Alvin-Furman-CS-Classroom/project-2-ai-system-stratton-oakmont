"""Demo: Module 2 Strategy Parameter Search."""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from src.module_2_strategy_search import search_top_strategies
from src.shared.market_data import generate_synthetic_ohlcv


def main() -> None:
    print("=== Module 2: Strategy Parameter Search Demo ===\n")
    print("Generating synthetic OHLCV (252 days)...")
    ohlcv = generate_synthetic_ohlcv(days=252, seed=42)
    print(f"History: {len(ohlcv)} bars\n")

    print("Running Beam Search for top 5 strategies...")
    top = search_top_strategies(ohlcv, top_k=5, method="beam")
    print(f"\nTop {len(top)} strategies by Sharpe ratio:\n")

    for idx, strategy in enumerate(top, 1):
        print(f"  {idx}. Sharpe={strategy.sharpe:.3f} | {strategy.explanation}")


if __name__ == "__main__":
    main()
