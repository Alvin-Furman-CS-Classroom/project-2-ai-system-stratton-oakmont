"""Demo: Compare Beam Search vs A* on synthetic OHLCV → top strategies by Sharpe."""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from src.module_2_strategy_search import search_top_strategies
from src.shared.market_data import generate_synthetic_ohlcv


def print_strategy_details(strategy, idx):
    """Print detailed info about a strategy."""
    print(f"\n  Strategy {idx}:")
    print(f"    Sharpe: {strategy.sharpe:.3f}")
    print(f"    Return: {strategy.total_return:.2%}")
    print(f"    Max Drawdown: {strategy.max_drawdown:.2%}")
    print(f"    Win Rate: {strategy.win_rate:.1%}")
    print(f"    Trades: {strategy.num_trades}")
    print(f"    Key Params:")
    print(f"      rsi_oversold: {strategy.params.get('rsi_oversold', 'N/A'):.2f}")
    print(f"      rsi_overbought: {strategy.params.get('rsi_overbought', 'N/A'):.2f}")
    print(f"      macd_epsilon: {strategy.params.get('macd_epsilon', 'N/A'):.4f}")
    print(f"      ma_crossover_margin: {strategy.params.get('ma_crossover_margin', 'N/A'):.4f}")


def run_search_comparison(ohlcv, top_k=5):
    """Run both search methods and compare results."""
    
    methods = ["beam", "astar"]
    results = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Running {method.upper()} Search...")
        print('='*60)
        
        strategies = search_top_strategies(ohlcv, top_k=top_k, method=method)
        results[method] = strategies
        
        print(f"\nTop {len(strategies)} strategies from {method.upper()}:")
        for idx, strategy in enumerate(strategies, 1):
            print_strategy_details(strategy, idx)
    
    return results


def compare_best_strategies(results):
    """Compare the best strategy from each method."""
    print(f"\n{'='*60}")
    print("COMPARISON: Best Strategy from Each Method")
    print('='*60)
    
    for method, strategies in results.items():
        if strategies:
            best = strategies[0]
            print(f"\n{method.upper()} Best:")
            print(f"  Sharpe: {best.sharpe:.3f}")
            print(f"  Return: {best.total_return:.2%}")
            print(f"  Max DD: {best.max_drawdown:.2%}")
            print(f"  Win Rate: {best.win_rate:.1%}")


def check_diversity(strategies, method_name):
    """Check if strategies are actually different."""
    print(f"\n{'='*60}")
    print(f"Diversity Check: {method_name.upper()}")
    print('='*60)
    
    if len(strategies) < 2:
        print("Not enough strategies to check diversity")
        return
    
    # Compare key params between strategies
    key_params = ['rsi_oversold', 'rsi_overbought', 'macd_epsilon', 'ma_crossover_margin']
    
    print("\nParameter Comparison Across Strategies:")
    for param in key_params:
        values = [s.params.get(param, 0) for s in strategies]
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val
        print(f"  {param:20s}: min={min_val:.4f}, max={max_val:.4f}, range={range_val:.4f}")
    
    # Check Sharpe diversity
    sharpes = [s.sharpe for s in strategies]
    print(f"\n  Sharpe ratios: {[f'{s:.3f}' for s in sharpes]}")
    print(f"  Sharpe range: {max(sharpes) - min(sharpes):.3f}")
    
    if max(sharpes) - min(sharpes) < 0.01:
        print("\n  ⚠️  WARNING: All strategies have nearly identical Sharpe ratios!")
        print("      This suggests the search converged to one local optimum.")


def main() -> None:
    print("="*60)
    print("Module 2: Strategy Parameter Search Demo")
    print("Comparing Beam Search vs A* Search")
    print("="*60)
    
    print("\nGenerating synthetic OHLCV (252 days)...")
    ohlcv = generate_synthetic_ohlcv(days=252, seed=42)
    print(f"Generated {len(ohlcv)} bars of market data")
    
    # Run both search methods
    top_k = 5
    results = run_search_comparison(ohlcv, top_k=top_k)
    
    # Compare best from each
    compare_best_strategies(results)
    
    # Check diversity within each method
    for method, strategies in results.items():
        check_diversity(strategies, method)
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)


if __name__ == "__main__":
    main()