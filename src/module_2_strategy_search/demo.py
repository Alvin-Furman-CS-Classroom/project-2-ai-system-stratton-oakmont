"""Demo: Compare Beam Search vs A* on synthetic OHLCV → top strategies by Sharpe.

Splits data into train (67%) and test (33%) to detect overfitting.
Shows all tuned parameters and a buy-and-hold benchmark.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from src.module_2_strategy_search import search_top_strategies
from src.module_2_strategy_search.backtest import backtest, sharpe_ratio, WARMUP_BARS
from src.shared.market_data import generate_synthetic_ohlcv


def print_strategy_details(strategy, idx):
    """Print detailed info about a strategy, showing all tuned parameters."""
    print(f"\n  Strategy {idx}:")
    print(f"    Sharpe: {strategy.sharpe:.3f}")
    print(f"    Return: {strategy.total_return:.2%}")
    print(f"    Max Drawdown: {strategy.max_drawdown:.2%}")
    print(f"    Win Rate: {strategy.win_rate:.1%}")
    print(f"    Trades: {strategy.num_trades}")
    print(f"    Parameters ({len(strategy.params)} total):")
    # Group params by category for readability
    groups = {
        "RSI": ["rsi_oversold", "rsi_overbought", "rsi_neutral_low", "rsi_neutral_high"],
        "MACD": ["macd_epsilon", "macd_strong_threshold"],
        "Trend": ["ma_crossover_margin"],
        "Volume": ["volume_high", "volume_surge_multiplier", "volume_average"],
        "Volatility": ["volatility_high", "volatility_low"],
    }
    for group_name, keys in groups.items():
        present = {k: strategy.params[k] for k in keys if k in strategy.params}
        if present:
            vals = ", ".join(f"{k}={v:.4f}" for k, v in present.items())
            print(f"      {group_name}: {vals}")


def compute_buy_and_hold(ohlcv):
    """Compute buy-and-hold benchmark on the tradable period (after warmup)."""
    close = ohlcv["Close"].values
    tradable = close[WARMUP_BARS:]
    rets = np.diff(tradable) / tradable[:-1]
    total_return = float(tradable[-1] / tradable[0] - 1)
    sharpe = float(np.sqrt(252) * rets.mean() / rets.std()) if rets.std() > 0 else 0.0
    return sharpe, total_return


def print_benchmark(ohlcv, label="Buy-and-Hold Benchmark"):
    """Print buy-and-hold performance as a baseline."""
    bh_sharpe, bh_return = compute_buy_and_hold(ohlcv)
    print(f"\n  {label}:")
    print(f"    Sharpe: {bh_sharpe:.3f}")
    print(f"    Return: {bh_return:.2%}")


def run_search_comparison(ohlcv_train, top_k=5):
    """Run both search methods on training data."""
    methods = ["beam", "astar"]
    results = {}

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Running {method.upper()} Search (on training data)...")
        print('='*60)

        strategies = search_top_strategies(ohlcv_train, top_k=top_k, method=method)
        results[method] = strategies

        print(f"\nTop {len(strategies)} strategies from {method.upper()}:")
        for idx, strategy in enumerate(strategies, 1):
            print_strategy_details(strategy, idx)

    return results


def validate_out_of_sample(results, ohlcv_test):
    """Backtest best strategies from each method on unseen test data."""
    print(f"\n{'='*60}")
    print("OUT-OF-SAMPLE VALIDATION (Test Set)")
    print('='*60)

    print_benchmark(ohlcv_test, label="Test-Set Buy-and-Hold")

    for method, strategies in results.items():
        if not strategies:
            continue
        best = strategies[0]
        rets, _ = backtest(ohlcv_test, best.params)
        oos_sharpe = sharpe_ratio(rets)
        oos_return = float(np.prod(1 + rets) - 1) if len(rets) > 0 else 0.0
        print(f"\n  {method.upper()} Best Strategy:")
        print(f"    In-sample  Sharpe: {best.sharpe:.3f}, Return: {best.total_return:.2%}")
        print(f"    Out-of-sample Sharpe: {oos_sharpe:.3f}, Return: {oos_return:.2%}")
        # Overfitting: in-sample much better than out-of-sample
        drop = best.sharpe - oos_sharpe
        if drop > 0.5:
            print(f"    ⚠️  In-sample >> out-of-sample (drop={drop:+.3f}) — likely overfitting")
        elif drop < -0.5:
            print(f"    📈  Out-of-sample >> in-sample (gap={drop:+.3f}) — favorable test conditions")
        else:
            print(f"    ✅  Reasonable gap ({drop:+.3f}) — strategy generalizes adequately")


def compare_best_strategies(results):
    """Compare the best strategy from each method (in-sample)."""
    print(f"\n{'='*60}")
    print("COMPARISON: Best Strategy from Each Method (Training Set)")
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

    # Compare all params between strategies
    all_params = sorted(strategies[0].params.keys())

    print("\nParameter Comparison Across Strategies:")
    for param in all_params:
        values = [s.params.get(param, 0) for s in strategies]
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val
        if range_val > 0:
            print(f"  {param:25s}: min={min_val:.4f}, max={max_val:.4f}, range={range_val:.4f}")

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

    # Generate data and split into train/test
    total_days = 378  # ~1.5 years: 252 train + 126 test
    train_days = 252
    print(f"\nGenerating synthetic OHLCV ({total_days} days, seed=42)...")
    ohlcv = generate_synthetic_ohlcv(days=total_days, seed=42)

    ohlcv_train = ohlcv.iloc[:train_days].copy()
    ohlcv_test = ohlcv.iloc[train_days:].copy()
    print(f"  Training set: {len(ohlcv_train)} bars")
    print(f"  Test set:     {len(ohlcv_test)} bars")

    # Buy-and-hold benchmark on training data
    print_benchmark(ohlcv_train, label="Training-Set Buy-and-Hold")

    # Run both search methods on training data
    top_k = 5
    results = run_search_comparison(ohlcv_train, top_k=top_k)

    # Compare best from each (in-sample)
    compare_best_strategies(results)

    # Out-of-sample validation
    validate_out_of_sample(results, ohlcv_test)

    # Check diversity within each method
    for method, strategies in results.items():
        check_diversity(strategies, method)

    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)


if __name__ == "__main__":
    main()