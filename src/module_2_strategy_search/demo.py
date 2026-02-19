"""Demo: Compare Beam Search vs A* on real/synthetic OHLCV → top strategies by Sharpe.

Splits data into train/test to validate generalization.
Shows key parameters and a buy-and-hold benchmark.
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
from src.shared.market_data import generate_synthetic_ohlcv, load_ohlcv_yahoo


def _fmt_strategy(strategy, idx):
    """Format a single strategy as a compact summary line."""
    p = strategy.params
    params_str = (
        f"RSI({p.get('rsi_oversold', 0):.0f}/{p.get('rsi_overbought', 0):.0f}), "
        f"MACD_eps={p.get('macd_epsilon', 0):.4f}, "
        f"MA_margin={p.get('ma_crossover_margin', 0):.4f}"
    )
    return (
        f"  {idx}. Sharpe={strategy.sharpe:+.3f}  "
        f"Return={strategy.total_return:+.2%}  "
        f"MaxDD={strategy.max_drawdown:+.2%}  "
        f"WinRate={strategy.win_rate:.0%}  "
        f"Trades={strategy.num_trades}  "
        f"[{params_str}]"
    )


def _buy_and_hold(ohlcv):
    """Compute buy-and-hold Sharpe and return on the tradable period."""
    close = ohlcv["Close"].values[WARMUP_BARS:]
    rets = np.diff(close) / close[:-1]
    total = float(close[-1] / close[0] - 1)
    sharpe = float(np.sqrt(252) * rets.mean() / rets.std()) if rets.std() > 0 else 0.0
    return sharpe, total


def main() -> None:
    print("=" * 60)
    print("Module 2 Demo: Beam Search vs A* Strategy Search")
    print("=" * 60)

    # Data configuration
    use_real_data = True
    symbol = "SPY"
    period = "2y"
    
    # Load data
    if use_real_data:
        try:
            ohlcv = load_ohlcv_yahoo(symbol=symbol, period=period)
            data_desc = f"{symbol} ({period})"
        except Exception as e:
            print(f"Failed to load {symbol}: {e}. Using synthetic data.")
            ohlcv = generate_synthetic_ohlcv(days=504, seed=42)
            data_desc = "synthetic (seed=42)"
    else:
        ohlcv = generate_synthetic_ohlcv(days=504, seed=42)
        data_desc = "synthetic (seed=42)"

    # Split train/test
    train_bars = len(ohlcv) // 2
    ohlcv_train = ohlcv.iloc[:train_bars].copy()
    ohlcv_test = ohlcv.iloc[train_bars:].copy()

    bh_sharpe, bh_ret = _buy_and_hold(ohlcv_train)
    print(f"\nData: {train_bars} train / {len(ohlcv) - train_bars} test bars ({data_desc})")
    print(f"Buy-and-Hold baseline (train): Sharpe={bh_sharpe:+.3f}, Return={bh_ret:+.2%}")

    top_k = 5
    for method in ("beam", "astar"):
        print(f"\n{'—'*60}")
        print(f"{method.upper()} SEARCH — top {top_k} strategies")
        print("—" * 60)
        strategies = search_top_strategies(ohlcv_train, top_k=top_k, method=method)
        for i, s in enumerate(strategies, 1):
            print(_fmt_strategy(s, i))

        # Validate on test set
        best = strategies[0]
        rets, _ = backtest(ohlcv_test, best.params)
        oos_sharpe = sharpe_ratio(rets)
        oos_ret = float(np.prod(1 + rets) - 1) if len(rets) > 0 else 0.0
        print(f"\n  Test set: Sharpe={oos_sharpe:+.3f}, Return={oos_ret:+.2%}")

    print(f"\n{'=' * 60}")
    print("Done.")


if __name__ == "__main__":
    main()