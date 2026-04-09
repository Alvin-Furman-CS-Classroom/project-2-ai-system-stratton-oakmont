"""Visualization demo for Module 5 position sizing strategy behavior.

Plots market data with technical indicators and highlights one concrete
entry/exit pair produced by the strategy's rule signals.

Usage:
    python -m src.module_5_position_sizing.demo_visualization
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Repo root must be on sys.path before any `from src...` imports.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.module_1_knowledge_base import evaluate_rules_on_indicators
from src.module_2_strategy_search import search_top_strategies
from src.module_2_strategy_search.backtest import WARMUP_BARS, indicators_from_ohlcv
from src.shared.market_data import generate_synthetic_ohlcv, load_ohlcv_yahoo
from src.shared.types import CandidateStrategy, TradingAction


@dataclass(frozen=True)
class TradeExample:
    """One concrete trade example extracted from strategy signals."""

    entry_idx: int
    exit_idx: int
    entry_action: TradingAction
    exit_action: TradingAction


def _compute_plot_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute indicator series used in the visualization panels."""
    close = ohlcv["Close"]

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()

    return pd.DataFrame(
        {
            "Close": close,
            "RSI": rsi,
            "MACD": macd,
            "MA20": ma20,
            "MA50": ma50,
        },
        index=ohlcv.index,
    )


def find_example_trade(ohlcv: pd.DataFrame, strategy: CandidateStrategy) -> TradeExample | None:
    """Find first entry/exit signal pair from rule-engine actions.

    Entry is any non-HOLD signal; exit is the first opposite non-HOLD signal.
    """
    indicators = indicators_from_ohlcv(ohlcv)

    entry_idx: int | None = None
    entry_action: TradingAction | None = None

    for idx in range(WARMUP_BARS, len(ohlcv)):
        ind = indicators[idx]
        if ind is None:
            continue

        action = evaluate_rules_on_indicators(ind, params=strategy.params).action
        if action == TradingAction.HOLD:
            continue

        if entry_idx is None:
            entry_idx = idx
            entry_action = action
            continue

        if action != entry_action:
            return TradeExample(
                entry_idx=entry_idx,
                exit_idx=idx,
                entry_action=entry_action,
                exit_action=action,
            )

    return None


def build_module5_strategy_figure(
    ohlcv: pd.DataFrame,
    strategy: CandidateStrategy,
    *,
    symbol: str,
) -> tuple[plt.Figure, TradeExample | None]:
    """Build a multi-panel figure with price, indicators, and trade markers."""
    data = _compute_plot_indicators(ohlcv)
    trade = find_example_trade(ohlcv, strategy)

    fig, (ax_price, ax_rsi, ax_macd) = plt.subplots(
        3,
        1,
        figsize=(13, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.2, 1.2]},
    )

    fig.suptitle("Module 5 Strategy Visualization", fontsize=14, fontweight="bold")

    # Price panel
    ax_price.plot(data.index, data["Close"], label="Close", color="#1f2937", linewidth=1.6)
    ax_price.plot(data.index, data["MA20"], label="MA20", color="#0ea5e9", linewidth=1.2)
    ax_price.plot(data.index, data["MA50"], label="MA50", color="#f97316", linewidth=1.2)
    ax_price.set_ylabel("Price")
    ax_price.grid(alpha=0.25)

    if trade is not None:
        entry_t = data.index[trade.entry_idx]
        exit_t = data.index[trade.exit_idx]
        entry_p = float(data["Close"].iloc[trade.entry_idx])
        exit_p = float(data["Close"].iloc[trade.exit_idx])

        ax_price.scatter([entry_t], [entry_p], marker="^", s=120, color="#16a34a", label="Entry")
        ax_price.scatter([exit_t], [exit_p], marker="v", s=120, color="#dc2626", label="Exit")
        ax_price.axvspan(entry_t, exit_t, color="#16a34a", alpha=0.08)
        ax_price.annotate(
            f"Entry ({trade.entry_action.value})",
            xy=(entry_t, entry_p),
            xytext=(12, 14),
            textcoords="offset points",
            color="#166534",
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": "#166534", "lw": 0.8},
        )
        ax_price.annotate(
            f"Exit ({trade.exit_action.value})",
            xy=(exit_t, exit_p),
            xytext=(12, -18),
            textcoords="offset points",
            color="#991b1b",
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": "#991b1b", "lw": 0.8},
        )

    subtitle = (
        f"{symbol} | Sharpe={strategy.sharpe:+.3f}, Return={strategy.total_return:+.2%}, "
        f"MaxDD={strategy.max_drawdown:+.2%}"
    )
    ax_price.set_title(subtitle, fontsize=10)
    ax_price.legend(loc="upper left", ncol=4)

    # RSI panel
    ax_rsi.plot(data.index, data["RSI"], color="#7c3aed", linewidth=1.2, label="RSI(14)")
    overbought = strategy.params.get("rsi_overbought", 70.0)
    oversold = strategy.params.get("rsi_oversold", 30.0)
    ax_rsi.axhline(overbought, color="#dc2626", linestyle="--", linewidth=1.0, label=f"Overbought ({overbought:.0f})")
    ax_rsi.axhline(oversold, color="#16a34a", linestyle="--", linewidth=1.0, label=f"Oversold ({oversold:.0f})")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel("RSI")
    ax_rsi.grid(alpha=0.25)
    ax_rsi.legend(loc="upper left", ncol=3, fontsize=8)

    # MACD panel
    ax_macd.plot(data.index, data["MACD"], color="#0284c7", linewidth=1.2, label="MACD")
    ax_macd.axhline(0.0, color="#111827", linestyle="--", linewidth=1.0)
    ax_macd.set_ylabel("MACD")
    ax_macd.set_xlabel("Date")
    ax_macd.grid(alpha=0.25)
    ax_macd.legend(loc="upper left")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, trade


def save_module5_figure(fig: plt.Figure, output_path: str | Path) -> Path:
    """Persist the generated visualization to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    return path


def _load_data_with_fallback(symbol: str, period: str) -> pd.DataFrame:
    try:
        return load_ohlcv_yahoo(symbol=symbol, period=period)
    except Exception:
        return generate_synthetic_ohlcv(days=252, seed=42)


def run_demo(
    *,
    symbol: str = "SPY",
    period: str = "1y",
    output_path: str | Path = "data/module_5_strategy_visualization.png",
) -> Path:
    """Run an end-to-end Module 5 visualization demo and save PNG output."""
    ohlcv = _load_data_with_fallback(symbol, period)
    top = search_top_strategies(ohlcv, top_k=1, method="beam")
    strategy = top[0]

    fig, trade = build_module5_strategy_figure(ohlcv, strategy, symbol=symbol)
    try:
        saved = save_module5_figure(fig, output_path)
    finally:
        plt.close(fig)

    if trade is not None:
        print(
            "Saved Module 5 visualization with example trade: "
            f"entry={trade.entry_action.value} at index {trade.entry_idx}, "
            f"exit={trade.exit_action.value} at index {trade.exit_idx}."
        )
    else:
        print("Saved Module 5 visualization (no opposite entry/exit signal pair found).")
    print(f"Output: {saved.resolve()}")
    return saved


if __name__ == "__main__":
    run_demo()