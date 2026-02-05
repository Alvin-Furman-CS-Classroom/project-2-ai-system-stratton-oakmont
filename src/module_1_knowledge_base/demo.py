"""Demo: Module 1 Trading Rule Knowledge Base usage.

Run from project root (where src/ lives):
  python -m src.module_1_knowledge_base.demo

If you run this file directly (e.g. from an IDE), the block below adds the
project root to sys.path so "from src...." imports work.
"""

import sys
from pathlib import Path

# Ensure project root is on path when run as __main__ (e.g. python demo.py or IDE Run).
if __name__ == "__main__":
    _project_root = Path(__file__).resolve().parent.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

from src.module_1_knowledge_base import evaluate_rules_on_indicators, indicators_to_facts
from src.shared import MarketIndicators


def print_inference_chain(result):
    """Print the inference chain showing how conclusions were derived."""
    if not result.inference_chain:
        print("  (no rules fired)")
        return
    for step in result.inference_chain:
        premises_str = ", ".join(
            f"{'¬' if lit.negated else ''}{lit.symbol}"
            for lit in step.supporting_literals
        )
        print(f"  {step.rule_id}: {premises_str} → {step.added_fact}")


def main():
    # Example: Bullish market conditions
    bullish = MarketIndicators(
        rsi=25,       # Oversold
        macd=1.0,     # Strong positive momentum
        ma20=105,     # Above MA50 (uptrend)
        ma50=100,
        volume=2_000_000,  # High volume
        volatility=0.01,   # Low volatility
    )

    print("=== Bullish Market Example ===")
    print(
        f"Indicators: RSI={bullish.rsi}, MACD={bullish.macd}, "
        f"MA20={bullish.ma20}, MA50={bullish.ma50}"
    )
    
    facts = indicators_to_facts(bullish)
    print(f"\nFacts (True only): {[k for k, v in facts.items() if v]}")
    
    result = evaluate_rules_on_indicators(bullish)
    print(f"\nAction: {result.action.value}")
    print(f"Fired Rules: {result.fired_rules}")
    print(f"Conflict: {result.conflict}")
    print("\nInference Chain:")
    print_inference_chain(result)

    # Example: Bearish market conditions
    bearish = MarketIndicators(
        rsi=75,       # Overbought
        macd=-1.0,    # Strong negative momentum
        ma20=95,      # Below MA50 (downtrend)
        ma50=100,
        volume=2_000_000,
        volatility=0.02,
    )

    print("\n=== Bearish Market Example ===")
    print(
        f"Indicators: RSI={bearish.rsi}, MACD={bearish.macd}, "
        f"MA20={bearish.ma20}, MA50={bearish.ma50}"
    )
    
    result = evaluate_rules_on_indicators(bearish)
    print(f"\nAction: {result.action.value}")
    print(f"Fired Rules: {result.fired_rules}")
    print("\nInference Chain:")
    print_inference_chain(result)

    # Example: Neutral market (no signal)
    neutral = MarketIndicators(
        rsi=50,
        macd=0.0,
        ma20=100,
        ma50=100,
        volume=500_000,
        volatility=0.02,
    )

    print("\n=== Neutral Market Example ===")
    result = evaluate_rules_on_indicators(neutral)
    print(f"Action: {result.action.value}")
    print(f"Fired Rules: {result.fired_rules}")
    print("\nInference Chain:")
    print_inference_chain(result)


if __name__ == "__main__":
    main()
