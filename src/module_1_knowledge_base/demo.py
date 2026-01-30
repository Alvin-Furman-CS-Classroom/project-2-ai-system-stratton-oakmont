"""Demo: Module 1 Trading Rule Knowledge Base usage."""

from src.module_1_knowledge_base import evaluate_rules_on_indicators, indicators_to_facts
from src.shared import MarketIndicators


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
    print(f"Indicators: RSI={bullish.rsi}, MACD={bullish.macd}, MA20={bullish.ma20}, MA50={bullish.ma50}")
    
    facts = indicators_to_facts(bullish)
    print(f"\nFacts (True only): {[k for k, v in facts.items() if v]}")
    
    result = evaluate_rules_on_indicators(bullish)
    print(f"\nAction: {result.action.value}")
    print(f"Fired Rules: {result.fired_rules}")
    print(f"Conflict: {result.conflict}")

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
    print(f"Indicators: RSI={bearish.rsi}, MACD={bearish.macd}, MA20={bearish.ma20}, MA50={bearish.ma50}")
    
    result = evaluate_rules_on_indicators(bearish)
    print(f"\nAction: {result.action.value}")
    print(f"Fired Rules: {result.fired_rules}")

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


if __name__ == "__main__":
    main()
