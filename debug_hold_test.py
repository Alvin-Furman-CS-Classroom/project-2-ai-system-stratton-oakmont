"""Debug script to show inference chain for test_hold_consolidation_no_breakout."""

from src.module_1_knowledge_base import evaluate_rules_on_indicators
from src.shared import MarketIndicators, TradingAction

ind = MarketIndicators(
    rsi=50,
    macd=0.1,
    ma20=100.5,
    ma50=100,
    volume=450_000,
    volatility=0.004,
)
result = evaluate_rules_on_indicators(ind)
print(f'Action: {result.action}')
print(f'Fired Rules: {result.fired_rules}')
print(f'Conflict: {result.conflict}')
print(f'\nInference Chain:')
if not result.inference_chain:
    print('  (no rules fired)')
else:
    for step in result.inference_chain:
        premises_str = ', '.join(
            f'{"¬" if lit.negated else ""}{lit.symbol}'
            for lit in step.supporting_literals
        )
        print(f'  {step.rule_id}: {premises_str} → {step.added_fact}')
