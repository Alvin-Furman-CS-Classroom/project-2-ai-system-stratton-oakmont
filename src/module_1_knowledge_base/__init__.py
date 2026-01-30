"""
Module 1: Trading Rule Knowledge Base (Propositional Logic, CNF/Horn, inference)

Public API:
- `evaluate_rules_on_indicators(...)` -> BUY/SELL/HOLD + fired rules + inference chain
"""

# Re-export things so other code can import from `src.module_1_knowledge_base`.
from .engine import (
    HornRule,
    InferenceResult,
    InferenceStep,
    Literal,
    evaluate_rules_on_indicators,
    forward_chain,
    horn_rule_from_cnf_clause,
)
from .facts import FactDefinition, DEFAULT_PARAMS, default_fact_definitions, indicators_to_facts

__all__ = [
    "HornRule",
    "InferenceResult",
    "InferenceStep",
    "Literal",
    "FactDefinition",
    "DEFAULT_PARAMS",
    "default_fact_definitions",
    "evaluate_rules_on_indicators",
    "forward_chain",
    "horn_rule_from_cnf_clause",
    "indicators_to_facts",
]
