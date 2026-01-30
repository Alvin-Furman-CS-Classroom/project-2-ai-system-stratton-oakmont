"""Unit tests for Module 1: Trading Rule Knowledge Base."""

from src.module_1_knowledge_base import (
    HornRule,
    Literal,
    evaluate_rules_on_indicators,
    horn_rule_from_cnf_clause,
    indicators_to_facts,
)
from src.shared import MarketIndicators, TradingAction


def test_indicators_to_facts_basic_flags():
    ind = MarketIndicators(
        rsi=25,
        macd=1.0,
        ma20=105,
        ma50=100,
        volume=2_000_000,
        volatility=0.01,
    )
    facts = indicators_to_facts(ind)
    assert facts["RSI_OVERSOLD"] is True
    assert facts["RSI_OVERBOUGHT"] is False
    assert facts["MACD_POSITIVE"] is True
    assert facts["GOLDEN_CROSS"] is True
    assert facts["VOLATILITY_HIGH"] is False
    assert facts["VOLUME_HIGH"] is True


def test_forward_chaining_default_rules_buy():
    ind = MarketIndicators(
        rsi=25,
        macd=1.0,
        ma20=105,
        ma50=100,
        volume=2_000_000,
        volatility=0.01,
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.BUY
    assert "BUY_1" in result.fired_rules
    assert any(step.added_fact == TradingAction.BUY.value for step in result.inference_chain)


def test_negated_premise_can_trigger():
    ind = MarketIndicators(
        rsi=25,
        macd=-0.5,
        ma20=100,
        ma50=100,
        volume=10,
        volatility=None,
    )
    # Fires if RSI_OVERBOUGHT is false.
    rules = [
        HornRule(
            rule_id="BUY_NOT_OVERBOUGHT",
            premises=[Literal("RSI_OVERBOUGHT", negated=True)],
            conclusion=TradingAction.BUY.value,
        )
    ]
    result = evaluate_rules_on_indicators(ind, rules=rules)
    assert result.action == TradingAction.BUY
    assert "BUY_NOT_OVERBOUGHT" in result.fired_rules


def test_conflict_buy_and_sell_returns_hold():
    ind = MarketIndicators(
        rsi=50,
        macd=0.0,
        ma20=100,
        ma50=100,
        volume=10,
        volatility=0.02,
    )
    rules = [
        HornRule(rule_id="BUY_ALWAYS", premises=[], conclusion=TradingAction.BUY.value),
        HornRule(rule_id="SELL_ALWAYS", premises=[], conclusion=TradingAction.SELL.value),
    ]
    result = evaluate_rules_on_indicators(ind, rules=rules)
    assert result.action == TradingAction.HOLD
    assert result.conflict is True


def test_horn_rule_from_cnf_clause_parses():
    r = horn_rule_from_cnf_clause(clause="(~A OR ~B OR C)", rule_id="R1")
    assert r.conclusion == "C"
    assert {lit.symbol for lit in r.premises} == {"A", "B"}
