"""Unit tests for Module 1: Trading Rule Knowledge Base."""

from src.module_1_knowledge_base import (
    DEFAULT_PARAMS,
    HornRule,
    Literal,
    default_fact_definitions,
    evaluate_rules_on_indicators,
    horn_rule_from_cnf_clause,
    indicators_to_facts,
)
from src.module_1_knowledge_base.engine import default_trading_rules
from src.shared import MarketIndicators, TradingAction


# =============================================================================
# FACT GENERATION TESTS
# =============================================================================


def test_indicators_to_facts_basic_flags():
    """Test that basic indicator-to-fact conversion works correctly."""
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


def test_indicators_to_facts_new_facts():
    """Test the new expanded facts: RSI_NEUTRAL, STRONG_UPTREND, MACD_STRONG, VOLATILITY_LOW."""
    # Neutral RSI, strong uptrend, strong MACD, low volatility
    ind = MarketIndicators(
        rsi=50,  # Neutral (40-60)
        macd=1.0,  # Strong positive (> 0.5)
        ma20=110,  # 10% above ma50 -> strong uptrend (> 2% margin)
        ma50=100,
        volume=1_500_000,  # High volume and surge (> 2x average of 500k)
        volatility=0.005,  # Low volatility (< 0.01)
    )
    facts = indicators_to_facts(ind)
    
    # RSI facts
    assert facts["RSI_NEUTRAL"] is True
    assert facts["RSI_OVERSOLD"] is False
    assert facts["RSI_OVERBOUGHT"] is False
    
    # MACD facts
    assert facts["MACD_POSITIVE"] is True
    assert facts["MACD_STRONG_POSITIVE"] is True
    assert facts["MACD_NEGATIVE"] is False
    assert facts["MACD_STRONG_NEGATIVE"] is False
    
    # Trend facts
    assert facts["GOLDEN_CROSS"] is True
    assert facts["STRONG_UPTREND"] is True
    assert facts["DEATH_CROSS"] is False
    assert facts["STRONG_DOWNTREND"] is False
    
    # Volume facts
    assert facts["VOLUME_HIGH"] is True
    assert facts["VOLUME_SURGE"] is True
    
    # Volatility facts
    assert facts["VOLATILITY_LOW"] is True
    assert facts["VOLATILITY_HIGH"] is False


def test_indicators_to_facts_downtrend():
    """Test facts for a strong downtrend scenario."""
    ind = MarketIndicators(
        rsi=75,  # Overbought
        macd=-1.0,  # Strong negative
        ma20=90,  # Below ma50 by more than 2%
        ma50=100,
        volume=100_000,  # Low volume
        volatility=0.05,  # High volatility
    )
    facts = indicators_to_facts(ind)
    
    assert facts["RSI_OVERBOUGHT"] is True
    assert facts["MACD_NEGATIVE"] is True
    assert facts["MACD_STRONG_NEGATIVE"] is True
    assert facts["DEATH_CROSS"] is True
    assert facts["STRONG_DOWNTREND"] is True
    assert facts["VOLATILITY_HIGH"] is True
    assert facts["VOLUME_HIGH"] is False
    assert facts["VOLUME_SURGE"] is False


def test_all_facts_are_defined():
    """Ensure we have all 16 expected facts defined."""
    facts_defs = default_fact_definitions()
    fact_names = {f.name for f in facts_defs}
    
    expected_facts = {
        # RSI
        "RSI_OVERSOLD", "RSI_OVERBOUGHT", "RSI_NEUTRAL",
        # MACD
        "MACD_POSITIVE", "MACD_NEGATIVE", "MACD_STRONG_POSITIVE", "MACD_STRONG_NEGATIVE",
        # Trend
        "GOLDEN_CROSS", "DEATH_CROSS", "STRONG_UPTREND", "STRONG_DOWNTREND",
        # Volume
        "VOLUME_HIGH", "VOLUME_SURGE",
        # Volatility
        "VOLATILITY_HIGH", "VOLATILITY_LOW", "VOLATILITY_UNKNOWN",
    }
    
    assert fact_names == expected_facts, f"Missing facts: {expected_facts - fact_names}"


# =============================================================================
# FORWARD CHAINING / RULE FIRING TESTS
# =============================================================================


def test_forward_chaining_default_rules_buy():
    """Test that the momentum buy rule fires correctly."""
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
    # The new rule ID is BUY_MOMENTUM_1 (renamed from BUY_1)
    assert "BUY_MOMENTUM_1" in result.fired_rules
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
    """Test CNF clause parsing into Horn rules."""
    r = horn_rule_from_cnf_clause(clause="(~A OR ~B OR C)", rule_id="R1")
    assert r.conclusion == "C"
    assert {lit.symbol for lit in r.premises} == {"A", "B"}


# =============================================================================
# EXPANDED RULE TESTS
# =============================================================================


def test_all_rules_are_defined():
    """Ensure we have all 14 expected rules defined."""
    rules = default_trading_rules()
    rule_ids = {r.rule_id for r in rules}
    
    expected_rules = {
        # Momentum continuation
        "BUY_MOMENTUM_1", "BUY_MOMENTUM_STRONG", "SELL_MOMENTUM_1", "SELL_MOMENTUM_STRONG",
        # Mean reversion
        "BUY_PULLBACK", "SELL_RALLY",
        # Volume confirmed
        "BUY_VOLUME_BREAKOUT", "SELL_VOLUME_BREAKDOWN",
        # Conservative
        "BUY_CONSERVATIVE", "SELL_CONSERVATIVE",
        # Aggressive
        "BUY_AGGRESSIVE", "SELL_AGGRESSIVE",
        # Low volatility
        "BUY_LOW_VOL", "SELL_LOW_VOL",
    }
    
    assert rule_ids == expected_rules, f"Missing rules: {expected_rules - rule_ids}"


def test_strong_momentum_buy():
    """Test strong momentum buy: strong uptrend + strong MACD + volume."""
    ind = MarketIndicators(
        rsi=50,  # Doesn't matter for this rule
        macd=1.0,  # Strong positive (> 0.5)
        ma20=110,  # 10% above ma50 -> strong uptrend
        ma50=100,
        volume=2_000_000,  # High volume
        volatility=0.02,
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.BUY
    assert "BUY_MOMENTUM_STRONG" in result.fired_rules


def test_strong_momentum_sell():
    """Test strong momentum sell: strong downtrend + strong negative MACD + volume."""
    ind = MarketIndicators(
        rsi=50,
        macd=-1.0,  # Strong negative
        ma20=90,  # Strong downtrend (< 98% of ma50)
        ma50=100,
        volume=2_000_000,  # High volume
        volatility=0.02,
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.SELL
    assert "SELL_MOMENTUM_STRONG" in result.fired_rules


def test_pullback_buy():
    """Test pullback buy: oversold in an uptrend."""
    ind = MarketIndicators(
        rsi=25,  # Oversold
        macd=0.1,  # Slightly positive
        ma20=102,  # In uptrend (golden cross) but not strong
        ma50=100,
        volume=500_000,
        volatility=0.02,  # Not high, not low
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.BUY
    # Both BUY_PULLBACK and BUY_MOMENTUM_1 should fire (same premises)
    assert "BUY_MOMENTUM_1" in result.fired_rules


def test_volume_breakout_buy():
    """Test volume breakout: uptrend + positive MACD + volume surge."""
    ind = MarketIndicators(
        rsi=55,  # Neutral
        macd=0.3,  # Positive but not strong
        ma20=105,
        ma50=100,
        volume=1_500_000,  # > 2x average (500k) = surge
        volatility=0.02,
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.BUY
    assert "BUY_VOLUME_BREAKOUT" in result.fired_rules


def test_aggressive_buy():
    """Test aggressive buy: just oversold + strong uptrend."""
    ind = MarketIndicators(
        rsi=25,  # Oversold
        macd=-0.1,  # Even slightly negative MACD
        ma20=110,  # Strong uptrend (10% above)
        ma50=100,
        volume=100_000,  # Low volume
        volatility=0.05,  # High volatility - doesn't matter for aggressive
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.BUY
    assert "BUY_AGGRESSIVE" in result.fired_rules


def test_aggressive_sell():
    """Test aggressive sell: just overbought + strong downtrend."""
    ind = MarketIndicators(
        rsi=75,  # Overbought
        macd=0.1,  # Even slightly positive MACD
        ma20=90,  # Strong downtrend
        ma50=100,
        volume=100_000,
        volatility=0.05,
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.SELL
    assert "SELL_AGGRESSIVE" in result.fired_rules


def test_low_volatility_buy():
    """Test low volatility buy: uptrend + positive MACD + low volatility."""
    ind = MarketIndicators(
        rsi=55,
        macd=0.2,  # Positive
        ma20=103,  # Uptrend
        ma50=100,
        volume=500_000,
        volatility=0.005,  # Low volatility
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.BUY
    assert "BUY_LOW_VOL" in result.fired_rules


def test_no_signal_neutral_market():
    """Test that neutral conditions produce HOLD."""
    ind = MarketIndicators(
        rsi=50,  # Neutral
        macd=0.0,  # Neutral
        ma20=100,  # No trend
        ma50=100,
        volume=500_000,  # Average
        volatility=0.02,  # Medium
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.HOLD
    # No buy or sell rules should fire
    assert len(result.fired_rules) == 0


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


def test_volatility_none_sets_unknown_fact():
    """Test that volatility=None sets VOLATILITY_UNKNOWN=True."""
    ind = MarketIndicators(
        rsi=50,
        macd=0.0,
        ma20=100,
        ma50=100,
        volume=500_000,
        volatility=None,
    )
    facts = indicators_to_facts(ind)
    assert facts["VOLATILITY_UNKNOWN"] is True
    assert facts["VOLATILITY_HIGH"] is False
    assert facts["VOLATILITY_LOW"] is False


def test_extreme_rsi_values():
    """Test RSI at boundaries (0 and 100)."""
    # RSI = 0 (extreme oversold)
    ind_low = MarketIndicators(rsi=0, macd=0, ma20=100, ma50=100, volume=100, volatility=0.02)
    facts_low = indicators_to_facts(ind_low)
    assert facts_low["RSI_OVERSOLD"] is True
    assert facts_low["RSI_OVERBOUGHT"] is False
    
    # RSI = 100 (extreme overbought)
    ind_high = MarketIndicators(rsi=100, macd=0, ma20=100, ma50=100, volume=100, volatility=0.02)
    facts_high = indicators_to_facts(ind_high)
    assert facts_high["RSI_OVERBOUGHT"] is True
    assert facts_high["RSI_OVERSOLD"] is False


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


def test_cnf_invalid_clause_raises():
    """Test that invalid CNF clauses raise ValueError."""
    import pytest
    
    # No positive literal (not Horn-convertible)
    with pytest.raises(ValueError):
        horn_rule_from_cnf_clause(clause="(~A OR ~B)", rule_id="R1")
    
    # Multiple positive literals (not Horn-convertible)
    with pytest.raises(ValueError):
        horn_rule_from_cnf_clause(clause="(A OR B OR C)", rule_id="R2")
    
    # Empty negated literal
    with pytest.raises(ValueError):
        horn_rule_from_cnf_clause(clause="(~ OR B)", rule_id="R3")
