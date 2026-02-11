"""Unit tests for Module 1: Trading Rule Knowledge Base."""

from src.module_1_knowledge_base import evaluate_rules_on_indicators
from src.shared import MarketIndicators, TradingAction


# =============================================================================
# BULLISH MARKET TESTS
# =============================================================================


def test_bullish_strong_oversold():
    """Bullish signal: RSI oversold with positive MACD and golden cross."""
    ind = MarketIndicators(
        rsi=25,  # Oversold
        macd=1.0,  # Strong positive
        ma20=105,
        ma50=100,
        volume=2_000_000,
        volatility=0.01,
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.BUY


def test_bullish_neutral_rsi_strong_momentum():
    """Bullish signal: Neutral RSI with strong uptrend and high volume."""
    ind = MarketIndicators(
        rsi=50,  # Neutral
        macd=1.2,  # Strong positive MACD
        ma20=110,
        ma50=100,
        volume=2_500_000,
        volatility=0.005,
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.BUY


def test_bullish_uptrend_with_momentum():
    """Bullish signal: Golden cross with positive MACD and moderate RSI."""
    ind = MarketIndicators(
        rsi=45,  # Slightly low but not oversold
        macd=0.8,  # Positive MACD
        ma20=103,  # Golden cross
        ma50=100,
        volume=1_800_000,
        volatility=0.012,
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.BUY


def test_bullish_strong_uptrend_low_volatility():
    """Bullish signal: Strong uptrend with low volatility and stable volume."""
    ind = MarketIndicators(
        rsi=35,  # Oversold
        macd=0.95,  # Strong positive
        ma20=115,  # 15% above MA50
        ma50=100,
        volume=2_100_000,
        volatility=0.008,
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.BUY


def test_bullish_recovery_pattern():
    """Bullish signal: RSI recovering from oversold with rising MACD."""
    ind = MarketIndicators(
        rsi=30,  # Just recovering from oversold
        macd=0.5,  # Positive and rising
        ma20=104,
        ma50=100,
        volume=1_600_000,
        volatility=0.015,
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.BUY


def test_bullish_sustained_momentum():
    """Bullish signal: Sustained bullish momentum with all indicators aligned."""
    ind = MarketIndicators(
        rsi=55,  # Neutral/slightly high
        macd=1.5,  # Very strong positive
        ma20=112,
        ma50=100,
        volume=3_000_000,
        volatility=0.007,
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.BUY


def test_bullish_moderate_uptrend():
    """Bullish signal: Moderate uptrend with good volume confirmation."""
    ind = MarketIndicators(
        rsi=48,  # Neutral
        macd=0.7,  # Moderately positive
        ma20=106,
        ma50=100,
        volume=2_200_000,
        volatility=0.011,
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.BUY


def test_bullish_gap_up_with_volume():
    """Bullish signal: Gap up with high volume and positive momentum."""
    ind = MarketIndicators(
        rsi=60,  # Elevated but not overbought
        macd=1.1,  # Strong positive
        ma20=120,
        ma50=100,
        volume=3_500_000,  # Very high volume
        volatility=0.010,
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.BUY


def test_bullish_breakout_scenario():
    """Bullish signal: Breakout above MA50 with accelerating momentum."""
    ind = MarketIndicators(
        rsi=52,
        macd=1.3,  # Accelerating MACD
        ma20=108,
        ma50=100,
        volume=2_800_000,
        volatility=0.009,
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.BUY


def test_bullish_consolidation_breakup():
    """Bullish signal: Consolidation period ending with upside breakout."""
    ind = MarketIndicators(
        rsi=40,  # Recovering
        macd=0.6,  # Positive and steady
        ma20=102,
        ma50=100,
        volume=1_900_000,
        volatility=0.006,  # Low volatility ending
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.BUY


# =============================================================================
# BEARISH MARKET TESTS
# =============================================================================


def test_bearish_strong_overbought():
    """Bearish signal: RSI overbought with negative MACD and death cross."""
    ind = MarketIndicators(
        rsi=75,  # Overbought
        macd=-1.0,  # Strong negative
        ma20=95,
        ma50=100,
        volume=2_000_000,
        volatility=0.01,
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.SELL


def test_bearish_neutral_rsi_strong_downtrend():
    """Bearish signal: Neutral RSI with strong downtrend and high volume."""
    ind = MarketIndicators(
        rsi=50,  # Neutral
        macd=-1.2,  # Strong negative MACD
        ma20=90,
        ma50=100,
        volume=2_500_000,
        volatility=0.005,
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.SELL


def test_bearish_downtrend_with_momentum():
    """Bearish signal: Death cross with negative MACD and moderate RSI."""
    ind = MarketIndicators(
        rsi=55,  # Slightly high but not overbought
        macd=-0.8,  # Negative MACD
        ma20=97,  # Death cross
        ma50=100,
        volume=1_800_000,
        volatility=0.012,
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.SELL


def test_bearish_strong_downtrend_low_volatility():
    """Bearish signal: Strong downtrend with low volatility and stable volume."""
    ind = MarketIndicators(
        rsi=65,  # Overbought
        macd=-0.95,  # Strong negative
        ma20=85,  # 15% below MA50
        ma50=100,
        volume=2_100_000,
        volatility=0.008,
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.SELL


def test_bearish_recovery_pattern():
    """Bearish signal: RSI falling from overbought with falling MACD."""
    ind = MarketIndicators(
        rsi=70,  # Just falling from overbought
        macd=-0.5,  # Negative and falling
        ma20=96,
        ma50=100,
        volume=1_600_000,
        volatility=0.015,
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.SELL


def test_bearish_sustained_downtrend():
    """Bearish signal: Sustained bearish momentum with all indicators aligned."""
    ind = MarketIndicators(
        rsi=45,  # Neutral/slightly low
        macd=-1.5,  # Very strong negative
        ma20=88,
        ma50=100,
        volume=3_000_000,
        volatility=0.007,
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.SELL


def test_bearish_moderate_downtrend():
    """Bearish signal: Moderate downtrend with good volume confirmation."""
    ind = MarketIndicators(
        rsi=52,  # Neutral
        macd=-0.7,  # Moderately negative
        ma20=94,
        ma50=100,
        volume=2_200_000,
        volatility=0.011,
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.SELL


def test_bearish_gap_down_with_volume():
    """Bearish signal: Gap down with high volume and negative momentum."""
    ind = MarketIndicators(
        rsi=40,  # Elevated but not oversold
        macd=-1.1,  # Strong negative
        ma20=80,
        ma50=100,
        volume=3_500_000,  # Very high volume
        volatility=0.010,
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.SELL


def test_bearish_breakdown_scenario():
    """Bearish signal: Breakdown below MA50 with accelerating negative momentum."""
    ind = MarketIndicators(
        rsi=48,
        macd=-1.3,  # Accelerating negative MACD
        ma20=92,
        ma50=100,
        volume=2_800_000,
        volatility=0.009,
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.SELL


def test_bearish_consolidation_breakdown():
    """Bearish signal: Consolidation period ending with downside breakdown."""
    ind = MarketIndicators(
        rsi=60,  # Falling
        macd=-0.6,  # Negative and steady
        ma20=98,
        ma50=100,
        volume=1_900_000,
        volatility=0.006,  # Low volatility ending
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.SELL


# =============================================================================
# NEUTRAL/HOLD MARKET TESTS
# =============================================================================


def test_hold_perfect_neutral():
    """Hold signal: All indicators perfectly neutral."""
    ind = MarketIndicators(
        rsi=50,  # Perfect neutral
        macd=0.0,  # No momentum
        ma20=100,  # No trend
        ma50=100,
        volume=500_000,  # Average volume
        volatility=0.02,  # Medium volatility
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.HOLD


def test_hold_slight_positive_signals():
    """Hold signal: Slight positive signals but not strong enough."""
    ind = MarketIndicators(
        rsi=48,  # Slightly low but not oversold
        macd=0.2,  # Weakly positive
        ma20=101,  # Barely above MA50
        ma50=100,
        volume=600_000,  # Slightly above average
        volatility=0.018,
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.HOLD


def test_hold_slight_negative_signals():
    """Hold signal: Slight negative signals but not strong enough."""
    ind = MarketIndicators(
        rsi=52,  # Slightly high but not overbought
        macd=-0.2,  # Weakly negative
        ma20=99,  # Barely below MA50
        ma50=100,
        volume=550_000,
        volatility=0.019,
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.HOLD


def test_hold_mixed_signals_bullish_bearish():
    """Hold signal: Mixed signals - some bullish, some bearish."""
    ind = MarketIndicators(
        rsi=35,  # Oversold (bullish)
        macd=-0.3,  # Negative (bearish)
        ma20=102,  # Golden cross (bullish)
        ma50=100,
        volume=700_000,
        volatility=0.016,
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.HOLD


def test_hold_consolidation_no_breakout():
    """Hold signal: Consolidation pattern with no clear breakout."""
    ind = MarketIndicators(
        rsi=50,
        macd=0.05,  # Very weak positive (just below threshold)
        ma20=100.3,  # Barely above (less than 1% difference)
        ma50=100,
        volume=450_000,  # Low volume
        volatility=0.02,  # Medium volatility (not low enough for BUY_LOW_VOL)
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.HOLD



def test_hold_moderate_rsi_neutral_macd():
    """Hold signal: Moderate RSI with neutral MACD."""
    ind = MarketIndicators(
        rsi=55,
        macd=0.0,  # Exactly neutral
        ma20=100,
        ma50=100,
        volume=500_000,
        volatility=0.02,
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.HOLD


def test_hold_weak_uptrend_weak_momentum():
    """Hold signal: Weak uptrend with very weak momentum."""
    ind = MarketIndicators(
        rsi=45,
        macd=0.15,  # Very weakly positive
        ma20=102,  # Slight uptrend
        ma50=100,
        volume=480_000,
        volatility=0.022,
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.HOLD


def test_hold_weak_downtrend_weak_momentum():
    """Hold signal: Weak downtrend with very weak momentum."""
    ind = MarketIndicators(
        rsi=55,
        macd=-0.15,  # Very weakly negative
        ma20=98,  # Slight downtrend
        ma50=100,
        volume=520_000,
        volatility=0.021,
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.HOLD


def test_hold_uncertainty_high_volatility():
    """Hold signal: Uncertain signals with high volatility."""
    ind = MarketIndicators(
        rsi=50,  # Neutral
        macd=0.05,  # Barely positive
        ma20=100.5,
        ma50=100,
        volume=600_000,
        volatility=0.045,  # High volatility creates uncertainty
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.HOLD


def test_hold_low_volume_no_confirmation():
    """Hold signal: Potential signals but no volume confirmation."""
    ind = MarketIndicators(
        rsi=30,  # Oversold signal
        macd=0.3,  # Positive signal
        ma20=103,  # Uptrend signal
        ma50=100,
        volume=100_000,  # Very low volume - no confirmation
        volatility=0.018,
    )
    result = evaluate_rules_on_indicators(ind)
    assert result.action == TradingAction.HOLD
