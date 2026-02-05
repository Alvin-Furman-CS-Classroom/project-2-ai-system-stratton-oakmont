from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

from src.shared import MarketIndicators


Params = Dict[str, float]


@dataclass(frozen=True)
class FactDefinition:
    """
    A named boolean proposition (fact) derived from numeric indicators.

    Module 1 uses propositional logic; this adapter layer turns numeric market
    indicators into boolean symbols that the Knowledge Base can reason over.
    """

    name: str
    predicate: Callable[[MarketIndicators, Params], bool]
    description: str


DEFAULT_PARAMS: Params = {
    # Simple defaults. Later modules can tune these numbers and pass them as `params`.
    # RSI thresholds
    "rsi_oversold": 30.0,
    "rsi_overbought": 70.0,
    "rsi_neutral_low": 40.0,
    "rsi_neutral_high": 60.0,
    # MACD thresholds
    "macd_epsilon": 0.0,
    "macd_strong_threshold": 0.5,
    # Trend strength (percentage margin for MA crossover)
    "ma_crossover_margin": 0.02,  # 2% margin for "strong" trend
    # Volume thresholds
    "volume_high": 1_000_000.0,
    "volume_surge_multiplier": 2.0,
    "volume_average": 500_000.0,
    # Volatility thresholds
    "volatility_high": 0.03,
    "volatility_low": 0.01,
}


def default_fact_definitions() -> List[FactDefinition]:
    """
    Comprehensive fact definitions covering:
    - RSI: oversold, overbought, neutral zones
    - MACD: positive/negative, strong momentum
    - Trend: golden/death cross, strong trends
    - Volume: high volume, volume surge
    - Volatility: high, low
    """
    return [
        # ========== RSI Facts ==========
        FactDefinition(
            name="RSI_OVERSOLD",
            predicate=lambda indicators, params: indicators.rsi < params["rsi_oversold"],
            description="RSI below 30 - potential buying opportunity",
        ),
        FactDefinition(
            name="RSI_OVERBOUGHT",
            predicate=lambda indicators, params: indicators.rsi > params["rsi_overbought"],
            description="RSI above 70 - potential selling opportunity",
        ),
        FactDefinition(
            name="RSI_NEUTRAL",
            predicate=lambda indicators, params: (
                params["rsi_neutral_low"] <= indicators.rsi <= params["rsi_neutral_high"]
            ),
            description="RSI in neutral zone (40-60) - no clear momentum signal",
        ),
        # ========== MACD Facts ==========
        FactDefinition(
            name="MACD_POSITIVE",
            predicate=lambda indicators, params: indicators.macd > params["macd_epsilon"],
            description="MACD is positive (bullish momentum)",
        ),
        FactDefinition(
            name="MACD_NEGATIVE",
            predicate=lambda indicators, params: indicators.macd < -params["macd_epsilon"],
            description="MACD is negative (bearish momentum)",
        ),
        FactDefinition(
            name="MACD_STRONG_POSITIVE",
            predicate=lambda indicators, params: (
                indicators.macd > params["macd_strong_threshold"]
            ),
            description="MACD strongly positive - strong bullish momentum",
        ),
        FactDefinition(
            name="MACD_STRONG_NEGATIVE",
            predicate=lambda indicators, params: (
                indicators.macd < -params["macd_strong_threshold"]
            ),
            description="MACD strongly negative - strong bearish momentum",
        ),
        # ========== Trend Facts (Moving Averages) ==========
        FactDefinition(
            name="GOLDEN_CROSS",
            predicate=lambda indicators, params: indicators.ma20 > indicators.ma50,
            description="MA20 above MA50 (uptrend)",
        ),
        FactDefinition(
            name="DEATH_CROSS",
            predicate=lambda indicators, params: indicators.ma20 < indicators.ma50,
            description="MA20 below MA50 (downtrend)",
        ),
        FactDefinition(
            name="STRONG_UPTREND",
            predicate=lambda indicators, params: (
                indicators.ma20 > indicators.ma50 * (1 + params["ma_crossover_margin"])
            ),
            description="MA20 significantly above MA50 - strong uptrend",
        ),
        FactDefinition(
            name="STRONG_DOWNTREND",
            predicate=lambda indicators, params: (
                indicators.ma20 < indicators.ma50 * (1 - params["ma_crossover_margin"])
            ),
            description="MA20 significantly below MA50 - strong downtrend",
        ),
        # ========== Volume Facts ==========
        FactDefinition(
            name="VOLUME_HIGH",
            predicate=lambda indicators, params: indicators.volume > params["volume_high"],
            description="Volume is above a high-volume threshold",
        ),
        FactDefinition(
            name="VOLUME_SURGE",
            predicate=lambda indicators, params: (
                indicators.volume
                > params["volume_average"] * params["volume_surge_multiplier"]
            ),
            description="Volume surge - unusually high trading activity",
        ),
        # ========== Volatility Facts ==========
        FactDefinition(
            name="VOLATILITY_HIGH",
            predicate=lambda indicators, params: (indicators.volatility is not None)
            and (indicators.volatility > params["volatility_high"]),
            description="Volatility is above a high-volatility threshold",
        ),
        FactDefinition(
            name="VOLATILITY_LOW",
            predicate=lambda indicators, params: (indicators.volatility is not None)
            and (indicators.volatility < params["volatility_low"]),
            description="Volatility is low - stable price action",
        ),
        FactDefinition(
            name="VOLATILITY_UNKNOWN",
            predicate=lambda indicators, params: indicators.volatility is None,
            description="Volatility data not available",
        ),
    ]


def indicators_to_facts(
    indicators: MarketIndicators,
    *,
    params: Optional[Params] = None,
    fact_definitions: Optional[Iterable[FactDefinition]] = None,
) -> Dict[str, bool]:
    """
    Convert numeric indicators to boolean facts for the inference engine.

    Args:
        indicators: Market data (RSI, MACD, MAs, volume, volatility).
        params: Threshold overrides (defaults in DEFAULT_PARAMS).
        fact_definitions: Custom facts (defaults in default_fact_definitions).

    Returns:
        Dict mapping fact names to truth values.
    """

    # Start with defaults, then overwrite any thresholds the caller provides.
    resolved_params = dict(DEFAULT_PARAMS)
    if params:
        resolved_params.update(params)

    # You can pass custom fact definitions if you want different facts.
    definitions = (
        list(fact_definitions)
        if fact_definitions is not None
        else default_fact_definitions()
    )
    return {
        definition.name: bool(definition.predicate(indicators, resolved_params))
        for definition in definitions
    }

