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
    "rsi_oversold": 30.0,
    "rsi_overbought": 70.0,
    "macd_epsilon": 0.0,
    "volatility_high": 0.03,
    "volume_high": 1_000_000.0,
}


def default_fact_definitions() -> List[FactDefinition]:
    return [
        FactDefinition(
            name="RSI_OVERSOLD",
            predicate=lambda ind, p: ind.rsi < p["rsi_oversold"],
            description="RSI indicates oversold conditions",
        ),
        FactDefinition(
            name="RSI_OVERBOUGHT",
            predicate=lambda ind, p: ind.rsi > p["rsi_overbought"],
            description="RSI indicates overbought conditions",
        ),
        FactDefinition(
            name="MACD_POSITIVE",
            predicate=lambda ind, p: ind.macd > p["macd_epsilon"],
            description="MACD is positive (bullish momentum)",
        ),
        FactDefinition(
            name="MACD_NEGATIVE",
            predicate=lambda ind, p: ind.macd < -p["macd_epsilon"],
            description="MACD is negative (bearish momentum)",
        ),
        FactDefinition(
            name="GOLDEN_CROSS",
            predicate=lambda ind, p: ind.ma20 > ind.ma50,
            description="MA20 above MA50 (uptrend)",
        ),
        FactDefinition(
            name="DEATH_CROSS",
            predicate=lambda ind, p: ind.ma20 < ind.ma50,
            description="MA20 below MA50 (downtrend)",
        ),
        FactDefinition(
            name="VOLUME_HIGH",
            predicate=lambda ind, p: ind.volume > p["volume_high"],
            description="Volume is above a high-volume threshold",
        ),
        FactDefinition(
            name="VOLATILITY_HIGH",
            predicate=lambda ind, p: (ind.volatility is not None)
            and (ind.volatility > p["volatility_high"]),
            description="Volatility is above a high-volatility threshold",
        ),
    ]


def indicators_to_facts(
    indicators: MarketIndicators,
    *,
    params: Optional[Params] = None,
    fact_definitions: Optional[Iterable[FactDefinition]] = None,
) -> Dict[str, bool]:
    """
    Compute boolean facts from numeric indicators.

    Returns a dict mapping symbol -> truth value, so the inference engine can
    evaluate positive and negated literals.
    """

    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)

    defs = list(fact_definitions) if fact_definitions is not None else default_fact_definitions()
    return {d.name: bool(d.predicate(indicators, p)) for d in defs}

