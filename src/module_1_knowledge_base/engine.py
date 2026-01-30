from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.shared import MarketIndicators, TradingAction

from .facts import FactDefinition, Params, indicators_to_facts


@dataclass(frozen=True)
class Literal:
    """A propositional symbol, optionally negated. Used as rule premises."""

    symbol: str
    negated: bool = False

    def is_satisfied(self, truth_assignments: Dict[str, bool]) -> bool:
        # If a symbol is missing, treat it as False.
        value = bool(truth_assignments.get(self.symbol, False))
        return (not value) if self.negated else value


@dataclass(frozen=True)
class HornRule:
    """
    A simple IF-THEN rule:
      IF (all the premises are true) THEN (the conclusion is true)

    We use this rule shape because it's easy to "chain" forward:
    once one rule makes something true, other rules can use it.
    """

    # `rule_id` lets us say which rule fired (for explainability).
    rule_id: str
    premises: Sequence[Literal]
    conclusion: str
    description: str = ""


@dataclass(frozen=True)
class InferenceStep:
    """One step in the inference chain, for explainability."""

    rule_id: str
    added_fact: str
    supporting_literals: Tuple[Literal, ...]


@dataclass(frozen=True)
class InferenceResult:
    """Complete output from rule evaluation: action, fired rules, and inference trace."""

    action: TradingAction
    fired_rules: Tuple[str, ...]
    inference_chain: Tuple[InferenceStep, ...]
    truth_assignments: Dict[str, bool]
    derived_facts: Tuple[str, ...]
    conflict: bool


def forward_chain(
    *,
    truth_assignments: Dict[str, bool],
    rules: Iterable[HornRule],
    max_steps: int = 256,
) -> Tuple[Set[str], List[str], List[InferenceStep]]:
    """
    Apply Horn rules until no new facts can be derived (fixed-point).

    Args:
        truth_assignments: Known facts (modified in-place as rules fire).
        rules: Horn rules to apply.
        max_steps: Safety limit to prevent infinite loops.

    Returns:
        Tuple of (derived_facts, fired_rule_ids, inference_chain).
    """

    # Keep track of what we've already concluded so we don't repeat forever.
    derived: Set[str] = set()
    fired: List[str] = []
    chain: List[InferenceStep] = []

    rules_list = list(rules)
    steps = 0
    changed = True

    while changed:
        changed = False
        for r in rules_list:
            if r.conclusion in derived:
                continue
            if bool(truth_assignments.get(r.conclusion, False)):
                # already true via facts or earlier derivation
                derived.add(r.conclusion)
                continue

            # A rule fires when ALL its premises match what we know.
            if all(lit.is_satisfied(truth_assignments) for lit in r.premises):
                # Mark the conclusion as true.
                truth_assignments[r.conclusion] = True
                derived.add(r.conclusion)
                fired.append(r.rule_id)
                # Save a "why" step for the explanation.
                chain.append(
                    InferenceStep(
                        rule_id=r.rule_id,
                        added_fact=r.conclusion,
                        supporting_literals=tuple(r.premises),
                    )
                )
                steps += 1
                changed = True

                if steps >= max_steps:
                    # Safety limit (shouldn't happen with sane rules).
                    return derived, fired, chain

    return derived, fired, chain


def choose_action(truth_assignments: Dict[str, bool]) -> Tuple[TradingAction, bool]:
    """Convert derived facts to a trading action. Returns (action, conflict_flag)."""
    buy = bool(truth_assignments.get(TradingAction.BUY.value, False))
    sell = bool(truth_assignments.get(TradingAction.SELL.value, False))
    if buy and sell:
        return TradingAction.HOLD, True
    if buy:
        return TradingAction.BUY, False
    if sell:
        return TradingAction.SELL, False
    return TradingAction.HOLD, False


def default_trading_rules() -> List[HornRule]:
    """
    Comprehensive trading rules organized by strategy type.
    
    Categories:
    - Momentum Continuation: Ride existing trends with confirmation
    - Mean Reversion: Enter on pullbacks within trends
    - Volume Confirmed: Require volume for signal validation
    - Conservative: Multiple confirmations, lower risk
    - Aggressive: Fewer conditions, faster entry
    
    Later modules (2 & 3) will tune parameters and evolve better rule combinations.
    """

    return [
        # ========== MOMENTUM CONTINUATION (BUY) ==========
        HornRule(
            rule_id="BUY_MOMENTUM_1",
            premises=[
                Literal("RSI_OVERSOLD"),
                Literal("MACD_POSITIVE"),
                Literal("GOLDEN_CROSS"),
                Literal("VOLATILITY_HIGH", negated=True),
            ],
            conclusion=TradingAction.BUY.value,
            description="Classic momentum buy: oversold RSI + positive MACD + uptrend, avoid high volatility",
        ),
        HornRule(
            rule_id="BUY_MOMENTUM_STRONG",
            premises=[
                Literal("STRONG_UPTREND"),
                Literal("MACD_STRONG_POSITIVE"),
                Literal("VOLUME_HIGH"),
            ],
            conclusion=TradingAction.BUY.value,
            description="Strong momentum buy: confirmed uptrend with strong MACD and volume",
        ),
        # ========== MOMENTUM CONTINUATION (SELL) ==========
        HornRule(
            rule_id="SELL_MOMENTUM_1",
            premises=[
                Literal("RSI_OVERBOUGHT"),
                Literal("MACD_NEGATIVE"),
                Literal("DEATH_CROSS"),
            ],
            conclusion=TradingAction.SELL.value,
            description="Classic momentum sell: overbought RSI + negative MACD + downtrend",
        ),
        HornRule(
            rule_id="SELL_MOMENTUM_STRONG",
            premises=[
                Literal("STRONG_DOWNTREND"),
                Literal("MACD_STRONG_NEGATIVE"),
                Literal("VOLUME_HIGH"),
            ],
            conclusion=TradingAction.SELL.value,
            description="Strong momentum sell: confirmed downtrend with strong bearish MACD and volume",
        ),
        # ========== MEAN REVERSION (Pullback entries) ==========
        HornRule(
            rule_id="BUY_PULLBACK",
            premises=[
                Literal("GOLDEN_CROSS"),  # Still in uptrend
                Literal("RSI_OVERSOLD"),  # But temporarily oversold (pullback)
                Literal("VOLATILITY_HIGH", negated=True),
            ],
            conclusion=TradingAction.BUY.value,
            description="Pullback buy: oversold in an uptrend - buy the dip",
        ),
        HornRule(
            rule_id="SELL_RALLY",
            premises=[
                Literal("DEATH_CROSS"),  # Still in downtrend
                Literal("RSI_OVERBOUGHT"),  # But temporarily overbought (relief rally)
                Literal("VOLATILITY_HIGH", negated=True),
            ],
            conclusion=TradingAction.SELL.value,
            description="Rally sell: overbought in a downtrend - sell the rip",
        ),
        # ========== VOLUME CONFIRMED ==========
        HornRule(
            rule_id="BUY_VOLUME_BREAKOUT",
            premises=[
                Literal("GOLDEN_CROSS"),
                Literal("MACD_POSITIVE"),
                Literal("VOLUME_SURGE"),
            ],
            conclusion=TradingAction.BUY.value,
            description="Volume breakout buy: uptrend confirmed by unusual volume surge",
        ),
        HornRule(
            rule_id="SELL_VOLUME_BREAKDOWN",
            premises=[
                Literal("DEATH_CROSS"),
                Literal("MACD_NEGATIVE"),
                Literal("VOLUME_SURGE"),
            ],
            conclusion=TradingAction.SELL.value,
            description="Volume breakdown sell: downtrend confirmed by unusual volume surge",
        ),
        # ========== CONSERVATIVE (Multiple confirmations) ==========
        HornRule(
            rule_id="BUY_CONSERVATIVE",
            premises=[
                Literal("RSI_OVERSOLD"),
                Literal("MACD_POSITIVE"),
                Literal("GOLDEN_CROSS"),
                Literal("VOLUME_HIGH"),
                Literal("VOLATILITY_HIGH", negated=True),
            ],
            conclusion=TradingAction.BUY.value,
            description="Conservative buy: all signals aligned - oversold, positive momentum, uptrend, volume, low volatility",
        ),
        HornRule(
            rule_id="SELL_CONSERVATIVE",
            premises=[
                Literal("RSI_OVERBOUGHT"),
                Literal("MACD_NEGATIVE"),
                Literal("DEATH_CROSS"),
                Literal("VOLUME_HIGH"),
                Literal("VOLATILITY_HIGH", negated=True),
            ],
            conclusion=TradingAction.SELL.value,
            description="Conservative sell: all signals aligned - overbought, negative momentum, downtrend, volume, low volatility",
        ),
        # ========== AGGRESSIVE (Faster entry, fewer conditions) ==========
        HornRule(
            rule_id="BUY_AGGRESSIVE",
            premises=[
                Literal("RSI_OVERSOLD"),
                Literal("STRONG_UPTREND"),
            ],
            conclusion=TradingAction.BUY.value,
            description="Aggressive buy: oversold in strong uptrend - fast entry",
        ),
        HornRule(
            rule_id="SELL_AGGRESSIVE",
            premises=[
                Literal("RSI_OVERBOUGHT"),
                Literal("STRONG_DOWNTREND"),
            ],
            conclusion=TradingAction.SELL.value,
            description="Aggressive sell: overbought in strong downtrend - fast exit",
        ),
        # ========== LOW VOLATILITY OPPORTUNITIES ==========
        HornRule(
            rule_id="BUY_LOW_VOL",
            premises=[
                Literal("GOLDEN_CROSS"),
                Literal("MACD_POSITIVE"),
                Literal("VOLATILITY_LOW"),
            ],
            conclusion=TradingAction.BUY.value,
            description="Low volatility buy: stable uptrend with positive momentum",
        ),
        HornRule(
            rule_id="SELL_LOW_VOL",
            premises=[
                Literal("DEATH_CROSS"),
                Literal("MACD_NEGATIVE"),
                Literal("VOLATILITY_LOW"),
            ],
            conclusion=TradingAction.SELL.value,
            description="Low volatility sell: stable downtrend with negative momentum",
        ),
    ]


def evaluate_rules_on_indicators(
    indicators: MarketIndicators,
    *,
    rules: Optional[Sequence[HornRule]] = None,
    params: Optional[Params] = None,
    fact_definitions: Optional[Sequence[FactDefinition]] = None,
) -> InferenceResult:
    """
    Main entrypoint: evaluate trading rules on market indicators.

    Args:
        indicators: Current market data.
        rules: Trading rules (defaults to default_trading_rules()).
        params: Threshold overrides for fact generation.
        fact_definitions: Custom fact definitions.

    Returns:
        InferenceResult with action (BUY/SELL/HOLD), fired rules, and inference chain.
    """

    # 1) Turn numbers into yes/no facts.
    base_truths = indicators_to_facts(
        indicators, params=params, fact_definitions=fact_definitions
    )
    # 2) Use rules to add more true facts.
    # Copy so we can assert derived conclusions as we chain.
    truth_assignments: Dict[str, bool] = dict(base_truths)

    applied_rules = list(rules) if rules is not None else default_trading_rules()
    derived, fired, chain = forward_chain(
        truth_assignments=truth_assignments, rules=applied_rules
    )
    # 3) Convert the final facts into BUY/SELL/HOLD.
    action, conflict = choose_action(truth_assignments)
    return InferenceResult(
        action=action,
        fired_rules=tuple(fired),
        inference_chain=tuple(chain),
        truth_assignments=truth_assignments,
        derived_facts=tuple(sorted(derived)),
        conflict=conflict,
    )


def horn_rule_from_cnf_clause(*, clause: str, rule_id: str, description: str = "") -> HornRule:
    """
    Turn one CNF clause string into a HornRule (when possible).

    We only support clauses like:
      (~A OR ~B OR C)
    which means:
      IF A and B are true, THEN C is true
    """

    raw = clause.strip()
    if raw.startswith("(") and raw.endswith(")"):
        raw = raw[1:-1].strip()

    # Split on "OR" to get each part.
    parts = [p.strip() for p in raw.split("OR")]
    if any(not p for p in parts):
        raise ValueError(f"Invalid clause: {clause!r}")

    positives: List[str] = []
    premises: List[Literal] = []

    for p in parts:
        if p.startswith("~"):
            sym = p[1:].strip()
            if not sym:
                raise ValueError(f"Invalid negated literal: {p!r}")
            # "~A" becomes the premise "A must be true".
            premises.append(Literal(sym, negated=False))
        elif p.upper().startswith("NOT "):
            sym = p[4:].strip()
            if not sym:
                raise ValueError(f"Invalid negated literal: {p!r}")
            premises.append(Literal(sym, negated=False))
        else:
            positives.append(p)

    if len(positives) != 1:
        raise ValueError(
            f"Clause must have exactly one positive literal to be Horn-convertible: {clause!r}"
        )

    return HornRule(
        rule_id=rule_id,
        premises=premises,
        conclusion=positives[0],
        description=description,
    )

