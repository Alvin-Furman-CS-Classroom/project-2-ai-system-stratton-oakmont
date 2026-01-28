from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.shared import MarketIndicators, TradingAction

from .facts import FactDefinition, Params, indicators_to_facts


@dataclass(frozen=True)
class Literal:
    symbol: str
    negated: bool = False

    def is_satisfied(self, truth_assignments: Dict[str, bool]) -> bool:
        value = bool(truth_assignments.get(self.symbol, False))
        return (not value) if self.negated else value


@dataclass(frozen=True)
class HornRule:
    """
    Horn rule: premises => conclusion

    This is a CNF-friendly representation: (¬p1 ∨ ¬p2 ∨ ... ∨ c)
    where c is the (single) positive literal conclusion.
    """

    rule_id: str
    premises: Sequence[Literal]
    conclusion: str
    description: str = ""


@dataclass(frozen=True)
class InferenceStep:
    rule_id: str
    added_fact: str
    supporting_literals: Tuple[Literal, ...]


@dataclass(frozen=True)
class InferenceResult:
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
    Forward chaining over Horn rules.

    We keep `truth_assignments` as symbol->bool for evaluating negated premises.
    Derived conclusions are asserted as True.
    """

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

            if all(lit.is_satisfied(truth_assignments) for lit in r.premises):
                truth_assignments[r.conclusion] = True
                derived.add(r.conclusion)
                fired.append(r.rule_id)
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
                    return derived, fired, chain

    return derived, fired, chain


def choose_action(truth_assignments: Dict[str, bool]) -> Tuple[TradingAction, bool]:
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
    Starter rule set (meant to be replaced/optimized by Modules 2 & 3).

    These are deliberately simple and explainable.
    """

    return [
        HornRule(
            rule_id="BUY_1",
            premises=[
                Literal("RSI_OVERSOLD"),
                Literal("MACD_POSITIVE"),
                Literal("GOLDEN_CROSS"),
                Literal("VOLATILITY_HIGH", negated=True),
            ],
            conclusion=TradingAction.BUY.value,
            description="Buy: oversold + positive momentum + uptrend, but not high volatility",
        ),
        HornRule(
            rule_id="SELL_1",
            premises=[
                Literal("RSI_OVERBOUGHT"),
                Literal("MACD_NEGATIVE"),
                Literal("DEATH_CROSS"),
            ],
            conclusion=TradingAction.SELL.value,
            description="Sell: overbought + negative momentum + downtrend",
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
    Module 1 main entrypoint.

    Input:
    - numeric indicators + optional parameter thresholds
    - a set of Horn/CNF-friendly rules (defaults provided)

    Output:
    - BUY/SELL/HOLD + fired rules + inference chain (explainability)
    """

    base_truths = indicators_to_facts(
        indicators, params=params, fact_definitions=fact_definitions
    )
    # Copy so we can assert derived conclusions as we chain.
    truth_assignments: Dict[str, bool] = dict(base_truths)

    applied_rules = list(rules) if rules is not None else default_trading_rules()
    derived, fired, chain = forward_chain(
        truth_assignments=truth_assignments, rules=applied_rules
    )
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
    Parse a *single* CNF clause into a HornRule when possible.

    Expected shape:
      (~A OR ~B OR CONCLUSION)
    i.e., exactly one positive literal (the conclusion) and 0+ negated literals
    (which become positive premises).
    """

    raw = clause.strip()
    if raw.startswith("(") and raw.endswith(")"):
        raw = raw[1:-1].strip()

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

