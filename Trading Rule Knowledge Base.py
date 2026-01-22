from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
from enum import Enum
import re


class TradingAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class MarketIndicators:
    """Current market state"""
    RSI: float  # Relative Strength Index (0-100)
    MACD: float  # Moving Average Convergence Divergence
    MA20: float  # 20-day Moving Average
    MA50: float  # 50-day Moving Average
    Volume: float  # Trading volume
    Price: float  # Current price
    
    def to_propositions(self) -> Set[str]:
        """
        Convert market indicators to propositional symbols based on conditions.
        Returns a set of true propositions about the current market state.
        """
        props = set()
        
        # RSI conditions
        if self.RSI > 70:
            props.add("RSI_OVERBOUGHT")
        elif self.RSI < 30:
            props.add("RSI_OVERSOLD")
        else:
            props.add("RSI_NEUTRAL")
        
        # MACD conditions
        if self.MACD > 0:
            props.add("MACD_POSITIVE")
        else:
            props.add("MACD_NEGATIVE")
        
        # Moving Average conditions
        if self.Price > self.MA20:
            props.add("PRICE_ABOVE_MA20")
        else:
            props.add("PRICE_BELOW_MA20")
            
        if self.Price > self.MA50:
            props.add("PRICE_ABOVE_MA50")
        else:
            props.add("PRICE_BELOW_MA50")
        
        if self.MA20 > self.MA50:
            props.add("MA20_ABOVE_MA50")  # Golden Cross indicator
        else:
            props.add("MA20_BELOW_MA50")  # Death Cross indicator
        
        # Volume conditions (simplified - you'd compare to average volume)
        if self.Volume > 1000000:  # Example threshold
            props.add("HIGH_VOLUME")
        else:
            props.add("LOW_VOLUME")
        
        return props


@dataclass
class Clause:
    """
    Represents a clause in CNF (Conjunctive Normal Form).
    A clause is a disjunction of literals: (A OR B OR NOT C)
    """
    literals: List[Tuple[str, bool]]  # (symbol, is_positive)
    
    def __str__(self):
        lit_strs = []
        for symbol, is_positive in self.literals:
            lit_strs.append(symbol if is_positive else f"NOT {symbol}")
        return "(" + " OR ".join(lit_strs) + ")"
    
    def is_unit_clause(self) -> bool:
        """Check if this is a unit clause (single literal)"""
        return len(self.literals) == 1
    
    def get_unit_literal(self) -> Optional[Tuple[str, bool]]:
        """Get the literal if this is a unit clause"""
        if self.is_unit_clause():
            return self.literals[0]
        return None


@dataclass
class Rule:
    """
    Represents a trading rule in the form: IF conditions THEN action
    Stored internally as CNF clauses.
    """
    name: str
    conditions: List[str]  # Antecedents (premises)
    action: str  # Consequent (conclusion)
    priority: int = 1  # Higher priority rules fire first
    
    def to_cnf_clauses(self) -> List[Clause]:
        """
        Convert rule to CNF format.
        IF A AND B THEN C becomes: (NOT A OR NOT B OR C)
        This is equivalent to: (A AND B) => C
        """
        # The implication A ∧ B => C is equivalent to ¬A ∨ ¬B ∨ C
        literals = [(cond, False) for cond in self.conditions]
        literals.append((self.action, True))
        return [Clause(literals)]
    
    def __str__(self):
        conditions_str = " AND ".join(self.conditions)
        return f"{self.name}: IF {conditions_str} THEN {self.action}"


class KnowledgeBase:
    """
    Knowledge Base that stores rules and facts in CNF format.
    Supports forward chaining inference.
    """
    
    def __init__(self):
        self.rules: List[Rule] = []
        self.facts: Set[str] = set()  # Known true propositions
        self.inference_chain: List[str] = []  # Track reasoning steps
    
    def add_rule(self, rule: Rule):
        """Add a trading rule to the knowledge base"""
        self.rules.append(rule)
        # Sort by priority (higher first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def add_fact(self, fact: str):
        """Add a known fact (true proposition)"""
        self.facts.add(fact)
    
    def set_facts(self, facts: Set[str]):
        """Set all facts at once (replaces existing facts)"""
        self.facts = facts.copy()
        self.inference_chain = []
    
    def forward_chaining(self) -> Set[str]:
        """
        Perform forward chaining inference.
        Returns all facts that can be inferred.
        """
        self.inference_chain = []
        inferred = set()
        agenda = list(self.facts)  # Queue of facts to process
        
        self.inference_chain.append(f"Initial facts: {sorted(self.facts)}")
        
        while agenda:
            fact = agenda.pop(0)
            
            if fact in inferred:
                continue
            
            inferred.add(fact)
            
            # Check each rule to see if it can fire
            for rule in self.rules:
                # Check if all conditions are satisfied
                if all(cond in inferred for cond in rule.conditions):
                    # Check if conclusion is new
                    if rule.action not in inferred:
                        agenda.append(rule.action)
                        step = (f"Rule '{rule.name}' fired: "
                               f"{' AND '.join(rule.conditions)} => {rule.action}")
                        self.inference_chain.append(step)
        
        return inferred
    
    def get_trading_action(self) -> Tuple[TradingAction, List[str], List[str]]:
        """
        Determine trading action based on current facts.
        Returns: (action, fired_rules, inference_chain)
        """
        # Perform inference
        inferred_facts = self.forward_chaining()
        
        # Track which rules fired
        fired_rules = []
        
        # Check for action conclusions in priority order
        action_map = {
            "ACTION_BUY": TradingAction.BUY,
            "ACTION_SELL": TradingAction.SELL,
            "ACTION_HOLD": TradingAction.HOLD
        }
        
        for action_symbol, action_enum in action_map.items():
            if action_symbol in inferred_facts:
                # Find which rule(s) led to this action
                for rule in self.rules:
                    if rule.action == action_symbol:
                        if all(cond in inferred_facts for cond in rule.conditions):
                            fired_rules.append(str(rule))
                
                return action_enum, fired_rules, self.inference_chain
        
        # Default to HOLD if no action inferred
        return TradingAction.HOLD, [], self.inference_chain


def create_default_trading_rules() -> KnowledgeBase:
    """
    Create a knowledge base with default trading rules.
    These rules can be optimized by Modules 2 and 3.
    """
    kb = KnowledgeBase()
    
    # Strong Buy Signals
    kb.add_rule(Rule(
        name="Strong_Buy_Signal",
        conditions=["RSI_OVERSOLD", "MACD_POSITIVE", "MA20_ABOVE_MA50"],
        action="ACTION_BUY",
        priority=10
    ))
    
    kb.add_rule(Rule(
        name="Golden_Cross_Buy",
        conditions=["MA20_ABOVE_MA50", "HIGH_VOLUME", "PRICE_ABOVE_MA20"],
        action="ACTION_BUY",
        priority=8
    ))
    
    # Strong Sell Signals
    kb.add_rule(Rule(
        name="Strong_Sell_Signal",
        conditions=["RSI_OVERBOUGHT", "MACD_NEGATIVE", "MA20_BELOW_MA50"],
        action="ACTION_SELL",
        priority=10
    ))
    
    kb.add_rule(Rule(
        name="Death_Cross_Sell",
        conditions=["MA20_BELOW_MA50", "HIGH_VOLUME", "PRICE_BELOW_MA20"],
        action="ACTION_SELL",
        priority=8
    ))
    
    # Moderate Signals
    kb.add_rule(Rule(
        name="Oversold_Buy",
        conditions=["RSI_OVERSOLD", "PRICE_BELOW_MA50"],
        action="ACTION_BUY",
        priority=5
    ))
    
    kb.add_rule(Rule(
        name="Overbought_Sell",
        conditions=["RSI_OVERBOUGHT", "PRICE_ABOVE_MA50"],
        action="ACTION_SELL",
        priority=5
    ))
    
    # Conservative Hold Rules
    kb.add_rule(Rule(
        name="Neutral_Hold",
        conditions=["RSI_NEUTRAL", "LOW_VOLUME"],
        action="ACTION_HOLD",
        priority=3
    ))
    
    kb.add_rule(Rule(
        name="Conflicting_Signals_Hold",
        conditions=["RSI_OVERSOLD", "MACD_NEGATIVE"],
        action="ACTION_HOLD",
        priority=6
    ))
    
    return kb


# Example usage and testing
def main():
    # Create knowledge base with rules
    kb = create_default_trading_rules()
    
    print("=" * 70)
    print("TRADING RULE KNOWLEDGE BASE")
    print("=" * 70)
    print("\nLoaded Rules:")
    for i, rule in enumerate(kb.rules, 1):
        print(f"{i}. {rule}")
    
    # Example 1: Bullish market scenario
    print("\n" + "=" * 70)
    print("SCENARIO 1: Bullish Market Conditions")
    print("=" * 70)
    
    market1 = MarketIndicators(
        RSI=25,          # Oversold
        MACD=5.2,        # Positive
        MA20=152.0,
        MA50=148.0,      # Golden cross
        Volume=1500000,  # High volume
        Price=153.0      # Above MA20
    )
    
    print(f"\nMarket Indicators:")
    print(f"  RSI: {market1.RSI}")
    print(f"  MACD: {market1.MACD}")
    print(f"  Price: ${market1.Price}")
    print(f"  MA20: ${market1.MA20}")
    print(f"  MA50: ${market1.MA50}")
    print(f"  Volume: {market1.Volume:,}")
    
    # Convert to propositions and add to KB
    propositions = market1.to_propositions()
    print(f"\nDerived Propositions: {sorted(propositions)}")
    
    kb.set_facts(propositions)
    action, fired_rules, inference_chain = kb.get_trading_action()
    
    print(f"\n--- Inference Chain ---")
    for step in inference_chain:
        print(f"  {step}")
    
    print(f"\n--- Decision ---")
    print(f"Recommended Action: {action.value}")
    print(f"\nRules That Fired:")
    for rule in fired_rules:
        print(f"  • {rule}")
    
    # Example 2: Bearish market scenario
    print("\n" + "=" * 70)
    print("SCENARIO 2: Bearish Market Conditions")
    print("=" * 70)
    
    market2 = MarketIndicators(
        RSI=78,          # Overbought
        MACD=-3.1,       # Negative
        MA20=145.0,
        MA50=148.0,      # Death cross
        Volume=1800000,  # High volume
        Price=144.0      # Below MA20
    )
    
    print(f"\nMarket Indicators:")
    print(f"  RSI: {market2.RSI}")
    print(f"  MACD: {market2.MACD}")
    print(f"  Price: ${market2.Price}")
    print(f"  MA20: ${market2.MA20}")
    print(f"  MA50: ${market2.MA50}")
    print(f"  Volume: {market2.Volume:,}")
    
    propositions2 = market2.to_propositions()
    print(f"\nDerived Propositions: {sorted(propositions2)}")
    
    kb.set_facts(propositions2)
    action2, fired_rules2, inference_chain2 = kb.get_trading_action()
    
    print(f"\n--- Inference Chain ---")
    for step in inference_chain2:
        print(f"  {step}")
    
    print(f"\n--- Decision ---")
    print(f"Recommended Action: {action2.value}")
    print(f"\nRules That Fired:")
    for rule in fired_rules2:
        print(f"  • {rule}")
    
    # Example 3: Neutral/Conflicting market
    print("\n" + "=" * 70)
    print("SCENARIO 3: Neutral/Conflicting Signals")
    print("=" * 70)
    
    market3 = MarketIndicators(
        RSI=50,          # Neutral
        MACD=0.5,        # Slightly positive
        MA20=150.0,
        MA50=150.5,      # Very close
        Volume=800000,   # Low volume
        Price=150.2
    )
    
    print(f"\nMarket Indicators:")
    print(f"  RSI: {market3.RSI}")
    print(f"  MACD: {market3.MACD}")
    print(f"  Price: ${market3.Price}")
    print(f"  MA20: ${market3.MA20}")
    print(f"  MA50: ${market3.MA50}")
    print(f"  Volume: {market3.Volume:,}")
    
    propositions3 = market3.to_propositions()
    print(f"\nDerived Propositions: {sorted(propositions3)}")
    
    kb.set_facts(propositions3)
    action3, fired_rules3, inference_chain3 = kb.get_trading_action()
    
    print(f"\n--- Inference Chain ---")
    for step in inference_chain3:
        print(f"  {step}")
    
    print(f"\n--- Decision ---")
    print(f"Recommended Action: {action3.value}")
    print(f"\nRules That Fired:")
    if fired_rules3:
        for rule in fired_rules3:
            print(f"  • {rule}")
    else:
        print(f"  (No specific rules fired - default to HOLD)")


if __name__ == "__main__":
    main()