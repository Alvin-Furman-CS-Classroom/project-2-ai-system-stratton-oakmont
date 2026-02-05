"""
Load trading rules from a config file (JSON) so Module 2/3 can write out
"best rules" and Module 1 can load them without code changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

from .engine import HornRule, Literal


# -----------------------------------------------------------------------------
# Config format (JSON)
# -----------------------------------------------------------------------------
# A rule is: { "rule_id": str, "premises": [ {"symbol": str, "negated": bool? } ], "conclusion": str, "description": str? }
# The file can be either:
#   - A list of rules: [ { ... }, { ... } ]
#   - An object with a "rules" key: { "rules": [ ... ] }
# "negated" defaults to false if omitted.
# -----------------------------------------------------------------------------


def _parse_premise(premise_raw: Any) -> Literal:
    """Turn one premise dict into a Literal. Validates shape."""
    if not isinstance(premise_raw, dict) or "symbol" not in premise_raw:
        raise ValueError(
            f"Each premise must be {{'symbol': str, 'negated': bool?}}; got {premise_raw!r}"
        )
    symbol = str(premise_raw["symbol"]).strip()
    if not symbol:
        raise ValueError(f"Premise symbol cannot be empty: {premise_raw!r}")
    negated = bool(premise_raw.get("negated", False))
    return Literal(symbol=symbol, negated=negated)


def _parse_rule(rule_raw: Any) -> HornRule:
    """Turn one rule dict into a HornRule. Validates required fields."""
    if not isinstance(rule_raw, dict):
        raise ValueError(f"Each rule must be a dict; got {type(rule_raw).__name__}")
    rule_id = rule_raw.get("rule_id")
    if rule_id is None or not str(rule_id).strip():
        raise ValueError(f"Rule must have non-empty 'rule_id'; got {rule_raw!r}")
    conclusion = rule_raw.get("conclusion")
    if conclusion is None or not str(conclusion).strip():
        raise ValueError(f"Rule must have non-empty 'conclusion'; got {rule_raw!r}")
    raw_premises = rule_raw.get("premises")
    if raw_premises is None:
        raw_premises = []
    if not isinstance(raw_premises, list):
        raise ValueError(
            f"'premises' must be a list; got {type(raw_premises).__name__}"
        )
    premises = [_parse_premise(prem) for prem in raw_premises]
    description = str(rule_raw.get("description", "")).strip()
    return HornRule(
        rule_id=str(rule_id).strip(),
        premises=premises,
        conclusion=str(conclusion).strip(),
        description=description,
    )


def load_rules_from_dict(data: Union[Dict[str, Any], List[Any]]) -> List[HornRule]:
    """
    Build a list of HornRules from a dict or list (e.g. from JSON).

    - If `data` is a list, it is treated as the list of rule objects.
    - If `data` is a dict, we look for key "rules"; if present, use that list;
      otherwise treat the whole dict as a single rule (one rule only).
    """
    if isinstance(data, list):
        return [_parse_rule(rule_item) for rule_item in data]
    if isinstance(data, dict):
        if "rules" in data:
            raw_rules = data["rules"]
            if not isinstance(raw_rules, list):
                raise ValueError(
                    f"'rules' must be a list; got {type(raw_rules).__name__}"
                )
            return [_parse_rule(rule_item) for rule_item in raw_rules]
        return [_parse_rule(data)]
    raise ValueError(f"Config must be a dict or list; got {type(data).__name__}")


def load_rules_from_file(path: Union[str, Path]) -> List[HornRule]:
    """
    Load rules from a JSON file. Path can be str or pathlib.Path.

    File format: either a JSON array of rule objects, or an object with
    a "rules" key containing that array.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Rules config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    return load_rules_from_dict(data)
