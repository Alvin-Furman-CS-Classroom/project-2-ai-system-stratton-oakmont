"""Unit tests for Module 3: strategy selection logic."""

from __future__ import annotations

from dataclasses import replace

from src.module_3_strategy_selection import (
    SelectionPreferences,
    StrategyConstraints,
    filter_candidates,
    score_strategy,
    select_strategy,
)
from src.shared import CandidateStrategy


def _make_candidate(
    sharpe: float = 0.0,
    total_return: float = 0.0,
    win_rate: float = 0.0,
    max_drawdown: float = 0.0,
    num_trades: int = 10,
) -> CandidateStrategy:
    return CandidateStrategy(
        params={},
        sharpe=sharpe,
        total_return=total_return,
        win_rate=win_rate,
        max_drawdown=max_drawdown,
        num_trades=num_trades,
    )


def test_filter_candidates_with_defaults_accepts_reasonable_strategies():
    candidates = [
        _make_candidate(sharpe=-0.1),  # below default min_sharpe=0.0
        _make_candidate(sharpe=0.1, total_return=0.05, win_rate=0.55, max_drawdown=-0.2),
    ]

    filtered = filter_candidates(candidates)
    assert len(filtered) == 1
    assert filtered[0].sharpe == 0.1


def test_filter_candidates_respects_custom_constraints():
    constraints = StrategyConstraints(
        min_sharpe=0.5,
        min_total_return=0.10,
        min_win_rate=0.60,
        max_drawdown_min=-0.15,
        min_trades=5,
    )

    good = _make_candidate(
        sharpe=0.6,
        total_return=0.12,
        win_rate=0.65,
        max_drawdown=-0.1,
        num_trades=10,
    )
    bad_sharpe = replace(good, sharpe=0.4)
    bad_return = replace(good, total_return=0.05)
    bad_win_rate = replace(good, win_rate=0.5)
    bad_drawdown = replace(good, max_drawdown=-0.3)
    bad_trades = replace(good, num_trades=2)

    filtered = filter_candidates(
        [good, bad_sharpe, bad_return, bad_win_rate, bad_drawdown, bad_trades],
        constraints=constraints,
    )

    assert filtered == [good]


def test_filter_candidates_empty_input_returns_empty_list():
    assert filter_candidates([]) == []


def test_score_strategy_prefers_higher_quality_metrics():
    prefs = SelectionPreferences()

    base = _make_candidate(
        sharpe=0.5,
        total_return=0.10,
        win_rate=0.55,
        max_drawdown=-0.2,
        num_trades=prefs.target_trades,
    )
    better = _make_candidate(
        sharpe=0.8,
        total_return=0.20,
        win_rate=0.60,
        max_drawdown=-0.1,  # shallower drawdown
        num_trades=prefs.target_trades,
    )

    base_score = score_strategy(base, prefs)
    better_score = score_strategy(better, prefs)
    assert better_score > base_score


def test_trades_far_from_target_are_penalized():
    prefs = SelectionPreferences(target_trades=50, trade_tolerance=20)

    near_target = _make_candidate(num_trades=50, sharpe=0.5)
    far_from_target = _make_candidate(num_trades=100, sharpe=0.5)

    near_score = score_strategy(near_target, prefs)
    far_score = score_strategy(far_from_target, prefs)
    assert near_score > far_score


def test_select_strategy_returns_best_candidate_when_multiple_pass():
    constraints = StrategyConstraints(min_sharpe=0.0)
    prefs = SelectionPreferences()

    c1 = _make_candidate(sharpe=0.5)
    c2 = _make_candidate(sharpe=1.0)
    c3 = _make_candidate(sharpe=0.8)

    best = select_strategy([c1, c2, c3], constraints=constraints, preferences=prefs)
    assert best is c2


def test_select_strategy_returns_none_when_all_fail_constraints():
    constraints = StrategyConstraints(min_sharpe=1.0)

    c1 = _make_candidate(sharpe=0.2)
    c2 = _make_candidate(sharpe=0.5)

    best = select_strategy([c1, c2], constraints=constraints)
    assert best is None

