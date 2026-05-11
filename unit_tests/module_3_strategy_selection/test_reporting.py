"""Unit tests for Module 3: reporting utilities."""

from __future__ import annotations

from src.module_3_strategy_selection import (
    SelectionPreferences,
    StrategyConstraints,
    summarize_selection,
)
from src.module_3_strategy_selection.reporting import _basic_stats, print_selection_steps
from src.shared import CandidateStrategy


def _make_candidate(sharpe: float) -> CandidateStrategy:
    return CandidateStrategy(
        params={},
        sharpe=sharpe,
        total_return=0.0,
        win_rate=0.0,
        max_drawdown=0.0,
        num_trades=10,
    )


def test_basic_stats_empty_returns_zeros():
    count, best, median = _basic_stats([])
    assert count == 0
    assert best == 0.0
    assert median == 0.0


def test_basic_stats_odd_and_even_counts():
    odd_candidates = [_make_candidate(s) for s in [-0.1, 0.0, 0.5]]
    count, best, median = _basic_stats(odd_candidates)
    assert count == 3
    assert best == 0.5
    assert median == 0.0

    even_candidates = [_make_candidate(s) for s in [-0.1, 0.0, 0.5, 1.0]]
    count2, best2, median2 = _basic_stats(even_candidates)
    assert count2 == 4
    assert best2 == 1.0
    assert median2 == (0.0 + 0.5) / 2.0


def test_summarize_selection_when_none_selected_mentions_constraints():
    constraints = StrategyConstraints(min_sharpe=0.5, min_total_return=0.1)
    prefs = SelectionPreferences()
    candidates = [_make_candidate(0.1), _make_candidate(0.2)]

    text = summarize_selection(
        selected=None,
        candidates=candidates,
        constraints=constraints,
        preferences=prefs,
    )

    assert "No strategy satisfied the selection constraints" in text
    assert "min_sharpe=0.50" in text
    assert "min_return=10.00%" in text


def test_summarize_selection_when_selected_includes_metrics():
    constraints = StrategyConstraints(min_sharpe=-1.0)
    prefs = SelectionPreferences()

    selected = CandidateStrategy(
        params={},
        sharpe=0.75,
        total_return=0.2,
        win_rate=0.6,
        max_drawdown=-0.15,
        num_trades=30,
    )
    candidates = [selected]

    text = summarize_selection(
        selected=selected,
        candidates=candidates,
        constraints=constraints,
        preferences=prefs,
    )

    assert "Selected final strategy based on Module 3 preferences" in text
    assert "Sharpe=+0.750" in text
    assert "Return=+20.00%" in text
    assert "MaxDD=-15.00%" in text
    assert "WinRate=60%" in text
    assert "Trades=30" in text


def test_print_selection_steps_smoke_test(capsys):
    """Ensure print_selection_steps runs without error and returns a strategy."""
    candidates = [
        CandidateStrategy(
            params={},
            sharpe=0.5,
            total_return=0.1,
            win_rate=0.55,
            max_drawdown=-0.2,
            num_trades=10,
        ),
        CandidateStrategy(
            params={},
            sharpe=0.8,
            total_return=0.15,
            win_rate=0.6,
            max_drawdown=-0.1,
            num_trades=12,
        ),
    ]

    best = print_selection_steps(candidates)
    captured = capsys.readouterr()

    assert best is not None
    # Check that some key section headers were printed.
    assert "Step 1/5: Receive Candidates" in captured.out
    assert "Step 5/5: Final Selection" in captured.out

