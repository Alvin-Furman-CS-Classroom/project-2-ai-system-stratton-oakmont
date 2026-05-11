"""Unit tests for selection scoring helpers (drawdown / trades terms)."""

from __future__ import annotations

import pytest

from src.module_3_strategy_selection.selection import _drawdown_term, _trades_term


@pytest.mark.parametrize(
    "dd, expected",
    [
        (0.0, 1.0),
        (-0.15, 0.85),
        (-1.0, 0.0),
        (0.5, 1.0),  # positive clamped to 0 → term 1.0
        (-2.0, 0.0),  # below -1 clamped → term 0.0
    ],
)
def test_drawdown_term_clamping(dd: float, expected: float):
    assert abs(_drawdown_term(dd) - expected) < 1e-9


def test_trades_term_target_zero_returns_zero():
    assert _trades_term(num_trades=50, target=0, tolerance=10) == 0.0


def test_trades_term_at_target_is_one():
    assert _trades_term(50, target=50, tolerance=30) == 1.0


def test_trades_term_at_tolerance_boundary_is_zero():
    assert _trades_term(20, target=50, tolerance=30) == 0.0
    assert _trades_term(80, target=50, tolerance=30) == 0.0


def test_trades_term_linear_midpoint():
    # diff=15, tolerance=30 → 1 - 15/30 = 0.5
    assert abs(_trades_term(35, target=50, tolerance=30) - 0.5) < 1e-9
