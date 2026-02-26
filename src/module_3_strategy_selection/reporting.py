"""Reporting utilities for Module 3 strategy selection.

Takes the selected strategy and the pool of candidates and generates a short
natural-language summary suitable for logs or demo output.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

from src.shared import CandidateStrategy

from .selection import SelectionPreferences, StrategyConstraints, score_strategy


def _basic_stats(candidates: Iterable[CandidateStrategy]) -> Tuple[int, float, float]:
    """Return (count, best_sharpe, median_sharpe) for context in the summary."""
    lst = list(candidates)
    if not lst:
        return 0, 0.0, 0.0
    sharpe_values = sorted(c.sharpe for c in lst)
    n = len(sharpe_values)
    best = sharpe_values[-1]
    median = sharpe_values[n // 2] if n % 2 == 1 else (
        sharpe_values[n // 2 - 1] + sharpe_values[n // 2]
    ) / 2.0
    return n, float(best), float(median)


def summarize_selection(
    selected: Optional[CandidateStrategy],
    candidates: Iterable[CandidateStrategy],
    constraints: Optional[StrategyConstraints] = None,
    preferences: Optional[SelectionPreferences] = None,
) -> str:
    """Produce a short human-readable summary of the selection step."""
    cand_list = list(candidates)
    total_n, best_sharpe, median_sharpe = _basic_stats(cand_list)

    if constraints is None:
        constraints = StrategyConstraints()
    if preferences is None:
        preferences = SelectionPreferences()

    if selected is None:
        return (
            "No strategy satisfied the selection constraints\n"
            f"- Candidates provided: {total_n}, best Sharpe={best_sharpe:+.3f}, "
            f"median Sharpe={median_sharpe:+.3f}\n"
            f"- Constraints: min_sharpe={constraints.min_sharpe:.2f}, "
            f"min_return={constraints.min_total_return:.2%}, "
            f"min_win_rate={constraints.min_win_rate:.2%}, "
            f"max_drawdown_min={constraints.max_drawdown_min:.2%}, "
            f"min_trades={constraints.min_trades}"
        )

    # Compute the composite score for the selected strategy for transparency.
    score = score_strategy(selected, preferences)

    return (
        "Selected final strategy based on Module 3 preferences\n"
        f"- Score={score:.3f} from weights "
        f"(Sharpe={preferences.w_sharpe:.2f}, "
        f"Return={preferences.w_return:.2f}, "
        f"WinRate={preferences.w_win_rate:.2f}, "
        f"Drawdown={preferences.w_drawdown:.2f}, "
        f"Trades={preferences.w_trades:.2f})\n"
        f"- Metrics: Sharpe={selected.sharpe:+.3f}, "
        f"Return={selected.total_return:+.2%}, "
        f"MaxDD={selected.max_drawdown:+.2%}, "
        f"WinRate={selected.win_rate:.0%}, "
        f"Trades={selected.num_trades}\n"
        f"- Candidates considered: {total_n}, best Sharpe in pool={best_sharpe:+.3f}, "
        f"median Sharpe in pool={median_sharpe:+.3f}\n"
        f"- Constraints: min_sharpe={constraints.min_sharpe:.2f}, "
        f"min_return={constraints.min_total_return:.2%}, "
        f"min_win_rate={constraints.min_win_rate:.2%}, "
        f"max_drawdown_min={constraints.max_drawdown_min:.2%}, "
        f"min_trades={constraints.min_trades}"
    )

