"""Reporting utilities for Module 3 strategy selection.

Takes the selected strategy and the pool of candidates and generates a short
natural-language summary suitable for logs or demo output.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

from src.shared import CandidateStrategy

from .selection import (
    SelectionPreferences,
    StrategyConstraints,
    _passes_constraints,
    _drawdown_term,
    _trades_term,
    filter_candidates,
    score_strategy,
)


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


# ---------------------------------------------------------------------------
# Verbose step-by-step trace
# ---------------------------------------------------------------------------

def _header(title: str) -> str:
    """Return a formatted section header."""
    bar = "=" * 60
    return f"\n{bar}\n  {title}\n{bar}"


def _format_strategy_row(
    idx: int,
    strategy: CandidateStrategy,
    score: Optional[float] = None,
    tag: str = "",
) -> str:
    """One-line summary of a strategy for table-style output."""
    parts = [
        f"  [{idx:>3}]",
        f"Sharpe={strategy.sharpe:+.3f}",
        f"Return={strategy.total_return:+.2%}",
        f"MaxDD={strategy.max_drawdown:+.2%}",
        f"WR={strategy.win_rate:.0%}",
        f"Trades={strategy.num_trades}",
    ]
    if score is not None:
        parts.append(f"Score={score:.4f}")
    if tag:
        parts.append(tag)
    return "  ".join(parts)


def _constraint_check_detail(
    strategy: CandidateStrategy, constraints: StrategyConstraints
) -> str:
    """Return which specific constraint(s) a strategy fails, if any."""
    failures: List[str] = []
    if strategy.sharpe < constraints.min_sharpe:
        failures.append(
            f"sharpe {strategy.sharpe:+.3f} < min {constraints.min_sharpe:.2f}"
        )
    if strategy.total_return < constraints.min_total_return:
        failures.append(
            f"return {strategy.total_return:+.2%} < min {constraints.min_total_return:.2%}"
        )
    if strategy.win_rate < constraints.min_win_rate:
        failures.append(
            f"win_rate {strategy.win_rate:.2%} < min {constraints.min_win_rate:.2%}"
        )
    if strategy.max_drawdown < constraints.max_drawdown_min:
        failures.append(
            f"max_dd {strategy.max_drawdown:+.2%} < limit {constraints.max_drawdown_min:.2%}"
        )
    if strategy.num_trades < constraints.min_trades:
        failures.append(
            f"trades {strategy.num_trades} < min {constraints.min_trades}"
        )
    if not failures:
        return "PASS"
    return "FAIL (" + "; ".join(failures) + ")"


def print_selection_steps(
    candidates: Iterable[CandidateStrategy],
    constraints: Optional[StrategyConstraints] = None,
    preferences: Optional[SelectionPreferences] = None,
) -> Optional[CandidateStrategy]:
    """Run the full selection pipeline while printing every step.

    This is the verbose counterpart of ``select_strategy``.  It prints a
    step-by-step trace showing:

    1. **Receive**   – all incoming candidate strategies
    2. **Constrain** – hard-filter each strategy and show pass/fail reasons
    3. **Score**     – compute the composite score for each survivor
    4. **Rank**      – sort survivors by score
    5. **Select**    – pick the best strategy and display the final summary

    Returns the selected ``CandidateStrategy`` (or ``None``).
    """
    if constraints is None:
        constraints = StrategyConstraints()
    if preferences is None:
        preferences = SelectionPreferences()

    cand_list = list(candidates)

    # ------------------------------------------------------------------
    # Step 1 – Receive candidates
    # ------------------------------------------------------------------
    print(_header("Step 1/5: Receive Candidates"))
    print(f"  Received {len(cand_list)} candidate strategies from Module 2.\n")
    for i, c in enumerate(cand_list, 1):
        print(_format_strategy_row(i, c))

    # ------------------------------------------------------------------
    # Step 2 – Apply hard constraints
    # ------------------------------------------------------------------
    print(_header("Step 2/5: Apply Hard Constraints (Filter)"))
    print(
        f"  Constraints: min_sharpe={constraints.min_sharpe:.2f}, "
        f"min_return={constraints.min_total_return:.2%}, "
        f"min_win_rate={constraints.min_win_rate:.2%}, "
        f"max_drawdown>={constraints.max_drawdown_min:.2%}, "
        f"min_trades={constraints.min_trades}\n"
    )

    passed: List[CandidateStrategy] = []
    for i, c in enumerate(cand_list, 1):
        result = _constraint_check_detail(c, constraints)
        status = "PASS" if _passes_constraints(c, constraints) else "REJECT"
        print(f"  [{i:>3}] {status}  {result}")
        if status == "PASS":
            passed.append(c)

    print(f"\n  >>> {len(passed)}/{len(cand_list)} strategies survived filtering.")

    if not passed:
        print(_header("Result: No Strategies Survived"))
        print("  All candidates were eliminated by the hard constraints.")
        print("  Consider relaxing your StrategyConstraints.\n")
        return None

    # ------------------------------------------------------------------
    # Step 3 – Score survivors
    # ------------------------------------------------------------------
    print(_header("Step 3/5: Score Surviving Strategies"))
    print(
        f"  Scoring weights: Sharpe={preferences.w_sharpe:.2f}, "
        f"Return={preferences.w_return:.2f}, "
        f"WinRate={preferences.w_win_rate:.2f}, "
        f"Drawdown={preferences.w_drawdown:.2f}, "
        f"Trades={preferences.w_trades:.2f}\n"
    )
    print(
        f"  Trade target={preferences.target_trades}, "
        f"tolerance=+/-{preferences.trade_tolerance}\n"
    )

    scored: List[Tuple[CandidateStrategy, float]] = []
    for i, c in enumerate(passed, 1):
        dd = _drawdown_term(c.max_drawdown)
        tr = _trades_term(c.num_trades, preferences.target_trades,
                          preferences.trade_tolerance)
        s = score_strategy(c, preferences)
        scored.append((c, s))

        print(f"  [{i:>3}] Composite Score = {s:.4f}")
        print(f"        sharpe_contrib  = {preferences.w_sharpe:.2f} * {c.sharpe:+.3f}  "
              f"= {preferences.w_sharpe * c.sharpe:+.4f}")
        print(f"        return_contrib  = {preferences.w_return:.2f} * {c.total_return:+.3f}  "
              f"= {preferences.w_return * c.total_return:+.4f}")
        print(f"        winrate_contrib = {preferences.w_win_rate:.2f} * {c.win_rate:.3f}  "
              f"= {preferences.w_win_rate * c.win_rate:+.4f}")
        print(f"        dd_term         = 1 + ({c.max_drawdown:+.3f}) = {dd:.3f}  "
              f"weighted = {preferences.w_drawdown * dd:+.4f}")
        print(f"        trades_term     = {tr:.3f} (target={preferences.target_trades}, "
              f"actual={c.num_trades})  weighted = {preferences.w_trades * tr:+.4f}")
        print()

    # ------------------------------------------------------------------
    # Step 4 – Rank
    # ------------------------------------------------------------------
    print(_header("Step 4/5: Rank Strategies by Score"))
    ranked = sorted(scored, key=lambda x: x[1], reverse=True)
    for rank, (c, s) in enumerate(ranked, 1):
        tag = "<-- BEST" if rank == 1 else ""
        print(_format_strategy_row(rank, c, score=s, tag=tag))

    # ------------------------------------------------------------------
    # Step 5 – Select
    # ------------------------------------------------------------------
    best_strategy, best_score = ranked[0]
    print(_header("Step 5/5: Final Selection"))
    print(f"  Winner: Score={best_score:.4f}")
    print(f"  Params : {best_strategy.params}")
    print(f"  Sharpe : {best_strategy.sharpe:+.3f}")
    print(f"  Return : {best_strategy.total_return:+.2%}")
    print(f"  MaxDD  : {best_strategy.max_drawdown:+.2%}")
    print(f"  WinRate: {best_strategy.win_rate:.0%}")
    print(f"  Trades : {best_strategy.num_trades}")
    if best_strategy.explanation:
        print(f"  Reason : {best_strategy.explanation}")
    print()

    return best_strategy

