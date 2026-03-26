"""Map sentiment regime + Module 3 candidate pool to a recommended strategy."""

from __future__ import annotations

from typing import Sequence

from src.shared.types import CandidateStrategy

from .regime_classifier import MarketRegime


def recommend_strategy_for_regime(
    regime: MarketRegime,
    candidates: Sequence[CandidateStrategy],
    *,
    m3_selected: CandidateStrategy | None = None,
) -> tuple[CandidateStrategy | None, str]:
    """
    Choose one strategy from ``candidates`` using simple regime-aware rules.

    - **Bullish**: prefer highest Sharpe (more aggressive / trend-following bias).
    - **Bearish**: prefer lowest max drawdown among strategies with Sharpe >= 0.
    - **Neutral**: prefer ``m3_selected`` if it appears in ``candidates``; else median Sharpe.

    If ``candidates`` is empty, returns ``(None, ...)``.
    """
    pool = list(candidates)
    if not pool:
        return None, "No candidate strategies provided."

    if regime is MarketRegime.BULLISH:
        best = max(pool, key=lambda s: s.sharpe)
        return best, (
            f"Bullish regime: selected highest-Sharpe candidate (Sharpe={best.sharpe:+.3f}) "
            "from the Module 3 pool."
        )

    if regime is MarketRegime.BEARISH:
        viable = [s for s in pool if s.sharpe >= 0.0]
        sub = viable if viable else pool
        # max_drawdown is negative (e.g. -0.15); pick least severe = max() by value
        best = max(sub, key=lambda s: s.max_drawdown)
        return best, (
            f"Bearish regime: selected defensive candidate (best max drawdown "
            f"{best.max_drawdown:+.2%} among Sharpe≥0 pool) from the Module 3 pool."
        )

    # Neutral
    if m3_selected is not None and m3_selected in pool:
        return m3_selected, (
            "Neutral regime: using Module 3 unified selection as the default recommendation."
        )

    ordered = sorted(pool, key=lambda s: s.sharpe)
    mid = ordered[len(ordered) // 2]
    return mid, (
        f"Neutral regime: no M3 match in pool; selected median-Sharpe candidate "
        f"(Sharpe={mid.sharpe:+.3f})."
    )
