"""Integration tests: Full pipeline (Module 1–5).

Verifies that Module 5 can consume Module 4 output end-to-end
without live API calls.
"""

from unittest.mock import patch

from src.module_4_sentiment.alpha_vantage_client import NewsArticle, NewsSentimentResult
from src.module_4_sentiment.pipeline import analyze_market_sentiment
from src.module_5_position_sizing.sizing import recommend_position_size, PositionSizingResult
from src.module_5_position_sizing.agent import PositionSizingAgent, QAgentConfig
from src.module_5_position_sizing.mdp import POSITION_PCTS, build_state
from src.shared.types import CandidateStrategy


def _article(score: float, label: str) -> NewsArticle:
    return NewsArticle(
        title="Integration test headline",
        url="https://example.com",
        time_published="20240101T120000",
        summary="Body text.",
        overall_sentiment_score=score,
        overall_sentiment_label=label,
        raw={},
    )


def _pool() -> list[CandidateStrategy]:
    return [
        CandidateStrategy(params={"rsi_buy": 30.0}, sharpe=0.6,
                          total_return=0.08, max_drawdown=-0.12),
        CandidateStrategy(params={"rsi_buy": 35.0}, sharpe=1.1,
                          total_return=0.18, max_drawdown=-0.28),
    ]


@patch("src.module_4_sentiment.pipeline.fetch_news_sentiment")
def test_m4_to_m5_untrained(mock_fetch):
    """M4 result feeds directly into M5 with an untrained agent."""
    feed = (
        [_article(-0.25, "Bearish") for _ in range(8)]
        + [_article(0.0, "Neutral") for _ in range(8)]
        + [_article(0.40, "Bullish") for _ in range(8)]
    )
    mock_fetch.return_value = NewsSentimentResult(feed=feed, raw={"feed": []})

    pool = _pool()
    m4 = analyze_market_sentiment(
        tickers="SPY",
        candidate_strategies=pool,
        m3_selected=pool[0],
        api_key="integration-test",
    )
    assert m4.recommended_strategy is not None

    result = recommend_position_size(
        m4, m4.recommended_strategy, volatility=0.15,
    )

    assert isinstance(result, PositionSizingResult)
    assert result.position_pct in POSITION_PCTS.values()
    assert result.dollar_amount > 0
    assert len(result.reasoning) > 0
    assert len(result.risk_assessment) > 0


@patch("src.module_4_sentiment.pipeline.fetch_news_sentiment")
def test_m4_to_m5_trained_agent(mock_fetch):
    """Train the agent briefly, then verify it produces a sensible result."""
    feed = [_article(0.45, "Bullish") for _ in range(20)]
    mock_fetch.return_value = NewsSentimentResult(feed=feed, raw={"feed": []})

    pool = _pool()
    m4 = analyze_market_sentiment(
        tickers="SPY",
        candidate_strategies=pool,
        m3_selected=pool[1],
        api_key="integration-test",
        fit_classifier_from_feed=True,
    )
    assert m4.recommended_strategy is not None

    strat = m4.recommended_strategy
    vol = 0.10
    agent = PositionSizingAgent(QAgentConfig(epsilon=0.0))
    train_state = build_state(
        m4.regime.value, m4.confidence, strat.sharpe, vol,
    )
    states = [train_state for _ in range(5)]
    for _ in range(300):
        agent.train_episode(states, strategy_return=strat.total_return,
                            max_drawdown=strat.max_drawdown, volatility=vol)

    result = recommend_position_size(
        m4, strat, volatility=vol, capital=25_000, agent=agent,
    )

    assert isinstance(result, PositionSizingResult)
    assert result.capital == 25_000
    assert result.position_pct >= 0.05
