"""Integration tests: Module 4 sentiment pipeline with Module 3-style candidates."""

from unittest.mock import patch

from src.module_4_sentiment.alpha_vantage_client import NewsArticle, NewsSentimentResult
from src.module_4_sentiment.pipeline import analyze_market_sentiment
from src.shared.types import CandidateStrategy


def _article(score: float, label: str) -> NewsArticle:
    return NewsArticle(
        title="Market headline",
        url="https://example.com/n",
        time_published="20240101T120000",
        summary="Summary text for the article.",
        overall_sentiment_score=score,
        overall_sentiment_label=label,
        raw={},
    )


@patch("src.module_4_sentiment.pipeline.fetch_news_sentiment")
def test_m4_pipeline_with_candidate_pool(mock_fetch):
    """End-to-end M4: news feed → regime → strategy pick (no live HTTP)."""
    feed = (
        [_article(-0.2, "Bearish") for _ in range(8)]
        + [_article(0.05, "Neutral") for _ in range(8)]
        + [_article(0.35, "Bullish") for _ in range(8)]
    )
    mock_fetch.return_value = NewsSentimentResult(feed=feed, raw={"feed": []})

    pool = [
        CandidateStrategy(params={"rsi_buy": 30.0}, sharpe=0.4, max_drawdown=-0.25),
        CandidateStrategy(params={"rsi_buy": 35.0}, sharpe=1.1, max_drawdown=-0.35),
    ]
    selected = pool[0]

    result = analyze_market_sentiment(
        tickers="SPY",
        candidate_strategies=pool,
        m3_selected=selected,
        api_key="integration-test",
    )

    assert result.recommended_strategy is not None
    assert result.recommended_strategy in pool
    assert len(result.top_headlines) <= 5
    assert result.classification_method in ("logistic_regression", "heuristic")
