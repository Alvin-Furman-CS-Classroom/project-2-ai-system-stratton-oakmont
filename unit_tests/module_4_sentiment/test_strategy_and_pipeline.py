"""Tests for strategy recommendation and sentiment pipeline."""

from unittest.mock import patch

from src.module_4_sentiment.alpha_vantage_client import NewsArticle, NewsSentimentResult
from src.module_4_sentiment.pipeline import analyze_market_sentiment
from src.module_4_sentiment.regime_classifier import MarketRegime
from src.module_4_sentiment.strategy_recommendation import recommend_strategy_for_regime
from src.shared.types import CandidateStrategy


def _c(params: str, sharpe: float, max_dd: float) -> CandidateStrategy:
    return CandidateStrategy(
        params={"id": params},
        sharpe=sharpe,
        total_return=0.1,
        max_drawdown=max_dd,
    )


def test_recommend_bullish_highest_sharpe():
    pool = [_c("a", 0.5, -0.2), _c("b", 1.2, -0.5)]
    s, reason = recommend_strategy_for_regime(MarketRegime.BULLISH, pool)
    assert s is not None
    assert s.params["id"] == "b"
    assert "Sharpe" in reason


def test_recommend_bearish_prefers_less_drawdown():
    pool = [_c("a", 0.3, -0.4), _c("b", 0.4, -0.1)]
    s, _ = recommend_strategy_for_regime(MarketRegime.BEARISH, pool)
    assert s is not None
    assert s.params["id"] == "b"


def test_recommend_neutral_uses_m3_when_present():
    m3 = _c("pick", 0.5, -0.2)
    pool = [m3, _c("other", 0.9, -0.3)]
    s, _ = recommend_strategy_for_regime(MarketRegime.NEUTRAL, pool, m3_selected=m3)
    assert s == m3


def _feed_article(score: float, label: str) -> NewsArticle:
    return NewsArticle(
        title="Headline",
        url="https://example.com",
        time_published="20240101T120000",
        summary="body",
        overall_sentiment_score=score,
        overall_sentiment_label=label,
        raw={},
    )


@patch("src.module_4_sentiment.pipeline.fetch_news_sentiment")
def test_analyze_market_sentiment_pipeline(mock_fetch):
    articles = (
        [_feed_article(-0.3, "Bearish") for _ in range(10)]
        + [_feed_article(0.0, "Neutral") for _ in range(10)]
        + [_feed_article(0.4, "Bullish") for _ in range(10)]
    )
    mock_fetch.return_value = NewsSentimentResult(feed=articles, raw={"feed": []})
    pool = [_c("low", 0.2, -0.15), _c("high", 1.0, -0.4)]
    out = analyze_market_sentiment(
        tickers="SPY",
        candidate_strategies=pool,
        m3_selected=pool[0],
        api_key="test",
        fit_classifier_from_feed=True,
    )
    assert len(out.top_headlines) >= 1
    assert out.recommended_strategy is not None
    assert out.regime in MarketRegime
