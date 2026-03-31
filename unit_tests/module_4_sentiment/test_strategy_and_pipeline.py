"""Tests for strategy recommendation and sentiment pipeline."""

from unittest.mock import patch

from src.module_4_sentiment.alpha_vantage_client import AlphaVantageError
from src.module_4_sentiment.alpha_vantage_client import NewsArticle, NewsSentimentResult
from src.module_4_sentiment.pipeline import analyze_market_sentiment
from src.module_4_sentiment.regime_classifier import MarketRegime
from src.module_4_sentiment.strategy_recommendation import recommend_strategy_for_regime
from src.shared.types import CandidateStrategy


def _c(param_id: float, sharpe: float, max_dd: float) -> CandidateStrategy:
    return CandidateStrategy(
        params={"id": param_id},
        sharpe=sharpe,
        total_return=0.1,
        max_drawdown=max_dd,
    )


def test_recommend_bullish_highest_sharpe():
    pool = [_c(1.0, 0.5, -0.2), _c(2.0, 1.2, -0.5)]
    s, reason = recommend_strategy_for_regime(MarketRegime.BULLISH, pool)
    assert s is not None
    assert s.params["id"] == 2.0
    assert "Sharpe" in reason


def test_recommend_bearish_prefers_less_drawdown():
    pool = [_c(1.0, 0.3, -0.4), _c(2.0, 0.4, -0.1)]
    s, _ = recommend_strategy_for_regime(MarketRegime.BEARISH, pool)
    assert s is not None
    assert s.params["id"] == 2.0


def test_recommend_neutral_uses_m3_when_present():
    m3 = _c(1.0, 0.5, -0.2)
    pool = [m3, _c(2.0, 0.9, -0.3)]
    s, _ = recommend_strategy_for_regime(MarketRegime.NEUTRAL, pool, m3_selected=m3)
    assert s == m3


def _feed_article(score: float, label: str, title: str = "Headline") -> NewsArticle:
    return NewsArticle(
        title=title,
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
    pool = [_c(1.0, 0.2, -0.15), _c(2.0, 1.0, -0.4)]
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


@patch("src.module_4_sentiment.pipeline.fetch_news_sentiment")
def test_analyze_market_sentiment_top_headlines_follow_regime(mock_fetch):
    articles = [
        _feed_article(-0.25, "Bearish", title=f"Bearish first {i}")
        for i in range(5)
    ] + [
        _feed_article(0.55 + 0.01 * i, "Bullish", title=f"Bullish strong {i}")
        for i in range(8)
    ]
    mock_fetch.return_value = NewsSentimentResult(feed=articles, raw={"feed": []})
    pool = [_c(1.0, 0.5, -0.2)]

    out = analyze_market_sentiment(
        tickers="SPY",
        candidate_strategies=pool,
        api_key="test",
    )

    assert out.regime is MarketRegime.BULLISH
    assert out.top_headlines
    assert out.top_headlines[0].startswith("Bullish strong")


@patch("src.module_4_sentiment.pipeline.fetch_news_sentiment")
def test_analyze_market_sentiment_default_no_fit_uses_heuristic(mock_fetch):
    articles = [_feed_article(0.3, "Bullish") for _ in range(12)]
    mock_fetch.return_value = NewsSentimentResult(feed=articles, raw={"feed": []})
    pool = [_c(1.0, 0.5, -0.2)]

    out = analyze_market_sentiment(
        tickers="SPY",
        candidate_strategies=pool,
        api_key="test",
    )

    assert out.classification_method == "heuristic"


@patch("src.module_4_sentiment.pipeline.fetch_news_sentiment")
def test_analyze_market_sentiment_fetch_fallback(mock_fetch):
    mock_fetch.side_effect = AlphaVantageError("rate limit")
    pool = [_c(1.0, 0.5, -0.2)]

    out = analyze_market_sentiment(
        tickers="SPY",
        candidate_strategies=pool,
        api_key="test",
    )

    assert out.regime is MarketRegime.NEUTRAL
    assert out.classification_method == "heuristic"
    assert "neutral fallback" in out.fallback_note.lower()
    assert "neutral fallback" in out.recommendation_reason.lower()


@patch("src.module_4_sentiment.pipeline.fetch_news_sentiment")
def test_analyze_market_sentiment_carries_m3_context(mock_fetch):
    articles = [_feed_article(0.0, "Neutral") for _ in range(10)]
    mock_fetch.return_value = NewsSentimentResult(feed=articles, raw={"feed": []})
    pool = [_c(1.0, 0.5, -0.2)]

    out = analyze_market_sentiment(
        tickers="SPY",
        candidate_strategies=pool,
        api_key="test",
        m3_context={
            "origin": "module_2_search",
            "reason": "Top composite score",
            "summary": "M2 beat GA by sharpe",
        },
    )

    assert out.m3_origin == "module_2_search"
    assert out.m3_reason == "Top composite score"
    assert out.m3_summary == "M2 beat GA by sharpe"
