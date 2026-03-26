"""End-to-end sentiment analysis: news → regime → strategy recommendation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.shared.types import CandidateStrategy

from .alpha_vantage_client import NewsArticle, fetch_news_sentiment
from .regime_classifier import MarketRegime, SentimentRegimeClassifier
from .strategy_recommendation import recommend_strategy_for_regime


@dataclass(frozen=True)
class SentimentAnalysisResult:
    """Module 4 output for Module 5 and reporting."""

    regime: MarketRegime
    confidence: float
    classification_method: str
    top_headlines: list[str]
    articles: tuple[NewsArticle, ...]
    recommended_strategy: CandidateStrategy | None
    recommendation_reason: str


def analyze_market_sentiment(
    *,
    tickers: str,
    candidate_strategies: Sequence[CandidateStrategy],
    m3_selected: CandidateStrategy | None = None,
    classifier: SentimentRegimeClassifier | None = None,
    news_limit: int = 50,
    api_key: str | None = None,
    fit_classifier_from_feed: bool = True,
) -> SentimentAnalysisResult:
    """
    Fetch news for ``tickers``, classify regime, recommend a strategy from the pool.

    When ``fit_classifier_from_feed`` is True (default), fits
    ``SentimentRegimeClassifier`` on the returned articles when possible; otherwise
    uses the passed ``classifier`` or a heuristic.
    """
    news = fetch_news_sentiment(
        tickers=tickers,
        limit=news_limit,
        api_key=api_key,
    )
    articles = list(news.feed)

    clf = classifier or SentimentRegimeClassifier()
    if fit_classifier_from_feed:
        clf.fit_from_articles(articles)

    regime, confidence, method = clf.predict_regime(articles)
    top_headlines = [a.title for a in articles[:5]]
    strat, reason = recommend_strategy_for_regime(
        regime,
        candidate_strategies,
        m3_selected=m3_selected,
    )

    return SentimentAnalysisResult(
        regime=regime,
        confidence=confidence,
        classification_method=method,
        top_headlines=top_headlines,
        articles=tuple(articles),
        recommended_strategy=strat,
        recommendation_reason=reason,
    )
