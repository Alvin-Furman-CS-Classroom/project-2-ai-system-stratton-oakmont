"""End-to-end sentiment analysis: news → regime → strategy recommendation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.shared.types import CandidateStrategy

from .alpha_vantage_client import AlphaVantageError, NewsArticle, article_headline, fetch_news_sentiment
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
    m3_origin: str | None = None
    m3_reason: str = ""
    m3_summary: str = ""
    fallback_note: str = ""
    regime_scores: dict[MarketRegime, float] = field(default_factory=dict)


def select_top_headlines_for_regime(
    articles: Sequence[NewsArticle],
    regime: MarketRegime,
    *,
    limit: int = 5,
) -> list[str]:
    """Pick representative headlines based on detected regime."""
    if limit <= 0:
        return []

    scored = [
        (
            float(a.overall_sentiment_score)
            if a.overall_sentiment_score is not None
            else 0.0,
            article_headline(a),
        )
        for a in articles
    ]
    if not scored:
        return []

    if regime is MarketRegime.BULLISH:
        ranked = sorted(scored, key=lambda x: x[0], reverse=True)
    elif regime is MarketRegime.BEARISH:
        ranked = sorted(scored, key=lambda x: x[0])
    else:
        ranked = sorted(scored, key=lambda x: abs(x[0]))

    return [title for _score, title in ranked[:limit]]


def analyze_market_sentiment(
    *,
    tickers: str,
    candidate_strategies: Sequence[CandidateStrategy],
    m3_selected: CandidateStrategy | None = None,
    m3_context: Mapping[str, Any] | None = None,
    classifier: SentimentRegimeClassifier | None = None,
    news_limit: int = 50,
    api_key: str | None = None,
    fit_classifier_from_feed: bool = True,
    fallback_on_fetch_error: bool = True,
) -> SentimentAnalysisResult:
    """
    Fetch news for ``tickers``, classify regime, recommend a strategy from the pool.

    By default, this attempts to fit ``SentimentRegimeClassifier`` on the
    returned articles when possible, so logistic regression is used whenever
    sufficient labeled data are available. If fitting is skipped or impossible,
    prediction falls back to a passed ``classifier`` (if already fitted) or the
    heuristic path.

    If news fetch fails and ``fallback_on_fetch_error`` is True, this returns a
    deterministic neutral fallback instead of raising.
    """
    fallback_note = ""
    try:
        news = fetch_news_sentiment(
            tickers=tickers,
            limit=news_limit,
            api_key=api_key,
        )
        articles = list(news.feed)
    except AlphaVantageError as exc:
        if not fallback_on_fetch_error:
            raise
        articles = []
        fallback_note = f"News fetch failed; used neutral fallback: {exc}"

    clf = classifier or SentimentRegimeClassifier()
    if fit_classifier_from_feed:
        clf.fit_from_articles(articles)

    regime, regime_scores, method = clf.predict_regime_with_scores(articles)
    confidence = float(regime_scores.get(regime, 0.0))
    top_headlines = select_top_headlines_for_regime(articles, regime, limit=5)
    strat, reason = recommend_strategy_for_regime(
        regime,
        candidate_strategies,
        m3_selected=m3_selected,
    )

    m3_origin = str(m3_context.get("origin")) if m3_context and m3_context.get("origin") is not None else None
    m3_reason = str(m3_context.get("reason")) if m3_context and m3_context.get("reason") is not None else ""
    m3_summary = str(m3_context.get("summary")) if m3_context and m3_context.get("summary") is not None else ""

    if fallback_note:
        reason = f"{reason} {fallback_note}".strip()

    return SentimentAnalysisResult(
        regime=regime,
        confidence=confidence,
        classification_method=method,
        top_headlines=top_headlines,
        articles=tuple(articles),
        recommended_strategy=strat,
        recommendation_reason=reason,
        m3_origin=m3_origin,
        m3_reason=m3_reason,
        m3_summary=m3_summary,
        fallback_note=fallback_note,
        regime_scores=regime_scores,
    )
