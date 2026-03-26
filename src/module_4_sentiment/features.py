"""Feature vectors from Alpha Vantage news articles (Module 4)."""

from __future__ import annotations

import numpy as np

from .alpha_vantage_client import NewsArticle


def label_to_regime_index(label: str | None) -> int | None:
    """
    Map Alpha Vantage ``overall_sentiment_label`` to a class index for training.

    Classes: 0 = bearish, 1 = neutral, 2 = bullish.
    Handles variants like ``Somewhat-Bullish`` / ``Somewhat-Bearish``.
    """
    if not label:
        return None
    s = label.lower()
    if "bearish" in s:
        return 0
    if "bullish" in s:
        return 2
    if "neutral" in s:
        return 1
    return None


def article_feature_vector(article: NewsArticle) -> np.ndarray:
    """Numeric features for one article (used by logistic regression)."""
    score = article.overall_sentiment_score if article.overall_sentiment_score is not None else 0.0
    return np.array(
        [score, np.log1p(len(article.summary)), np.log1p(len(article.title))],
        dtype=np.float64,
    )


def build_training_arrays(
    articles: list[NewsArticle],
) -> tuple[np.ndarray, np.ndarray]:
    """Rows with known labels (from API) for fitting the classifier."""
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for a in articles:
        idx = label_to_regime_index(a.overall_sentiment_label)
        if idx is None:
            continue
        xs.append(article_feature_vector(a))
        ys.append(idx)
    if not xs:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    return np.stack(xs, axis=0), np.array(ys, dtype=np.int64)


def aggregate_sentiment_score(articles: list[NewsArticle]) -> float:
    """Mean overall sentiment score (0.0 if none available)."""
    scores = [a.overall_sentiment_score for a in articles if a.overall_sentiment_score is not None]
    if not scores:
        return 0.0
    return float(np.mean(scores))
