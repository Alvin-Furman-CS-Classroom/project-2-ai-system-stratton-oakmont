"""Tests for Module 4 features and regime classifier."""

import numpy as np

from src.module_4_sentiment.alpha_vantage_client import NewsArticle
from src.module_4_sentiment.features import (
    article_feature_vector,
    build_training_arrays,
    label_to_regime_index,
)
from src.module_4_sentiment.regime_classifier import (
    MarketRegime,
    SentimentRegimeClassifier,
    heuristic_regime,
)


def _article(score: float, label: str) -> NewsArticle:
    return NewsArticle(
        title="t",
        url="https://x",
        time_published="20240101T000000",
        summary="summary text",
        overall_sentiment_score=score,
        overall_sentiment_label=label,
        raw={},
    )


def test_label_to_regime_index():
    assert label_to_regime_index("Bullish") == 2
    assert label_to_regime_index("Somewhat-Bearish") == 0
    assert label_to_regime_index("Neutral") == 1
    assert label_to_regime_index(None) is None


def test_build_training_arrays_filters_unknown():
    xs, ys = build_training_arrays([_article(0.1, "Bullish"), _article(0.0, "Unknown")])
    assert xs.shape[0] == 1
    assert ys[0] == 2


def test_heuristic_bullish():
    articles = [_article(0.5, "Bullish")] * 3
    r, c = heuristic_regime(articles)
    assert r is MarketRegime.BULLISH
    assert c > 0


def test_classifier_fits_and_predicts():
    articles = (
        [_article(-0.4 + 0.01 * i, "Bearish") for i in range(8)]
        + [_article(0.0, "Neutral") for _ in range(8)]
        + [_article(0.4, "Bullish") for _ in range(8)]
    )
    clf = SentimentRegimeClassifier()
    assert clf.fit_from_articles(articles) is True
    regime, conf, method = clf.predict_regime([_article(0.45, "Bullish")])
    assert method == "logistic_regression"
    assert regime is MarketRegime.BULLISH
    assert 0.0 <= conf <= 1.0


def test_article_feature_vector_shape():
    v = article_feature_vector(_article(0.2, "Neutral"))
    assert v.shape == (3,)
    assert not np.isnan(v).any()
