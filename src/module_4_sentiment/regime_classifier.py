"""Market regime classification: logistic regression + score-based fallback."""

from __future__ import annotations

from enum import Enum

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .alpha_vantage_client import NewsArticle
from .features import aggregate_sentiment_score, article_feature_vector, build_training_arrays


class MarketRegime(str, Enum):
    """Coarse market sentiment regime for downstream modules."""

    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    BULLISH = "BULLISH"


_REGIME_TO_INDEX = {
    MarketRegime.BEARISH: 0,
    MarketRegime.NEUTRAL: 1,
    MarketRegime.BULLISH: 2,
}
_INDEX_TO_REGIME = {v: k for k, v in _REGIME_TO_INDEX.items()}

# Heuristic confidence/threshold constants.
HEURISTIC_REGIME_THRESHOLD = 0.12
HEURISTIC_CONFIDENCE_SCALE = 0.45
HEURISTIC_NEUTRAL_CONFIDENCE_CAP = 0.6


def _index_to_regime(i: int) -> MarketRegime:
    return _INDEX_TO_REGIME[int(i)]


def heuristic_regime(articles: list[NewsArticle]) -> tuple[MarketRegime, float]:
    """
    Rule-based regime when the model is unavailable or data is sparse.

    Uses mean sentiment score with symmetric thresholds (Alpha Vantage scores
    are typically in roughly [-1, 1]).
    """
    mean_score = aggregate_sentiment_score(articles)
    if not articles:
        return MarketRegime.NEUTRAL, 0.0

    thresh = HEURISTIC_REGIME_THRESHOLD
    if mean_score > thresh:
        conf = min(1.0, abs(mean_score) / HEURISTIC_CONFIDENCE_SCALE)
        return MarketRegime.BULLISH, float(conf)
    if mean_score < -thresh:
        conf = min(1.0, abs(mean_score) / HEURISTIC_CONFIDENCE_SCALE)
        return MarketRegime.BEARISH, float(conf)
    conf = max(0.0, 1.0 - abs(mean_score) / thresh) * HEURISTIC_NEUTRAL_CONFIDENCE_CAP
    return MarketRegime.NEUTRAL, float(conf)


class SentimentRegimeClassifier:
    """
    Multinomial logistic regression over per-article features.

    Trained on API-provided sentiment labels. At inference, class probabilities
    are averaged across articles in the current feed, then the argmax regime is
    returned with that averaged max probability as confidence.
    """

    def __init__(self) -> None:
        self._pipe: Pipeline | None = None

    @property
    def is_fitted(self) -> bool:
        return self._pipe is not None

    def fit_from_articles(self, articles: list[NewsArticle]) -> bool:
        """
        Fit on articles whose labels could be parsed. Returns False if fitting
        was skipped (too little data or sklearn error).
        """
        X, y = build_training_arrays(articles)
        if X.shape[0] < 6:
            self._pipe = None
            return False
        if len(np.unique(y)) < 2:
            self._pipe = None
            return False

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        try:
            pipe.fit(X, y)
        except ValueError:
            self._pipe = None
            return False

        self._pipe = pipe
        return True

    def predict_regime(self, articles: list[NewsArticle]) -> tuple[MarketRegime, float, str]:
        """
        Returns ``(regime, confidence, method)`` where method is
        ``\"logistic_regression\"`` or ``\"heuristic\"``.
        """
        if not articles:
            return MarketRegime.NEUTRAL, 0.0, "heuristic"

        if self._pipe is None:
            r, c = heuristic_regime(articles)
            return r, c, "heuristic"

        X = np.stack([article_feature_vector(a) for a in articles], axis=0)
        probs = self._pipe.predict_proba(X)
        mean_p = np.mean(probs, axis=0)
        classes = self._pipe.named_steps["clf"].classes_
        idx = int(np.argmax(mean_p))
        regime = _index_to_regime(int(classes[idx]))
        confidence = float(mean_p[idx])
        return regime, confidence, "logistic_regression"
