# Module 4: Market Sentiment Classifier (Supervised Learning)

from .alpha_vantage_client import (
    AlphaVantageError,
    NewsArticle,
    NewsSentimentResult,
    fetch_news_sentiment,
)
from .pipeline import SentimentAnalysisResult, analyze_market_sentiment
from .regime_classifier import MarketRegime, SentimentRegimeClassifier
from .strategy_recommendation import recommend_strategy_for_regime

__all__ = [
    "AlphaVantageError",
    "NewsArticle",
    "NewsSentimentResult",
    "fetch_news_sentiment",
    "SentimentAnalysisResult",
    "analyze_market_sentiment",
    "MarketRegime",
    "SentimentRegimeClassifier",
    "recommend_strategy_for_regime",
]
