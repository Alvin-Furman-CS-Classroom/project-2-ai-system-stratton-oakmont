"""Unit tests for Module 4: Alpha Vantage news sentiment client."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.module_4_sentiment.alpha_vantage_client import (
    AlphaVantageError,
    fetch_news_sentiment,
)


def _fake_response(payload: dict) -> MagicMock:
    data = json.dumps(payload).encode("utf-8")
    mock = MagicMock()
    mock.read = MagicMock(return_value=data)
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    return mock


@patch("src.module_4_sentiment.alpha_vantage_client.urllib.request.urlopen")
def test_fetch_news_sentiment_parses_feed(mock_urlopen):
    payload = {
        "feed": [
            {
                "title": "Test headline",
                "url": "https://example.com/a",
                "time_published": "20240101T120000",
                "summary": "Summary text",
                "overall_sentiment_score": "0.15",
                "overall_sentiment_label": "Neutral",
            }
        ]
    }
    mock_urlopen.return_value = _fake_response(payload)

    result = fetch_news_sentiment(tickers="IBM", api_key="test-key", limit=10)

    assert len(result.feed) == 1
    article = result.feed[0]
    assert article.title == "Test headline"
    assert article.url == "https://example.com/a"
    assert article.overall_sentiment_score == pytest.approx(0.15)
    assert article.overall_sentiment_label == "Neutral"
    assert result.raw == payload

    called_url = mock_urlopen.call_args[0][0].full_url
    assert "function=NEWS_SENTIMENT" in called_url
    assert "tickers=IBM" in called_url
    assert "apikey=test-key" in called_url


@patch("src.module_4_sentiment.alpha_vantage_client.urllib.request.urlopen")
def test_fetch_raises_on_error_message(mock_urlopen):
    mock_urlopen.return_value = _fake_response({"Error Message": "invalid api call"})
    with pytest.raises(AlphaVantageError, match="invalid api call"):
        fetch_news_sentiment(api_key="x")


@patch("src.module_4_sentiment.alpha_vantage_client.urllib.request.urlopen")
def test_fetch_raises_on_note_rate_limit(mock_urlopen):
    mock_urlopen.return_value = _fake_response({"Note": "Thank you for using Alpha Vantage"})
    with pytest.raises(AlphaVantageError, match="rate limit"):
        fetch_news_sentiment(api_key="x")


@patch("src.module_4_sentiment.alpha_vantage_client._load_dotenv_if_present")
def test_missing_api_key(_mock_dotenv):
    """Avoid loading a real .env file (which would supply a key and hit the network)."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(AlphaVantageError, match="ALPHA_VANTAGE_API_KEY"):
            fetch_news_sentiment()
