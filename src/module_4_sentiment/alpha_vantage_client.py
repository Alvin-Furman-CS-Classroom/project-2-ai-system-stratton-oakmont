"""Alpha Vantage Market News & Sentiment API (NEWS_SENTIMENT).

Docs: https://www.alphavantage.co/documentation/#news-sentiment

Set ``ALPHA_VANTAGE_API_KEY`` in the environment, or pass ``api_key`` explicitly.
Optional: a ``.env`` file in the project root (loaded on first fetch if ``python-dotenv`` is installed).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

ALPHA_VANTAGE_QUERY_URL = "https://www.alphavantage.co/query"


class AlphaVantageError(RuntimeError):
    """Raised when the API returns an error, rate-limit notice, or invalid payload."""


def _load_dotenv_if_present() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _get_api_key(explicit: str | None) -> str:
    if explicit:
        return explicit.strip()
    _load_dotenv_if_present()
    key = os.environ.get("ALPHA_VANTAGE_API_KEY", "").strip()
    if not key:
        raise AlphaVantageError(
            "Missing API key: set environment variable ALPHA_VANTAGE_API_KEY "
            "or pass api_key= to fetch_news_sentiment."
        )
    return key


def _check_api_payload(data: dict[str, Any]) -> None:
    if "Error Message" in data:
        raise AlphaVantageError(str(data["Error Message"]))
    if "Note" in data:
        raise AlphaVantageError(f"Alpha Vantage rate limit or note: {data['Note']}")
    if "Information" in data:
        raise AlphaVantageError(str(data["Information"]))


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class NewsArticle:
    """One item from the NEWS_SENTIMENT ``feed``."""

    title: str
    url: str
    time_published: str
    summary: str
    overall_sentiment_score: float | None
    overall_sentiment_label: str | None
    raw: dict[str, Any] = field(repr=False)


@dataclass(frozen=True)
class NewsSentimentResult:
    """Parsed NEWS_SENTIMENT response."""

    feed: list[NewsArticle]
    raw: dict[str, Any]


def _parse_feed(data: dict[str, Any]) -> list[NewsArticle]:
    feed = data.get("feed")
    if not isinstance(feed, list):
        return []
    out: list[NewsArticle] = []
    for item in feed:
        if not isinstance(item, dict):
            continue
        out.append(
            NewsArticle(
                title=str(item.get("title", "")),
                url=str(item.get("url", "")),
                time_published=str(item.get("time_published", "")),
                summary=str(item.get("summary", "")),
                overall_sentiment_score=_parse_float(item.get("overall_sentiment_score")),
                overall_sentiment_label=(
                    str(item["overall_sentiment_label"])
                    if item.get("overall_sentiment_label") is not None
                    else None
                ),
                raw=item,
            )
        )
    return out


def fetch_news_sentiment(
    *,
    tickers: str | None = None,
    topics: str | None = None,
    time_from: str | None = None,
    time_to: str | None = None,
    sort: str | None = "LATEST",
    limit: int | None = 50,
    api_key: str | None = None,
    timeout_s: float = 30.0,
) -> NewsSentimentResult:
    """
    Call Alpha Vantage ``NEWS_SENTIMENT`` and return parsed articles.

    Args:
        tickers: Comma-separated tickers (e.g. ``\"IBM\"`` or ``\"AAPL,MSFT\"``).
        topics: Optional topic filter (see Alpha Vantage docs for allowed values).
        time_from: Optional start ``YYYYMMDDTHHMM`` (UTC).
        time_to: Optional end ``YYYYMMDDTHHMM`` (UTC).
        sort: ``LATEST``, ``EARLIEST``, or ``RELEVANCE``.
        limit: Max number of articles (API default is typically 50).
        api_key: Override ``ALPHA_VANTAGE_API_KEY``.
        timeout_s: HTTP timeout in seconds.

    Returns:
        NewsSentimentResult with ``feed`` entries and the full JSON ``raw`` payload.
    """
    params: dict[str, str] = {"function": "NEWS_SENTIMENT"}
    if tickers:
        params["tickers"] = tickers
    if topics:
        params["topics"] = topics
    if time_from:
        params["time_from"] = time_from
    if time_to:
        params["time_to"] = time_to
    if sort:
        params["sort"] = sort
    if limit is not None:
        params["limit"] = str(limit)
    params["apikey"] = _get_api_key(api_key)

    url = f"{ALPHA_VANTAGE_QUERY_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "module_4_sentiment/1.0"})

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raise AlphaVantageError(f"HTTP {e.code}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise AlphaVantageError(f"Request failed: {e.reason}") from e

    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        raise AlphaVantageError(f"Invalid JSON from Alpha Vantage: {e}") from e

    if not isinstance(data, dict):
        raise AlphaVantageError("Unexpected response: top-level JSON is not an object.")

    _check_api_payload(data)
    return NewsSentimentResult(feed=_parse_feed(data), raw=data)
