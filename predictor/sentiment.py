"""
News sentiment analysis using VADER + free RSS feeds.

Sources used (no API key required):
  - Moneycontrol RSS
  - Economic Times Markets RSS
  - LiveMint RSS
  - Business Standard RSS

Sentiment score contributed to final stock score: [-30, +30].
"""

import logging
import re
import time
from typing import Dict, List, Optional

import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RSS feeds — all freely accessible, no auth needed
# ---------------------------------------------------------------------------
RSS_FEEDS = [
    "https://www.moneycontrol.com/rss/business.xml",
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://www.livemint.com/rss/markets",
    "https://www.business-standard.com/rss/markets-106.rss",
    "https://feeds.feedburner.com/ndtvprofit-latest",
]

_analyzer = SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Strip HTML tags and extra whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fetch_headlines(feed_url: str, timeout: int = 8) -> List[str]:
    """
    Parse an RSS feed and return a list of title+summary strings.
    Returns empty list on any error.
    """
    try:
        feed = feedparser.parse(feed_url, request_headers={"User-Agent": "Mozilla/5.0"})
        headlines = []
        for entry in feed.entries:
            title = _clean_text(getattr(entry, "title", ""))
            summary = _clean_text(getattr(entry, "summary", ""))
            if title:
                headlines.append(f"{title}. {summary}"[:500])
        return headlines
    except Exception as exc:
        logger.debug("Feed error (%s): %s", feed_url, exc)
        return []


def fetch_all_headlines(max_per_feed: int = 20) -> List[str]:
    """Aggregate headlines from all configured RSS feeds."""
    all_headlines: List[str] = []
    for feed_url in RSS_FEEDS:
        headlines = _fetch_headlines(feed_url)
        all_headlines.extend(headlines[:max_per_feed])
        time.sleep(0.3)  # polite crawl delay

    logger.info("Fetched %d total headlines from %d feeds", len(all_headlines), len(RSS_FEEDS))
    return all_headlines


# ---------------------------------------------------------------------------
# Keyword matching
# ---------------------------------------------------------------------------

def _build_keywords(symbol: str, company_name: str) -> List[str]:
    """
    Build search keywords for a stock from its symbol and company name.
    e.g.  RELIANCE.NS / Reliance Industries → ["Reliance", "RELIANCE"]
    """
    ticker = symbol.replace(".NS", "").replace("-", " ")
    keywords = [ticker, ticker.lower(), ticker.upper()]

    # First word of company name (e.g. "Tata" from "Tata Consultancy Services")
    words = company_name.split()
    if words:
        keywords.append(words[0])
        if len(words) > 1:
            keywords.append(" ".join(words[:2]))

    return list(set(keywords))


def _score_headlines(headlines: List[str], keywords: List[str]) -> Optional[float]:
    """
    Score headlines relevant to a stock using VADER.

    Returns:
        Mean compound sentiment score in [-1, +1] across matched headlines,
        or None if no relevant headlines found.
    """
    relevant_scores = []
    for headline in headlines:
        if any(kw.lower() in headline.lower() for kw in keywords):
            vs = _analyzer.polarity_scores(headline)
            relevant_scores.append(vs["compound"])

    if not relevant_scores:
        return None

    return round(sum(relevant_scores) / len(relevant_scores), 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_sentiment_scores(
    symbols: List[str],
    symbol_names: Dict[str, str],
    headlines: Optional[List[str]] = None,
) -> Dict[str, dict]:
    """
    Compute sentiment scores for all symbols.

    Args:
        symbols:       List of NSE tickers, e.g. ["RELIANCE.NS", ...]
        symbol_names:  Mapping from ticker → company name
        headlines:     Pre-fetched headlines (fetched fresh if None)

    Returns:
        dict mapping symbol → {
            "raw":   float | None,   # VADER compound [-1, +1]
            "score": float,          # contribution to final score [-30, +30]
            "count": int,            # number of matching headlines
        }
    """
    if headlines is None:
        headlines = fetch_all_headlines()

    results: Dict[str, dict] = {}
    for symbol in symbols:
        company = symbol_names.get(symbol, symbol.replace(".NS", ""))
        keywords = _build_keywords(symbol, company)

        # Count matching headlines
        count = sum(
            1 for h in headlines
            if any(kw.lower() in h.lower() for kw in keywords)
        )

        raw = _score_headlines(headlines, keywords)
        if raw is None:
            # No news found → neutral, slight negative bias (unknown = caution)
            score = -2.0
        else:
            score = raw * 30.0  # scale to [-30, +30]

        results[symbol] = {
            "raw": raw,
            "score": round(score, 2),
            "count": count,
        }

    return results
