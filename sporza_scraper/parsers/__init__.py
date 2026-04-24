"""HTML parsers for the Sporza scraper.

Each parser is a pure function of raw HTML: no I/O, no network, no logging
side effects beyond WARNING lines when optional fields are missing. This
keeps them trivially unit-testable against recorded fixtures.
"""

from .football_results import parse_football_results, discover_football_result_links
from .live_match import parse_live_match
from .news import discover_news_links, parse_news_article

__all__ = [
    "parse_news_article",
    "discover_news_links",
    "parse_football_results",
    "discover_football_result_links",
    "parse_live_match",
]
