"""Parser unit tests with recorded HTML fixtures (no network)."""
from __future__ import annotations

from pathlib import Path

import pytest

from sporza_scraper.parsers import (
    discover_news_links,
    parse_football_results,
    parse_live_match,
    parse_news_article,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------


def test_news_article_happy_path() -> None:
    article = parse_news_article(_read("news_article_happy.html"))
    assert article is not None
    assert article.title == "Voorbeeldtitel van het artikel"
    assert article.lead == "Korte lead van het artikel."
    assert "Jan Janssens" in article.authors
    assert "Els Peeters" in article.authors
    assert article.category == "Voetbal"
    assert "Jupiler Pro League" in article.tags
    assert "Anderlecht" in article.tags
    assert article.published_at and article.published_at.startswith("2026-04-15")
    assert article.updated_at and article.updated_at.startswith("2026-04-15")
    assert any("Eerste paragraaf" in p for p in article.body_paragraphs)
    assert any(img.url.endswith("inline.jpg") for img in article.images)
    # tracking param stripped
    assert not any("utm_source" in img.url for img in article.images)
    assert any("ander-artikel-987654" in link for link in article.related_links)
    assert article.slug  # slug derived from URL


def test_news_article_missing_fields_does_not_raise() -> None:
    article = parse_news_article(_read("news_article_missing_fields.html"))
    assert article is not None
    assert article.title == "Alleen een titel"
    assert article.published_at is None
    assert article.authors == []
    assert article.body_paragraphs  # at least the single paragraph


def test_news_article_malformed_date_returns_none_field() -> None:
    html = (
        "<html><head>"
        "<meta property='article:published_time' content='not-a-date'>"
        "</head><body><article><h1>T</h1><p>x</p></article></body></html>"
    )
    article = parse_news_article(html)
    assert article is not None
    assert article.published_at is None


def test_discover_news_links_filters_and_dedupes() -> None:
    links = discover_news_links(_read("news_index.html"))
    assert "https://sporza.be/nl/voetbal/artikel-een-111111/" in links
    assert "https://sporza.be/nl/wielrennen/artikel-twee-222222/" in links
    assert len(links) == len(set(links))  # no duplicates
    assert all("example.com" not in l for l in links)
    assert all("/video/" not in l for l in links)


def test_discover_news_links_empty_page() -> None:
    links = discover_news_links("<html><body></body></html>")
    assert links == []


# ---------------------------------------------------------------------------
# Football results
# ---------------------------------------------------------------------------


def test_football_results_happy_path() -> None:
    matches = parse_football_results(_read("football_results.html"))
    assert len(matches) == 2
    first = matches[0]
    assert first.home_team == "Club Brugge"
    assert first.away_team == "Anderlecht"
    assert first.home_score == 2
    assert first.away_score == 1
    assert first.halftime_home_score == 1
    assert first.halftime_away_score == 0
    assert first.status == "finished"
    assert first.venue == "Jan Breydel"
    assert first.kickoff_at and first.kickoff_at.startswith("2026-04-14")
    assert first.competition == "Jupiler Pro League"

    second = matches[1]
    assert second.status == "postponed"
    assert second.home_score is None


def test_football_results_empty_page() -> None:
    matches = parse_football_results("<html><body><p>geen wedstrijden</p></body></html>")
    assert matches == []


# ---------------------------------------------------------------------------
# Live match
# ---------------------------------------------------------------------------


def test_live_match_live_state() -> None:
    live = parse_live_match(_read("live_match_live.html"), url="https://sporza.be/nl/matches/brugge-anderlecht/")
    assert live is not None
    assert live.home_team == "Club Brugge"
    assert live.away_team == "Anderlecht"
    assert live.home_score == 1
    assert live.away_score == 1
    assert live.current_minute == 62
    assert live.status == "live"
    assert live.is_live is True
    assert live.polled_at  # UTC timestamp set
    assert len(live.events) == 2
    assert live.events[0].event_type == "goal"
    assert live.events[1].event_type == "yellow_card"
    assert any(l.team and "Club Brugge" in l.team for l in live.lineups)


def test_live_match_finished_state() -> None:
    live = parse_live_match(_read("live_match_finished.html"), url="https://sporza.be/nl/matches/gent-antwerp/")
    assert live is not None
    assert live.status == "finished"
    assert live.is_live is False
    assert live.home_score == 0
    assert live.away_score == 2
    assert live.events == []


def test_live_match_unrecognisable_page_returns_none() -> None:
    result = parse_live_match("<html><body>just a 404</body></html>")
    assert result is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
