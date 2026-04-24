"""Parser for Sporza football results / fixtures pages."""
from __future__ import annotations

import logging
import re
from typing import Iterable

from bs4 import BeautifulSoup, Tag

from ..models import FootballMatch, MatchStatus
from ..utils import (
    clean_text,
    dedupe_preserve_order,
    normalise_url,
    parse_any_datetime,
    to_iso_string,
)

log = logging.getLogger(__name__)

_SCORE_RE = re.compile(r"(?P<h>\d{1,3})\s*[-–:]\s*(?P<a>\d{1,3})")
_HT_RE = re.compile(
    r"(rust|halftime|ht)[^\d]*(?P<h>\d{1,3})\s*[-–:]\s*(?P<a>\d{1,3})",
    re.IGNORECASE,
)

_STATUS_KEYWORDS: dict[MatchStatus, tuple[str, ...]] = {
    "live": ("live", "bezig", "aan de gang", "1e helft", "2e helft", "eerste helft",
             "tweede helft"),
    "finished": ("ft", "afgelopen", "gespeeld", "einde", "beëindigd"),
    "postponed": ("uitgesteld", "afgelast"),
    "scheduled": ("wedstrijd", "aftrap", "begint", "start om"),
}


def discover_football_result_links(html: str) -> list[str]:
    """Return likely match-detail URLs from a results index/listing."""
    soup = BeautifulSoup(html, "lxml")
    urls: list[str] = []
    for a in soup.select("a[href*='/matches/'], a[href*='/wedstrijd']"):
        href = a.get("href")
        if isinstance(href, str):
            normalised = normalise_url(href)
            if normalised:
                urls.append(normalised)
    return dedupe_preserve_order(urls)


def parse_football_results(html: str, competition_hint: str | None = None) -> list[FootballMatch]:
    """Parse a football results / fixtures page into a list of matches.

    Two strategies are attempted in order:

    1. Semantic: elements with ``data-testid`` or ``itemprop`` attributes
       identifying match rows.
    2. Heuristic: any block that contains two team names and a score pattern.

    Unparseable rows are skipped with a WARNING log, never raising.
    """
    soup = BeautifulSoup(html, "lxml")
    matches: list[FootballMatch] = []

    rows = list(_find_match_rows(soup))
    if not rows:
        log.warning("parse_football_results: no match rows detected")
        return matches

    for row in rows:
        try:
            match = _parse_match_row(row, competition_hint)
            if match:
                matches.append(match)
        except Exception as exc:  # keep the pipeline alive on malformed rows
            log.warning("parse_football_results: skipping malformed row (%s)", exc)
            continue
    return matches


# ---------------------------------------------------------------------------


def _find_match_rows(soup: BeautifulSoup) -> Iterable[Tag]:
    selectors = (
        "[data-testid*='match']",
        "[data-testid*='fixture']",
        "article[itemtype*='SportsEvent']",
        "li.match, div.match, tr.match",
    )
    for selector in selectors:
        found = soup.select(selector)
        if found:
            yield from (n for n in found if isinstance(n, Tag))
            return
    # Last resort: any list item containing a score-like token.
    for node in soup.find_all(["li", "tr", "article", "div"]):
        if not isinstance(node, Tag):
            continue
        if _SCORE_RE.search(node.get_text(" ", strip=True) or ""):
            yield node


def _parse_match_row(row: Tag, competition_hint: str | None) -> FootballMatch | None:
    text = row.get_text(" ", strip=True) or ""

    home_team, away_team = _extract_team_names(row)
    if not home_team and not away_team:
        return None

    score_match = _SCORE_RE.search(text)
    home_score = int(score_match.group("h")) if score_match else None
    away_score = int(score_match.group("a")) if score_match else None

    ht_match = _HT_RE.search(text)
    ht_home = int(ht_match.group("h")) if ht_match else None
    ht_away = int(ht_match.group("a")) if ht_match else None

    status = _extract_status(text)
    kickoff = _extract_kickoff(row)

    venue = _extract_venue(row)
    matchday = _extract_matchday(row)
    competition = _extract_competition(row) or competition_hint
    match_url = _extract_match_url(row)

    return FootballMatch(
        match_url=match_url,
        competition=competition,
        matchday=matchday,
        kickoff_at=to_iso_string(kickoff),
        home_team=home_team,
        away_team=away_team,
        home_score=home_score,
        away_score=away_score,
        halftime_home_score=ht_home,
        halftime_away_score=ht_away,
        venue=venue,
        status=status,
    )


def _extract_team_names(row: Tag) -> tuple[str | None, str | None]:
    # 1) Try dedicated attributes.
    home_node = row.select_one(
        "[data-testid*='home'], [itemprop='homeTeam'], .team-home, .home-team"
    )
    away_node = row.select_one(
        "[data-testid*='away'], [itemprop='awayTeam'], .team-away, .away-team"
    )
    home = clean_text(home_node.get_text(" ", strip=True)) if isinstance(home_node, Tag) else None
    away = clean_text(away_node.get_text(" ", strip=True)) if isinstance(away_node, Tag) else None
    if home or away:
        return home, away

    # 2) Fallback: two team-like nodes.
    team_nodes = row.select(".team, [class*='team']")
    names = [
        clean_text(n.get_text(" ", strip=True))
        for n in team_nodes
        if isinstance(n, Tag)
    ]
    names = [n for n in names if n]
    if len(names) >= 2:
        return names[0], names[1]
    return None, None


def _extract_status(text: str) -> MatchStatus:
    lowered = text.lower()
    for status, keywords in _STATUS_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return status
    if _SCORE_RE.search(text):
        return "finished"
    return "unknown"


def _extract_kickoff(row: Tag) -> object | None:
    time_tag = row.find("time")
    if isinstance(time_tag, Tag):
        attr = time_tag.get("datetime") or time_tag.get_text(" ", strip=True)
        if isinstance(attr, str):
            parsed = parse_any_datetime(attr)
            if parsed:
                return parsed
    meta = row.find("meta", attrs={"itemprop": "startDate"})
    if isinstance(meta, Tag):
        content = meta.get("content")
        if isinstance(content, str):
            return parse_any_datetime(content)
    return None


def _extract_venue(row: Tag) -> str | None:
    node = row.select_one("[itemprop='location'], .venue, [data-testid*='venue']")
    if isinstance(node, Tag):
        return clean_text(node.get_text(" ", strip=True))
    return None


def _extract_matchday(row: Tag) -> str | None:
    node = row.select_one(".matchday, [data-testid*='matchday'], [data-testid*='round']")
    if isinstance(node, Tag):
        return clean_text(node.get_text(" ", strip=True))
    return None


def _extract_competition(row: Tag) -> str | None:
    node = row.select_one(".competition, [data-testid*='competition']")
    if isinstance(node, Tag):
        return clean_text(node.get_text(" ", strip=True))
    # Walk up to find a section heading.
    for parent in row.parents:
        if isinstance(parent, Tag):
            heading = parent.find(["h2", "h3", "h4"])
            if isinstance(heading, Tag):
                return clean_text(heading.get_text(" ", strip=True))
    return None


def _extract_match_url(row: Tag) -> str | None:
    a = row.find("a", href=True)
    if isinstance(a, Tag):
        href = a.get("href")
        if isinstance(href, str):
            return normalise_url(href)
    return None
