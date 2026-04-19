"""Parser for Sporza live match pages."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable

from bs4 import BeautifulSoup, Tag

from ..models import EventType, LiveMatch, Lineup, MatchEvent, MatchStatus
from ..utils import (
    clean_text,
    dedupe_preserve_order,
    normalise_url,
    stable_hash,
    utcnow_iso,
)

log = logging.getLogger(__name__)

_SCORE_RE = re.compile(r"(?P<h>\d{1,3})\s*[-–:]\s*(?P<a>\d{1,3})")
_MINUTE_RE = re.compile(r"(?P<m>\d{1,3})['’`]?")
_EVENT_KEYWORDS: dict[EventType, tuple[str, ...]] = {
    "goal": ("goal", "doelpunt"),
    "own_goal": ("owngoal", "own goal", "eigen doel"),
    "penalty": ("penalty", "strafschop"),
    "penalty_missed": ("penalty gemist", "strafschop gemist"),
    "yellow_card": ("yellow", "gele kaart"),
    "red_card": ("red", "rode kaart"),
    "substitution": ("substitution", "wissel", "vervanging"),
}


def parse_live_match(html: str, url: str | None = None) -> LiveMatch | None:
    """Parse a live match page into a ``LiveMatch`` snapshot.

    Tries an embedded JSON blob first (``__NEXT_DATA__``) and falls back to
    heuristic HTML parsing. Always returns a ``LiveMatch`` with
    ``polled_at`` set, or ``None`` if the page is not recognisable.
    """
    soup = BeautifulSoup(html, "lxml")
    polled_at = utcnow_iso()

    blob = _extract_next_data(soup)
    if blob:
        result = _parse_from_json(blob, url=url, polled_at=polled_at)
        if result:
            return result
        log.warning(
            "parse_live_match: __NEXT_DATA__ present but unrecognised; "
            "falling back to HTML parsing"
        )

    return _parse_from_html(soup, url=url, polled_at=polled_at)


# ---------------------------------------------------------------------------
# JSON path (preferred)
# ---------------------------------------------------------------------------


def _extract_next_data(soup: BeautifulSoup) -> dict[str, Any] | None:
    script = soup.find("script", attrs={"id": "__NEXT_DATA__"})
    if not isinstance(script, Tag):
        return None
    raw = script.string or ""
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        log.warning("parse_live_match: could not parse __NEXT_DATA__ JSON")
        return None


def _parse_from_json(
    blob: dict[str, Any], *, url: str | None, polled_at: str
) -> LiveMatch | None:
    match_obj = _find_match_object(blob)
    if not match_obj:
        return None

    match_id = _coerce_str(match_obj.get("id") or match_obj.get("matchId"))
    if not match_id:
        match_id = stable_hash(url or json.dumps(match_obj, sort_keys=True)[:200])

    home = match_obj.get("homeTeam") or {}
    away = match_obj.get("awayTeam") or {}

    home_team = _coerce_str(
        (home.get("name") if isinstance(home, dict) else None) or match_obj.get("home")
    )
    away_team = _coerce_str(
        (away.get("name") if isinstance(away, dict) else None) or match_obj.get("away")
    )

    score = match_obj.get("score") or {}
    home_score = _coerce_int((score.get("home") if isinstance(score, dict) else None)
                             or match_obj.get("homeScore"))
    away_score = _coerce_int((score.get("away") if isinstance(score, dict) else None)
                             or match_obj.get("awayScore"))

    minute = _coerce_int(match_obj.get("minute") or match_obj.get("currentMinute"))
    status_raw = _coerce_str(match_obj.get("status") or match_obj.get("state")) or ""
    status, is_live = _normalise_status(status_raw)
    competition = _coerce_str(
        match_obj.get("competition") or match_obj.get("tournament")
    )

    events = _parse_events_json(match_obj.get("events") or match_obj.get("timeline"))
    lineups = _parse_lineups_json(match_obj.get("lineups"))

    return LiveMatch(
        match_id=match_id,
        match_url=normalise_url(url) if url else None,
        competition=competition,
        home_team=home_team,
        away_team=away_team,
        home_score=home_score,
        away_score=away_score,
        current_minute=minute,
        status=status,
        is_live=is_live,
        polled_at=polled_at,
        events=events,
        lineups=lineups,
    )


def _find_match_object(blob: Any) -> dict[str, Any] | None:
    """Depth-first search for an object looking like a match payload."""
    def looks_like_match(obj: dict[str, Any]) -> bool:
        keys = {k.lower() for k in obj.keys()}
        return ("hometeam" in keys or "home" in keys) and (
            "awayteam" in keys or "away" in keys
        )

    stack: list[Any] = [blob]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if looks_like_match(node):
                return node
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return None


def _parse_events_json(raw: Any) -> list[MatchEvent]:
    if not isinstance(raw, list):
        return []
    events: list[MatchEvent] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        event_type = _normalise_event_type(
            _coerce_str(item.get("type") or item.get("eventType")) or ""
        )
        events.append(
            MatchEvent(
                minute=_coerce_int(item.get("minute")),
                team=_coerce_str(item.get("team")),
                player=_coerce_str(item.get("player") or item.get("playerName")),
                event_type=event_type,
                detail=_coerce_str(item.get("detail") or item.get("description")),
            )
        )
    return events


def _parse_lineups_json(raw: Any) -> list[Lineup]:
    if not isinstance(raw, list):
        return []
    lineups: list[Lineup] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        starting = item.get("startingXI") or item.get("starters") or []
        bench = item.get("bench") or item.get("substitutes") or []
        lineups.append(
            Lineup(
                team=_coerce_str(item.get("team")),
                coach=_coerce_str(item.get("coach") or item.get("manager")),
                starting_xi=[_coerce_str(x) or "" for x in starting if _coerce_str(x)],
                bench=[_coerce_str(x) or "" for x in bench if _coerce_str(x)],
            )
        )
    return lineups


# ---------------------------------------------------------------------------
# HTML fallback
# ---------------------------------------------------------------------------


def _parse_from_html(soup: BeautifulSoup, *, url: str | None, polled_at: str) -> LiveMatch | None:
    home_node = soup.select_one("[data-testid*='home'], .team-home, .home-team")
    away_node = soup.select_one("[data-testid*='away'], .team-away, .away-team")
    if not isinstance(home_node, Tag) and not isinstance(away_node, Tag):
        log.warning("parse_live_match: could not find team nodes in HTML")
        return None

    home_team = clean_text(home_node.get_text(" ", strip=True)) if isinstance(home_node, Tag) else None
    away_team = clean_text(away_node.get_text(" ", strip=True)) if isinstance(away_node, Tag) else None

    score_text = ""
    score_node = soup.select_one(".score, [data-testid*='score']")
    if isinstance(score_node, Tag):
        score_text = score_node.get_text(" ", strip=True)
    else:
        score_text = soup.get_text(" ", strip=True)

    m = _SCORE_RE.search(score_text)
    home_score = int(m.group("h")) if m else None
    away_score = int(m.group("a")) if m else None

    minute = None
    minute_node = soup.select_one(".minute, [data-testid*='minute'], [data-testid*='clock']")
    if isinstance(minute_node, Tag):
        mm = _MINUTE_RE.search(minute_node.get_text(" ", strip=True))
        if mm:
            try:
                minute = int(mm.group("m"))
            except ValueError:
                minute = None

    status_text = ""
    status_node = soup.select_one(".status, [data-testid*='status']")
    if isinstance(status_node, Tag):
        status_text = status_node.get_text(" ", strip=True)
    status, is_live = _normalise_status(status_text)

    events = list(_parse_events_html(soup))
    lineups = list(_parse_lineups_html(soup))

    match_id = stable_hash(url or (home_team or "") + (away_team or ""))

    return LiveMatch(
        match_id=match_id,
        match_url=normalise_url(url) if url else None,
        competition=None,
        home_team=home_team,
        away_team=away_team,
        home_score=home_score,
        away_score=away_score,
        current_minute=minute,
        status=status,
        is_live=is_live,
        polled_at=polled_at,
        events=events,
        lineups=lineups,
    )


def _parse_events_html(soup: BeautifulSoup) -> Iterable[MatchEvent]:
    for node in soup.select(".event, [data-testid*='event']"):
        if not isinstance(node, Tag):
            continue
        text = node.get_text(" ", strip=True)
        event_type = _normalise_event_type(text.lower())
        minute_match = _MINUTE_RE.search(text)
        minute = int(minute_match.group("m")) if minute_match else None
        team_node = node.select_one(".team, [data-testid*='team']")
        player_node = node.select_one(".player, [data-testid*='player']")
        yield MatchEvent(
            minute=minute,
            team=clean_text(team_node.get_text(" ", strip=True)) if isinstance(team_node, Tag) else None,
            player=clean_text(player_node.get_text(" ", strip=True)) if isinstance(player_node, Tag) else None,
            event_type=event_type,
            detail=clean_text(text),
        )


def _parse_lineups_html(soup: BeautifulSoup) -> Iterable[Lineup]:
    for node in soup.select(".lineup, [data-testid*='lineup']"):
        if not isinstance(node, Tag):
            continue
        team_node = node.find(["h2", "h3", "h4"])
        team = clean_text(team_node.get_text(" ", strip=True)) if isinstance(team_node, Tag) else None
        starters: list[str] = []
        bench: list[str] = []
        for li in node.select(".starter, .starting li"):
            text = clean_text(li.get_text(" ", strip=True))
            if text:
                starters.append(text)
        for li in node.select(".bench li, .substitute"):
            text = clean_text(li.get_text(" ", strip=True))
            if text:
                bench.append(text)
        coach_node = node.select_one(".coach, [data-testid*='coach']")
        coach = clean_text(coach_node.get_text(" ", strip=True)) if isinstance(coach_node, Tag) else None
        yield Lineup(
            team=team,
            coach=coach,
            starting_xi=dedupe_preserve_order(starters),
            bench=dedupe_preserve_order(bench),
        )


# ---------------------------------------------------------------------------


def _normalise_status(value: str) -> tuple[MatchStatus, bool]:
    lowered = (value or "").lower()
    if any(k in lowered for k in ("live", "bezig", "helft")):
        return "live", True
    if any(k in lowered for k in ("ft", "afgelopen", "einde", "finished", "full time")):
        return "finished", False
    if any(k in lowered for k in ("uitgesteld", "afgelast", "postponed")):
        return "postponed", False
    if any(k in lowered for k in ("scheduled", "aftrap", "nog niet")):
        return "scheduled", False
    return "unknown", False


def _normalise_event_type(value: str) -> EventType:
    lowered = value.lower()
    for candidate, keywords in _EVENT_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return candidate
    return "other"


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return clean_text(value)
    return clean_text(str(value))


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
