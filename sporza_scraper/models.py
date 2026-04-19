"""Typed dataclasses representing scraped entities."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

MatchStatus = Literal["scheduled", "live", "finished", "postponed", "unknown"]
EventType = Literal[
    "goal",
    "own_goal",
    "penalty",
    "penalty_missed",
    "yellow_card",
    "red_card",
    "substitution",
    "other",
]


@dataclass(slots=True)
class Image:
    url: str
    alt: str | None = None


@dataclass(slots=True)
class NewsArticle:
    url: str
    slug: str
    title: str | None
    lead: str | None
    authors: list[str] = field(default_factory=list)
    published_at: str | None = None  # ISO 8601, Europe/Brussels
    updated_at: str | None = None
    category: str | None = None
    tags: list[str] = field(default_factory=list)
    body_paragraphs: list[str] = field(default_factory=list)
    images: list[Image] = field(default_factory=list)
    related_links: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FootballMatch:
    match_url: str | None
    competition: str | None
    matchday: str | None
    kickoff_at: str | None
    home_team: str | None
    away_team: str | None
    home_score: int | None
    away_score: int | None
    halftime_home_score: int | None
    halftime_away_score: int | None
    venue: str | None
    status: MatchStatus

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MatchEvent:
    minute: int | None
    team: str | None
    player: str | None
    event_type: EventType
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Lineup:
    team: str | None
    coach: str | None
    starting_xi: list[str] = field(default_factory=list)
    bench: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LiveMatch:
    match_id: str
    match_url: str | None
    competition: str | None
    home_team: str | None
    away_team: str | None
    home_score: int | None
    away_score: int | None
    current_minute: int | None
    status: MatchStatus
    is_live: bool
    polled_at: str  # ISO 8601 UTC
    events: list[MatchEvent] = field(default_factory=list)
    lineups: list[Lineup] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
