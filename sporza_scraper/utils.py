"""Utility helpers: date parsing, URL normalisation, text cleaning."""
from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from datetime import datetime, timezone
from typing import Iterable
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse
from zoneinfo import ZoneInfo

from .config import BASE_URL, DUTCH_MONTHS, TIMEZONE_NAME, TRACKING_QUERY_PARAMS

log = logging.getLogger(__name__)

BRUSSELS_TZ = ZoneInfo(TIMEZONE_NAME)

_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(value: str | None) -> str | None:
    """Collapse whitespace and strip. Return None if empty."""
    if value is None:
        return None
    normalised = unicodedata.normalize("NFKC", value)
    collapsed = _WHITESPACE_RE.sub(" ", normalised).strip()
    return collapsed or None


def slugify(value: str) -> str:
    """Create a filesystem-safe slug from an arbitrary string."""
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return value or "untitled"


def stable_hash(value: str, length: int = 10) -> str:
    """Deterministic short hash, useful for fallback filenames."""
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def absolute_url(href: str | None, base: str = BASE_URL) -> str | None:
    """Convert a relative URL to absolute; leave absolute URLs alone."""
    if not href:
        return None
    return urljoin(base, href.strip())


def strip_tracking(url: str | None) -> str | None:
    """Remove known tracking query parameters from a URL."""
    if not url:
        return None
    parsed = urlparse(url)
    kept = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=False)
            if k.lower() not in TRACKING_QUERY_PARAMS]
    cleaned = parsed._replace(query=urlencode(kept), fragment="")
    return urlunparse(cleaned)


def normalise_url(href: str | None, base: str = BASE_URL) -> str | None:
    """Absolute + tracking-stripped URL."""
    return strip_tracking(absolute_url(href, base))


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    """Deduplicate items while preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

_ISO_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?"
)
_DUTCH_DATE_RE = re.compile(
    r"(?P<day>\d{1,2})\s+(?P<month>[a-zA-Z]+)\s+(?P<year>\d{4})"
    r"(?:[^\d]+(?P<hour>\d{1,2})[:\.](?P<minute>\d{2}))?",
    re.IGNORECASE,
)


def parse_iso_datetime(value: str | None) -> datetime | None:
    """Parse an ISO 8601 datetime string. Returns timezone-aware datetime."""
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    # datetime.fromisoformat handles most variants in 3.11+.
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=BRUSSELS_TZ)
    return dt


def parse_dutch_datetime(value: str | None) -> datetime | None:
    """Parse a Dutch-language date like '15 april 2026 20:45'."""
    if not value:
        return None
    match = _DUTCH_DATE_RE.search(value)
    if not match:
        return None
    month = DUTCH_MONTHS.get(match.group("month").lower())
    if not month:
        return None
    try:
        day = int(match.group("day"))
        year = int(match.group("year"))
        hour = int(match.group("hour") or 0)
        minute = int(match.group("minute") or 0)
        return datetime(year, month, day, hour, minute, tzinfo=BRUSSELS_TZ)
    except (ValueError, TypeError):
        return None


def parse_any_datetime(value: str | None) -> datetime | None:
    """Best-effort datetime parser: ISO first, then Dutch text."""
    if not value:
        return None
    return parse_iso_datetime(value) or parse_dutch_datetime(value)


def to_iso_string(dt: datetime | None) -> str | None:
    """Render a datetime as ISO 8601 in Europe/Brussels."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=BRUSSELS_TZ)
    return dt.astimezone(BRUSSELS_TZ).isoformat()


def utcnow_iso() -> str:
    """Current UTC time as ISO 8601."""
    return datetime.now(tz=timezone.utc).isoformat()
