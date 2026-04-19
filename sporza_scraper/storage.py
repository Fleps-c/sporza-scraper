"""Atomic JSON writers for scraped entities."""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_OUTPUT_ROOT,
    FOOTBALL_LIVE_SUBDIR,
    FOOTBALL_RESULTS_SUBDIR,
    NEWS_SUBDIR,
    PREMIER_LEAGUE_SUBDIR,
    STATS_SUBDIR,
)
from .models import FootballMatch, LiveMatch, NewsArticle
from .utils import parse_iso_datetime

log = logging.getLogger(__name__)

_SAFE_SEGMENT_RE = re.compile(r"[^A-Za-z0-9._-]+")


class Storage:
    """Filesystem storage with atomic writes and an opinionated layout."""

    def __init__(self, root: Path = DEFAULT_OUTPUT_ROOT, dry_run: bool = False) -> None:
        self.root = Path(root)
        self.dry_run = dry_run

    # ------------------------------------------------------------------
    # Public writers
    # ------------------------------------------------------------------

    def save_news(self, article: NewsArticle) -> Path:
        dt = parse_iso_datetime(article.published_at) or datetime.now(tz=timezone.utc)
        folder = self.root / NEWS_SUBDIR / f"{dt.year:04d}" / f"{dt.month:02d}" / f"{dt.day:02d}"
        path = folder / f"{_safe_segment(article.slug)}.json"
        self._write_json(path, article.to_dict())
        return path

    def save_football_results(self, date_iso: str, matches: list[FootballMatch]) -> Path:
        folder = self.root / FOOTBALL_RESULTS_SUBDIR
        path = folder / f"{_safe_segment(date_iso)}.json"
        payload = {
            "date": date_iso,
            "count": len(matches),
            "matches": [m.to_dict() for m in matches],
        }
        self._write_json(path, payload)
        return path

    def save_pl_article(self, article: NewsArticle, enrichment: dict) -> Path:
        """Save a Premier League article with enrichment data."""
        dt = parse_iso_datetime(article.published_at) or datetime.now(tz=timezone.utc)
        folder = (
            self.root
            / PREMIER_LEAGUE_SUBDIR
            / f"{dt.year:04d}"
            / f"{dt.month:02d}"
            / f"{dt.day:02d}"
        )
        slug = _safe_segment(article.slug)
        path = folder / f"{slug}.json"
        payload = article.to_dict()
        payload["pl_enrichment"] = enrichment
        self._write_json(path, payload)
        return path

    def save_stats_csv(self, season: str, data: bytes) -> Path:
        """Save downloaded CSV to data/stats/E0_{season}.csv."""
        folder = self.root / STATS_SUBDIR
        path = folder / f"E0_{season}.csv"
        if self.dry_run:
            log.info("[dry-run] would write %s", path)
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        log.info("wrote %s (%d bytes)", path, len(data))
        return path

    def save_live_snapshot(self, live: LiveMatch) -> Path:
        folder = self.root / FOOTBALL_LIVE_SUBDIR / _safe_segment(live.match_id)
        ts = live.polled_at.replace(":", "-")
        path = folder / f"{_safe_segment(ts)}.json"
        self._write_json(path, live.to_dict())
        return path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_json(self, path: Path, payload: Any) -> None:
        if self.dry_run:
            log.info("[dry-run] would write %s", path)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        data = json.dumps(
            _normalise_for_json(payload),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        tmp.write_text(data, encoding="utf-8")
        os.replace(tmp, path)
        log.info("wrote %s", path)


def _safe_segment(value: str) -> str:
    """Make a string safe to use as a single path segment."""
    if not value:
        return "untitled"
    return _SAFE_SEGMENT_RE.sub("-", value).strip("-") or "untitled"


def _normalise_for_json(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _normalise_for_json(asdict(value))
    if isinstance(value, dict):
        return {k: _normalise_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise_for_json(v) for v in value]
    return value
