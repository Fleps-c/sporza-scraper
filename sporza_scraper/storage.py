"""Atomic JSON writers for scraped entities, with optional SQLite persistence."""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
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
from .player_extractor import PLAYER_TO_CLUB
from .utils import parse_iso_datetime

log = logging.getLogger(__name__)

_SAFE_SEGMENT_RE = re.compile(r"[^A-Za-z0-9._-]+")


class Storage:
    """Filesystem storage with atomic writes and an opinionated layout.

    Premier League articles are also persisted to a local SQLite database
    (``sporza_predictions.db``) for fast querying by the dashboard.
    """

    def __init__(self, root: Path = DEFAULT_OUTPUT_ROOT, dry_run: bool = False) -> None:
        self.root = Path(root)
        self.dry_run = dry_run

        # SQLite database setup
        self.db_path = self.root / "sporza_predictions.db"

        # Ensure the data folder exists
        self.root.mkdir(parents=True, exist_ok=True)

        if not self.dry_run:
            self._init_db()

    def _init_db(self) -> None:
        """Create database tables if they don't exist yet."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                slug TEXT PRIMARY KEY,
                title TEXT,
                published_at TEXT,
                url TEXT,
                article_type TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_slug TEXT,
                player_name TEXT,
                club TEXT,
                signal_type TEXT,
                score REAL,
                FOREIGN KEY (article_slug) REFERENCES articles (slug)
            )
        ''')

        # Index for fast lookups when re-scraping and dashboard queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_signals_article_slug
            ON signals (article_slug)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_signals_club
            ON signals (club)
        ''')

        conn.commit()
        conn.close()

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

        # 1. JSON file storage (always)
        self._write_json(path, payload)

        # 2. SQLite database storage
        if not self.dry_run:
            self._save_pl_to_db(article, enrichment, slug)

        return path

    def _save_pl_to_db(self, article: NewsArticle, enrichment: dict, slug: str) -> None:
        """Persist the article and its signals to the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Upsert article (INSERT OR REPLACE handles re-scrapes)
            cursor.execute('''
                INSERT OR REPLACE INTO articles (slug, title, published_at, url, article_type)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                slug,
                article.title,
                article.published_at,
                article.url,
                enrichment.get("article_type", "unknown"),
            ))

            # Remove old signals for this article (clean slate on re-scrape)
            cursor.execute('DELETE FROM signals WHERE article_slug = ?', (slug,))

            # Insert new signals
            signals = enrichment.get("performance_signals", [])
            for signal in signals:
                # PerformanceSignal.to_dict() uses key "player", not "player_name"
                player_name = signal.get("player") or signal.get("player_name")
                club = PLAYER_TO_CLUB.get(player_name, "Unknown")

                cursor.execute('''
                    INSERT INTO signals (article_slug, player_name, club, signal_type, score)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    slug,
                    player_name,
                    club,
                    signal.get("signal_type"),
                    signal.get("score"),
                ))

            conn.commit()
            log.info("Saved article '%s' and %d signals to database.", slug, len(signals))

        except sqlite3.Error as e:
            log.error("Database error for article %s: %s", slug, e)
        finally:
            conn.close()

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
