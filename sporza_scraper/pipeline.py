"""Orchestration: discover → fetch → parse → store."""
from __future__ import annotations

import logging
import signal
import time
from datetime import date, datetime, timedelta
from typing import Iterable

from .config import (
    DEFAULT_POLL_INTERVAL_SECONDS,
    FOOTBALL_INDEX_URL,
    LIVE_INDEX_URL,
    MIN_POLL_INTERVAL_SECONDS,
    NEWS_FALLBACK_INDEX_URLS,
    NEWS_INDEX_URL,
    PL_INDEX_URLS,
    PL_SEARCH_URL_TEMPLATE,
)
from .http_client import ThrottledClient
from .parsers import (
    discover_football_result_links,
    discover_news_links,
    parse_football_results,
    parse_live_match,
    parse_news_article,
)
from .robots import RobotsPolicy
from .storage import Storage
from .utils import parse_iso_datetime

log = logging.getLogger(__name__)


class Pipeline:
    """Coordinates the HTTP client, robots policy, parsers, and storage."""

    def __init__(self, client: ThrottledClient, robots: RobotsPolicy, storage: Storage) -> None:
        self.client = client
        self.robots = robots
        self.storage = storage

    # ------------------------------------------------------------------
    # News
    # ------------------------------------------------------------------

    def scrape_news(self, limit: int, since: date | None = None) -> int:
        log.info("Starting news scrape (limit=%d)", limit)
        all_links: list[str] = []
        seen_urls: set[str] = set()

        def _add_links(new_links: list[str]) -> None:
            for l in new_links:
                if l not in seen_urls:
                    seen_urls.add(l)
                    all_links.append(l)

        # Primary: homepage.
        index_html = self._fetch_text(NEWS_INDEX_URL)
        if index_html:
            _add_links(discover_news_links(index_html))
        log.info("Homepage: %d candidate article URLs", len(all_links))

        # Fallback: section pages, stopping once we have enough to satisfy
        # ``limit`` (with some headroom for parser rejections).
        for fallback_url in NEWS_FALLBACK_INDEX_URLS:
            if len(all_links) >= limit * 3:
                break
            if not self.robots.can_fetch(fallback_url):
                continue
            html = self._fetch_text(fallback_url)
            if not html:
                continue
            before = len(all_links)
            _add_links(discover_news_links(html))
            log.info(
                "Section %s: +%d new URLs (total %d)",
                fallback_url,
                len(all_links) - before,
                len(all_links),
            )

        log.info("Discovered %d candidate article URLs in total", len(all_links))

        saved = 0
        seen: set[str] = set()
        for link in all_links:
            if saved >= limit:
                break
            if link in seen:
                continue
            seen.add(link)
            if not self.robots.can_fetch(link):
                log.info("robots.txt disallows %s — skipping", link)
                continue
            html = self._fetch_text(link)
            if not html:
                continue
            try:
                article = parse_news_article(html, url=link)
            except Exception as exc:
                log.warning("Failed to parse %s: %s", link, exc)
                continue
            if not article:
                continue
            if since and article.published_at:
                published = parse_iso_datetime(article.published_at)
                if published and published.date() < since:
                    log.debug("Skipping %s: older than --since", link)
                    continue
            self.storage.save_news(article)
            saved += 1
            log.info("saved article %d/%d: %s", saved, limit, article.slug)
        log.info("News scrape complete: %d articles saved", saved)
        return saved

    # ------------------------------------------------------------------
    # Premier League scraping
    # ------------------------------------------------------------------

    def scrape_premier_league(
        self,
        limit: int,
        search_queries: list[str] | None = None,
    ) -> int:
        """Scrape PL-focused articles from index pages and search results.

        Discovers links from ``PL_INDEX_URLS``, then optionally from
        search queries (e.g. player names). Each article is enriched
        with player mentions and performance signals before saving.
        """
        from .player_extractor import extract_player_mentions
        from .signals import classify_article_type, detect_signals

        log.info("Starting PL scrape (limit=%d)", limit)
        all_links: list[str] = []
        seen_urls: set[str] = set()

        def _add_links(new_links: list[str]) -> None:
            for l in new_links:
                if l not in seen_urls:
                    seen_urls.add(l)
                    all_links.append(l)

        # 1. Index pages.
        for index_url in PL_INDEX_URLS:
            if not self.robots.can_fetch(index_url):
                log.warning("robots.txt disallows PL index %s — skipping", index_url)
                continue
            html = self._fetch_text(index_url)
            if not html:
                log.warning("Could not fetch PL index %s", index_url)
                continue
            log.debug("PL index %s: fetched %d bytes", index_url, len(html))
            from .parsers import discover_news_links
            new_links = discover_news_links(html)
            log.info(
                "PL index %s: discovered %d article links", index_url, len(new_links),
            )
            _add_links(new_links)

            # Also try broader link discovery: any /nl/ link containing
            # PL-related keywords, even without ~digits pattern.
            broad = self._discover_pl_links_broad(html)
            if broad:
                log.info(
                    "PL index %s: +%d broad links", index_url, len(broad),
                )
                _add_links(broad)

            if len(all_links) >= limit * 3:
                break

        # 2. Search queries (optional — Sporza search is JS-rendered, so
        #    these often return 0 links. We try anyway in case it works.)
        if search_queries:
            for query in search_queries:
                if len(all_links) >= limit * 3:
                    break
                search_url = PL_SEARCH_URL_TEMPLATE.format(query=query)
                if not self.robots.can_fetch(search_url):
                    log.debug("robots.txt disallows search %s", search_url)
                    continue
                html = self._fetch_text(search_url)
                if not html:
                    continue
                from .parsers import discover_news_links
                new_links = discover_news_links(html)
                if new_links:
                    log.info("Search '%s': discovered %d links", query, len(new_links))
                    _add_links(new_links)
                broad = self._discover_pl_links_broad(html)
                if broad:
                    log.info("Search '%s': +%d broad links", query, len(broad))
                    _add_links(broad)

        log.info("PL discovery: %d candidate URLs", len(all_links))

        saved = 0
        for link in all_links:
            if saved >= limit:
                break
            if not self.robots.can_fetch(link):
                log.debug("robots.txt disallows %s — skipping", link)
                continue
            html = self._fetch_text(link)
            if not html:
                log.debug("Could not fetch article %s", link)
                continue
            try:
                from .parsers import parse_news_article
                article = parse_news_article(html, url=link)
            except Exception as exc:
                log.warning("Failed to parse %s: %s", link, exc)
                continue
            if not article:
                log.debug("Parser returned None for %s", link)
                continue
            if not article.body_paragraphs:
                log.debug("Empty body for %s — skipping", link)
                continue

            # --- PL enrichment ---
            extraction = extract_player_mentions(
                article.body_paragraphs, premier_league_only=True,
            )
            signals = detect_signals(
                article.body_paragraphs, extraction.unique_names,
            )
            article_type = classify_article_type(
                article.title, article.body_paragraphs,
            )
            enrichment = {
                "article_type": article_type,
                "players_mentioned": [
                    {"name": m.name, "mention_count": 1, "context": m.context}
                    for m in extraction.mentions
                ],
                "unique_players": extraction.unique_names,
                "performance_signals": [s.to_dict() for s in signals],
            }

            self.storage.save_pl_article(article, enrichment)
            saved += 1
            log.info(
                "PL saved %d/%d: %s (%d players, %d signals)",
                saved, limit, article.slug,
                len(extraction.unique_names), len(signals),
            )

        log.info("PL scrape complete: %d articles saved", saved)
        return saved

    # ------------------------------------------------------------------
    # Football results
    # ------------------------------------------------------------------

    def scrape_football_results(self, dates: Iterable[date]) -> int:
        dates = list(dates)
        log.info("Starting football results scrape for %d date(s)", len(dates))
        total = 0
        for d in dates:
            url = f"{FOOTBALL_INDEX_URL}?date={d.isoformat()}"
            if not self.robots.can_fetch(url):
                log.info("robots.txt disallows %s — skipping", url)
                continue
            html = self._fetch_text(url)
            if not html:
                continue
            try:
                matches = parse_football_results(html)
            except Exception as exc:
                log.warning("Failed to parse results for %s: %s", d, exc)
                continue
            self.storage.save_football_results(d.isoformat(), matches)
            total += len(matches)
            log.info("date=%s matches=%d", d.isoformat(), len(matches))
        log.info("Football results scrape complete: %d matches total", total)
        return total

    # ------------------------------------------------------------------
    # Live matches
    # ------------------------------------------------------------------

    def scrape_live_matches(
        self,
        poll_interval: int = DEFAULT_POLL_INTERVAL_SECONDS,
        duration_minutes: int | None = None,
    ) -> int:
        if poll_interval < MIN_POLL_INTERVAL_SECONDS:
            log.warning(
                "poll-interval %d raised to minimum %d",
                poll_interval,
                MIN_POLL_INTERVAL_SECONDS,
            )
            poll_interval = MIN_POLL_INTERVAL_SECONDS

        stop_at: float | None = None
        if duration_minutes is not None:
            stop_at = time.monotonic() + duration_minutes * 60

        stop_flag = {"stop": False}

        def _handle_sigint(signum: int, frame: object) -> None:  # noqa: ARG001
            log.info("SIGINT received — finishing current poll and exiting")
            stop_flag["stop"] = True

        previous = signal.signal(signal.SIGINT, _handle_sigint)
        snapshots = 0
        try:
            while not stop_flag["stop"]:
                snapshots += self._poll_live_once()
                if stop_at is not None and time.monotonic() >= stop_at:
                    log.info("Duration reached; exiting live poll loop")
                    break
                if stop_flag["stop"]:
                    break
                time.sleep(poll_interval)
        finally:
            signal.signal(signal.SIGINT, previous)
        log.info("Live scrape complete: %d snapshots", snapshots)
        return snapshots

    def _poll_live_once(self) -> int:
        if not self.robots.can_fetch(LIVE_INDEX_URL):
            log.info("robots.txt disallows %s — skipping", LIVE_INDEX_URL)
            return 0
        index_html = self._fetch_text(LIVE_INDEX_URL)
        if not index_html:
            return 0
        links = discover_football_result_links(index_html)
        saved = 0
        for link in links:
            if not self.robots.can_fetch(link):
                continue
            html = self._fetch_text(link)
            if not html:
                continue
            try:
                snapshot = parse_live_match(html, url=link)
            except Exception as exc:
                log.warning("Failed to parse live match %s: %s", link, exc)
                continue
            if not snapshot or not snapshot.is_live:
                continue
            self.storage.save_live_snapshot(snapshot)
            saved += 1
        log.info("live poll wrote %d snapshots", saved)
        return saved

    # ------------------------------------------------------------------

    @staticmethod
    def _discover_pl_links_broad(html: str) -> list[str]:
        """Broader link discovery: find any sporza.be /nl/ link that might
        be a PL article, even without the strict ~digits filter.

        This catches articles whose URLs use a different format (slug-only).
        """
        import re
        from bs4 import BeautifulSoup
        from .utils import normalise_url

        soup = BeautifulSoup(html, "lxml")
        # PL-related keywords that signal a link is about Premier League
        pl_keywords = (
            "premier-league", "engeland", "manchester", "liverpool",
            "arsenal", "chelsea", "tottenham", "city", "united",
            "de-bruyne", "trossard", "doku", "onana", "lukaku",
        )
        excluded_segments = (
            "/categorie/", "/pas-verschenen/", "/video/", "/audio/",
            "/matchcenter/", "/live/", "/podcast/",
        )
        links: list[str] = []
        seen: set[str] = set()

        for a in soup.select("a[href]"):
            href = a.get("href")
            if not href or not isinstance(href, str):
                continue
            normalised = normalise_url(href, base="https://sporza.be")
            if not normalised or normalised in seen:
                continue
            if "sporza.be" not in normalised:
                continue
            lowered = normalised.lower()
            if not lowered.split("sporza.be", 1)[-1].startswith("/nl/"):
                continue
            # Skip known non-article sections
            if any(seg in lowered for seg in excluded_segments):
                continue
            # Must be a deep-enough path (at least 3 segments after /nl/)
            path = lowered.split("sporza.be", 1)[-1]
            parts = [p for p in path.strip("/").split("/") if p]
            if len(parts) < 3:
                continue
            # Accept if the URL or link text contains PL keywords
            link_text = (a.get_text(" ", strip=True) or "").lower()
            combined = lowered + " " + link_text
            if any(kw in combined for kw in pl_keywords):
                seen.add(normalised)
                links.append(normalised)

        return links

    def _fetch_text(self, url: str) -> str | None:
        try:
            response = self.client.get(url)
        except Exception as exc:
            log.warning("fetch failed for %s: %s", url, exc)
            return None
        if response.status_code >= 400:
            log.warning("non-2xx %d for %s", response.status_code, url)
            return None
        return response.text


def daterange(start: date, end: date) -> Iterable[date]:
    """Yield consecutive dates from ``start`` to ``end`` inclusive."""
    if end < start:
        raise ValueError("end date must be >= start date")
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def parse_cli_date(value: str) -> date:
    """Parse a YYYY-MM-DD CLI argument into a ``date``."""
    return datetime.strptime(value, "%Y-%m-%d").date()
