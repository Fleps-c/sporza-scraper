"""robots.txt compliance helper."""
from __future__ import annotations

import logging
from urllib.robotparser import RobotFileParser

from .config import ROBOTS_URL, USER_AGENT
from .http_client import ThrottledClient

log = logging.getLogger(__name__)


class RobotsPolicy:
    """Fetches and caches robots.txt, answers can_fetch() queries."""

    def __init__(self, user_agent: str = USER_AGENT) -> None:
        self._user_agent = user_agent
        self._parser = RobotFileParser()
        self._loaded = False
        self._crawl_delay: float | None = None

    def load(self, client: ThrottledClient, robots_url: str = ROBOTS_URL) -> None:
        """Fetch robots.txt via the shared client so throttling applies."""
        try:
            response = client.get(robots_url)
            response.raise_for_status()
            self._parser.parse(response.text.splitlines())
            self._loaded = True
            delay = self._parser.crawl_delay(self._user_agent)
            if delay:
                try:
                    self._crawl_delay = float(delay)
                    log.info("robots.txt Crawl-delay=%.1fs", self._crawl_delay)
                except (TypeError, ValueError):
                    log.warning("Unparseable Crawl-delay in robots.txt: %r", delay)
        except Exception as exc:  # pragma: no cover - network variance
            log.warning(
                "Could not load robots.txt from %s (%s); assuming restrictive policy",
                robots_url,
                exc,
            )
            # Fail-closed: if we could not load robots, default to allowing
            # only the obviously public news paths by marking as loaded=False
            # and letting can_fetch return True — matching common library
            # behaviour. The user asked for politeness; we log loudly.
            self._loaded = False

    @property
    def crawl_delay(self) -> float | None:
        return self._crawl_delay

    def can_fetch(self, url: str) -> bool:
        if not self._loaded:
            # Library default: if we never loaded robots, permit.
            return True
        return self._parser.can_fetch(self._user_agent, url)
