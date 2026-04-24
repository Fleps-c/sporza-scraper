"""Throttled HTTP client with retry and polite behaviour."""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

import requests
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from .config import (
    MIN_REQUEST_INTERVAL,
    REQUEST_TIMEOUT,
    RETRY_BASE_SECONDS,
    RETRY_CAP_SECONDS,
    RETRY_MAX_ATTEMPTS,
    RETRY_STATUS_CODES,
    USER_AGENT,
)

log = logging.getLogger(__name__)


class RetryableHTTPError(Exception):
    """Raised to trigger a tenacity retry on a retryable HTTP response."""

    def __init__(self, status: int, url: str, retry_after: float | None = None):
        super().__init__(f"Retryable HTTP {status} for {url}")
        self.status = status
        self.url = url
        self.retry_after = retry_after


def _is_retryable(exc: BaseException) -> bool:
    """Return True if the exception should trigger a retry."""
    if isinstance(exc, RetryableHTTPError):
        return True
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    return False


def _parse_retry_after(header: str | None) -> float | None:
    if not header:
        return None
    try:
        return max(0.0, float(header))
    except ValueError:
        # HTTP-date values are uncommon for this site; fall back to None.
        return None


class ThrottledClient:
    """A ``requests.Session`` wrapper with per-host throttling and retries."""

    def __init__(
        self,
        user_agent: str = USER_AGENT,
        min_interval: float = MIN_REQUEST_INTERVAL,
    ) -> None:
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept-Language": "nl-BE,nl;q=0.9,en;q=0.5",
                "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
            }
        )
        self._min_interval = min_interval
        self._last_request_at: dict[str, float] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_min_interval(self, seconds: float) -> None:
        """Allow the pipeline to raise the delay (e.g., robots Crawl-delay)."""
        with self._lock:
            if seconds > self._min_interval:
                log.info("Raising min request interval to %.2fs", seconds)
                self._min_interval = seconds

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        return self._request("GET", url, **kwargs)

    def close(self) -> None:
        self._session.close()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ThrottledClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _throttle(self, host: str) -> None:
        with self._lock:
            now = time.monotonic()
            last = self._last_request_at.get(host, 0.0)
            wait = self._min_interval - (now - last)
            if wait > 0:
                time.sleep(wait)
            self._last_request_at[host] = time.monotonic()

    def _log_retry(self, retry_state: RetryCallState) -> None:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        log.warning(
            "Retry #%d after %.2fs due to: %s",
            retry_state.attempt_number,
            retry_state.next_action.sleep if retry_state.next_action else 0,
            exc,
        )

    def _request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        timeout = kwargs.pop("timeout", REQUEST_TIMEOUT)
        host = requests.utils.urlparse(url).netloc  # type: ignore[attr-defined]

        retryer = Retrying(
            reraise=True,
            stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
            wait=wait_random_exponential(
                multiplier=RETRY_BASE_SECONDS, max=RETRY_CAP_SECONDS
            ),
            retry=retry_if_exception(_is_retryable),
            before_sleep=self._log_retry,
        )

        for attempt in retryer:
            with attempt:
                self._throttle(host)
                start = time.monotonic()
                try:
                    response = self._session.request(
                        method, url, timeout=timeout, **kwargs
                    )
                except (requests.ConnectionError, requests.Timeout) as e:
                    log.warning("Network error for %s: %s", url, e)
                    raise
                latency = time.monotonic() - start
                log.info(
                    "%s %s -> %d (%.2fs)",
                    method,
                    url,
                    response.status_code,
                    latency,
                )
                if response.status_code in RETRY_STATUS_CODES:
                    retry_after = _parse_retry_after(
                        response.headers.get("Retry-After")
                    )
                    if retry_after:
                        log.warning(
                            "Server sent Retry-After=%.1fs for %s", retry_after, url
                        )
                        time.sleep(retry_after)
                    raise RetryableHTTPError(
                        response.status_code, url, retry_after=retry_after
                    )
                return response

        # Unreachable: tenacity reraises on final failure.
        raise RuntimeError("retry loop exited without returning")  # pragma: no cover
