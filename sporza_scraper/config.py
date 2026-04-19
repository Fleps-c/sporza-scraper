"""Configuration constants for the Sporza scraper."""
from __future__ import annotations

from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Target site
# ---------------------------------------------------------------------------
BASE_URL: Final[str] = "https://sporza.be"
ROBOTS_URL: Final[str] = f"{BASE_URL}/robots.txt"

# Section / index URLs used for discovery. The scraper falls back gracefully
# if any of these change — see parsers for heuristic discovery as a backup.
NEWS_INDEX_URL: Final[str] = f"{BASE_URL}/nl/"
FOOTBALL_INDEX_URL: Final[str] = f"{BASE_URL}/nl/voetbal/"
LIVE_INDEX_URL: Final[str] = f"{BASE_URL}/nl/matchcenter/"

# Premier League specific index pages for targeted scraping.
# Note: Sporza restructured their URLs — /categorie/voetbal/premier-league/
# and /categorie/voetbal/engeland/ return 404. We use the general football
# section plus match-center pages which do work.
PL_INDEX_URLS: Final[tuple[str, ...]] = (
    f"{BASE_URL}/nl/categorie/voetbal/",
    f"{BASE_URL}/nl/",
    f"{BASE_URL}/nl/categorie/voetbal/champions-league/",
    f"{BASE_URL}/nl/categorie/voetbal/europa-league/",
    f"{BASE_URL}/nl/pas-verschenen/",
)
# Sporza search is JS-rendered, so we use direct section URLs instead.
PL_SEARCH_URL_TEMPLATE: Final[str] = f"{BASE_URL}/nl/zoek/?query={{query}}"

# Additional index pages used as fallbacks when the homepage alone does not
# yield enough article links. Ordered from most to least general.
NEWS_FALLBACK_INDEX_URLS: Final[tuple[str, ...]] = (
    f"{BASE_URL}/nl/pas-verschenen/",
    f"{BASE_URL}/nl/categorie/voetbal/",
    f"{BASE_URL}/nl/categorie/voetbal/jupiler-pro-league/",
    f"{BASE_URL}/nl/categorie/voetbal/champions-league/",
    f"{BASE_URL}/nl/categorie/wielrennen/",
    f"{BASE_URL}/nl/categorie/tennis/",
    f"{BASE_URL}/nl/categorie/auto-motor/formule-1/",
    f"{BASE_URL}/nl/categorie/zaalsport/basketbal/",
)

# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------
USER_AGENT: Final[str] = "SporzaScraperBot/1.0 (+contact: <email placeholder>)"

# Connect / read timeouts (seconds)
CONNECT_TIMEOUT: Final[float] = 15.0
READ_TIMEOUT: Final[float] = 30.0
REQUEST_TIMEOUT: Final[tuple[float, float]] = (CONNECT_TIMEOUT, READ_TIMEOUT)

# Minimum delay between requests to the same host (seconds)
MIN_REQUEST_INTERVAL: Final[float] = 1.5

# Retry configuration (tenacity)
RETRY_MAX_ATTEMPTS: Final[int] = 5
RETRY_BASE_SECONDS: Final[float] = 2.0
RETRY_CAP_SECONDS: Final[float] = 60.0

# HTTP status codes that trigger a retry
RETRY_STATUS_CODES: Final[frozenset[int]] = frozenset(
    {408, 425, 429, 500, 502, 503, 504}
)

# ---------------------------------------------------------------------------
# Live-match polling
# ---------------------------------------------------------------------------
MIN_POLL_INTERVAL_SECONDS: Final[int] = 30
DEFAULT_POLL_INTERVAL_SECONDS: Final[int] = 60

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_ROOT: Final[Path] = Path("data")

NEWS_SUBDIR: Final[str] = "news"
FOOTBALL_RESULTS_SUBDIR: Final[str] = "football/results"
FOOTBALL_LIVE_SUBDIR: Final[str] = "football/live"
PREMIER_LEAGUE_SUBDIR: Final[str] = "premier-league"
STATS_SUBDIR: Final[str] = "stats"

# ---------------------------------------------------------------------------
# Football-data.co.uk CSV
# ---------------------------------------------------------------------------
FOOTBALL_DATA_CSV_URL_TEMPLATE: Final[str] = (
    "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
)
DEFAULT_CSV_SEASON: Final[str] = "2425"

# Team name normalisation: maps variant names → canonical form.
# The CSV uses short names; our player_extractor uses long names.
TEAM_NAME_ALIASES: Final[dict[str, str]] = {
    # CSV short names → canonical
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Nott'm Forest": "Nottingham Forest",
    "Spurs": "Tottenham",
    "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
    "Newcastle": "Newcastle United",
    "Leicester": "Leicester City",
    "Ipswich": "Ipswich Town",
    "Brighton": "Brighton & Hove Albion",
    "Bournemouth": "AFC Bournemouth",
    # Long names → canonical (identity or synonym)
    "Manchester City": "Manchester City",
    "Manchester United": "Manchester United",
    "Nottingham Forest": "Nottingham Forest",
    "West Ham United": "West Ham United",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers",
    "Newcastle United": "Newcastle United",
    "Leicester City": "Leicester City",
    "Ipswich Town": "Ipswich Town",
    "Brighton & Hove Albion": "Brighton & Hove Albion",
    "AFC Bournemouth": "AFC Bournemouth",
    "Tottenham Hotspur": "Tottenham",
    "Tottenham": "Tottenham",
    "Arsenal": "Arsenal",
    "Liverpool": "Liverpool",
    "Chelsea": "Chelsea",
    "Aston Villa": "Aston Villa",
    "Brentford": "Brentford",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Southampton": "Southampton",
    "Leeds": "Leeds United",
    "Leeds United": "Leeds United",
    "Luton": "Luton Town",
    "Luton Town": "Luton Town",
    "Burnley": "Burnley",
    "Sheffield United": "Sheffield United",
    "Sheffield Utd": "Sheffield United",
}

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
# Query-string parameters to strip from stored URLs.
TRACKING_QUERY_PARAMS: Final[frozenset[str]] = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "gclid",
        "fbclid",
        "mc_cid",
        "mc_eid",
        "msclkid",
        "ref",
        "ref_src",
    }
)

# Dutch month names (lowercase) for date parsing fallbacks.
DUTCH_MONTHS: Final[dict[str, int]] = {
    "januari": 1,
    "februari": 2,
    "maart": 3,
    "april": 4,
    "mei": 5,
    "juni": 6,
    "juli": 7,
    "augustus": 8,
    "september": 9,
    "oktober": 10,
    "november": 11,
    "december": 12,
}

TIMEZONE_NAME: Final[str] = "Europe/Brussels"
