# Sporza Scraper

A polite, resilient Python scraper for **sporza.be** (VRT's Flemish sports site) that collects **news articles**, **football match results**, and **live match snapshots** into structured JSON.

## Ethical & Legal Notice

This scraper accesses publicly available pages on sporza.be. Users are responsible for reviewing and complying with the VRT/Sporza terms of use and applicable Belgian and EU law (including GDPR) before running it. The tool identifies itself honestly, obeys robots.txt, rate-limits requests, and does not bypass any access controls, paywalls, or anti-bot measures. Do not republish scraped content without permission from the rights holders.

**Intended use:** personal research and technical experimentation only.

## Installation

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the **sentiment analysis** feature, you also need to download the Dutch
spaCy language model (one-time, ~15 MB):

```bash
python -m spacy download nl_core_news_md
```

(If you skip this step, the scraper will download it automatically on first
run of the `sentiment` command.)

## CLI Usage

```bash
# News — latest 5 articles
python -m sporza_scraper news --limit 5

# News since a date
python -m sporza_scraper news --limit 20 --since 2026-04-01

# Football results for a single date
python -m sporza_scraper football-results --date 2026-04-14

# Football results across a range
python -m sporza_scraper football-results --range 2026-04-10 2026-04-14

# Live poll every 60s for 5 minutes
python -m sporza_scraper live --poll-interval 60 --duration 5
```

Global flags: `--out PATH`, `--log-level {DEBUG,INFO,WARNING,ERROR}`, `--dry-run`.

### Sentiment analysis

After scraping some news articles, analyse how Sporza writes about Premier
League players:

```bash
# Step 1: Scrape articles (the more, the better)
python -m sporza_scraper news --limit 30

# Step 2: Run sentiment analysis on the scraped data
python -m sporza_scraper sentiment

# Top-10 most-mentioned players only
python -m sporza_scraper sentiment --top 10

# Include ALL detected player names, not just Premier League
python -m sporza_scraper sentiment --all-players

# Save full report as JSON
python -m sporza_scraper sentiment --report report.json
```

## Output Layout

```
data/
├── news/YYYY/MM/DD/<slug>.json
├── football/results/YYYY-MM-DD.json
└── football/live/<match_id>/<ISO_timestamp>.json
```

All JSON is written atomically (`.tmp` → `os.replace`), pretty-printed with
sorted keys and `ensure_ascii=False` so that Dutch diacritics stay readable.

## Data Schemas

### News article (excerpt)

```json
{
  "url": "https://sporza.be/nl/voetbal/voorbeeld-artikel-123456/",
  "slug": "voorbeeld-artikel-123456",
  "title": "Voorbeeldtitel van het artikel",
  "lead": "Korte lead van het artikel.",
  "authors": ["Jan Janssens"],
  "published_at": "2026-04-15T18:30:00+02:00",
  "updated_at": "2026-04-15T19:00:00+02:00",
  "category": "Voetbal",
  "tags": ["Jupiler Pro League", "Anderlecht"],
  "body_paragraphs": ["Eerste paragraaf…", "Tweede paragraaf…"],
  "images": [{"url": "https://sporza.be/img/hero.jpg", "alt": "Hero"}],
  "related_links": ["https://sporza.be/nl/voetbal/ander-artikel-987654/"]
}
```

### Football match

```json
{
  "match_url": "https://sporza.be/nl/matches/brugge-anderlecht/",
  "competition": "Jupiler Pro League",
  "matchday": "Speeldag 30",
  "kickoff_at": "2026-04-14T20:45:00+02:00",
  "home_team": "Club Brugge",
  "away_team": "Anderlecht",
  "home_score": 2,
  "away_score": 1,
  "halftime_home_score": 1,
  "halftime_away_score": 0,
  "venue": "Jan Breydel",
  "status": "finished"
}
```

### Live match snapshot

```json
{
  "match_id": "abc123",
  "match_url": "https://sporza.be/nl/matches/brugge-anderlecht/",
  "competition": "Jupiler Pro League",
  "home_team": "Club Brugge",
  "away_team": "Anderlecht",
  "home_score": 1,
  "away_score": 1,
  "current_minute": 62,
  "status": "live",
  "is_live": true,
  "polled_at": "2026-04-15T20:47:12+00:00",
  "events": [
    {"minute": 23, "team": "Club Brugge", "player": "De Ketelaere", "event_type": "goal", "detail": "…"}
  ],
  "lineups": [
    {"team": "Club Brugge", "coach": "Nicky Hayen", "starting_xi": ["Mignolet", "Sabbe"], "bench": ["Jackers"]}
  ]
}
```

### Sentiment report (example output)

```json
{
  "total_articles_scanned": 25,
  "total_mentions": 42,
  "players": [
    {
      "player_name": "De Bruyne",
      "mention_count": 8,
      "avg_sentiment": 0.4512,
      "label": "positive",
      "mentions": [
        {
          "article_url": "https://sporza.be/nl/sport/voetbal/~3333687/",
          "article_title": "De Bruyne schittert in derby",
          "context": "Kevin De Bruyne was opnieuw de beste man op het veld met twee assists.",
          "sentiment_score": 0.65,
          "sentiment_label": "positive"
        }
      ]
    }
  ]
}
```

The terminal summary looks like:

```
============================================================
  PREMIER LEAGUE PLAYER SENTIMENT REPORT
============================================================
  Articles scanned : 25
  Total mentions   : 42
  Players found    : 7
============================================================

  Player                    Mentions  Avg Score      Label
  ------------------------- -------- ---------- ----------
  De Bruyne                        8     +0.451 + positive
  Trossard                         5     +0.220 + positive
  Salah                            4     -0.130 - negative
```

## Design Notes

- **Politeness first.** One `requests.Session`; a single worker; a minimum 1.5s delay between requests (raised automatically if `robots.txt` declares a larger `Crawl-delay`); `Retry-After` honoured on 429.
- **Resilient retries.** `tenacity` with exponential backoff (base 2s, cap 60s, jitter, max 5 attempts) on 5xx/408/425/429 and network errors.
- **Pure parsers.** Every parser takes raw HTML and returns typed `@dataclass` objects. No I/O inside parsers, so they are trivially unit-testable against recorded fixtures.
- **JSON-LD first.** News parsing prefers `application/ld+json` blocks and `<meta property="article:*">` tags over class-name chains. Live match parsing tries `__NEXT_DATA__` first and falls back to HTML heuristics.
- **Dutch dates.** Falls back to parsing Dutch month names (`januari`, `februari`, …) when no machine-readable `datetime` attribute is present.
- **Idempotent writes.** Re-running on the same day overwrites cleanly via `.tmp` + `os.replace`.
- **Graceful SIGINT.** `live` mode installs a signal handler so Ctrl-C finishes the in-flight poll and exits cleanly.
- **Sentiment analysis.** A built-in Dutch sentiment lexicon (~350 words, scored -1 to +1) with negation handling ("niet goed" → flipped score) and intensifier support ("zeer sterk" → boosted). No GPU needed — runs instantly on CPU.
- **Player extraction.** spaCy's `nl_core_news_md` Dutch NER model detects PERSON entities; a curated Premier League player/club set filters to PL context. Works on Sporza's Dutch text out of the box.

## Testing

All tests use recorded HTML fixtures and never touch the network:

```bash
pip install pytest
PYTHONPATH=. pytest -v
```

Fixtures live under `tests/fixtures/` and cover: happy-path article, article with missing fields, empty index, football results with FT and postponed states, and live-match live / finished pages.

## Limitations

- Selectors are heuristic and may need updating if Sporza redesigns its site. Fallbacks are in place (JSON-LD, meta tags, `itemprop`), but not every field is guaranteed.
- The scraper is sequential by design. Do not parallelise it — the site is a public broadcaster and deserves polite load.
- The `live` mode discovery relies on the match-center index resolving to plain HTML. If the page goes fully client-rendered, use the `__NEXT_DATA__` fallback or skip live polling.

## What this scraper will **not** do

- No headless browsers, no bot-detection bypass.
- No scraping of user comments, private pages, or anything behind authentication.
- No storage of personal data beyond the public bylines already printed on the page.
- No publishing or redistribution of scraped content.
