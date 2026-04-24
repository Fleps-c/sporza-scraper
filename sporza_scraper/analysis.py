"""Sentiment-per-player analysis orchestrator.

Reads previously scraped news article JSON files, extracts player mentions
using spaCy NER, scores the surrounding text using the Dutch sentiment
lexicon, and produces a per-player summary.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .player_extractor import extract_player_mentions
from .sentiment import SentimentResult, analyse_text

log = logging.getLogger(__name__)


@dataclass(slots=True)
class PlayerMentionSentiment:
    """Sentiment for a single mention of a player in one article."""
    article_url: str
    article_title: str | None
    context: str
    sentiment_score: float
    sentiment_label: str


@dataclass(slots=True)
class PlayerSentimentSummary:
    """Aggregated sentiment for one player across all articles."""
    player_name: str
    mention_count: int
    avg_sentiment: float
    label: str  # positive / negative / neutral
    mentions: list[PlayerMentionSentiment] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnalysisReport:
    """Full analysis report covering all discovered players."""
    total_articles_scanned: int
    total_mentions: int
    players: list[PlayerSentimentSummary] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_analysis(
    data_root: Path,
    *,
    premier_league_only: bool = True,
    top_n: int | None = None,
) -> AnalysisReport:
    """Scan all news JSON files in ``data_root`` and produce a sentiment report.

    Parameters
    ----------
    data_root:
        The root output directory that contains ``news/YYYY/MM/DD/*.json``.
    premier_league_only:
        If True, only extract names linked to the Premier League.
    top_n:
        If set, only include the top-N most-mentioned players in the report.
    """
    news_root = data_root / "news"
    if not news_root.exists():
        log.warning("No news data found at %s — run 'news' scrape first", news_root)
        return AnalysisReport(total_articles_scanned=0, total_mentions=0)

    json_files = sorted(news_root.rglob("*.json"))
    log.info("Found %d news article JSON files to analyse", len(json_files))

    # Accumulator: player name → list of (article_url, article_title, context, sentiment)
    player_data: dict[str, list[PlayerMentionSentiment]] = {}
    total_articles = 0

    for path in json_files:
        try:
            article = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Skipping %s: %s", path, exc)
            continue

        paragraphs = article.get("body_paragraphs") or []
        if not paragraphs:
            continue
        total_articles += 1

        url = article.get("url", str(path))
        title = article.get("title")

        extraction = extract_player_mentions(
            paragraphs, premier_league_only=premier_league_only
        )
        if not extraction.mentions:
            continue

        log.info(
            "Article '%s': %d player mentions (%s)",
            title or path.name,
            len(extraction.mentions),
            ", ".join(extraction.unique_names[:5]),
        )

        for mention in extraction.mentions:
            sentiment = analyse_text(mention.context)
            entry = PlayerMentionSentiment(
                article_url=url,
                article_title=title,
                context=mention.context[:300],
                sentiment_score=sentiment.score,
                sentiment_label=sentiment.label,
            )
            player_data.setdefault(mention.name, []).append(entry)

    # Build per-player summaries.
    summaries: list[PlayerSentimentSummary] = []
    for name, entries in player_data.items():
        scores = [e.sentiment_score for e in entries]
        avg = sum(scores) / len(scores) if scores else 0.0
        label = "positive" if avg > 0.05 else "negative" if avg < -0.05 else "neutral"
        summaries.append(
            PlayerSentimentSummary(
                player_name=name,
                mention_count=len(entries),
                avg_sentiment=round(avg, 4),
                label=label,
                mentions=entries,
            )
        )

    # Sort by mention count (descending), then alphabetical.
    summaries.sort(key=lambda s: (-s.mention_count, s.player_name))
    if top_n:
        summaries = summaries[:top_n]

    total_mentions = sum(s.mention_count for s in summaries)
    log.info(
        "Analysis complete: %d articles, %d players, %d total mentions",
        total_articles,
        len(summaries),
        total_mentions,
    )

    return AnalysisReport(
        total_articles_scanned=total_articles,
        total_mentions=total_mentions,
        players=summaries,
    )


def save_report(report: AnalysisReport, output_path: Path) -> Path:
    """Write the analysis report as pretty-printed JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        report.to_dict(),
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    )
    output_path.write_text(data, encoding="utf-8")
    log.info("Sentiment report written to %s", output_path)
    return output_path


def print_summary(report: AnalysisReport) -> None:
    """Print a human-readable summary to stdout."""
    print(f"\n{'=' * 60}")
    print(f"  PREMIER LEAGUE PLAYER SENTIMENT REPORT")
    print(f"{'=' * 60}")
    print(f"  Articles scanned : {report.total_articles_scanned}")
    print(f"  Total mentions   : {report.total_mentions}")
    print(f"  Players found    : {len(report.players)}")
    print(f"{'=' * 60}\n")

    if not report.players:
        print("  No Premier League player mentions found.")
        print("  Try scraping more articles first:\n")
        print("    python -m sporza_scraper news --limit 30\n")
        return

    # Table header
    print(f"  {'Player':<25} {'Mentions':>8} {'Avg Score':>10} {'Label':>10}")
    print(f"  {'-' * 25} {'-' * 8} {'-' * 10} {'-' * 10}")

    for p in report.players:
        emoji = "+" if p.label == "positive" else "-" if p.label == "negative" else "~"
        print(f"  {p.player_name:<25} {p.mention_count:>8} {p.avg_sentiment:>+10.3f} {emoji:>1} {p.label}")

    print()
