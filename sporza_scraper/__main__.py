"""CLI entry-point for the Sporza scraper."""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from .config import DEFAULT_CSV_SEASON, DEFAULT_OUTPUT_ROOT, DEFAULT_POLL_INTERVAL_SECONDS
from .http_client import ThrottledClient
from .pipeline import Pipeline, daterange, parse_cli_date
from .robots import RobotsPolicy
from .storage import Storage

log = logging.getLogger("sporza_scraper")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sporza_scraper",
        description="Polite webscraper for sporza.be (news, football results, live).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root directory (default: ./data)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and parse but do not write files",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_news = sub.add_parser("news", help="Scrape news articles")
    p_news.add_argument("--limit", type=int, default=10, help="Max articles (default: 10)")
    p_news.add_argument(
        "--since",
        type=parse_cli_date,
        default=None,
        help="Only keep articles published on/after YYYY-MM-DD",
    )

    p_res = sub.add_parser("football-results", help="Scrape football results")
    group = p_res.add_mutually_exclusive_group()
    group.add_argument("--date", type=parse_cli_date, help="Single date YYYY-MM-DD")
    group.add_argument(
        "--range",
        nargs=2,
        metavar=("START", "END"),
        type=parse_cli_date,
        help="Inclusive date range",
    )

    p_live = sub.add_parser("live", help="Poll live match data")
    p_live.add_argument(
        "--poll-interval",
        type=int,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
        help="Seconds between polls (minimum 30, default 60)",
    )
    p_live.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Stop after N minutes (default: run until Ctrl-C)",
    )

    p_sent = sub.add_parser(
        "sentiment",
        help="Analyse Premier League player sentiment from scraped articles",
    )
    p_sent.add_argument(
        "--top",
        type=int,
        default=None,
        help="Only show the top-N most-mentioned players",
    )
    p_sent.add_argument(
        "--all-players",
        action="store_true",
        help="Include ALL detected players, not just Premier League",
    )
    p_sent.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Path to save the full JSON report (optional)",
    )

    p_pl = sub.add_parser("pl-scrape", help="Scrape Premier League articles with enrichment")
    p_pl.add_argument("--limit", type=int, default=50, help="Max articles (default: 50)")
    p_pl.add_argument(
        "--search",
        nargs="*",
        default=None,
        help="Custom search queries (default: PL + Belgian player names)",
    )

    # --- stats subcommand ---
    p_stats = sub.add_parser("stats", help="Download and inspect match statistics")
    p_stats.add_argument(
        "--download",
        action="store_true",
        help="Download the CSV from football-data.co.uk",
    )
    p_stats.add_argument(
        "--season",
        type=str,
        default=DEFAULT_CSV_SEASON,
        help=f"Season code (default: {DEFAULT_CSV_SEASON})",
    )
    p_stats.add_argument(
        "--table",
        action="store_true",
        help="Show current league table",
    )
    p_stats.add_argument(
        "--team",
        type=str,
        default=None,
        help="Show team info",
    )
    p_stats.add_argument(
        "--form",
        type=int,
        default=5,
        help="Number of recent matches for --team form (default: 5)",
    )
    p_stats.add_argument(
        "--h2h",
        nargs=2,
        metavar=("TEAM_A", "TEAM_B"),
        default=None,
        help="Show head-to-head record between two teams",
    )

    # --- benchmark subcommand ---
    p_bench = sub.add_parser(
        "benchmark",
        help="Compare our model's accuracy against betting site odds",
    )
    p_bench.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Path to save the JSON benchmark report (optional)",
    )

    # --- predict subcommand ---
    p_pred = sub.add_parser("predict", help="Generate PL predictions (players or match)")
    p_pred.add_argument(
        "--player",
        type=str,
        default=None,
        help="Filter by player name (substring match)",
    )
    p_pred.add_argument(
        "--club",
        type=str,
        default=None,
        help="Filter by club name (substring match)",
    )
    p_pred.add_argument(
        "--top",
        type=int,
        default=None,
        help="Only show top-N players",
    )
    p_pred.add_argument(
        "--match",
        nargs=2,
        metavar=("HOME", "AWAY"),
        default=None,
        help="Predict outcome of a specific match (e.g. --match Arsenal \"Manchester City\")",
    )
    p_pred.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Path to save the JSON prediction report (optional)",
    )

    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)

    # Offline commands — no network needed (unless --download).
    if args.command == "sentiment":
        _run_sentiment(args)
        return 0
    if args.command == "predict":
        _run_predictions(args)
        return 0
    if args.command == "stats":
        _run_stats(args)
        return 0
    if args.command == "benchmark":
        _run_benchmark(args)
        return 0

    storage = Storage(root=args.out, dry_run=args.dry_run)
    robots = RobotsPolicy()

    with ThrottledClient() as client:
        robots.load(client)
        if robots.crawl_delay:
            client.set_min_interval(robots.crawl_delay)

        pipeline = Pipeline(client=client, robots=robots, storage=storage)

        if args.command == "news":
            pipeline.scrape_news(limit=args.limit, since=args.since)
        elif args.command == "football-results":
            dates: list[date]
            if args.date:
                dates = [args.date]
            elif args.range:
                dates = list(daterange(args.range[0], args.range[1]))
            else:
                from datetime import date as _date
                dates = [_date.today()]
            pipeline.scrape_football_results(dates)
        elif args.command == "live":
            pipeline.scrape_live_matches(
                poll_interval=args.poll_interval,
                duration_minutes=args.duration,
            )
        elif args.command == "pl-scrape":
            pipeline.scrape_premier_league(
                limit=args.limit,
                search_queries=args.search,
            )
        else:  # pragma: no cover
            log.error("unknown command: %s", args.command)
            return 2
    return 0


def _run_sentiment(args: argparse.Namespace) -> None:
    from .analysis import print_summary, run_analysis, save_report

    report = run_analysis(
        data_root=args.out,
        premier_league_only=not args.all_players,
        top_n=args.top,
    )
    print_summary(report)
    if args.report:
        save_report(report, args.report)


def _run_predictions(args: argparse.Namespace) -> None:
    from .predictor import (
        predict_match,
        print_match_prediction,
        print_predictions,
        run_predictions,
        save_predictions,
    )

    # Match prediction mode.
    if args.match:
        home, away = args.match
        mp = predict_match(home, away, data_root=args.out)
        print_match_prediction(mp)
        if args.report:
            save_predictions(mp, args.report)
        return

    # Player prediction mode (default).
    report = run_predictions(
        data_root=args.out,
        player_filter=args.player,
        club_filter=args.club,
        top_n=args.top,
    )
    print_predictions(report)
    if args.report:
        save_predictions(report, args.report)


def _run_stats(args: argparse.Namespace) -> None:
    from .match_stats import (
        MatchDatabase,
        print_head_to_head,
        print_league_table,
        print_team_form,
    )

    # Download if requested.
    if args.download:
        csv_path = MatchDatabase.download(
            season=args.season,
            output_dir=args.out / "stats",
        )
        print(f"Downloaded to {csv_path}")
        if not (args.table or args.team or args.h2h):
            return

    # Load the database.
    stats_dir = args.out / "stats"
    csvs = sorted(stats_dir.glob("E0_*.csv")) if stats_dir.exists() else []
    if not csvs:
        print("No stats CSV found. Run with --download first.")
        return
    db = MatchDatabase(csvs[-1])
    if not db.loaded:
        print("Failed to load stats CSV.")
        return

    # Show requested views.
    if args.table:
        print_league_table(db.league_table())

    if args.team:
        team = db.normalise_team_name(args.team)
        print_team_form(db.team_form(team, last_n=args.form))
        stats = db.team_season_stats(team)
        ou = db.over_under_tendency(team)
        print(f"  Season: {stats['played']}P {stats['wins']}W {stats['draws']}D"
              f" {stats['losses']}L — {stats['goals_per_game']} GPG")
        print(f"  Attack rating:  {db.team_attack_rating(team):.2f}x league avg")
        print(f"  Defence rating: {db.team_defence_rating(team):.2f}x league avg")
        print(f"  Over 2.5: {ou['over_2_5_pct']:.0%}  |  Avg total goals: {ou['avg_total_goals']}")
        print()

    if args.h2h:
        h2h = db.head_to_head(args.h2h[0], args.h2h[1])
        print_head_to_head(h2h)


def _run_benchmark(args: argparse.Namespace) -> None:
    from .benchmark import print_benchmark, run_benchmark, save_benchmark

    report = run_benchmark(data_root=args.out)
    print_benchmark(report)
    if args.report:
        save_benchmark(report, args.report)


if __name__ == "__main__":
    sys.exit(main())
