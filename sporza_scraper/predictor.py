"""Hybrid prediction engine: statistical baselines + Sporza signal modifiers.

Combines quantitative match data from football-data.co.uk with qualitative
performance signals extracted from Sporza articles to produce:
1. **Match predictions** — Poisson-based outcome probabilities.
2. **Player predictions** — data-driven scoring/assist/start likelihoods.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .config import DEFAULT_CSV_SEASON, DEFAULT_OUTPUT_ROOT, PREMIER_LEAGUE_SUBDIR, STATS_SUBDIR
from .match_stats import (
    MatchDatabase,
    build_goal_matrix,
    outcomes_from_matrix,
)
from .player_extractor import PLAYER_TO_CLUB
from .utils import parse_iso_datetime

log = logging.getLogger(__name__)


# ====================================================================
# Dataclasses
# ====================================================================


@dataclass(slots=True)
class SignalEntry:
    signal_type: str
    score: float
    confidence: float
    evidence: str
    article_date: date | None
    article_url: str


@dataclass(slots=True)
class Prediction:
    likely_to_score: float
    likely_to_assist: float
    likely_to_start: float
    confidence: str  # high / medium / low
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PlayerProfile:
    player_name: str
    club: str | None
    total_mentions: int
    date_range: str
    signal_summary: dict[str, dict[str, Any]] = field(default_factory=dict)
    overall_form_score: float = 0.0
    trend: str = "stable"
    recent_7d_score: float = 0.0
    prediction: Prediction | None = None
    _signals: list[SignalEntry] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "player": self.player_name,
            "club": self.club,
            "total_mentions": self.total_mentions,
            "date_range": self.date_range,
            "signal_summary": self.signal_summary,
            "overall_form_score": self.overall_form_score,
            "trend": self.trend,
            "recent_7d_score": self.recent_7d_score,
            "prediction": self.prediction.to_dict() if self.prediction else None,
        }


@dataclass(slots=True)
class PredictionReport:
    total_articles_scanned: int
    date_range: str
    players: list[PlayerProfile] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_articles_scanned": self.total_articles_scanned,
            "date_range": self.date_range,
            "players": [p.to_dict() for p in self.players],
        }


@dataclass(slots=True)
class MatchPrediction:
    home_team: str
    away_team: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    expected_home_goals: float
    expected_away_goals: float
    over_2_5_prob: float
    btts_prob: float
    confidence: str
    key_factors: list[str] = field(default_factory=list)
    sporza_modifiers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FixturePredictionReport:
    fixtures: list[MatchPrediction] = field(default_factory=list)
    league_table: list[dict[str, Any]] = field(default_factory=list)
    generated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixtures": [f.to_dict() for f in self.fixtures],
            "league_table": self.league_table,
            "generated_at": self.generated_at,
        }


# ====================================================================
# Load helpers
# ====================================================================


def _load_match_db(data_root: Path) -> MatchDatabase | None:
    """Try to load the stats CSV, return None if not available."""
    stats_dir = data_root / STATS_SUBDIR
    if not stats_dir.exists():
        return None
    csvs = sorted(stats_dir.glob("E0_*.csv"))
    if not csvs:
        return None
    db = MatchDatabase(csvs[-1])
    return db if db.loaded else None


def _load_sporza_signals(data_root: Path) -> tuple[
    dict[str, list[SignalEntry]],
    dict[str, int],
    list[date],
    int,
    dict[str, list[SignalEntry]],
]:
    """Read PL article JSONs. Returns (player_signals, player_mentions,
    article_dates, article_count, team_signals)."""
    pl_root = data_root / PREMIER_LEAGUE_SUBDIR
    if not pl_root.exists():
        return {}, {}, [], 0, {}

    json_files = sorted(pl_root.rglob("*.json"))
    player_signals: dict[str, list[SignalEntry]] = {}
    player_mentions: dict[str, int] = {}
    team_signals: dict[str, list[SignalEntry]] = {}
    article_dates: list[date] = []

    for path in json_files:
        try:
            article = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        art_date = _parse_article_date(article)
        if art_date:
            article_dates.append(art_date)
        art_url = article.get("url", str(path))

        enrichment = article.get("pl_enrichment", {})
        signals_raw = enrichment.get("performance_signals", [])
        players_raw = enrichment.get("players_mentioned", [])

        for pm in players_raw:
            if isinstance(pm, dict):
                name = pm.get("name", "")
                count = pm.get("mention_count", 1)
                player_mentions[name] = player_mentions.get(name, 0) + count

        for sig in signals_raw:
            if not isinstance(sig, dict):
                continue
            player = sig.get("player", "")
            if not player:
                continue
            entry = SignalEntry(
                signal_type=sig.get("signal", sig.get("signal_type", "unknown")),
                score=float(sig.get("score", 0)),
                confidence=float(sig.get("confidence", 0.5)),
                evidence=sig.get("evidence", ""),
                article_date=art_date,
                article_url=art_url,
            )
            player_signals.setdefault(player, []).append(entry)

            # Aggregate team-level signals.
            club = PLAYER_TO_CLUB.get(player)
            if club:
                team_signals.setdefault(club, []).append(entry)

    return player_signals, player_mentions, article_dates, len(json_files), team_signals


# ====================================================================
# Match prediction (Poisson model + Sporza modifiers)
# ====================================================================


def predict_match(
    home_team: str,
    away_team: str,
    data_root: Path = DEFAULT_OUTPUT_ROOT,
) -> MatchPrediction:
    """Predict outcome of a specific fixture."""
    db = _load_match_db(data_root)
    if not db:
        log.warning("No match stats CSV found — using league-average defaults")

    _, _, _, article_count, team_signals = _load_sporza_signals(data_root)
    today = date.today()

    # Normalise names.
    if db:
        h = db.normalise_team_name(home_team)
        a = db.normalise_team_name(away_team)
    else:
        h, a = home_team, away_team

    key_factors: list[str] = []
    sporza_mods: list[str] = []

    # --- Statistical baselines ---
    if db and db.loaded:
        h_attack = db.team_attack_rating(h)
        h_defence = db.team_defence_rating(h)
        a_attack = db.team_attack_rating(a)
        a_defence = db.team_defence_rating(a)

        avg_home_goals = db.league_avg_home_goals()
        avg_away_goals = db.league_avg_away_goals()

        h_form = db.team_form(h, last_n=5)
        a_form = db.team_form(a, last_n=5)

        key_factors.append(
            f"{h}: {h_form['form_string']} in last 5 ({h_form['goals_per_game']} GPG)"
        )
        key_factors.append(
            f"{a}: {a_form['form_string']} in last 5 ({a_form['goals_per_game']} GPG)"
        )
        key_factors.append(f"{h} attack rating {h_attack:.2f}x league avg")
        key_factors.append(f"{a} defence concedes {a_defence:.2f}x league avg")

        # Market odds.
        implied = db.implied_probability(h, a)
        key_factors.append(f"Market odds imply {implied['home_win']:.0%} {h} win")

        # Home advantage factor (typically ~1.05-1.15).
        home_advantage = 1.0
        if avg_home_goals > avg_away_goals:
            home_advantage = avg_home_goals / ((avg_home_goals + avg_away_goals) / 2)
            home_advantage = min(home_advantage, 1.25)  # cap

        lambda_home = avg_home_goals * h_attack * a_defence * home_advantage
        lambda_away = avg_away_goals * a_attack * h_defence

        match_count = db.match_count
    else:
        # No CSV: use sensible PL defaults.
        lambda_home = 1.55
        lambda_away = 1.20
        match_count = 0
        key_factors.append("No match CSV loaded — using PL average defaults")

    # --- Sporza signal modifiers ---
    h_modifier = _team_sporza_modifier(h, team_signals, today)
    a_modifier = _team_sporza_modifier(a, team_signals, today)

    if abs(h_modifier - 1.0) > 0.01:
        adj = (h_modifier - 1.0) * 100
        sporza_mods.append(f"{h} Sporza form modifier: {adj:+.0f}%")
    if abs(a_modifier - 1.0) > 0.01:
        adj = (a_modifier - 1.0) * 100
        sporza_mods.append(f"{a} Sporza form modifier: {adj:+.0f}%")

    # Add player-specific injury modifiers for the two teams only.
    seen_evidence: set[str] = set()
    for team_name in (h, a):
        for sig in team_signals.get(team_name, []):
            if sig.signal_type in ("injury_doubt", "injury_confirmed"):
                snippet = sig.evidence[:60]
                if snippet not in seen_evidence:
                    seen_evidence.add(snippet)
                    label = "injury" if sig.signal_type == "injury_confirmed" else "doubt"
                    sporza_mods.append(f"{team_name}: {snippet} [{label}]")

    lambda_home *= h_modifier
    lambda_away *= a_modifier

    # Clamp lambdas to sane range.
    lambda_home = max(0.3, min(lambda_home, 4.5))
    lambda_away = max(0.2, min(lambda_away, 4.0))

    # --- Poisson model ---
    matrix = build_goal_matrix(lambda_home, lambda_away, max_goals=7)
    outcomes = outcomes_from_matrix(matrix)

    # Confidence.
    if match_count >= 20 and article_count >= 5:
        confidence = "high"
    elif match_count >= 10 or article_count >= 3:
        confidence = "medium"
    else:
        confidence = "low"

    return MatchPrediction(
        home_team=h,
        away_team=a,
        home_win_prob=outcomes["home_win"],
        draw_prob=outcomes["draw"],
        away_win_prob=outcomes["away_win"],
        expected_home_goals=round(lambda_home, 2),
        expected_away_goals=round(lambda_away, 2),
        over_2_5_prob=outcomes["over_2_5"],
        btts_prob=outcomes["btts"],
        confidence=confidence,
        key_factors=key_factors,
        sporza_modifiers=sporza_mods,
    )


def _team_sporza_modifier(
    team: str,
    team_signals: dict[str, list[SignalEntry]],
    today: date,
) -> float:
    """Compute a multiplier (around 1.0) from Sporza signals for a team.

    Positive signals push > 1.0, negative signals push < 1.0.
    """
    signals = team_signals.get(team, [])
    if not signals:
        return 1.0

    cutoff = today - timedelta(days=14)
    recent = [s for s in signals if s.article_date and s.article_date >= cutoff]
    if not recent:
        return 1.0

    # Weighted average of signal scores.
    total_w = 0.0
    total_s = 0.0
    for s in recent:
        w = s.confidence
        total_w += w
        total_s += s.score * w

    if total_w == 0:
        return 1.0

    avg_score = total_s / total_w  # range roughly -1 to +1
    # Convert to a multiplier: avg_score of +0.5 → 1.10, -0.5 → 0.90
    modifier = 1.0 + (avg_score * 0.20)
    return max(0.75, min(modifier, 1.30))


# ====================================================================
# Player predictions (hybrid: stats baseline + Sporza modifiers)
# ====================================================================


def run_predictions(
    data_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    player_filter: str | None = None,
    club_filter: str | None = None,
    top_n: int | None = None,
) -> PredictionReport:
    """Build player profiles from PL article JSONs and generate predictions."""
    db = _load_match_db(data_root)
    player_signals, player_mentions, article_dates, article_count, _ = (
        _load_sporza_signals(data_root)
    )

    if not player_signals:
        log.info("No performance signals found in PL data")
        return PredictionReport(
            total_articles_scanned=article_count,
            date_range=_date_range_str(article_dates),
        )

    today = date.today()
    profiles: list[PlayerProfile] = []

    for name, signals in player_signals.items():
        club = PLAYER_TO_CLUB.get(name)
        if player_filter and player_filter.lower() not in name.lower():
            continue
        if club_filter and club and club_filter.lower() not in club.lower():
            continue

        profile = _build_profile(
            name, club, signals, player_mentions.get(name, 0), today, db,
        )
        profiles.append(profile)

    profiles.sort(key=lambda p: (-p.total_mentions, -p.overall_form_score))
    if top_n:
        profiles = profiles[:top_n]

    return PredictionReport(
        total_articles_scanned=article_count,
        date_range=_date_range_str(article_dates),
        players=profiles,
    )


# ====================================================================
# Profile building
# ====================================================================


def _build_profile(
    name: str,
    club: str | None,
    signals: list[SignalEntry],
    mention_count: int,
    today: date,
    db: MatchDatabase | None,
) -> PlayerProfile:
    dates = [s.article_date for s in signals if s.article_date]
    date_range = _date_range_str(dates)

    by_type: dict[str, list[float]] = {}
    for s in signals:
        by_type.setdefault(s.signal_type, []).append(s.score)
    signal_summary = {
        st: {"count": len(scores), "avg_score": round(sum(scores) / len(scores), 4)}
        for st, scores in by_type.items()
    }

    overall = _recency_weighted_avg(signals, today, window_days=30)
    recent_7d = _recency_weighted_avg(signals, today, window_days=7)

    score_30d = _recency_weighted_avg(signals, today, window_days=30)
    diff = recent_7d - score_30d
    trend = "improving" if diff >= 0.1 else "declining" if diff <= -0.1 else "stable"

    prediction = _compute_prediction(signal_summary, mention_count, trend, recent_7d, club, db)

    return PlayerProfile(
        player_name=name,
        club=club,
        total_mentions=mention_count,
        date_range=date_range,
        signal_summary=signal_summary,
        overall_form_score=round(overall, 4),
        trend=trend,
        recent_7d_score=round(recent_7d, 4),
        prediction=prediction,
        _signals=signals,
    )


def _recency_weighted_avg(
    signals: list[SignalEntry], today: date, window_days: int,
) -> float:
    cutoff = today - timedelta(days=window_days)
    half_window = window_days / 2
    total_weight = 0.0
    total_score = 0.0
    for s in signals:
        if s.article_date and s.article_date < cutoff:
            continue
        days_ago = (today - s.article_date).days if s.article_date else window_days
        weight = 2.0 if days_ago <= half_window else 1.0
        weight *= s.confidence
        total_weight += weight
        total_score += s.score * weight
    if total_weight == 0:
        return 0.0
    return total_score / total_weight


def _compute_prediction(
    summary: dict[str, dict[str, Any]],
    mention_count: int,
    trend: str,
    recent_score: float,
    club: str | None,
    db: MatchDatabase | None,
) -> Prediction:
    def _avg(signal_type: str) -> float:
        entry = summary.get(signal_type)
        return entry["avg_score"] if entry else 0.0

    def _count(signal_type: str) -> int:
        entry = summary.get(signal_type)
        return entry["count"] if entry else 0

    # --- Data-driven baselines ---
    team_gpg = 1.35  # PL average fallback
    if db and db.loaded and club:
        stats = db.team_season_stats(club)
        if stats.get("played", 0) > 0:
            team_gpg = stats["goals_per_game"]

    base_score_prob = team_gpg / 11  # ~0.12-0.18
    base_assist_prob = team_gpg / 14  # slightly lower

    # Form bonus from stats (team's last-5 PPG deviation from season avg).
    team_form_bonus = 0.0
    if db and db.loaded and club:
        form = db.team_form(club, last_n=5)
        season = db.team_season_stats(club)
        if season.get("played", 0) > 0:
            season_ppg = season["points"] / season["played"]
            form_ppg = form["points"] / form["played"] if form["played"] else season_ppg
            team_form_bonus = (form_ppg - season_ppg) / 3  # normalised

    # --- Sporza modifiers ---
    goal_avg = _avg("goal_threat")
    form_avg = _avg("positive_form")
    injury_doubt_avg = abs(_avg("injury_doubt"))
    assist_avg = _avg("assist_threat")

    likely_score = (
        base_score_prob
        + (goal_avg * 0.20)
        + (form_avg * 0.08)
        - (injury_doubt_avg * 0.30)
        + (team_form_bonus * 0.05)
    )
    likely_score = max(0.0, min(1.0, likely_score))

    likely_assist = (
        base_assist_prob
        + (assist_avg * 0.25)
        + (form_avg * 0.06)
        - (injury_doubt_avg * 0.20)
        + (team_form_bonus * 0.03)
    )
    likely_assist = max(0.0, min(1.0, likely_assist))

    injury_conf_count = _count("injury_confirmed")
    injury_doubt_count = _count("injury_doubt")
    likely_start = 1.0 - (injury_conf_count * 0.9) - (injury_doubt_count * 0.15)
    likely_start = max(0.0, min(1.0, likely_start))

    # Confidence.
    has_stats = db is not None and db.loaded
    if mention_count >= 10 and has_stats:
        confidence = "high"
    elif mention_count >= 5 or has_stats:
        confidence = "medium"
    else:
        confidence = "low"

    # Reasoning.
    parts: list[str] = []
    if has_stats and club:
        parts.append(f"Team scores {team_gpg:.1f} GPG (base {base_score_prob:.0%})")
    if form_avg > 0.3:
        parts.append(f"Positive form signals (avg {form_avg:+.2f})")
    elif form_avg < -0.1:
        parts.append(f"Poor form signals (avg {form_avg:+.2f})")
    goal_cnt = _count("goal_threat")
    if goal_cnt:
        parts.append(f"{goal_cnt} goal-related mention{'s' if goal_cnt != 1 else ''}")
    assist_cnt = _count("assist_threat")
    if assist_cnt:
        parts.append(f"{assist_cnt} assist signal{'s' if assist_cnt != 1 else ''}")
    if injury_conf_count:
        parts.append(f"Injury confirmed ({injury_conf_count}x)")
    elif injury_doubt_count:
        parts.append(f"Minor injury doubt ({injury_doubt_count}x)")
    if trend == "improving":
        parts.append("Form trending upward")
    elif trend == "declining":
        parts.append("Form trending downward")
    if team_form_bonus > 0.1:
        parts.append("Team in above-average recent form")
    elif team_form_bonus < -0.1:
        parts.append("Team in below-average recent form")
    reasoning = ". ".join(parts) + "." if parts else "Insufficient data for detailed reasoning."

    return Prediction(
        likely_to_score=round(likely_score, 2),
        likely_to_assist=round(likely_assist, 2),
        likely_to_start=round(likely_start, 2),
        confidence=confidence,
        reasoning=reasoning,
    )


# ====================================================================
# Output
# ====================================================================


def save_predictions(report: PredictionReport | MatchPrediction, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(report.to_dict(), indent=2, ensure_ascii=False, sort_keys=True)
    output_path.write_text(data, encoding="utf-8")
    log.info("Prediction report saved to %s", output_path)
    return output_path


def print_predictions(report: PredictionReport) -> None:
    print(f"\n{'=' * 60}")
    print(f"  PREMIER LEAGUE PERFORMANCE PREDICTIONS")
    print(f"  Based on {report.total_articles_scanned} Sporza articles ({report.date_range})")
    print(f"{'=' * 60}")

    if not report.players:
        print("\n  No player data found.")
        print("  Try scraping PL articles first:\n")
        print("    python -m sporza_scraper pl-scrape --limit 50\n")
        return

    by_club: dict[str, list[PlayerProfile]] = {}
    for p in report.players:
        club = p.club or "Unknown"
        by_club.setdefault(club, []).append(p)

    for club, players in by_club.items():
        print(f"\n  {club.upper()}")
        print(f"  {'─' * 50}")
        for p in players:
            pred = p.prediction
            if not pred:
                continue
            trend_sym = "▲" if p.trend == "improving" else "▼" if p.trend == "declining" else "─"
            conf = pred.confidence[:3].upper()
            print(
                f"  {p.player_name:<20} Form: {p.overall_form_score:+.2f} {trend_sym}  "
                f"Score: {pred.likely_to_score:3.0%}  "
                f"Assist: {pred.likely_to_assist:3.0%}  "
                f"Start: {pred.likely_to_start:3.0%}  "
                f"[{conf}]"
            )

    scorers = sorted(
        [p for p in report.players if p.prediction],
        key=lambda p: -(p.prediction.likely_to_score if p.prediction else 0),
    )[:5]
    if scorers:
        print(f"\n{'=' * 60}")
        print(f"  HOTLIST — Top 5 most likely to score")
        for i, p in enumerate(scorers, 1):
            pred = p.prediction
            if not pred:
                continue
            club_short = (p.club or "?")[:15]
            trend_sym = "▲" if p.trend == "improving" else "▼" if p.trend == "declining" else "─"
            goal_cnt = p.signal_summary.get("goal_threat", {}).get("count", 0)
            print(
                f"  {i}. {p.player_name} ({club_short})"
                f"{'':>{30 - len(p.player_name) - len(club_short)}}"
                f"{pred.likely_to_score:3.0%}  "
                f"[form {trend_sym}, {goal_cnt} goal signal{'s' if goal_cnt != 1 else ''}]"
            )
    print(f"{'=' * 60}\n")


def print_match_prediction(mp: MatchPrediction) -> None:
    """Pretty-print a match prediction."""
    bar_width = 20

    def _bar(pct: float) -> str:
        filled = round(pct * bar_width)
        return "█" * filled + "░" * (bar_width - filled)

    print(f"\n  MATCH PREDICTION: {mp.home_team} vs {mp.away_team}")
    print(f"  {'═' * 50}")
    print()
    print(f"  {mp.home_team + ' (H)':<22} {mp.home_win_prob:5.1%}   {_bar(mp.home_win_prob)}")
    print(f"  {'Draw':<22} {mp.draw_prob:5.1%}   {_bar(mp.draw_prob)}")
    print(f"  {mp.away_team + ' (A)':<22} {mp.away_win_prob:5.1%}   {_bar(mp.away_win_prob)}")
    print()
    print(f"  Expected score: {mp.home_team} {mp.expected_home_goals:.1f}"
          f" — {mp.expected_away_goals:.1f} {mp.away_team}")
    print(f"  Over 2.5 goals: {mp.over_2_5_prob:.0%}  |  BTTS: {mp.btts_prob:.0%}")

    if mp.key_factors:
        print(f"\n  KEY FACTORS:")
        for f in mp.key_factors:
            print(f"  • {f}")

    if mp.sporza_modifiers:
        print(f"\n  SPORZA SIGNAL ADJUSTMENTS:")
        for m in mp.sporza_modifiers:
            print(f"  • {m}")

    print(f"\n  Confidence: {mp.confidence.upper()}")
    print(f"  {'═' * 50}\n")


# ====================================================================
# Helpers
# ====================================================================


def _parse_article_date(article: dict[str, Any]) -> date | None:
    raw = article.get("published_at")
    if isinstance(raw, str):
        dt = parse_iso_datetime(raw)
        if dt:
            return dt.date()
    return None


def _date_range_str(dates: list[date]) -> str:
    if not dates:
        return "no dates"
    earliest = min(dates)
    latest = max(dates)
    return f"{earliest.isoformat()} to {latest.isoformat()}"
