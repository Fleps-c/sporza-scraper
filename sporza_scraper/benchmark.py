"""Accuracy benchmark: compare our Poisson baseline model against betting sites.

For each completed match in the football-data.co.uk CSV, this module:
1. Builds our model using only data from **prior** matches (no future leakage).
2. Compares our predicted probabilities to Bet365 and market average odds.
3. Measures who picks the correct outcome more often and how well-calibrated
   the probabilities are (Brier score).

Usage via CLI::

    python -m sporza_scraper benchmark
    python -m sporza_scraper benchmark --report benchmark.json
"""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import DEFAULT_OUTPUT_ROOT, STATS_SUBDIR
from .match_stats import MatchDatabase, build_goal_matrix, outcomes_from_matrix

log = logging.getLogger(__name__)


# ====================================================================
# Dataclasses
# ====================================================================


@dataclass(slots=True)
class SourceResult:
    """Accuracy metrics for a single prediction source."""

    name: str
    correct: int = 0
    total: int = 0
    brier_sum: float = 0.0

    @property
    def accuracy_pct(self) -> float:
        return round(self.correct / self.total * 100, 1) if self.total else 0.0

    @property
    def brier_score(self) -> float:
        return round(self.brier_sum / self.total, 4) if self.total else 0.0


@dataclass(slots=True)
class BenchmarkReport:
    """Full comparison report."""

    match_count: int = 0
    predicted_count: int = 0
    our_model: SourceResult = field(default_factory=lambda: SourceResult("Our Poisson Model"))
    bet365: SourceResult = field(default_factory=lambda: SourceResult("Bet365"))
    market_avg: SourceResult = field(default_factory=lambda: SourceResult("Market Average"))
    agreement_pct: float = 0.0
    our_edge_count: int = 0
    their_edge_count: int = 0
    our_edge_examples: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_count": self.match_count,
            "predicted_count": self.predicted_count,
            "our_model": {
                "correct": self.our_model.correct,
                "total": self.our_model.total,
                "accuracy_pct": self.our_model.accuracy_pct,
                "brier_score": self.our_model.brier_score,
            },
            "bet365": {
                "correct": self.bet365.correct,
                "total": self.bet365.total,
                "accuracy_pct": self.bet365.accuracy_pct,
                "brier_score": self.bet365.brier_score,
            },
            "market_avg": {
                "correct": self.market_avg.correct,
                "total": self.market_avg.total,
                "accuracy_pct": self.market_avg.accuracy_pct,
                "brier_score": self.market_avg.brier_score,
            },
            "agreement_with_bet365_pct": self.agreement_pct,
            "our_edge_count": self.our_edge_count,
            "their_edge_count": self.their_edge_count,
            "our_edge_examples": self.our_edge_examples,
        }


# ====================================================================
# Helpers
# ====================================================================


def _poisson_predict(
    lam_h: float, lam_a: float,
) -> dict[str, float]:
    """Predict H/D/A probabilities from expected goals."""
    matrix = build_goal_matrix(lam_h, lam_a, max_goals=7)
    outcomes = outcomes_from_matrix(matrix)
    return {
        "H": outcomes["home_win"],
        "D": outcomes["draw"],
        "A": outcomes["away_win"],
    }


def _odds_to_probs(
    h_odds: float, d_odds: float, a_odds: float,
) -> dict[str, float] | None:
    """Convert decimal odds to normalised probabilities."""
    if h_odds <= 0 or d_odds <= 0 or a_odds <= 0:
        return None
    raw_h, raw_d, raw_a = 1 / h_odds, 1 / d_odds, 1 / a_odds
    total = raw_h + raw_d + raw_a
    return {"H": raw_h / total, "D": raw_d / total, "A": raw_a / total}


def _brier(probs: dict[str, float], actual: str) -> float:
    """Brier score for a single prediction (lower = better)."""
    return sum(
        (probs[o] - (1 if o == actual else 0)) ** 2
        for o in ("H", "D", "A")
    )


# ====================================================================
# Main benchmark
# ====================================================================


def run_benchmark(data_root: Path = DEFAULT_OUTPUT_ROOT) -> BenchmarkReport:
    """Run the accuracy comparison using the stats CSV.

    The model is built **progressively**: for each match, only data from
    earlier matches is used, so there is no information leakage.
    """
    stats_dir = data_root / STATS_SUBDIR
    if not stats_dir.exists():
        log.warning("No stats directory at %s", stats_dir)
        return BenchmarkReport()

    csvs = sorted(stats_dir.glob("E0_*.csv"))
    if not csvs:
        log.warning("No E0_*.csv found in %s", stats_dir)
        return BenchmarkReport()

    db = MatchDatabase(csvs[-1])
    if not db.loaded:
        log.warning("Failed to load CSV")
        return BenchmarkReport()

    matches = db._matches  # noqa: SLF001 – internal access for benchmark
    report = BenchmarkReport(match_count=len(matches))

    # Progressive accumulators (only past data used for each prediction).
    team_gf: dict[str, int] = defaultdict(int)
    team_ga: dict[str, int] = defaultdict(int)
    team_mp: dict[str, int] = defaultdict(int)
    total_hg = total_ag = total_matches = 0

    agree_count = 0
    compare_count = 0

    for m in matches:
        h, a = m["home_team"], m["away_team"]
        ftr = m.get("FTR", "")
        hg, ag = m.get("FTHG", 0), m.get("FTAG", 0)

        # Need enough prior data before we start predicting.
        if team_mp[h] < 5 or team_mp[a] < 5 or total_matches < 20:
            team_gf[h] += hg; team_ga[h] += ag
            team_gf[a] += ag; team_ga[a] += hg
            team_mp[h] += 1; team_mp[a] += 1
            total_hg += hg; total_ag += ag
            total_matches += 1
            continue

        # ── Our model (from prior data only) ──
        avg_hg = total_hg / total_matches
        avg_ag = total_ag / total_matches
        league_avg = (total_hg + total_ag) / (total_matches * 2)

        h_att = (team_gf[h] / team_mp[h]) / league_avg if league_avg else 1.0
        h_def = (team_ga[h] / team_mp[h]) / league_avg if league_avg else 1.0
        a_att = (team_gf[a] / team_mp[a]) / league_avg if league_avg else 1.0
        a_def = (team_ga[a] / team_mp[a]) / league_avg if league_avg else 1.0

        home_adv = min(avg_hg / ((avg_hg + avg_ag) / 2), 1.25)

        lam_h = max(0.3, min(avg_hg * h_att * a_def * home_adv, 4.5))
        lam_a = max(0.2, min(avg_ag * a_att * h_def, 4.0))

        our_probs = _poisson_predict(lam_h, lam_a)
        our_pick = max(our_probs, key=our_probs.get)

        # ── Bookmaker odds ──
        b365_probs = _odds_to_probs(
            m.get("B365H", 0), m.get("B365D", 0), m.get("B365A", 0),
        )
        avg_probs = _odds_to_probs(
            m.get("AvgH", 0), m.get("AvgD", 0), m.get("AvgA", 0),
        )

        # ── Score ──
        report.our_model.total += 1
        if our_pick == ftr:
            report.our_model.correct += 1
        report.our_model.brier_sum += _brier(our_probs, ftr)

        if b365_probs:
            b365_pick = max(b365_probs, key=b365_probs.get)
            report.bet365.total += 1
            if b365_pick == ftr:
                report.bet365.correct += 1
            report.bet365.brier_sum += _brier(b365_probs, ftr)

            compare_count += 1
            if our_pick == b365_pick:
                agree_count += 1

            if our_pick == ftr and b365_pick != ftr:
                report.our_edge_count += 1
                if len(report.our_edge_examples) < 5:
                    report.our_edge_examples.append({
                        "match": f"{h} vs {a}",
                        "actual": ftr,
                        "our_pick": our_pick,
                        "bet365_pick": b365_pick,
                    })
            elif b365_pick == ftr and our_pick != ftr:
                report.their_edge_count += 1

        if avg_probs:
            avg_pick = max(avg_probs, key=avg_probs.get)
            report.market_avg.total += 1
            if avg_pick == ftr:
                report.market_avg.correct += 1
            report.market_avg.brier_sum += _brier(avg_probs, ftr)

        # ── Update stats AFTER prediction ──
        team_gf[h] += hg; team_ga[h] += ag
        team_gf[a] += ag; team_ga[a] += hg
        team_mp[h] += 1; team_mp[a] += 1
        total_hg += hg; total_ag += ag
        total_matches += 1

    report.predicted_count = report.our_model.total
    report.agreement_pct = (
        round(agree_count / compare_count * 100, 1) if compare_count else 0.0
    )

    return report


# ====================================================================
# Output
# ====================================================================


def save_benchmark(report: BenchmarkReport, output_path: Path) -> Path:
    """Write benchmark results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
    output_path.write_text(data, encoding="utf-8")
    log.info("Benchmark saved to %s", output_path)
    return output_path


def print_benchmark(report: BenchmarkReport) -> None:
    """Pretty-print benchmark results to stdout."""
    print(f"\n{'=' * 60}")
    print("  ACCURACY BENCHMARK: Our Model vs Betting Sites")
    print(f"  {report.match_count} matches in CSV, {report.predicted_count} predicted")
    print(f"{'=' * 60}")

    for src in (report.our_model, report.bet365, report.market_avg):
        if not src.total:
            continue
        print(f"\n  {src.name}:")
        print(f"    Correct picks : {src.correct} / {src.total}  ({src.accuracy_pct}%)")
        print(f"    Brier score   : {src.brier_score}  (lower = better)")

    print(f"\n{'─' * 60}")
    print(f"  Agreement with Bet365: {report.agreement_pct}%")
    print(f"  We beat Bet365:        {report.our_edge_count} matches")
    print(f"  Bet365 beat us:        {report.their_edge_count} matches")

    if report.our_edge_examples:
        print(f"\n  Examples where our model outperformed Bet365:")
        for ex in report.our_edge_examples:
            print(
                f"    {ex['match']:<30s}  Actual: {ex['actual']}"
                f"  |  Ours: {ex['our_pick']}  Bet365: {ex['bet365_pick']}"
            )

    print(f"{'=' * 60}\n")
