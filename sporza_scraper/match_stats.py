"""Premier League match statistics from football-data.co.uk CSV.

Downloads, parses, and queries the E0.csv file to extract team-level
and league-level statistics used as statistical baselines for the
hybrid prediction model.
"""
from __future__ import annotations

import csv
import io
import logging
import math
import urllib.request
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final

from .config import (
    DEFAULT_CSV_SEASON,
    DEFAULT_OUTPUT_ROOT,
    FOOTBALL_DATA_CSV_URL_TEMPLATE,
    STATS_SUBDIR,
    TEAM_NAME_ALIASES,
    USER_AGENT,
)

log = logging.getLogger(__name__)

# Columns we parse as int (goals, shots, cards, etc.).
_INT_COLS: Final[tuple[str, ...]] = (
    "FTHG", "FTAG", "HTHG", "HTAG",
    "HS", "AS", "HST", "AST",
    "HF", "AF", "HC", "AC",
    "HY", "AY", "HR", "AR",
)

# Columns we parse as float (odds).
_FLOAT_COLS: Final[tuple[str, ...]] = (
    "B365H", "B365D", "B365A",
    "AvgH", "AvgD", "AvgA",
    "MaxH", "MaxD", "MaxA",
)


# ====================================================================
# MatchDatabase
# ====================================================================


class MatchDatabase:
    """Load and query Premier League match statistics from CSV."""

    def __init__(self, csv_path: Path | None = None) -> None:
        self._matches: list[dict[str, Any]] = []
        self._teams: set[str] = set()
        self._league_table_cache: list[dict[str, Any]] | None = None
        if csv_path and csv_path.exists():
            self.load(csv_path)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, csv_path: Path) -> None:
        """Parse a football-data.co.uk CSV into internal match list."""
        text = csv_path.read_text(encoding="utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        self._matches = []
        self._league_table_cache = None
        for row in reader:
            if not row.get("HomeTeam") or not row.get("AwayTeam"):
                continue
            match = self._parse_row(row)
            if match:
                self._matches.append(match)
                self._teams.add(match["home_team"])
                self._teams.add(match["away_team"])
        log.info("Loaded %d matches, %d teams", len(self._matches), len(self._teams))

    @staticmethod
    def download(
        season: str = DEFAULT_CSV_SEASON,
        output_dir: Path = DEFAULT_OUTPUT_ROOT / STATS_SUBDIR,
    ) -> Path:
        """Download E0.csv for the given season."""
        url = FOOTBALL_DATA_CSV_URL_TEMPLATE.format(season=season)
        output_dir.mkdir(parents=True, exist_ok=True)
        dest = output_dir / f"E0_{season}.csv"
        log.info("Downloading %s → %s", url, dest)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        dest.write_bytes(data)
        log.info("Downloaded %d bytes to %s", len(data), dest)
        return dest

    @property
    def loaded(self) -> bool:
        return len(self._matches) > 0

    @property
    def teams(self) -> list[str]:
        return sorted(self._teams)

    @property
    def match_count(self) -> int:
        return len(self._matches)

    # ------------------------------------------------------------------
    # Team-level queries
    # ------------------------------------------------------------------

    def team_form(self, team: str, last_n: int = 5) -> dict[str, Any]:
        """Return last N matches: W/D/L record, goals scored/conceded."""
        canon = self.normalise_team_name(team)
        matches = self._team_matches(canon)[-last_n:]
        w = d = l = gf = ga = cs = 0
        results: list[str] = []
        for m in matches:
            scored, conceded = self._goals_for_against(m, canon)
            gf += scored
            ga += conceded
            if scored > conceded:
                w += 1
                results.append("W")
            elif scored == conceded:
                d += 1
                results.append("D")
            else:
                l += 1
                results.append("L")
            if conceded == 0:
                cs += 1
        played = len(matches)
        return {
            "team": canon,
            "played": played,
            "wins": w, "draws": d, "losses": l,
            "goals_for": gf, "goals_against": ga,
            "points": w * 3 + d,
            "clean_sheets": cs,
            "goals_per_game": round(gf / played, 2) if played else 0.0,
            "conceded_per_game": round(ga / played, 2) if played else 0.0,
            "results": results,
            "form_string": "".join(results),
        }

    def team_season_stats(self, team: str) -> dict[str, Any]:
        """Full season aggregates with home/away split."""
        canon = self.normalise_team_name(team)
        matches = self._team_matches(canon)
        if not matches:
            return {"team": canon, "played": 0}

        total = {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0,
                 "shots": 0, "shots_on_target": 0, "corners": 0, "fouls": 0}
        home = {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0}
        away = {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0}

        for m in matches:
            is_home = m["home_team"] == canon
            scored, conceded = self._goals_for_against(m, canon)
            bucket = home if is_home else away

            total["P"] += 1
            bucket["P"] += 1
            total["GF"] += scored
            total["GA"] += conceded
            bucket["GF"] += scored
            bucket["GA"] += conceded

            if scored > conceded:
                total["W"] += 1; bucket["W"] += 1
            elif scored == conceded:
                total["D"] += 1; bucket["D"] += 1
            else:
                total["L"] += 1; bucket["L"] += 1

            total["shots"] += (m.get("HS", 0) if is_home else m.get("AS", 0))
            total["shots_on_target"] += (m.get("HST", 0) if is_home else m.get("AST", 0))
            total["corners"] += (m.get("HC", 0) if is_home else m.get("AC", 0))
            total["fouls"] += (m.get("HF", 0) if is_home else m.get("AF", 0))

        p = total["P"]
        return {
            "team": canon,
            "played": p,
            "wins": total["W"], "draws": total["D"], "losses": total["L"],
            "goals_for": total["GF"], "goals_against": total["GA"],
            "goal_difference": total["GF"] - total["GA"],
            "points": total["W"] * 3 + total["D"],
            "goals_per_game": round(total["GF"] / p, 2) if p else 0.0,
            "conceded_per_game": round(total["GA"] / p, 2) if p else 0.0,
            "avg_shots": round(total["shots"] / p, 1) if p else 0.0,
            "avg_shots_on_target": round(total["shots_on_target"] / p, 1) if p else 0.0,
            "avg_corners": round(total["corners"] / p, 1) if p else 0.0,
            "avg_fouls": round(total["fouls"] / p, 1) if p else 0.0,
            "home": home,
            "away": away,
        }

    def head_to_head(self, team_a: str, team_b: str, last_n: int = 5) -> dict[str, Any]:
        """H2H record between two teams."""
        a = self.normalise_team_name(team_a)
        b = self.normalise_team_name(team_b)
        h2h = [
            m for m in self._matches
            if {m["home_team"], m["away_team"]} == {a, b}
        ][-last_n:]

        a_wins = draws = b_wins = a_goals = b_goals = 0
        for m in h2h:
            ga, gb = self._goals_for_against(m, a)
            a_goals += ga
            b_goals += gb
            if ga > gb:
                a_wins += 1
            elif ga == gb:
                draws += 1
            else:
                b_wins += 1

        return {
            "team_a": a, "team_b": b,
            "matches_played": len(h2h),
            "a_wins": a_wins, "draws": draws, "b_wins": b_wins,
            "a_goals": a_goals, "b_goals": b_goals,
        }

    def league_table(self) -> list[dict[str, Any]]:
        """Compute the current league standings from match results."""
        if self._league_table_cache is not None:
            return self._league_table_cache

        standings: dict[str, dict[str, int]] = {}
        for m in self._matches:
            for team_key, is_home in [("home_team", True), ("away_team", False)]:
                team = m[team_key]
                if team not in standings:
                    standings[team] = {
                        "P": 0, "W": 0, "D": 0, "L": 0,
                        "GF": 0, "GA": 0, "Pts": 0,
                    }
                s = standings[team]
                scored, conceded = self._goals_for_against(m, team)
                s["P"] += 1
                s["GF"] += scored
                s["GA"] += conceded
                if scored > conceded:
                    s["W"] += 1; s["Pts"] += 3
                elif scored == conceded:
                    s["D"] += 1; s["Pts"] += 1
                else:
                    s["L"] += 1

        table = [
            {"team": t, **s, "GD": s["GF"] - s["GA"]}
            for t, s in standings.items()
        ]
        table.sort(key=lambda r: (-r["Pts"], -r["GD"], -r["GF"], r["team"]))
        for i, row in enumerate(table, 1):
            row["pos"] = i

        self._league_table_cache = table
        return table

    # ------------------------------------------------------------------
    # Derived analytics
    # ------------------------------------------------------------------

    def team_attack_rating(self, team: str) -> float:
        """Goals scored per game relative to league average. >1.0 = above avg."""
        canon = self.normalise_team_name(team)
        stats = self.team_season_stats(canon)
        if not stats["played"]:
            return 1.0
        league_avg = self._league_avg_goals_per_game()
        if league_avg == 0:
            return 1.0
        return stats["goals_per_game"] / league_avg

    def team_defence_rating(self, team: str) -> float:
        """Goals conceded per game relative to league average. <1.0 = above avg."""
        canon = self.normalise_team_name(team)
        stats = self.team_season_stats(canon)
        if not stats["played"]:
            return 1.0
        league_avg = self._league_avg_goals_per_game()
        if league_avg == 0:
            return 1.0
        return stats["conceded_per_game"] / league_avg

    def implied_probability(self, home_team: str, away_team: str) -> dict[str, float]:
        """Use average market odds from the most recent H2H meeting."""
        h = self.normalise_team_name(home_team)
        a = self.normalise_team_name(away_team)

        # Find most recent match where h was home vs a.
        recent = None
        for m in reversed(self._matches):
            if m["home_team"] == h and m["away_team"] == a:
                recent = m
                break

        if recent and recent.get("AvgH") and recent.get("AvgD") and recent.get("AvgA"):
            return _odds_to_probs(recent["AvgH"], recent["AvgD"], recent["AvgA"])

        # Fallback: use league-wide home/draw/away rates.
        total = len(self._matches) or 1
        hw = sum(1 for m in self._matches if m.get("FTR") == "H")
        dr = sum(1 for m in self._matches if m.get("FTR") == "D")
        aw = total - hw - dr
        return {
            "home_win": round(hw / total, 3),
            "draw": round(dr / total, 3),
            "away_win": round(aw / total, 3),
        }

    def over_under_tendency(self, team: str) -> dict[str, Any]:
        """What % of this team's matches go over/under 2.5 goals."""
        canon = self.normalise_team_name(team)
        matches = self._team_matches(canon)
        if not matches:
            return {"team": canon, "played": 0, "over_2_5_pct": 0.5, "under_2_5_pct": 0.5}
        over = sum(1 for m in matches if m["FTHG"] + m["FTAG"] > 2)
        p = len(matches)
        return {
            "team": canon,
            "played": p,
            "over_2_5_pct": round(over / p, 3),
            "under_2_5_pct": round((p - over) / p, 3),
            "avg_total_goals": round(sum(m["FTHG"] + m["FTAG"] for m in matches) / p, 2),
        }

    # ------------------------------------------------------------------
    # League-wide averages
    # ------------------------------------------------------------------

    def league_avg_home_goals(self) -> float:
        if not self._matches:
            return 1.5
        return sum(m["FTHG"] for m in self._matches) / len(self._matches)

    def league_avg_away_goals(self) -> float:
        if not self._matches:
            return 1.2
        return sum(m["FTAG"] for m in self._matches) / len(self._matches)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def normalise_team_name(self, name: str) -> str:
        """Map variant names to canonical form."""
        if name in self._teams:
            return name
        canonical = TEAM_NAME_ALIASES.get(name)
        if canonical:
            # The canonical form might itself differ from CSV names.
            # Check if the canonical is in our known teams.
            if canonical in self._teams:
                return canonical
            # Try reverse: maybe the CSV uses the short form.
            for csv_name, canon in TEAM_NAME_ALIASES.items():
                if canon == canonical and csv_name in self._teams:
                    return csv_name
            return canonical
        # Case-insensitive fallback.
        lower = name.lower()
        for t in self._teams:
            if t.lower() == lower:
                return t
        for alias, canon in TEAM_NAME_ALIASES.items():
            if alias.lower() == lower:
                return canon
        return name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_row(self, row: dict[str, str]) -> dict[str, Any] | None:
        parsed: dict[str, Any] = {}
        parsed["home_team"] = row.get("HomeTeam", "").strip()
        parsed["away_team"] = row.get("AwayTeam", "").strip()
        parsed["FTR"] = row.get("FTR", "").strip()

        # Date
        date_str = row.get("Date", "").strip()
        if date_str:
            try:
                parsed["date"] = datetime.strptime(date_str, "%d/%m/%Y").date()
            except ValueError:
                try:
                    parsed["date"] = datetime.strptime(date_str, "%d/%m/%y").date()
                except ValueError:
                    parsed["date"] = None
        else:
            parsed["date"] = None

        # Int columns
        for col in _INT_COLS:
            val = row.get(col, "").strip()
            try:
                parsed[col] = int(val) if val else 0
            except ValueError:
                parsed[col] = 0

        # Float columns
        for col in _FLOAT_COLS:
            val = row.get(col, "").strip()
            try:
                parsed[col] = float(val) if val else 0.0
            except ValueError:
                parsed[col] = 0.0

        if not parsed["home_team"] or not parsed["away_team"]:
            return None
        return parsed

    def _team_matches(self, team: str) -> list[dict[str, Any]]:
        """All matches involving a team, sorted by date."""
        return [
            m for m in self._matches
            if m["home_team"] == team or m["away_team"] == team
        ]

    @staticmethod
    def _goals_for_against(match: dict[str, Any], team: str) -> tuple[int, int]:
        if match["home_team"] == team:
            return match["FTHG"], match["FTAG"]
        return match["FTAG"], match["FTHG"]

    def _league_avg_goals_per_game(self) -> float:
        if not self._matches:
            return 1.35
        total = sum(m["FTHG"] + m["FTAG"] for m in self._matches)
        return total / (len(self._matches) * 2)


# ====================================================================
# Poisson model helpers (used by predictor.py)
# ====================================================================


def poisson_pmf(k: int, lam: float) -> float:
    """P(X = k) for Poisson with rate lam."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def build_goal_matrix(
    lambda_home: float, lambda_away: float, max_goals: int = 6,
) -> list[list[float]]:
    """Build a (max_goals × max_goals) probability matrix.

    ``matrix[i][j]`` = P(home scores i AND away scores j).
    """
    matrix: list[list[float]] = []
    for i in range(max_goals):
        row: list[float] = []
        for j in range(max_goals):
            row.append(poisson_pmf(i, lambda_home) * poisson_pmf(j, lambda_away))
        matrix.append(row)
    return matrix


def outcomes_from_matrix(matrix: list[list[float]]) -> dict[str, float]:
    """Sum the goal matrix into H/D/A probabilities, O/U 2.5, BTTS."""
    n = len(matrix)
    home_win = draw = away_win = 0.0
    over_2_5 = 0.0
    btts_yes = 0.0

    for i in range(n):
        for j in range(n):
            p = matrix[i][j]
            if i > j:
                home_win += p
            elif i == j:
                draw += p
            else:
                away_win += p
            if i + j > 2:
                over_2_5 += p
            if i >= 1 and j >= 1:
                btts_yes += p

    total = home_win + draw + away_win
    if total > 0:
        home_win /= total
        draw /= total
        away_win /= total

    return {
        "home_win": round(home_win, 4),
        "draw": round(draw, 4),
        "away_win": round(away_win, 4),
        "over_2_5": round(over_2_5, 4),
        "btts": round(btts_yes, 4),
    }


# ====================================================================
# Odds conversion
# ====================================================================


def _odds_to_probs(h_odds: float, d_odds: float, a_odds: float) -> dict[str, float]:
    """Convert decimal odds to implied probabilities (normalised)."""
    if h_odds <= 0 or d_odds <= 0 or a_odds <= 0:
        return {"home_win": 0.33, "draw": 0.33, "away_win": 0.34}
    raw_h = 1.0 / h_odds
    raw_d = 1.0 / d_odds
    raw_a = 1.0 / a_odds
    total = raw_h + raw_d + raw_a
    return {
        "home_win": round(raw_h / total, 4),
        "draw": round(raw_d / total, 4),
        "away_win": round(raw_a / total, 4),
    }


# ====================================================================
# Terminal output helpers
# ====================================================================


def print_league_table(table: list[dict[str, Any]]) -> None:
    print(f"\n{'Pos':>3}  {'Team':<25} {'P':>3} {'W':>3} {'D':>3} {'L':>3}"
          f" {'GF':>3} {'GA':>3} {'GD':>4} {'Pts':>4}")
    print("─" * 70)
    for r in table:
        print(f"{r['pos']:>3}  {r['team']:<25} {r['P']:>3} {r['W']:>3} {r['D']:>3} {r['L']:>3}"
              f" {r['GF']:>3} {r['GA']:>3} {r['GD']:>+4} {r['Pts']:>4}")
    print()


def print_team_form(form: dict[str, Any]) -> None:
    print(f"\n  {form['team']} — Last {form['played']} matches")
    print(f"  {'─' * 40}")
    print(f"  Record: W{form['wins']}-D{form['draws']}-L{form['losses']}"
          f"  ({form['points']} pts)")
    print(f"  Goals:  {form['goals_for']} scored, {form['goals_against']} conceded"
          f"  ({form['goals_per_game']} GPG)")
    print(f"  Clean sheets: {form['clean_sheets']}")
    print(f"  Form: {form['form_string']}")
    print()


def print_head_to_head(h2h: dict[str, Any]) -> None:
    a, b = h2h["team_a"], h2h["team_b"]
    print(f"\n  {a} vs {b} — Last {h2h['matches_played']} meetings")
    print(f"  {'─' * 40}")
    print(f"  {a}: {h2h['a_wins']} wins  |  Draws: {h2h['draws']}  |  {b}: {h2h['b_wins']} wins")
    print(f"  Goals: {a} {h2h['a_goals']} — {h2h['b_goals']} {b}")
    print()
