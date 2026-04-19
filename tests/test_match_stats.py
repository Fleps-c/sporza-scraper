"""Unit tests for match_stats.py — CSV parsing, league table, form, Poisson."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from sporza_scraper.match_stats import (
    MatchDatabase,
    build_goal_matrix,
    outcomes_from_matrix,
    poisson_pmf,
)


# ── 10-row fixture CSV ────────────────────────────────────────────────
FIXTURE_CSV = textwrap.dedent("""\
Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,Referee,HS,AS,HST,AST,HF,AF,HC,AC,HY,AY,HR,AR,B365H,B365D,B365A,AvgH,AvgD,AvgA
E0,16/08/2024,20:00,Man United,Fulham,1,0,H,1,0,H,Smith,12,8,5,3,10,11,6,3,2,1,0,0,1.50,4.50,7.00,1.55,4.40,6.80
E0,17/08/2024,15:00,Arsenal,Wolves,2,0,H,1,0,H,Jones,18,4,8,1,8,12,7,2,1,3,0,0,1.20,7.00,15.00,1.22,6.80,14.50
E0,17/08/2024,15:00,Everton,Brighton,0,3,A,0,1,A,Brown,6,15,2,7,14,9,3,8,3,2,0,0,3.00,3.50,2.30,3.10,3.40,2.25
E0,17/08/2024,15:00,Newcastle,Southampton,1,0,H,0,0,D,Taylor,14,6,6,2,9,13,5,4,2,2,0,0,1.40,5.00,8.00,1.42,4.90,7.80
E0,18/08/2024,14:00,Nott'm Forest,Bournemouth,1,1,D,0,1,A,Clark,10,11,4,5,11,10,4,5,1,1,0,0,2.50,3.40,2.90,2.55,3.35,2.85
E0,24/08/2024,15:00,Brighton,Man United,2,1,A,1,0,H,Smith,16,10,7,4,8,12,9,3,1,3,0,0,2.10,3.60,3.40,2.15,3.50,3.35
E0,24/08/2024,15:00,Wolves,Arsenal,0,2,A,0,1,A,Jones,5,14,2,6,13,7,2,8,2,1,0,0,5.50,4.00,1.60,5.40,3.90,1.62
E0,24/08/2024,15:00,Fulham,Leicester,2,1,H,1,1,D,White,11,9,5,3,10,11,6,4,2,2,0,0,2.00,3.50,3.80,2.05,3.45,3.70
E0,25/08/2024,14:00,Southampton,Nott'm Forest,0,1,A,0,0,D,Clark,7,9,3,4,12,10,3,5,3,1,0,0,2.30,3.40,3.20,2.35,3.35,3.15
E0,31/08/2024,15:00,Arsenal,Brighton,1,1,D,0,0,D,Taylor,15,12,6,5,7,9,8,6,0,2,0,0,1.55,4.00,6.50,1.58,3.90,6.30
""")


@pytest.fixture()
def csv_path(tmp_path: Path) -> Path:
    p = tmp_path / "E0_2425.csv"
    p.write_text(FIXTURE_CSV, encoding="utf-8")
    return p


@pytest.fixture()
def db(csv_path: Path) -> MatchDatabase:
    return MatchDatabase(csv_path)


# ── Loading ───────────────────────────────────────────────────────────

class TestLoading:
    def test_loads_all_matches(self, db: MatchDatabase) -> None:
        assert db.loaded
        assert db.match_count == 10

    def test_teams_populated(self, db: MatchDatabase) -> None:
        teams = db.teams
        assert "Arsenal" in teams
        # CSV short name kept as-is.
        assert "Man United" in teams or "Manchester United" in teams

    def test_empty_csv_not_loaded(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.csv"
        p.write_text("HomeTeam,AwayTeam,FTHG,FTAG\n", encoding="utf-8")
        db = MatchDatabase(p)
        assert not db.loaded

    def test_nonexistent_csv(self, tmp_path: Path) -> None:
        db = MatchDatabase(tmp_path / "nope.csv")
        assert not db.loaded


# ── League table ──────────────────────────────────────────────────────

class TestLeagueTable:
    def test_table_has_all_teams(self, db: MatchDatabase) -> None:
        table = db.league_table()
        team_names = [r["team"] for r in table]
        assert len(team_names) == len(set(team_names)), "No duplicates"
        # Arsenal played 3 matches in the fixture.
        ars = next(r for r in table if r["team"] == "Arsenal")
        assert ars["P"] == 3

    def test_table_sorted_by_points(self, db: MatchDatabase) -> None:
        table = db.league_table()
        points = [r["Pts"] for r in table]
        assert points == sorted(points, reverse=True)

    def test_table_cached(self, db: MatchDatabase) -> None:
        t1 = db.league_table()
        t2 = db.league_table()
        assert t1 is t2  # same object, cached

    def test_arsenal_results(self, db: MatchDatabase) -> None:
        table = db.league_table()
        ars = next(r for r in table if r["team"] == "Arsenal")
        # Arsenal: beat Wolves 2-0, beat Wolves(A) 2-0, drew Brighton 1-1 = 2W 1D 0L
        assert ars["W"] == 2
        assert ars["D"] == 1
        assert ars["L"] == 0
        assert ars["Pts"] == 7


# ── Team form ─────────────────────────────────────────────────────────

class TestTeamForm:
    def test_form_last_5(self, db: MatchDatabase) -> None:
        form = db.team_form("Arsenal", last_n=5)
        assert form["team"] == "Arsenal"
        assert form["played"] <= 5

    def test_form_record_sums(self, db: MatchDatabase) -> None:
        form = db.team_form("Arsenal", last_n=10)
        assert form["wins"] + form["draws"] + form["losses"] == form["played"]

    def test_goals_per_game(self, db: MatchDatabase) -> None:
        form = db.team_form("Arsenal", last_n=10)
        assert form["goals_per_game"] >= 0
        expected = round(form["goals_for"] / form["played"], 2) if form["played"] else 0
        assert form["goals_per_game"] == expected


# ── Season stats ──────────────────────────────────────────────────────

class TestSeasonStats:
    def test_stats_structure(self, db: MatchDatabase) -> None:
        stats = db.team_season_stats("Arsenal")
        assert "played" in stats
        assert "goals_per_game" in stats
        assert "home" in stats and "away" in stats

    def test_home_away_split(self, db: MatchDatabase) -> None:
        stats = db.team_season_stats("Arsenal")
        assert stats["home"]["P"] + stats["away"]["P"] == stats["played"]

    def test_unknown_team_returns_zero(self, db: MatchDatabase) -> None:
        stats = db.team_season_stats("Nonexistent FC")
        assert stats["played"] == 0


# ── Head to head ──────────────────────────────────────────────────────

class TestH2H:
    def test_h2h_structure(self, db: MatchDatabase) -> None:
        h2h = db.head_to_head("Arsenal", "Brighton")
        assert h2h["team_a"] == "Arsenal"
        assert "matches_played" in h2h

    def test_h2h_goals_consistent(self, db: MatchDatabase) -> None:
        h2h = db.head_to_head("Arsenal", "Brighton")
        total = h2h["a_wins"] + h2h["draws"] + h2h["b_wins"]
        assert total == h2h["matches_played"]


# ── Attack/defence ratings ────────────────────────────────────────────

class TestRatings:
    def test_attack_rating_positive(self, db: MatchDatabase) -> None:
        rating = db.team_attack_rating("Arsenal")
        assert rating > 0

    def test_defence_rating_positive(self, db: MatchDatabase) -> None:
        rating = db.team_defence_rating("Arsenal")
        assert rating > 0


# ── Over/under tendency ──────────────────────────────────────────────

class TestOverUnder:
    def test_structure(self, db: MatchDatabase) -> None:
        ou = db.over_under_tendency("Arsenal")
        assert abs(ou["over_2_5_pct"] + ou["under_2_5_pct"] - 1.0) < 0.01


# ── Name normalisation ───────────────────────────────────────────────

class TestNormalisation:
    def test_csv_name_identity(self, db: MatchDatabase) -> None:
        assert db.normalise_team_name("Arsenal") == "Arsenal"

    def test_alias_resolution(self, db: MatchDatabase) -> None:
        # "Nott'm Forest" is in the CSV; alias maps it.
        result = db.normalise_team_name("Nottingham Forest")
        assert result in ("Nott'm Forest", "Nottingham Forest")

    def test_case_insensitive(self, db: MatchDatabase) -> None:
        result = db.normalise_team_name("arsenal")
        assert result == "Arsenal"


# ── Implied probability ──────────────────────────────────────────────

class TestImpliedProbability:
    def test_probabilities_sum_to_one(self, db: MatchDatabase) -> None:
        prob = db.implied_probability("Arsenal", "Brighton")
        total = prob["home_win"] + prob["draw"] + prob["away_win"]
        assert abs(total - 1.0) < 0.01

    def test_fallback_when_no_h2h(self, db: MatchDatabase) -> None:
        # Teams that never met with one as home.
        prob = db.implied_probability("Everton", "Newcastle")
        total = prob["home_win"] + prob["draw"] + prob["away_win"]
        assert abs(total - 1.0) < 0.01


# ── Poisson model ────────────────────────────────────────────────────

class TestPoisson:
    def test_poisson_pmf_zero(self) -> None:
        # P(X=0 | λ=1.5) = e^{-1.5}
        import math
        expected = math.exp(-1.5)
        assert abs(poisson_pmf(0, 1.5) - expected) < 1e-10

    def test_poisson_pmf_sums_near_one(self) -> None:
        total = sum(poisson_pmf(k, 2.0) for k in range(20))
        assert abs(total - 1.0) < 1e-6

    def test_poisson_pmf_zero_lambda(self) -> None:
        assert poisson_pmf(0, 0.0) == 1.0
        assert poisson_pmf(1, 0.0) == 0.0

    def test_goal_matrix_shape(self) -> None:
        matrix = build_goal_matrix(1.5, 1.2, max_goals=7)
        assert len(matrix) == 7
        assert all(len(row) == 7 for row in matrix)

    def test_goal_matrix_sums_near_one(self) -> None:
        matrix = build_goal_matrix(1.5, 1.2, max_goals=7)
        total = sum(cell for row in matrix for cell in row)
        assert abs(total - 1.0) < 0.01  # small truncation at max_goals

    def test_outcomes_sum_to_one(self) -> None:
        matrix = build_goal_matrix(1.5, 1.2, max_goals=7)
        out = outcomes_from_matrix(matrix)
        total = out["home_win"] + out["draw"] + out["away_win"]
        assert abs(total - 1.0) < 0.01

    def test_home_advantage_reflected(self) -> None:
        # Higher home lambda → higher home win prob.
        matrix = build_goal_matrix(2.5, 0.8, max_goals=7)
        out = outcomes_from_matrix(matrix)
        assert out["home_win"] > out["away_win"]

    def test_high_lambda_means_over_2_5(self) -> None:
        matrix = build_goal_matrix(2.0, 2.0, max_goals=7)
        out = outcomes_from_matrix(matrix)
        assert out["over_2_5"] > 0.5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
