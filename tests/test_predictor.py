"""Tests for sporza_scraper.predictor — player + match (hybrid) predictions."""
from __future__ import annotations

import json
import tempfile
import textwrap
from datetime import date, timedelta
from pathlib import Path

import pytest

from sporza_scraper.match_stats import MatchDatabase
from sporza_scraper.predictor import (
    MatchPrediction,
    Prediction,
    PredictionReport,
    PlayerProfile,
    predict_match,
    print_match_prediction,
    run_predictions,
    save_predictions,
    print_predictions,
)

# ---------------------------------------------------------------------------
# Fixture CSV (10 rows, same as test_match_stats.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_pl_article(
    slug: str,
    published_at: str,
    players: list[dict],
    signals: list[dict],
) -> dict:
    """Create a minimal PL article JSON for testing."""
    return {
        "url": f"https://sporza.be/nl/voetbal/~123/{slug}",
        "slug": slug,
        "title": f"Test article: {slug}",
        "published_at": published_at,
        "body_paragraphs": ["Test body paragraph."],
        "pl_enrichment": {
            "article_type": "general coverage",
            "players_mentioned": players,
            "unique_players": [p["name"] for p in players],
            "performance_signals": signals,
        },
    }


def _write_articles(tmp_dir: Path, articles: list[dict]) -> None:
    pl_root = tmp_dir / "premier-league" / "2026" / "04" / "15"
    pl_root.mkdir(parents=True, exist_ok=True)
    for i, article in enumerate(articles):
        path = pl_root / f"article-{i}.json"
        path.write_text(json.dumps(article), encoding="utf-8")


# ---------------------------------------------------------------------------
# run_predictions
# ---------------------------------------------------------------------------


class TestRunPredictions:
    def test_no_data_directory(self, tmp_path: Path):
        report = run_predictions(tmp_path)
        assert report.total_articles_scanned == 0
        assert report.players == []

    def test_empty_directory(self, tmp_path: Path):
        (tmp_path / "premier-league").mkdir()
        report = run_predictions(tmp_path)
        assert report.total_articles_scanned == 0

    def test_basic_prediction(self, tmp_path: Path):
        today = date.today().isoformat()
        articles = [
            _make_pl_article(
                "haaland-scores",
                today,
                players=[{"name": "Haaland", "mention_count": 5}],
                signals=[
                    {
                        "signal_type": "goal_threat",
                        "player": "Haaland",
                        "score": 0.8,
                        "confidence": 0.95,
                        "evidence": "Haaland scoorde twee keer.",
                    },
                    {
                        "signal_type": "positive_form",
                        "player": "Haaland",
                        "score": 0.75,
                        "confidence": 0.9,
                        "evidence": "Haaland was uitstekend.",
                    },
                ],
            ),
        ]
        _write_articles(tmp_path, articles)
        report = run_predictions(tmp_path)
        assert report.total_articles_scanned == 1
        assert len(report.players) >= 1
        haaland = next((p for p in report.players if p.player_name == "Haaland"), None)
        assert haaland is not None
        assert haaland.prediction is not None
        assert haaland.prediction.likely_to_score > 0.15  # above baseline

    def test_player_filter(self, tmp_path: Path):
        today = date.today().isoformat()
        articles = [
            _make_pl_article(
                "multi-player",
                today,
                players=[
                    {"name": "Haaland", "mention_count": 3},
                    {"name": "Saka", "mention_count": 2},
                ],
                signals=[
                    {"signal_type": "goal_threat", "player": "Haaland",
                     "score": 0.8, "confidence": 0.9, "evidence": "test"},
                    {"signal_type": "positive_form", "player": "Saka",
                     "score": 0.6, "confidence": 0.8, "evidence": "test"},
                ],
            ),
        ]
        _write_articles(tmp_path, articles)
        report = run_predictions(tmp_path, player_filter="Haaland")
        names = [p.player_name for p in report.players]
        assert "Haaland" in names
        assert "Saka" not in names

    def test_club_filter(self, tmp_path: Path):
        today = date.today().isoformat()
        articles = [
            _make_pl_article(
                "club-test",
                today,
                players=[
                    {"name": "Haaland", "mention_count": 3},
                    {"name": "Saka", "mention_count": 2},
                ],
                signals=[
                    {"signal_type": "goal_threat", "player": "Haaland",
                     "score": 0.8, "confidence": 0.9, "evidence": "test"},
                    {"signal_type": "positive_form", "player": "Saka",
                     "score": 0.6, "confidence": 0.8, "evidence": "test"},
                ],
            ),
        ]
        _write_articles(tmp_path, articles)
        # Haaland → Manchester City, Saka → Arsenal
        report = run_predictions(tmp_path, club_filter="Arsenal")
        names = [p.player_name for p in report.players]
        assert "Saka" in names
        assert "Haaland" not in names

    def test_top_n(self, tmp_path: Path):
        today = date.today().isoformat()
        articles = [
            _make_pl_article(
                "many-players",
                today,
                players=[
                    {"name": "Haaland", "mention_count": 10},
                    {"name": "Saka", "mention_count": 8},
                    {"name": "Salah", "mention_count": 6},
                ],
                signals=[
                    {"signal_type": "goal_threat", "player": "Haaland",
                     "score": 0.8, "confidence": 0.9, "evidence": "test"},
                    {"signal_type": "positive_form", "player": "Saka",
                     "score": 0.6, "confidence": 0.8, "evidence": "test"},
                    {"signal_type": "goal_threat", "player": "Salah",
                     "score": 0.7, "confidence": 0.85, "evidence": "test"},
                ],
            ),
        ]
        _write_articles(tmp_path, articles)
        report = run_predictions(tmp_path, top_n=2)
        assert len(report.players) == 2

    def test_injury_reduces_start_likelihood(self, tmp_path: Path):
        today = date.today().isoformat()
        articles = [
            _make_pl_article(
                "injury-test",
                today,
                players=[{"name": "Rodri", "mention_count": 3}],
                signals=[
                    {
                        "signal_type": "injury_confirmed",
                        "player": "Rodri",
                        "score": -0.9,
                        "confidence": 0.95,
                        "evidence": "Rodri is definitief out.",
                    },
                ],
            ),
        ]
        _write_articles(tmp_path, articles)
        report = run_predictions(tmp_path)
        rodri = next((p for p in report.players if p.player_name == "Rodri"), None)
        assert rodri is not None
        assert rodri.prediction is not None
        assert rodri.prediction.likely_to_start < 0.5  # significantly reduced


# ---------------------------------------------------------------------------
# Prediction dataclass
# ---------------------------------------------------------------------------


class TestPrediction:
    def test_to_dict(self):
        pred = Prediction(
            likely_to_score=0.35,
            likely_to_assist=0.20,
            likely_to_start=0.90,
            confidence="medium",
            reasoning="Good form.",
        )
        d = pred.to_dict()
        assert d["likely_to_score"] == 0.35
        assert d["confidence"] == "medium"


# ---------------------------------------------------------------------------
# save_predictions & print_predictions
# ---------------------------------------------------------------------------


class TestSavePredictions:
    def test_save_creates_json(self, tmp_path: Path):
        report = PredictionReport(
            total_articles_scanned=5,
            date_range="2026-04-01 to 2026-04-15",
            players=[],
        )
        out_path = tmp_path / "predictions.json"
        result = save_predictions(report, out_path)
        assert result == out_path
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["total_articles_scanned"] == 5


class TestPrintPredictions:
    def test_print_empty_report(self, capsys):
        report = PredictionReport(
            total_articles_scanned=0,
            date_range="",
            players=[],
        )
        print_predictions(report)
        captured = capsys.readouterr()
        assert "No player data found" in captured.out

    def test_print_with_players(self, capsys):
        profile = PlayerProfile(
            player_name="Haaland",
            club="Manchester City",
            total_mentions=10,
            date_range="2026-04-01 to 2026-04-15",
            overall_form_score=0.65,
            trend="improving",
            recent_7d_score=0.70,
            prediction=Prediction(
                likely_to_score=0.45,
                likely_to_assist=0.20,
                likely_to_start=0.95,
                confidence="high",
                reasoning="Strong form.",
            ),
        )
        report = PredictionReport(
            total_articles_scanned=20,
            date_range="2026-04-01 to 2026-04-15",
            players=[profile],
        )
        print_predictions(report)
        captured = capsys.readouterr()
        assert "Haaland" in captured.out
        assert "MANCHESTER CITY" in captured.out
        assert "HOTLIST" in captured.out


# ---------------------------------------------------------------------------
# Helper: set up data dir with stats CSV + PL articles
# ---------------------------------------------------------------------------


def _setup_hybrid_data(tmp_path: Path) -> Path:
    """Create data dir with stats CSV and a PL article."""
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()
    (stats_dir / "E0_2425.csv").write_text(FIXTURE_CSV, encoding="utf-8")

    today = date.today()
    pl_dir = (
        tmp_path / "premier-league"
        / f"{today.year:04d}" / f"{today.month:02d}" / f"{today.day:02d}"
    )
    pl_dir.mkdir(parents=True)

    article = {
        "url": "https://sporza.be/nl/test-hybrid-123456/",
        "title": "Hybrid test article",
        "published_at": today.isoformat(),
        "pl_enrichment": {
            "players_mentioned": [
                {"name": "Bukayo Saka", "mention_count": 4},
            ],
            "performance_signals": [
                {"player": "Bukayo Saka", "signal": "goal_threat",
                 "score": 0.7, "confidence": 0.85, "evidence": "Saka scoorde"},
                {"player": "Bukayo Saka", "signal": "positive_form",
                 "score": 0.6, "confidence": 0.9, "evidence": "Saka in vorm"},
            ],
        },
    }
    (pl_dir / "hybrid-test.json").write_text(
        json.dumps(article, ensure_ascii=False), encoding="utf-8",
    )
    return tmp_path


# ---------------------------------------------------------------------------
# Match predictions (Poisson + Sporza modifiers)
# ---------------------------------------------------------------------------


class TestMatchPrediction:
    def test_probabilities_sum_to_one(self, tmp_path: Path):
        data_root = _setup_hybrid_data(tmp_path)
        mp = predict_match("Arsenal", "Brighton", data_root=data_root)
        total = mp.home_win_prob + mp.draw_prob + mp.away_win_prob
        assert abs(total - 1.0) < 0.02

    def test_expected_goals_positive(self, tmp_path: Path):
        data_root = _setup_hybrid_data(tmp_path)
        mp = predict_match("Arsenal", "Brighton", data_root=data_root)
        assert mp.expected_home_goals > 0
        assert mp.expected_away_goals > 0

    def test_output_fields(self, tmp_path: Path):
        data_root = _setup_hybrid_data(tmp_path)
        mp = predict_match("Arsenal", "Brighton", data_root=data_root)
        assert isinstance(mp, MatchPrediction)
        assert mp.home_team
        assert mp.away_team
        assert mp.confidence in ("high", "medium", "low")
        assert 0 <= mp.over_2_5_prob <= 1
        assert 0 <= mp.btts_prob <= 1

    def test_key_factors_populated(self, tmp_path: Path):
        data_root = _setup_hybrid_data(tmp_path)
        mp = predict_match("Arsenal", "Brighton", data_root=data_root)
        assert isinstance(mp.key_factors, list)
        assert len(mp.key_factors) > 0

    def test_to_dict(self, tmp_path: Path):
        data_root = _setup_hybrid_data(tmp_path)
        mp = predict_match("Arsenal", "Brighton", data_root=data_root)
        d = mp.to_dict()
        assert isinstance(d, dict)
        assert "home_win_prob" in d
        assert "expected_home_goals" in d

    def test_no_csv_fallback(self, tmp_path: Path):
        """Match prediction works even without stats CSV."""
        pl_dir = tmp_path / "premier-league"
        pl_dir.mkdir(parents=True)
        mp = predict_match("Arsenal", "Brighton", data_root=tmp_path)
        total = mp.home_win_prob + mp.draw_prob + mp.away_win_prob
        assert abs(total - 1.0) < 0.02

    def test_save_match_prediction(self, tmp_path: Path):
        data_root = _setup_hybrid_data(tmp_path)
        mp = predict_match("Arsenal", "Brighton", data_root=data_root)
        out = tmp_path / "match.json"
        save_predictions(mp, out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert "home_win_prob" in data

    def test_print_no_crash(self, tmp_path: Path, capsys):
        data_root = _setup_hybrid_data(tmp_path)
        mp = predict_match("Arsenal", "Brighton", data_root=data_root)
        print_match_prediction(mp)
        captured = capsys.readouterr()
        assert "MATCH PREDICTION" in captured.out


# ---------------------------------------------------------------------------
# Hybrid player predictions (stats baseline + Sporza modifiers)
# ---------------------------------------------------------------------------


class TestHybridPlayerPredictions:
    def test_stats_baseline_used(self, tmp_path: Path):
        """With stats CSV, player predictions use data-driven baselines."""
        data_root = _setup_hybrid_data(tmp_path)
        report = run_predictions(data_root=data_root)
        saka = next((p for p in report.players if p.player_name == "Bukayo Saka"), None)
        assert saka is not None
        assert saka.prediction is not None
        # With positive goal_threat signals, score prob should be > 0.
        assert saka.prediction.likely_to_score > 0

    def test_predictions_bounded(self, tmp_path: Path):
        data_root = _setup_hybrid_data(tmp_path)
        report = run_predictions(data_root=data_root)
        for p in report.players:
            if p.prediction:
                assert 0 <= p.prediction.likely_to_score <= 1
                assert 0 <= p.prediction.likely_to_assist <= 1
                assert 0 <= p.prediction.likely_to_start <= 1

    def test_confidence_with_stats(self, tmp_path: Path):
        """With stats CSV available, confidence should be at least medium."""
        data_root = _setup_hybrid_data(tmp_path)
        report = run_predictions(data_root=data_root)
        saka = next((p for p in report.players if p.player_name == "Bukayo Saka"), None)
        assert saka is not None
        assert saka.prediction is not None
        assert saka.prediction.confidence in ("medium", "high")

    def test_reasoning_mentions_team(self, tmp_path: Path):
        data_root = _setup_hybrid_data(tmp_path)
        report = run_predictions(data_root=data_root)
        saka = next((p for p in report.players if p.player_name == "Bukayo Saka"), None)
        assert saka is not None
        assert saka.prediction is not None
        assert "GPG" in saka.prediction.reasoning or "goal" in saka.prediction.reasoning.lower()
